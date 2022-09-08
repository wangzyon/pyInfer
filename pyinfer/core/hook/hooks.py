import cv2
import numpy as np
from typing import List
from PIL import Image, ImageDraw
from ..job import Job
from ...utils.common.registry import HOOKS
from ...utils.detection.bbox import QuadrangleBBox

__all__ = ["DrawBBoxHook"]


class Hook():

    def __init__(self):
        pass

    def after_set_result(self, job):
        pass


@HOOKS.register_module()
class DrawBBoxHook(Hook):

    def __init__(self, out_dir, prefix = "infer_") -> None:
        self.out_dir = out_dir
        self.prefix = prefix

    def after_set_result(self, job: Job):
        """绘制目标框"""
        image_np, filename = job.input.image.astype(np.uint8), job.input.filename
        bboxes: List[QuadrangleBBox] = job.future.result().bboxes
         
        image_background = Image.fromarray(image_np)
        draw_background = ImageDraw.Draw(image_background)
        for bbox in bboxes:
            draw_background.rectangle((bbox.left,bbox.top,bbox.right,bbox.bottom),fill='#FF00FF')
        
        image = Image.fromarray(cv2.addWeighted(image_np, 0.8, np.array(image_background),0.2, 0))
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle((bbox.left,bbox.top,bbox.right,bbox.bottom),outline='#FF00FF')
            draw.text((bbox.left,bbox.top),bbox.labelname)
        cv2.imwrite(f"{self.out_dir}/{self.prefix}{filename}", np.array(image))
        
        

