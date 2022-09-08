from typing import List

from .detection import DetectionInfer
from ...utils.detection.bbox import QuadrangleBBox
from ...utils.common.registry import INFERS
from ..job import Job

__all__ = ["MMDetectionInfer"]


@INFERS.register_module()
class MMDetectionInfer(DetectionInfer):

    def cpu_decode(self, job: Job) -> List[QuadrangleBBox]:
        """
        解码, 将模型推理结果解码为QuadrangleBBox格式
        """
        bboxes = []
        # 从独占数据资源中取出模型推理结果, results=List[bbox]格式，其中bbox为[left,top,right,bottom,confidence,label]
        results = job.mono_data.output
        for result in results:
            left, top, right, bottom, confidence, label = result
            if confidence < self.start_params.confidence_threshold:
                continue
            label = int(label)
            # 通过逆仿射变换将bbox坐标映射回原图
            left, top = job.traits.to_src_coord(left, top)
            right, bottom = job.traits.to_src_coord(right, bottom)
            labelname = self.start_params.labelnames[label]
            bbox = QuadrangleBBox(x1=left,
                                  y1=top,
                                  x2=right,
                                  y2=top,
                                  x3=right,
                                  y3=bottom,
                                  x4=left,
                                  y4=bottom,
                                  confidence=confidence,
                                  label=label,
                                  labelname=labelname,
                                  keepflag=True)
            bboxes.append(bbox)
        return bboxes

    def postprocess(self, job):
        """后处理"""
        bboxes = self.cpu_decode(job)
        job.output = self.Output(bboxes=bboxes, bbox_num=len(bboxes))
        return job
