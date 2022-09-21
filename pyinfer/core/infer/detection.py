import cv2
import os
import numpy as np
from typing import List, Union, Dict, Tuple
from pydantic import BaseModel, Field
from .base import Infer

from ..job import Job, JobSet

from ...utils.detection.nms import cpu_nms
from ...utils.detection.bbox import QuadrangleBBox
from ...utils.functional.slice import slice_one_image
from ...utils.functional.traits import WarpAffineTraits
from ..build import INFERS

@INFERS.register_module()
class DetectionInfer(Infer):
    class Input(BaseModel):
        filename: str = Field(describe="图像文件名称")
        image: Union[np.ndarray, None] = Field(
            default=None, describe="图像解析后numpy对象")
        ox: float = Field(default=0, describe="image坐标系原点x坐标")
        oy: float = Field(default=0, describe="image坐标系原点y坐标")

        class Config:
            arbitrary_types_allowed = True

    class Output(BaseModel):
        bbox_num: int
        bboxes: List[QuadrangleBBox]

        def to_array(self):
            ret = []
            for bbox in self.bboxes:
                ret.append([bbox.left, bbox.right, bbox.top,
                            bbox.bottom, bbox.label])
            return np.array(ret).astype(np.int)

    class StartParams(BaseModel):
        engine: Dict
        confidence_threshold: float = Field(gt=0, describe="置信度阈值")
        nms_threshold: float = Field(gt=0.0, describe="NMS阈值")
        max_batch_size: int = Field(gt=0, describe="推理时最大batch_size")
        max_object: int = Field(gt=0, describe="图像包含的最大目标数量")
        width: int = Field(gt=0, describe="推理时送入网络的图像宽度")
        height: int = Field(gt=0, describe="推理时送入网络的图像高度")
        workspace: str = Field(default="workspace/")
        labelnames: Tuple = Field(describe="label names")
        timeout: int = Field(default=10, describe="提交job的最大等待时间，若超时，则提交任务失败.")
        hooks: List[Dict] = Field(description="钩子", default=[])
        slice: bool = Field(description="是否启动分片推理模式", default=False)
        device: str = Field(description="推理设备", default="cuda:0")
        # 分片推理模式下需要配置的参数
        subsize: int = Field(default=640, describe="瓦片大小")
        rate: int = Field(default=1, describe="大图resize系数")
        gap: int = Field(default=200, describe="瓦片间重叠大小")
        padding: bool = Field(default=True, describe="瓦片是否padding至subsize大小")

    def parse_raw(self, filename, raw_data):
        """数据解析为Input"""
        if self.start_params.slice:
            return self.__slice_parse_raw(filename, raw_data)
        else:
            return self.__parse_raw(filename, raw_data)

    def __parse_raw(self, filename, raw_data):
        """数据解析为Input, 非分片推理"""
        image = cv2.imdecode(np.frombuffer(raw_data, np.uint8),
                             cv2.IMREAD_COLOR).astype(np.float32)
        inp = self.Input.parse_obj(dict(filename=filename, image=image))
        return inp

    def __slice_parse_raw(self, filename, raw_data):
        """数据解析为Input, 分片推理，将整图拆分为多个Input"""
        image = cv2.imdecode(np.frombuffer(raw_data, np.uint8),
                             cv2.IMREAD_COLOR).astype(np.float32)
        image_base_name, extension = os.path.splitext(filename)
        patches = slice_one_image(image=image,
                                  image_base_name=image_base_name,
                                  subsize=self.start_params.subsize,
                                  rate=self.start_params.rate,
                                  gap=self.start_params.gap,
                                  padding=self.start_params.padding,
                                  bboxes=None)
        # sub input
        inps = [
            self.Input.parse_obj(
                dict(filename=f"{patch_name}{extension}", image=patch_image, ox=left, oy=top))
            for patch_name, rate, left, top, patch_image, patch_bboxes in patches
        ]
        # real input
        inps.insert(0, self.Input.parse_obj(
            dict(filename=filename, image=image, left=0, top=0)))
        return inps

    def collect_fn(self, jobset: JobSet):
        """slice分片结果融合"""
        bboxes: List[QuadrangleBBox] = []
        for job in jobset.jobs:
            ox, oy = job.input.ox, job.input.oy
            # 取每个瓦片图的检测框
            patch_bboxes: List[QuadrangleBBox] = job.output.bboxes
            # 检测框更新原点至(0,0)
            bboxes.extend([bbox.reset_origin(-ox, -oy)
                           for bbox in patch_bboxes])
        bboxes = cpu_nms(bboxes, self.start_params.nms_threshold)  # nms
        output = self.Output(bboxes=bboxes, bbox_num=len(bboxes))  # 返回大图检测框
        return output

    def preprocess(self, job: Job):
        """预处理"""
        if job.input is None or job.input.image is None:
            self.logger.error(
                f"input image is empty. Please check {job.input.filename}.")
            return False

        # 预处理要求获取独占数据资源
        mono_data = self._mono_allocator.query(self.start_params.timeout)
        if mono_data is None:
            self.logger.error("query mono data timeout.")
            return False
        else:
            job.mono_data = mono_data

        # 通过warpaffineTraits萃取器，计算仿射变换矩阵、逆矩阵、并完成仿射变换
        job.traits = WarpAffineTraits(job.input.image.shape[1], job.input.image.shape[0], self.start_params.width,
                                      self.start_params.height)
        # mono_data.input存储预处理结果，作为模型推理的输入
        job.mono_data.input = job.traits(job.input.image)
        return True
    
    def decode(self, job):
        bboxes = []
        # 从独占数据资源中取出模型推理结果, results=List[bbox]格式，其中bbox为[left,top,right,width, height, confidence, *scores]
        results = job.mono_data.output
        for result in results:
            xc, yc, width, height, confidence, *scores = result
            if confidence < self.start_params.confidence_threshold:
                continue
            
            max_score_index = np.argmax(scores)
            max_score = scores[max_score_index]
            if max_score*confidence < self.start_params.confidence_threshold:
                continue
            
            label = max_score_index
            # 通过逆仿射变换将bbox坐标映射回原图
            left, top = job.traits.to_src_coord(xc - width/2, yc - height/2)
            right, bottom = job.traits.to_src_coord(xc + width/2, yc + height/2)
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
    
    def nms(self, bboxes:List[QuadrangleBBox]):
        return cpu_nms(bboxes, self.start_params.nms_threshold)

    def postprocess(self, job):
        """后处理"""
        
        bboxes = self.nms(self.decode(job))
        job.output = self.Output(bbox_num=len(bboxes), bboxes = bboxes)