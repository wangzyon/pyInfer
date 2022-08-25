import threading
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Union
from abc import abstractmethod, ABCMeta
from pydantic import BaseModel, Field
from concurrent import futures

from .traits import WarpAffineTraits
from .mono_allocator import MonoAllocator
from .logger import Logger
from .registry import INFER
from .build import build_engine
from .visual import DetectionPlot

__all__ = ["DetectionInfer"]


class Job():

    def __init__(self, asyncio_event_loop=None) -> None:
        """
        推理job
        
        @input: 推理输入
        @output: 推理输出
        @mono_data: 独占数据资源，相当于workspace,推理任务必须获取独占数据资源，才能被提交至推理队列；
                mono_data.input记录输入预处理后数据，等待被送入模型，mono_data.output记录模型输出数据，等待被后处理；
        @traits: 特征萃取，提取当前输入的相关特征，辅助预处理和后处理；
        @future: 非阻塞，记录响应结果；asyncio_event_loop非None时，为异步job，需要在异步函数配合await使用，否则为同步；

        """
        self.input = None
        self.output = None
        self.traits = None
        self.mono_data = None
        self.future = futures.Future() if asyncio_event_loop is None else asyncio_event_loop.create_future()


class Infer(metaclass=ABCMeta):

    class Input(BaseModel):
        """输入"""
        pass

    class Output(BaseModel):
        """输出"""
        pass

    class StartParams(BaseModel):
        """推理配置参数"""
        max_batch_size: int
        workspace: str = Field(default="workspace/")

    def __init__(self, log_cfg={}):
        self._loop = asyncio.get_running_loop()    # 绑定异步事件循环，实现基于异步future的事件通知机制；
        self._cond = threading.Condition()
        self._jobs = []
        self.log = Logger(**log_cfg)

    def startup(self, start_params: StartParams):
        self.start_params = start_params
        self._run = True
        self._mono_allocator = MonoAllocator(self.start_params.max_batch_size * 2)
        # 启动推理消费线程，同步等待模型引擎初始化完成
        start_job = Job()
        t = threading.Thread(target=self.work, args=(start_job,))
        t.start()
        return start_job.future.result()

    @abstractmethod
    def preprocess(self, job: Job):
        """前处理"""
        pass

    @abstractmethod
    def postprocess(self, job: Job):
        """后处理"""
        pass

    @abstractmethod
    def work(self, start_job: Job):
        """推理"""
        pass

    def commit(self, inp: Input):
        "提交任务"
        job = Job(self._loop)
        job.input = inp

        if not self.preprocess(job):
            return

        # 预处理完成，将job添加至任务队列
        with self._cond:
            self._jobs.append(job)
            self._cond.notify()

        return job.future

    def commits(self, inps: List[Input]):
        return [self.commit(inp) for inp in inps]

    def wait_for_job(self) -> Job:
        """获取任务"""
        with self._cond:
            self._cond.wait_for(lambda: (len(self._jobs) > 0 or not self._run))
            if not self._run:
                return
            job = self._jobs.pop(0)
            return job

    def wait_for_jobs(self) -> List[Job]:
        """获取一批任务"""
        with self._cond:
            self._cond.wait_for(lambda: (len(self._jobs) > 0 or not self._run))
            if not self._run:
                return []
            max_size = min(self.start_params.max_batch_size, len(self._jobs))
            fetch_jobs = self._jobs[:max_size]
            self._jobs = self._jobs[max_size:]
            return fetch_jobs

    def __del__(self):
        with self._cond:
            self._run = False
            self._cond.notify()
        self.log.info(f"{self.__class__.__name__} is destoryed.")

    def destory(self):
        self.__del__()


@INFER.register_module()
class DetectionInfer(Infer):

    class Input(BaseModel):
        filename: str = Field(describe="图像文件名称")
        image: Union[np.ndarray, None] = Field(default=None, describe="图像解析后numpy对象")

        class Config:
            arbitrary_types_allowed = True

    class Output(BaseModel):

        class BBox(BaseModel):
            left: float
            right: float
            top: float
            bottom: float
            confidence: float
            label: int
            labelname: str
            keepflag: bool = Field(default=True, title="Reserve this bbox")

        bbox_num: int
        bboxes: List[BBox]

        def to_array(self):
            ret = []
            for bbox in self.bboxes:
                ret.append([bbox.left, bbox.right, bbox.top, bbox.bottom, bbox.label])
            return np.array(ret).astype(np.int)

    class StartParams(BaseModel):
        engine: Dict
        confidence_threshold: float = Field(gt=0, describe="置信度阈值")
        nms_threshold: float = Field(gt=0, describe="NMS阈值")
        max_batch_size: int = Field(gt=0, describe="推理时最大batch_size")
        max_object: int = Field(gt=0, describe="图像包含的最大目标数量")
        width: int = Field(gt=0, describe="推理时送入网络的图像宽度")
        height: int = Field(gt=0, describe="推理时送入网络的图像高度")
        plot_result: bool = Field(default=True, describe="可视化检测结果至workspace")
        workspace: str = Field(default="workspace/")
        classes: Tuple = Field(describe="class name")
        timeout: int = Field(default=10, describe="提交job的最大等待时间，若超时，则提交任务失败.")

    def preprocess(self, job: Job):
        """预处理"""
        if job.input is None or job.input.image is None:
            self.log.error(f"input image is empty. Please check {job.input.filename}.")
            return False

        # 预处理要求获取独占数据资源
        mono_data = self._mono_allocator.query(self.start_params.timeout)
        if mono_data is None:
            self.log.error("query mono data timeout.")
            return False
        else:
            job.mono_data = mono_data

        # 通过warpaffineTraits萃取器，计算仿射变换矩阵、逆矩阵、并完成仿射变换
        job.traits = WarpAffineTraits(job.input.image.shape[1], job.input.image.shape[0], self.start_params.width,
                                      self.start_params.height)
        # mono_data.input存储预处理结果，作为模型推理的输入
        job.mono_data.input = job.traits(job.input.image)
        return True

    def cpu_decode(self, job: Job) -> List[Output.BBox]:
        """
        解码, 将模型推理结果解码为Output.BBox格式
        """
        bboxes = []
        # 从独占数据资源中取出模型推理结果, results=List[bbox]格式，其中bbox为[x,y,width,height,confidence,label]
        results = job.mono_data.output
        for result in results:
            left, top, right, bottom, confidence, label = result
            if confidence < self.start_params.confidence_threshold:
                continue
            label = int(label)
            # 通过逆仿射变换将bbox坐标映射回原图
            left, top = job.traits.to_src_coord(left, top)
            right, bottom = job.traits.to_src_coord(right, bottom)
            labelname = self.start_params.classes[label]
            bbox = self.Output.BBox(
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                confidence=confidence,
                label=label,
                labelname=labelname,
                keepflag=True)
            bboxes.append(bbox)
        return bboxes

    def cpu_nms(self, bboxes: List[Output.BBox]) -> List[Output.BBox]:

        def box_iou(a, b):
            cleft = max(a.left, b.left)
            ctop = max(a.top, b.top)
            cright = min(a.right, b.right)
            cbottom = min(a.bottom, b.bottom)
            c_area = max(cbottom - ctop, 0) * max(cright - cleft, 0)
            if c_area == 0:
                return 0
            a_area = max(a.right - a.left, 0) * max(a.bottom - a.top, 0)
            b_area = max(b.right - b.left, 0) * max(b.bottom - b.top, 0)
            return c_area / (a_area + b_area - c_area)

        for i in range(len(bboxes)):
            for j in range(len(bboxes)):
                # label不同或同一个bbox跳过nms
                if i == j or bboxes[i].label != bboxes[j].label:
                    continue

                # 置信度相同，保留靠后bbox, 即j<i时不对i进行nms
                if bboxes[j].confidence == bboxes[i].confidence and j < i:
                    continue

                iou = box_iou(bboxes[i], bboxes[j])
                if (iou > self.start_params.nms_threshold):
                    bboxes[i].keepflag = 0

        return list(filter(lambda bbox: bbox.keepflag, bboxes))

    def postprocess(self, job):
        """后处理"""
        bboxes = self.cpu_decode(job)
        # bboxes = self.cpu_nms(bboxes)
        job.output = self.Output(bboxes=bboxes, bbox_num=len(bboxes))
        return job

    def work(self, start_job: Job):
        engine = build_engine(self.start_params.engine, log=self.log)    # 加载模型，创建目标检测推理引擎
        start_job.future.set_result(engine)
        if engine is None:
            self.log.error(f"加载模型失败.")
            return
        self.log.info(f"加载模型成功.")

        while self._run:
            fetch_jobs = self.wait_for_jobs()

            # 推理停止运行，fetch_jobs返回为空，未完成job推理结果置None
            if len(fetch_jobs) == 0:
                for job in self._jobs:
                    job.future.set_result(job.output)
                break

            # fetch_jobs非空，将多个job的input组合成batch
            batch_input = np.stack([job.mono_data.input for job in fetch_jobs])
            # 推理
            batch_output = engine.forward(batch_input)
            self.log.info(f"Infer batch size({len(fetch_jobs)}), wait jobs({len(self._jobs)})")
            for index, job in enumerate(fetch_jobs):
                # mono_data.output存储推理结果，作为后处理的输入
                job.mono_data.output = batch_output[index]
                # 后处理得到job.output
                job = self.postprocess(job)
                # 释放独占数据资源
                job.mono_data.release()
                job.mono_data = None
                # 可视化推理结果
                if self.start_params.plot_result:
                    self.plot_result(job)

                job.future.set_result(job.output)    # 返回推理结果

    def plot_result(self, job: Job):
        plot = DetectionPlot(job.input.image, job.output.bboxes).plot()
        plot.save(f"{self.start_params.workspace}/infer_{job.input.filename}")
