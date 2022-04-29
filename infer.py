from abc import abstractmethod
from locale import currency
from queue import Queue
import threading
import asyncio
import concurrent
import uuid
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from configs import general_image_model
from mmdet.apis import init_detector, inference_detector  # 图像框架相关

__all__ = ["create_general_image_infer"]


class Job:
    """推理任务"""
    def __init__(self, inp, future):
        self.future = future
        self.inp = inp


def create_job(inp, loop):
    if loop is None:
        future = currency.futures.Future()
    else:
        future = loop.create_future()
    return Job(inp, future)


class Infer():
    """推理器接口基类，子类重新forward方法"""
    @abstractmethod
    def forward(self, job):
        pass


class GeneralImageInferResult:
    """图像识别结果封装"""
    class Target():
        def __init__(self):
            self.bbox = []
            self.label = None
            self.score = None
            self.id = uuid.uuid1().hex

        @staticmethod
        def ltrb_to_four_point(bbox):
            """4个点表示，与任意四边形box格式统一"""
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[1]
            x3, y3 = bbox[2], bbox[3]
            x4, y4 = bbox[0], bbox[3]
            return [x1, y1, x2, y2, x3, y3, x4, y4]

        def state_dict(self):
            return {
                "BBOX": self.ltrb_to_four_point(self.bbox),
                "LABEL": self.label,
                "SCORE": self.score,
                "TARGETID": self.id
            }

    def make_target(self):
        return self.Target()

    def __init__(self):
        self.width = None
        self.height = None
        self.targets = []

    def state_dict(self):
        return {"WIDTH": self.width, "HEIGHT": self.height, "TARGETS": [t.state_dict() for t in self.targets]}


class GeneralImageInferImpl(Infer):
    """
    识图推理器
    """
    def __init__(self, config, checkpoint, job_limit_size, max_batch_size, score_thr):
        """

        Args:
            config (string): 模型配置文件，例如mmdetection的config.py
            checkpoint (string): 模型文件，torch.pt
            job_limit_size (int): 最大任务数量，限制溢出
            max_batch_size (int): 推理batch上限，取决于GPU显存
            score_thr (float): 置信度阈值
        """
        self._config = config
        self._checkpoint = checkpoint
        self._model = None
        self._running = False
        self._jobs = Queue()
        self._cv = threading.Condition()
        self._worker_thread = None
        self._job_limit_size = job_limit_size
        self._max_batch_size = max_batch_size
        self._score_thr = score_thr
        self._event_loop = None  # 异步推理事件循环，保证在消费线程中触发其他线程（例如生产线程）中的事件

    def startup(self):
        """创建模型"""
        self._running = True
        future = concurrent.futures.Future()
        self._worker_thread = threading.Thread(target=self.dowork, args=(future, ))  # 将模型创建及初始化放到线程内部做，利用资源回收
        self._worker_thread.start()
        return future.result()  # 阻塞，确保模型初始化完成

    def dowork(self, future: asyncio.Future):
        """推理线程，消费者"""
        try:
            self._model = init_detector(self._config, self._checkpoint, device='cuda:0')
        except:
            future.set_result(False)  # 模型创建失败
            return

        future.set_result(True)  # 模型创建成功
        fetch_jobs = []
        while self._running:
            with self._cv:
                if self._jobs.empty():
                    self._cv.notify()
                    self._cv.wait()  # 阻塞，避免轮询浪费CPU
                if not self._running: break
                for _ in range(min(self._jobs.qsize(), self._max_batch_size)):  # 动态batch
                    fetch_jobs.append(self._jobs.get())

            # 推理
            print(f"inference with batch size {len(fetch_jobs)}")
            results = self.pipeline([job.inp for job in fetch_jobs])
            for result, job in zip(results, fetch_jobs):
                if self._event_loop is None:
                    job.future.set_result(result)  # 同步场景，填充推理结果
                else:
                    self._event_loop.call_soon_threadsafe(job.future.set_result, result)  # 异步场景，填充推理结果
            fetch_jobs.clear()

    def forward(self, inps):
        """将输入封装为任务，添加到待推理任务队列中"""
        inps = inps if isinstance(inps, list) else [inps]
        jobs = [create_job(inp, self._event_loop) for inp in inps]
        with self._cv:  # 获取锁
            if self._jobs.qsize() > self._job_limit_size:  # 任务数量达到上限，阻塞任务添加，避免资源占用过多程序崩溃
                self._cv.wait()
            for job in jobs:
                self._jobs.put(job)  # 添加任务，python队列是线程安全的
            print(f"increase job num:{len(jobs)},current job num:{self._jobs.qsize()}")
            self._cv.notify()  # 通知消费
        return [job.future for job in jobs]

    def pipeline(self, urls):
        """实际模型推理和后处理"""
        images = list(map(lambda url: np.array(Image.open(BytesIO(requests.get(url).content))), urls))
        results = inference_detector(self._model, images)

        def transform(image, result):
            r = GeneralImageInferResult()
            r.height = image.shape[0]
            r.width = image.shape[1]
            for label_index, boxes in enumerate(result):
                boxes = boxes.tolist()
                for box_index in range(len(boxes)):
                    target = r.make_target()
                    target.label = self._model.CLASSES[label_index]
                    target.bbox = list(map(int, boxes[box_index][:4]))
                    target.score = boxes[box_index][4]
                    if target.score > self._score_thr:
                        r.targets.append(target)
            return r.state_dict()

        return [transform(image, result) for image, result in zip(images, results)]

    def shutdown(self):
        """停止推理，当推理器对象销毁时，触发del方法，触发shutdown，通过join等待线程正常退出"""
        if self._running:
            self._running = False
            with self._cv:
                self._cv.notify()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join()

    def bind_event_loop(self, loop):
        """绑定事件循环"""
        self._event_loop = loop
        return self

    def __del__(self):
        """对象析构时调用"""
        self.shutdown()


def create_general_image_infer():
    """基于RAII的接口封装"""
    infer = GeneralImageInferImpl(**general_image_model)
    if infer.startup():
        return infer
    else:
        return None
