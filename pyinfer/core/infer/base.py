import asyncio
import threading
import numpy as np
from abc import abstractmethod, ABCMeta
from typing import List, Union, Any, Dict
from pydantic import BaseModel, Field

from ..mono_allocator import MonoAllocator
from ..job import Job, JobSet
from ...utils.common.logger import Logger
from ..build import build_engine, build_hook

__all__ = ["Infer"]


class Infer(metaclass=ABCMeta):
    """
    推理器基类，接口类

              parser_raw          commit   preprocess        append       await job future
    生产者：raw_input----------------> infer input--------->job-------------->valid job--------->job queue--------------------->

                   engine forward              postprocess         set_result
    消费者：job.mono_data.input---------------------> job.mono_data.output---------------->job.output-----------------> future result
    """
    class Input(BaseModel):
        """输入"""
        pass

    class Output(BaseModel):
        """输出"""
        pass

    class StartParams(BaseModel):
        """推理配置参数"""
        engine: Dict = Field(description="infer engine config dict")
        hooks: List[Dict] = Field(description="钩子", default=[])
        max_batch_size: int
        workspace: str = Field(default="workspace/")

    def __init__(self, logger=None):
        self._loop = asyncio.get_running_loop()  # 绑定异步事件循环，实现基于异步future的事件通知机制；
        self._cond = threading.Condition()
        self._job_queue = []
        self.logger = Logger() if logger is None else logger
        self._hooks = []

    def startup(self, start_params: StartParams):
        """启动推理消费线程"""
        self.start_params = start_params
        self._run = True
        self._mono_allocator = MonoAllocator(
            self.start_params.max_batch_size * 2)
        start_job = Job()
        t = threading.Thread(target=self.work, args=(start_job, ))
        t.start()
        # 阻塞，确保模型引擎初始化完成
        return start_job.future.result()

    @abstractmethod
    def parse_raw(self, filename, raw_data) -> Union[Input, List[Input]]:
        """
        由网络输入解析为Infer的Input格式，或List[Input]格式
        """
        pass

    @abstractmethod
    def preprocess(self, job: Job):
        """前处理"""
        pass

    @abstractmethod
    def postprocess(self, job: Job):
        """后处理"""
        pass



    def commit(self, inp: Union[Input, List[Input]]):
        """
        提交任务

        将输入封装成job，提交至任务队列；
        """
        if isinstance(inp, list):
            future = self.__commits(inp)
        else:
            future = self.__commit(inp)
        return future


    def __commit(self, inp: Input):
        "提交单个任务"
        job = Job(inp, self._loop)

        if not self.preprocess(job):
            return

        # 预处理完成，将valid job添加至任务队列
        with self._cond:
            self._job_queue.append(job)
            self._cond.notify()

        # 添加钩子
        for hook in self._hooks:
            job.add_done_call_back(hook.after_set_result)

        return job.future

    def __commits(self, inps: List[Input]):
        """
        提交任务集

        inps[0]表示原始输入，inps[1:]表示原始输入拆分生成的n个子输入；
        """
        assert len(inps) >= 2, "inps num less than 2, check parse_raw method."
        real_inp, *sub_inps = inps

        # 创建子任务
        jobs = [Job(sub_inp, self._loop) for sub_inp in sub_inps]
        # 若存在无法预处理的子任务，则提交失败；
        if not all([self.preprocess(job) for job in jobs]):
            return

        # 预处理完成，将子任务添加至任务队列
        with self._cond:
            self._job_queue.extend(jobs)
            self._cond.notify()

        # 创建任务集
        jobset = JobSet(jobs, self.collect_fn, real_inp, self._loop)

        # 添加钩子
        for hook in self._hooks:
            jobset.add_done_call_back(hook.after_set_result)
        return jobset.future

    def collect_fn(self, jobset: JobSet) -> Any:
        """
        任务集结果融合：汇集所有子任务推理结果，collect_fn返回值作为任务集结果；当推理输入被拆分为多个子输入分别推理时，collect_fn必须重写；

        Args:
            jobset: 任务集

        return: Any, 返回值保存在jobset.future.result；
        """
        raise NotImplementedError

    def wait_for_job(self) -> Job:
        """获取任务"""
        with self._cond:
            self._cond.wait_for(
                lambda: (len(self._job_queue) > 0 or not self._run))
            if not self._run:
                return
            job = self._job_queue.pop(0)
            return job

    def wait_for_jobs(self) -> List[Job]:
        """获取一批任务"""
        with self._cond:
            self._cond.wait_for(
                lambda: (len(self._job_queue) > 0 or not self._run))
            if not self._run:
                return []
            max_size = min(self.start_params.max_batch_size,
                           len(self._job_queue))
            fetch_jobs = self._job_queue[:max_size]
            self._job_queue = self._job_queue[max_size:]
            return fetch_jobs
        
    def work(self, start_job: Job):
        """推理"""

        # 1.推理引擎创建并初始化
        engine = build_engine(self.start_params.engine, device =self.start_params.device , logger=self.logger)
        if engine is None:
            start_job.future.set_result(False)  # 初始化推理引擎失败，退出推理消费者线程
            return

        for hook_cfg in self.start_params.hooks:
            hook = build_hook(hook_cfg)
            if hook is None:
                start_job.future.set_result(False)  # 初始化钩子失败，退出推理消费者线程
                return
            self._hooks.append(hook)

        # 启动成功
        start_job.future.set_result(True)

        while self._run:
            # 2.取任务：从任务队里中取出预处理完成的job
            fetch_jobs = self.wait_for_jobs()
            if len(fetch_jobs) == 0:  # 当推理停止时，fetch_jobs返回为空
                for job in self._job_queue:
                    job.future.set_result(job.output)  # 未完成job推理结果置None
                break

            # 3.组合batch
            batch_input = np.stack([job.mono_data.input for job in fetch_jobs])
            # 4.推理
            batch_output = engine.forward(batch_input)
            self.logger.info(
                f"Infer batch size({len(fetch_jobs)}), wait jobs({len(self._job_queue)})")
            for index, job in enumerate(fetch_jobs):
                # 5.取模型输出结果：engine->job.mono_data.output
                job.mono_data.output = batch_output[index]
                # 6.后处理，job.mono_data.output->postprocess->job.output
                self.postprocess(job)
                # 7.释放独占数据资源
                job.mono_data.release()
                job.mono_data = None
                # 8.返回
                job.future.set_result(job.output)
                
    def __del__(self):
        with self._cond:
            self._run = False
            self._cond.notify()
        self.logger.info(f"{self.__class__.__name__} is destoryed.")

    def destory(self):
        self.__del__()
