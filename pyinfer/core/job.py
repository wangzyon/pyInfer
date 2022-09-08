from concurrent import futures
from typing import List


class Job():
    def __init__(self, inp=None, loop=None) -> None:
        """
        推理job

        :param input: 推理输入
        :param output: 推理输出
        :param mono_data: 独占数据资源，相当于workspace,推理任务必须获取独占数据资源，才能被提交至推理队列；
                mono_data.input记录输入预处理后数据，等待被送入模型，mono_data.output记录模型输出数据，等待被后处理；
        :param traits: 特征萃取，提取当前输入的相关特征，辅助预处理和后处理；
        :param future: 非阻塞，记录响应结果；loop非None时，为异步job，需要在异步函数配合await使用，否则为同步；

        """
        self.input = inp
        self.output = None
        self.traits = None
        self.mono_data = None
        self.future = futures.Future() if loop is None else loop.create_future()
        self.call_backs = []
        self.future.add_done_callback(self.run_call_back)

    def add_done_call_back(self, call_back_fn):
        self.call_backs.append(call_back_fn)

    def run_call_back(self, future):
        for cb in self.call_backs:
            cb(self)


class JobSet(Job):
    def __init__(self, jobs: List[Job], collect_fn, inp=None, loop=None) -> None:
        """
        任务集合：当所有子任务均完成时，JobSet完成


        :param jobs: 任务列表
        :param loop: loop非None时，为异步job，需要在异步函数配合await使用，否则为同步；
        :param collect_fn: 推理结果回收，collect_fn输入为所有job的future.result列表，collect_fn返回值为JobSet.future.result，即整个任务集合的结果；
        """
        self.collect_fn = collect_fn
        self.jobs = jobs
        for job in self.jobs:
            job.add_done_call_back(self.notify)

        super().__init__(inp, loop)

    def done(self):
        # 所有job完成，且future未执行set_result时，避免多次重复set_result
        return all([job.future.done() for job in self.jobs]) and not self.future.done()

    def notify(self, job):
        if self.done():
            self.future.set_result(self.collect_fn(self))
