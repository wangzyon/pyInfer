import threading
import weakref
import time
from .logger import Logger

__all__ = ["MonoAllocator"]


class MonoAllocator():
    """独占数据资源分配器"""

    class MonoData():
        """独占数据资源"""

        def __init__(self, name, allocator, available=True) -> None:
            """
            input: 存储模型网络的输入数据
            output: 存储模型网络的输出数据
            """
            self.name = name
            self._available = available
            self._allocator = allocator
            self.input = None
            self.output = None

        def release(self):
            self._allocator.release_one(self)

    def __init__(self, size: int) -> None:
        """
        params:
            size      : 独占数据资源数量，建议设置为batch_size*2；
            self.datas   : 独占数据资源列表；
            self._num_avail : 剩余资源数量；
            self._num_thread_wait: 等待资源的线程数量；
            self._run: 当程序终止时，触发析构__del__, _run=False，停止资源分配，在等待资源的线程陆续退出等待；
            self._lock：线程锁；
            self._cond：线程同步，同步资源数量；
            self._cond_exit：线程同步，同步等待资源的线程数量，停止资源分配_run=False时，退出时需要等待_num_thread_wait为0；
            
        """
        self.datas = [self.MonoData(f"mono_data_{i}", weakref.proxy(self)) for i in range(size)]
        self._num_avail = size
        self._num_thread_wait = 0
        self._run = True
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._cond_exit = threading.Condition(self._lock)
        self._log = Logger()

    def query(self, timeout=10):
        """请求独占数据资源"""
        with self._cond:
            if not self._run:
                # 推理终止时停止分配数据资源
                return

            # 等待独占数据资源
            if self._num_avail == 0:
                self._num_thread_wait += 1    # 排队线程数+1
                state = self._cond.wait_for(lambda: (self._num_avail > 0 or not self._run), timeout)
                self._num_thread_wait -= 1    # 排队线程数-1
                # 更新析构状态
                self._cond_exit.notify()
                # 未获取到独占数据资源，可能是请求超时或分配器停止分配
                if not state or self._num_avail == 0 or not self._run:
                    return

            # 返回请求得到的独占数据资源
            for mono_data in self.datas:
                if mono_data._available:
                    self._log.debug(f"occupy {mono_data.name}")
                    mono_data._available = False
                    self._num_avail -= 1
                    return mono_data

    def release_one(self, mono_data: MonoData):
        """释放独占数据资源所有权"""
        with self._cond:
            self._num_avail += 1
            mono_data._available = True
            mono_data.input = None
            mono_data.output = None
            self._cond.notify_all()
            time.sleep(1e-4)
        self._log.debug(f"release {mono_data.name}")

    def __del__(self):
        """对象析构时调用"""
        with self._cond:
            self._run = False
            self._cond.notify_all()

        with self._cond_exit:
            # 等待所有等待线程退出wait状态
            self._cond_exit.wait_for(lambda: (self._num_thread_wait == 0))

        self._log.info("MonoAllocator is destoryed.")

    def destory(self):
        self.__del__()
