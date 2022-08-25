![](doc/head.png)

# 概述

异步推理服务，基本流程：

![](doc/基本流程.svg)

1. 异步服务：通过http访问FastAPI注册的路由 `route func`，`route func`为异步协程，可并发访问；
2. 异步推理：`route func`解析请求参数获取输入，将输入封装成job，并立即返回未完成工作 `future`；`route func`通过await挂起，直到协程 `future`返回结果；
3. 执行任务：创建消费者线程，从任务队列 `queue`中获取任务并执行具体的推理工作，并将推理结果填充至 `future`；
4. 结果返回：异步协程 `future`在结果填充后返回，程序回到 `route func`，收集推理结果通过FastAPI返回，完成一次推理；

推理过程中数据流动：

![](doc\数据流动.svg)

1. infer通过commit将http输入封装成job；
2. job记录当前推理任务的输入input、输出output；
3. job独占数据资源mono_data存储中间数据，相当于workspace；
4. traits萃取器，概念来自于C++，做特征提取，例如计算仿射变换矩阵和逆矩阵，服务于预处理和后处理；
5. future做非阻塞及事件通知；

# 安装

```
pip install fastapi[all]
```

# 设计优势

## 异步动态batch推理

1. 异步：`future`协程实现推理的异步，`route func`实现网络通信的异步，两者均是协程，基于事件循环 `EvenLoop`进行切换，从而满足并发；

本项目中使用两种future，对应两种job：

```python
# 同步非阻塞，start_job使用，同步等待模型启动成功；
from concurrent import futures
future = futures.Future()

# 异步非阻塞，infer job, 配合await使用, 异步等待推理结果；
import asyncio
loop = asyncio.get_running_loop()
future = loop.create_future()

#future,在多线程需通过如下方式保证跨线程事件触发；（待验证）
event_loop.call_soon_threadsafe(job.future.set_result, result)
```

2. 动态推理

基于生产线程、消费线程、任务队里、线程通信等机制，推理时可将多个任务合并推理，提高推理并行度；

## 独占资源分配器

独占资源分配器实现两个功能：内存防溢出、预处理和推理解耦；

![](doc\独占资源分配器.svg)

由job结构和数据流动可知，要想将job加入任务队列必须经过预处理preprocess，而预处理必须申请独占数据资源（python类对象），才能够存储预处理结果和模型推理结果，因此，设计独占资源分配器，可以巧妙控制整个推理的进行；

1. 内存防溢出：独占资源分配器设定指定数量的独占资源，当所有资源被分配占用时，再次commit必须等待之前的任务推理完成，释放资源；因此内存占用恒定，大量commit不会造成服务器内存溢出崩溃；
2. 预处理和推理解耦：独占数据资源数量一般为最大batch_size两倍，此时，满足一个batch_size进行预处理，一个batch_size进行推理，即预备一个batch_size，实现prefetch的功能，从而使预处理和推理同时进行，提高推理效率；

# 测试

1. 下载models：models包含mmdetection框架下，balloon数据集的目标检测模型文件；

[下载地址](https://pan.baidu.com/s/1CgXf7Q59BtgFL8aOAFGouA )，提取码：2sh3 

```
models/
	yolox_s_8x8_300e_coco.py
	model.pth
```

2. 启动服务

切换路径至根目录下：

```bash
uvicorn app:app
```

3. 测试接口

上传workspace目录下balloon.jpg；

![](doc\demo_api.png)

# 关于异步

1. 函数、生成器、协程、异步生成器

```python
# 函数
def function():
    return 1
# 生成器
def generator():
    yield 1 
# 协程
async def async_function():
    return 1
# 异步生成器
async def async_generator():
    yield 1
```

> async标识普通函数即为协程

2. `await`和 `await for`

```python
async def async_function():
    return 1
# 对于协程async_function, await等价于try...except...e.value
try:
    async_function().send(None)
except StopIteration as e:
    value = e.value   
# 等价于
value = await async_function()
# 异步生成器可由await for获取结果
async def async_generator():
    for i in range(10):
    	yield i
    
await for v in async_generator():
    print(v)
```

推荐阅读：[Python Async/Await入门指南](https://zhuanlan.zhihu.com/p/27258289)
