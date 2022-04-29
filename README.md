![](images\head.png)


# 概述

异步推理服务，基本流程：

![](images\async_infer.svg)

1. 服务并发：通过http访问aiohttp注册的路由`route func`，`route func`为异步协程，可并发访问；
2. 异步推理：`route func`解析请求参数`input`作为`forward`输入，`forward`将输入封装成job，并立即返回未完成工作`future`；`route func`通过await挂起，直到协程`future`返回结果；
3. 执行任务：创建消费者线程，从任务队列`queue`中获取任务并执行具体的推理工作，并将推理结果填充至`future`；
4. 结果返回：异步协程`future`在结果填充后返回，程序回到`route func`，收集推理结果通过aiohttp返回，完成一次推理；


# 安装
```
pip install aiohttp
```

# 设计细节

1. `future`协程实现推理的异步，`route func`实现网络通信的异步，两者均是协程，基于事件循环`EvenLoop`进行切换；
2. `future`在多线程中，必须通过`event_loop.call_soon_threadsafe(job.future.set_result, result)`方式，才能保证跨线程事件触发；
3. 动态batch推理，提高推理效率；
4. 任务数量限制，溢出时生产等待，任务数量减少后自动恢复；
5. RAII接口封装思想；

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

2. `await`和`await for`

```python
async def async_function():
    return 1
# 对于协程async_function, await等价于try...except...e.value
try:
    async_function().send(None)
except StopIteration as e:
    print(e.value)   
# 等价于
value = await async_function()
# 异步生成器可由await for获取结果
async def async_generator():
    for i in range(10):
    	yield i
        
async for v in async_generator():
    print(v)
```

推荐阅读：[Python Async/Await入门指南](https://zhuanlan.zhihu.com/p/27258289)