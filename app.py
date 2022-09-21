import uvicorn
import argparse
from enum import Enum
from fastapi import FastAPI, UploadFile, File

from pyinfer.core.build import build_infer, build_logger
from pyinfer.core.infer import Infer
from pyinfer.utils.common.config import Config




app = FastAPI()

infers = {}    # 存储初始化后的推理器


class OnlineInferName(str, Enum):
    """app上线的推理服务"""
    DetectionInfer = "Detection目标检测推理"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/volume/wzy/project/PyInfer/applications/balloon/config.py")
    return parser.parse_args()


def infer_func(infer_name):

    async def wrapper(file: UploadFile = File(...)):
        """
        \f
        :param infer_name: 推理器名称；
        :param file: API接口输入文件流；

        1. 获取推理器；
        2. 解析文件流至推理器输入；
        3. 提交输入；
        4. 返回推理结果；

        """
        infer: Infer = infers.get(infer_name)
        input = infer.parse_raw(filename=file.filename, raw_data=await file.read())
        future = infer.commit(input)
        return future if future is None else await future

    return wrapper


@app.on_event("startup")
async def startup_event():
    """
    初始化所有待上线的推理器

    1. 依据推理器名称，获取相应的配置参数；
    2. 创建并初始化推理器；
    3. 一个推理器初始化成功，则添加其对应的推理服务路由

    """
    args = parse_args()
    cfg = Config.fromfile(args.config)

    logger = build_logger(cfg.log)
    for item in OnlineInferName:
        if cfg.infer.get(item.name) is None:    # 1
            logger.error(f"{item.name} config lack, build infer failed.")
            continue

        infer = build_infer(cfg.infer.get(item.name), logger=logger)    # 2
        app.post(f'/infer/{item.name}', response_model=infer.Output,
                 tags=[item.value])(infer_func(item.name))    # 3
        infers[item.name] = infer


@app.on_event("shutdown")
def shutdown_event():
    """销毁推理器"""
    for infer in infers.values():
        infer.destory()


@app.get('/health', tags=["服务状态"], summary="")
async def health():
    """服务网络状态测试"""
    return {'START': 'UP'}


if __name__ == "__main__":
    uvicorn.run(app, port = 8805)
