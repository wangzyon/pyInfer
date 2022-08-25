import cv2
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File

from core.build import build_infer
from core import DetectionInfer
from appConfigs import detection_infer_cfg, log_cfg

app = FastAPI()

infers = {}


@app.on_event("startup")
async def startup_event():
    infers["detection_infer"] = build_infer(detection_infer_cfg, log_cfg=log_cfg)


@app.on_event("shutdown")
def shutdown_event():
    for infer in infers.values():
        infer.destory()


async def decode_upload_img(file: UploadFile = File(...)):
    buffer = await file.read()
    buffer = np.frombuffer(buffer, np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR).astype(np.float32)
    return img


@app.get('/health', tags=["服务状态"], summary="infer one image")
async def health():
    return {'START': 'UP'}


@app.post('/infer', response_model=DetectionInfer.Output, tags=["推理"], summary="infer one image")
async def infer(img_file: UploadFile = File(...)):
    """
    推理
    \f
    :param img_file: 推理时图像输入.
    """
    input = DetectionInfer.Input.parse_obj(dict(filename=img_file.filename, image=await decode_upload_img(img_file)))
    infer = infers.get("detection_infer")
    future = infer.commit(input)
    if future is None:
        return future
    else:
        return await future
    #return await future if future is None else await future
