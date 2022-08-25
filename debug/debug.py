import pickle
import numpy as np
from pydantic import BaseModel, Field
from core.build import build_infer
from appConfigs import detection_infer_cfg
from typing import Union


class Input(BaseModel):
    filename: str = Field(describe="图像文件名称")
    image: Union[np.ndarray, None] = Field(default=None, describe="图像解析后numpy对象")

    class Config:
        arbitrary_types_allowed = True


detection_infer = build_infer(detection_infer_cfg)

# app中存储
# with open("xxx", 'wb') as f:
#     image = await decode_upload_img(img_file)
#     pickle.dump(image, f)
#     raise

with open("xxx", 'rb') as f:
    img_file = pickle.load(f)

input = detection_infer.Input.parse_obj(dict(filename='test.jpg', image=img_file))
future = detection_infer.commit(input)
result = future.result()
detection_infer.destory()