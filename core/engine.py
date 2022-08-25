from abc import abstractmethod, ABCMeta
from .registry import ENGINE
from .logger import Logger
import numpy as np

__all__ = ["MMDetectionInferEngine"]

try:
    from mmdet.apis import init_detector, inference_detector
    MMDETECTION_ENABLE = True
except:
    MMDETECTION_ENABLE = False


class InferEngine(metaclass=ABCMeta):

    def __init__(self, log=None, **kwargs) -> None:
        self.log = Logger() if log is None else log

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def forward(self, job):
        pass


@ENGINE.register_module()
class MMDetectionInferEngine(InferEngine):

    def build(self, model_file, config_file, device):
        if not MMDETECTION_ENABLE:
            self.log.fatal("Cannot import mmdet, build model failed.")
        self.model = init_detector(config_file, model_file, device=device)
        return True

    def forward(self, batch_input: np.ndarray):
        """
        对于推理输入，batch_input格式为(batch_size, height, width, channel),需要将batch_input转换为mmdetection输入格式
        mmdetection输入为List[input], 其中input为(height, width, channel)
        
        对于推理输出：
        mmdetection results格式为:
        [
            [
                array([[xmin, ymin, xmax, ymax, confidence],
                    [xmin, ymin, xmax, ymax, confidence]], dtype=float32),
                array([[xmin, ymin, xmax, ymax, confidence]], dtype=float32),
            ]
        ]
        第一层索引i表示第i个样本, 第二层索引j为类别label, 第三层数组shape为(n,5), n表示输出bbox数量；
        
        将mmdetection results格式转换为如下格式：
        [
            [
                [xmin, ymin, xmax, ymax, confidence, label],
                [xmin, ymin, xmax, ymax, confidence, label],
                [xmin, ymin, xmax, ymax, confidence, label],
            ]
        ]
        第一层索引i表示第i个样本, 第二层索引j为第i个样本的第j个bbox 
        """
        # 输入格式转换
        inputs = [batch_input[i] for i in range(len(batch_input))]
        results = inference_detector(self.model, inputs)

        # 输出格式转换
        new_results = []
        for input_i, input_i_result in enumerate(results):
            bboxes = []
            for label, bbox_array in enumerate(input_i_result):
                bbox_num = len(bbox_array)
                bbox_array = bbox_array.astype(np.float)    # 避免出现其他数据类型，影响网络序列化传输
                label_array = np.zeros((bbox_num, 1)) * label
                bbox_array = np.concatenate([bbox_array, label_array], axis=1)
                bboxes.extend(bbox_array.tolist())
            new_results.append(bboxes)
        return new_results


class MMPoseDetectionInferEngine(InferEngine):

    def build(self, model_file, config_file):
        pass

    def forward(self, job):
        pass