from abc import abstractmethod, ABCMeta
import numpy as np
import onnxruntime
from ...utils.common.registry import ENGINES
from ...utils.common.logger import Logger

__all__ = ["OnnxInferEngine"]




class InferEngine(metaclass=ABCMeta):

    def __init__(self, device, logger=None, **kwargs) -> None:
        self.device = device
        self.logger = Logger() if logger is None else logger

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def forward(self, job):
        pass


@ENGINES.register_module()
class OnnxInferEngine(InferEngine):
    __providers__ = {"cuda":"CUDAExecutionProvider", "TensorRT":"TensorrtExecutionProvider", "cpu":"TensorrtExecutionProvider"}

    def build(self, onnx_file):
        providers = onnxruntime.get_available_providers()
        
        if self.__providers__[self.device] not in providers:
            self.logger.fatal(f"Onnxruntime lack {self.__providers__[self.device]}, build model failed.")
            return False
        
        self.logger.info(f"onnxruntime use [{self.__providers__[self.device]}], support [{','.join(providers)}]")
        self.session = onnxruntime.InferenceSession(onnx_file, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        return True

    def forward(self, batch_input: np.ndarray):
        """
        默认onnx单输入单输出
        
        batch_input[batch_size, ...]
        """
        # 输入格式转换
        batch_input = batch_input.transpose(0,3,1,2)
        batch_output = self.session.run(self.output_names, {self.input_names[0]:batch_input})[0]
        return batch_output
