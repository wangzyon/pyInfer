__all__ = ["INFER", "ENGINE"]


class Register():

    def __init__(self, name=None) -> None:
        self.module_dict = {}

    def register_module(self):

        def _register(module):
            self.module_dict[module.__name__] = module
            return module

        return _register

    def get(self, module_name):
        return self.module_dict.get(module_name)


INFER = Register("INFER")
ENGINE = Register("ENGINE")