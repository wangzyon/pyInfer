__all__ = ["INFER", "ENGINE"]


class Register():

    def __init__(self, name=None) -> None:
        self.module_dict = {}

    def register_module(self):

        def _register(module):
            self.module_dict[module.__name__] = module
            return module

        return _register

    def build(self, cfg):
        module_name = cfg.pop('type')
        return self.get(module_name)(**dict(cfg))

    def get(self, module_name):
        return self.module_dict.get(module_name)


INFERS = Register("INFERS")
ENGINES = Register("ENGINES")
HOOKS = Register("HOOKS")
