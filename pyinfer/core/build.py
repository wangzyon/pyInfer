from typing import Dict
from ..utils.common.registry import INFERS, ENGINES, HOOKS
from ..utils.common.logger import Logger

__all__ = ["build_infer", "build_engine", "build_hook"]


def build_infer(config: Dict, **kwargs):
    if INFERS.get(config.get('type')) is None:
        raise KeyError(f"Cannot found infer type {config.get('type')}.")

    infer = INFERS.get(config.pop('type'))(**kwargs)
    start_params = infer.StartParams.parse_obj(config)
    if not infer.startup(start_params):
        return
    else:
        return infer


def build_engine(config: Dict, **kwargs):
    if ENGINES.get(config.get('type')) is None:
        raise KeyError(f"Cannot found infer engine {config.get('type')}.")

    engine = ENGINES.get(config.pop('type'))(**kwargs)
    if not engine.build(**config):
        return
    else:
        return engine


def build_hook(config):
    return HOOKS.build(config)


def build_logger(config) -> Logger:
    return Logger(**dict(config))
