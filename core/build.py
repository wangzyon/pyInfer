from typing import Dict
from .registry import INFER, ENGINE

__all__ = ["build_infer", "build_engine"]


def build_infer(config: Dict, **kwargs):
    if INFER.get(config.get('type')) is None:
        raise KeyError(f"Cannot found infer type {config.get('type')}.")

    infer = INFER.get(config.pop('type'))(**kwargs)
    start_params = infer.StartParams.parse_obj(config)
    if not infer.startup(start_params):
        return
    else:
        return infer


def build_engine(config: Dict, **kwargs):
    if ENGINE.get(config.get('type')) is None:
        raise KeyError(f"Cannot found infer engine {config.get('type')}.")

    engine = ENGINE.get(config.pop('type'))(**kwargs)
    if not engine.build(**config):
        return
    else:
        return engine
