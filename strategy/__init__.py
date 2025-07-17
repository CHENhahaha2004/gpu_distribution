from .data_parallel import DataParallelStrategy
from .tensor_parallel import TensorParallelStrategy
from .pipeline_parallel import PipelineParallelStrategy

STRATEGY_REGISTRY = {
    "data": DataParallelStrategy,
    "tensor": TensorParallelStrategy,
    "pipeline": PipelineParallelStrategy,
}

def get_strategy(name: str, **kwargs):
    key = name.lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unsupported strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[key](**kwargs)

__all__ = [
    "get_strategy",
    "DataParallelStrategy",
    "TensorParallelStrategy",
    "PipelineParallelStrategy",
] 