import simpy
from abc import ABC, abstractmethod
from typing import List


class ParallelStrategy(ABC):
    """Base class for different parallel training strategies."""

    def __init__(self, micro_batches: int = 1, flops_per_batch: float = 4e12, comm_size: int = 200 * 1024 * 1024):
        self.micro_batches = micro_batches
        self.flops_per_batch = flops_per_batch
        self.comm_size = comm_size  # bytes

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, env: simpy.Environment, gpus: List["GPU"], recorder):
        """Return a SimPy generator that orchestrates the training step."""
        raise NotImplementedError 