import simpy
from typing import List

from .base import ParallelStrategy
from simulator.gpu import GPU


class DataParallelStrategy(ParallelStrategy):
    """Data parallelism: replicate model, different data, gradient all-reduce."""

    def run(self, env: simpy.Environment, gpus: List[GPU], recorder):
        num_gpus = len(gpus)

        for mb in range(self.micro_batches):
            # ------------------------------------------------------------------
            # Forward & backward computation on all GPUs in parallel
            # ------------------------------------------------------------------
            compute_events = [env.process(g.compute(f"mb{mb}_compute", self.flops_per_batch)) for g in gpus]
            yield simpy.events.AllOf(env, compute_events)

            # ------------------------------------------------------------------
            # Gradient all-reduce (simplified as pairwise sends)
            # ------------------------------------------------------------------
            comm_events = []
            for g in gpus:
                dst = (g.id + 1) % num_gpus  # simple ring next hop
                comm_events.append(env.process(g.send(dst, self.comm_size)))
            yield simpy.events.AllOf(env, comm_events) 