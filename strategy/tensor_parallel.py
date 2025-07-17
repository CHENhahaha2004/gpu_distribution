import simpy
from typing import List

from .base import ParallelStrategy
from simulator.gpu import GPU


class TensorParallelStrategy(ParallelStrategy):
    """Tensor parallelism: split tensor, partial compute, then all-reduce."""

    def run(self, env: simpy.Environment, gpus: List[GPU], recorder):
        num_parts = len(gpus)
        partial_flop = self.flops_per_batch / num_parts

        for mb in range(self.micro_batches):
            # Partial computation across GPUs
            compute_events = [env.process(g.compute(f"mb{mb}_partial_compute", partial_flop)) for g in gpus]
            yield simpy.events.AllOf(env, compute_events)

            # All-reduce to gather partial results (simplified as pairwise sends)
            comm_events = []
            for g in gpus:
                dst = (g.id + 1) % num_parts
                comm_events.append(env.process(g.send(dst, self.comm_size)))
            yield simpy.events.AllOf(env, comm_events) 