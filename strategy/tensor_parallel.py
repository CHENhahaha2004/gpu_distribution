import simpy
from typing import List

from .base import ParallelStrategy
from simulator.gpu import GPU


class TensorParallelStrategy(ParallelStrategy):
    """Tensor parallelism: split tensor, partial compute, then all-reduce."""

    def run(self, env: simpy.Environment, gpus: List[GPU], recorder):
        from collections import defaultdict

        num_parts = len(gpus)
        partial_flop = self.flops_per_batch / num_parts

        # Pre-compute node grouping and leaders (same as data-parallel)
        node_to_gpus = defaultdict(list)
        for g in gpus:
            node_to_gpus[g.node_id].append(g)

        leaders = [grp[0] for grp in node_to_gpus.values()]

        for mb in range(self.micro_batches):
            # -------- Forward --------
            # Step-1: Partial forward compute on each GPU
            compute_events = [env.process(g.compute(f"mb{mb}_fwd_partial", partial_flop)) for g in gpus]
            yield simpy.events.AllOf(env, compute_events)

            # Step-2~4: Hierarchical all-reduce of forward outputs
            # (same pattern used earlier, we wrap into helper for reuse)
            yield simpy.events.AllOf(env, self._hierarchical_all_reduce(env, gpus, node_to_gpus, leaders))

            # -------- Backward --------
            # Step-5: Partial backward compute on each GPU
            compute_events = [env.process(g.compute(f"mb{mb}_bwd_partial", partial_flop)) for g in gpus]
            yield simpy.events.AllOf(env, compute_events)

            # Step-6~8: Hierarchical all-reduce of gradients
            yield simpy.events.AllOf(env, self._hierarchical_all_reduce(env, gpus, node_to_gpus, leaders))

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _hierarchical_all_reduce(self, env: simpy.Environment, gpus: List[GPU], node_to_gpus, leaders):
        """Return list of events representing intra-node reduce, inter-node ring, and broadcast."""
        events = []

        # Intra-node reduce (non-leader -> leader)
        for grp in node_to_gpus.values():
            leader = grp[0]
            for g in grp[1:]:
                events.append(env.process(g.send(leader.id, self.comm_size)))

        # Inter-node ring among leaders (if >1 nodes)
        if len(leaders) > 1:
            num_leaders = len(leaders)
            for idx, g in enumerate(leaders):
                dst = leaders[(idx + 1) % num_leaders].id
                events.append(env.process(g.send(dst, self.comm_size)))

        # Broadcast back within node
        for grp in node_to_gpus.values():
            leader = grp[0]
            for g in grp[1:]:
                events.append(env.process(leader.send(g.id, self.comm_size)))

        return events 