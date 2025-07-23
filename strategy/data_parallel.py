import simpy
from typing import List

from .base import ParallelStrategy
from simulator.gpu import GPU


class DataParallelStrategy(ParallelStrategy):
    """Data parallelism: replicate model, different data, gradient all-reduce."""

    def run(self, env: simpy.Environment, gpus: List[GPU], recorder):
        # Group GPUs by their residing node for hierarchical collectives
        from collections import defaultdict

        node_to_gpus = defaultdict(list)
        for g in gpus:
            node_to_gpus[g.node_id].append(g)

        # Pick the *leader* (first GPU) of each node for inter-node traffic
        leaders = [grp[0] for grp in node_to_gpus.values()]

        for mb in range(self.micro_batches):
            # --------------------------------------------------------------
            # Step-1: Local forward/backward on all GPUs (no change)
            # --------------------------------------------------------------
            compute_events = [env.process(g.compute(f"mb{mb}_compute", self.flops_per_batch)) for g in gpus]
            yield simpy.events.AllOf(env, compute_events)

            # --------------------------------------------------------------
            # Step-2: Intra-node gradient reduce – each non-leader GPU sends
            #         its gradients to the node leader. (internal comms)
            # --------------------------------------------------------------
            comm_events = []
            for node_id, grp in node_to_gpus.items():
                leader = grp[0]
                for g in grp[1:]:  # skip leader itself
                    comm_events.append(env.process(g.send(leader.id, self.comm_size)))
            yield simpy.events.AllOf(env, comm_events)

            # --------------------------------------------------------------
            # Step-3: Inter-node all-reduce among leaders (ring over leaders)
            # --------------------------------------------------------------
            if len(leaders) > 1:
                comm_events = []
                num_leaders = len(leaders)
                for idx, g in enumerate(leaders):
                    dst = leaders[(idx + 1) % num_leaders].id
                comm_events.append(env.process(g.send(dst, self.comm_size)))
                yield simpy.events.AllOf(env, comm_events)

            # --------------------------------------------------------------
            # Step-4: Broadcast reduced gradients back to local GPUs
            # --------------------------------------------------------------
            comm_events = []
            for node_id, grp in node_to_gpus.items():
                leader = grp[0]
                for g in grp[1:]:
                    comm_events.append(env.process(leader.send(g.id, self.comm_size)))
            yield simpy.events.AllOf(env, comm_events) 