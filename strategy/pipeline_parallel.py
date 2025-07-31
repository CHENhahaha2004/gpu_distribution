import simpy
from typing import List

from .base import ParallelStrategy
from ..simulator.gpu import GPU


class PipelineParallelStrategy(ParallelStrategy):
    """Pipeline parallelism with micro-batching (GPipe style) showing forward & backward."""

    def run(self, env: simpy.Environment, gpus: List[GPU], recorder):
        num_stages = len(gpus)
        flops_per_stage = self.flops_per_batch / num_stages
        act_size = self.comm_size  # both activations & gradients, simplified

        # ------------------------------------------------------------------
        # Helper structures – token events to synchronise between stages
        # fwd_ready[s][mb]  : activation mb has arrived at stage s (s>0)
        # grad_ready[s][mb] : gradient mb has arrived at stage s (s<num_stages-1)
        # ------------------------------------------------------------------
        fwd_ready = [
            [env.event() for _ in range(self.micro_batches)] for _ in range(num_stages)
        ]
        grad_ready = [
            [env.event() for _ in range(self.micro_batches)] for _ in range(num_stages)
        ]

        # ------------------------------------------------------------------
        # Per-stage process encapsulating forward & backward for all micro-batches
        # ------------------------------------------------------------------
        def stage_proc(stage_id: int):
            gpu = gpus[stage_id]

            # Forward passes for all micro-batches
            for mb in range(self.micro_batches):
                # Wait for activation from previous stage (except stage 0)
                if stage_id > 0:
                    yield fwd_ready[stage_id][mb]

                # Compute forward
                yield env.process(gpu.compute(f"mb{mb}_s{stage_id}_fwd", flops_per_stage))

                # Send activation to next stage (except last stage)
                if stage_id < num_stages - 1:
                    yield env.process(gpu.send(stage_id + 1, act_size, data_type="activation", priority=2))
                    # Notify next stage that activation is ready
                    fwd_ready[stage_id + 1][mb].succeed()

            # Backward passes (reverse micro-batch order for 1F1B overlap)
            for mb in reversed(range(self.micro_batches)):
                # Wait for gradient from next stage (except last stage)
                if stage_id < num_stages - 1:
                    yield grad_ready[stage_id][mb]

                # Compute backward
                yield env.process(gpu.compute(f"mb{mb}_s{stage_id}_bwd", flops_per_stage))

                # Send gradient to previous stage (except first stage)
                if stage_id > 0:
                    yield env.process(gpu.send(stage_id - 1, act_size, data_type="gradient", priority=1))
                    # Notify previous stage that gradient is ready
                    grad_ready[stage_id - 1][mb].succeed()

        # ------------------------------------------------------------------
        # Launch all stage processes and wait for completion
        # ------------------------------------------------------------------
        stage_events = [env.process(stage_proc(s)) for s in range(num_stages)]
        yield simpy.events.AllOf(env, stage_events) 