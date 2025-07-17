import simpy
from typing import List

from .base import ParallelStrategy
from simulator.gpu import GPU


class PipelineParallelStrategy(ParallelStrategy):
    """Pipeline parallelism across GPUs (each GPU is a stage)."""

    def run(self, env: simpy.Environment, gpus: List[GPU], recorder):
        num_stages = len(gpus)
        flops_per_stage = self.flops_per_batch / num_stages
        activation_size = self.comm_size  # simplistic assumption

        def stage_proc(stage_id: int, micro_batch: int):
            gpu = gpus[stage_id]
            # Compute for this stage
            yield env.process(gpu.compute(f"mb{micro_batch}_stage{stage_id}_compute", flops_per_stage))
            # Send activations to next stage unless last stage
            if stage_id < num_stages - 1:
                yield env.process(gpu.send(stage_id + 1, activation_size))

        # Launch pipeline for each micro batch
        for mb in range(self.micro_batches):
            prev_event = None
            for stage_id in range(num_stages):
                # Define nested function to capture current prev_event
                def launch(stage=stage_id, mb_idx=mb, wait_event=prev_event):
                    if wait_event is not None:
                        yield wait_event  # wait for dependency
                    yield env.process(stage_proc(stage, mb_idx))
                prev_event = env.process(launch())
            # Wait for last stage of this micro-batch to finish before next micro-batch
            yield prev_event 