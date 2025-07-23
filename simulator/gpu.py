import simpy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from trace.recorder import Recorder  # noqa: F401
    from net_model.base import BaseNetModel  # noqa: F401


class GPU:
    """A minimal GPU abstraction used by the simulator.

    Parameters
    ----------
    env : simpy.Environment
        The simulation environment.
    gpu_id : int
        Identifier for the GPU.
    flops : float
        Peak floating-point operations per second of the GPU.
    nic_bw : float
        Network interface bandwidth (bytes/sec).
    recorder : Recorder
        Trace recorder instance.
    net_model : BaseNetModel
        Network model responsible for point-to-point transfers.
    """

    def __init__(self, env: simpy.Environment, gpu_id: int, node_id: int, flops: float, nic_bw: float,
                 recorder: "Recorder", net_model: "BaseNetModel") -> None:
        self.env = env
        self.id = gpu_id
        self.node_id = node_id
        self.flops = flops  # FLOPS (float operations per second)
        self.nic_bw = nic_bw  # bytes / second
        self.recorder = recorder
        self.net_model = net_model

    # ---------------------------------------------------------------------
    # Public SimPy processes
    # ---------------------------------------------------------------------
    def compute(self, name: str, flop: float):
        """Yield a *compute* process that takes time proportional to FLOPs."""
        start = self.env.now
        duration = flop / self.flops if self.flops > 0 else 0.0
        yield self.env.timeout(duration)
        end = self.env.now
        self.recorder.log(self.id, "compute", name, start, end)

    def send(self, dst_gpu: int, size: int):
        """Yield a *communication* process via the provided network model."""
        start = self.env.now
        # Delegate to network model which can insert contention/delay.
        yield self.env.process(self.net_model.transfer(self.id, dst_gpu, size, self.nic_bw))
        end = self.env.now
        # Record on sender side
        self.recorder.log(self.id, "comm", f"{self.id}->{dst_gpu}", start, end)
        # Also record the corresponding receive event on dst GPU for timeline completeness
        self.recorder.log(dst_gpu, "comm", f"{self.id}->{dst_gpu}", start, end)

    def recv(self, src_gpu: int, size: int):
        """Explicit receive call (not currently used by strategies)."""
        start = self.env.now
        yield self.env.process(self.net_model.transfer(src_gpu, self.id, size, self.nic_bw))
        end = self.env.now
        self.recorder.log(self.id, "comm", f"{src_gpu}->{self.id}", start, end) 