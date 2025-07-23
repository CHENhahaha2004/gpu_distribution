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

        # Register mapping (used by send() to identify intra/inter-node links)
        GPU._id_to_node[self.id] = self.node_id

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
        """Yield a *communication* process via the provided network model.

        This variant additionally annotates whether the transfer is intra-node
        or inter-node so that downstream analysis can easily distinguish the
        two cases without changing the Recorder interface.
        """
        start = self.env.now
        # Delegate to network model which can insert contention/delay.
        yield self.env.process(self.net_model.transfer(self.id, dst_gpu, size, self.nic_bw))
        end = self.env.now

        # -----------------------------
        # Determine comm scope
        # -----------------------------
        # GPU instances may live on different nodes. We record the scope in the
        # *name* field so that existing aggregation logic (which relies on the
        # *type* field being exactly "compute"/"comm") remains untouched.
        dst_node = getattr(self, "_id_to_node", {}).get(dst_gpu)
        scope = "intra" if dst_node == self.node_id else "inter"
        name_suffix = f" ({scope})"

        # Record on sender side
        self.recorder.log(self.id, "comm", f"{self.id}->{dst_gpu}{name_suffix}", start, end)
        # Also record the corresponding receive event on dst GPU for timeline completeness
        self.recorder.log(dst_gpu, "comm", f"{self.id}->{dst_gpu}{name_suffix}", start, end)

    def recv(self, src_gpu: int, size: int):
        """Explicit receive call (not currently used by strategies)."""
        start = self.env.now
        yield self.env.process(self.net_model.transfer(src_gpu, self.id, size, self.nic_bw))
        end = self.env.now
        self.recorder.log(self.id, "comm", f"{src_gpu}->{self.id}", start, end) 

    # ---------------------------------------------------------------------
    # Class-level helpers
    # ---------------------------------------------------------------------
    # Maintain a global mapping so that *send* can cheaply look up the node of
    # the destination GPU.
    _id_to_node: dict[int, int] = {} 