import simpy
from abc import ABC, abstractmethod


class BaseNetModel(ABC):
    """Abstract base class representing a network fabric used by GPUs.

    Subclasses implement :py:meth:`transfer` which should *yield* a
    :class:`simpy.events.Event` representing the completion of the data
    transfer. This design allows later replacement with more sophisticated
    models (e.g. RDMA with contention, latency jitter, congestion control)
    without touching caller code.
    """

    def __init__(self, env: simpy.Environment):
        self.env = env

    # ------------------------------------------------------------------
    # Interface – must be implemented by concrete subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def transfer(self, src: int, dst: int, size: int, bw: float, *, priority: int = 2):
        """Return a SimPy *event* for a point-to-point transfer.

        Parameters
        ----------
        src : int
            Source GPU id.
        dst : int
            Destination GPU id.
        size : int
            Payload size in **bytes**.
        bw : float
            Nominal NIC bandwidth (bytes/sec) available to *src* GPU.
        """
        raise NotImplementedError 