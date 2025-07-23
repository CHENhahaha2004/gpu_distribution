from __future__ import annotations

from typing import Callable, Dict, Optional

import simpy

from .base import BaseNetModel

# Type alias for a link-level scheduler function
LinkScheduler = Callable[[simpy.Environment, int, int, int, float, float], simpy.events.Event]


class RDMANet(BaseNetModel):
    """RDMA network model with intra/inter-node parameters and pluggable scheduler.

    Default scheduler is *ideal* (no contention): time = latency + size / bw.
    Down-the-road features such as shared-link contention、queueing、credit-based flow
    control can be injected by supplying a custom *scheduler* callable, or by calling
    :py:meth:`register_scheduler` with a replacement.
    """

    def __init__(
        self,
        env: simpy.Environment,
        gpu_to_node: Dict[int, int],
        *,
        intra_bw: float = 100e9,
        inter_bw: float = 25e9,
        intra_lat: float = 5e-6,
        inter_lat: float = 2.5e-5,
        scheduler: Optional[LinkScheduler] = None,
    ) -> None:
        super().__init__(env)
        self.gpu_to_node = gpu_to_node
        self.intra_bw = intra_bw
        self.inter_bw = inter_bw
        self.intra_lat = intra_lat
        self.inter_lat = inter_lat

        # Link scheduler determines how a (src, dst) pair experiences delay
        self._scheduler: LinkScheduler = scheduler or self._ideal_scheduler

    # ------------------------------------------------------------------
    # Public API – extension hook
    # ------------------------------------------------------------------
    def register_scheduler(self, scheduler: LinkScheduler) -> None:
        """Replace internal link scheduler with *scheduler*.

        Parameters
        ----------
        scheduler : Callable
            Signature ``(env, src, dst, size, bw, latency) -> simpy.Event``.
        """
        self._scheduler = scheduler

    # ------------------------------------------------------------------
    # BaseNetModel implementation
    # ------------------------------------------------------------------
    def transfer(self, src: int, dst: int, size: int, bw_unused: float):  # noqa: D401
        src_node = self.gpu_to_node[src]
        dst_node = self.gpu_to_node[dst]

        if src_node == dst_node:
            bw = self.intra_bw
            latency = self.intra_lat
        else:
            bw = self.inter_bw
            latency = self.inter_lat

        # Delegate to active scheduler (may model queueing / contention)
        yield self.env.process(self._scheduler(self.env, src, dst, size, bw, latency))

    # ------------------------------------------------------------------
    # Built-in link scheduler (no contention)
    # ------------------------------------------------------------------
    @staticmethod
    def _ideal_scheduler(env: simpy.Environment, src: int, dst: int, size: int, bw: float, latency: float):
        """Ideal delay = latency + size / bw."""
        duration = latency + (size / bw if bw > 0 else 0.0)
        yield env.timeout(duration)
