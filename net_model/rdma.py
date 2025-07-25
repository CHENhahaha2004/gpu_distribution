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

        # ------------------------------------------------------------------
        # Internal shared link modelling RDMA contention across all GPUs
        # ------------------------------------------------------------------
        # According to the lab spec the physical RDMA fabric is modelled as a
        # *single* full-duplex link shared by all communications with:
        #   • base latency   : 10 µs (provided as *latency* arg to scheduler)
        #   • total bandwidth: 100 Gbps  (≈12.5 GB/s)
        #   • max concurrency: 4 inflight transfers; surplus queue in FIFO order
        #
        # The link object encapsulates a SimPy Resource to arbitrate access and
        # exposes a *transfer* coroutine that realises the timing semantics and
        # captures rich metrics for later visualisation.
        # ------------------------------------------------------------------

        class _RDMALink:
            """Shared RDMA link with limited concurrency & statistics."""

            def __init__(self, env: simpy.Environment, bandwidth: float, base_latency: float, max_conc: int = 4):
                self.env = env
                self.total_bw = bandwidth  # bytes/sec
                self.base_lat = base_latency  # seconds
                self._res = simpy.Resource(env, capacity=max_conc)

                # Metrics ---------------------------------------------------
                self._inflight = 0  # current active transfers (<=capacity)
                self._records: list[dict] = []  # per-transfer metrics
                self._utilisation: list[dict] = []  # timeline of link state

                # Helper: log current state (called on every state change)
                def _log_state():
                    self._utilisation.append({
                        "time": self.env.now,
                        "concurrency": self._inflight,
                        "queue_len": len(self._res.queue),
                        "throughput": self.total_bw if self._inflight > 0 else 0.0,
                    })

                self._log_state = _log_state  # attach for reuse

            # -----------------------------------------
            # Public helpers
            # -----------------------------------------
            def stats_transfer(self) -> list[dict]:
                return self._records

            def stats_link(self) -> list[dict]:
                return self._utilisation

            # -----------------------------------------
            # SimPy processes
            # -----------------------------------------
            def transfer(self, size: int):
                """SimPy process representing a single RDMA P2P transfer."""
                start = self.env.now

                with self._res.request() as req:
                    yield req  # waiting constitutes *queueing* time
                    q_exit = self.env.now

                    # state: one request has left queue & entered active
                    self._inflight += 1
                    self._log_state()

                    concurrency = self._inflight
                    share_bw = self.total_bw / concurrency  # bytes/sec each

                    # Transmission (includes base latency)
                    tx_time = size / share_bw if share_bw > 0 else 0.0
                    duration = self.base_lat + tx_time
                    yield self.env.timeout(duration)

                    end = self.env.now

                    # Persist per-comm metrics
                    self._records.append({
                        "start": start,
                        "queue_time": q_exit - start,
                        "transfer_time": end - q_exit,
                        "end": end,
                    })

                    # Release capacity
                    self._inflight -= 1
                    self._log_state()

        # Instantiate a single shared link (inter- & intra-node unified model)
        self._link = _RDMALink(env, bandwidth=100 * 1e9 / 8, base_latency=10e-6)  # 100 Gbps → bytes/sec

        # Choose scheduler: user-supplied overrides built-in RDMA contention model
        self._scheduler: LinkScheduler = scheduler or self._rdma_scheduler

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------
    def link_metrics(self) -> dict[str, list[dict]]:
        """Return dictionaries containing per-transfer & per-second link stats."""
        return {
            "transfers": self._link.stats_transfer(),
            "utilisation": self._link.stats_link(),
        }

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

    # ------------------------------------------------------------------
    # Built-in contention-aware scheduler (default for RDMANet)
    # ------------------------------------------------------------------
    def _rdma_scheduler(self, env: simpy.Environment, src: int, dst: int, size: int, bw: float, latency: float):
        """Wrap the shared *_link.transfer* coroutine so that signature matches
        the *LinkScheduler* protocol expected by :py:meth:`transfer`. The *bw*
        & *latency* parameters are ignored because the shared link already
        encodes them via *total_bw* & *base_lat* as specified by the lab.
        """

        # NOTE: For now intra-/inter-node distinction is ignored ‑- all traffic
        # shares the same link as the bottleneck, matching the experimental
        # focus on cross-machine RDMA contention.

        yield from self._link.transfer(size)
