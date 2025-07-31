from __future__ import annotations

import simpy
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..trace.recorder import Recorder  # noqa: F401
    from ..net_model.base import BaseNetModel  # noqa: F401


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
    def compute(self, name: str, flop: float, *, scenario: Optional[str] = None,
                optimization_strategy: Optional[str] = None):
        """SimPy 进程：执行计算，并写入带完整字段的日志。"""

        start = self.env.now
        duration = flop / self.flops if self.flops > 0 else 0.0
        yield self.env.timeout(duration)
        end = self.env.now

        # 记录事件（compute 无通信量、带宽等字段）
        self.recorder.log_event(
            source_node=self.node_id,
            source_gpu=self.id,
            target_node=self.node_id,
            target_gpu=self.id,
            event_type="compute",
            data_type=name,
            start=start,
            end=end,
            data_size=0,
            scenario=scenario,
            optimization_strategy=optimization_strategy,
        )

    def send(
        self,
        dst_gpu: int,
        size: int,
        *,
        data_type: str = "activation",
        priority: int = 2,
        scenario: Optional[str] = None,
        optimization_strategy: Optional[str] = None,
    ):
        """SimPy 进程：执行 point-to-point 通信，并记录完整事件。

        Parameters
        ----------
        dst_gpu : int
            目标 GPU id。
        size : int
            传输数据量（bytes）。
        data_type : str
            activation / gradient / log 等。
        priority : int
            优先级（1=highest）。
        scenario / optimization_strategy : str | None
            仅在任务 1.3 中用于标记场景。
        """

        start = self.env.now

        # network.transfer 可能返回度量字典
        result = yield self.env.process(
            self.net_model.transfer(self.id, dst_gpu, size, self.nic_bw, priority=priority)
        )
        end = self.env.now

        # 解析可能的附加度量
        bw_used_pct: Optional[float] = None
        wait_ms: Optional[float] = None
        if isinstance(result, dict):
            share_bw = result.get("share_bw")  # bytes/sec
            queue_time = result.get("queue_time")  # sec
            total_bw = result.get("total_bw")  # bytes/sec
            if share_bw is not None and total_bw:
                bw_used_pct = share_bw / total_bw * 100.0
            if queue_time is not None:
                wait_ms = queue_time * 1e3

        # 写入 recorder
        dst_node = GPU._id_to_node.get(dst_gpu, -1)
        self.recorder.log_event(
            source_node=self.node_id,
            source_gpu=self.id,
            target_node=dst_node,
            target_gpu=dst_gpu,
            event_type="comm",
            data_type=data_type,
            start=start,
            end=end,
            data_size=size,
            bandwidth_used=bw_used_pct,
            wait_time=wait_ms,
            scenario=scenario,
            optimization_strategy=optimization_strategy,
        )

    def recv(self, src_gpu: int, size: int):
        """Explicit receive call (not currently used by strategies)."""
        start = self.env.now
        yield self.env.process(self.net_model.transfer(src_gpu, self.id, size, self.nic_bw))
        end = self.env.now
        self.recorder.log_event(
            source_node=GPU._id_to_node.get(src_gpu, -1),
            source_gpu=src_gpu,
            target_node=self.node_id,
            target_gpu=self.id,
            event_type="comm",
            data_type="recv_placeholder",
            start=start,
            end=end,
            data_size=size,
        ) 

    # ---------------------------------------------------------------------
    # Class-level helpers
    # ---------------------------------------------------------------------
    # Maintain a global mapping so that *send* can cheaply look up the node of
    # the destination GPU.
    _id_to_node: dict[int, int] = {} 