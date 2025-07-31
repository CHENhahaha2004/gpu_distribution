from __future__ import annotations

"""Fat-tree network model for the GPU simulator.

本实现并非学术级完整胖树，而是简化三层（edge / core）拓扑，
能够体现以下关键点：
    • 不同链路拥有独立带宽 / 时延与并发 capacity；
    • 每次 GPU → GPU 传输沿预先计算好的最短路径逐跳排队；

简化假设：
    1. 每个 *compute node*（由 ``gpu_to_node`` 提供）被视为『服务器』直接挂在
       所在 *edge switch* 上。
    2. 同一个 edge 下的两台服务器通信只经过 *一条* 下行链路；跨 edge / pod
       通信要先上行到 *core* 再下行。
    3. 所有链路均视为全双工，但两个方向互不影响（用独立 ``_Link`` 对象）。

若后续需要完整 k-ary fat tree，只需替换 ``_build_topology`` 产出的
``self._adj`` 与 ``self._links`` 数据结构，不影响调度逻辑。"""

from typing import Dict, List, Tuple, Optional, Set
import simpy

from .base import BaseNetModel

__all__ = ["FatTreeNet"]


class _Link:
    """Point-to-point link with limited concurrent transfers & simple stats."""

    def __init__(self, env: simpy.Environment, bandwidth: float, latency: float, capacity: int = 4):
        self.env = env
        self.bw = bandwidth          # bytes / sec
        self.lat = latency           # seconds
        self._res = simpy.Resource(env, capacity=capacity)

    # ------------------------ SimPy process ------------------------
    def transmit(self, size: int):
        """SimPy 进程：在本链路上传输 *size* 字节数据并返回度量字典。"""
        start = self.env.now
        with self._res.request() as req:
            yield req
            q_exit = self.env.now

            # 当前并发数 (<=capacity)
            concurrency = len(self._res.users)
            share_bw = self.bw / concurrency if concurrency > 0 else self.bw

            duration = self.lat + size / share_bw if share_bw > 0 else 0.0
            yield self.env.timeout(duration)
            end = self.env.now

        return {
            "share_bw": share_bw,
            "queue_time": q_exit - start,
            "total_bw": self.bw,
            "link_duration": end - start,
        }


class FatTreeNet(BaseNetModel):
    """Simplified fat-tree network model with deterministic shortest-path routing."""

    def __init__(
        self,
        env: simpy.Environment,
        gpu_to_node: Dict[int, int],
        *,
        k: int = 4,
        bw_edge: float = 25e9,
        bw_core: float = 25e9,
        lat_edge: float = 2e-6,
        lat_core: float = 5e-6,
        link_capacity: int = 4,
    ) -> None:
        super().__init__(env)
        if k % 2 != 0:
            raise ValueError("k 必须为偶数 (k-ary fat tree)")

        self.gpu_to_node = gpu_to_node  # GPU id -> compute-node id
        self._links: Dict[Tuple[str, str], _Link] = {}
        self._adj: Dict[str, Set[str]] = {}

        # ------------------------------------------------------------------
        # Build a *very* small 2-level fat tree (edge + core) 适配任意节点数
        # ------------------------------------------------------------------
        self._build_topology(k, bw_edge, bw_core, lat_edge, lat_core, link_capacity)

    # ------------------------------------------------------------------
    # Topology construction helpers
    # ------------------------------------------------------------------
    def _add_link(self, u: str, v: str, bw: float, lat: float, cap: int):
        """创建双向链路 (u <-> v)。"""
        link_fwd = _Link(self.env, bw, lat, cap)
        link_rev = _Link(self.env, bw, lat, cap)
        self._links[(u, v)] = link_fwd
        self._links[(v, u)] = link_rev
        self._adj.setdefault(u, set()).add(v)
        self._adj.setdefault(v, set()).add(u)

    def _build_topology(self, k: int, bw_e: float, bw_c: float, lat_e: float, lat_c: float, cap: int):
        """非常简化的 fat-tree：每个 *pod* 只有 1 个 edge switch，与单一 core 互联。"""
        num_nodes = len(set(self.gpu_to_node.values()))
        pods = [f"pod{pid}" for pid in range(k)]
        core_switch = "core"

        # Edge switch per pod
        edge_switches = {pod: f"edge_{pod}" for pod in pods}
        for esw in edge_switches.values():
            # 上联到 core
            self._add_link(esw, core_switch, bw_c, lat_c, cap)

        # Connect compute nodes to edge switches round-robin
        for node_id in range(num_nodes):
            pod = pods[node_id % k]
            esw = edge_switches[pod]
            srv = f"srv_{node_id}"
            self._add_link(srv, esw, bw_e, lat_e, cap)

        # 保存映射: compute-node id -> server node name
        self._node_to_srv: Dict[int, str] = {nid: f"srv_{nid}" for nid in range(num_nodes)}

    # ------------------------------------------------------------------
    # Shortest-path routing (BFS, unweighted)
    # ------------------------------------------------------------------
    def _shortest_path(self, src: str, dst: str) -> List[Tuple[str, str]]:
        if src == dst:
            return []
        from collections import deque

        visited = {src}
        parent: Dict[str, str] = {}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in self._adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    if v == dst:
                        queue.clear()
                        break
                    queue.append(v)
        # reconstruct
        path_nodes: List[str] = [dst]
        while path_nodes[-1] != src:
            path_nodes.append(parent[path_nodes[-1]])
        path_nodes.reverse()
        # convert to list of directed edges
        return list(zip(path_nodes[:-1], path_nodes[1:]))

    # ------------------------------------------------------------------
    # BaseNetModel 入口
    # ------------------------------------------------------------------
    def transfer(self, src: int, dst: int, size: int, bw_unused: float, *, priority: int = 2):  # noqa: D401
        """GPU→GPU 传输，沿最短路径逐跳排队。"""
        src_srv = self._node_to_srv[self.gpu_to_node[src]]
        dst_srv = self._node_to_srv[self.gpu_to_node[dst]]
        edge_list = self._shortest_path(src_srv, dst_srv)

        # 并行启动每条链路的 transmit，然后按串行顺序等待（逐跳）
        metrics: Dict[str, List] = {"share_bw": [], "queue_time": []}
        for u, v in edge_list:
            link = self._links[(u, v)]
            result = yield self.env.process(link.transmit(size))
            if isinstance(result, dict):
                metrics["share_bw"].append(result.get("share_bw"))
                metrics["queue_time"].append(result.get("queue_time"))

        # 可汇总返回给 GPU 日志；这里简单返回平均带宽占比、累积等待时间
        if metrics["share_bw"]:
            share_bw_avg = sum(metrics["share_bw"]) / len(metrics["share_bw"])
            queue_time_total = sum(metrics["queue_time"])
            total_bw = self._links[edge_list[0]].bw  # 所有链路带宽相同
            return {
                "share_bw": share_bw_avg,
                "queue_time": queue_time_total,
                "total_bw": total_bw,
            }
        return {} 