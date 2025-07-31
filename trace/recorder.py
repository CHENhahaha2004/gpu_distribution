from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional


class Recorder:
    """Collects simulation events and exports them for analysis/visualization."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._next_eid: int = 0  # 全局自增 event_id

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 兼容旧接口：仍允许简单记录 (GPU 视角)
    # ------------------------------------------------------------------
    def log(self, gpu_id: int, event_type: str, name: str, start: float, end: float) -> None:
        """旧版接口，仅在 *run_sim.py* 中的概览视图需要，用 minimal 字段记录。"""

        self._records.append({
            "event_id": self._next_eid,
            "source_node": -1,
            "source_gpu": gpu_id,
            "target_node": -1,
            "target_gpu": -1,
            "event_type": event_type,
            "data_type": name,  # 直接塞进 data_type，方便后处理；compute 可存算子名
            "start_time": start,
            "end_time": end,
            "data_size": 0,
            # legacy columns
            "gpu_id": gpu_id,
            "type": event_type,
            "name": name,
            "start": start,
            "end": end,
        })
        self._next_eid += 1

    # ------------------------------------------------------------------
    # 新接口：实验要求的完整字段
    # ------------------------------------------------------------------
    def log_event(
        self,
        *,
        source_node: int,
        source_gpu: int,
        target_node: int,
        target_gpu: int,
        event_type: str,
        data_type: str,
        start: float,
        end: float,
        data_size: int,
        bandwidth_used: Optional[float] = None,
        wait_time: Optional[float] = None,
        scenario: Optional[str] = None,
        optimization_strategy: Optional[str] = None,
    ) -> None:
        """记录带有完整字段的一条事件。各数值单位均符合实验描述。"""

        record: Dict[str, Any] = {
            "event_id": self._next_eid,
            "source_node": source_node,
            "source_gpu": source_gpu,
            "target_node": target_node,
            "target_gpu": target_gpu,
            "event_type": event_type,
            "data_type": data_type,
            "start_time": start,
            "end_time": end,
            "data_size": data_size,
            # 兼容旧版字段，用于已有可视化/统计逻辑--------------------------------
            "gpu_id": source_gpu,
            "type": event_type,
            "name": data_type,
            "start": start,
            "end": end,
        }

        # 可选字段按需写入（保持列统一）
        if bandwidth_used is not None:
            record["bandwidth_used"] = bandwidth_used
        if wait_time is not None:
            record["wait_time"] = wait_time
        if scenario is not None:
            record["scenario"] = scenario
        if optimization_strategy is not None:
            record["optimization_strategy"] = optimization_strategy

        self._records.append(record)
        self._next_eid += 1

    def to_csv(self, path: str) -> None:
        """将记录写入 *path*，列集取自所有记录并保持稳定顺序。"""

        if not self._records:
            return

        # 根据首次记录收集所有字段，以保证列顺序固定且兼容下游
        fieldnames = list(self._records[0].keys())
        # 若后续记录出现新字段，则补到末尾
        for rec in self._records[1:]:
            for k in rec.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return basic metrics aggregated from the trace.

        Returns
        -------
        dict
            Keys include:
            - total_time: overall makespan (seconds)
            - compute_time: sum of all compute durations across GPUs
            - comm_time: sum of all communication durations across GPUs
            - comm_ratio: comm_time / (compute_time + comm_time)
        """
        if not self._records:
            return {}

        # 兼容两种字段命名
        if "end_time" in self._records[0]:
            # 新格式
            total_time = max(r["end_time"] for r in self._records)
            compute_time = sum((r["end_time"] - r["start_time"]) for r in self._records if r["event_type"] == "compute")
            comm_time = sum((r["end_time"] - r["start_time"]) for r in self._records if r["event_type"] == "comm")
        else:
            total_time = max(r["end_time"] for r in self._records)
            compute_time = sum((r["end_time"] - r["start_time"]) for r in self._records if r["event_type"] == "compute")
            comm_time = sum((r["end_time"] - r["start_time"]) for r in self._records if r["event_type"] == "comm")
        denom = compute_time + comm_time
        comm_ratio = comm_time / denom if denom > 0 else 0.0

        return {
            "total_time": total_time,
            "compute_time": compute_time,
            "comm_time": comm_time,
            "comm_ratio": comm_ratio,
        } 