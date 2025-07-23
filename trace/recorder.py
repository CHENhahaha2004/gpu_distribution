import csv
from typing import Any, Dict, List


class Recorder:
    """Collects simulation events and exports them for analysis/visualization."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def log(self, gpu_id: int, event_type: str, name: str, start: float, end: float) -> None:
        self._records.append({
            "gpu_id": gpu_id,
            "type": event_type,
            "name": name,
            "start": start,
            "end": end,
        })

    def to_csv(self, path: str) -> None:
        """Write the trace into *path* as a CSV file."""
        fieldnames = ["gpu_id", "type", "name", "start", "end"]
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

        total_time = max(r["end"] for r in self._records)
        compute_time = sum((r["end"] - r["start"]) for r in self._records if r["type"] == "compute")
        comm_time = sum((r["end"] - r["start"]) for r in self._records if r["type"] == "comm")
        denom = compute_time + comm_time
        comm_ratio = comm_time / denom if denom > 0 else 0.0

        return {
            "total_time": total_time,
            "compute_time": compute_time,
            "comm_time": comm_time,
            "comm_ratio": comm_ratio,
        } 