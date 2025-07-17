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