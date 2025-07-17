from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def plot_gantt(csv_path: str, *, show: bool = True, save_path: Optional[str] = None) -> None:
    """Render a Gantt chart from an event trace CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV trace produced by :class:`trace.recorder.Recorder`.
    show : bool, optional
        Whether to display the chart on screen.
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    df = pd.read_csv(csv_path)
    # Sort by start time for consistent drawing order
    df = df.sort_values("start")

    colors = {
        "compute": "tab:blue",
        "comm": "tab:orange",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in df.iterrows():
        ax.barh(y=row["gpu_id"], width=row["end"] - row["start"], left=row["start"],
                color=colors.get(row["type"], "gray"), edgecolor="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU ID")
    ax.set_title("GPU Activity Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig) 