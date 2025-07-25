import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# -----------------------------------------------------------------------------
# RDMA link metrics visualisation helpers
# -----------------------------------------------------------------------------


def _to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Safely convert *records* list[dict] to DataFrame (empty if none)."""
    if records:
        return pd.DataFrame.from_records(records)
    # ensure columns exist even when empty to avoid KeyError downstream
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 1) 折线图：时间 vs 链路流量 / 并发通信数 / 排队队列长度
# -----------------------------------------------------------------------------

def plot_link_utilisation(util_records: List[Dict[str, Any]], *, show: bool = True,
                           save_path: Optional[str] = None) -> None:
    """Draw line chart for throughput / concurrency / queue length."""
    df = _to_df(util_records)
    if df.empty:
        print("[plot_link_utilisation] 无数据可绘制 util_records 为空")
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Convert throughput to GB/s for readability
    ax1.plot(df["time"], df["throughput"] / 1e9, label="Throughput (GB/s)", color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Throughput (GB/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Secondary axis for concurrency & queue length
    ax2 = ax1.twinx()
    ax2.plot(df["time"], df["concurrency"], label="Concurrency", color="tab:orange")
    ax2.plot(df["time"], df["queue_len"], label="Queue length", color="tab:green")
    ax2.set_ylabel("Count", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    # Combine legends
    lines, labels = [], []
    for ax in (ax1, ax2):
        lns, lbls = ax.get_legend_handles_labels()
        lines.extend(lns)
        labels.extend(lbls)
    ax1.legend(lines, labels, loc="upper right")

    plt.title("RDMA Link Utilisation Over Time")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2) 柱状图：单次通信排队 + 传输耗时
# -----------------------------------------------------------------------------

def plot_comm_durations(trans_records: List[Dict[str, Any]], *, show: bool = True,
                         save_path: Optional[str] = None, top_n: Optional[int] = None) -> None:
    """Stacked bar chart of queue vs transfer time per communication event.

    Parameters
    ----------
    trans_records : list[dict]
        Output of RDMANet.link_metrics()["transfers"].
    top_n : int, optional
        Limit to first *n* events (keep timeline order) to avoid overcrowding.
    """
    df = _to_df(trans_records).reset_index().rename(columns={"index": "event"})
    if df.empty:
        print("[plot_comm_durations] 无数据可绘制 trans_records 为空")
        return

    if top_n is not None:
        df = df.head(top_n)

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.4), 4))

    ax.bar(df["event"], df["queue_time"], label="Queue", color="tab:red")
    ax.bar(df["event"], df["transfer_time"], bottom=df["queue_time"],
           label="Transfer", color="tab:purple")

    ax.set_xlabel("Comm Event Index (Timeline order)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-Communication Delay Breakdown")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3) 表格：通信耗时明细
# -----------------------------------------------------------------------------

def comm_duration_table(trans_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return pandas DataFrame with computed durations & ratios."""
    df = _to_df(trans_records).copy()
    if df.empty:
        print("[comm_duration_table] trans_records 为空 – 返回空 DataFrame")
        return df

    df["total_time"] = df["queue_time"] + df["transfer_time"]
    df["queue_pct"] = df["queue_time"] / df["total_time"]
    df["transfer_pct"] = df["transfer_time"] / df["total_time"]
    return df[["start", "queue_time", "transfer_time", "total_time", "queue_pct", "transfer_pct"]]


__all__ = [
    "plot_link_utilisation",
    "plot_comm_durations",
    "comm_duration_table",
] 