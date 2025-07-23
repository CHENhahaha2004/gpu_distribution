from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Matplotlib backend (existing)
# -----------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------
# Plotly backend (interactive)
# -----------------------------------------------------------------------------

import plotly.express as px


def _insert_idle_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Given a trace dataframe, append *idle* rows for gaps in each GPU timeline."""
    records = []
    for gpu_id, group in df.groupby("gpu_id"):
        sorted_ev = group.sort_values("start").to_dict("records")
        # Leading idle
        if sorted_ev[0]["start"] > 0:
            records.append({
                "gpu_id": gpu_id,
                "type": "idle",
                "name": "idle",
                "start": 0.0,
                "end": sorted_ev[0]["start"],
            })
        # Gaps between events
        for prev, nxt in zip(sorted_ev, sorted_ev[1:]):
            if nxt["start"] > prev["end"]:
                records.append({
                    "gpu_id": gpu_id,
                    "type": "idle",
                    "name": "idle",
                    "start": prev["end"],
                    "end": nxt["start"],
                })
        # Trailing idle (optional): not needed for visualization of active time
    if records:
        df = pd.concat([df, pd.DataFrame.from_records(records)], ignore_index=True)
    return df


def plot_gantt_plotly(csv_path: str, *, show: bool = True, save_html: str | None = None) -> None:
    """Render an interactive Plotly Gantt chart from event CSV."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("start")

    # Insert idle intervals for better overview
    df = _insert_idle_intervals(df)

    color_map = {
        "compute": "#1f77b4",   # blue
        "comm": "#d62728",      # red
        "idle": "#aaaaaa",      # gray
    }

    fig = px.timeline(df, x_start="start", x_end="end", y="gpu_id", color="type",
                       color_discrete_map=color_map, hover_data=["name"])
    fig.update_yaxes(title="GPU ID", autorange="reversed")
    fig.update_xaxes(title="Time (s)", type="linear", range=[0, df["end"].max()])
    fig.update_layout(title="GPU Activity Timeline (Interactive)")

    if save_html:
        fig.write_html(save_html)
    if show:
        fig.show()

__all__ = [
    "plot_gantt",
    "plot_gantt_plotly",
    "plot_gantt_by_channel",
    "plot_gantt_split",
]


# -----------------------------------------------------------------------------
# Channel-split view (compute vs comm) – Plotly
# -----------------------------------------------------------------------------


def plot_gantt_by_channel(csv_path: str, *, show: bool = True, save_html: str | None = None) -> None:
    """Render interactive Gantt chart with two lanes: 计算通道 & 通信通道.

    Events across所有 GPU 会合并到对应通道，便于观察整体计算/通信时序。
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("start")

    # ---------------------------------
    # Extract micro-batch id for styling
    # ---------------------------------
    import re

    def extract_mb(name: str) -> int | None:
        m = re.search(r"mb(\d+)", str(name))
        return int(m.group(1)) if m else None

    df["mb"] = df["name"].apply(extract_mb)

    # Cycle through color shades per micro-batch to aid visual separation
    mb_palette = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#cab2d6", "#6a3d9a", "#ffff99", "#ff7f00"]

    import math

    def mb_color(mb_id, default: str):
        """Return palette color based on micro-batch id, fallback to *default*.

        Handles *None*, *NaN* or non-int convertible values gracefully.
        """
        try:
            if mb_id is None or (isinstance(mb_id, float) and math.isnan(mb_id)):
                return default
            idx = int(mb_id)
            return mb_palette[idx % len(mb_palette)]
        except Exception:
            return default

    # Map to channel names (English)
    channel_map = {
        "compute": "Compute",
        "comm": "Communication",
    }
    df["channel"] = df["type"].map(channel_map).fillna("Other")

    # Base color map per channel
    color_map = {
        "Compute": "#1f77b4",
        "Communication": "#d62728",
        "Other": "#aaaaaa",
    }

    # --------------------------------------------------
    # Merge overlapping intervals per channel to avoid
    # overplotting identical bars (improves visibility)
    # --------------------------------------------------
    agg_records = []
    for ch, grp in df.groupby("channel"):
        intervals = grp[["start", "end"]].sort_values("start").to_numpy()
        if len(intervals) == 0:
            continue
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:  # overlap
                cur_e = max(cur_e, e)
            else:
                agg_records.append({"channel": ch, "start": cur_s, "end": cur_e})
                cur_s, cur_e = s, e
        agg_records.append({"channel": ch, "start": cur_s, "end": cur_e})

    agg_df = pd.DataFrame.from_records(agg_records)

    # Hover shows duration
    agg_df["hover"] = agg_df.apply(lambda r: f"{r['channel']} {r['end']-r['start']:.4f}s", axis=1)

    # Use channel names as color categories

    fig = px.timeline(agg_df, x_start="start", x_end="end", y="channel", color="channel",
                       color_discrete_map=color_map,
                       hover_data=["hover"])
    fig.update_yaxes(categoryorder="array", categoryarray=["Communication", "Compute"])
    fig.update_xaxes(title="Time (s)", type="linear", range=[0, df["end"].max()])

    # Add dashed lines at micro-batch boundaries
    if df["mb"].notna().any():
        mb_boundaries = (df.groupby("mb")["end"].max().sort_index().values)
        for t in mb_boundaries[:-1]:  # skip last boundary
            fig.add_vline(x=t, line_dash="dash", line_color="red", opacity=0.7)

    fig.update_layout(title="Compute / Communication Channel View")

    if save_html:
        fig.write_html(save_html)
    if show:
        fig.show() 

# -----------------------------------------------------------------------------
# Split view (Matplotlib): 通信通道 + GPU 计算通道
# -----------------------------------------------------------------------------


def plot_gantt_split(csv_path: str, *, show: bool = True, save_path: str | None = None) -> None:  # noqa: D401
    """使用 Matplotlib 绘制 *Split View*，避免 Plotly 时间轴解析问题。

    - y=0  : 通信通道（聚合所有 comm 事件）
    - y=1… : 各 GPU 计算通道

    颜色规则：
    • forward   -> 蓝色
    • backward  -> 绿色
    • intra     -> 橙色
    • inter     -> 红色
    • 其他      -> 灰色
    """

    df = pd.read_csv(csv_path)

    # ----------------------------------
    # 分类与颜色
    # ----------------------------------
    def classify_phase(row):
        if row["type"] == "compute":
            name = str(row["name"]).lower()
            if "bwd" in name or "backward" in name:
                return "backward"
            # default all other compute as forward
            return "forward"
        else:  # comm
            name = str(row["name"]).lower()
            if "inter" in name:
                return "inter"
            return "intra"  # default classify as intra if not explicit

    # Extract micro-batch id (assumes pattern mb{num}_ in name)
    import re
    def extract_mb(name: str):
        m = re.search(r"mb(\d+)", str(name))
        return int(m.group(1)) if m else None

    df["mb"] = df["name"].apply(extract_mb)

    df["phase"] = df.apply(classify_phase, axis=1)

    # Base palettes for forward/backward alternating colors per micro-batch
    # Alternate between two colors for better visual contrast without getting too light
    fwd_palette = ["#1f77b4", "#4e79b7"]  # dark & medium blue
    bwd_palette = ["#2ca02c", "#4daf4a"]  # dark & medium green

    color_map = {
        "other": "#7f7f7f",
        "intra": "#ff7f0e",
        "inter": "#9e9e9e",
    }

    # ----------------------------------
    # y-axis mapping: Communication rows split into intra / inter
    # ----------------------------------
    gpu_ids = sorted(df[df["type"] == "compute"]["gpu_id"].unique())
    y_map = {"comm_intra": 0, "comm_inter": 1}
    for idx, gid in enumerate(gpu_ids, start=2):  # GPUs start after two comm rows
        y_map[gid] = idx

    # Figure size based on GPU count
    height = 1.0 + 0.6 * len(gpu_ids)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, height))

    # Precompute timeline span for dynamic min-width
    total_span = df["end"].max() - df["start"].min()
    min_width = total_span * 0.005  # 1% of overall span

    # Draw events with visibility boost for comm intervals
    for _, r in df.sort_values("start").iterrows():
        if r["type"] == "comm":
            y = y_map["comm_inter"] if r["phase"] == "inter" else y_map["comm_intra"]
        else:
            y = y_map[r["gpu_id"]]
        duration = r["end"] - r["start"]
        # Enforce minimum visual width for comm events
        if r["type"] == "comm" and duration < min_width:
            width = min_width
        else:
            width = duration
        # Determine color
        phase = r["phase"]
        if phase == "forward":
            mb_raw = r["mb"]
            mb_id = 0
            try:
                import math, numpy as np
                if mb_raw is not None and not (isinstance(mb_raw, float) and math.isnan(mb_raw)):
                    mb_id = int(mb_raw)
            except Exception:
                mb_id = 0
            color = fwd_palette[mb_id % len(fwd_palette)]
        elif phase == "backward":
            mb_raw = r["mb"]
            mb_id = 0
            try:
                import math, numpy as np
                if mb_raw is not None and not (isinstance(mb_raw, float) and math.isnan(mb_raw)):
                    mb_id = int(mb_raw)
            except Exception:
                mb_id = 0
            color = bwd_palette[mb_id % len(bwd_palette)]
        else:
            color = color_map.get(phase, "#cccccc")

        # Use hatch for inter-node communication to make it pop out
        hatch = "//" if phase == "inter" else None

        bar = ax.barh(y=y, width=width, left=r["start"],
                      color=color, edgecolor="black")
        if hatch:
            # Apply hatch pattern to bar container
            for patch in bar:
                patch.set_hatch(hatch)

    # Y ticks & labels
    y_ticks = [y_map["comm_intra"], y_map["comm_inter"]] + [y_map[gid] for gid in gpu_ids]
    y_labels = ["Comm-Intra", "Comm-Inter"] + [f"GPU{gid}" for gid in gpu_ids]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Time (s)")
    ax.set_title("GPU Compute / Communication Timeline (Split View)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch

    # Build legend: forward/backward sample, intra/inter
    legend_handles = [
        Patch(color=fwd_palette[0], label="Forward (mb0)") ,
        Patch(color=bwd_palette[0], label="Backward (mb0)"),
        Patch(color=color_map["intra"], label="Intra-node Comm"),
        Patch(facecolor=color_map["inter"], hatch="//", edgecolor="black", label="Inter-node Comm"),
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return fig 