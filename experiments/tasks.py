"""Lab tasks 1.1~1.3 high-level helpers.

Each *run_taskX* function spins up a fresh SimPy environment, configures
network / strategy according to specification, and dumps trace CSVs under
`trace/` folder so that downstream visualisation & analysis scripts can
pick them up.

该模块对 CLI 参数零依赖，可直接在其他脚本或单元测试中调用。"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

import simpy

from ..simulator.gpu import GPU
from ..net_model.ideal import IdealNet
from ..net_model.rdma import RDMANet
from ..strategy import get_strategy
from ..trace.recorder import Recorder

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _ensure_trace_dir() -> Path:
    path = Path(__file__).resolve().parent.parent / "trace"
    path.mkdir(exist_ok=True)
    return path


def _setup_cluster(env: simpy.Environment, recorder: Recorder, *, nodes: int, gpus_per_node: int, flops: float,
                   nic_bw: float, net_model) -> list[GPU]:
    """Create GPUs & populate global id→node mapping."""
    gpus: list[GPU] = []
    for gid in range(nodes * gpus_per_node):
        gpus.append(
            GPU(
                env,
                gpu_id=gid,
                node_id=gid // gpus_per_node,
                flops=flops,
                nic_bw=nic_bw,
                recorder=recorder,
                net_model=net_model,
            )
        )
    return gpus

# -----------------------------------------------------------------------------
# Baseline & optimised RDMA schedulers used in task-1.3
# -----------------------------------------------------------------------------

def make_baseline_scheduler(link):
    """Return baseline scheduler with ±20% latency jitter & congestion (>80% util)."""

    def _scheduler(env, src, dst, size, bw, latency, priority):  # noqa: D401
        # 添加随机延迟抖动
        latency *= 1 + random.uniform(-0.2, 0.2)

        # 基线延迟 (含抖动)
        yield env.timeout(latency)

        # 走共享队列，拿到真实传输 cost（已包含 10µs 固定 base_lat）
        metrics = yield from link.transfer(size)

        # 拥塞判定 & 额外等待
        util = metrics["share_bw"] / metrics["total_bw"]  # 0~1
        if util > 0.8:
            residual_bw = metrics["total_bw"] - metrics["share_bw"]
            if residual_bw > 0:
                extra = size / residual_bw
                yield env.timeout(extra)
                metrics["queue_time"] += extra
        return metrics

    return _scheduler


def make_priority_scheduler(link):
    """简化优先级调度器：1 级梯度 > 2 级激活 > 3 级日志。"""

    # 按优先级分组的请求队列
    queues = {1: [], 2: [], 3: []}
    active_requests = []  # 正在处理的请求

    def _scheduler(env, src, dst, size, bw, latency, priority):  # noqa: D401
        # 创建完成事件
        completion_event = env.event()
        
        # 将请求加入对应优先级队列
        queues[priority].append((size, completion_event))
        
        # 如果没有正在处理的请求，开始处理队列
        if not active_requests:
            env.process(_process_queues(env))
        
        # 等待完成
        metrics = yield completion_event
        return metrics

    def _process_queues(env):
        """处理所有队列中的请求，按优先级顺序。"""
        while any(queues.values()):  # 只要还有队列不为空
            # 按优先级 1->2->3 处理
            for priority in (1, 2, 3):
                while queues[priority]:  # 处理当前优先级的所有请求
                    size, completion_event = queues[priority].pop(0)
                    active_requests.append(completion_event)
                    
                    # 执行传输
                    transfer_metrics = yield from link.transfer(size)
                    
                    # 标记完成
                    completion_event.succeed(transfer_metrics)
                    active_requests.remove(completion_event)

    return _scheduler

# -----------------------------------------------------------------------------
# Task entry-points
# -----------------------------------------------------------------------------

def run_task1_1(*, output_dir: str | os.PathLike | None = None, strategy_name: str = "pipeline"):
    """基础并行模拟（理想网络）。"""
    output_dir = Path(output_dir or _ensure_trace_dir())

    env = simpy.Environment()
    recorder = Recorder()

    # Cluster config (4×4)
    nodes, gpus_per_node = 4, 4
    net = IdealNet(env)
    gpus = _setup_cluster(env, recorder, nodes=nodes, gpus_per_node=gpus_per_node, flops=1e13, nic_bw=100e9, net_model=net)

    # Pipeline strategy – 4 stage, 4 micro-batch / 2 mini-batch align to spec
    strategy = get_strategy(strategy_name, micro_batches=4, flops_per_batch=10e9, comm_size=8 * 1024 * 1024)
    env.process(strategy.run(env, gpus, recorder))
    env.run()

    out_path = output_dir / f"task1_1_{strategy_name}.csv"
    recorder.to_csv(str(out_path))
    print(f"[Task1.1] Trace saved → {out_path.relative_to(output_dir.parent)}")


def run_task1_2(*, output_dir: str | os.PathLike | None = None, strategy_name: str = "pipeline"):
    """RDMA 网络模拟（无抖动，有资源争用）"""
    output_dir = Path(output_dir or _ensure_trace_dir())

    env = simpy.Environment()
    recorder = Recorder()

    nodes, gpus_per_node = 4, 4
    gpu_to_node = {gid: gid // gpus_per_node for gid in range(nodes * gpus_per_node)}

    net = RDMANet(env, gpu_to_node)
    gpus = _setup_cluster(env, recorder, nodes=nodes, gpus_per_node=gpus_per_node, flops=1e13, nic_bw=100e9, net_model=net)

    strategy = get_strategy(strategy_name, micro_batches=4, flops_per_batch=10e9, comm_size=8 * 1024 * 1024)
    env.process(strategy.run(env, gpus, recorder))
    env.run()

    out_path = output_dir / f"task1_2_{strategy_name}_rdma.csv"
    recorder.to_csv(str(out_path))
    print(f"[Task1.2] Trace saved → {out_path.relative_to(output_dir.parent)}")


def run_task1_3(*, output_dir: str | os.PathLike | None = None, strategy_name: str = "pipeline"):
    """拥塞与优化比较，生成 baseline 与 optimised 两份 trace 并计算分析 CSV。"""

    output_dir = Path(output_dir or _ensure_trace_dir())
    nodes, gpus_per_node = 4, 4

    def _run_once(scenario: Literal["baseline", "optimized"], scheduler_factory):
        env = simpy.Environment()
        recorder = Recorder()

        gpu_to_node = {gid: gid // gpus_per_node for gid in range(nodes * gpus_per_node)}
        net = RDMANet(env, gpu_to_node)
        # 注册调度器
        net.register_scheduler(scheduler_factory(net._link))

        gpus = _setup_cluster(env, recorder, nodes=nodes, gpus_per_node=gpus_per_node, flops=1e13, nic_bw=100e9, net_model=net)
        strategy = get_strategy(strategy_name, micro_batches=4, flops_per_batch=10e9, comm_size=8 * 1024 * 1024)

        env.process(strategy.run(env, gpus, recorder))
        env.run()

        # 写回 scenario 字段
        for rec in recorder._records:
            rec["scenario"] = scenario
            if scenario == "optimized":
                rec["optimization_strategy"] = "priority_scheduling+time_slicing"

        fname = f"task1_3_{strategy_name}_{scenario}.csv"
        recorder.to_csv(str(output_dir / fname))
        return recorder.summary(), output_dir / fname

    # baseline
    summary_base, path_base = _run_once("baseline", make_baseline_scheduler)
    # optimised
    summary_opt, path_opt = _run_once("optimized", make_priority_scheduler)

    # ------------------ 生成分析 CSV ------------------
    import csv

    analysis_file = output_dir / f"task1_3_{strategy_name}_analysis.csv"
    with open(analysis_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "baseline",
                "optimized",
                "improvement(%)",
            ],
        )
        writer.writeheader()

        def _pct(a, b):
            return (a - b) / a * 100 if a else 0.0

        # total time
        writer.writerow({
            "metric": "total_time(s)",
            "baseline": summary_base["total_time"],
            "optimized": summary_opt["total_time"],
            "improvement(%)": _pct(summary_base["total_time"], summary_opt["total_time"]),
        })
        # comm ratio
        writer.writerow({
            "metric": "comm_ratio",
            "baseline": summary_base["comm_ratio"],
            "optimized": summary_opt["comm_ratio"],
            "improvement(%)": _pct(summary_base["comm_ratio"], summary_opt["comm_ratio"]),
        })

    print("[Task1.3] Analysis saved →", analysis_file.relative_to(output_dir.parent)) 