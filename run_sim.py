#!/usr/bin/env python3
"""Command-line driver for GPU distributed training simulation (phase 1.1).

Usage example
-------------
python run_sim.py --gpus 4 --strategy tensor --micro_batches 2 --plot
"""
import argparse
import simpy

from simulator.gpu import GPU
# Network models
from net_model.ideal import IdealNet
from net_model.rdma import RDMANet
from trace.recorder import Recorder
from strategy import get_strategy
from visualization.plot import plot_gantt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU parallel-training simulator (phase 1.1).")
    # Cluster hierarchy arguments
    parser.add_argument("--nodes", type=int, default=1, help="Number of compute nodes")
    parser.add_argument("--gpus_per_node", type=int, default=4, help="GPUs per node")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Total GPUs to simulate (overrides nodes * gpus_per_node)")
    parser.add_argument("--strategy", type=str, choices=["data", "tensor", "pipeline"], default="data",
                        help="Parallel strategy to use")
    parser.add_argument("--micro_batches", type=int, default=2, help="Number of micro-batches to process")
    parser.add_argument("--flops", type=float, default=4e12, help="FLOPs per micro-batch (before splitting)")
    parser.add_argument("--comm_size", type=int, default=200 * 1024 * 1024, help="Communication size in bytes")
    parser.add_argument("--bw", type=float, default=100e9, help="NIC bandwidth in bytes/sec")
    # RDMA-specific overrides (optional)
    parser.add_argument("--intra_bw", type=float, default=100e9, help="Intra-node RDMA bandwidth (bytes/s)")
    parser.add_argument("--inter_bw", type=float, default=25e9, help="Inter-node RDMA bandwidth (bytes/s)")
    parser.add_argument("--intra_lat", type=float, default=5e-6, help="Intra-node latency (s)")
    parser.add_argument("--inter_lat", type=float, default=2.5e-5, help="Inter-node latency (s)")
    parser.add_argument("--net_conf", type=str, default=None, help="Path to JSON config for network model")
    parser.add_argument("--net", type=str, choices=["ideal", "rdma"], default="ideal",
                        help="Network model to use (ideal or rdma)")
    parser.add_argument("--out", type=str, default="trace.csv", help="Path to save CSV trace")
    parser.add_argument("--plot", action="store_true", help="Plot the resulting Gantt chart")
    parser.add_argument("--plot_backend", type=str, choices=["matplotlib", "plotly"], default="matplotlib",
                        help="Backend used for plotting the Gantt chart")
    parser.add_argument("--plot_view", type=str, choices=["per_gpu", "per_channel"], default="per_gpu",
                        help="Gantt view: each GPU row or aggregated compute/comm channels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = simpy.Environment()
    recorder = Recorder()

    # Determine total GPUs based on CLI args
    num_gpus = args.gpus if args.gpus is not None else args.nodes * args.gpus_per_node

    # Build network model
    if args.net == "ideal":
        net = IdealNet(env)
    else:
        # GPU ID 到节点 ID 
        gpu_to_node = {i: (i // args.gpus_per_node) for i in range(num_gpus)}

        # Load overrides from JSON if provided
        import json, os
        #准备 RDMANet 的初始化参数
        rdma_kwargs = dict(
            intra_bw=args.intra_bw,
            inter_bw=args.inter_bw,
            intra_lat=args.intra_lat,
            inter_lat=args.inter_lat,
        )
        #若提供了网络配置 JSON 文件，则覆盖默认参数
        if args.net_conf and os.path.isfile(args.net_conf):
            with open(args.net_conf, "r") as f:
                cfg = json.load(f)
            rdma_kwargs.update(cfg)
        #创建基于 RDMA 的网络模型
        net = RDMANet(env, gpu_to_node, **rdma_kwargs)

    # GPUs实例化
    gpus = [GPU(env, gpu_id=i, node_id=(i // args.gpus_per_node), flops=args.flops, nic_bw=args.bw,
                recorder=recorder, net_model=net) for i in range(num_gpus)]

    # Build selected strategy
    strategy = get_strategy(args.strategy, micro_batches=args.micro_batches, flops_per_batch=args.flops,
                             comm_size=args.comm_size)

    # Launch simulation
    env.process(strategy.run(env, gpus, recorder))
    env.run()

    # Export trace
    recorder.to_csv(args.out)
    print(f"Simulation finished. Trace saved to {args.out}")

    # Print summary metrics
    summary = recorder.summary()
    if summary:
        print("Summary metrics:")
        print(f"  Total time      : {summary['total_time']:.6f} s")
        print(f"  Compute time    : {summary['compute_time']:.6f} s")
        print(f"  Communication   : {summary['comm_time']:.6f} s ({summary['comm_ratio']*100:.2f}% of active time)")

    # Visualise if requested
    if args.plot:
        try:
            if args.plot_backend == "matplotlib":
                # Matplotlib backend目前仅支持 per_gpu 视图
                from visualization.plot import plot_gantt
                plot_gantt(args.out)
            else:  # plotly backend
                if args.plot_view == "per_gpu":
                    from visualization.plot import plot_gantt_plotly
                    plot_gantt_plotly(args.out)
                else:
                    from visualization.plot import plot_gantt_by_channel
                    plot_gantt_by_channel(args.out)
        except Exception as exc:  # pragma: no cover
            print("Failed to plot Gantt chart:", exc)
    import pandas as pd
    print(pd.read_csv('trace.csv').head())

if __name__ == "__main__":
    main() 