#!/usr/bin/env python3
"""Command-line driver for GPU distributed training simulation (phase 1.1).

Usage example
-------------
python run_sim.py --gpus 4 --strategy tensor --micro_batches 2 --plot
"""
import argparse
import simpy

from simulator.gpu import GPU
from net_model.ideal import IdealNet
from trace.recorder import Recorder
from strategy import get_strategy
from visualization.plot import plot_gantt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU parallel-training simulator (phase 1.1).")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs to simulate")
    parser.add_argument("--strategy", type=str, choices=["data", "tensor", "pipeline"], default="data",
                        help="Parallel strategy to use")
    parser.add_argument("--micro_batches", type=int, default=2, help="Number of micro-batches to process")
    parser.add_argument("--flops", type=float, default=4e12, help="FLOPs per micro-batch (before splitting)")
    parser.add_argument("--comm_size", type=int, default=200 * 1024 * 1024, help="Communication size in bytes")
    parser.add_argument("--bw", type=float, default=100e9, help="NIC bandwidth in bytes/sec")
    parser.add_argument("--out", type=str, default="trace.csv", help="Path to save CSV trace")
    parser.add_argument("--plot", action="store_true", help="Plot the resulting Gantt chart")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = simpy.Environment()
    recorder = Recorder()
    net = IdealNet(env)

    # Instantiate GPUs
    gpus = [GPU(env, gpu_id=i, flops=args.flops, nic_bw=args.bw, recorder=recorder, net_model=net)
            for i in range(args.gpus)]

    # Build selected strategy
    strategy = get_strategy(args.strategy, micro_batches=args.micro_batches, flops_per_batch=args.flops,
                             comm_size=args.comm_size)

    # Launch simulation
    env.process(strategy.run(env, gpus, recorder))
    env.run()

    # Export trace
    recorder.to_csv(args.out)
    print(f"Simulation finished. Trace saved to {args.out}")

    # Visualise if requested
    if args.plot:
        try:
            plot_gantt(args.out)
        except Exception as exc:  # pragma: no cover
            print("Failed to plot Gantt chart:", exc)


if __name__ == "__main__":
    main() 