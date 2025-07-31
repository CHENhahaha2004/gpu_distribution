#!/usr/bin/env python3
"""命令行入口：一键运行实验任务 1.1 / 1.2 / 1.3。

示例：
  python -m gpu_distribution.run_tasks --task 1.2
  python -m gpu_distribution.run_tasks --task 1.3 --output-dir ./trace
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

from gpu_distribution.experiments import tasks as exp_tasks

# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU Distributed Training Lab – task runner")
    parser.add_argument("--task", required=True, choices=["1.1", "1.2", "1.3", "task1_1", "task1_2", "task1_3"],
                        help="Which task to run (1.1 / 1.2 / 1.3)")
    parser.add_argument("--output-dir", default=None, help="Directory to dump CSV traces & analysis")
    parser.add_argument("--strategy", default="pipeline", choices=["pipeline", "data", "tensor"],
                        help="Parallel strategy to use")
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)

    output = ns.output_dir
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)

    task_map = {
        "1.1": exp_tasks.run_task1_1,
        "task1_1": exp_tasks.run_task1_1,
        "1.2": exp_tasks.run_task1_2,
        "task1_2": exp_tasks.run_task1_2,
        "1.3": exp_tasks.run_task1_3,
        "task1_3": exp_tasks.run_task1_3,
    }

    func = task_map[ns.task]
    if output is not None:
        func(output_dir=output, strategy_name=ns.strategy)
    else:
        func(strategy_name=ns.strategy)


if __name__ == "__main__":
    main() 