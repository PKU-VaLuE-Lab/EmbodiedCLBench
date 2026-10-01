from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import TongSimGraphEnv
from .grader import grade_trace
from .loader import load_tongsim_task
from .validator import validate_task


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TongSIM offline DAG utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--task-dir", required=True)

    observe_parser = subparsers.add_parser("observe")
    observe_parser.add_argument("--task-dir", required=True)

    rollout_parser = subparsers.add_parser("rollout")
    rollout_parser.add_argument("--task-dir", required=True)
    rollout_parser.add_argument("--actions", nargs="*", default=[])
    rollout_parser.add_argument("--output-dir")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    task_dir = Path(args.task_dir)

    if args.command == "validate":
        print(json.dumps(validate_task(task_dir), indent=2))
        return 0

    task = load_tongsim_task(task_dir)
    env = TongSimGraphEnv(task=task, task_dir=task_dir)

    if args.command == "observe":
        print(json.dumps(env.reset(), indent=2))
        return 0

    env.reset()
    for action in args.actions:
        env.step(action)
    if args.output_dir:
        env.save_trace(Path(args.output_dir))
    print(json.dumps(grade_trace(task, env.trace), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
