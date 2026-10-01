from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tongbench_eval.graph_env.runner import run_tongsim_task


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a TongSIM offline DAG task")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--agent-command")
    parser.add_argument("--workspace-dir")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--memory-file")
    parser.add_argument("--unique-state-image-dir")
    args = parser.parse_args()

    agent_command = None if args.oracle else args.agent_command
    result = run_tongsim_task(
        task_dir=Path(args.task_dir),
        output_dir=Path(args.output_dir),
        agent_command=agent_command,
        workspace_dir=Path(args.workspace_dir) if args.workspace_dir else None,
        max_steps=args.max_steps,
        timeout_sec=args.timeout_sec,
        memory_file=Path(args.memory_file) if args.memory_file else None,
        unique_state_image_dir=Path(args.unique_state_image_dir) if args.unique_state_image_dir else None,
    )
    print(json.dumps(result["score"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
