from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tongbench_eval.graph_env.sequence_runner import compare_memory_vs_no_memory, run_tongsim_sequence


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a TongSIM offline DAG sequence")
    parser.add_argument("--sequence-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["memory", "no_memory"], default="no_memory")
    parser.add_argument("--compare-memory", action="store_true")
    parser.add_argument("--agent-command")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    agent_command = args.agent_command
    if args.compare_memory:
        result = compare_memory_vs_no_memory(
            sequence_file=Path(args.sequence_file),
            output_dir=Path(args.output_dir),
            agent_command=agent_command,
            max_steps=args.max_steps,
            timeout_sec=args.timeout_sec,
        )
    else:
        result = run_tongsim_sequence(
            sequence_file=Path(args.sequence_file),
            output_dir=Path(args.output_dir),
            mode=args.mode,
            agent_command=agent_command,
            max_steps=args.max_steps,
            timeout_sec=args.timeout_sec,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
