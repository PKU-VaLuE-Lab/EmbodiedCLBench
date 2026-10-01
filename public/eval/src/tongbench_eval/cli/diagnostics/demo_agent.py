from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tongbench_eval.utils.python_utils import current_python


def _run_env(*args: str, cwd: Path) -> dict:
    result = subprocess.run(
        [current_python(), "tongsim_env.py", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal local TongSIM demo agent")
    parser.add_argument("--max-loops", type=int, default=30)
    args = parser.parse_args()

    workspace = Path.cwd()
    for _ in range(args.max_loops):
        observation = _run_env("observe", cwd=workspace)
        if observation.get("done"):
            break
        candidate_actions = observation.get("candidate_actions", [])
        if not candidate_actions:
            break
        action_id = str(candidate_actions[0]["action_id"])
        step = _run_env("act", "--action_id", action_id, cwd=workspace)
        if step.get("done"):
            break
    final_state = _run_env("finish", cwd=workspace)
    print(json.dumps(final_state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
