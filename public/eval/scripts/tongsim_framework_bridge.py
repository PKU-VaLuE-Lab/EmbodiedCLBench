from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tongbench_eval.graph_env.framework_bridge import serve_framework_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a hidden TongSIM session for framework agents.")
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    args = parser.parse_args()

    serve_framework_session(
        Path(args.runtime_root),
        args.session_id,
        poll_interval=float(args.poll_interval),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
