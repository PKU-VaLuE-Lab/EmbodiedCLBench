from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from .framework_runtime import TongSimFrameworkRuntime


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def serve_framework_session(
    runtime_root: Path,
    session_id: str,
    *,
    poll_interval: float = 0.2,
) -> None:
    runtime = TongSimFrameworkRuntime(runtime_root)
    session = runtime.load_session(session_id)
    bridge_dir = Path(session.bridge_dir)
    requests_dir = bridge_dir / "requests"
    responses_dir = bridge_dir / "responses"
    stop_file = bridge_dir / "STOP"

    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    while True:
        if stop_file.exists():
            break

        pending = sorted(requests_dir.glob("*.json"))
        if not pending:
            time.sleep(poll_interval)
            continue

        for request_path in pending:
            response_path = responses_dir / request_path.name
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                command = str(request.get("command", "")).strip().lower()
                payload = request.get("payload", {})
                if command == "observe":
                    result = runtime.observe(session_id)
                elif command == "act":
                    result = runtime.act(session_id, payload if isinstance(payload, dict) else {})
                elif command == "status":
                    result = runtime.status(session_id)
                elif command == "finish":
                    result = runtime.finish(session_id)
                else:
                    raise ValueError(f"Unsupported TongSIM framework command: {command}")
                _write_json(response_path, {"ok": True, "result": result})
            except Exception as exc:
                _write_json(
                    response_path,
                    {
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            finally:
                request_path.unlink(missing_ok=True)
