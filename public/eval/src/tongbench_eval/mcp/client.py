from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _session_config_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / ".tongsim_session.json", Path.cwd() / ".tongsim_session.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Missing .tongsim_session.json next to tongsim_client.py or in the current directory.")


def _load_session_config() -> dict[str, Any]:
    return json.loads(_session_config_path().read_text(encoding="utf-8"))


def _request_paths(config: dict[str, Any], request_id: str) -> tuple[Path, Path]:
    bridge_mount = Path(str(config.get("bridge_mount", "/tongsim_bridge")))
    return bridge_mount / "requests" / f"{request_id}.json", bridge_mount / "responses" / f"{request_id}.json"


def _send_request(command: str, payload: dict[str, Any] | None = None, timeout_sec: float = 120.0) -> dict[str, Any]:
    config = _load_session_config()
    request_id = uuid.uuid4().hex
    request_path, response_path = _request_paths(config, request_id)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps(
            {
                "command": command,
                "payload": payload or {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if response_path.exists():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response_path.unlink(missing_ok=True)
            if not response.get("ok", False):
                error = response.get("error", "Unknown TongSIM bridge error")
                traceback_text = response.get("traceback", "")
                raise RuntimeError(f"{error}\n{traceback_text}".strip())
            return dict(response.get("result", {}))
        time.sleep(0.1)
    raise TimeoutError(f"TongSIM bridge timed out waiting for response to {command}.")


def _resolve_local_workspace_root() -> Path:
    return _session_config_path().parent


def _mirror_observation_images(payload: dict[str, Any]) -> dict[str, Any]:
    config = _load_session_config()
    local_root = _resolve_local_workspace_root()
    observation_subdir = str(config.get("observation_subdir", "current_observation"))
    local_observation_dir = local_root / observation_subdir
    if local_observation_dir.exists():
        shutil.rmtree(local_observation_dir)
    local_observation_dir.mkdir(parents=True, exist_ok=True)

    rewritten: list[str] = []
    for raw_path in payload.get("image_paths", []) or []:
        source = Path(str(raw_path))
        if not source.exists():
            continue
        destination = local_observation_dir / source.name
        shutil.copy2(source, destination)
        rewritten.append(str(destination.resolve()))
    payload["image_paths"] = rewritten
    (local_root / "last_observation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def _load_decision_payload(raw_value: str) -> dict[str, Any]:
    candidate = Path(raw_value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(raw_value)


def _parse_role_bindings(raw_roles: list[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for item in raw_roles:
        if "=" not in item:
            raise ValueError(f"Invalid --role value {item!r}; expected role=OBJECT_ID.")
        role, object_id = item.split("=", 1)
        role = role.strip()
        object_id = object_id.strip()
        if not role or not object_id:
            raise ValueError(f"Invalid --role value {item!r}; expected role=OBJECT_ID.")
        bindings[role] = object_id
    return bindings


def _parse_visual_evidence(raw_items: list[str]) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for item in raw_items:
        if "::" not in item:
            raise ValueError(
                f"Invalid --evidence value {item!r}; expected predicate::evidence text."
            )
        predicate, text = item.split("::", 1)
        predicate = predicate.strip()
        text = text.strip()
        if not predicate or not text:
            raise ValueError(
                f"Invalid --evidence value {item!r}; expected predicate::evidence text."
            )
        evidence.append({"predicate": predicate, "evidence": text})
    return evidence


def _build_simple_decision_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "perceived_state_predicates": list(args.perceived or []),
        "uncertain_predicates": list(args.uncertain or []),
        "visual_evidence": _parse_visual_evidence(list(args.evidence or [])),
        "selected_action": {
            "action_level": args.action_level,
            "template_id": args.template_id,
            "action_type": args.action_type,
            "role_bindings": _parse_role_bindings(list(args.role or [])),
        },
        "brief_reason": args.brief_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TongSIM framework bridge client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("observe")
    act_parser = subparsers.add_parser("act")
    act_parser.add_argument("--decision-json", required=True)
    act_simple_parser = subparsers.add_parser("act-simple")
    act_simple_parser.add_argument("--action-level", required=True)
    act_simple_parser.add_argument("--template-id", required=True)
    act_simple_parser.add_argument("--action-type", required=True)
    act_simple_parser.add_argument("--role", action="append", default=[])
    act_simple_parser.add_argument("--perceived", action="append", default=[])
    act_simple_parser.add_argument("--uncertain", action="append", default=[])
    act_simple_parser.add_argument("--evidence", action="append", default=[])
    act_simple_parser.add_argument("--brief-reason", default="")
    subparsers.add_parser("status")
    subparsers.add_parser("finish")

    args = parser.parse_args()

    try:
        if args.command == "observe":
            payload = _mirror_observation_images(_send_request("observe"))
        elif args.command == "act":
            payload = _send_request("act", _load_decision_payload(args.decision_json))
            (_resolve_local_workspace_root() / "last_action_result.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        elif args.command == "act-simple":
            payload = _send_request("act", _build_simple_decision_payload(args))
            (_resolve_local_workspace_root() / "last_action_result.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        elif args.command == "status":
            payload = _send_request("status")
        else:
            payload = _send_request("finish")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
