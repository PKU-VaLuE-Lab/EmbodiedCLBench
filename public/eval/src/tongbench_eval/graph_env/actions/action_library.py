from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_factorizer import build_object_choices, normalize_short_name
from tongbench_eval.graph_env.io.object_normalizer import infer_object_type_from_id, normalize_object_type


_PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z0-9_]+)\}")


def _infer_required_roles(*values: Any) -> list[str]:
    roles: set[str] = set()
    for value in values:
        if isinstance(value, str):
            roles.update(_PLACEHOLDER_PATTERN.findall(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    roles.update(_PLACEHOLDER_PATTERN.findall(item))
    ordered = sorted(roles, key=lambda item: (0 if item == "object" else 1, item))
    return ordered


def _description_from_subtask(raw: dict[str, Any]) -> str:
    subtask_type = str(raw.get("subtask_type", "subtask"))
    primitive_sequence = raw.get("primitive_sequence", [])
    primitives: list[str] = []
    for step in primitive_sequence:
        if isinstance(step, dict):
            template_id = step.get("template_id")
            if template_id is not None:
                primitives.append(str(template_id))
    if primitives:
        return f"{subtask_type} via {' -> '.join(primitives)}"
    return f"{subtask_type} subtask"


@dataclass(slots=True)
class AtomicTemplate:
    template_id: str
    action_space: list[str] = field(default_factory=list)
    task_description: str = ""
    available_objects: list[str] = field(default_factory=list)
    pre_state: list[str] = field(default_factory=list)
    post_state: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubtaskTemplate:
    subtask_id: str
    subtask_type: str
    length: int = 0
    primitive_sequence: list[dict[str, Any]] = field(default_factory=list)
    roles: dict[str, list[str]] = field(default_factory=dict)
    pre_state: list[str] = field(default_factory=list)
    post_state: list[str] = field(default_factory=list)
    net_add: list[str] = field(default_factory=list)
    net_delete: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionCandidate:
    action_level: str
    template_id: str
    action_type: str
    description: str
    required_roles: list[str]
    role_constraints: dict[str, list[str]]
    pre_state: list[str]
    post_state: list[str]
    net_add: list[str]
    net_delete: list[str]
    primitive_sequence: list[dict[str, Any]] = field(default_factory=list)
    candidate_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_level": self.action_level,
            "template_id": self.template_id,
            "action_type": self.action_type,
            "description": self.description,
            "required_roles": list(self.required_roles),
            "role_constraints": {key: list(value) for key, value in self.role_constraints.items()},
            "pre_state": list(self.pre_state),
            "post_state": list(self.post_state),
            "net_add": list(self.net_add),
            "net_delete": list(self.net_delete),
            "primitive_sequence": list(self.primitive_sequence),
            "candidate_source": self.candidate_source,
        }


@dataclass(slots=True)
class ObjectCandidate:
    object_id: str
    short_name: str
    type_guess: str
    current_properties: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "short_name": self.short_name,
            "type_guess": self.type_guess,
            "current_properties": list(self.current_properties),
        }


@dataclass(slots=True)
class AtomicActionLibrary:
    templates: list[AtomicTemplate]

    def candidates(self, placement_only: bool = False, candidate_source_override: str | None = None) -> list[ActionCandidate]:
        output: list[ActionCandidate] = []
        for template in self.templates:
            if placement_only and template.template_id not in {"place_{object1}_on_{object2}", "place_{object1}_in_{object2}"}:
                continue
            required_roles = _infer_required_roles(template.template_id, template.pre_state, template.post_state)
            action_type = template.action_space[0] if template.action_space else template.template_id
            raw_available_objects = template.raw.get("available_objects", template.available_objects)
            if isinstance(raw_available_objects, dict):
                role_constraints = {
                    role: [normalize_object_type(str(item)) for item in raw_available_objects.get(role, [])]
                    for role in required_roles
                }
            else:
                normalized_available = [normalize_object_type(str(item)) for item in template.available_objects]
                role_constraints = {
                    role: list(normalized_available)
                    for role in required_roles
                }
            output.append(
                ActionCandidate(
                    action_level="atomic",
                    template_id=template.template_id,
                    action_type=action_type,
                    description=template.task_description,
                    required_roles=required_roles,
                    role_constraints=role_constraints,
                    pre_state=list(template.pre_state),
                    post_state=list(template.post_state),
                    net_add=list(template.post_state),
                    net_delete=[],
                    candidate_source=candidate_source_override or "atomic_templates",
                )
            )
        return output


@dataclass(slots=True)
class SubtaskLibrary:
    templates: list[SubtaskTemplate]

    def candidates(self) -> list[ActionCandidate]:
        output: list[ActionCandidate] = []
        for template in self.templates:
            role_constraints = {
                role: [normalize_object_type(item) for item in values]
                for role, values in template.roles.items()
            }
            output.append(
                ActionCandidate(
                    action_level="subtask",
                    template_id=template.subtask_id,
                    action_type=template.subtask_type,
                    description=_description_from_subtask(template.raw),
                    required_roles=list(template.roles.keys()),
                    role_constraints=role_constraints,
                    pre_state=list(template.pre_state),
                    post_state=list(template.post_state),
                    net_add=list(template.net_add),
                    net_delete=list(template.net_delete),
                    primitive_sequence=list(template.primitive_sequence),
                    candidate_source="subtask_templates",
                )
            )
        return output


def load_atomic_templates(path: Path) -> AtomicActionLibrary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    templates: list[AtomicTemplate] = []
    for item in payload.get("templates", []):
        if not isinstance(item, dict):
            continue
        templates.append(
            AtomicTemplate(
                template_id=str(item.get("template_id", "")),
                action_space=[str(value) for value in item.get("action_space", [])],
                task_description=str(item.get("task_description", "")),
                available_objects=[str(value) for value in item.get("available_objects", [])],
                pre_state=[str(value) for value in item.get("pre_state", [])],
                post_state=[str(value) for value in item.get("post_state", [])],
                raw=dict(item),
            )
        )
    return AtomicActionLibrary(templates=templates)


def load_subtask_templates(path: Path) -> SubtaskLibrary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    templates: list[SubtaskTemplate] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        templates.append(
            SubtaskTemplate(
                subtask_id=str(item.get("subtask_id", "")),
                subtask_type=str(item.get("subtask_type", "")),
                length=int(item.get("length", 0)),
                primitive_sequence=[
                    dict(step) for step in item.get("primitive_sequence", []) if isinstance(step, dict)
                ],
                roles={
                    str(role): [str(value) for value in values]
                    for role, values in item.get("roles", {}).items()
                    if isinstance(values, list)
                },
                pre_state=[str(value) for value in item.get("pre_state", [])],
                post_state=[str(value) for value in item.get("post_state", [])],
                net_add=[str(value) for value in item.get("net_add", [])],
                net_delete=[str(value) for value in item.get("net_delete", [])],
                raw=dict(item),
            )
        )
    return SubtaskLibrary(templates=templates)


def _predicate_object_ids(current_state_predicates: list[str]) -> set[str]:
    object_ids: set[str] = set()
    for predicate in current_state_predicates:
        if "(" not in predicate or ")" not in predicate:
            continue
        inner = predicate.split("(", 1)[1].rsplit(")", 1)[0]
        for item in inner.split(","):
            value = item.strip()
            if value:
                object_ids.add(value)
    return object_ids


def build_task_object_candidates(task: Any, current_state_predicates: list[str]) -> list[dict[str, Any]]:
    raw_task = task if isinstance(task, dict) else getattr(task, "raw", {})
    object_choices = build_object_choices(raw_task, current_state_predicates)
    return [
        ObjectCandidate(
            object_id=str(item["object_id"]),
            short_name=str(item.get("short_name") or normalize_short_name(str(item["object_id"]))),
            type_guess=infer_object_type_from_id(str(item["object_id"])),
            current_properties=sorted(str(value) for value in item.get("current_properties", [])),
        ).to_dict()
        for item in object_choices
    ]


def build_model_facing_action_candidates(
    task: Any,
    current_state_predicates: list[str],
    action_library_mode: str,
    atomic_library: AtomicActionLibrary | None,
    subtask_library: SubtaskLibrary | None,
) -> list[dict[str, Any]]:
    raw_task = task if isinstance(task, dict) else getattr(task, "raw", {})
    task_objects = [str(item["object_id"]) for item in build_object_choices(raw_task, current_state_predicates)]
    task_types = {infer_object_type_from_id(object_id) for object_id in task_objects}
    candidates: list[ActionCandidate] = []
    expose_all_atomic = action_library_mode == "atomic_only"
    if action_library_mode in {"atomic_only", "atomic_plus_subtask"} and atomic_library is not None:
        candidates.extend(atomic_library.candidates())
    elif action_library_mode == "subtask_only" and atomic_library is not None:
        candidates.extend(
            atomic_library.candidates(
                placement_only=True,
                candidate_source_override="subtask_only_required_atomic_placement",
            )
        )
    if action_library_mode != "atomic_only" and subtask_library is not None:
        candidates.extend(subtask_library.candidates())
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        if expose_all_atomic and candidate.action_level == "atomic":
            output.append(candidate.to_dict())
            continue
        if not candidate.required_roles:
            output.append(candidate.to_dict())
            continue
        overlaps = False
        for _, allowed_types in candidate.role_constraints.items():
            if not allowed_types or task_types.intersection(set(allowed_types)):
                overlaps = True
                break
        if overlaps:
            output.append(candidate.to_dict())
    return output
