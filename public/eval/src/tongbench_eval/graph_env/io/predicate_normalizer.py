from __future__ import annotations

import re
from typing import Any


_PREDICATE_PATTERN = re.compile(r"^([A-Za-z0-9_]+)\((.*)\)$")


def normalize_predicate(predicate: str) -> str:
    value = re.sub(r"\s+", " ", str(predicate).strip())
    match = _PREDICATE_PATTERN.match(value)
    if not match:
        return value
    name = match.group(1).strip()
    args = [re.sub(r"\s+", " ", item.strip()) for item in match.group(2).split(",")]
    args = [item for item in args if item]
    return f"{name}({', '.join(args)})"


def extract_predicate_list(predicates: Any) -> list[str]:
    if predicates is None:
        return []
    if isinstance(predicates, dict):
        if isinstance(predicates.get("predicates"), (list, tuple, set)):
            return [str(item) for item in predicates.get("predicates", [])]
        return []
    if isinstance(predicates, (list, tuple, set)):
        return [str(item) for item in predicates]
    return [str(predicates)]


def normalize_predicate_set(predicates: list[str] | tuple[str, ...] | set[str] | Any) -> list[str]:
    predicates = extract_predicate_list(predicates)
    output: list[str] = []
    seen: set[str] = set()
    for predicate in predicates:
        normalized = normalize_predicate(str(predicate))
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output
