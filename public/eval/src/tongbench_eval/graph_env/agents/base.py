from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AgentDecision:
    selected_action_id: str | None
    action_level: str | None = None
    template_id: str | None = None
    action_type: str | None = None
    object_id: str | None = None
    target_id: str | None = None
    role_bindings: dict[str, str] = field(default_factory=dict)
    confidence: float | None = None
    brief_reason: str | None = None
    perceived_state: str | None = None
    perceived_state_predicates: list[str] = field(default_factory=list)
    uncertain_predicates: list[str] = field(default_factory=list)
    visual_evidence: list[dict[str, Any]] = field(default_factory=list)
    rejected_actions: list[str] = field(default_factory=list)
    rejected_options: list[str] = field(default_factory=list)
    raw_response: str = ""
    parsed_json: dict[str, Any] | None = None
    parse_error: str | None = None
    parse_error_details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLMBackend(ABC):
    name: str

    @abstractmethod
    def choose_action(
        self,
        observation: dict[str, Any],
        prompt: str,
        image_paths: list[Path],
    ) -> AgentDecision:
        raise NotImplementedError
