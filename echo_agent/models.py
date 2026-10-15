from __future__ import annotations

import re

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class StateTransitionError(ValueError):
    """Raised when a business object receives an invalid status transition."""


class ProjectStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    RUNNING = "running"
    STOPED = "stoped"
    COMPLETED = "completed"

    @classmethod
    def coerce(cls, value: str | "ProjectStatus") -> "ProjectStatus":
        if isinstance(value, cls):
            return value
        if value == "stopped":
            value = cls.STOPED.value
        return cls(value)


class IntentStatus(str, Enum):
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def coerce(cls, value: str | "IntentStatus") -> "IntentStatus":
        if isinstance(value, cls):
            return value
        return cls(value)


class FactKind(str, Enum):
    START = "start"
    GOAL = "goal"
    EVIDENCE = "evidence"
    REFUTED = "refuted"
    ERROR = "error"

    @classmethod
    def coerce(cls, value: str | "FactKind" | None) -> "FactKind":
        if isinstance(value, cls):
            return value
        if value is None or not str(value).strip():
            return cls.EVIDENCE
        return cls(str(value))


class IntentOutcome(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REFUTED = "refuted"
    ERROR = "error"
    EMPTY = "empty"

    @classmethod
    def coerce(cls, value: str | "IntentOutcome" | None) -> "IntentOutcome":
        if isinstance(value, cls):
            return value
        if value is None or not str(value).strip():
            return cls.PENDING
        return cls(str(value))


PROJECT_STATUS_TRANSITIONS: dict[ProjectStatus, set[ProjectStatus]] = {
    ProjectStatus.INACTIVE: {ProjectStatus.ACTIVE},
    ProjectStatus.ACTIVE: {ProjectStatus.RUNNING},
    ProjectStatus.RUNNING: {ProjectStatus.STOPED, ProjectStatus.COMPLETED},
    ProjectStatus.STOPED: {ProjectStatus.RUNNING},
    ProjectStatus.COMPLETED: set(),
}

INTENT_STATUS_TRANSITIONS: dict[IntentStatus, set[IntentStatus]] = {
    IntentStatus.ACTIVE: {IntentStatus.RUNNING},
    IntentStatus.RUNNING: {IntentStatus.ACTIVE, IntentStatus.COMPLETED, IntentStatus.FAILED},
    IntentStatus.COMPLETED: set(),
    IntentStatus.FAILED: set(),
}


@dataclass(slots=True)
class Fact:
    id: int
    description: str
    kind: FactKind = FactKind.EVIDENCE

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("fact id must be non-negative")
        if not self.description.strip():
            raise ValueError("fact description cannot be empty")
        self.kind = FactKind.coerce(self.kind)

    @property
    def is_plannable(self) -> bool:
        return self.kind in {FactKind.START, FactKind.EVIDENCE}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        return cls(
            id=int(data["id"]),
            description=str(data["description"]),
            kind=FactKind.coerce(data.get("kind")),
        )


def normalize_action_key(value: object) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^a-z0-9-]+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:64]


def normalize_intent_description(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", " ", text).strip()
    return text


@dataclass(slots=True)
class Intent:
    id: int
    from_fact_id: int | None = None
    to_fact_id: int | None = None
    description: str = ""
    status: IntentStatus = IntentStatus.ACTIVE
    outcome: IntentOutcome = IntentOutcome.PENDING
    note: str = ""
    action_key: str = ""

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("intent id must be non-negative")
        if self.from_fact_id is not None and self.from_fact_id < 0:
            raise ValueError("intent from_fact_id must be non-negative")
        if self.to_fact_id is not None and self.to_fact_id < 0:
            raise ValueError("intent to_fact_id must be non-negative")
        self.status = IntentStatus.coerce(self.status)
        self.outcome = IntentOutcome.coerce(self.outcome)
        self.note = str(self.note or "")
        self.action_key = normalize_action_key(self.action_key)

    @property
    def is_executable(self) -> bool:
        return self.status == IntentStatus.ACTIVE

    @property
    def is_completed(self) -> bool:
        return self.status == IntentStatus.COMPLETED

    def transition_to(self, status: IntentStatus | str) -> None:
        next_status = IntentStatus.coerce(status)
        allowed = INTENT_STATUS_TRANSITIONS[self.status]
        if next_status not in allowed:
            raise StateTransitionError(
                f"invalid intent status transition: {self.status.value} -> {next_status.value}"
            )
        self.status = next_status

    def mark_running(self) -> None:
        self.transition_to(IntentStatus.RUNNING)

    def release(self) -> None:
        self.transition_to(IntentStatus.ACTIVE)

    def mark_completed(
        self,
        to_fact_id: int | None = None,
        outcome: IntentOutcome | str = IntentOutcome.VERIFIED,
    ) -> None:
        if to_fact_id is not None:
            if to_fact_id < 0:
                raise ValueError("intent to_fact_id must be non-negative")
            self.to_fact_id = to_fact_id
        self.outcome = IntentOutcome.coerce(outcome)
        self.transition_to(IntentStatus.COMPLETED)

    def mark_failed(self, outcome: IntentOutcome | str, note: str = "") -> None:
        self.outcome = IntentOutcome.coerce(outcome)
        if note:
            self.note = note
        self.transition_to(IntentStatus.FAILED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from": self.from_fact_id,
            "to": self.to_fact_id,
            "description": self.description,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "note": self.note,
            "action_key": self.action_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intent":
        return cls(
            id=int(data["id"]),
            from_fact_id=data.get("from"),
            to_fact_id=data.get("to"),
            description=str(data.get("description", "")),
            status=IntentStatus.coerce(data.get("status", IntentStatus.ACTIVE)),
            outcome=IntentOutcome.coerce(data.get("outcome")),
            note=str(data.get("note", "")),
            action_key=str(data.get("action_key", "")),
        )


@dataclass(slots=True)
class Worker:
    docker_session: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "docker_session": self.docker_session,
        }


@dataclass(slots=True)
class Project:
    id: int
    name: str
    status: ProjectStatus = ProjectStatus.INACTIVE
    facts: dict[int, Fact] = field(default_factory=dict)
    intents: dict[int, Intent] = field(default_factory=dict)
    docker: str | None = None
    target_url: str | None = None
    hint: str = ""
    objective: str = ""

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("project id must be non-negative")
        if not self.name.strip():
            raise ValueError("project name cannot be empty")
        self.status = ProjectStatus.coerce(self.status)
        self.facts = _normalize_items(self.facts, Fact)
        self.intents = _normalize_items(self.intents, Intent)
        self._validate_intent_fact_refs(self.intents.values())

    @property
    def can_execute(self) -> bool:
        return self.status in {ProjectStatus.ACTIVE, ProjectStatus.STOPED}

    @property
    def is_running(self) -> bool:
        return self.status == ProjectStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        return self.status == ProjectStatus.COMPLETED

    @property
    def is_first_run(self) -> bool:
        return not self.intents

    def transition_to(self, status: ProjectStatus | str) -> None:
        next_status = ProjectStatus.coerce(status)
        allowed = PROJECT_STATUS_TRANSITIONS[self.status]
        if next_status not in allowed:
            raise StateTransitionError(
                f"invalid project status transition: {self.status.value} -> {next_status.value}"
            )
        self.status = next_status

    def activate(self) -> None:
        self.transition_to(ProjectStatus.ACTIVE)

    def start(self) -> None:
        self.transition_to(ProjectStatus.RUNNING)

    def stop(self) -> None:
        self.transition_to(ProjectStatus.STOPED)

    def complete(self) -> None:
        self.transition_to(ProjectStatus.COMPLETED)

    def add_fact(
        self,
        description: str,
        fact_id: int | None = None,
        kind: FactKind | str | None = None,
    ) -> Fact:
        next_id = self._next_id(self.facts) if fact_id is None else fact_id
        if next_id in self.facts:
            raise ValueError(f"fact id already exists: {next_id}")
        resolved_kind = FactKind.coerce(kind)
        if kind is None and next_id == 0:
            resolved_kind = FactKind.START
        elif kind is None and next_id == 1:
            resolved_kind = FactKind.GOAL
        fact = Fact(id=next_id, description=description, kind=resolved_kind)
        self.facts[fact.id] = fact
        return fact

    def add_intent(
        self,
        description: str = "",
        from_fact_id: int | None = None,
        to_fact_id: int | None = None,
        intent_id: int | None = None,
        action_key: str = "",
    ) -> Intent:
        next_id = self._next_id(self.intents) if intent_id is None else intent_id
        if next_id in self.intents:
            raise ValueError(f"intent id already exists: {next_id}")
        intent = Intent(
            id=next_id,
            from_fact_id=from_fact_id,
            to_fact_id=to_fact_id,
            description=description,
            action_key=action_key,
        )
        self._validate_intent_fact_refs([intent])
        self.intents[intent.id] = intent
        return intent

    def active_intents(self) -> list[Intent]:
        return sorted(
            (intent for intent in self.intents.values() if intent.status == IntentStatus.ACTIVE),
            key=lambda intent: intent.id,
        )

    def next_active_intent(self) -> Intent | None:
        intents = self.active_intents()
        return intents[0] if intents else None

    def complete_intent(self, intent_id: int, result_fact_description: str | None = None) -> Intent:
        intent = self.intents[intent_id]
        if intent.status == IntentStatus.ACTIVE:
            intent.mark_running()
        if result_fact_description is not None:
            fact = self.add_fact(result_fact_description, kind=FactKind.EVIDENCE)
            intent.mark_completed(to_fact_id=fact.id, outcome=IntentOutcome.VERIFIED)
        else:
            intent.mark_completed(outcome=IntentOutcome.VERIFIED)
        return intent

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "facts": [fact.to_dict() for fact in sorted(self.facts.values(), key=lambda item: item.id)],
            "intents": [
                intent.to_dict() for intent in sorted(self.intents.values(), key=lambda item: item.id)
            ],
            "docker": self.docker,
            "target_url": self.target_url,
            "hint": self.hint,
            "objective": self.objective,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            status=ProjectStatus.coerce(data.get("status", ProjectStatus.INACTIVE)),
            facts={fact.id: fact for fact in map(Fact.from_dict, data.get("facts", []))},
            intents={intent.id: intent for intent in map(Intent.from_dict, data.get("intents", []))},
            docker=data.get("docker"),
            target_url=data.get("target_url"),
            hint=str(data.get("hint", "")),
            objective=str(data.get("objective", "")),
        )

    @staticmethod
    def _next_id(items: dict[int, object]) -> int:
        return max(items, default=-1) + 1

    def _validate_intent_fact_refs(self, intents: Iterable[Intent]) -> None:
        for intent in intents:
            for attr_name, fact_id in (
                ("from_fact_id", intent.from_fact_id),
                ("to_fact_id", intent.to_fact_id),
            ):
                if fact_id is not None and fact_id not in self.facts:
                    raise ValueError(f"intent {attr_name} references unknown fact id: {fact_id}")


def _normalize_items(
    items: dict[int, Fact] | dict[int, Intent] | Iterable[Fact] | Iterable[Intent],
    expected_type: type[Fact] | type[Intent],
) -> dict[int, Fact] | dict[int, Intent]:
    values = items.values() if isinstance(items, dict) else items
    normalized: dict[int, Fact] | dict[int, Intent] = {}
    for item in values:
        if not isinstance(item, expected_type):
            raise TypeError(f"expected {expected_type.__name__}, got {type(item).__name__}")
        if item.id in normalized:
            raise ValueError(f"duplicate {expected_type.__name__.lower()} id: {item.id}")
        normalized[item.id] = item
    return normalized
