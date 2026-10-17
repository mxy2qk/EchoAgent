from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Protocol

from echo_agent.models import Intent, Project, ProjectStatus


class DispatchMode(str, Enum):
    SKIPPED = "skipped"
    FIRST_RUN = "first_run"
    EXPLORE = "explore"
    REASON = "reason"
    COMPLETED = "completed"


@dataclass(slots=True)
class WorkerObservation:
    answer: str | None = None
    fact_description: str | None = None
    outcome: str | None = None
    # 答案在原始输出中出现的上下文片段，用于让 reason 核验答案来源。
    answer_source: str | None = None


@dataclass(slots=True)
class WorkerResult:
    answer: str | None = None
    fact_description: str | None = None
    retryable: bool = False
    detail: str | None = None
    outcome: str | None = None
    prior_results: list[WorkerObservation] = field(default_factory=list)
    # 答案在原始输出中出现的上下文片段（落盘时写入 fact，供 reason 核验）。
    answer_source: str | None = None

    @property
    def has_answer(self) -> bool:
        return self.answer is not None


@dataclass(slots=True)
class ReasonResult:
    completed: bool = False
    next_intent_descriptions: list[str] | None = None
    next_intents: list[dict[str, object]] | None = None
    answer: str | None = None
    stop_reason: str = ""
    raw_output: str = ""
    # 该结果来自超时/中断后的 salvage，而不是模型正常收尾。
    # 空规划在这种情况不代表「没有方向」，不应计入 no-progress。
    salvaged: bool = False


@dataclass(slots=True)
class DispatchResult:
    project_id: int
    mode: DispatchMode
    status: ProjectStatus
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["status"] = self.status.value
        return data


@dataclass(slots=True)
class IntentExecutionResult:
    intent: Intent
    result: WorkerResult


@dataclass(slots=True)
class AgentRunResult:
    output: str
    # True 表示 output 是从超时/非零退出的原始流里 salvage 出来的，而非正常收尾。
    salvaged: bool = False


AgentLogCallback = Callable[[str, str, dict[str, object] | None], None]


class WorkerAdapter(Protocol):
    def execute_intent(self, project: Project, intent: Intent) -> WorkerResult:
        raise NotImplementedError


class Reasoner(Protocol):
    def plan(self, project: Project) -> ReasonResult:
        raise NotImplementedError


class AgentAdapter(Protocol):
    def run(self, project: Project, prompt: str, mode: DispatchMode) -> AgentRunResult:
        raise NotImplementedError
