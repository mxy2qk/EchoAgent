from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Mapping

from echo_agent.dispatcher.agents import CodexDockerAgent, CodexReasoner, CodexWorker
from echo_agent.dispatcher.client import ProjectClient
from echo_agent.dispatcher.prompts import PromptBuilder
from echo_agent.dispatcher.types import (
    AgentAdapter,
    DispatchMode,
    DispatchResult,
    IntentExecutionResult,
    ReasonResult,
    Reasoner,
    WorkerAdapter,
    WorkerResult,
)
from echo_agent.dispatcher.utils import callable_accepts_mode
from echo_agent.models import Intent, Project, ProjectStatus


class Dispatcher:
    def __init__(
        self,
        client: ProjectClient,
        worker: WorkerAdapter | None = None,
        reasoner: Reasoner | None = None,
        agent: AgentAdapter | None = None,
        prompt_builder: PromptBuilder | None = None,
        max_explore_workers: int = 4,
        codex_model: str | None = None,
        codex_mode_models: Mapping[DispatchMode | str, str] | None = None,
        codex_workspace: str = "/home/kali/workspace",
        codex_timeout_seconds: int = 1800,
        codex_no_output_timeout_seconds: int = 900,
        codex_mode_no_output_timeout_seconds: Mapping[DispatchMode | str, int] | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        codex_command: tuple[str, ...] | None = None,
        codex_provider_name: str = "echo_agent_proxy",
        codex_provider_supports_websockets: bool = False,
        codex_request_max_retries: int = 2,
        codex_stream_max_retries: int = 2,
        codex_stream_idle_timeout_ms: int = 60_000,
    ) -> None:
        self.client = client
        if max_explore_workers < 1:
            raise ValueError("max_explore_workers must be at least 1")
        self.max_explore_workers = max_explore_workers
        if agent is None:
            agent = CodexDockerAgent(
                codex_command=codex_command,
                model=codex_model,
                mode_models=codex_mode_models,
                workspace=codex_workspace,
                timeout_seconds=codex_timeout_seconds,
                no_output_timeout_seconds=codex_no_output_timeout_seconds,
                mode_no_output_timeout_seconds=codex_mode_no_output_timeout_seconds,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                provider_name=codex_provider_name,
                provider_supports_websockets=codex_provider_supports_websockets,
                request_max_retries=codex_request_max_retries,
                stream_max_retries=codex_stream_max_retries,
                stream_idle_timeout_ms=codex_stream_idle_timeout_ms,
            )
        self.worker = worker or CodexWorker(agent, prompt_builder)
        self.reasoner = reasoner or CodexReasoner(agent, prompt_builder)
        self._worker_accepts_mode = callable_accepts_mode(self.worker.execute_intent)
        self._worker_accepts_log = callable_accepts_mode(self.worker.execute_intent, parameter="log")
        self._reasoner_accepts_log = callable_accepts_mode(self.reasoner.plan, parameter="log")

    def run_once(self, project_id: int, release_project: bool = True) -> DispatchResult:
        project = self.client.get_project(project_id)
        return self._run_project_step(project, release_project=release_project)

    def run_loop(
        self,
        project_id: int,
        interval_seconds: float = 3.0,
        max_steps: int | None = None,
    ) -> list[DispatchResult]:
        results: list[DispatchResult] = []
        steps = 0
        if max_steps == 0:
            return results
        while max_steps is None or steps < max_steps:
            project = self.client.get_project(project_id)
            result = self._run_project_step(project, release_project=False)
            results.append(result)
            steps += 1
            if result.mode in {DispatchMode.COMPLETED, DispatchMode.SKIPPED}:
                break
            time.sleep(interval_seconds)
        self._release_project_if_running(project_id)
        return results

    def _run_project_step(self, project: Project, release_project: bool) -> DispatchResult:
        if project.is_completed:
            return DispatchResult(
                project.id,
                DispatchMode.COMPLETED,
                project.status,
                "project already completed",
            )

        if not project.can_execute and project.status != ProjectStatus.RUNNING:
            return DispatchResult(
                project.id,
                DispatchMode.SKIPPED,
                project.status,
                f"project is not executable in status {project.status.value}",
            )

        mode = self._enter_running(project)
        self.client.save_project(project)
        self._log(project.id, f"project entered {mode.value} mode", "info", {"status": project.status.value})

        if project.is_first_run:
            intent = self._create_initial_intent(project)
            return self._first_run(project, intent, release_project=release_project)

        intents = project.active_intents()
        if intents:
            return self._explore(project, intents, mode, release_project=release_project)
        return self._reason(project, release_project=release_project)

    def _first_run(self, project: Project, intent: Intent, release_project: bool) -> DispatchResult:
        intent.mark_running()
        self.client.save_project(project)
        self._log(project.id, "first-run initial intent started", "info", {"intent_id": intent.id})

        try:
            result = self._execute_intent(project, intent, DispatchMode.FIRST_RUN)
        except Exception as exc:
            result = WorkerResult(
                answer=None,
                fact_description=self._execution_failed_fact("first-run", exc),
            )
            self._log(
                project.id,
                f"first-run initial intent failed: {type(exc).__name__}: {exc}",
                "error",
                {"intent_id": intent.id},
            )
        fact_text = self._worker_result_fact_text(result)
        if fact_text:
            fact = project.add_fact(fact_text)
            self._complete_intent_with_fact(project, intent, fact.id)
        else:
            intent.mark_completed()
        if release_project:
            self._release_project(project)
        self.client.save_project(project)
        self._log(project.id, "first-run finished without flag", "info", {"intent_id": intent.id})
        return DispatchResult(project.id, DispatchMode.FIRST_RUN, project.status, "initial intent executed")

    def _explore(
        self,
        project: Project,
        intents: list[Intent],
        mode: DispatchMode,
        release_project: bool,
    ) -> DispatchResult:
        for intent in intents:
            intent.mark_running()
        self.client.save_project(project)
        self._log(
            project.id,
            f"explore started with {len(intents)} active intents",
            "info",
            {"intent_ids": [intent.id for intent in intents]},
        )

        completed_count = 0
        for item in self._execute_intents(project, intents):
            completed_count += 1
            self._apply_intent_result(project, item)
            self.client.save_project(project)
            self._log(
                project.id,
                f"explore intent {item.intent.id} finished",
                "info",
                {
                    "intent_id": item.intent.id,
                    "completed_count": completed_count,
                    "total_count": len(intents),
                    "has_fact": bool(item.result.fact_description),
                    "has_answer": bool(item.result.answer),
                },
            )

        if release_project:
            self._release_project(project)
        self.client.save_project(project)
        self._log(project.id, f"explore finished {completed_count} intents", "info")
        return DispatchResult(project.id, mode, project.status, f"executed {completed_count} intents")

    def _reason(self, project: Project, release_project: bool) -> DispatchResult:
        self._log(project.id, "reason started", "info")
        if self._reasoner_accepts_log:
            result = self.reasoner.plan(project, log=self._agent_log(project.id))  # type: ignore[call-arg]
        else:
            result = self.reasoner.plan(project)
        if result.completed:
            if result.answer:
                self._set_end_fact_result(project, result.answer)
                self._connect_answer_fact_to_goal(project, result.answer)
            self._complete_project(project)
            self.client.save_project(project)
            self._log(project.id, "reason marked project completed", "success", {"has_answer": bool(result.answer)})
            return DispatchResult(project.id, DispatchMode.COMPLETED, project.status, "reason marked project completed")

        planned_intents = self._planned_intents_from_reason_result(project, result)
        planned_count = 0
        for item in planned_intents:
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            to_fact_id = self._valid_fact_id(project, item.get("to"))
            if to_fact_id is None and 1 in project.facts:
                to_fact_id = 1
            project.add_intent(
                description=description,
                from_fact_id=self._valid_fact_id(project, item.get("from")),
                to_fact_id=to_fact_id,
            )
            planned_count += 1
        if release_project:
            self._release_project(project)
        self.client.save_project(project)
        self._log(project.id, f"reason planned {planned_count} new intents", "info")
        return DispatchResult(
            project.id,
            DispatchMode.REASON,
            project.status,
            f"planned {planned_count} intents",
        )

    @classmethod
    def _planned_intents_from_reason_result(
        cls,
        project: Project,
        result: ReasonResult,
    ) -> list[dict[str, object]]:
        fallback_from_fact_id = cls._select_reason_fallback_fact_id(project)
        planned_intents = result.next_intents
        if not planned_intents:
            planned_intents = [
                {"description": description, "from": fallback_from_fact_id, "to": None}
                for description in (result.next_intent_descriptions or [])
            ]
        if planned_intents:
            normalized: list[dict[str, object]] = []
            for item in planned_intents:
                normalized.append(
                    {
                        "description": item.get("description", ""),
                        "from": item.get("from") if item.get("from") is not None else fallback_from_fact_id,
                        "to": item.get("to"),
                    }
                )
            return normalized
        return [
            {
                "description": cls._fallback_reason_intent_description(project, fallback_from_fact_id),
                "from": fallback_from_fact_id,
                "to": None,
            }
        ]

    @staticmethod
    def _select_reason_fallback_fact_id(project: Project) -> int | None:
        if not project.facts:
            return None
        from_fact_ids = {intent.from_fact_id for intent in project.intents.values() if intent.from_fact_id is not None}
        candidate_ids = sorted(fact_id for fact_id in project.facts if fact_id not in from_fact_ids and fact_id != 1)
        if candidate_ids:
            return candidate_ids[-1]
        if 1 in project.facts and len(project.facts) > 1:
            return max(fact_id for fact_id in project.facts if fact_id != 1)
        return max(project.facts)

    @staticmethod
    def _fallback_reason_intent_description(project: Project, from_fact_id: int | None) -> str:
        if from_fact_id is not None and from_fact_id in project.facts:
            fact_text = project.facts[from_fact_id].description.strip().splitlines()[0]
            if fact_text:
                return f"基于 Fact #{from_fact_id} 分析下一步可验证的漏洞方向"
        return "分析当前事实图并验证下一步可利用方向"

    @staticmethod
    def _valid_fact_id(project: Project, value: object) -> int | None:
        if isinstance(value, int) and value in project.facts:
            return value
        return None

    def _execute_intents(self, project: Project, intents: list[Intent]) -> Iterable[IntentExecutionResult]:
        max_workers = min(self.max_explore_workers, len(intents))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._execute_intent, project, intent, DispatchMode.EXPLORE): intent
                for intent in intents
            }
            for future in as_completed(futures):
                intent = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = WorkerResult(
                        fact_description=self._execution_failed_fact(f"intent {intent.id}", exc)
                    )
                yield IntentExecutionResult(intent=intent, result=result)

    @staticmethod
    def _execution_failed_fact(scope: str, exc: Exception) -> str:
        text = str(exc)
        if "no-output timed out" in text:
            return f"{scope} 执行超时：Codex 长时间无 stdout/stderr 输出，可能卡在长期运行或无输出命令；需要 reason 重新规划更小的验证步骤"
        if len(text) > 300:
            text = f"{text[:297]}..."
        return f"{scope} execution failed: {type(exc).__name__}: {text}"

    @staticmethod
    def _apply_intent_result(project: Project, item: IntentExecutionResult) -> None:
        fact_text = Dispatcher._worker_result_fact_text(item.result)
        if fact_text:
            fact = project.add_fact(fact_text)
            Dispatcher._complete_intent_with_fact(project, item.intent, fact.id)
        else:
            item.intent.mark_completed()

    def _execute_intent(self, project: Project, intent: Intent, mode: DispatchMode) -> WorkerResult:
        if self._worker_accepts_mode and self._worker_accepts_log:
            return self.worker.execute_intent(project, intent, mode, log=self._agent_log(project.id))  # type: ignore[call-arg]
        if self._worker_accepts_mode:
            return self.worker.execute_intent(project, intent, mode)  # type: ignore[call-arg]
        if self._worker_accepts_log:
            return self.worker.execute_intent(project, intent, log=self._agent_log(project.id))  # type: ignore[call-arg]
        return self.worker.execute_intent(project, intent)

    @staticmethod
    def _create_initial_intent(project: Project) -> Intent:
        if 0 in project.facts and 1 in project.facts:
            return project.add_intent(
                description="初始探测目标并提取可继续分析的关键事实",
                from_fact_id=0,
                to_fact_id=1,
            )
        return project.add_intent(description="初始探测目标并提取可继续分析的关键事实")

    @staticmethod
    def _complete_intent_with_fact(project: Project, intent: Intent, fact_id: int) -> None:
        intent.mark_completed(to_fact_id=fact_id)

    @staticmethod
    def _worker_result_fact_text(result: WorkerResult) -> str | None:
        if result.fact_description:
            return result.fact_description
        if result.answer:
            return f"agent reported possible final answer: {result.answer}"
        return None

    @staticmethod
    def _set_end_fact_result(project: Project, answer: str) -> None:
        if 1 not in project.facts:
            return
        project.facts[1].description = f"result\nanswer: {answer}"

    @staticmethod
    def _connect_answer_fact_to_goal(project: Project, answer: str) -> None:
        if 1 not in project.facts:
            return
        answer_text = str(answer).strip()
        if not answer_text:
            return
        source_fact = next(
            (
                fact
                for fact in sorted(project.facts.values(), key=lambda item: item.id, reverse=True)
                if fact.id != 1 and answer_text in fact.description
            ),
            None,
        )
        if source_fact is None:
            return
        already_connected = any(
            intent.from_fact_id == source_fact.id and intent.to_fact_id == 1
            for intent in project.intents.values()
        )
        if already_connected:
            return
        intent = project.add_intent(
            description="确认该事实包含最终答案",
            from_fact_id=source_fact.id,
            to_fact_id=1,
        )
        intent.mark_running()
        intent.mark_completed()

    @staticmethod
    def _enter_running(project: Project) -> DispatchMode:
        if project.status == ProjectStatus.ACTIVE:
            mode = DispatchMode.FIRST_RUN if project.is_first_run else DispatchMode.EXPLORE
            project.start()
            return mode
        if project.status == ProjectStatus.STOPED:
            project.start()
        return DispatchMode.FIRST_RUN if project.is_first_run else DispatchMode.EXPLORE

    @staticmethod
    def _complete_project(project: Project) -> None:
        if project.status == ProjectStatus.COMPLETED:
            return
        if project.status == ProjectStatus.INACTIVE:
            project.activate()
        if project.status in {ProjectStatus.ACTIVE, ProjectStatus.STOPED}:
            project.start()
        if project.status == ProjectStatus.RUNNING:
            project.complete()

    @staticmethod
    def _release_project(project: Project) -> None:
        if project.status == ProjectStatus.RUNNING:
            project.stop()

    def _release_project_if_running(self, project_id: int) -> None:
        project = self.client.get_project(project_id)
        if project.status == ProjectStatus.RUNNING:
            self._release_project(project)
            self.client.save_project(project)
            self._log(project.id, "project released to stoped", "info")

    def _log(
        self,
        project_id: int,
        message: str,
        level: str = "info",
        metadata: dict[str, object] | None = None,
    ) -> None:
        add_log = getattr(self.client, "add_project_log", None)
        if add_log is None:
            return
        try:
            add_log(project_id, message, level=level, metadata=metadata or {})
        except Exception:
            return

    def _agent_log(self, project_id: int):
        def log(message: str, level: str = "info", metadata: dict[str, object] | None = None) -> None:
            self._log(project_id, message, level=level, metadata=metadata or {})

        return log
