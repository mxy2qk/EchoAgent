from __future__ import annotations

import copy
import threading
import time
import unittest
from typing import Any

from echo_agent.dispatcher.core import Dispatcher
from echo_agent.dispatcher.types import DispatchMode, ReasonResult, WorkerResult
from echo_agent.models import FactKind, IntentOutcome, Project, ProjectStatus


class MemoryProjectClient:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.logs: list[dict[str, Any]] = []
        self.saved_projects: list[Project] = []

    def get_project(self, project_id: int) -> Project:
        self.assert_project_id(project_id)
        return copy.deepcopy(self.project)

    def save_project(self, project: Project) -> dict[str, Any]:
        self.assert_project_id(project.id)
        self.project = copy.deepcopy(project)
        self.saved_projects.append(copy.deepcopy(project))
        return self.project.to_dict()

    def add_project_log(
        self,
        project_id: int,
        message: str,
        level: str = "info",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.assert_project_id(project_id)
        self.logs.append({"message": message, "level": level, "metadata": metadata or {}})

    def assert_project_id(self, project_id: int) -> None:
        if project_id != self.project.id:
            raise AssertionError(f"unexpected project id: {project_id}")


class StaticWorker:
    def __init__(self, fact_description: str) -> None:
        self.fact_description = fact_description

    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        return WorkerResult(answer=None, fact_description=self.fact_description)


class FailingWorker:
    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        raise RuntimeError("agent stalled")


class OrderedConcurrentWorker:
    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        if intent.id == 0:
            time.sleep(0.01)
            return WorkerResult(answer=None, fact_description="first fact")
        time.sleep(0.1)
        return WorkerResult(answer=None, fact_description="second fact")


class IntentFactWorker:
    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        return WorkerResult(answer=None, fact_description=f"fact from intent {intent.id}")


class FlagWorker:
    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        return WorkerResult(answer="flag{ok}", fact_description=None)


class PlaceholderAnswerWorker:
    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        return WorkerResult(answer="真实 flag 字符串", fact_description=None)


class FlagFactWorker:
    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        return WorkerResult(answer=None, fact_description="confirmed final answer flag{fact}")


class StaticReasoner:
    def plan(self, project: Project) -> ReasonResult:
        return ReasonResult(
            completed=False,
            next_intents=[
                {"from": 2, "to": None, "description": "枚举目录"},
                {"from": 2, "to": None, "description": "测试上传点"},
            ],
        )


class EmptyReasoner:
    def plan(self, project: Project) -> ReasonResult:
        return ReasonResult(completed=False, next_intents=[])


class CompletedReasoner:
    def plan(self, project: Project) -> ReasonResult:
        return ReasonResult(completed=True, next_intents=[])


class FlagReasoner:
    def plan(self, project: Project) -> ReasonResult:
        return ReasonResult(completed=True, answer="flag{reason}", next_intents=[])


class GraphAnswerReasoner:
    def plan(self, project: Project) -> ReasonResult:
        answers = [
            fact.description.removeprefix("agent reported possible final answer: ").strip()
            for fact in project.facts.values()
            if fact.description.startswith("agent reported possible final answer: ")
        ]
        if answers:
            return ReasonResult(completed=True, answer=answers[-1], next_intents=[])
        return ReasonResult(completed=False, next_intents=[])


class LegacyReasoner:
    def plan(self, project: Project) -> ReasonResult:
        return ReasonResult(completed=False, next_intent_descriptions=["分析是否存在文件上传漏洞"])


class DispatcherFactInsertionTests(unittest.TestCase):
    def test_first_run_fact_completes_initial_intent_without_auto_bridge(self) -> None:
        project = Project(
            id=1,
            name="test",
            status=ProjectStatus.ACTIVE,
            target_url="http://target",
            objective="get flag",
        )
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("discovered web framework: flask"),  # type: ignore[arg-type]
        )

        dispatcher.run_once(project.id)

        saved = client.project
        self.assertEqual(saved.facts[2].description, "discovered web framework: flask")
        self.assertEqual(saved.intents[0].from_fact_id, 0)
        self.assertEqual(saved.intents[0].to_fact_id, 2)
        self.assertTrue(saved.intents[0].is_completed)
        self.assertEqual(len(saved.intents), 1)
        self.assertEqual(saved.active_intents(), [])

    def test_first_run_failure_becomes_fact_for_reason(self) -> None:
        project = Project(
            id=1,
            name="test",
            status=ProjectStatus.ACTIVE,
            target_url="http://target",
            objective="get flag",
        )
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=FailingWorker(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id)

        saved = client.project
        self.assertEqual(result.mode, DispatchMode.FIRST_RUN)
        self.assertNotIn(2, saved.facts)
        self.assertEqual(saved.intents[0].status.value, "failed")
        self.assertEqual(saved.intents[0].outcome.value, "error")
        self.assertEqual(saved.intents[0].from_fact_id, 0)
        self.assertIn("first-run execution failed", saved.intents[0].note)
        self.assertEqual(saved.status, ProjectStatus.STOPED)

    def test_no_output_timeout_failure_fact_is_actionable(self) -> None:
        fact = Dispatcher._execution_failed_fact(
            "intent 3",
            RuntimeError("codex agent no-output timed out in explore mode after 900s without stdout/stderr"),
        )

        self.assertIn("Codex 长时间无 stdout/stderr 输出", fact)
        self.assertIn("重新规划更小的验证步骤", fact)

    def test_reason_can_branch_multiple_intents_from_same_fact(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("web service exposes upload form", fact_id=2)
        seed_intent = project.add_intent("initial discovery", from_fact_id=0, to_fact_id=2)
        seed_intent.mark_running()
        seed_intent.mark_completed()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=StaticReasoner(),  # type: ignore[arg-type]
        )

        dispatcher.run_once(project.id)

        saved = client.project
        branch_intents = [intent for intent in saved.intents.values() if intent.from_fact_id == 2]
        self.assertEqual(len(branch_intents), 2)
        self.assertEqual([intent.description for intent in branch_intents], ["枚举目录", "测试上传点"])
        self.assertTrue(all(intent.to_fact_id == 1 for intent in branch_intents))

    def test_reason_creates_next_intents_after_first_run_fact(self) -> None:
        project = Project(
            id=1,
            name="test",
            status=ProjectStatus.ACTIVE,
            target_url="http://target",
            objective="get flag",
        )
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("web service exposes upload form"),  # type: ignore[arg-type]
            reasoner=StaticReasoner(),  # type: ignore[arg-type]
        )

        first_result = dispatcher.run_once(project.id)
        second_result = dispatcher.run_once(project.id)

        self.assertEqual(first_result.mode, DispatchMode.FIRST_RUN)
        self.assertEqual(second_result.mode, DispatchMode.REASON)
        saved = client.project
        self.assertEqual(saved.intents[0].from_fact_id, 0)
        self.assertEqual(saved.intents[0].to_fact_id, 2)
        branch_intents = [intent for intent in saved.intents.values() if intent.from_fact_id == 2]
        self.assertEqual(len(branch_intents), 2)
        self.assertEqual([intent.description for intent in branch_intents], ["枚举目录", "测试上传点"])
        self.assertTrue(all(intent.to_fact_id == 1 for intent in branch_intents))

    def test_explore_saves_each_completed_intent_before_all_finish(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("web service exposes upload form", fact_id=2)
        project.add_intent("fast intent", from_fact_id=2, to_fact_id=1)
        project.add_intent("slow intent", from_fact_id=2, to_fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=OrderedConcurrentWorker(),  # type: ignore[arg-type]
            max_explore_workers=2,
        )

        dispatcher.run_once(project.id)

        progressive_snapshots = [
            snapshot
            for snapshot in client.saved_projects
            if (
                len(snapshot.facts) == 4
                and snapshot.intents[0].is_completed
                and snapshot.intents[0].to_fact_id == 3
                and snapshot.intents[1].status.value == "running"
                and snapshot.intents[1].to_fact_id == 1
            )
        ]
        self.assertTrue(progressive_snapshots)
        saved = client.project
        self.assertTrue(saved.intents[0].is_completed)
        self.assertTrue(saved.intents[1].is_completed)
        self.assertEqual(saved.intents[0].to_fact_id, 3)
        self.assertEqual(saved.intents[1].to_fact_id, 4)
        self.assertEqual(saved.facts[3].description, "first fact")
        self.assertEqual(saved.facts[4].description, "second fact")

    def test_explore_answer_is_inserted_as_fact_for_reason_judgement(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("web service exposes upload form", fact_id=2)
        project.add_intent("try exploit", from_fact_id=2, to_fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=FlagWorker(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id)

        self.assertEqual(result.mode, DispatchMode.EXPLORE)
        self.assertEqual(client.project.status, ProjectStatus.STOPED)
        self.assertEqual(client.project.intents[0].to_fact_id, 3)
        self.assertTrue(client.project.intents[0].is_completed)
        self.assertEqual(client.project.facts[1].description, "goal")
        self.assertEqual(client.project.facts[3].description, "agent reported possible final answer: flag{ok}")

    def test_first_run_answer_is_inserted_as_fact_for_reason_judgement(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=FlagWorker(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id)

        self.assertEqual(result.mode, DispatchMode.FIRST_RUN)
        self.assertEqual(client.project.facts[1].description, "goal")
        self.assertEqual(client.project.facts[2].description, "agent reported possible final answer: flag{ok}")

    def test_explore_fact_with_flag_form_does_not_complete_without_reason(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("web service exposes upload form", fact_id=2)
        project.add_intent("try exploit", from_fact_id=2, to_fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=FlagFactWorker(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id)

        self.assertEqual(result.mode, DispatchMode.EXPLORE)
        self.assertEqual(client.project.status, ProjectStatus.STOPED)
        self.assertEqual(client.project.facts[1].description, "goal")
        self.assertEqual(client.project.facts[3].description, "confirmed final answer flag{fact}")

    def test_explore_placeholder_answer_does_not_complete_or_update_end_fact(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("web service exposes upload form", fact_id=2)
        project.add_intent("try exploit", from_fact_id=2, to_fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=PlaceholderAnswerWorker(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id)

        self.assertEqual(result.mode, DispatchMode.EXPLORE)
        self.assertEqual(client.project.status, ProjectStatus.STOPED)
        self.assertEqual(client.project.facts[1].description, "goal")
        self.assertTrue(client.project.intents[0].is_completed)

    def test_reason_completion_requires_answer_in_evidence(self) -> None:
        project = self._project_ready_for_reason()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=FlagReasoner(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id)

        self.assertEqual(result.mode, DispatchMode.REASON)
        self.assertEqual(client.project.facts[1].description, "goal")
        self.assertNotEqual(client.project.status, ProjectStatus.COMPLETED)
        self.assertFalse(
            any(intent.description == "确认该事实包含最终答案" for intent in client.project.intents.values())
        )

    def test_existing_graph_answer_only_completes_after_reason_judgement(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("agent reported possible final answer: flag{graph}", fact_id=2)
        seed_intent = project.add_intent("initial discovery", from_fact_id=0, to_fact_id=2)
        seed_intent.mark_running()
        seed_intent.mark_completed()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=GraphAnswerReasoner(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id)

        self.assertEqual(result.mode, DispatchMode.COMPLETED)
        self.assertEqual(client.project.facts[1].description, "result\nanswer: flag{graph}")
        answer_edges = [
            intent
            for intent in client.project.intents.values()
            if intent.from_fact_id == 2 and intent.to_fact_id == 1
        ]
        self.assertEqual(len(answer_edges), 1)
        self.assertEqual(answer_edges[0].description, "确认该事实包含最终答案")
        self.assertTrue(answer_edges[0].is_completed)

    def test_reason_answer_does_not_create_goal_edge_without_source_fact(self) -> None:
        project = self._project_ready_for_reason()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=FlagReasoner(),  # type: ignore[arg-type]
        )

        dispatcher.run_once(project.id)

        self.assertFalse(
            any(intent.to_fact_id == 1 and intent.description == "确认该事实包含最终答案" for intent in client.project.intents.values())
        )

    def test_run_loop_continues_to_reason_after_parallel_explore_facts_without_flag(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("web service exposes upload form", fact_id=2)
        for index in range(5):
            project.add_intent(f"intent {index}", from_fact_id=2)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=IntentFactWorker(),  # type: ignore[arg-type]
            reasoner=StaticReasoner(),  # type: ignore[arg-type]
            max_explore_workers=5,
        )

        results = dispatcher.run_loop(project.id, interval_seconds=0, max_steps=2)

        self.assertEqual([result.mode for result in results], [DispatchMode.EXPLORE, DispatchMode.REASON])
        saved = client.project
        self.assertEqual(len([intent for intent in saved.intents.values() if intent.is_completed]), 5)
        branch_intents = [
            intent
            for intent in saved.intents.values()
            if intent.from_fact_id == 2 and intent.is_executable
        ]
        self.assertEqual([intent.description for intent in branch_intents], ["枚举目录", "测试上传点"])
        self.assertTrue(all(intent.to_fact_id == 1 for intent in branch_intents))

    def test_reason_empty_plan_creates_fallback_intent(self) -> None:
        project = self._project_ready_for_reason()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=EmptyReasoner(),  # type: ignore[arg-type]
        )

        dispatcher.run_once(project.id)

        active_intents = client.project.active_intents()
        self.assertEqual(len(active_intents), 1)
        self.assertEqual(active_intents[0].from_fact_id, 2)
        self.assertIn("分析下一步可验证的漏洞方向", active_intents[0].description)

    def test_reason_completed_without_answer_still_completes_by_codex_judgement(self) -> None:
        project = self._project_ready_for_reason()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=CompletedReasoner(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id)

        self.assertEqual(result.mode, DispatchMode.REASON)
        self.assertNotEqual(client.project.status, ProjectStatus.COMPLETED)
        self.assertEqual(client.project.facts[1].description, "goal")

    def test_legacy_reason_descriptions_use_leaf_fact_as_fallback(self) -> None:
        project = self._project_ready_for_reason()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=LegacyReasoner(),  # type: ignore[arg-type]
        )

        dispatcher.run_once(project.id)

        active_intents = client.project.active_intents()
        self.assertEqual(len(active_intents), 1)
        self.assertEqual(active_intents[0].from_fact_id, 2)
        self.assertEqual(active_intents[0].description, "分析是否存在文件上传漏洞")

    @staticmethod
    def _project_ready_for_reason() -> Project:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("web service exposes upload form", fact_id=2)
        seed_intent = project.add_intent("initial discovery", from_fact_id=0, to_fact_id=2)
        seed_intent.mark_running()
        seed_intent.mark_completed()
        return project


class SlotGateWorker:
    def __init__(self) -> None:
        self.snapshot: dict[int, str] = {}

    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        if not self.snapshot:
            time.sleep(0.05)
            self.snapshot = {item.id: item.status.value for item in project.intents.values()}
        return WorkerResult(fact_description=f"fact from intent {intent.id}")


class RetryableFailureWorker:
    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        raise RuntimeError("sqlite database is locked")


class FlakyRetryWorker:
    def __init__(self) -> None:
        self.calls = 0

    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("HTTP 429 too many requests")
        return WorkerResult(fact_description="recovered after retry")


class CodexSlotTests(unittest.TestCase):
    def test_waiting_intent_stays_active_until_codex_slot_is_free(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_intent("first", from_fact_id=0, to_fact_id=1)
        project.add_intent("second", from_fact_id=0, to_fact_id=1)
        client = MemoryProjectClient(project)
        worker = SlotGateWorker()
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=worker,  # type: ignore[arg-type]
            max_explore_workers=2,
            max_codex_slots=1,
        )

        dispatcher.run_once(project.id)

        self.assertEqual(worker.snapshot[0], "running")
        self.assertEqual(worker.snapshot[1], "active")
        self.assertTrue(client.project.intents[0].is_completed)
        self.assertTrue(client.project.intents[1].is_completed)

    def test_retryable_failure_returns_intent_to_active_without_fact(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_intent("probe", from_fact_id=0, to_fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=RetryableFailureWorker(),  # type: ignore[arg-type]
            max_codex_slots=1,
        )

        result = dispatcher.run_once(project.id)

        self.assertEqual(result.mode, DispatchMode.EXPLORE)
        self.assertEqual(client.project.intents[0].status.value, "active")
        self.assertEqual(set(client.project.facts), {0, 1})
        self.assertTrue(any("retry failed" in log["message"] for log in client.logs))

    def test_retryable_failure_retries_once_then_completes(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_intent("probe", from_fact_id=0, to_fact_id=1)
        client = MemoryProjectClient(project)
        worker = FlakyRetryWorker()
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=worker,  # type: ignore[arg-type]
        )

        dispatcher.run_once(project.id)

        self.assertEqual(worker.calls, 2)
        self.assertTrue(client.project.intents[0].is_completed)
        self.assertEqual(client.project.facts[2].description, "recovered after retry")

    def test_stop_before_slot_does_not_start_intent(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_intent("probe", from_fact_id=0, to_fact_id=1)
        client = MemoryProjectClient(project)
        stop_event = threading.Event()
        stop_event.set()
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("should not run"),  # type: ignore[arg-type]
            stop_event=stop_event,
        )

        dispatcher.run_once(project.id)

        self.assertEqual(client.project.intents[0].status.value, "active")
        self.assertNotIn(2, client.project.facts)


class RaisingReasoner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, project: Project) -> ReasonResult:
        self.calls += 1
        raise RuntimeError("HTTP 503 service unavailable")


class ExplodingReasoner:
    def plan(self, project: Project) -> ReasonResult:
        raise AssertionError("reason should not run")


class OuterFailureTests(unittest.TestCase):
    def test_stranded_running_intent_is_recovered_before_reason(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        stranded = project.add_intent("probe", from_fact_id=0, to_fact_id=1)
        stranded.mark_running()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("recovered fact"),  # type: ignore[arg-type]
            reasoner=ExplodingReasoner(),  # type: ignore[arg-type]
        )

        result = dispatcher.run_once(project.id, release_project=False)

        self.assertEqual(result.mode, DispatchMode.EXPLORE)
        self.assertTrue(client.project.intents[0].is_completed)
        self.assertEqual(client.project.facts[2].description, "recovered fact")
        self.assertTrue(any("stranded" in log["message"] for log in client.logs))

    def test_retry_budget_turns_persistent_failure_into_fact(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_intent("probe", from_fact_id=0, to_fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=RetryableFailureWorker(),  # type: ignore[arg-type]
            max_retryable_attempts=2,
        )

        first = dispatcher.run_once(project.id, release_project=False)
        second = dispatcher.run_once(project.id, release_project=False)

        self.assertEqual(first.mode, DispatchMode.EXPLORE)
        self.assertEqual(second.mode, DispatchMode.EXPLORE)
        self.assertEqual(client.project.intents[0].status.value, "failed")
        self.assertEqual(client.project.intents[0].outcome.value, "error")
        self.assertNotIn(2, client.project.facts)
        self.assertIn("database is locked", client.project.intents[0].note)
        self.assertIn("execution failed", client.project.intents[0].note)

    def test_transient_gateway_error_is_retryable(self) -> None:
        self.assertTrue(
            Dispatcher._is_retryable_failure(RuntimeError("upstream returned HTTP 503 service unavailable"))
        )
        self.assertFalse(
            Dispatcher._is_retryable_failure(
                RuntimeError("codex agent no-output timed out in explore mode after 900s without stdout/stderr")
            )
        )

    def test_reason_exception_does_not_escape_or_complete_project(self) -> None:
        project = DispatcherFactInsertionTests._project_ready_for_reason()
        client = MemoryProjectClient(project)
        reasoner = RaisingReasoner()
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=reasoner,  # type: ignore[arg-type]
            max_retryable_attempts=2,
        )

        result = dispatcher.run_once(project.id, release_project=False)

        self.assertEqual(result.mode, DispatchMode.REASON)
        self.assertEqual(result.message, "reason failed; will retry")
        self.assertEqual(client.project.status, ProjectStatus.RUNNING)
        self.assertEqual(len(client.project.intents), 1)
        self.assertEqual(reasoner.calls, 1)

    def test_repeated_empty_reason_does_not_grow_fallback_intents(self) -> None:
        project = DispatcherFactInsertionTests._project_ready_for_reason()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("fallback explored"),  # type: ignore[arg-type]
            reasoner=EmptyReasoner(),  # type: ignore[arg-type]
            max_retryable_attempts=2,
        )

        results = dispatcher.run_loop(project.id, interval_seconds=0, max_steps=6)

        fallbacks = [
            intent
            for intent in client.project.intents.values()
            if "分析下一步可验证的漏洞方向" in intent.description
        ]
        self.assertEqual(len(fallbacks), 1)
        self.assertEqual(results[-1].mode, DispatchMode.SKIPPED)
        self.assertNotEqual(client.project.status, ProjectStatus.COMPLETED)


class RefutedWorker:
    def execute_intent(self, project: Project, intent, mode: DispatchMode) -> WorkerResult:
        return WorkerResult(fact_description="上传点不存在可利用漏洞", outcome="refuted")


class GraphContractTests(unittest.TestCase):
    def test_refuted_path_does_not_become_next_from(self) -> None:
        project = Project(id=1, name="test", status=ProjectStatus.ACTIVE)
        project.add_fact("start", fact_id=0)
        project.add_fact("goal", fact_id=1)
        project.add_fact("发现上传点", fact_id=2)
        project.add_intent("分析上传", from_fact_id=2, to_fact_id=1)
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=RefutedWorker(),  # type: ignore[arg-type]
            reasoner=EmptyReasoner(),  # type: ignore[arg-type]
        )

        dispatcher.run_once(project.id, release_project=False)
        dispatcher.run_once(project.id, release_project=False)

        saved = client.project
        self.assertEqual(saved.facts[3].kind, FactKind.REFUTED)
        self.assertEqual(saved.intents[0].outcome, IntentOutcome.REFUTED)
        self.assertTrue(saved.intents[0].is_completed)
        fallback = saved.active_intents()[0]
        self.assertEqual(fallback.from_fact_id, 2)
        self.assertNotEqual(fallback.from_fact_id, 3)

    def test_completed_description_is_not_planned_again(self) -> None:
        project = DispatcherFactInsertionTests._project_ready_for_reason()
        done = project.add_intent("分析是否存在文件上传漏洞", from_fact_id=2, to_fact_id=1)
        done.mark_running()
        done.mark_completed()
        client = MemoryProjectClient(project)
        dispatcher = Dispatcher(
            client=client,  # type: ignore[arg-type]
            worker=StaticWorker("unused"),  # type: ignore[arg-type]
            reasoner=LegacyReasoner(),  # type: ignore[arg-type]
        )

        dispatcher.run_once(project.id)

        matches = [
            intent
            for intent in client.project.intents.values()
            if intent.description == "分析是否存在文件上传漏洞"
        ]
        self.assertEqual(len(matches), 1)



if __name__ == "__main__":
    unittest.main()
