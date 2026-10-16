from __future__ import annotations

import threading
from dataclasses import dataclass

from echo_agent.config import DispatcherSettings
from echo_agent.dispatcher import Dispatcher
from echo_agent.dispatcher.core import step_consumes_budget, step_requests_trailing_explore
from echo_agent.models import Project, ProjectStatus
from echo_agent.server.docker_runtime import DockerProjectRuntime, DockerRuntimeError
from echo_agent.storage import SQLiteProjectStore


@dataclass(slots=True)
class DispatcherRunState:
    project_id: int
    running: bool
    stop_requested: bool = False
    last_message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "running": self.running,
            "stop_requested": self.stop_requested,
            "last_message": self.last_message,
        }


class DispatcherRunner:
    def __init__(self, store: SQLiteProjectStore, runtime: DockerProjectRuntime | None = None) -> None:
        self.store = store
        self.runtime = runtime or DockerProjectRuntime.from_settings()
        self._lock = threading.Lock()
        self._runs: dict[int, _RunHandle] = {}

    def start(
        self,
        project_id: int,
        interval_seconds: float = 3.0,
        max_steps: int | None = None,
        max_workers: int = 4,
    ) -> DispatcherRunState:
        project = self.store.get_project(project_id)
        if project.status == ProjectStatus.INACTIVE:
            raise ValueError("project must be activated before it can run")
        if project.status == ProjectStatus.COMPLETED:
            raise ValueError("completed project cannot run")
        with self._lock:
            existing = self._runs.get(project_id)
            if existing and existing.thread.is_alive():
                return existing.state

            stop_event = threading.Event()
            state = DispatcherRunState(project_id=project_id, running=True)
            thread = threading.Thread(
                target=self._run_project,
                args=(project_id, interval_seconds, max_steps, max_workers, stop_event, state),
                daemon=True,
                name=f"echo-agent-dispatcher-{project_id}",
            )
            self._runs[project_id] = _RunHandle(thread=thread, stop_event=stop_event, state=state)
            thread.start()
            return state

    def stop(self, project_id: int, *, join_timeout: float = 5.0) -> DispatcherRunState:
        with self._lock:
            handle = self._runs.get(project_id)
            if handle is None:
                state = DispatcherRunState(project_id=project_id, running=False, last_message="not running")
                dispatcher = None
                thread = None
            else:
                handle.stop_event.set()
                handle.state.stop_requested = True
                handle.state.last_message = "stop requested; running codex processes cancelled"
                state = handle.state
                dispatcher = handle.dispatcher
                thread = handle.thread
        if dispatcher is not None:
            dispatcher.request_stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                self._force_stop_container(project_id)
                thread.join(timeout=join_timeout)
        self._release_idle_project(project_id)
        state.running = thread is not None and thread.is_alive()
        return state

    def _force_stop_container(self, project_id: int) -> None:
        project = self.store.get_project(project_id)
        self.runtime.force_stop_project_container(
            project,
            log=lambda message, level="info", metadata=None: self.store.add_project_log(
                project.id,
                message,
                level=level,
                metadata=metadata or {},
            ),
        )

    def _release_idle_project(self, project_id: int) -> None:
        project = self.store.get_project(project_id)
        if project.status != ProjectStatus.RUNNING:
            return
        self.runtime.stop_project_container(
            project,
            log=lambda message, level="info", metadata=None: self.store.add_project_log(
                project.id,
                message,
                level=level,
                metadata=metadata or {},
            ),
        )
        project.stop()
        self.store.add_project_log(
            project.id,
            "project status transition: running -> stoped",
            metadata={"old_status": "running", "new_status": "stoped", "forced": False},
        )
        self.store.save_project(project)

    def status(self, project_id: int) -> DispatcherRunState:
        with self._lock:
            handle = self._runs.get(project_id)
            if handle is None:
                return DispatcherRunState(project_id=project_id, running=False, last_message="not running")
            handle.state.running = handle.thread.is_alive()
            return handle.state

    def _run_project(
        self,
        project_id: int,
        interval_seconds: float,
        max_steps: int | None,
        max_workers: int,
        stop_event: threading.Event,
        state: DispatcherRunState,
    ) -> None:
        client = StoreDispatcherClient(self.store, self.runtime)
        settings = DispatcherSettings.from_env()
        dispatcher = Dispatcher(
            client,
            max_explore_workers=max_workers,
            max_codex_slots=settings.codex_max_slots,
            stop_event=stop_event,
            codex_model=settings.codex_model,
            codex_mode_models=settings.codex_mode_models,
            codex_workspace=settings.codex_workspace,
            codex_timeout_seconds=settings.codex_timeout_seconds,
            codex_no_output_timeout_seconds=settings.codex_no_output_timeout_seconds,
            codex_mode_no_output_timeout_seconds=settings.codex_mode_no_output_timeout_seconds,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            codex_command=settings.codex_command,
            codex_provider_name=settings.codex_provider_name,
            codex_provider_supports_websockets=settings.codex_provider_supports_websockets,
            codex_request_max_retries=settings.codex_request_max_retries,
            codex_stream_max_retries=settings.codex_stream_max_retries,
            codex_stream_idle_timeout_ms=settings.codex_stream_idle_timeout_ms,
        )
        with self._lock:
            handle = self._runs.get(project_id)
            if handle is not None:
                handle.dispatcher = dispatcher
        if stop_event.is_set():
            dispatcher.request_stop()
        steps = 0
        observed = 0
        try:
            client.add_project_log(
                project_id,
                "server dispatcher started",
                metadata={
                    "max_steps": max_steps,
                    "codex_model": settings.codex_model,
                    "codex_mode_models": settings.codex_mode_models,
                },
            )
            consecutive_failures = 0
            while not stop_event.is_set() and (max_steps is None or steps < max_steps):
                try:
                    result = dispatcher.run_once(project_id, release_project=False)
                except Exception as exc:
                    consecutive_failures += 1
                    state.last_message = f"{type(exc).__name__}: {exc}"
                    client.add_project_log(project_id, state.last_message, level="error")
                    if consecutive_failures >= 3:
                        break
                    stop_event.wait(interval_seconds)
                    continue
                consecutive_failures = 0
                observed += 1
                if step_consumes_budget(result):
                    steps += 1
                state.last_message = result.message
                client.add_project_log(
                    project_id,
                    f"server dispatcher step {observed} finished",
                    metadata={
                        "mode": result.mode.value,
                        "status": result.status.value,
                        "message": result.message,
                        "budget_steps": steps,
                    },
                )
                if result.mode.value in {"completed", "skipped"}:
                    break
                if step_consumes_budget(result) and max_steps is not None and steps >= max_steps:
                    if step_requests_trailing_explore(result) and not stop_event.is_set():
                        try:
                            trailing = dispatcher.run_once(project_id, release_project=False)
                        except Exception as exc:
                            state.last_message = f"{type(exc).__name__}: {exc}"
                            client.add_project_log(project_id, state.last_message, level="error")
                        else:
                            observed += 1
                            state.last_message = trailing.message
                            client.add_project_log(
                                project_id,
                                f"server dispatcher step {observed} finished",
                                metadata={
                                    "mode": trailing.mode.value,
                                    "status": trailing.status.value,
                                    "message": trailing.message,
                                    "budget_steps": steps,
                                    "trailing_explore": True,
                                },
                            )
                        break
                    break
                stop_event.wait(interval_seconds)
        except Exception as exc:
            state.last_message = f"{type(exc).__name__}: {exc}"
            client.add_project_log(project_id, state.last_message, level="error")
        finally:
            dispatcher._release_project_if_running(project_id)
            state.running = False
            client.add_project_log(
                project_id,
                "server dispatcher stopped",
                metadata={"steps": observed, "budget_steps": steps},
            )

class StoreDispatcherClient:
    def __init__(self, store: SQLiteProjectStore, runtime: DockerProjectRuntime | None = None) -> None:
        self.store = store
        self.runtime = runtime or DockerProjectRuntime.from_settings()

    def get_project(self, project_id: int) -> Project:
        return self.store.get_project(project_id)

    def save_project(self, project: Project) -> Project:
        old_status = self.store.get_project(project.id).status
        if old_status != project.status:
            self.add_project_log(
                project.id,
                f"project status transition: {old_status.value} -> {project.status.value}",
                metadata={"old_status": old_status.value, "new_status": project.status.value},
            )
        try:
            self.runtime.apply_status_transition(
                project,
                old_status,
                log=lambda message, level="info", metadata=None: self.add_project_log(
                    project.id,
                    message,
                    level=level,
                    metadata=metadata or {},
                ),
            )
        except DockerRuntimeError as exc:
            self.add_project_log(
                project.id,
                f"project container lifecycle failed for {project.status.value}: {exc}",
                level="error",
                metadata={"docker": project.docker, "status": project.status.value},
            )
            raise
        if old_status != project.status:
            self.add_project_log(
                project.id,
                f"project container lifecycle applied for {project.status.value}",
                metadata={"docker": project.docker, "status": project.status.value},
            )
        return self.store.save_project(project)

    def add_project_log(
        self,
        project_id: int,
        message: str,
        level: str = "info",
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return self.store.add_project_log(project_id, message, level=level, metadata=metadata)


@dataclass(slots=True)
class _RunHandle:
    thread: threading.Thread
    stop_event: threading.Event
    state: DispatcherRunState
    dispatcher: Dispatcher | None = None
