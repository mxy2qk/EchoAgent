from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from echo_agent.models import ProjectStatus
from echo_agent.server.dispatcher_runner import DispatcherRunner, DispatcherRunState, _RunHandle
from echo_agent.server.docker_runtime import DockerProjectRuntime
from echo_agent.storage import SQLiteProjectStore


class ForceStopRuntime(DockerProjectRuntime):
    def __init__(self) -> None:
        super().__init__(
            image="worker:test",
            context=Path("container"),
            dockerfile=Path("container/Dockerfile"),
        )
        object.__setattr__(self, "commands", [])

    def _container_state(self, container: str) -> str | None:
        return "running"

    def _run(self, command, log=None, output_prefix: str = "docker") -> None:
        self.commands.append(list(command))


class DispatcherRunnerStopTests(unittest.TestCase):
    def test_stop_releases_running_project_without_force_killing_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteProjectStore(Path(directory) / "state.db")
            project = store.create_project(project_id=7, target_url="http://target", objective="get flag")
            project.activate()
            project.start()
            store.save_project(project)
            runtime = ForceStopRuntime()
            runner = DispatcherRunner(store, runtime=runtime)

            state = runner.stop(project.id)

            saved = store.get_project(project.id)
            self.assertFalse(state.running)
            self.assertEqual(saved.status, ProjectStatus.STOPED)
            self.assertEqual(runtime.commands, [["docker", "stop", "echo-agent-task-7"]])
            self.assertNotIn("-t", runtime.commands[0])


    def test_stop_cancels_running_codex_processes_without_force_stopping_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteProjectStore(Path(directory) / "state.db")
            project = store.create_project(project_id=7, target_url="http://target", objective="get flag")
            project.activate()
            project.start()
            store.save_project(project)
            runtime = ForceStopRuntime()
            runner = DispatcherRunner(store, runtime=runtime)
            stop_event = threading.Event()
            started = threading.Event()

            def wait_for_stop() -> None:
                started.set()
                stop_event.wait(2)

            thread = threading.Thread(target=wait_for_stop)
            thread.start()
            started.wait(1)

            class CancellingDispatcher:
                def __init__(self) -> None:
                    self.stopped = False

                def request_stop(self) -> None:
                    self.stopped = True

            dispatcher = CancellingDispatcher()
            runner._runs[project.id] = _RunHandle(
                thread=thread,
                stop_event=stop_event,
                state=DispatcherRunState(project_id=project.id, running=True),
                dispatcher=dispatcher,  # type: ignore[arg-type]
            )

            state = runner.stop(project.id)
            thread.join(timeout=1)

            saved = store.get_project(project.id)
            self.assertTrue(dispatcher.stopped)
            self.assertTrue(state.stop_requested)
            self.assertFalse(state.running)
            self.assertEqual(saved.status, ProjectStatus.STOPED)
            self.assertEqual(runtime.commands, [["docker", "stop", "echo-agent-task-7"]])


if __name__ == "__main__":
    unittest.main()
