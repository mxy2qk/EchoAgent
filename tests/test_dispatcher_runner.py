from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from echo_agent.models import ProjectStatus
from echo_agent.server.dispatcher_runner import DispatcherRunner
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
    def test_stop_force_stops_docker_and_releases_running_project(self) -> None:
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
            self.assertEqual(runtime.commands, [["docker", "stop", "-t", "0", "echo-agent-task-7"]])


if __name__ == "__main__":
    unittest.main()
