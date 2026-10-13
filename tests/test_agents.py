from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from echo_agent.dispatcher.agents import CodexDockerAgent
from echo_agent.dispatcher.types import DispatchMode


class CodexDockerAgentCommandTests(unittest.TestCase):
    def test_proxy_base_url_is_configured_as_explicit_provider(self) -> None:
        agent = CodexDockerAgent(
            model="test-model",
            openai_api_key="test-key",
            openai_base_url="https://proxy.example/v1",
        )

        command = agent._codex_command_for_mode(DispatchMode.FIRST_RUN)

        self.assertIn('model_provider="echo_agent_proxy"', command)
        self.assertIn(
            'model_providers.echo_agent_proxy.base_url="https://proxy.example/v1"',
            command,
        )
        self.assertIn("model_providers.echo_agent_proxy.supports_websockets=false", command)
        self.assertIn("model_providers.echo_agent_proxy.stream_idle_timeout_ms=60000", command)
        self.assertEqual(command[-3:], ("--model", "test-model", "-"))

    def test_docker_command_passes_base_url_as_explicit_env_value(self) -> None:
        agent = CodexDockerAgent(
            model="test-model",
            openai_api_key="test-key",
            openai_base_url="https://proxy.example/v1",
        )
        project = type("ProjectStub", (), {"docker": "worker-container"})()

        command = agent._docker_command(project, DispatchMode.EXPLORE)  # type: ignore[arg-type]
        safe_command = agent._safe_command(command)

        self.assertIn("OPENAI_API_KEY=test-key", command)
        self.assertIn("OPENAI_BASE_URL=https://proxy.example/v1", command)
        self.assertNotIn("OPENAI_API_KEY=test-key", safe_command)
        self.assertNotIn("OPENAI_BASE_URL=https://proxy.example/v1", safe_command)

    def test_default_openai_provider_is_unchanged_without_base_url(self) -> None:
        agent = CodexDockerAgent(model="test-model", openai_api_key="test-key")

        command = agent._codex_command_for_mode(DispatchMode.EXPLORE)

        self.assertFalse(any(item.startswith("model_provider=") for item in command))
        self.assertEqual(command[-3:], ("--model", "test-model", "-"))

    def test_custom_provider_command_is_not_overridden(self) -> None:
        agent = CodexDockerAgent(
            codex_command=(
                "codex",
                "exec",
                "-c",
                'model_provider="custom"',
                "-",
            ),
            openai_base_url="https://proxy.example/v1",
        )

        command = agent._codex_command_for_mode(DispatchMode.REASON)

        self.assertEqual(command.count('model_provider="custom"'), 1)
        self.assertFalse(any("echo_agent_proxy" in item for item in command))

    def test_process_output_keeps_stderr_out_of_result(self) -> None:
        agent = CodexDockerAgent(timeout_seconds=5)
        logs: list[str] = []

        output = agent._run_process(
            (
                "sh",
                "-c",
                "cat >/dev/null; echo 'progress line 1' >&2; echo 'progress line 2' >&2; echo '{\"answer\":\"done\",\"fact_description\":\"ok\"}'",
            ),
            prompt="hello",
            mode=DispatchMode.EXPLORE,
            env=None,
            log=lambda message, _level, _metadata: logs.append(message),
        )

        self.assertEqual(output, '{"answer":"done","fact_description":"ok"}')
        self.assertIn("codex output:\nprogress line 1\nprogress line 2", logs)
        self.assertIn('codex result:\n{"answer":"done","fact_description":"ok"}', logs)

    def test_nonzero_reason_process_salvages_json_from_stderr(self) -> None:
        agent = CodexDockerAgent(timeout_seconds=5)
        logs: list[tuple[str, str]] = []

        output = agent._run_process(
            (
                "sh",
                "-c",
                "cat >/dev/null; echo '{\"completed\": false, \"next_intents\": [{\"from\": 2, \"to\": null, \"description\": \"枚举目录\"}]}' >&2; echo 'ERROR: stream disconnected' >&2; exit 1",
            ),
            prompt="hello",
            mode=DispatchMode.REASON,
            env=None,
            log=lambda message, level, _metadata: logs.append((level, message)),
        )

        self.assertIn('"completed": false', output)
        self.assertIn('"next_intents"', output)
        self.assertTrue(any(level == "warning" and "salvaged JSON output" in message for level, message in logs))

    def test_process_is_killed_after_no_output_timeout(self) -> None:
        agent = CodexDockerAgent(timeout_seconds=10, no_output_timeout_seconds=1)
        logs: list[tuple[str, str]] = []

        with self.assertRaisesRegex(RuntimeError, "no-output timed out"):
            agent._run_process(
                (
                    "sh",
                    "-c",
                    "cat >/dev/null; sleep 5",
                ),
                prompt="hello",
                mode=DispatchMode.EXPLORE,
                env=None,
                log=lambda message, level, _metadata: logs.append((level, message)),
            )

        self.assertTrue(
            any(level == "error" and "no-output timed out" in message for level, message in logs)
        )

    def test_mode_specific_no_output_timeout_overrides_default(self) -> None:
        agent = CodexDockerAgent(
            no_output_timeout_seconds=900,
            mode_no_output_timeout_seconds={"explore": 1},
        )

        self.assertEqual(agent._no_output_timeout_for_mode(DispatchMode.EXPLORE), 1)
        self.assertEqual(agent._no_output_timeout_for_mode(DispatchMode.FIRST_RUN), 900)


    def test_intent_command_uses_isolated_codex_home_and_workspace(self) -> None:
        agent = CodexDockerAgent(workspace="/home/kali/workspace")
        project = type("ProjectStub", (), {"docker": "worker-container"})()

        command = agent._docker_command(project, DispatchMode.EXPLORE, runtime_key="intents/4")  # type: ignore[arg-type]

        self.assertIn("CODEX_HOME=/home/kali/.codex/intents/4", command)
        self.assertIn("TMUX_TMPDIR=/home/kali/.codex/intents/4/tmux", command)
        self.assertIn("/home/kali/workspace/intents/4", command)
        self.assertEqual(
            agent.intent_runtime_paths(4),
            ("/home/kali/.codex/intents/4", "/home/kali/workspace/intents/4"),
        )

    def test_preflight_runs_once_per_container(self) -> None:
        agent = CodexDockerAgent()
        project = type("ProjectStub", (), {"docker": "worker-container"})()
        calls: list[list[str]] = []

        def fake_run(command, **kwargs):
            calls.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        with patch("echo_agent.dispatcher.agents.subprocess.run", fake_run):
            agent._run_preflight(project, log=None)  # type: ignore[arg-type]
            agent._run_preflight(project, log=None)  # type: ignore[arg-type]

        self.assertEqual(len(calls), 1)

    def test_cancel_running_kills_registered_process_group(self) -> None:
        agent = CodexDockerAgent()
        process = subprocess.Popen(["sleep", "30"], start_new_session=True)
        agent._register_process(process)

        self.assertTrue(agent.cancel_running())
        process.wait(timeout=2)

        self.assertIsNotNone(process.returncode)
        self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
