from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from echo_agent.config import DispatcherSettings, ServerSettings


class ConfigFileTests(unittest.TestCase):
    def test_toml_config_populates_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "echo_agent.toml"
            config_path.write_text(
                """
[server]
host = "0.0.0.0"
port = 9000
database_path = "state/test.db"

[dispatcher]
server_url = "http://server:9000"
interval_seconds = 1.5
explore_max_workers = 7

[codex]
command = ["codex", "exec", "--json", "-"]
model = "main-model"
first_run_model = "first-model"
explore_model = "explore-model"
reason_model = "reason-model"
workspace = "/workspace"
timeout_seconds = 90
no_output_timeout_seconds = 12
first_run_no_output_timeout_seconds = 30
explore_no_output_timeout_seconds = 8
reason_no_output_timeout_seconds = 10
api_key = "file-key"
base_url = "https://proxy.example/v1"
provider_name = "test_proxy"
supports_websockets = true
request_max_retries = 3
stream_max_retries = 4
stream_idle_timeout_ms = 5000

[docker]
image = "worker:test"
context = "docker-context"
dockerfile = "docker-context/Dockerfile"
build_network = "bridge"

[resources]
root = "state/resources"
items = [
  { name = "tools", path = "tools", container_path = "/opt/tools", read_only = false },
  { name = "kb", url = "https://example.test/kb.txt", path = "kb/kb.txt", container_path = "/opt/kb/kb.txt", read_only = true },
]
""",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ECHO_AGENT_CONFIG": str(config_path)}, clear=True):
                server = ServerSettings.from_env()
                dispatcher = DispatcherSettings.from_env()

        config_dir = config_path.resolve().parent
        self.assertEqual(server.host, "0.0.0.0")
        self.assertEqual(server.port, 9000)
        self.assertEqual(server.database_path, config_dir / "state/test.db")
        self.assertEqual(dispatcher.codex_command, ("codex", "exec", "--json", "-"))
        self.assertEqual(dispatcher.codex_model, "main-model")
        self.assertEqual(dispatcher.codex_mode_models["first_run"], "first-model")
        self.assertEqual(dispatcher.codex_mode_models["explore"], "explore-model")
        self.assertEqual(dispatcher.codex_mode_models["reason"], "reason-model")
        self.assertEqual(dispatcher.codex_timeout_seconds, 90)
        self.assertEqual(
            dispatcher.codex_mode_timeout_seconds,
            {"first_run": 90, "explore": 90, "reason": 90},
        )
        self.assertEqual(dispatcher.codex_no_output_timeout_seconds, 12)
        self.assertEqual(
            dispatcher.codex_mode_no_output_timeout_seconds,
            {"first_run": 30, "explore": 8, "reason": 10},
        )
        self.assertEqual(dispatcher.codex_provider_name, "test_proxy")
        self.assertTrue(dispatcher.codex_provider_supports_websockets)
        self.assertEqual(dispatcher.codex_request_max_retries, 3)
        self.assertEqual(dispatcher.project_docker_image, "worker:test")
        self.assertEqual(dispatcher.project_docker_context, config_dir / "docker-context")
        self.assertEqual(dispatcher.project_dockerfile, config_dir / "docker-context/Dockerfile")
        self.assertEqual(dispatcher.project_resource_root, config_dir / "state/resources")
        self.assertEqual(len(dispatcher.project_resources), 2)
        self.assertEqual(dispatcher.project_resources[0].name, "tools")
        self.assertEqual(dispatcher.project_resources[0].container_path, "/opt/tools")
        self.assertFalse(dispatcher.project_resources[0].read_only)
        self.assertEqual(dispatcher.project_resources[1].url, "https://example.test/kb.txt")

    def test_environment_overrides_toml_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "echo_agent.toml"
            config_path.write_text('[codex]\nmodel = "file-model"\n', encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "ECHO_AGENT_CONFIG": str(config_path),
                    "ECHO_AGENT_CODEX_MODEL": "env-model",
                    "ECHO_AGENT_CODEX_COMMAND": "codex exec --json -",
                },
                clear=True,
            ):
                settings = DispatcherSettings.from_env()

        self.assertEqual(settings.codex_model, "env-model")
        self.assertEqual(settings.codex_command, ("codex", "exec", "--json", "-"))

    def test_relative_resource_root_is_resolved_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "echo_agent.toml"
            config_path.write_text("[resources]\nroot = \".echo_agent/resources\"\n", encoding="utf-8")
            with patch.dict(os.environ, {"ECHO_AGENT_CONFIG": str(config_path)}, clear=True):
                settings = DispatcherSettings.from_env()

        self.assertEqual(
            settings.project_resource_root,
            config_path.resolve().parent / ".echo_agent/resources",
        )

    def test_missing_user_toml_uses_example_defaults_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "echo_agent.toml"
            config_path.write_text("[codex]\n", encoding="utf-8")
            with patch.dict(os.environ, {"ECHO_AGENT_CONFIG": str(config_path)}, clear=True):
                settings = DispatcherSettings.from_env()

        self.assertEqual(settings.codex_model, "gpt-5.5")
        self.assertEqual(
            settings.codex_mode_models,
            {"first_run": "gpt-5.5", "explore": "gpt-5.5", "reason": "gpt-5.5"},
        )
        self.assertEqual(settings.codex_mode_timeout_seconds["first_run"], settings.codex_timeout_seconds)
        self.assertIsNone(settings.openai_api_key)
        self.assertEqual(settings.openai_base_url, "https://your-provider.example/v1")
        self.assertEqual(settings.codex_max_slots, 2)

    def test_mode_timeout_can_override_the_global_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "echo_agent.toml"
            config_path.write_text(
                "[codex]\ntimeout_seconds = 90\nfirst_run_timeout_seconds = 120\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "ECHO_AGENT_CONFIG": str(config_path),
                    "ECHO_AGENT_CODEX_REASON_TIMEOUT": "15",
                },
                clear=True,
            ):
                settings = DispatcherSettings.from_env()

        self.assertEqual(settings.codex_timeout_seconds, 90)
        self.assertEqual(settings.codex_first_run_timeout_seconds, 120)
        self.assertEqual(settings.codex_explore_timeout_seconds, 90)
        self.assertEqual(settings.codex_reason_timeout_seconds, 15)


if __name__ == "__main__":
    unittest.main()


class AuthTokenConfigTests(unittest.TestCase):
    def test_auth_token_is_optional_and_shared_with_dispatcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "echo_agent.toml"
            config_path.write_text('[server]\nauth_token = "abc"\n', encoding="utf-8")
            with patch.dict(os.environ, {"ECHO_AGENT_CONFIG": str(config_path)}, clear=True):
                server = ServerSettings.from_env()
                dispatcher = DispatcherSettings.from_env()
        self.assertEqual(server.auth_token, "abc")
        self.assertEqual(dispatcher.server_auth_token, "abc")

    def test_blank_auth_token_stays_unset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "echo_agent.toml"
            config_path.write_text('[server]\nauth_token = ""\n', encoding="utf-8")
            with patch.dict(os.environ, {"ECHO_AGENT_CONFIG": str(config_path)}, clear=True):
                server = ServerSettings.from_env()
        self.assertIsNone(server.auth_token)

    def test_example_reason_idle_timeout_is_long_enough_to_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "echo_agent.toml"
            config_path.write_text("[codex]\n", encoding="utf-8")
            with patch.dict(os.environ, {"ECHO_AGENT_CONFIG": str(config_path)}, clear=True):
                settings = DispatcherSettings.from_env()
        self.assertEqual(settings.codex_reason_no_output_timeout_seconds, 900)

