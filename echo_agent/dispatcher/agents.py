from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Mapping, Sequence

from echo_agent.dispatcher.parsing import parse_reason_result, parse_worker_result
from echo_agent.dispatcher.prompts import PromptBuilder
from echo_agent.dispatcher.types import (
    AgentAdapter,
    AgentLogCallback,
    AgentRunResult,
    DispatchMode,
    ReasonResult,
    WorkerResult,
)
from echo_agent.models import Intent, Project


class CodexDockerAgent:
    def __init__(
        self,
        codex_command: Sequence[str] | None = None,
        timeout_seconds: int = 1800,
        workspace: str = "/home/kali/workspace",
        model: str | None = None,
        mode_models: Mapping[DispatchMode | str, str] | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        provider_name: str = "echo_agent_proxy",
        provider_supports_websockets: bool = False,
        request_max_retries: int = 2,
        stream_max_retries: int = 2,
        stream_idle_timeout_ms: int = 60_000,
        no_output_timeout_seconds: int = 900,
        mode_no_output_timeout_seconds: Mapping[DispatchMode | str, int] | None = None,
    ) -> None:
        self.codex_command = tuple(
            codex_command
            or (
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "-",
            )
        )
        self.timeout_seconds = timeout_seconds
        self.workspace = workspace
        self.model = model
        self.mode_models = {self._mode_key(key): value for key, value in (mode_models or {}).items()}
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.provider_name = provider_name
        self.provider_supports_websockets = provider_supports_websockets
        self.request_max_retries = request_max_retries
        self.stream_max_retries = stream_max_retries
        self.stream_idle_timeout_ms = stream_idle_timeout_ms
        self.no_output_timeout_seconds = no_output_timeout_seconds
        self.mode_no_output_timeout_seconds = {
            self._mode_key(key): int(value)
            for key, value in (mode_no_output_timeout_seconds or {}).items()
        }

    def run(
        self,
        project: Project,
        prompt: str,
        mode: DispatchMode,
        log: AgentLogCallback | None = None,
    ) -> AgentRunResult:
        if not project.docker:
            raise RuntimeError("project.docker is required to run codex in explore/reason mode")

        command = self._docker_command(project, mode)
        env = self._subprocess_env()
        self._log(
            log,
            f"codex agent process starting in {mode.value} mode",
            metadata={
                "docker": project.docker,
                "workspace": self.workspace,
                "command": self._safe_command(command),
                "model": self.mode_models.get(mode.value, self.model),
                "timeout_seconds": self.timeout_seconds,
                "no_output_timeout_seconds": self._no_output_timeout_for_mode(mode),
                "prompt_chars": len(prompt),
            },
        )
        self._run_preflight(project, log=log)
        output = self._run_process(command, prompt, mode, env=env, log=log)
        self._log(log, f"codex agent process finished in {mode.value} mode", metadata={"output_chars": len(output)})
        return AgentRunResult(output=output)

    def _docker_command(self, project: Project, mode: DispatchMode) -> list[str]:
        command = ["docker", "exec", "-i"]
        if self.openai_api_key:
            command.extend(["-e", f"OPENAI_API_KEY={self.openai_api_key}"])
        if self.openai_base_url:
            command.extend(["-e", f"OPENAI_BASE_URL={self.openai_base_url}"])
        command.extend(["-w", self.workspace, project.docker, *self._codex_command_for_mode(mode)])
        return command

    def _preflight_command(self, project: Project) -> list[str]:
        command = ["docker", "exec", "-i"]
        if self.openai_api_key:
            command.extend(["-e", f"OPENAI_API_KEY={self.openai_api_key}"])
        if self.openai_base_url:
            command.extend(["-e", f"OPENAI_BASE_URL={self.openai_base_url}"])
        script = (
            'echo "workspace=$(pwd)"; '
            'echo "codex_path=$(command -v codex || true)"; '
            'codex --version; '
            'if [ -n "$OPENAI_API_KEY" ]; then echo "openai_api_key=present"; else echo "openai_api_key=missing"; fi; '
            'if [ -n "$OPENAI_BASE_URL" ]; then echo "openai_base_url=$OPENAI_BASE_URL"; else echo "openai_base_url=missing"; fi'
        )
        command.extend(["-w", self.workspace, project.docker, "sh", "-lc", script])
        return command

    def _run_preflight(self, project: Project, log: AgentLogCallback | None) -> None:
        command = self._preflight_command(project)
        self._log(log, "codex preflight started", metadata={"command": self._safe_command(command)})
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=self._subprocess_env(),
        )
        output = "\n".join(
            line for line in [completed.stdout.strip(), completed.stderr.strip()] if line
        )
        if output:
            for line in output.splitlines():
                self._log(log, f"codex preflight: {self._truncate(line)}")
        if completed.returncode != 0:
            raise RuntimeError(f"codex preflight failed with exit code {completed.returncode}: {output}")
        self._log(log, "codex preflight finished")

    def _subprocess_env(self) -> dict[str, str] | None:
        if not self.openai_api_key and not self.openai_base_url:
            return None
        env = os.environ.copy()
        if self.openai_api_key:
            env["OPENAI_API_KEY"] = self.openai_api_key
        if self.openai_base_url:
            env["OPENAI_BASE_URL"] = self.openai_base_url
        return env

    def _run_process(
        self,
        command: Sequence[str],
        prompt: str,
        mode: DispatchMode,
        env: dict[str, str] | None,
        log: AgentLogCallback | None,
    ) -> str:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(prompt)
        process.stdin.close()
        self._log(log, "codex prompt written to stdin", metadata={"prompt_chars": len(prompt)})

        output_lines: list[str] = []
        stderr_lines: list[str] = []
        recent_output: deque[str] = deque(maxlen=40)
        stdout_log_lines: list[str] = []
        stderr_log_lines: list[str] = []
        output_lock = threading.Lock()
        total_activity_count = 0

        def pop_log_block(lines: list[str]) -> str | None:
            if not lines:
                return None
            block = "\n".join(lines)
            lines.clear()
            return block

        def should_flush_log_block(lines: list[str]) -> bool:
            return len(lines) >= 20 or sum(len(line) for line in lines) >= 4000

        def flush_log_blocks() -> None:
            with output_lock:
                stdout_block = pop_log_block(stdout_log_lines)
                stderr_block = pop_log_block(stderr_log_lines)
            if stdout_block:
                self._log(log, f"codex result:\n{self._truncate(stdout_block, limit=5000)}")
            if stderr_block:
                self._log(log, f"codex output:\n{self._truncate(stderr_block, limit=5000)}")

        def read_stdout() -> None:
            nonlocal total_activity_count
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                flush_needed = False
                with output_lock:
                    total_activity_count += 1
                    output_lines.append(line)
                    recent_output.append(f"stdout: {line}")
                    stdout_log_lines.append(line)
                    flush_needed = should_flush_log_block(stdout_log_lines)
                if flush_needed:
                    flush_log_blocks()

        def read_stderr() -> None:
            nonlocal total_activity_count
            for raw_line in process.stderr:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                flush_needed = False
                with output_lock:
                    total_activity_count += 1
                    stderr_lines.append(line)
                    recent_output.append(f"stderr: {line}")
                    stderr_log_lines.append(line)
                    flush_needed = should_flush_log_block(stderr_log_lines)
                if flush_needed:
                    flush_log_blocks()

        stdout_reader = threading.Thread(target=read_stdout, daemon=True)
        stderr_reader = threading.Thread(target=read_stderr, daemon=True)
        stdout_reader.start()
        stderr_reader.start()
        started_at = time.monotonic()
        last_activity_at = started_at
        last_heartbeat_at = started_at
        last_log_flush_at = started_at
        last_activity_count = 0
        activity_count = 0
        no_output_timeout_seconds = self._no_output_timeout_for_mode(mode)
        while process.poll() is None:
            elapsed = time.monotonic() - started_at
            with output_lock:
                current_activity_count = total_activity_count
            if current_activity_count != activity_count:
                activity_count = current_activity_count
                last_activity_at = time.monotonic()
            if elapsed > self.timeout_seconds:
                self._kill_process(process)
                process.wait()
                stdout_reader.join(timeout=1)
                stderr_reader.join(timeout=1)
                flush_log_blocks()
                process.stdout.close()
                process.stderr.close()
                with output_lock:
                    output_count = len(output_lines)
                self._log(
                    log,
                    f"codex agent timed out in {mode.value} mode",
                    "error",
                    {"timeout_seconds": self.timeout_seconds, "output_lines": output_count},
                )
                raise RuntimeError(f"codex agent timed out in {mode.value} mode after {self.timeout_seconds}s")
            idle_seconds = time.monotonic() - last_activity_at
            if no_output_timeout_seconds > 0 and idle_seconds > no_output_timeout_seconds:
                self._kill_process(process)
                process.wait()
                stdout_reader.join(timeout=1)
                stderr_reader.join(timeout=1)
                flush_log_blocks()
                process.stdout.close()
                process.stderr.close()
                with output_lock:
                    output_count = len(output_lines)
                    detail = "\n".join(recent_output).strip()
                self._log(
                    log,
                    f"codex agent no-output timed out in {mode.value} mode",
                    "error",
                    {
                        "no_output_timeout_seconds": no_output_timeout_seconds,
                        "elapsed_seconds": int(elapsed),
                        "idle_seconds": int(idle_seconds),
                        "output_lines": output_count,
                        "recent_output": detail,
                    },
                )
                raise RuntimeError(
                    f"codex agent no-output timed out in {mode.value} mode after "
                    f"{no_output_timeout_seconds}s without stdout/stderr"
                )
            if time.monotonic() - last_heartbeat_at >= 60:
                with output_lock:
                    output_count = len(output_lines)
                    heartbeat_activity_count = total_activity_count
                idle_seconds = time.monotonic() - last_activity_at
                remaining_idle_seconds = (
                    max(0, no_output_timeout_seconds - int(idle_seconds))
                    if no_output_timeout_seconds > 0
                    else None
                )
                if heartbeat_activity_count == last_activity_count:
                    self._log(
                        log,
                        "codex agent still running with no new output",
                        metadata={
                            "elapsed_seconds": int(elapsed),
                            "idle_seconds": int(idle_seconds),
                            "remaining_idle_seconds": remaining_idle_seconds,
                            "output_lines": output_count,
                        },
                    )
                else:
                    self._log(
                        log,
                        "codex agent still running",
                        metadata={"elapsed_seconds": int(elapsed), "output_lines": output_count},
                    )
                last_activity_count = heartbeat_activity_count
                last_heartbeat_at = time.monotonic()
            if time.monotonic() - last_log_flush_at >= 2:
                flush_log_blocks()
                last_log_flush_at = time.monotonic()
            time.sleep(1)
        return_code = process.returncode
        stdout_reader.join(timeout=1)
        stderr_reader.join(timeout=1)
        flush_log_blocks()
        process.stdout.close()
        process.stderr.close()

        with output_lock:
            output = "\n".join(output_lines).strip()
            stderr_output = "\n".join(stderr_lines).strip()
            detail = "\n".join(recent_output).strip()
        if return_code != 0:
            salvaged_output = self._extract_mode_json_output(
                "\n".join(item for item in [output, stderr_output] if item),
                mode,
            )
            if salvaged_output is not None:
                self._log(
                    log,
                    f"codex agent exited with code {return_code} in {mode.value} mode, salvaged JSON output",
                    "warning",
                    {"recent_output": detail},
                )
                return salvaged_output
            self._log(
                log,
                f"codex agent failed in {mode.value} mode with exit code {return_code}",
                "error",
                {"recent_output": detail},
            )
            raise RuntimeError(f"codex agent failed in {mode.value} mode with exit code {return_code}: {detail}")
        return output

    @classmethod
    def _extract_mode_json_output(cls, text: str, mode: DispatchMode) -> str | None:
        decoder = json.JSONDecoder()
        best: str | None = None
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and cls._looks_like_mode_output(value, mode):
                best = text[index : index + end]
        return best

    @staticmethod
    def _looks_like_mode_output(value: dict[str, object], mode: DispatchMode) -> bool:
        if mode == DispatchMode.REASON:
            return "completed" in value and (
                "next_intents" in value or "next_intent_descriptions" in value
            )
        if mode in {DispatchMode.FIRST_RUN, DispatchMode.EXPLORE}:
            return "answer" in value or "fact_description" in value
        return False

    def _no_output_timeout_for_mode(self, mode: DispatchMode) -> int:
        return self.mode_no_output_timeout_seconds.get(mode.value, self.no_output_timeout_seconds)

    @staticmethod
    def _kill_process(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            process.kill()

    def _codex_command_for_mode(self, mode: DispatchMode) -> tuple[str, ...]:
        model = self.mode_models.get(mode.value, self.model)
        command = list(self.codex_command)

        extra_args: list[str] = []
        if self.openai_base_url and not self._command_already_sets_provider(command):
            # OPENAI_BASE_URL is not a reliable Codex CLI provider override. In
            # particular, recent Codex releases may still use the built-in
            # api.openai.com WebSocket endpoint. Configure an explicit provider
            # so proxy-backed runs use the requested Responses API endpoint.
            provider = self.provider_name
            extra_args.extend(
                [
                    "-c",
                    f"model_provider={json.dumps(provider)}",
                    "-c",
                    f"model_providers.{provider}.name={json.dumps('EchoAgent Proxy')}",
                    "-c",
                    f"model_providers.{provider}.base_url={json.dumps(self.openai_base_url)}",
                    "-c",
                    f"model_providers.{provider}.env_key={json.dumps('OPENAI_API_KEY')}",
                    "-c",
                    f"model_providers.{provider}.wire_api={json.dumps('responses')}",
                    "-c",
                    f"model_providers.{provider}.supports_websockets={str(self.provider_supports_websockets).lower()}",
                    "-c",
                    f"model_providers.{provider}.request_max_retries={self.request_max_retries}",
                    "-c",
                    f"model_providers.{provider}.stream_max_retries={self.stream_max_retries}",
                    "-c",
                    f"model_providers.{provider}.stream_idle_timeout_ms={self.stream_idle_timeout_ms}",
                ]
            )
        if model and not self._command_already_sets_model():
            extra_args.extend(["--model", model])

        if extra_args:
            if command and command[-1] == "-":
                command[-1:-1] = extra_args
            else:
                command.extend(extra_args)
        return tuple(command)

    def _command_already_sets_model(self) -> bool:
        return "--model" in self.codex_command or "-m" in self.codex_command

    @staticmethod
    def _command_already_sets_provider(command: Sequence[str]) -> bool:
        return any(
            item.startswith("model_provider=") or item.startswith("model_providers.")
            for item in command
        )

    @staticmethod
    def _mode_key(mode: DispatchMode | str) -> str:
        if isinstance(mode, DispatchMode):
            return mode.value
        return mode

    @staticmethod
    def _safe_command(command: Sequence[str]) -> list[str]:
        return ["<env>" if item.startswith("OPENAI_") else item for item in command]

    @staticmethod
    def _truncate(value: str, limit: int = 1000) -> str:
        if len(value) <= limit:
            return value
        return f"{value[:limit - 3]}..."

    @staticmethod
    def _log(
        log: AgentLogCallback | None,
        message: str,
        level: str = "info",
        metadata: dict[str, object] | None = None,
    ) -> None:
        if log is not None:
            log(message, level, metadata or {})


class CodexWorker:
    def __init__(self, agent: AgentAdapter, prompt_builder: PromptBuilder | None = None) -> None:
        self.agent = agent
        self.prompt_builder = prompt_builder or PromptBuilder()

    def execute_intent(
        self,
        project: Project,
        intent: Intent,
        mode: DispatchMode = DispatchMode.EXPLORE,
        log: AgentLogCallback | None = None,
    ) -> WorkerResult:
        if mode == DispatchMode.FIRST_RUN:
            prompt = self.prompt_builder.build_first_run_prompt(project, intent)
        else:
            prompt = self.prompt_builder.build_explore_prompt(project, intent)
        result = self.agent.run(project, prompt, mode, log=log)  # type: ignore[call-arg]
        return parse_worker_result(result.output)


class CodexReasoner:
    def __init__(self, agent: AgentAdapter, prompt_builder: PromptBuilder | None = None) -> None:
        self.agent = agent
        self.prompt_builder = prompt_builder or PromptBuilder()

    def plan(self, project: Project, log: AgentLogCallback | None = None) -> ReasonResult:
        prompt = self.prompt_builder.build_reason_prompt(project)
        result = self.agent.run(project, prompt, DispatchMode.REASON, log=log)  # type: ignore[call-arg]
        return parse_reason_result(result.output)


class NoopWorker:
    def execute_intent(self, project: Project, intent: Intent) -> WorkerResult:
        return WorkerResult(fact_description="intent executed without adapter output")


class NoopReasoner:
    def plan(self, project: Project) -> ReasonResult:
        return ReasonResult(completed=False, next_intent_descriptions=[])
