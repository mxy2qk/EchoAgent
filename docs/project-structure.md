# Project Structure

EchoAgent keeps runtime code under the `echo_agent` package and separates operational assets, documentation, and tests at the repository root.

## Runtime Package

```text
echo_agent/
├── config.py
├── models.py
├── storage.py
├── dispatcher/
└── server/
```

- `config.py`: TOML/env-backed settings for the server, dispatcher, Codex, Docker, and shared resources.
- `models.py`: graph domain objects, statuses, and state transitions.
- `storage.py`: SQLite persistence for projects, facts, intents, and logs.
- `dispatcher/`: project execution loop, Codex Docker adapter, prompts, parsing, and HTTP client.
- `server/`: FastAPI routes, Pydantic schemas, background dispatcher runner, and static web UI.

## Operational Assets

```text
container/
├── Dockerfile
└── AGENTS.md
```

This directory defines the Kali/Codex execution environment used by project containers.
Large tools, PoCs, templates, and knowledge bases are not baked into the image; they are prepared under `.echo_agent/resources` and mounted into containers according to `echo_agent.toml`.

## Documentation

```text
docs/
├── design.md
├── image.png
└── project-structure.md
```

`design.md` is the original design note. `project-structure.md` is the current engineering layout reference.

## Tests

```text
tests/
├── test_agents.py
├── test_config.py
├── test_dispatcher.py
├── test_dispatcher_runner.py
├── test_docker_runtime.py
└── test_parsing.py
```

Run tests with:

```bash
python -m unittest
node --check echo_agent/server/static/app.js
```
