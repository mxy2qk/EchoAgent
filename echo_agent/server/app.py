from __future__ import annotations

import secrets
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

import uvicorn

from echo_agent.models import Project
from echo_agent.server.docker_runtime import DockerProjectRuntime, DockerRuntimeError
from echo_agent.server.schemas import (
    FactCreateRequest,
    IntentCreateRequest,
    ProjectCreateRequest,
    ProjectLogCreateRequest,
    ProjectRunRequest,
    ProjectSaveRequest,
    ProjectStatusRequest,
)
from echo_agent.server.dispatcher_runner import DispatcherRunner
from echo_agent.storage import ProjectNotFoundError, SQLiteProjectStore


class MissingServerDependencyError(RuntimeError):
    pass


def create_app(store: SQLiteProjectStore, auth_token: str | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:
        raise MissingServerDependencyError(
            "FastAPI is required to run the server. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    app = FastAPI(title="EchoAgent Server")
    app.state.store = store
    app.state.project_runtime = DockerProjectRuntime.from_settings()
    app.state.project_runtime.prepare_shared_resources()
    app.state.dispatcher_runner = DispatcherRunner(store, runtime=app.state.project_runtime)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    if auth_token:
        # 只保护数据/控制接口；首页与静态资源保持开放，前端自行携带 token。
        # 目的：worker 容器内的 agent 即使能经宿主机代理打到本机端口，也拿不到项目数据。
        @app.middleware("http")
        async def _require_auth_token(request, call_next):  # type: ignore[no-untyped-def]
            path = request.url.path
            if path in {"/", "/health"} or path.startswith("/static"):
                return await call_next(request)
            provided = request.headers.get("x-echo-agent-token", "")
            if not secrets.compare_digest(provided, auth_token):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            return await call_next(request)

    def apply_project_status_transition(project: Project, old_project: Project) -> None:
        if old_project.status == project.status:
            return
        app.state.store.add_project_log(
            project.id,
            f"project status transition: {old_project.status.value} -> {project.status.value}",
            metadata={"old_status": old_project.status.value, "new_status": project.status.value},
        )
        app.state.store.add_project_log(
            project.id,
            f"project container lifecycle started for {project.status.value}",
            metadata={"docker": project.docker, "status": project.status.value},
        )
        try:
            app.state.project_runtime.apply_status_transition(
                project,
                old_project.status,
                log=lambda message, level="info", metadata=None: app.state.store.add_project_log(
                    project.id,
                    message,
                    level=level,
                    metadata=metadata or {},
                ),
            )
        except DockerRuntimeError as exc:
            app.state.store.add_project_log(
                project.id,
                f"project container lifecycle failed for {project.status.value}: {exc}",
                level="error",
                metadata={"docker": project.docker, "status": project.status.value},
            )
            raise
        app.state.store.add_project_log(
            project.id,
            f"project container lifecycle applied for {project.status.value}",
            metadata={"docker": project.docker, "status": project.status.value},
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/projects")
    def list_projects() -> list[dict[str, Any]]:
        return [project.to_dict() for project in app.state.store.list_projects()]

    @app.post("/projects", status_code=HTTPStatus.CREATED)
    def create_project(data: ProjectCreateRequest) -> dict[str, Any]:
        try:
            project = app.state.store.create_project(
                name=data.name,
                project_id=data.id,
                docker=data.docker,
                target_url=data.target_url,
                hint=data.hint,
                objective=data.objective,
            )
            return project.to_dict()
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    @app.get("/projects/{project_id}")
    def get_project(project_id: int) -> dict[str, Any]:
        try:
            return app.state.store.get_project(project_id).to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc

    @app.delete("/projects/{project_id}")
    def delete_project(project_id: int) -> dict[str, Any]:
        try:
            run_state = app.state.dispatcher_runner.status(project_id)
            if run_state.running:
                raise HTTPException(
                    status_code=HTTPStatus.CONFLICT,
                    detail="project dispatcher is running; stop it before deleting the project",
                )
            project = app.state.store.get_project(project_id)
            app.state.project_runtime.remove_project_container(project)
            return app.state.store.delete_project(project_id).to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc
        except DockerRuntimeError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    @app.put("/projects/{project_id}")
    def save_project(project_id: int, data: ProjectSaveRequest) -> dict[str, Any]:
        try:
            old_project = app.state.store.get_project(project_id)
            payload = data.model_dump(mode="json")
            payload["id"] = project_id
            project = Project.from_dict(payload)
            apply_project_status_transition(project, old_project)
            return app.state.store.save_project(project).to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc
        except DockerRuntimeError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    @app.patch("/projects/{project_id}/status")
    def update_project_status(project_id: int, data: ProjectStatusRequest) -> dict[str, Any]:
        try:
            project = app.state.store.get_project(project_id)
            old_project = Project.from_dict(project.to_dict())
            project.transition_to(data.status)
            apply_project_status_transition(project, old_project)
            return app.state.store.save_project(project).to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc
        except DockerRuntimeError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/projects/{project_id}/facts", status_code=HTTPStatus.CREATED)
    def add_fact(project_id: int, data: FactCreateRequest) -> dict[str, Any]:
        try:
            project = app.state.store.get_project(project_id)
            fact = project.add_fact(data.description, fact_id=data.id)
            app.state.store.save_project(project)
            return fact.to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/projects/{project_id}/intents", status_code=HTTPStatus.CREATED)
    def add_intent(project_id: int, data: IntentCreateRequest) -> dict[str, Any]:
        try:
            project = app.state.store.get_project(project_id)
            intent = project.add_intent(
                description=data.description,
                from_fact_id=data.from_fact_id,
                to_fact_id=data.to_fact_id,
                intent_id=data.id,
            )
            app.state.store.save_project(project)
            return intent.to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    @app.get("/projects/{project_id}/logs")
    def list_project_logs(project_id: int, limit: int = 200) -> list[dict[str, Any]]:
        try:
            return app.state.store.list_project_logs(project_id, limit=limit)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc

    @app.post("/projects/{project_id}/logs", status_code=HTTPStatus.CREATED)
    def add_project_log(project_id: int, data: ProjectLogCreateRequest) -> dict[str, Any]:
        try:
            return app.state.store.add_project_log(
                project_id,
                message=data.message,
                level=data.level,
                metadata=data.metadata,
            )
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    @app.delete("/projects/{project_id}/logs")
    def clear_project_logs(project_id: int) -> dict[str, Any]:
        try:
            return app.state.store.clear_project_logs(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc

    @app.get("/projects/{project_id}/run")
    def get_project_run_status(project_id: int) -> dict[str, Any]:
        try:
            app.state.store.get_project(project_id)
            return app.state.dispatcher_runner.status(project_id).to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc

    @app.post("/projects/{project_id}/run", status_code=HTTPStatus.ACCEPTED)
    def start_project_run(project_id: int, data: ProjectRunRequest) -> dict[str, Any]:
        try:
            return app.state.dispatcher_runner.start(
                project_id,
                interval_seconds=data.interval_seconds,
                max_steps=data.max_steps,
                max_workers=data.max_workers,
            ).to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    @app.delete("/projects/{project_id}/run")
    def stop_project_run(project_id: int) -> dict[str, Any]:
        try:
            app.state.store.get_project(project_id)
            return app.state.dispatcher_runner.stop(project_id).to_dict()
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="project not found") from exc

    return app


class UvicornServerHandle:
    def __init__(self, host: str, port: int, app: Any) -> None:
        self._config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(self._config)

    @property
    def server_port(self) -> int:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            port = self._bound_port()
            if port is not None:
                return port
            time.sleep(0.01)
        raise RuntimeError("server did not start within 5 seconds")

    def serve_forever(self) -> None:
        self._server.run()

    def shutdown(self) -> None:
        self._server.should_exit = True

    def server_close(self) -> None:
        return

    def _bound_port(self) -> int | None:
        servers = getattr(self._server, "servers", None)
        if not servers:
            return None
        sockets = getattr(servers[0], "sockets", None)
        if not sockets:
            return None
        return int(sockets[0].getsockname()[1])


def create_server(host: str, port: int, database_path: str, auth_token: str | None = None) -> UvicornServerHandle:
    return UvicornServerHandle(host, port, create_app(SQLiteProjectStore(database_path), auth_token=auth_token))
