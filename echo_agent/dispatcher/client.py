from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from echo_agent.models import Project


class ProjectAPIError(RuntimeError):
    pass


class ProjectClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def list_projects(self) -> list[Project]:
        return [Project.from_dict(item) for item in self._request("GET", "/projects")]

    def get_project(self, project_id: int) -> Project:
        return Project.from_dict(self._request("GET", f"/projects/{project_id}"))

    def delete_project(self, project_id: int) -> Project:
        return Project.from_dict(self._request("DELETE", f"/projects/{project_id}"))

    def create_project(
        self,
        name: str | None = None,
        project_id: int | None = None,
        docker: str | None = None,
        target_url: str | None = None,
        hint: str = "",
        objective: str = "",
    ) -> Project:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if project_id is not None:
            payload["id"] = project_id
        if docker is not None:
            payload["docker"] = docker
        if target_url is not None:
            payload["target_url"] = target_url
        if hint:
            payload["hint"] = hint
        if objective:
            payload["objective"] = objective
        return Project.from_dict(self._request("POST", "/projects", payload))

    def save_project(self, project: Project) -> Project:
        return Project.from_dict(self._request("PUT", f"/projects/{project.id}", project.to_dict()))

    def update_project_status(self, project_id: int, status: str) -> Project:
        return Project.from_dict(self._request("PATCH", f"/projects/{project_id}/status", {"status": status}))

    def activate_project(self, project_id: int) -> Project:
        return self.update_project_status(project_id, "active")

    def add_fact(self, project_id: int, description: str, fact_id: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"description": description}
        if fact_id is not None:
            payload["id"] = fact_id
        return self._request("POST", f"/projects/{project_id}/facts", payload)

    def add_intent(
        self,
        project_id: int,
        description: str = "",
        from_fact_id: int | None = None,
        to_fact_id: int | None = None,
        intent_id: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"description": description}
        if intent_id is not None:
            payload["id"] = intent_id
        if from_fact_id is not None:
            payload["from"] = from_fact_id
        if to_fact_id is not None:
            payload["to"] = to_fact_id
        return self._request("POST", f"/projects/{project_id}/intents", payload)

    def list_project_logs(self, project_id: int, limit: int = 200) -> list[dict[str, Any]]:
        return self._request("GET", f"/projects/{project_id}/logs?limit={limit}")

    def clear_project_logs(self, project_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/projects/{project_id}/logs")

    def add_project_log(
        self,
        project_id: int,
        message: str,
        level: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/projects/{project_id}/logs",
            {
                "level": level,
                "message": message,
                "metadata": metadata or {},
            },
        )

    def get_project_run_status(self, project_id: int) -> dict[str, Any]:
        return self._request("GET", f"/projects/{project_id}/run")

    def start_project_run(
        self,
        project_id: int,
        interval_seconds: float = 3.0,
        max_steps: int | None = None,
        max_workers: int = 4,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/projects/{project_id}/run",
            {
                "interval_seconds": interval_seconds,
                "max_steps": max_steps,
                "max_workers": max_workers,
            },
        )

    def stop_project_run(self, project_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/projects/{project_id}/run")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                data = json.loads(raw)
                detail = data.get("error", data.get("detail", raw))
            except json.JSONDecodeError:
                detail = raw
            raise ProjectAPIError(f"{method} {path} failed: {exc.code} {detail}") from exc
