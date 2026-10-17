from __future__ import annotations

from echo_agent.config import ServerSettings
from echo_agent.server.app import create_app
from echo_agent.storage import SQLiteProjectStore


settings = ServerSettings.from_env()
app = create_app(SQLiteProjectStore(settings.database_path), auth_token=settings.auth_token)
