"""HTTP REST API — FastAPI wrapper around the MambaMemory engine.

Provides a universal HTTP interface for any client that can't use MCP.

Authentication:
  - Set MAMBA_MEMORY_API_KEY env var to enable Bearer token auth
  - If not set, API is open (for local / development use)

Usage:
    mamba-memory serve --http --port 8420
    # or
    MAMBA_MEMORY_API_KEY=your-secret uvicorn mamba_memory.server.http.app:create_app --factory
"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from mamba_memory.core.engine import MambaMemoryEngine
from mamba_memory.core.types import EngineConfig, IngestResult, MemoryStatus, RecallResult

# Store engine in module-level variable for lifespan management
_engine: MambaMemoryEngine | None = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_api_key() -> str | None:
    """Get the configured API key, or None if auth is disabled."""
    return os.environ.get("MAMBA_MEMORY_API_KEY")


async def _verify_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Verify Bearer token if MAMBA_MEMORY_API_KEY is set."""
    required_key = _get_api_key()
    if required_key is None:
        return  # Auth disabled

    # Allow /health without auth
    if request.url.path == "/health":
        return

    if credentials is None:
        raise HTTPException(
            401,
            detail="Missing Authorization header. Use: Authorization: Bearer <api-key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(credentials.credentials, required_key):
        raise HTTPException(
            403,
            detail="Invalid API key",
        )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    content: str
    role: str = "user"
    tags: list[str] = Field(default_factory=list)
    force: bool = False
    summary: str | None = None
    entities: list[str] | None = None


class RecallRequest(BaseModel):
    query: str
    limit: int = 5
    layers: list[str] | None = None
    min_score: float = 0.0


class ForgetRequest(BaseModel):
    query: str


class CompactRequest(BaseModel):
    layer: str = "all"


class ForgetResponse(BaseModel):
    forgotten: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """Create a FastAPI app wired to MambaMemoryEngine."""
    _config = config or EngineConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _engine
        _engine = MambaMemoryEngine(_config)
        await _engine.start()
        yield
        await _engine.shutdown()
        _engine = None

    auth_enabled = _get_api_key() is not None

    app = FastAPI(
        title="MambaMemory",
        description="AI Agent Cognitive Memory Engine — SSM-inspired selective memory",
        version="0.1.0",
        lifespan=lifespan,
    )

    def _get_engine() -> MambaMemoryEngine:
        if _engine is None:
            raise HTTPException(503, "Engine not started")
        return _engine

    # -- Endpoints -----------------------------------------------------------

    @app.post("/ingest", response_model=IngestResult, dependencies=[Depends(_verify_auth)])
    async def ingest(req: IngestRequest) -> IngestResult:
        engine = _get_engine()
        return await engine.ingest(
            req.content,
            role=req.role,
            tags=req.tags if req.tags else None,
            force=req.force,
            pre_summary=req.summary,
            pre_entities=req.entities,
        )

    @app.post("/recall", response_model=RecallResult, dependencies=[Depends(_verify_auth)])
    async def recall(req: RecallRequest) -> RecallResult:
        engine = _get_engine()
        return await engine.recall(
            req.query,
            limit=req.limit,
            layers=req.layers,
            min_score=req.min_score,
        )

    @app.post("/forget", response_model=ForgetResponse, dependencies=[Depends(_verify_auth)])
    async def forget(req: ForgetRequest) -> ForgetResponse:
        engine = _get_engine()
        count = await engine.forget(req.query)
        return ForgetResponse(forgotten=count)

    @app.post("/compact", dependencies=[Depends(_verify_auth)])
    async def compact(req: CompactRequest) -> dict:
        engine = _get_engine()
        return await engine.compact(req.layer)

    @app.get("/status", response_model=MemoryStatus, dependencies=[Depends(_verify_auth)])
    async def status() -> MemoryStatus:
        engine = _get_engine()
        return engine.status()

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "engine": _engine is not None,
            "auth": auth_enabled,
        }

    # -- Dashboard UI --------------------------------------------------------

    @app.get("/ui", include_in_schema=False)
    async def dashboard():
        from fastapi.responses import HTMLResponse
        from mamba_memory.server.http.dashboard import DASHBOARD_HTML
        return HTMLResponse(DASHBOARD_HTML)

    return app
