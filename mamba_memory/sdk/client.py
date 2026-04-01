"""Python SDK client for MambaMemory.

Two usage modes:
1. Direct (in-process): import and use MambaMemoryEngine directly
2. HTTP client: connect to a running MambaMemory HTTP server

Example (direct)::

    from mamba_memory.sdk import MambaMemory

    async with MambaMemory() as mem:
        await mem.ingest("用户偏好Docker部署")
        result = await mem.recall("部署方式")
        for m in result.memories:
            print(f"[{m.layer}] {m.content}")

Example (HTTP)::

    from mamba_memory.sdk import MambaMemoryHTTP

    client = MambaMemoryHTTP("http://localhost:8420")
    result = await client.recall("部署方式")
"""

from __future__ import annotations

from typing import Any

from mamba_memory.core.engine import MambaMemoryEngine
from mamba_memory.core.types import (
    EngineConfig,
    IngestResult,
    MemoryStatus,
    RecallResult,
)


class MambaMemory:
    """Direct (in-process) MambaMemory client.

    Wraps the engine with a simpler async context manager interface.
    """

    def __init__(self, config: EngineConfig | None = None, db_path: str | None = None) -> None:
        cfg = config or EngineConfig()
        if db_path:
            cfg.l3.db_path = db_path
        self._engine = MambaMemoryEngine(cfg)

    async def __aenter__(self) -> MambaMemory:
        await self._engine.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._engine.shutdown()

    async def start(self, session_id: str = "") -> None:
        await self._engine.start(session_id)

    async def shutdown(self) -> None:
        await self._engine.shutdown()

    async def ingest(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        force: bool = False,
    ) -> IngestResult:
        return await self._engine.ingest(content, tags=tags, force=force)

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        layers: list[str] | None = None,
    ) -> RecallResult:
        return await self._engine.recall(query, limit=limit, layers=layers)

    async def forget(self, query: str) -> int:
        return await self._engine.forget(query)

    def status(self) -> MemoryStatus:
        return self._engine.status()

    async def compact(self, layer: str = "all") -> dict:
        return await self._engine.compact(layer)


class MambaMemoryHTTP:
    """HTTP client for a remote MambaMemory server."""

    def __init__(self, base_url: str = "http://localhost:8420") -> None:
        self._base = base_url.rstrip("/")

    async def ingest(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        force: bool = False,
    ) -> dict:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/ingest",
                json={"content": content, "tags": tags or [], "force": force},
            )
            resp.raise_for_status()
            return resp.json()

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        layers: list[str] | None = None,
    ) -> dict:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/recall",
                json={"query": query, "limit": limit, "layers": layers},
            )
            resp.raise_for_status()
            return resp.json()

    async def forget(self, query: str) -> dict:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._base}/forget", json={"query": query})
            resp.raise_for_status()
            return resp.json()

    async def status(self) -> dict:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self._base}/status")
            resp.raise_for_status()
            return resp.json()
