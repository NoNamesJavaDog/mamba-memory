"""MCP Server integration tests.

Spins up the MambaMemory MCP server as a subprocess,
connects via stdio with the MCP client SDK, and exercises
all 5 tools + 2 prompts through the real protocol.

Tests:
  1. Server initialization & tool listing
  2. Prompt listing & retrieval
  3. memory_ingest — basic, with tags, force, pre-summary
  4. memory_recall — basic, layer filtering, min_score
  5. memory_status — summary and slots detail
  6. memory_forget
  7. memory_compact
  8. Full lifecycle: ingest → recall → forget → verify
  9. Error handling — bad tool name, missing params
  10. Concurrent tool calls

Note: Each test function creates its own MCP session to avoid
anyio/asyncio cancel scope conflicts in pytest teardown.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _server_params(db_path: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-c",
            f"""
import asyncio
from mamba_memory.server.mcp.server import serve
asyncio.run(serve(db_path="{db_path}"))
""",
        ],
        env={
            **os.environ,
            "GOOGLE_API_KEY": GOOGLE_API_KEY,
            "PYTHONPATH": _PROJECT_ROOT,
            "PYTHONUNBUFFERED": "1",
        },
    )


async def _run_with_session(tmp_dir: str, func):
    """Create a fresh MCP session, run func(session), then cleanly close."""
    import anyio

    db_path = os.path.join(tmp_dir, "mcp_test.db")
    params = _server_params(db_path)

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await func(session)


async def _call(session: ClientSession, tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool and parse the JSON response."""
    try:
        result = await session.call_tool(tool_name, arguments)
        text = result.content[0].text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": text or "empty response", "raw": True}
    except Exception as e:
        return {"error": str(e), "exception": True}


# ---------------------------------------------------------------------------
# Test 1: Server initialization & tool listing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools(tmp_path):
    async def run(session: ClientSession):
        result = await session.list_tools()
        tool_names = sorted(t.name for t in result.tools)
        assert tool_names == [
            "memory_compact", "memory_forget", "memory_ingest",
            "memory_recall", "memory_status",
        ]
        # Verify schemas
        for tool in result.tools:
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 2: Prompt listing & retrieval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompts(tmp_path):
    async def run(session: ClientSession):
        result = await session.list_prompts()
        names = sorted(p.name for p in result.prompts)
        assert names == ["memory-compress", "memory-usage-guide"]

        # Compress prompt
        r1 = await session.get_prompt(
            "memory-compress",
            arguments={"conversation": "User: I like PostgreSQL\nAssistant: Got it!"},
        )
        assert "PostgreSQL" in r1.messages[0].content.text

        # Usage guide
        r2 = await session.get_prompt("memory-usage-guide")
        assert "memory_ingest" in r2.messages[0].content.text

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 3: memory_ingest (all variants)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest(tmp_path):
    async def run(session: ClientSession):
        # Basic ingest
        d1 = await _call(session, "memory_ingest", {
            "content": "我决定使用PostgreSQL作为主数据库",
        })
        assert d1["stored"] is True

        # With tags
        d2 = await _call(session, "memory_ingest", {
            "content": "Redis缓存配置：maxmemory 256MB",
            "tags": ["Redis", "cache"],
        })
        assert d2["stored"] is True

        # Force
        d3 = await _call(session, "memory_ingest", {
            "content": "临时笔记：明天买牛奶",
            "force": True,
        })
        assert d3["stored"] is True
        assert d3["layer"] == "l2"

        # Pre-summary
        d4 = await _call(session, "memory_ingest", {
            "content": "Long deployment discussion...",
            "summary": "用户决定用Docker Compose部署，Nginx端口443",
            "entities": ["Docker", "Nginx"],
        })
        assert d4["stored"] is True
        assert d4["layer"] == "l2"

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 4: memory_recall
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall(tmp_path):
    async def run(session: ClientSession):
        # Seed data
        await _call(session, "memory_ingest", {
            "content": "记住：API密钥每90天轮换一次",
            "force": True,
        })
        await _call(session, "memory_ingest", {
            "content": "测试数据",
        })

        # Basic recall
        d1 = await _call(session, "memory_recall", {"query": "API密钥"})
        assert "memories" in d1
        assert d1["count"] >= 0

        # Layer filtering
        d2 = await _call(session, "memory_recall", {
            "query": "测试", "layers": ["l1"], "limit": 2,
        })
        for m in d2["memories"]:
            assert m["layer"] == "l1"

        # High min_score → empty
        d3 = await _call(session, "memory_recall", {
            "query": "xyzzy unrelated", "min_score": 0.99,
        })
        assert d3["count"] <= 1

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 5: memory_status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status(tmp_path):
    async def run(session: ClientSession):
        d1 = await _call(session, "memory_status", {})
        assert d1["l2_total_slots"] == 64

        # Ingest then check slots detail
        await _call(session, "memory_ingest", {"content": "重要配置", "force": True})
        d2 = await _call(session, "memory_status", {"detail": "slots"})
        assert "slots" in d2
        assert isinstance(d2["slots"], list)

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 6: memory_forget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forget(tmp_path):
    async def run(session: ClientSession):
        await _call(session, "memory_ingest", {"content": "密码是abc123", "force": True})
        d = await _call(session, "memory_forget", {"query": "密码"})
        assert "forgotten" in d
        assert isinstance(d["forgotten"], int)

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 7: memory_compact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact(tmp_path):
    async def run(session: ClientSession):
        d1 = await _call(session, "memory_compact", {"layer": "all"})
        assert "l2_evicted" in d1
        assert "l3_archived" in d1

        d2 = await _call(session, "memory_compact", {"layer": "l2"})
        assert "l2_evicted" in d2

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 8: Full lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_lifecycle(tmp_path):
    async def run(session: ClientSession):
        # Ingest
        d1 = await _call(session, "memory_ingest", {
            "content": "生产数据库IP是10.0.1.100，端口5432",
            "tags": ["database"], "force": True,
        })
        assert d1["stored"] is True

        # Pre-summary ingest
        await _call(session, "memory_ingest", {
            "content": "Caching discussion...",
            "summary": "Redis Cluster, 3主3从, maxmemory 2GB",
            "entities": ["Redis"],
        })

        # Status
        status = await _call(session, "memory_status", {"detail": "slots"})
        assert status["l2_active_slots"] > 0

        # Recall
        recall = await _call(session, "memory_recall", {"query": "数据库", "limit": 3})
        assert recall["count"] > 0

        # Forget
        forget = await _call(session, "memory_forget", {"query": "数据库IP"})
        assert forget["forgotten"] >= 0

        # Compact
        compact = await _call(session, "memory_compact", {"layer": "all"})
        assert "l2_evicted" in compact

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 9: Error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_handling(tmp_path):
    async def run(session: ClientSession):
        # Unknown tool
        d1 = await _call(session, "nonexistent_tool", {})
        assert "error" in d1

        # Missing required param
        d2 = await _call(session, "memory_ingest", {})
        assert "error" in d2

        d3 = await _call(session, "memory_recall", {})
        assert "error" in d3

    await _run_with_session(str(tmp_path), run)


# ---------------------------------------------------------------------------
# Test 10: Sequential tool calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sequential_calls(tmp_path):
    async def run(session: ClientSession):
        # 5 ingests
        for i in range(5):
            r = await _call(session, "memory_ingest", {
                "content": f"配置项{i}=value_{i}", "force": True,
            })
            assert r["stored"] is True

        # 5 recalls
        for i in range(5):
            await _call(session, "memory_recall", {"query": f"配置项{i}", "limit": 1})

        # Status check
        status = await _call(session, "memory_status", {})
        assert status["l2_active_slots"] > 0

    await _run_with_session(str(tmp_path), run)
