"""Tests for L1 Session Layer."""

import pytest

from mamba_memory.core.l1.session import SessionLayer
from mamba_memory.core.types import ConversationTurn, L1Config


@pytest.fixture
def session():
    config = L1Config(window_size=3, max_compressed_segments=2)
    return SessionLayer(config)


def _turn(content: str, role: str = "user") -> ConversationTurn:
    return ConversationTurn(role=role, content=content, tokens=len(content) // 4)


@pytest.mark.asyncio
async def test_window_keeps_recent_turns(session: SessionLayer):
    for i in range(3):
        await session.add_turn(_turn(f"message {i}"))

    assert len(session.window) == 3
    assert session.window[0].content == "message 0"
    assert session.window[2].content == "message 2"


@pytest.mark.asyncio
async def test_overflow_compresses(session: SessionLayer):
    """When window exceeds size, oldest turns are compressed."""
    for i in range(5):
        await session.add_turn(_turn(f"message {i}"))

    # Window should be trimmed to 3
    assert len(session.window) == 3
    # Compressed should have segments
    assert len(session.compressed) > 0


@pytest.mark.asyncio
async def test_compressed_overflow_returns_segments(session: SessionLayer):
    """When compressed segments exceed max, overflow is returned for L2."""
    overflow_segments = []
    for i in range(10):
        overflow = await session.add_turn(_turn(f"message {i}"))
        overflow_segments.extend(overflow)

    # Should have generated overflow segments for L2
    assert len(session.compressed) <= session.config.max_compressed_segments


@pytest.mark.asyncio
async def test_get_full_context(session: SessionLayer):
    for i in range(3):
        await session.add_turn(_turn(f"hello {i}"))

    ctx = session.get_full_context()
    assert len(ctx) == 3
    assert "[user]" in ctx[0]


@pytest.mark.asyncio
async def test_clear(session: SessionLayer):
    await session.add_turn(_turn("test"))
    session.clear()
    assert len(session.window) == 0
    assert len(session.compressed) == 0
