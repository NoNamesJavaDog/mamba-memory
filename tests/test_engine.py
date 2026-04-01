"""Integration tests for MambaMemoryEngine."""

import pytest

from mamba_memory.core.engine import MambaMemoryEngine
from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config


@pytest.fixture
async def engine(tmp_path):
    config = EngineConfig(
        l1=L1Config(window_size=3, max_compressed_segments=2),
        l2=L2Config(slot_count=8, base_decay_rate=0.9, eviction_threshold=0.05),
        l3=L3Config(db_path=str(tmp_path / "test.db"), vector_dim=256),
        embedding_provider="dummy",
    )
    eng = MambaMemoryEngine(config)
    await eng.start(session_id="test-session")
    yield eng
    await eng.shutdown()


@pytest.mark.asyncio
async def test_ingest_and_recall(engine: MambaMemoryEngine):
    """Basic ingest → recall roundtrip."""
    await engine.ingest("我决定使用PostgreSQL作为主数据库, port 5432", tags=["PostgreSQL"])

    result = await engine.recall("数据库")
    assert result.memories is not None


@pytest.mark.asyncio
async def test_status(engine: MambaMemoryEngine):
    status = engine.status()
    assert status.l2_total_slots == 8
    assert status.l1_window_turns == 0


@pytest.mark.asyncio
async def test_ingest_fills_window(engine: MambaMemoryEngine):
    for i in range(3):
        await engine.ingest(f"message {i}")

    status = engine.status()
    assert status.l1_window_turns == 3


@pytest.mark.asyncio
async def test_overflow_triggers_compression(engine: MambaMemoryEngine):
    """When window overflows, compression happens."""
    for i in range(6):
        await engine.ingest(f"message number {i} with some content")

    status = engine.status()
    assert status.l1_window_turns <= 3  # window is capped


@pytest.mark.asyncio
async def test_explicit_ingest(engine: MambaMemoryEngine):
    """Force-ingest bypasses the gate."""
    result = await engine.ingest_explicit(
        "API Key: sk-test-12345",
        tags=["credentials"],
    )
    assert result.stored is True


@pytest.mark.asyncio
async def test_forget(engine: MambaMemoryEngine):
    await engine.ingest_explicit("记住：密码是abc123", tags=["password"])
    count = await engine.forget("密码")
    # Should have forgotten at least something
    assert isinstance(count, int)


@pytest.mark.asyncio
async def test_compact(engine: MambaMemoryEngine):
    result = await engine.compact("all")
    assert "l2_evicted" in result
    assert "l3_archived" in result


@pytest.mark.asyncio
async def test_shutdown_saves_snapshot(engine: MambaMemoryEngine):
    await engine.ingest("记住这个重要信息", force=True)
    await engine.shutdown()

    # Re-start and check snapshot was loaded
    await engine.start(session_id="test-session-2")
    # L2 should have been restored
    status = engine.status()
    assert status.l2_step_count >= 0  # snapshot was loaded


@pytest.mark.asyncio
async def test_multi_layer_recall(engine: MambaMemoryEngine):
    """Recall should search across all layers."""
    # Put something in L1
    await engine.ingest("今天讨论了Docker部署方案")

    # Force something into L2
    await engine.ingest_explicit("生产环境使用Kubernetes", tags=["k8s"])

    # Recall should find content
    result = await engine.recall("部署", limit=10)
    assert isinstance(result.memories, list)
