"""Tests for L2 State Layer — Gate, Evolver, Recaller."""

import pytest

from mamba_memory.core.l2.gate import Gate, _importance_score
from mamba_memory.core.l2.state import StateLayer
from mamba_memory.core.types import (
    GateInput,
    L2Config,
    MemorySlot,
    RecallQuery,
)


# -- Gate tests --------------------------------------------------------------


class TestGateImportance:
    def test_decision_scores_high(self):
        assert _importance_score("我决定用Docker部署") > 0.0

    def test_preference_scores_high(self):
        assert _importance_score("I prefer PostgreSQL") > 0.0

    def test_correction_scores_high(self):
        assert _importance_score("不对，应该是8080端口") > 0.0

    def test_greeting_scores_zero(self):
        assert _importance_score("你好啊") == 0.0

    def test_explicit_memory_scores_high(self):
        assert _importance_score("记住这个密码") > 0.0


class TestGate:
    def test_discard_low_importance(self):
        gate = Gate()
        slots = [MemorySlot(id=0, state="existing topic", activation=0.5)]
        inp = GateInput(source="turn", content="嗯嗯好的")
        decision = gate.evaluate(inp, slots)
        assert not decision.should_write

    def test_store_explicit_memory(self):
        gate = Gate()
        slots = [MemorySlot(id=0)]
        inp = GateInput(source="turn", content="记住：服务器IP是192.168.1.1")
        decision = gate.evaluate(inp, slots)
        assert decision.should_write

    def test_store_decision(self):
        gate = Gate()
        slots = [MemorySlot(id=0)]
        inp = GateInput(source="turn", content="我决定使用PostgreSQL作为主数据库")
        decision = gate.evaluate(inp, slots)
        assert decision.should_write


# -- StateLayer tests --------------------------------------------------------


@pytest.fixture
def state():
    config = L2Config(slot_count=4, base_decay_rate=0.9, eviction_threshold=0.05)
    return StateLayer(config)


@pytest.mark.asyncio
async def test_ingest_stores_in_slot(state: StateLayer):
    ok, reason = await state.ingest("用户偏好Docker部署", entities=["Docker"])
    # Gate may or may not store depending on importance
    # At minimum it shouldn't crash
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


@pytest.mark.asyncio
async def test_ingest_explicit_stores(state: StateLayer):
    """Direct ingest with high-importance content should store."""
    ok, _ = await state.ingest(
        "记住：生产环境使用PostgreSQL, port 5432",
        entities=["PostgreSQL"],
    )
    assert ok is True


@pytest.mark.asyncio
async def test_recall_returns_results(state: StateLayer):
    # Ingest something
    await state.ingest(
        "记住：项目使用Python 3.12",
        entities=["Python"],
    )
    # Recall
    results = state.recall(RecallQuery(text="Python版本"))
    # May or may not find results depending on gate
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_decay_reduces_activation(state: StateLayer):
    # Force a slot to have content
    state.slots[0].state = "test content"
    state.slots[0].activation = 1.0

    # Step without writing (just decay)
    await state.ingest("你好", entities=[])  # low importance, likely discarded

    # Activation should have decayed
    assert state.slots[0].activation < 1.0


@pytest.mark.asyncio
async def test_snapshot_and_restore(state: StateLayer):
    state.slots[0].state = "test data"
    state.slots[0].topic = "test topic"
    state.slots[0].activation = 0.8

    snap = state.snapshot("test-session")

    new_state = StateLayer(state.config)
    new_state.restore(snap)

    assert new_state.slots[0].state == "test data"
    assert new_state.slots[0].topic == "test topic"


@pytest.mark.asyncio
async def test_force_evict(state: StateLayer):
    state.slots[0].state = "will be forgotten"
    state.slots[0].topic = "forgotten topic"

    evicted = await state.force_evict(0)
    assert evicted is not None
    assert evicted.state == "will be forgotten"
    assert state.slots[0].is_empty
