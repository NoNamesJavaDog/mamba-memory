"""Tests for L3 Persistent Layer."""

import os
import tempfile

import pytest

from mamba_memory.core.l3.store import PersistentLayer
from mamba_memory.core.types import (
    EntityNode,
    EntityRelation,
    L3Config,
    MemoryRecord,
    MemorySource,
    StateSnapshot,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    config = L3Config(db_path=db_path, vector_dim=8)
    s = PersistentLayer(config)
    yield s
    s.close()


class TestMemoryRecords:
    def test_save_and_get(self, store: PersistentLayer):
        record = MemoryRecord(
            content="用户偏好Docker部署",
            topic="deployment",
            source=MemorySource.CONVERSATION,
            importance=0.8,
            entities=["Docker"],
        )
        rid = store.save_memory(record)
        loaded = store.get_memory(rid)
        assert loaded is not None
        assert loaded.content == "用户偏好Docker部署"
        assert loaded.topic == "deployment"
        assert loaded.importance == 0.8

    def test_search_by_entity(self, store: PersistentLayer):
        store.save_memory(MemoryRecord(
            content="PostgreSQL is the main DB", topic="db", entities=["PostgreSQL"],
        ))
        store.save_memory(MemoryRecord(
            content="Redis for caching", topic="cache", entities=["Redis"],
        ))

        results = store.search_by_entity("PostgreSQL")
        assert len(results) == 1
        assert results[0].content == "PostgreSQL is the main DB"

    def test_archive(self, store: PersistentLayer):
        record = MemoryRecord(content="old info", topic="test")
        rid = store.save_memory(record)
        store.archive_memory(rid)
        loaded = store.get_memory(rid)
        assert loaded is not None
        assert loaded.archived is True

    def test_record_count(self, store: PersistentLayer):
        assert store.record_count() == 0
        store.save_memory(MemoryRecord(content="a", topic="t"))
        store.save_memory(MemoryRecord(content="b", topic="t"))
        assert store.record_count() == 2


class TestSnapshots:
    def test_save_and_load(self, store: PersistentLayer):
        snap = StateSnapshot(step_count=42, slots_json='[{"id":0}]', session_id="s1")
        store.save_snapshot(snap)

        loaded = store.load_latest_snapshot()
        assert loaded is not None
        assert loaded.step_count == 42
        assert loaded.session_id == "s1"

    def test_load_latest(self, store: PersistentLayer):
        store.save_snapshot(StateSnapshot(step_count=1, slots_json="[]"))
        store.save_snapshot(StateSnapshot(step_count=2, slots_json="[]"))
        loaded = store.load_latest_snapshot()
        assert loaded is not None
        assert loaded.step_count == 2


class TestEntityGraph:
    def test_upsert_entity(self, store: PersistentLayer):
        store.upsert_entity(EntityNode(name="Docker", type="tool"))
        entity = store.get_entity("Docker")
        assert entity is not None
        assert entity.type == "tool"

    def test_upsert_increments_count(self, store: PersistentLayer):
        store.upsert_entity(EntityNode(name="Python", type="tool", mention_count=1))
        store.upsert_entity(EntityNode(name="Python", type="tool", mention_count=1))
        entity = store.get_entity("Python")
        assert entity is not None
        assert entity.mention_count == 2

    def test_relations(self, store: PersistentLayer):
        store.upsert_entity(EntityNode(name="App", type="project"))
        store.upsert_entity(EntityNode(name="Docker", type="tool"))
        store.upsert_relation(EntityRelation(
            from_entity="App", to_entity="Docker", relation_type="uses",
        ))
        related = store.get_related_entities("App")
        assert len(related) == 1
        assert related[0][0].name == "Docker"
        assert related[0][1] == "uses"
