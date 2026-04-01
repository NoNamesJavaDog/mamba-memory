"""Pluggable storage backend abstraction.

Defines the interface that any storage backend must implement,
enabling swap between SQLite (default), PostgreSQL, Redis, or custom backends.

Architecture:
    Engine → PersistentLayer(backend=...) → StorageBackend
                                              ├── SQLiteBackend (default, built-in)
                                              ├── PostgreSQLBackend (optional)
                                              └── RedisBackend (optional, L2 state only)

Usage:
    # Default (SQLite)
    store = PersistentLayer(config)

    # PostgreSQL
    store = PersistentLayer(config, backend=PostgreSQLBackend(dsn="..."))

    # Redis for L2 state + PostgreSQL for L3
    # (implemented at Engine level, not here)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mamba_memory.core.types import (
    EntityNode,
    EntityRelation,
    MemoryRecord,
    StateSnapshot,
)


class StorageBackend(ABC):
    """Abstract storage backend interface.

    All methods are synchronous (SQLite is sync, async wrappers added at engine level).
    PostgreSQL/Redis backends can use connection pools internally.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Create tables/indices if needed."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close connections and flush pending writes."""
        ...

    # -- Memory Records --

    @abstractmethod
    def save_memory(self, record: MemoryRecord, namespace: str) -> str:
        ...

    @abstractmethod
    def get_memory(self, record_id: str) -> MemoryRecord | None:
        ...

    @abstractmethod
    def search_by_entity(self, entity: str, namespace: str, limit: int) -> list[MemoryRecord]:
        ...

    @abstractmethod
    def search_by_time(self, start: float, end: float, namespace: str, limit: int) -> list[MemoryRecord]:
        ...

    @abstractmethod
    def mark_loaded(self, record_id: str) -> None:
        ...

    @abstractmethod
    def archive_memory(self, record_id: str, superseded_by: str | None) -> None:
        ...

    @abstractmethod
    def record_count(self, namespace: str, archived: bool | None) -> int:
        ...

    # -- Snapshots --

    @abstractmethod
    def save_snapshot(self, snapshot: StateSnapshot, namespace: str) -> str:
        ...

    @abstractmethod
    def load_latest_snapshot(self, namespace: str) -> StateSnapshot | None:
        ...

    # -- Entity Graph --

    @abstractmethod
    def upsert_entity(self, entity: EntityNode) -> None:
        ...

    @abstractmethod
    def upsert_relation(self, relation: EntityRelation) -> None:
        ...

    @abstractmethod
    def get_entity(self, name: str) -> EntityNode | None:
        ...

    @abstractmethod
    def get_related_entities(self, name: str, limit: int) -> list[tuple[EntityNode, str]]:
        ...

    @abstractmethod
    def entity_count(self) -> int:
        ...

    # -- Embeddings --

    @abstractmethod
    def store_embedding(self, memory_id: str, vector: bytes, hnsw_id: int | None) -> None:
        ...

    @abstractmethod
    def load_embeddings(self) -> list[dict[str, Any]]:
        """Load all embeddings for HNSW index rebuild."""
        ...


class PostgreSQLBackend(StorageBackend):
    """PostgreSQL storage backend (stub — requires asyncpg or psycopg).

    Install: pip install mamba-memory[postgres]

    Advantages over SQLite:
      - Concurrent multi-process access
      - pgvector extension for native vector search (replaces HNSW)
      - Better suited for multi-instance deployment
      - Full ACID with WAL replication

    Usage:
        backend = PostgreSQLBackend(dsn="postgresql://user:pass@host:5432/mamba")
        store = PersistentLayer(config, backend=backend)
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Any = None

    def initialize(self) -> None:
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "psycopg2 required for PostgreSQL backend: pip install psycopg2-binary"
            ) from e

        self._conn = psycopg2.connect(self._dsn)
        # Create tables (same schema as SQLite, adapted for PostgreSQL)
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'conversation',
                    session_id TEXT DEFAULT '',
                    importance REAL NOT NULL DEFAULT 0.5,
                    entities JSONB NOT NULL DEFAULT '[]',
                    embedding_id TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    last_loaded_at DOUBLE PRECISION,
                    load_count INTEGER NOT NULL DEFAULT 0,
                    archived BOOLEAN NOT NULL DEFAULT FALSE,
                    superseded_by TEXT,
                    namespace TEXT DEFAULT 'default'
                );
                CREATE INDEX IF NOT EXISTS idx_memories_ns ON memories(namespace);
                CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id TEXT PRIMARY KEY,
                    step_count INTEGER NOT NULL,
                    slots_json TEXT NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    session_id TEXT DEFAULT '',
                    namespace TEXT DEFAULT 'default'
                );

                CREATE TABLE IF NOT EXISTS entities (
                    name TEXT PRIMARY KEY,
                    type TEXT NOT NULL DEFAULT 'concept',
                    first_seen DOUBLE PRECISION NOT NULL,
                    last_seen DOUBLE PRECISION NOT NULL,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    description TEXT,
                    namespace TEXT DEFAULT 'default'
                );

                CREATE TABLE IF NOT EXISTS entity_relations (
                    from_entity TEXT NOT NULL,
                    to_entity TEXT NOT NULL,
                    relation_type TEXT NOT NULL DEFAULT 'related',
                    weight REAL NOT NULL DEFAULT 1.0,
                    last_seen DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (from_entity, to_entity, relation_type)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    memory_id TEXT PRIMARY KEY,
                    vector BYTEA NOT NULL,
                    hnsw_id INTEGER
                );
            """)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    # Stub implementations — each method follows the same pattern as SQLite
    # but uses psycopg2 parameterized queries with %s instead of ?

    def save_memory(self, record: MemoryRecord, namespace: str) -> str:
        import json
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO memories
                   (id, content, topic, source, session_id, importance, entities,
                    created_at, load_count, archived, namespace)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content""",
                (record.id, record.content, record.topic, record.source.value,
                 record.session_id, record.importance,
                 json.dumps(record.entities), record.created_at,
                 record.load_count, record.archived, namespace),
            )
        self._conn.commit()
        return record.id

    def get_memory(self, record_id: str) -> MemoryRecord | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM memories WHERE id = %s", (record_id,))
            row = cur.fetchone()
        return None  # TODO: row → MemoryRecord conversion

    def search_by_entity(self, entity: str, namespace: str, limit: int) -> list[MemoryRecord]:
        return []  # TODO

    def search_by_time(self, start: float, end: float, namespace: str, limit: int) -> list[MemoryRecord]:
        return []  # TODO

    def mark_loaded(self, record_id: str) -> None:
        pass  # TODO

    def archive_memory(self, record_id: str, superseded_by: str | None) -> None:
        pass  # TODO

    def record_count(self, namespace: str, archived: bool | None) -> int:
        return 0  # TODO

    def save_snapshot(self, snapshot: StateSnapshot, namespace: str) -> str:
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO state_snapshots (id, step_count, slots_json, created_at, session_id, namespace)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (snapshot.id, snapshot.step_count, snapshot.slots_json,
                 snapshot.created_at, snapshot.session_id, namespace),
            )
        self._conn.commit()
        return snapshot.id

    def load_latest_snapshot(self, namespace: str) -> StateSnapshot | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM state_snapshots WHERE namespace = %s ORDER BY created_at DESC LIMIT 1",
                (namespace,),
            )
            row = cur.fetchone()
        return None  # TODO: row → StateSnapshot conversion

    def upsert_entity(self, entity: EntityNode) -> None:
        pass  # TODO

    def upsert_relation(self, relation: EntityRelation) -> None:
        pass  # TODO

    def get_entity(self, name: str) -> EntityNode | None:
        return None  # TODO

    def get_related_entities(self, name: str, limit: int) -> list[tuple[EntityNode, str]]:
        return []  # TODO

    def entity_count(self) -> int:
        return 0  # TODO

    def store_embedding(self, memory_id: str, vector: bytes, hnsw_id: int | None) -> None:
        pass  # TODO

    def load_embeddings(self) -> list[dict[str, Any]]:
        return []  # TODO


class RedisStateBackend:
    """Redis backend for L2 state layer only (not full StorageBackend).

    Redis is used for fast L2 state serialization across multiple
    engine instances sharing the same memory state.

    Install: pip install redis

    Usage:
        redis_state = RedisStateBackend(url="redis://localhost:6379/0")
        # Save/load L2 snapshots to Redis instead of SQLite
        redis_state.save_snapshot(state_json, namespace="agent-1")
        state_json = redis_state.load_snapshot(namespace="agent-1")
    """

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "mamba") -> None:
        try:
            import redis
        except ImportError as e:
            raise ImportError("redis required: pip install redis") from e

        self._client = redis.from_url(url)
        self._prefix = prefix

    def _key(self, namespace: str, suffix: str) -> str:
        return f"{self._prefix}:{namespace}:{suffix}"

    def save_snapshot(self, state_json: str, namespace: str = "default") -> None:
        """Save L2 state snapshot to Redis (replaces SQLite snapshots)."""
        self._client.set(self._key(namespace, "l2_state"), state_json)

    def load_snapshot(self, namespace: str = "default") -> str | None:
        """Load L2 state snapshot from Redis."""
        data = self._client.get(self._key(namespace, "l2_state"))
        return data.decode("utf-8") if data else None

    def save_slot(self, slot_id: int, slot_json: str, namespace: str = "default") -> None:
        """Save a single L2 slot (for real-time sync between instances)."""
        self._client.hset(self._key(namespace, "slots"), str(slot_id), slot_json)

    def load_slot(self, slot_id: int, namespace: str = "default") -> str | None:
        data = self._client.hget(self._key(namespace, "slots"), str(slot_id))
        return data.decode("utf-8") if data else None

    def close(self) -> None:
        self._client.close()
