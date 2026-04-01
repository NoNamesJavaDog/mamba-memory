"""L3 Persistent Layer — SQLite + HNSW vector index.

Long-term storage for:
  - Memory records (evicted L2 slots, explicit saves, conversation archives)
  - L2 state snapshots (for cold-start recovery)
  - Entity graph (nodes + relations)
  - Vector embeddings (HNSW index for semantic search)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from mamba_memory.core.types import (
    EntityNode,
    EntityRelation,
    L3Config,
    MemoryRecord,
    MemorySource,
    StateSnapshot,
)

# Optional: hnswlib for vector search
try:
    import hnswlib  # type: ignore[import-untyped]

    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False


class PersistentLayer:
    """L3: SQLite structured store + HNSW vector index.

    Provides:
      - CRUD for memory records
      - State snapshot save/load (cold-start recovery)
      - Hybrid search (semantic + entity + time)
      - Entity graph maintenance
    """

    SCHEMA_VERSION = 2  # Bump when schema changes

    def __init__(self, config: L3Config | None = None, namespace: str = "default") -> None:
        self.config = config or L3Config()
        self.namespace = namespace
        db_path = os.path.expanduser(self.config.db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_tables()
        self._migrate()

        # Vector index
        self._index: hnswlib.Index | None = None
        self._id_map: dict[int, str] = {}  # hnsw internal id → memory record id
        self._next_hnsw_id = 0
        if HAS_HNSW:
            self._init_vector_index()

    def close(self) -> None:
        self._save_hnsw_index()
        self._conn.close()

    # -- Schema --------------------------------------------------------------

    def _init_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                content       TEXT NOT NULL,
                topic         TEXT NOT NULL DEFAULT '',
                source        TEXT NOT NULL DEFAULT 'conversation',
                session_id    TEXT DEFAULT '',
                importance    REAL NOT NULL DEFAULT 0.5,
                entities      TEXT NOT NULL DEFAULT '[]',
                embedding_id  TEXT,
                created_at    REAL NOT NULL,
                last_loaded_at REAL,
                load_count    INTEGER NOT NULL DEFAULT 0,
                archived      INTEGER NOT NULL DEFAULT 0,
                superseded_by TEXT,
                namespace     TEXT DEFAULT 'default'
            );

            CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);

            CREATE TABLE IF NOT EXISTS state_snapshots (
                id          TEXT PRIMARY KEY,
                step_count  INTEGER NOT NULL,
                slots_json  TEXT NOT NULL,
                created_at  REAL NOT NULL,
                session_id  TEXT DEFAULT '',
                namespace   TEXT DEFAULT 'default'
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_created ON state_snapshots(created_at);

            CREATE TABLE IF NOT EXISTS entities (
                name          TEXT PRIMARY KEY,
                type          TEXT NOT NULL DEFAULT 'concept',
                first_seen    REAL NOT NULL,
                last_seen     REAL NOT NULL,
                mention_count INTEGER NOT NULL DEFAULT 1,
                description   TEXT,
                namespace     TEXT DEFAULT 'default'
            );

            CREATE TABLE IF NOT EXISTS entity_relations (
                from_entity   TEXT NOT NULL,
                to_entity     TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'related',
                weight        REAL NOT NULL DEFAULT 1.0,
                last_seen     REAL NOT NULL,
                PRIMARY KEY (from_entity, to_entity, relation_type)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id TEXT PRIMARY KEY,
                vector    BLOB NOT NULL,
                hnsw_id   INTEGER
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );
        """)
        self._conn.commit()

        # Initialize schema version if empty
        row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if not row:
            self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (self.SCHEMA_VERSION,))
            self._conn.commit()

    def _migrate(self) -> None:
        """Run schema migrations if needed."""
        row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        current = row["version"] if row else 1

        if current < 2:
            self._migrate_v2()
            self._conn.execute("UPDATE schema_version SET version = 2")
            self._conn.commit()

    def _migrate_v2(self) -> None:
        """V2: add namespace column to memories, state_snapshots, entities."""
        for table in ("memories", "state_snapshots", "entities"):
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN namespace TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:
                pass  # Column already exists
        try:
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def _hnsw_index_path(self) -> str:
        """Path for the serialized HNSW index file (next to the SQLite DB)."""
        return os.path.splitext(os.path.expanduser(self.config.db_path))[0] + ".hnsw"

    def _init_vector_index(self) -> None:
        """Load HNSW index from disk, or rebuild from stored embeddings."""
        self._index = hnswlib.Index(space="cosine", dim=self.config.vector_dim)

        index_path = self._hnsw_index_path()

        # Try loading serialized index first (fast path)
        if os.path.exists(index_path):
            try:
                self._index.load_index(index_path, max_elements=self.config.hnsw_max_elements)
                self._index.set_ef(50)
                # Rebuild id_map from DB
                rows = self._conn.execute(
                    "SELECT memory_id, hnsw_id FROM embeddings WHERE hnsw_id IS NOT NULL"
                ).fetchall()
                for row in rows:
                    self._id_map[row["hnsw_id"]] = row["memory_id"]
                    self._next_hnsw_id = max(self._next_hnsw_id, row["hnsw_id"] + 1)
                return
            except Exception:
                pass  # Corrupted or incompatible, rebuild below

        # Slow path: rebuild from stored vectors
        self._index.init_index(
            max_elements=self.config.hnsw_max_elements,
            ef_construction=self.config.hnsw_ef_construction,
            M=self.config.hnsw_m,
        )
        self._index.set_ef(50)

        rows = self._conn.execute(
            "SELECT memory_id, vector, hnsw_id FROM embeddings WHERE hnsw_id IS NOT NULL"
        ).fetchall()

        if rows:
            for row in rows:
                hid = row["hnsw_id"]
                self._id_map[hid] = row["memory_id"]
                self._next_hnsw_id = max(self._next_hnsw_id, hid + 1)

            vectors = np.array(
                [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
            )
            ids = np.array([r["hnsw_id"] for r in rows])
            self._index.add_items(vectors, ids)

            # Save rebuilt index to disk
            self._save_hnsw_index()

    def _save_hnsw_index(self) -> None:
        """Persist the HNSW index to disk."""
        if self._index is not None and self._index.get_current_count() > 0:
            self._index.save_index(self._hnsw_index_path())

    # -- Memory Records ------------------------------------------------------

    def save_memory(self, record: MemoryRecord, embedding: list[float] | None = None) -> str:
        """Persist a memory record. Optionally index its embedding."""
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, topic, source, session_id, importance, entities,
                embedding_id, created_at, last_loaded_at, load_count, archived, superseded_by, namespace)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.content,
                record.topic,
                record.source.value,
                record.session_id,
                record.importance,
                json.dumps(record.entities, ensure_ascii=False),
                record.embedding_id,
                record.created_at,
                record.last_loaded_at,
                record.load_count,
                int(record.archived),
                record.superseded_by,
                self.namespace,
            ),
        )
        self._conn.commit()

        if embedding is not None:
            self._store_embedding(record.id, embedding)

        return record.id

    def get_memory(self, record_id: str) -> MemoryRecord | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (record_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def search_by_entity(self, entity: str, limit: int = 10) -> list[MemoryRecord]:
        """Find memories mentioning a specific entity."""
        pattern = f'%"{entity}"%'
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE archived = 0 AND entities LIKE ? AND namespace = ?
               ORDER BY importance DESC, created_at DESC
               LIMIT ?""",
            (pattern, self.namespace, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search_by_time(self, start: float, end: float, limit: int = 50) -> list[MemoryRecord]:
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE archived = 0 AND created_at BETWEEN ? AND ? AND namespace = ?
               ORDER BY created_at DESC LIMIT ?""",
            (start, end, self.namespace, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search_semantic(self, embedding: list[float], limit: int = 10) -> list[MemoryRecord]:
        """Vector nearest-neighbor search via HNSW index."""
        if self._index is None or self._index.get_current_count() == 0:
            return []

        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        actual_limit = min(limit, self._index.get_current_count())
        ids, distances = self._index.knn_query(vec, k=actual_limit)

        results: list[MemoryRecord] = []
        for hid, dist in zip(ids[0], distances[0]):
            mem_id = self._id_map.get(int(hid))
            if mem_id:
                record = self.get_memory(mem_id)
                if record and not record.archived:
                    results.append(record)
        return results

    def search_hybrid(
        self,
        embedding: list[float] | None = None,
        entity: str | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        """Combined semantic + entity search with dedup."""
        seen: set[str] = set()
        results: list[MemoryRecord] = []

        if embedding is not None:
            for r in self.search_semantic(embedding, limit=limit * 2):
                if r.id not in seen:
                    seen.add(r.id)
                    results.append(r)

        if entity is not None:
            for r in self.search_by_entity(entity, limit=limit):
                if r.id not in seen:
                    seen.add(r.id)
                    results.append(r)

        results.sort(key=lambda r: r.importance, reverse=True)
        return results[:limit]

    def mark_loaded(self, record_id: str) -> None:
        """Update last_loaded_at and increment load_count."""
        now = time.time()
        self._conn.execute(
            "UPDATE memories SET last_loaded_at = ?, load_count = load_count + 1 WHERE id = ?",
            (now, record_id),
        )
        self._conn.commit()

    def archive_memory(self, record_id: str, superseded_by: str | None = None) -> None:
        self._conn.execute(
            "UPDATE memories SET archived = 1, superseded_by = ? WHERE id = ?",
            (superseded_by, record_id),
        )
        self._conn.commit()

    # -- Snapshots -----------------------------------------------------------

    def save_snapshot(self, snapshot: StateSnapshot) -> str:
        self._conn.execute(
            """INSERT INTO state_snapshots (id, step_count, slots_json, created_at, session_id, namespace)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (snapshot.id, snapshot.step_count, snapshot.slots_json, snapshot.created_at,
             snapshot.session_id, self.namespace),
        )
        self._conn.commit()
        return snapshot.id

    def load_latest_snapshot(self) -> StateSnapshot | None:
        row = self._conn.execute(
            "SELECT * FROM state_snapshots WHERE namespace = ? ORDER BY created_at DESC LIMIT 1",
            (self.namespace,),
        ).fetchone()
        if not row:
            return None
        return StateSnapshot(
            id=row["id"],
            step_count=row["step_count"],
            slots_json=row["slots_json"],
            created_at=row["created_at"],
            session_id=row["session_id"],
        )

    # -- Entity Graph --------------------------------------------------------

    def upsert_entity(self, entity: EntityNode) -> None:
        self._conn.execute(
            """INSERT INTO entities (name, type, first_seen, last_seen, mention_count, description)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                 last_seen = MAX(last_seen, excluded.last_seen),
                 mention_count = mention_count + excluded.mention_count,
                 description = COALESCE(excluded.description, description)""",
            (entity.name, entity.type, entity.first_seen, entity.last_seen,
             entity.mention_count, entity.description),
        )
        self._conn.commit()

    def upsert_relation(self, rel: EntityRelation) -> None:
        self._conn.execute(
            """INSERT INTO entity_relations (from_entity, to_entity, relation_type, weight, last_seen)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(from_entity, to_entity, relation_type) DO UPDATE SET
                 weight = weight + excluded.weight,
                 last_seen = MAX(last_seen, excluded.last_seen)""",
            (rel.from_entity, rel.to_entity, rel.relation_type, rel.weight, rel.last_seen),
        )
        self._conn.commit()

    def get_entity(self, name: str) -> EntityNode | None:
        row = self._conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return EntityNode(
            name=row["name"], type=row["type"],
            first_seen=row["first_seen"], last_seen=row["last_seen"],
            mention_count=row["mention_count"], description=row["description"],
        )

    def get_related_entities(self, name: str, limit: int = 20) -> list[tuple[EntityNode, str]]:
        """Get entities related to the given entity."""
        rows = self._conn.execute(
            """SELECT e.*, er.relation_type
               FROM entity_relations er
               JOIN entities e ON e.name = er.to_entity
               WHERE er.from_entity = ?
               ORDER BY er.weight DESC LIMIT ?""",
            (name, limit),
        ).fetchall()
        return [
            (EntityNode(
                name=r["name"], type=r["type"],
                first_seen=r["first_seen"], last_seen=r["last_seen"],
                mention_count=r["mention_count"], description=r["description"],
            ), r["relation_type"])
            for r in rows
        ]

    def entity_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()
        return row["c"] if row else 0

    # -- Stats ---------------------------------------------------------------

    def record_count(self, archived: bool | None = None) -> int:
        if archived is None:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE namespace = ?", (self.namespace,)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE archived = ? AND namespace = ?",
                (int(archived), self.namespace),
            ).fetchone()
        return row["c"] if row else 0

    # -- Internal ------------------------------------------------------------

    def _store_embedding(self, memory_id: str, embedding: list[float]) -> None:
        vec = np.array(embedding, dtype=np.float32)
        blob = vec.tobytes()

        hnsw_id: int | None = None
        if self._index is not None:
            hnsw_id = self._next_hnsw_id
            self._next_hnsw_id += 1
            self._index.add_items(vec.reshape(1, -1), np.array([hnsw_id]))
            self._id_map[hnsw_id] = memory_id

        self._conn.execute(
            """INSERT OR REPLACE INTO embeddings (memory_id, vector, hnsw_id)
               VALUES (?, ?, ?)""",
            (memory_id, blob, hnsw_id),
        )
        self._conn.commit()

        # Periodically persist index to disk (every 100 inserts)
        if self._index is not None and self._next_hnsw_id % 100 == 0:
            self._save_hnsw_index()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            topic=row["topic"],
            source=MemorySource(row["source"]),
            session_id=row["session_id"],
            importance=row["importance"],
            entities=json.loads(row["entities"]),
            embedding_id=row["embedding_id"],
            created_at=row["created_at"],
            last_loaded_at=row["last_loaded_at"],
            load_count=row["load_count"],
            archived=bool(row["archived"]),
            superseded_by=row["superseded_by"],
        )
