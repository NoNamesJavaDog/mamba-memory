"""Core type definitions for the three-layer memory architecture."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


def _now() -> float:
    return time.time()


def _uuid() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# L1 — Session Layer
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    """A single conversation turn with full fidelity."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: float = Field(default_factory=_now)
    tags: list[str] = Field(default_factory=list)
    tokens: int = 0


class CompressedSegment(BaseModel):
    """Compressed summary of multiple conversation turns that slid out of the window."""

    time_range: tuple[float, float]
    turn_count: int
    summary: str
    entities: list[str] = Field(default_factory=list)
    tokens: int = 0


class L1Config(BaseModel):
    """Configuration for the session layer."""

    window_size: int = 8
    """How many recent turns to keep at full fidelity."""

    max_compressed_segments: int = 20
    """Maximum compressed segments before overflow triggers L2 gate."""

    segment_target_tokens: int = 200
    """Target token count per compressed segment."""


# ---------------------------------------------------------------------------
# L2 — State Layer (SSM-inspired)
# ---------------------------------------------------------------------------


class MemorySlot(BaseModel):
    """A single slot in the fixed-dimension state vector.

    Each slot represents one cognitive topic — a compressed piece of knowledge
    that evolves over time through gating, decay, and activation.
    """

    id: int
    topic: str = ""
    state: str = ""
    entities: list[str] = Field(default_factory=list)

    activation: float = 1.0
    """Activity score (0–1). Decays over time, boosted on read/write."""

    last_active_step: int = 0
    last_write_step: int = 0
    created_step: int = 0

    last_active_time: float = 0.0
    """Wall-clock timestamp of last activation (for time-aware decay)."""

    rehearsal_count: int = 0
    """How many times this slot has been recalled (strengthens memory)."""

    source_refs: list[str] = Field(default_factory=list)
    """IDs of L3 MemoryRecords linked to this slot."""

    embedding: list[float] | None = None

    @property
    def is_empty(self) -> bool:
        return self.state == ""


class WriteMode(str, Enum):
    UPDATE = "update"
    """Append new info to existing state."""
    OVERWRITE = "overwrite"
    """Replace state entirely (old version archived to L3)."""
    MERGE = "merge"
    """LLM-assisted deep merge + dedup."""


class GateDecision(BaseModel):
    """Result of the selective gate evaluation."""

    should_write: bool
    target_slot: int = -1
    """Slot ID to write to, or -1 to create a new slot."""

    write_mode: WriteMode = WriteMode.UPDATE
    gate_strength: float = 0.5
    """How strongly the new content affects the slot (0–1)."""

    co_activations: list[int] = Field(default_factory=list)
    """Slot IDs that get a co-activation boost."""

    reason: str = ""


class GateInput(BaseModel):
    """Input to the gate evaluator."""

    source: Literal["turn", "compressed_segment", "l3_recall"]
    content: str
    entities: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class RecallQuery(BaseModel):
    """Query for selective recall from L2."""

    text: str
    embedding: list[float] | None = None
    required_entities: list[str] | None = None
    exclude_slots: list[int] | None = None
    limit: int = 5


class RecalledMemory(BaseModel):
    """A single recalled memory from L2."""

    slot_id: int
    topic: str
    content: str
    score: float
    breakdown: ScoreBreakdown


class ScoreBreakdown(BaseModel):
    semantic: float = 0.0
    activation: float = 0.0
    recency: float = 0.0


class L2Config(BaseModel):
    """Configuration for the state layer."""

    slot_count: int = 64
    """Number of memory slots (state dimension)."""

    slot_max_tokens: int = 150
    """Maximum tokens per slot state text."""

    base_decay_rate: float = 0.95
    """Per-step decay multiplier for activation (0–1)."""

    activation_boost: float = 1.5
    """Multiplier for activation boost on write."""

    eviction_threshold: float = 0.05
    """Slots below this activation get evicted to L3."""

    snapshot_interval: int = 50
    """Save full state snapshot to L3 every N steps."""

    time_decay_enabled: bool = True
    """Use wall-clock time decay in addition to step-based decay."""

    time_decay_halflife: float = 3600.0
    """Time-based decay half-life in seconds (default: 1 hour).
    After one half-life with no access, activation loses ~50%.
    Rehearsal (repeated recall) extends the half-life per Ebbinghaus."""

    # Recall scoring weights
    weight_semantic: float = 0.6
    weight_activation: float = 0.15
    weight_recency: float = 0.15
    weight_importance: float = 0.1


# ---------------------------------------------------------------------------
# L3 — Persistent Layer
# ---------------------------------------------------------------------------


class MemorySource(str, Enum):
    CONVERSATION = "conversation"
    COMPRESSION = "compression"
    EVICTION = "eviction"
    EXPLICIT = "explicit"
    EXTERNAL = "external"


class MemoryRecord(BaseModel):
    """A persistent memory record stored in L3."""

    id: str = Field(default_factory=_uuid)
    content: str
    topic: str = ""
    source: MemorySource = MemorySource.CONVERSATION
    session_id: str = ""
    importance: float = 0.5
    entities: list[str] = Field(default_factory=list)
    embedding_id: str | None = None
    created_at: float = Field(default_factory=_now)
    last_loaded_at: float | None = None
    load_count: int = 0
    archived: bool = False
    superseded_by: str | None = None


class StateSnapshot(BaseModel):
    """Serialized snapshot of the entire L2 state for persistence / cold-start recovery."""

    id: str = Field(default_factory=_uuid)
    step_count: int
    slots_json: str
    """JSON-serialized list of MemorySlot dicts."""

    created_at: float = Field(default_factory=_now)
    session_id: str = ""


class EntityNode(BaseModel):
    """An entity in the knowledge graph."""

    name: str
    type: Literal["person", "project", "tool", "concept", "location", "org"] = "concept"
    first_seen: float = Field(default_factory=_now)
    last_seen: float = Field(default_factory=_now)
    mention_count: int = 1
    description: str | None = None


class EntityRelation(BaseModel):
    """A directed relation between two entities."""

    from_entity: str
    to_entity: str
    relation_type: str = "related"
    weight: float = 1.0
    last_seen: float = Field(default_factory=_now)


class L3Config(BaseModel):
    """Configuration for the persistent layer."""

    db_path: str = "~/.mamba-memory/default.db"
    """SQLite database file path."""

    vector_dim: int = 256
    """Embedding vector dimension."""

    hnsw_max_elements: int = 100_000
    hnsw_ef_construction: int = 200
    hnsw_m: int = 16

    archive_after_days: int = 90
    """Auto-archive memories older than this (0 = disabled)."""


# ---------------------------------------------------------------------------
# Engine-level config
# ---------------------------------------------------------------------------


class EngineConfig(BaseModel):
    """Top-level configuration for MambaMemoryEngine."""

    l1: L1Config = Field(default_factory=L1Config)
    l2: L2Config = Field(default_factory=L2Config)
    l3: L3Config = Field(default_factory=L3Config)

    embedding_provider: str = "auto"
    """Which embedding backend to use: 'openai', 'local', 'none', or 'auto'."""

    compression_model: str = "auto"
    """Which LLM to use for compression/merge: 'openai', 'local', or 'auto'."""

    namespace: str = "default"
    """Namespace for multi-agent isolation. Each namespace has independent memory."""


# ---------------------------------------------------------------------------
# Recall / Ingest results (used by Engine public API)
# ---------------------------------------------------------------------------


class IngestResult(BaseModel):
    """Result of ingesting content into the memory system."""

    stored: bool
    layer: Literal["l1", "l2", "discarded"]
    slot_id: int | None = None
    reason: str = ""


class RecallResult(BaseModel):
    """Combined recall result across all three layers."""

    memories: list[RecallItem] = Field(default_factory=list)
    total_tokens: int = 0


class RecallItem(BaseModel):
    """A single item in the recall result."""

    content: str
    topic: str = ""
    score: float = 0.0
    layer: Literal["l1", "l2", "l3"]
    source_id: str | None = None


class MemoryStatus(BaseModel):
    """Current status of the memory system."""

    l1_window_turns: int = 0
    l1_compressed_segments: int = 0
    l2_active_slots: int = 0
    l2_total_slots: int = 0
    l2_step_count: int = 0
    l3_total_records: int = 0
    l3_archived_records: int = 0
    l3_entity_count: int = 0
