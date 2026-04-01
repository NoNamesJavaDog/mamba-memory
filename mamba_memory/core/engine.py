"""MambaMemory Engine — orchestrates the three-layer memory system.

This is the single public entry point. All external interfaces (MCP, HTTP,
SDK) talk to the Engine; the Engine coordinates L1 ↔ L2 ↔ L3 data flow.

Write path:  input → L1 window → overflow → L2 gate → L2 state → evict → L3
Read path:   query → L1 context → L2 recall → L3 search → merged result
Cold start:  L3 snapshot → restore L2 → empty L1
"""

from __future__ import annotations

import asyncio
import time

from mamba_memory.core.embedding import EmbeddingProvider, create_provider
from mamba_memory.core.l1.session import SessionLayer
from mamba_memory.core.l2.state import StateLayer
from mamba_memory.core.l3.store import PersistentLayer
from mamba_memory.core.types import (
    ConversationTurn,
    EngineConfig,
    IngestResult,
    MemoryRecord,
    MemorySource,
    MemoryStatus,
    RecallItem,
    RecallQuery,
    RecallResult,
)


class _EvictionHandler:
    """Routes evicted L2 slots to L3 persistent storage."""

    def __init__(self, l3: PersistentLayer, embedder: EmbeddingProvider) -> None:
        self._l3 = l3
        self._embedder = embedder

    async def on_evict(self, slot) -> None:
        record = MemoryRecord(
            content=slot.state,
            topic=slot.topic,
            source=MemorySource.EVICTION,
            importance=slot.activation,
            entities=slot.entities,
        )
        embedding = slot.embedding
        self._l3.save_memory(record, embedding=embedding)


class MambaMemoryEngine:
    """Top-level engine that wires L1, L2, L3 together.

    Usage::

        engine = MambaMemoryEngine()
        await engine.start()

        # Ingest conversation turns
        result = await engine.ingest("用户决定用Docker部署", source="session-1")

        # Recall relevant memories
        memories = await engine.recall("部署方式是什么？")

        # Check status
        status = engine.status()

        await engine.shutdown()
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._lock = asyncio.Lock()
        self._embedder: EmbeddingProvider | None = None
        self._l1: SessionLayer | None = None
        self._l2: StateLayer | None = None
        self._l3: PersistentLayer | None = None
        self._session_id = ""
        self._started = False

    @property
    def l1(self) -> SessionLayer:
        assert self._l1 is not None, "Engine not started"
        return self._l1

    @property
    def l2(self) -> StateLayer:
        assert self._l2 is not None, "Engine not started"
        return self._l2

    @property
    def l3(self) -> PersistentLayer:
        assert self._l3 is not None, "Engine not started"
        return self._l3

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, session_id: str = "") -> None:
        """Initialize all layers and restore state from L3 if available."""
        self._session_id = session_id or f"s-{int(time.time())}"

        # Embedding provider
        self._embedder = create_provider(self.config.embedding_provider)

        # Update L3 config with actual embedding dim
        self.config.l3.vector_dim = self._embedder.dim

        # L3 first (needed for cold-start restore and eviction target)
        self._l3 = PersistentLayer(self.config.l3, namespace=self.config.namespace)

        # LLM compressor / merger (if configured)
        llm_compressor = None
        llm_merger = None
        if self.config.compression_model != "none":
            try:
                from mamba_memory.core.llm import create_compressor, create_merger

                llm_compressor = create_compressor(self.config.compression_model)
                llm_merger = create_merger(
                    self.config.compression_model,
                    max_tokens=self.config.l2.slot_max_tokens,
                )
            except Exception:
                pass  # Fall back to naive compressor/merger

        # L2 with eviction handler wired to L3
        eviction = _EvictionHandler(self._l3, self._embedder)
        self._l2 = StateLayer(
            self.config.l2,
            merger=llm_merger,
            on_evict=eviction,
        )

        # Cold-start: restore L2 from latest snapshot
        snapshot = self._l3.load_latest_snapshot()
        if snapshot:
            elapsed = time.time() - snapshot.created_at
            self._l2.restore(snapshot, elapsed_seconds=elapsed)

        # L1
        self._l1 = SessionLayer(self.config.l1)

        # Wire LLM compressor to L1
        if llm_compressor:
            self._l1.set_compressor(llm_compressor)

        self._started = True

    async def shutdown(self) -> None:
        """Save final snapshot and close resources."""
        if self._l2 and self._l3:
            snap = self._l2.snapshot(self._session_id)
            self._l3.save_snapshot(snap)
            self._l3.close()
        self._started = False

    # -- Write path ----------------------------------------------------------

    async def ingest(
        self,
        content: str,
        *,
        role: str = "user",
        source: str = "",
        tags: list[str] | None = None,
        force: bool = False,
        pre_summary: str | None = None,
        pre_entities: list[str] | None = None,
    ) -> IngestResult:
        """Ingest content into the memory system.

        Flow:
        1. Content enters L1 as a conversation turn
        2. If L1 window overflows, compressed segments go to L2 gate
        3. L2 gate decides whether to store in state layer
        4. Evicted L2 slots are persisted to L3

        Args:
            content: Text to remember
            role: Conversation role (user/assistant/system/tool)
            source: Source identifier (session ID, etc.)
            tags: Entity tags
            force: Skip gate, force storage in L2
            pre_summary: Client-provided summary (skips internal LLM compression)
            pre_entities: Client-provided entities (used with pre_summary)
        """
        self._ensure_started()
        async with self._lock:
            return await self._ingest_locked(
                content, role=role, source=source, tags=tags,
                force=force, pre_summary=pre_summary, pre_entities=pre_entities,
            )

    async def _ingest_locked(
        self,
        content: str,
        *,
        role: str,
        source: str,
        tags: list[str] | None,
        force: bool,
        pre_summary: str | None,
        pre_entities: list[str] | None,
    ) -> IngestResult:
        tokens = _estimate_tokens(content)

        turn = ConversationTurn(
            role=role,
            content=content,
            tags=tags or [],
            tokens=tokens,
        )

        # Pre-compressed path: client already summarized, go straight to L2
        if pre_summary:
            embedding = await self._embed(pre_summary)
            ok, r = await self.l2.ingest(
                pre_summary,
                source="turn",
                entities=pre_entities or tags,
                embedding=embedding,
            )
            # Still add raw turn to L1 window for recent context
            await self.l1.add_turn(turn)
            return IngestResult(stored=ok, layer="l2" if ok else "l1", reason=r)

        # L1: add turn, get overflow segments
        overflow = await self.l1.add_turn(turn)

        if not overflow and not force:
            return IngestResult(stored=True, layer="l1", reason="in window")

        # Process overflow through L2
        stored_in_l2 = False
        reason = "no overflow"

        for segment in overflow:
            embedding = await self._embed(segment.summary)
            ok, r = await self.l2.ingest_segment(segment, embedding=embedding)
            if ok:
                stored_in_l2 = True
                reason = r

        # Force-ingest directly to L2 (bypasses gate entirely)
        if force:
            embedding = await self._embed(content)
            await self.l2.force_write(
                content,
                entities=tags or [],
                embedding=embedding,
            )
            stored_in_l2 = True
            reason = "force-ingested, gate bypassed"

        # Update entity graph in L3 (with typed relation extraction)
        all_entities = list(set(tags or []) | set(pre_entities or []))
        if all_entities:
            self._update_entity_graph(all_entities, content=content)

        # Periodic L2 → L3 snapshot
        if self.l2.step_count % self.config.l2.snapshot_interval == 0 and self.l2.step_count > 0:
            snap = self.l2.snapshot(self._session_id)
            self.l3.save_snapshot(snap)

        return IngestResult(
            stored=stored_in_l2,
            layer="l2" if stored_in_l2 else "l1",
            reason=reason,
        )

    async def ingest_explicit(
        self,
        content: str,
        tags: list[str] | None = None,
    ) -> IngestResult:
        """Explicitly store a memory (user said 'remember this'). Bypasses gate."""
        return await self.ingest(content, role="system", tags=tags, force=True)

    # -- Read path -----------------------------------------------------------

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        layers: list[str] | None = None,
        min_score: float = 0.0,
    ) -> RecallResult:
        """Recall relevant memories across all layers.

        Args:
            query: Query text
            limit: Max results
            layers: Which layers to query ('l1', 'l2', 'l3'). Default: all.
            min_score: Minimum relevance score to include
        """
        self._ensure_started()
        async with self._lock:
            return await self._recall_locked(query, limit=limit, layers=layers, min_score=min_score)

    async def _recall_locked(
        self,
        query: str,
        *,
        limit: int,
        layers: list[str] | None,
        min_score: float,
    ) -> RecallResult:
        active_layers = set(layers or ["l1", "l2", "l3"])
        items: list[RecallItem] = []

        embedding = await self._embed(query)

        # L1: keyword match in recent context
        if "l1" in active_layers:
            l1_items = self._recall_l1(query)
            items.extend(l1_items)

        # L2: selective recall from state vector
        if "l2" in active_layers:
            rq = RecallQuery(text=query, embedding=embedding, limit=limit)
            recalled = self.l2.recall(rq)
            for m in recalled:
                if m.score >= min_score:
                    items.append(RecallItem(
                        content=m.content,
                        topic=m.topic,
                        score=m.score,
                        layer="l2",
                        source_id=str(m.slot_id),
                    ))

        # L3: persistent search (semantic + entity graph)
        if "l3" in active_layers:
            # Semantic search
            if embedding:
                l3_records = self.l3.search_semantic(embedding, limit=limit)
                for r in l3_records:
                    items.append(RecallItem(
                        content=r.content,
                        topic=r.topic,
                        score=r.importance,
                        layer="l3",
                        source_id=r.id,
                    ))
                    self.l3.mark_loaded(r.id)

            # Entity graph search — find entities mentioned in query
            # then look up related memories via entity index
            query_words = set(query.split())
            for word in query_words:
                entity = self.l3.get_entity(word)
                if entity:
                    entity_records = self.l3.search_by_entity(word, limit=3)
                    for r in entity_records:
                        items.append(RecallItem(
                            content=r.content,
                            topic=r.topic,
                            score=r.importance * 0.9,
                            layer="l3",
                            source_id=r.id,
                        ))

        # Dedup and sort
        seen: set[str] = set()
        unique: list[RecallItem] = []
        for item in items:
            key = item.content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(item)

        unique.sort(key=lambda x: x.score, reverse=True)
        unique = unique[:limit]

        total_tokens = sum(_estimate_tokens(i.content) for i in unique)

        return RecallResult(memories=unique, total_tokens=total_tokens)

    # -- Management ----------------------------------------------------------

    async def forget(self, query: str) -> int:
        """Forget memories matching the query. Returns count of forgotten items."""
        self._ensure_started()
        async with self._lock:
            return await self._forget_locked(query)

    async def _forget_locked(self, query: str) -> int:
        count = 0

        embedding = await self._embed(query)

        # Forget from L2
        rq = RecallQuery(text=query, embedding=embedding, limit=5)
        matches = self.l2.recall(rq)
        for m in matches:
            if m.score > 0.5:
                await self.l2.force_evict(m.slot_id)
                count += 1

        # Archive from L3
        if embedding:
            l3_matches = self.l3.search_semantic(embedding, limit=5)
            for r in l3_matches:
                self.l3.archive_memory(r.id)
                count += 1

        return count

    async def compact(self, layer: str = "all") -> dict:
        """Manual compaction trigger.

        - 'l2': evict low-activation slots to L3
        - 'l3': archive old records
        - 'all': both
        """
        self._ensure_started()
        async with self._lock:
            return await self._compact_locked(layer)

    async def _compact_locked(self, layer: str) -> dict:
        result = {"l2_evicted": 0, "l3_archived": 0}

        if layer in ("l2", "all"):
            for slot in self.l2.slots:
                if not slot.is_empty and slot.activation < self.config.l2.eviction_threshold * 2:
                    await self.l2.force_evict(slot.id)
                    result["l2_evicted"] += 1

        if layer in ("l3", "all") and self.config.l3.archive_after_days > 0:
            cutoff = time.time() - (self.config.l3.archive_after_days * 86400)
            old = self.l3.search_by_time(0, cutoff, limit=1000)
            for r in old:
                self.l3.archive_memory(r.id)
                result["l3_archived"] += 1

        return result

    def status(self) -> MemoryStatus:
        """Get current memory system status."""
        self._ensure_started()
        return MemoryStatus(
            l1_window_turns=len(self.l1.window),
            l1_compressed_segments=len(self.l1.compressed),
            l2_active_slots=self.l2.active_slot_count,
            l2_total_slots=self.config.l2.slot_count,
            l2_step_count=self.l2.step_count,
            l3_total_records=self.l3.record_count(),
            l3_archived_records=self.l3.record_count(archived=True),
            l3_entity_count=self.l3.entity_count(),
        )

    # -- Internal ------------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("Engine not started. Call await engine.start() first.")

    async def _embed(self, text: str) -> list[float] | None:
        if self._embedder is None:
            return None
        try:
            return await self._embedder.embed(text)
        except Exception:
            return None

    def _update_entity_graph(self, entities: list[str], content: str = "") -> None:
        """Insert entities and typed relations into L3 knowledge graph."""
        from mamba_memory.core.l3.knowledge_graph import extract_relations
        from mamba_memory.core.types import EntityNode, EntityRelation

        for name in entities:
            self.l3.upsert_entity(EntityNode(name=name))

        # Co-occurrence relations
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                self.l3.upsert_relation(
                    EntityRelation(from_entity=a, to_entity=b, relation_type="co_mentioned")
                )
                self.l3.upsert_relation(
                    EntityRelation(from_entity=b, to_entity=a, relation_type="co_mentioned")
                )

        # Typed relations extracted from content
        if content:
            entity_set = set(entities)
            typed_rels = extract_relations(content, known_entities=entity_set)
            for rel in typed_rels:
                self.l3.upsert_entity(EntityNode(name=rel.from_entity))
                self.l3.upsert_entity(EntityNode(name=rel.to_entity))
                self.l3.upsert_relation(rel)

    def _recall_l1(self, query: str) -> list[RecallItem]:
        """Ngram + token overlap search in L1 window + compressed segments."""
        from mamba_memory.core.text import text_relevance

        items: list[RecallItem] = []

        # Window turns
        for turn in self.l1.window:
            score = text_relevance(query, turn.content)
            if score > 0.05:
                items.append(RecallItem(
                    content=turn.content,
                    topic=f"[{turn.role}]",
                    score=score * 0.8,  # L1 scores slightly discounted
                    layer="l1",
                ))

        # Compressed segments
        for seg in self.l1.compressed:
            score = text_relevance(query, seg.summary)
            if score > 0.05:
                items.append(RecallItem(
                    content=seg.summary,
                    topic="[compressed]",
                    score=score * 0.6,
                    layer="l1",
                ))

        return items


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.5 tokens per CJK char, ~0.25 per ASCII char."""
    cjk = sum(1 for c in text if ord(c) > 0x2E80)
    ascii_chars = len(text) - cjk
    return int(cjk * 1.5 + ascii_chars * 0.25)
