"""Selective Recall — input-driven memory retrieval from L2 state.

Corresponds to the SSM output projection C:  y(t) = C · h(t)

Instead of returning all state, recall selectively activates relevant
slots based on the current query. Score is a weighted blend of:
  - Semantic similarity (embedding cosine)
  - Activation level (includes decay history)
  - Recency (time since last active)
  - Importance (initial gate strength)

Recalled slots get an activation boost, creating positive feedback:
useful memories persist longer.
"""

from __future__ import annotations

from mamba_memory.core.l2.gate import cosine_similarity
from mamba_memory.core.types import (
    L2Config,
    MemorySlot,
    RecallQuery,
    RecalledMemory,
    ScoreBreakdown,
)


class Recaller:
    """Retrieves relevant memories from the L2 state vector."""

    def __init__(self, config: L2Config | None = None) -> None:
        self.config = config or L2Config()

    def recall(
        self,
        query: RecallQuery,
        slots: list[MemorySlot],
        step_count: int,
    ) -> list[RecalledMemory]:
        """Score all non-empty slots against the query and return top-K.

        Side effect: recalled slots get an activation boost.
        """
        exclude = set(query.exclude_slots or [])
        candidates: list[RecalledMemory] = []

        for slot in slots:
            if slot.is_empty or slot.id in exclude:
                continue

            breakdown = self._score_slot(query, slot, step_count)
            total = (
                self.config.weight_semantic * breakdown.semantic
                + self.config.weight_activation * breakdown.activation
                + self.config.weight_recency * breakdown.recency
            )

            # Entity bonus: if query requires specific entities, boost matching slots
            if query.required_entities:
                overlap = set(query.required_entities) & set(slot.entities)
                if overlap:
                    total += 0.2 * (len(overlap) / len(query.required_entities))

            candidates.append(
                RecalledMemory(
                    slot_id=slot.id,
                    topic=slot.topic,
                    content=slot.state,
                    score=total,
                    breakdown=breakdown,
                )
            )

        # Sort by score descending
        candidates.sort(key=lambda m: m.score, reverse=True)
        top_k = candidates[: query.limit]

        # Side effect: boost activation + rehearsal of recalled slots
        import time as _time

        recalled_ids = {m.slot_id for m in top_k}
        now = _time.time()
        for slot in slots:
            if slot.id in recalled_ids:
                slot.activation = min(1.0, slot.activation + 0.1)
                slot.last_active_step = step_count
                slot.last_active_time = now
                slot.rehearsal_count += 1  # Ebbinghaus: rehearsal extends half-life

        return top_k

    def _score_slot(
        self,
        query: RecallQuery,
        slot: MemorySlot,
        step_count: int,
    ) -> ScoreBreakdown:
        """Compute per-dimension scores for a single slot."""

        # Semantic similarity
        semantic = 0.0
        if query.embedding is not None and slot.embedding is not None:
            semantic = max(0.0, cosine_similarity(query.embedding, slot.embedding))

        # Activation (already incorporates decay history)
        activation = slot.activation

        # Recency: exponential decay based on steps since last active
        steps_ago = max(1, step_count - slot.last_active_step)
        recency = 1.0 / (1.0 + 0.1 * steps_ago)

        return ScoreBreakdown(
            semantic=semantic,
            activation=activation,
            recency=recency,
        )
