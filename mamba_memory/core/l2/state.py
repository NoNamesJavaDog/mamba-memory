"""L2 State Layer — the SSM-inspired cognitive state manager.

Orchestrates Gate + Evolver + Recaller to maintain a fixed-dimension
state vector that evolves with each interaction.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mamba_memory.core.l2.evolver import EvictionCallback, Evolver, SlotMerger
from mamba_memory.core.l2.gate import Gate
from mamba_memory.core.l2.recaller import Recaller
from mamba_memory.core.types import (
    CompressedSegment,
    GateInput,
    L2Config,
    MemorySlot,
    RecallQuery,
    RecalledMemory,
    StateSnapshot,
)

if TYPE_CHECKING:
    pass


class StateLayer:
    """L2: Fixed-dimension cognitive state with selective gating and decay.

    The state layer maintains ``slot_count`` memory slots. Each slot holds
    one compressed topic. Slots evolve through:
      - **Gate**: decides what to write and where
      - **Evolver**: applies decay, writes, and evicts dead slots
      - **Recaller**: selectively retrieves relevant slots for a query
    """

    def __init__(
        self,
        config: L2Config | None = None,
        merger: SlotMerger | None = None,
        on_evict: EvictionCallback | None = None,
    ) -> None:
        self.config = config or L2Config()

        # Initialize empty slots
        self.slots: list[MemorySlot] = [
            MemorySlot(id=i) for i in range(self.config.slot_count)
        ]

        self.gate = Gate(self.config)
        self.evolver = Evolver(self.config, merger=merger, on_evict=on_evict)
        self.recaller = Recaller(self.config)

    @property
    def step_count(self) -> int:
        return self.evolver.step_count

    @property
    def active_slot_count(self) -> int:
        return sum(1 for s in self.slots if not s.is_empty)

    # -- Write path ----------------------------------------------------------

    async def ingest(
        self,
        content: str,
        *,
        source: str = "compressed_segment",
        entities: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> tuple[bool, str]:
        """Evaluate and potentially write content into the state layer.

        Returns (stored: bool, reason: str).
        """
        inp = GateInput(
            source=source,
            content=content,
            entities=entities or [],
            embedding=embedding,
        )

        decision = self.gate.evaluate(inp, self.slots)

        if not decision.should_write:
            # Still step the evolver (decay happens regardless)
            await self.evolver.step(self.slots, None)
            return False, decision.reason

        await self.evolver.step(
            self.slots,
            decision,
            content=content,
            entities=entities,
            embedding=embedding,
        )

        return True, decision.reason

    async def force_write(
        self,
        content: str,
        *,
        entities: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Force-write content to L2, bypassing gate evaluation entirely.

        Used for explicit user memory requests ("remember this").
        Finds or allocates a slot and writes with maximum gate strength.
        If overwriting a non-empty slot, the old content is evicted to L3 first.
        """
        from mamba_memory.core.types import GateDecision, WriteMode

        target = self.gate._find_free_slot(self.slots)
        if target == -1:
            # Even when saturated, force-write evicts the weakest
            target = min(self.slots, key=lambda s: s.activation).id

        # Evict old content to L3 before overwriting
        target_slot = self.slots[target]
        if not target_slot.is_empty and self.evolver._on_evict:
            await self.evolver._on_evict.on_evict(target_slot)

        decision = GateDecision(
            should_write=True,
            target_slot=target,
            write_mode=WriteMode.OVERWRITE,
            gate_strength=0.8,  # slightly below 1.0 so all force-writes don't max out
            reason="force-write, gate bypassed",
        )

        await self.evolver.step(
            self.slots,
            decision,
            content=content,
            entities=entities,
            embedding=embedding,
        )

    async def ingest_segment(
        self,
        segment: CompressedSegment,
        embedding: list[float] | None = None,
    ) -> tuple[bool, str]:
        """Convenience: ingest a compressed segment from L1 overflow."""
        return await self.ingest(
            content=segment.summary,
            source="compressed_segment",
            entities=segment.entities,
            embedding=embedding,
        )

    # -- Read path -----------------------------------------------------------

    def recall(self, query: RecallQuery) -> list[RecalledMemory]:
        """Selectively recall relevant slots based on query."""
        return self.recaller.recall(query, self.slots, self.step_count)

    def get_slot(self, slot_id: int) -> MemorySlot | None:
        """Get a specific slot by ID."""
        if 0 <= slot_id < len(self.slots):
            return self.slots[slot_id]
        return None

    def get_active_slots(self) -> list[MemorySlot]:
        """Get all non-empty slots sorted by activation (highest first)."""
        active = [s for s in self.slots if not s.is_empty]
        active.sort(key=lambda s: s.activation, reverse=True)
        return active

    # -- Snapshot / Restore --------------------------------------------------

    def snapshot(self, session_id: str = "") -> StateSnapshot:
        """Serialize the full state for persistence to L3."""
        slots_data = [s.model_dump() for s in self.slots]
        return StateSnapshot(
            step_count=self.step_count,
            slots_json=json.dumps(slots_data, ensure_ascii=False),
            session_id=session_id,
        )

    def restore(self, snapshot: StateSnapshot, elapsed_seconds: float = 0) -> None:
        """Restore state from a snapshot, with optional cold-start decay."""
        slots_data = json.loads(snapshot.slots_json)
        self.slots = [MemorySlot(**d) for d in slots_data]

        # Pad or trim to match current config
        while len(self.slots) < self.config.slot_count:
            self.slots.append(MemorySlot(id=len(self.slots)))
        self.slots = self.slots[: self.config.slot_count]

        self.evolver._step_count = snapshot.step_count

        # Apply cold-start decay
        if elapsed_seconds > 0:
            self.evolver.apply_cold_start_decay(self.slots, elapsed_seconds)

    # -- Maintenance ---------------------------------------------------------

    async def force_evict(self, slot_id: int) -> MemorySlot | None:
        """Manually evict a specific slot (e.g., user-requested forget)."""
        if 0 <= slot_id < len(self.slots):
            slot = self.slots[slot_id]
            if not slot.is_empty:
                copy = slot.model_copy(deep=True)
                slot.state = ""
                slot.topic = ""
                slot.entities = []
                slot.activation = 0.0
                slot.embedding = None
                slot.source_refs = []
                return copy
        return None
