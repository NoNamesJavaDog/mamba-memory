"""State Evolver — SSM discrete state transition with time-aware decay.

Implements the core SSM recurrence at each time step:

    h(t) = Ā · h(t-1) + B̄ · x(t)

Where:
    Ā  = decay matrix  (all slots decay — step-based + time-based)
    B̄·x = gated write  (selected slot receives new information)

Decay model:
    Step decay:  activation *= base_decay_rate  (per interaction step)
    Time decay:  activation *= 2^(-Δt / halflife)  (Ebbinghaus-inspired)
    Rehearsal:   halflife *= (1 + 0.5 * rehearsal_count)  (spaced repetition)

The combined decay means:
    - Unused memories fade with both steps AND wall-clock time
    - Frequently recalled memories have longer half-lives
    - A memory recalled 5 times has 3.5x the half-life of one never recalled
"""

from __future__ import annotations

import math
import time as _time
from typing import Protocol

from mamba_memory.core.types import (
    GateDecision,
    L2Config,
    MemorySlot,
    WriteMode,
)


class SlotMerger(Protocol):
    """Interface for merging new content into an existing slot state."""

    async def merge(self, existing: str, new_content: str, mode: WriteMode) -> str:
        """Merge new_content into existing state according to mode."""
        ...


class EvictionCallback(Protocol):
    """Called when a slot is evicted — typically persists it to L3."""

    async def on_evict(self, slot: MemorySlot) -> None: ...


class Evolver:
    """Drives the state vector forward one step per interaction.

    Each step:
    1. Global decay — all slots lose activation
    2. Selective write — gate-selected slot receives new content
    3. Co-activation — related slots get an activation bump
    4. Eviction — dead slots are flushed to L3 and cleared
    """

    def __init__(
        self,
        config: L2Config | None = None,
        merger: SlotMerger | None = None,
        on_evict: EvictionCallback | None = None,
    ) -> None:
        self.config = config or L2Config()
        self._merger = merger
        self._on_evict = on_evict
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    async def step(
        self,
        slots: list[MemorySlot],
        gate: GateDecision | None,
        content: str = "",
        entities: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> list[MemorySlot]:
        """Execute one evolution step. Returns evicted slots (if any)."""
        self._step_count += 1
        evicted: list[MemorySlot] = []

        # 1. Global decay
        self._apply_decay(slots)

        # 2. Selective write
        if gate and gate.should_write:
            await self._apply_write(slots, gate, content, entities or [], embedding)

        # 3. Co-activation
        if gate and gate.co_activations:
            self._apply_co_activation(slots, gate)

        # 4. Eviction check
        evicted = await self._check_eviction(slots)

        return evicted

    def apply_cold_start_decay(self, slots: list[MemorySlot], elapsed_seconds: float) -> None:
        """Apply catch-up decay after system restart.

        Simulates the decay that would have happened during downtime.
        Uses average step duration to estimate missed steps.
        """
        avg_step_duration = 30.0  # assume ~30s per interaction on average
        missed_steps = int(elapsed_seconds / avg_step_duration)
        if missed_steps <= 0:
            return

        decay_factor = self.config.base_decay_rate ** missed_steps
        for slot in slots:
            if not slot.is_empty:
                slot.activation *= decay_factor
                slot.activation = max(slot.activation, 0.0)

    # -- Internal steps ------------------------------------------------------

    def _apply_decay(self, slots: list[MemorySlot]) -> None:
        """All slots lose activation via combined step + time decay.

        Step decay:  activation *= base_decay_rate
        Time decay:  activation *= 2^(-Δt / effective_halflife)
        Effective halflife = base_halflife * (1 + 0.5 * rehearsal_count)

        The time decay implements an Ebbinghaus-like forgetting curve where
        rehearsal (repeated recall) strengthens memory retention.
        """
        now = _time.time()
        for slot in slots:
            if slot.is_empty:
                continue

            # Step-based decay (always applied)
            slot.activation *= self.config.base_decay_rate

            # Time-based decay (Ebbinghaus curve)
            if self.config.time_decay_enabled and slot.last_active_time > 0:
                elapsed = now - slot.last_active_time
                if elapsed > 0:
                    # Rehearsal extends half-life: more recalls = slower forgetting
                    effective_halflife = self.config.time_decay_halflife * (
                        1.0 + 0.5 * slot.rehearsal_count
                    )
                    time_factor = math.pow(2.0, -elapsed / effective_halflife)
                    slot.activation *= time_factor

            slot.activation = max(slot.activation, 0.0)

    async def _apply_write(
        self,
        slots: list[MemorySlot],
        gate: GateDecision,
        content: str,
        entities: list[str],
        embedding: list[float] | None,
    ) -> None:
        """Write content to the target slot based on gate decision."""
        target_id = gate.target_slot

        # Find or validate target slot
        target: MemorySlot | None = None
        for slot in slots:
            if slot.id == target_id:
                target = slot
                break

        if target is None:
            return

        # Apply write based on mode
        if gate.write_mode == WriteMode.OVERWRITE:
            target.state = content
            target.entities = entities
        elif gate.write_mode == WriteMode.UPDATE:
            target.state = await self._merge_content(target.state, content, WriteMode.UPDATE)
            target.entities = list(set(target.entities + entities))
        elif gate.write_mode == WriteMode.MERGE:
            target.state = await self._merge_content(target.state, content, WriteMode.MERGE)
            target.entities = list(set(target.entities + entities))

        # Update metadata
        if not target.topic and content:
            # Auto-generate topic from first ~50 chars
            target.topic = content[:50].strip()

        if embedding is not None:
            target.embedding = embedding

        # Activation boost
        boost = gate.gate_strength * self.config.activation_boost
        target.activation = min(1.0, target.activation + boost)
        target.last_active_step = self._step_count
        target.last_write_step = self._step_count
        target.last_active_time = _time.time()

        if target.created_step == 0:
            target.created_step = self._step_count

    def _apply_co_activation(self, slots: list[MemorySlot], gate: GateDecision) -> None:
        """Bump activation on slots related to the written slot."""
        co_set = set(gate.co_activations)
        bump = 0.1 * gate.gate_strength
        now = _time.time()
        for slot in slots:
            if slot.id in co_set and not slot.is_empty:
                slot.activation = min(1.0, slot.activation * (1.0 + bump))
                slot.last_active_step = self._step_count
                slot.last_active_time = now

    async def _check_eviction(self, slots: list[MemorySlot]) -> list[MemorySlot]:
        """Evict slots below the activation threshold."""
        evicted: list[MemorySlot] = []
        for slot in slots:
            if slot.is_empty:
                continue
            if slot.activation < self.config.eviction_threshold:
                evicted.append(slot.model_copy(deep=True))
                if self._on_evict:
                    await self._on_evict.on_evict(slot)
                # Clear the slot for reuse
                slot.state = ""
                slot.topic = ""
                slot.entities = []
                slot.activation = 0.0
                slot.embedding = None
                slot.source_refs = []
        return evicted

    async def _merge_content(self, existing: str, new: str, mode: WriteMode) -> str:
        """Merge content using the configured merger or simple fallback."""
        if not existing:
            return new

        if self._merger:
            return await self._merger.merge(existing, new, mode)

        # Fallback: simple append with separator
        merged = f"{existing}\n---\n{new}"
        # Truncate if too long (rough guard)
        if len(merged) > self.config.slot_max_tokens * 4:  # ~4 chars per token
            merged = merged[-(self.config.slot_max_tokens * 4) :]
        return merged
