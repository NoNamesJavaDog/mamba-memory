"""L1 Session Layer — sliding window + compressed segments.

Keeps the last K conversation turns at full fidelity.
When the window overflows, the oldest turns are compressed into summaries.
When compressed segments overflow, they are pushed to L2 via the gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from mamba_memory.core.types import (
    CompressedSegment,
    ConversationTurn,
    L1Config,
)

if TYPE_CHECKING:
    pass


class Compressor(Protocol):
    """Interface for compressing conversation turns into summaries."""

    async def compress(self, turns: list[ConversationTurn], target_tokens: int) -> CompressedSegment:
        """Compress a batch of turns into a single summary segment."""
        ...


class SessionLayer:
    """L1: Recent conversation window + compressed history segments.

    The session layer is the fastest tier — pure in-memory, sub-millisecond
    read/write. It holds:
      - ``window``: the last K turns at full fidelity
      - ``compressed``: summaries of older turns that slid out of the window

    When compressed segments exceed the limit, the oldest segments are returned
    as overflow for the L2 gate to evaluate.
    """

    def __init__(self, config: L1Config | None = None) -> None:
        self.config = config or L1Config()
        self.window: list[ConversationTurn] = []
        self.compressed: list[CompressedSegment] = []
        self._compressor: Compressor | None = None

    def set_compressor(self, compressor: Compressor) -> None:
        self._compressor = compressor

    # -- Public API ----------------------------------------------------------

    async def add_turn(self, turn: ConversationTurn) -> list[CompressedSegment]:
        """Add a conversation turn. Returns overflow segments (if any) for L2 gate.

        Flow:
        1. Push turn into window
        2. If window exceeds size, pop oldest turns and compress
        3. If compressed list exceeds limit, pop oldest as overflow
        4. Return overflow segments → caller feeds them to L2 gate
        """
        self.window.append(turn)
        overflow: list[CompressedSegment] = []

        if len(self.window) > self.config.window_size:
            overflow = await self._compress_overflow()

        return overflow

    def get_recent_turns(self, limit: int | None = None) -> list[ConversationTurn]:
        """Get the most recent turns from the window."""
        if limit is None:
            return list(self.window)
        return list(self.window[-limit:])

    def get_compressed(self) -> list[CompressedSegment]:
        """Get all compressed segments (newest first)."""
        return list(self.compressed)

    def get_full_context(self, max_tokens: int | None = None) -> list[str]:
        """Build context strings from compressed segments + window turns.

        Returns a list of text blocks ordered oldest-to-newest:
        compressed summaries first, then raw window turns.
        """
        blocks: list[str] = []
        token_budget = max_tokens or float("inf")
        used = 0

        # Compressed segments (oldest first for chronological order)
        for seg in reversed(self.compressed):
            if used + seg.tokens > token_budget:
                break
            blocks.append(f"[Summary] {seg.summary}")
            used += seg.tokens

        # Window turns
        for turn in self.window:
            if used + turn.tokens > token_budget:
                break
            blocks.append(f"[{turn.role}] {turn.content}")
            used += turn.tokens

        return blocks

    def clear(self) -> None:
        """Clear all session state."""
        self.window.clear()
        self.compressed.clear()

    @property
    def total_tokens(self) -> int:
        window_tokens = sum(t.tokens for t in self.window)
        compressed_tokens = sum(s.tokens for s in self.compressed)
        return window_tokens + compressed_tokens

    # -- Internal ------------------------------------------------------------

    async def _compress_overflow(self) -> list[CompressedSegment]:
        """Compress excess window turns and manage compressed segment overflow.

        When compressed segments overflow, instead of pushing each one individually
        to L2, we merge multiple overflow segments into a single higher-level
        summary (recursive summarization chain). This produces denser, more
        informative segments for L2 to evaluate.

        Chain: turns → L1 segment → [overflow] → merge N segments → super-segment → L2
        """
        excess = len(self.window) - self.config.window_size
        if excess <= 0:
            return []

        evicted = self.window[:excess]
        self.window = self.window[excess:]

        # Compress evicted turns into a segment
        segment = await self._do_compress(evicted)
        self.compressed.insert(0, segment)  # newest first

        # Collect overflow segments
        raw_overflow: list[CompressedSegment] = []
        while len(self.compressed) > self.config.max_compressed_segments:
            raw_overflow.append(self.compressed.pop())  # pop oldest

        if not raw_overflow:
            return []

        # Recursive merge: combine multiple overflow segments into one
        if len(raw_overflow) >= 2:
            merged = await self._merge_segments(raw_overflow)
            return [merged]

        return raw_overflow

    async def _merge_segments(self, segments: list[CompressedSegment]) -> CompressedSegment:
        """Merge multiple compressed segments into a single higher-level summary.

        This is the recursive step in the summarization chain:
        summary-of-summaries, preserving only the most important facts
        across a longer time span.
        """
        if len(segments) == 1:
            return segments[0]

        # Build pseudo-turns from segment summaries for re-compression
        pseudo_turns = [
            ConversationTurn(
                role="system",
                content=f"[Summary of {seg.turn_count} turns] {seg.summary}",
                timestamp=seg.time_range[0],
                tags=seg.entities,
                tokens=seg.tokens,
            )
            for seg in segments
        ]

        merged = await self._do_compress(pseudo_turns)

        # Preserve the full time range across all merged segments
        all_starts = [s.time_range[0] for s in segments]
        all_ends = [s.time_range[1] for s in segments]
        merged.time_range = (min(all_starts), max(all_ends))
        merged.turn_count = sum(s.turn_count for s in segments)

        # Merge entities from all segments
        all_entities: set[str] = set()
        for s in segments:
            all_entities.update(s.entities)
        merged.entities = list(all_entities)

        return merged

    async def _do_compress(self, turns: list[ConversationTurn]) -> CompressedSegment:
        """Compress a batch of turns using the configured compressor or fallback."""
        if self._compressor is not None:
            return await self._compressor.compress(turns, self.config.segment_target_tokens)

        # Fallback: simple concatenation (no LLM)
        return _fallback_compress(turns)


def _fallback_compress(turns: list[ConversationTurn]) -> CompressedSegment:
    """Structured fallback compression without LLM.

    Strategy:
    1. Extract key facts (sentences with structured data / decision signals)
    2. Auto-extract entities from text
    3. Build structured summary from facts, not raw truncation
    4. Merge with any tags already on the turns
    """
    if not turns:
        return CompressedSegment(time_range=(0, 0), turn_count=0, summary="", tokens=0)

    from mamba_memory.core.text import compress_turns_structured, extract_entities_simple

    turn_pairs = [(t.role, t.content) for t in turns]
    summary, auto_entities = compress_turns_structured(turn_pairs, max_chars=600)

    # Merge auto-extracted entities with pre-tagged entities
    all_entities = list(set(auto_entities))
    for t in turns:
        all_entities.extend(t.tags)
    all_entities = list(set(all_entities))

    total_tokens = sum(t.tokens for t in turns)

    return CompressedSegment(
        time_range=(turns[0].timestamp, turns[-1].timestamp),
        turn_count=len(turns),
        summary=summary,
        entities=list(set(all_entities)),
        tokens=min(total_tokens, 200),
    )
