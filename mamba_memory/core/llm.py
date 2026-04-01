"""LLM-powered compressor and merger.

Replaces the naive fallback implementations with actual LLM calls for:
  1. L1 Compressor — compress conversation turns into structured summaries
  2. L2 SlotMerger — intelligently merge/update/overwrite slot states

Supports multiple backends:
  - OpenAI-compatible API (GPT, DeepSeek, Moonshot, local vLLM, etc.)
  - Anthropic Claude
  - Any endpoint that speaks the OpenAI chat completions format

The LLM is called with structured prompts and expected to return JSON.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from mamba_memory.core.l2.evolver import SlotMerger
from mamba_memory.core.types import (
    CompressedSegment,
    ConversationTurn,
    WriteMode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract LLM backend
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    """Minimal interface for chat completion calls."""

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """Send chat messages and return the assistant's reply text."""
        ...


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible chat completions (works with any compatible endpoint)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4.1-mini",
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self._model = model

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e

        self._client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self._model = model

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        # Anthropic requires separating system message
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system or "You are a memory compression assistant.",
            messages=chat_msgs,
            temperature=temperature,
        )
        return resp.content[0].text


class GeminiBackend(LLMBackend):
    """Google Gemini backend via google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError("google-genai required: pip install google-genai") from e

        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self._model = model

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        from google.genai import types

        # Build Gemini contents from chat messages
        system_instruction = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            elif m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=m["content"])]))

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return resp.text or ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMPRESS_SYSTEM = """You are a memory compression engine. Your job is to compress conversation turns into a concise, information-dense summary.

Rules:
1. PRESERVE: decisions, preferences, facts, action items, corrections, names, numbers, configurations
2. DISCARD: greetings, filler, repetition, verbose explanations, politeness
3. EXTRACT: key entities (people, projects, tools, concepts) as tags
4. Keep the summary factual and third-person ("the user decided..." not "I decided...")
5. Target approximately {target_tokens} tokens

Output ONLY valid JSON:
{{"summary": "...", "entities": ["..."]}}"""

COMPRESS_USER = """Compress these conversation turns:

{turns}"""

MERGE_SYSTEM = """You are a memory state manager. You maintain concise cognitive state entries about a user.

Your job depends on the mode:

- UPDATE: Append the new information to the existing state. Keep both old and new facts.
- OVERWRITE: The new information supersedes the old. Replace outdated facts but note what changed.
- MERGE: Combine old and new, deduplicating overlapping information. Produce the most concise accurate result.

Rules:
1. Output ONLY the updated state text, no explanation
2. Be concise — every word must carry information
3. Resolve contradictions: new information wins
4. Keep entity names, numbers, and configurations exact
5. Maximum ~{max_tokens} tokens"""

MERGE_USER = """Mode: {mode}

Existing state:
{existing}

New information:
{new_content}

Output the merged state:"""


# ---------------------------------------------------------------------------
# LLM Compressor (for L1)
# ---------------------------------------------------------------------------


class LLMCompressor:
    """LLM-powered conversation compressor for L1 session layer.

    Implements the ``Compressor`` protocol from ``l1.session``.

    Usage::

        backend = OpenAIBackend(model="gpt-4.1-mini")
        compressor = LLMCompressor(backend)
        session.set_compressor(compressor)
    """

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    async def compress(
        self, turns: list[ConversationTurn], target_tokens: int
    ) -> CompressedSegment:
        """Compress conversation turns into a structured summary via LLM."""
        if not turns:
            return CompressedSegment(
                time_range=(0, 0), turn_count=0, summary="", tokens=0
            )

        # Format turns for the prompt
        turns_text = "\n".join(
            f"[{t.role}] {t.content}" for t in turns
        )

        system_msg = COMPRESS_SYSTEM.format(target_tokens=target_tokens)
        user_msg = COMPRESS_USER.format(turns=turns_text)

        try:
            raw = await self._backend.chat([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])

            parsed = _parse_json(raw)
            summary = parsed.get("summary", raw)
            entities = parsed.get("entities", [])

        except Exception as e:
            logger.warning("LLM compression failed, using fallback: %s", e)
            # Fallback to simple truncation
            summary = " | ".join(f"{t.role}: {t.content[:100]}" for t in turns)
            if len(summary) > 500:
                summary = summary[:500] + "..."
            entities = []

        return CompressedSegment(
            time_range=(turns[0].timestamp, turns[-1].timestamp),
            turn_count=len(turns),
            summary=summary,
            entities=entities,
            tokens=_estimate_tokens(summary),
        )


# ---------------------------------------------------------------------------
# LLM Slot Merger (for L2)
# ---------------------------------------------------------------------------


class LLMSlotMerger:
    """LLM-powered slot state merger for L2 state layer.

    Implements the ``SlotMerger`` protocol from ``l2.evolver``.

    Usage::

        backend = OpenAIBackend(model="gpt-4.1-mini")
        merger = LLMSlotMerger(backend)
        state_layer = StateLayer(config, merger=merger)
    """

    def __init__(self, backend: LLMBackend, max_tokens: int = 150) -> None:
        self._backend = backend
        self._max_tokens = max_tokens

    async def merge(self, existing: str, new_content: str, mode: WriteMode) -> str:
        """Merge new content into existing slot state via LLM."""
        if not existing:
            return new_content

        system_msg = MERGE_SYSTEM.format(max_tokens=self._max_tokens)
        user_msg = MERGE_USER.format(
            mode=mode.value.upper(),
            existing=existing,
            new_content=new_content,
        )

        try:
            result = await self._backend.chat([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])
            return result.strip()

        except Exception as e:
            logger.warning("LLM merge failed, using fallback: %s", e)
            # Fallback: simple append
            merged = f"{existing}\n---\n{new_content}"
            if len(merged) > self._max_tokens * 4:
                merged = merged[-(self._max_tokens * 4):]
            return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_backend(
    provider: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBackend:
    """Create an LLM backend.

    Args:
        provider: 'google', 'openai', 'anthropic', or 'auto'
        model: Model name (uses sensible defaults if omitted)
        api_key: API key (falls back to env vars)
        base_url: Custom endpoint URL (for compatible APIs like DeepSeek, vLLM)

    'auto' checks env vars: GOOGLE_API_KEY → Gemini, ANTHROPIC → Claude, else OpenAI.
    """
    if provider == "google":
        return GeminiBackend(api_key=api_key, model=model or "gemini-2.0-flash")

    if provider == "anthropic":
        return AnthropicBackend(api_key=api_key, model=model or "claude-sonnet-4-6")

    if provider == "openai":
        return OpenAIBackend(api_key=api_key, base_url=base_url, model=model or "gpt-4.1-mini")

    # Auto-detect
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            return GeminiBackend(model=model or "gemini-2.0-flash")
        except ImportError:
            pass

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend(model=model or "claude-sonnet-4-6")
        except ImportError:
            pass

    try:
        return OpenAIBackend(
            api_key=api_key, base_url=base_url, model=model or "gpt-4.1-mini",
        )
    except ImportError:
        pass

    raise ImportError(
        "No LLM backend available. Install one of: "
        "google-genai, openai, anthropic"
    )


def create_compressor(
    provider: str = "auto",
    model: str | None = None,
    **kwargs: Any,
) -> LLMCompressor:
    """Convenience: create an LLMCompressor with one call."""
    backend = create_llm_backend(provider, model, **kwargs)
    return LLMCompressor(backend)


def create_merger(
    provider: str = "auto",
    model: str | None = None,
    max_tokens: int = 150,
    **kwargs: Any,
) -> LLMSlotMerger:
    """Convenience: create an LLMSlotMerger with one call."""
    backend = create_llm_backend(provider, model, **kwargs)
    return LLMSlotMerger(backend, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Try finding first { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {}


def _estimate_tokens(text: str) -> int:
    cjk = sum(1 for c in text if ord(c) > 0x2E80)
    ascii_chars = len(text) - cjk
    return int(cjk * 1.5 + ascii_chars * 0.25)
