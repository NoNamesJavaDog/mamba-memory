"""Embedding provider abstraction.

Supports multiple backends:
  - OpenAI API (text-embedding-3-small/large)
  - Local sentence-transformers models
  - No-op dummy (for testing / no-embedding mode)

The provider interface is intentionally minimal so custom backends
can be plugged in easily.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding vector dimension."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default: sequential calls to embed()."""
        return [await self.embed(t) for t in texts]


class DummyEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embeddings for testing (no ML model needed)."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        # Expand hash to fill dim
        repeats = (self._dim * 4 // len(h)) + 1
        raw = (h * repeats)[: self._dim * 4]
        # Convert bytes to floats in [-1, 1]
        values = [((b / 255.0) * 2 - 1) for b in raw[: self._dim]]
        # Normalize
        norm = sum(v * v for v in values) ** 0.5
        if norm > 0:
            values = [v / norm for v in values]
        return values


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings via the openai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dim: int = 256,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install mamba-memory[embeddings]"
            ) from e

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            input=text,
            model=self._model,
            dimensions=self._dim,
        )
        return resp.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dim,
        )
        return [d.embedding for d in resp.data]


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embeddings via the google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        dim: int = 768,
    ) -> None:
        try:
            from google import genai
            from google.genai import types  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "google-genai required: pip install google-genai"
            ) from e

        import os

        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        from google.genai import types

        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self._dim),
        )
        return list(result.embeddings[0].values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embeddings (runs on CPU/GPU)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required: pip install mamba-memory[embeddings]"
            ) from e

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()


def create_provider(
    provider_type: str = "auto",
    **kwargs,
) -> EmbeddingProvider:
    """Factory function to create an embedding provider.

    Args:
        provider_type: 'openai', 'google', 'local', 'dummy', or 'auto'
        **kwargs: Passed to the provider constructor

    'auto' tries google → openai → local → dummy.
    """
    if provider_type == "openai":
        return OpenAIEmbeddingProvider(**kwargs)
    if provider_type == "google":
        return GoogleEmbeddingProvider(**kwargs)
    if provider_type == "local":
        return LocalEmbeddingProvider(**kwargs)
    if provider_type == "dummy":
        return DummyEmbeddingProvider(**kwargs)

    # Auto: try google → openai → local → dummy
    import os

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            return GoogleEmbeddingProvider(**kwargs)
        except ImportError:
            pass

    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddingProvider(**kwargs)
        except ImportError:
            pass

    try:
        return LocalEmbeddingProvider(**kwargs)
    except ImportError:
        pass

    return DummyEmbeddingProvider(**kwargs)
