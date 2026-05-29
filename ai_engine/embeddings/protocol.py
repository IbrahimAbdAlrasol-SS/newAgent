"""
Embedding Protocol — shared interface for all embedding backends.

Both ``EmbeddingService`` (local sentence-transformers) and
``OpenAIEmbeddingService`` satisfy this protocol, enabling
type-safe dependency injection without tight coupling.

Usage:
    def build_rag(embedder: EmbeddingProtocol) -> RAGPipeline:
        ...
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Structural interface every embedding backend must satisfy."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query and return its vector."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts and return a list of vectors."""
        ...

    def health_check(self) -> bool:
        """Return *True* if the backend is healthy and ready to serve."""
        ...


__all__ = ["EmbeddingProtocol"]
