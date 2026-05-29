"""
Embeddings module for multilingual text embeddings.

Provides high-quality embeddings for Arabic and English text
using state-of-the-art models.
"""

from .embedding_service import EmbeddingService
from .factory import get_embedding_service
from .openai_embedding_service import OpenAIEmbeddingService
from .protocol import EmbeddingProtocol

__all__ = ["EmbeddingService", "OpenAIEmbeddingService", "get_embedding_service", "EmbeddingProtocol"]
