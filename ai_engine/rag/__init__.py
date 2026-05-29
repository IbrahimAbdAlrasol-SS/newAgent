"""
RAG (Retrieval-Augmented Generation) module.

Provides vector storage and retrieval for multi-tenant product catalogs.
"""

from .catalog_extractor import CatalogExtractor, ExtractedProduct
from .vector_store import MultiTenantVectorStore

__all__ = ["MultiTenantVectorStore", "CatalogExtractor", "ExtractedProduct"]
