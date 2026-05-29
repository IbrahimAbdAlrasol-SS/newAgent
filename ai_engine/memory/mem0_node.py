"""
Mem0 Long-Term Memory Node.

Integrates mem0 for persistent cross-session customer memory.
Degrades gracefully if mem0 is not installed.

Usage:
    node = Mem0MemoryNode()
    # Before context packing: inject relevant memories into state
    updates = await node.search(state)
    # After response generation: persist turn
    await node.add(state, response_content)

Memory is namespaced per tenant to prevent cross-tenant data leakage:
    mem0 user_id = "{tenant_id}:{social_user_id}"
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


def _get_mem0_client() -> Any:
    """Lazy-init mem0 Memory client. Returns None if mem0 unavailable."""
    try:
        from mem0 import Memory
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": os.environ.get("MEM0_LLM_MODEL", "gpt-4o-mini"),
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": os.environ.get("MEM0_EMBED_MODEL", "text-embedding-3-small"),
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": os.environ.get("QDRANT_HOST", "localhost"),
                    "port": int(os.environ.get("QDRANT_PORT", "6333")),
                    "collection_name": "agent_memory",
                },
            },
        }
        return Memory.from_config(config)
    except ImportError:
        logger.warning("mem0 not installed — long-term memory disabled. Install: pip install mem0ai")
        return None
    except Exception as exc:
        logger.warning(f"mem0 init failed (non-fatal): {exc}")
        return None


class Mem0MemoryNode:
    """
    Long-term memory integration via mem0.

    Searches customer memories to enrich context before response generation,
    and persists new facts from each conversation turn.
    """

    _client: Any = None
    _initialized: bool = False

    @classmethod
    def _get_client(cls) -> Any:
        if not cls._initialized:
            cls._client = _get_mem0_client()
            cls._initialized = True
        return cls._client

    async def search(self, state) -> dict:
        """
        Search mem0 for memories relevant to the current user message.
        Returns state-update dict enriching customer_profile with memories.
        """
        client = self._get_client()
        if client is None:
            return {}

        tenant_id = getattr(state, "tenant_id", None)
        user_id = getattr(state, "social_user_id", None)
        if not tenant_id or not user_id:
            return {}

        mem_user_id = f"{tenant_id}:{user_id}"

        messages = getattr(state, "messages", [])
        query = ""
        for msg in reversed(messages):
            if getattr(msg, "role", None) == "user":
                query = getattr(msg, "content", "")
                break

        if not query:
            return {}

        try:
            results = client.search(query, user_id=mem_user_id, limit=5)
            if not results:
                return {}

            memories_text = "\n".join(
                f"- {r['memory']}" for r in results if r.get("memory")
            )
            current_profile = getattr(state, "customer_profile", None) or {}
            updated_profile = {**current_profile, "long_term_memories": memories_text}

            logger.info(f"🧠 MEM0 — retrieved {len(results)} memories for {user_id}")
            return {"customer_profile": updated_profile}

        except Exception as exc:
            logger.warning(f"mem0 search failed (non-fatal): {exc}")
            return {}

    async def add(self, state, response_content: str) -> None:
        """
        Persist the current turn to mem0 (fire-and-forget).
        Called after response generation. Failures are non-fatal.
        """
        client = self._get_client()
        if client is None:
            return

        tenant_id = getattr(state, "tenant_id", None)
        user_id = getattr(state, "social_user_id", None)
        if not tenant_id or not user_id:
            return

        mem_user_id = f"{tenant_id}:{user_id}"

        messages = getattr(state, "messages", [])
        last_user = ""
        for msg in reversed(messages):
            if getattr(msg, "role", None) == "user":
                last_user = getattr(msg, "content", "")
                break

        if not last_user:
            return

        try:
            client.add(
                [
                    {"role": "user", "content": last_user},
                    {"role": "assistant", "content": response_content},
                ],
                user_id=mem_user_id,
            )
            logger.debug(f"🧠 MEM0 — persisted turn for {user_id}")
        except Exception as exc:
            logger.warning(f"mem0 add failed (non-fatal): {exc}")


__all__ = ["Mem0MemoryNode"]
