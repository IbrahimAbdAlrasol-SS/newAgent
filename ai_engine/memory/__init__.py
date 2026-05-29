"""
ai_engine/memory — Memory Layer (Layer 2: Core Capabilities)

هذه الحزمة تمثّل طبقة الذاكرة بأنواعها:

- memory_guard.py   → Layer 3 Governance: يمنع حقن الذاكرة
- (مستقبلاً) long_term.py  → واجهة الذاكرة طويلة الأمد
- (مستقبلاً) short_term.py → واجهة Redis session
"""

from ai_engine.memory.memory_guard import MemoryGuard, memory_guard

__all__ = ["MemoryGuard", "memory_guard"]
