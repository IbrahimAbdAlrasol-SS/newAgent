"""
PersonaLoader — single source of truth for AI prompts.

Loads markdown persona files and exposes named sections via `get(key)`.
This is a thin engine — all content lives in `personas/default.md`
(and optional per-tenant overrides in `personas/tenants/{tenant_id}.md`).

Section format (in MD):

    ## section.key

    section body text (can include $variables and {placeholders})

JSON sections must wrap their content in a ```json ... ``` fenced block.

Usage:
    from ai_engine.personas.loader import PersonaLoader

    persona = PersonaLoader.for_tenant(tenant_id)  # falls back to default
    raw    = persona.get("system.base.ar")
    block  = persona.render("intent.product_search", user_message=msg, products_list=p)
    cfg    = persona.get_json("personality.luxury")
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from string import Template
from typing import Any

logger = logging.getLogger(__name__)

_PERSONAS_DIR = Path(__file__).parent
_DEFAULT_FILE = _PERSONAS_DIR / "default.md"
_TENANTS_DIR = _PERSONAS_DIR / "tenants"

_SECTION_RE = re.compile(r"^## ([a-z0-9_.]+)\s*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


class PersonaLoader:
    """Parses an MD persona file into named sections."""

    _cache: dict[str, "PersonaLoader"] = {}
    _lock = threading.RLock()  # reentrant: for_tenant() may call default() while holding lock

    def __init__(self, path: Path, fallback: "PersonaLoader | None" = None) -> None:
        self.path = path
        self.fallback = fallback
        self._sections: dict[str, str] = {}
        self._meta: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------ loading

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Persona file not found: {self.path}")

        text = self.path.read_text(encoding="utf-8")

        # Optional YAML-ish frontmatter (we only need a few keys, so naive parse)
        fm_match = _FRONTMATTER_RE.match(text)
        if fm_match:
            for line in fm_match.group(1).splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    self._meta[k.strip()] = v.strip()
            text = text[fm_match.end():]

        # Split by `## key` headings
        parts = _SECTION_RE.split(text)
        # parts = [preamble, key1, body1, key2, body2, ...]
        for i in range(1, len(parts), 2):
            key = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            # Strip H1 comment headings (lines starting with "# ") embedded in body
            body = re.sub(r"^# .*$", "", body, flags=re.MULTILINE).strip()
            self._sections[key] = body

        logger.info(
            "PersonaLoader loaded %d sections from %s (meta=%s)",
            len(self._sections), self.path.name, self._meta,
        )

    # ------------------------------------------------------------------ lookup

    def has(self, key: str) -> bool:
        if key in self._sections:
            return True
        if self.fallback is not None:
            return self.fallback.has(key)
        return False

    def get(self, key: str, default: str | None = None) -> str:
        """Return raw section text. Falls back to default persona if missing."""
        if key in self._sections:
            return self._sections[key]
        if self.fallback is not None:
            return self.fallback.get(key, default=default)
        if default is not None:
            return default
        raise KeyError(f"Persona section not found: {key}")

    def get_json(self, key: str) -> Any:
        """Parse a section whose body is a ```json``` fenced block."""
        body = self.get(key)
        m = _JSON_FENCE_RE.search(body)
        payload = m.group(1) if m else body
        return json.loads(payload)

    def render(self, key: str, **vars: Any) -> str:
        """Render a section by substituting $var placeholders (safe — missing → blank-keeps-token)."""
        body = self.get(key)
        return Template(body).safe_substitute(**vars)

    def format(self, key: str, **vars: Any) -> str:
        """Render a section using str.format-style {placeholders}."""
        body = self.get(key)
        try:
            return body.format(**vars)
        except (KeyError, IndexError) as e:
            logger.warning("Persona.format missing variable for %s: %s", key, e)
            return body

    def variants(self, key: str) -> list[str]:
        """Return non-empty lines from a section as variant choices (used by fallbacks)."""
        body = self.get(key, default="")
        return [line.strip() for line in body.splitlines() if line.strip()]

    # ----------------------------------------------------------------- factory

    @classmethod
    def default(cls) -> "PersonaLoader":
        with cls._lock:
            if "__default__" not in cls._cache:
                cls._cache["__default__"] = cls(_DEFAULT_FILE)
            return cls._cache["__default__"]

    @classmethod
    def for_tenant(cls, tenant_id: str | None) -> "PersonaLoader":
        """Return tenant-specific persona, falling back to default."""
        if not tenant_id:
            return cls.default()
        key = f"tenant:{tenant_id}"
        with cls._lock:
            if key in cls._cache:
                return cls._cache[key]
            tenant_file = _TENANTS_DIR / f"{tenant_id}.md"
            if tenant_file.exists():
                loaded = cls(tenant_file, fallback=cls.default())
            else:
                loaded = cls.default()
            cls._cache[key] = loaded
            return loaded

    @classmethod
    def reload(cls) -> None:
        """Clear cache — call after editing MD files to pick up changes."""
        with cls._lock:
            cls._cache.clear()
        logger.info("PersonaLoader cache cleared")
