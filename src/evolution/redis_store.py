"""Redis state layer for Auto-ChemInstruct evolution.

Provides atomic counters, archive storage, lineage tracking, and MAP-Elites
cell storage backed by Redis. Includes in-memory fallback for development.

Key schema:
    autochem:{session}:hypothesis:{id}   → ReactionHypothesis JSON
    autochem:{session}:result:{id}       → VerificationResult JSON
    autochem:{session}:cell:{x}:{y}:{z}  → Island archive cell (hash)
    autochem:{session}:lineage:{id}      → Parent→child links (list)
    autochem:{session}:counter:generated → Atomic counter
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import redis  # type: ignore[import-untyped]
from loguru import logger


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    key_prefix: str = "autochem"
    ttl_seconds: int = 604800  # 7 days


class ReactionArchive:
    """Redis-backed archive for reaction hypotheses, results, and evolution state."""

    def __init__(
        self,
        session_id: UUID | None = None,
        config: RedisConfig | None = None,
        redis_client: redis.Redis | None = None,
        use_fallback: bool = False,
    ):
        self.config = config or RedisConfig()
        self.session_id = session_id or uuid4()
        self._use_fallback = use_fallback
        self._fallback: dict[str, Any] = {}

        if not use_fallback:
            try:
                self._redis = redis_client or redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    decode_responses=True,
                )
                self._redis.ping()
                logger.info("Connected to Redis at {}:{}", self.config.host, self.config.port)
            except (redis.ConnectionError, redis.RedisError) as e:
                logger.warning("Redis unavailable ({}), using in-memory fallback", e)
                self._use_fallback = True
                self._fallback = {}

    def _key(self, *parts: str) -> str:
        return f"{self.config.key_prefix}:{self.session_id}:" + ":".join(parts)

    # ── Hash operations (for hypothesis/result storage) ──

    def set_json(self, entity_type: str, entity_id: str, data: dict) -> None:
        key = self._key(entity_type, entity_id)
        if self._use_fallback:
            self._fallback[key] = json.dumps(data)
        else:
            self._redis.set(key, json.dumps(data), ex=self.config.ttl_seconds)

    def get_json(self, entity_type: str, entity_id: str) -> dict | None:
        key = self._key(entity_type, entity_id)
        if self._use_fallback:
            raw = self._fallback.get(key)
            return json.loads(raw) if raw else None
        raw = self._redis.get(key)
        return json.loads(raw) if raw else None

    def delete(self, entity_type: str, entity_id: str) -> None:
        key = self._key(entity_type, entity_id)
        if self._use_fallback:
            self._fallback.pop(key, None)
        else:
            self._redis.delete(key)

    # ── Atomic counters ──

    def increment(self, counter_name: str, amount: int = 1) -> int:
        key = self._key("counter", counter_name)
        if self._use_fallback:
            current = int(self._fallback.get(key, 0))
            self._fallback[key] = current + amount
            return self._fallback[key]
        return int(self._redis.incrby(key, amount))

    def get_counter(self, counter_name: str) -> int:
        key = self._key("counter", counter_name)
        if self._use_fallback:
            return int(self._fallback.get(key, 0))
        return int(self._redis.get(key) or 0)

    # ── MAP-Elites cell storage ──

    def set_cell_elite(
        self,
        island_id: str,
        cell_coords: tuple[int, int, int],
        hypothesis_id: str,
        fitness: float,
    ) -> None:
        x, y, z = cell_coords
        key = self._key("island", island_id, "cell", str(x), str(y), str(z))
        data = json.dumps({"hypothesis_id": hypothesis_id, "fitness": fitness})
        if self._use_fallback:
            self._fallback[key] = data
        else:
            self._redis.set(key, data, ex=self.config.ttl_seconds)

    def get_cell_elite(self, island_id: str, cell_coords: tuple[int, int, int]) -> dict | None:
        x, y, z = cell_coords
        key = self._key("island", island_id, "cell", str(x), str(y), str(z))
        if self._use_fallback:
            raw = self._fallback.get(key)
            return json.loads(raw) if raw else None
        raw = self._redis.get(key)
        return json.loads(raw) if raw else None

    def get_all_cells(self, island_id: str) -> dict[tuple[int, int, int], dict]:
        pattern = self._key("island", island_id, "cell", "*", "*", "*")
        result: dict[tuple[int, int, int], dict] = {}
        if self._use_fallback:
            for key, val in self._fallback.items():
                if key.startswith(self._key("island", island_id, "cell")):
                    parts = key.split(":")
                    coords = tuple(int(p) for p in parts[-3:])
                    result[coords] = json.loads(val)
        else:
            for key in self._redis.scan_iter(match=pattern):
                val = self._redis.get(key)
                if val:
                    parts = key.split(":")
                    coords = tuple(int(p) for p in parts[-3:])
                    result[coords] = json.loads(val)
        return result

    def cell_count(self, island_id: str) -> int:
        return len(self.get_all_cells(island_id))

    # ── Lineage tracking ──

    def add_child(self, parent_id: str, child_id: str) -> None:
        key = self._key("lineage", parent_id)
        if self._use_fallback:
            children = json.loads(self._fallback.get(key, "[]"))
            children.append(child_id)
            self._fallback[key] = json.dumps(children)
        else:
            self._redis.rpush(key, child_id)
            self._redis.expire(key, self.config.ttl_seconds)

    def get_children(self, parent_id: str) -> list[str]:
        key = self._key("lineage", parent_id)
        if self._use_fallback:
            return json.loads(self._fallback.get(key, "[]"))
        return self._redis.lrange(key, 0, -1)

    def get_ancestors(self, hypothesis_id: str) -> list[str]:
        """Walk up the lineage tree to find all ancestors."""
        ancestors: list[str] = []
        # Reverse lookup: find which key has hypothesis_id as child
        pattern = self._key("lineage", "*")
        current = hypothesis_id
        max_depth = 50
        for _ in range(max_depth):
            found = None
            if self._use_fallback:
                for key in self._fallback:
                    if key.startswith(self._key("lineage")):
                        children = json.loads(self._fallback.get(key, "[]"))
                        if current in children:
                            found = key.split(":")[-1]
                            break
            else:
                for key in self._redis.scan_iter(match=pattern):
                    children = self._redis.lrange(key, 0, -1)
                    if current in children:
                        found = key.split(":")[-1]
                        break
            if found is None:
                break
            ancestors.insert(0, found)
            current = found
        return ancestors

    # ── Session management ──

    def clear_session(self) -> int:
        pattern = self._key("*")
        count = 0
        if self._use_fallback:
            to_delete = [k for k in self._fallback if k.startswith(self._key(""))]
            for k in to_delete:
                del self._fallback[k]
            count = len(to_delete)
        else:
            keys = list(self._redis.scan_iter(match=pattern))
            if keys:
                count = self._redis.delete(*keys)
        logger.info("Cleared {} keys for session {}", count, self.session_id)
        return count

    def get_stats(self) -> dict:
        """Return summary statistics for the session."""
        return {
            "session_id": str(self.session_id),
            "hypotheses_generated": self.get_counter("generated"),
            "hypotheses_passed": self.get_counter("passed"),
            "hypotheses_failed": self.get_counter("failed"),
            "diversity_island_cells": self.cell_count("diversity_island"),
            "quality_island_cells": self.cell_count("quality_island"),
            "novelty_island_cells": self.cell_count("novelty_island"),
            "yield_island_cells": self.cell_count("yield_island"),
            "use_fallback": self._use_fallback,
        }
