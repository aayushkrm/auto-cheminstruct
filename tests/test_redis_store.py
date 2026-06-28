"""Tests for Redis state layer with in-memory fallback."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.evolution.redis_store import ReactionArchive, RedisConfig


class TestReactionArchive:
    """Tests for the ReactionArchive class using in-memory fallback."""

    @pytest.fixture
    def archive(self):
        session_id = uuid4()
        config = RedisConfig(key_prefix="test")
        return ReactionArchive(
            session_id=session_id,
            config=config,
            use_fallback=True,
        )

    def test_initialization(self, archive):
        assert archive._use_fallback is True
        assert archive.session_id is not None

    def test_set_and_get_json(self, archive):
        archive.set_json("hypothesis", "h1", {"reactants": ["CC"], "products": ["CCC"]})
        result = archive.get_json("hypothesis", "h1")
        assert result is not None
        assert result["reactants"] == ["CC"]

    def test_get_missing_key(self, archive):
        result = archive.get_json("hypothesis", "nonexistent")
        assert result is None

    def test_delete(self, archive):
        archive.set_json("hypothesis", "h1", {"data": "test"})
        archive.delete("hypothesis", "h1")
        assert archive.get_json("hypothesis", "h1") is None

    def test_counter_increment(self, archive):
        assert archive.get_counter("generated") == 0
        archive.increment("generated")
        assert archive.get_counter("generated") == 1
        archive.increment("generated", 5)
        assert archive.get_counter("generated") == 6

    def test_multiple_counters(self, archive):
        archive.increment("generated", 10)
        archive.increment("passed", 7)
        archive.increment("failed", 3)
        assert archive.get_counter("generated") == 10
        assert archive.get_counter("passed") == 7
        assert archive.get_counter("failed") == 3

    def test_cell_elite_set_and_get(self, archive):
        archive.set_cell_elite("diversity_island", (1, 3, 7), "h_abc", 0.85)
        elite = archive.get_cell_elite("diversity_island", (1, 3, 7))
        assert elite is not None
        assert elite["hypothesis_id"] == "h_abc"
        assert elite["fitness"] == 0.85

    def test_cell_elite_overwrite(self, archive):
        archive.set_cell_elite("diversity_island", (0, 0, 0), "old", 0.5)
        archive.set_cell_elite("diversity_island", (0, 0, 0), "new", 0.9)
        elite = archive.get_cell_elite("diversity_island", (0, 0, 0))
        assert elite["hypothesis_id"] == "new"
        assert elite["fitness"] == 0.9

    def test_missing_cell(self, archive):
        elite = archive.get_cell_elite("diversity_island", (99, 99, 99))
        assert elite is None

    def test_cell_count_empty(self, archive):
        assert archive.cell_count("diversity_island") == 0

    def test_cell_count_multiple_islands(self, archive):
        archive.set_cell_elite("diversity_island", (0, 0, 0), "d1", 0.5)
        archive.set_cell_elite("diversity_island", (1, 1, 1), "d2", 0.6)
        archive.set_cell_elite("quality_island", (0, 0, 0), "q1", 0.8)
        assert archive.cell_count("diversity_island") == 2
        assert archive.cell_count("quality_island") == 1

    def test_lineage_add_child(self, archive):
        archive.add_child("parent_1", "child_a")
        archive.add_child("parent_1", "child_b")
        children = archive.get_children("parent_1")
        assert children == ["child_a", "child_b"]

    def test_lineage_empty(self, archive):
        assert archive.get_children("orphan") == []

    def test_lineage_ancestors_single_level(self, archive):
        archive.add_child("grandparent", "parent")
        archive.add_child("parent", "child")
        ancestors = archive.get_ancestors("child")
        assert ancestors == ["grandparent", "parent"]

    def test_lineage_ancestors_deep(self, archive):
        archive.add_child("g1", "g2")
        archive.add_child("g2", "g3")
        archive.add_child("g3", "g4")
        archive.add_child("g4", "g5")
        ancestors = archive.get_ancestors("g5")
        assert ancestors == ["g1", "g2", "g3", "g4"]

    def test_lineage_no_ancestors(self, archive):
        assert archive.get_ancestors("root") == []

    def test_get_stats(self, archive):
        archive.increment("generated", 20)
        archive.increment("passed", 15)
        archive.increment("failed", 5)
        archive.set_cell_elite("diversity_island", (0, 0, 0), "d1", 0.5)
        archive.set_cell_elite("quality_island", (0, 0, 0), "q1", 0.8)
        stats = archive.get_stats()
        assert stats["hypotheses_generated"] == 20
        assert stats["hypotheses_passed"] == 15
        assert stats["hypotheses_failed"] == 5
        assert stats["diversity_island_cells"] == 1
        assert stats["quality_island_cells"] == 1
        assert stats["use_fallback"] is True

    def test_clear_session(self, archive):
        archive.set_json("hypothesis", "h1", {"data": 1})
        archive.increment("generated", 5)
        archive.set_cell_elite("diversity_island", (0, 0, 0), "d1", 0.5)
        count = archive.clear_session()
        assert count > 0
        assert archive.get_json("hypothesis", "h1") is None
        assert archive.get_counter("generated") == 0
        assert archive.cell_count("diversity_island") == 0

    def test_ttl_default(self):
        config = RedisConfig()
        assert config.ttl_seconds == 604800  # 7 days
