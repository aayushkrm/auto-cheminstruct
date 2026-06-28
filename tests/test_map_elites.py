"""Tests for MAP-Elites archive, mutation operators, islands, migration, and orchestrator."""

from __future__ import annotations

import random

import pytest

from src.evolution.map_elites import (
    DEFAULT_DIMS,
    DEFAULT_MUTATORS,
    BehaviorDimension,
    EliteArchive,
    Island,
    IslandConfig,
    MapCell,
    MapElitesOrchestrator,
    MutationOperator,
    _compute_coords,
    migrate_between_islands,
    mutate_condition_optimization,
    mutate_insight_guided,
    mutate_reactant_substitution,
    mutate_reaction_type_crossover,
    mutate_scaffold_hopping,
    select_mutator,
)

# ── BehaviorDimension tests ──


class TestBehaviorDimension:
    def test_linear_binning(self):
        dim = BehaviorDimension(name="x", bins=5, bounds=(0, 10))
        assert dim.bin_index(0) == 0
        assert dim.bin_index(2.4) == 1
        assert dim.bin_index(5.0) == 2
        assert dim.bin_index(7.5) == 3
        assert dim.bin_index(9.9) == 4
        assert dim.bin_index(10.0) == 4

    def test_linear_below_bound(self):
        dim = BehaviorDimension(name="x", bins=5, bounds=(0, 10))
        assert dim.bin_index(-5.0) == 0

    def test_linear_above_bound(self):
        dim = BehaviorDimension(name="x", bins=5, bounds=(0, 10))
        assert dim.bin_index(999.0) == 4

    def test_log_binning(self):
        dim = BehaviorDimension(name="mw", bins=3, bounds=(10, 1000), binning="log")
        assert dim.bin_index(10) == 0
        assert dim.bin_index(100) == 1
        assert dim.bin_index(1000) == 2

    def test_coords_helper(self):
        coords = _compute_coords(DEFAULT_DIMS, 0, 1e-6, 0.0)
        assert len(coords) == 3
        assert all(0 <= c < d.bins for c, d in zip(coords, DEFAULT_DIMS, strict=False))


# ── MapCell tests ──


class TestMapCell:
    def test_defaults(self):
        cell = MapCell(dims=(0, 0, 0), hypothesis_id="h1", fitness=0.5)
        assert cell.generation == 0
        assert cell.island_id == "default"
        assert cell.parent_id is None
        assert cell.mutation_name is None

    def test_with_lineage(self):
        cell = MapCell(
            dims=(1, 2, 3),
            hypothesis_id="child",
            fitness=0.8,
            generation=5,
            parent_id="parent",
            mutation_name="scaffold_hopping",
        )
        assert cell.parent_id == "parent"
        assert cell.mutation_name == "scaffold_hopping"


# ── EliteArchive tests ──


class TestEliteArchive:
    @pytest.fixture
    def archive(self):
        return EliteArchive()

    def test_empty_archive(self, archive):
        assert archive.size == 0
        assert archive.coverage == 0.0
        assert archive.random_elite() is None
        assert archive.highest_fitness_elite() is None

    def test_offers_improvement_empty(self, archive):
        assert archive.offers_improvement((0, 0, 0), 0.5)

    def test_add_elite(self, archive):
        assert archive.add_elite((0, 0, 0), "h1", 0.5)
        assert archive.size == 1
        cell = archive.grid[(0, 0, 0)]
        assert cell.hypothesis_id == "h1"
        assert cell.fitness == 0.5

    def test_elitism_lower_rejected(self, archive):
        archive.add_elite((0, 0, 0), "h1", 0.5)
        assert not archive.offers_improvement((0, 0, 0), 0.3)
        added = archive.add_elite((0, 0, 0), "h2", 0.3)
        assert not added
        assert archive.grid[(0, 0, 0)].hypothesis_id == "h1"

    def test_elitism_higher_accepted(self, archive):
        archive.add_elite((0, 0, 0), "h1", 0.5)
        assert archive.offers_improvement((0, 0, 0), 0.9)
        added = archive.add_elite((0, 0, 0), "h2", 0.9)
        assert added
        assert archive.grid[(0, 0, 0)].hypothesis_id == "h2"

    def test_coverage_calculation(self, archive):
        max_cells = archive.grid_shape[0] * archive.grid_shape[1] * archive.grid_shape[2]
        archive.add_elite((0, 0, 0), "h1", 0.5)
        archive.add_elite((0, 0, 1), "h2", 0.6)
        assert archive.coverage == 2 / max_cells

    def test_random_elite_returns_cell(self, archive):
        archive.add_elite((0, 0, 0), "h1", 0.5)
        cell = archive.random_elite()
        assert cell is not None
        assert cell.hypothesis_id == "h1"

    def test_highest_fitness_elite(self, archive):
        archive.add_elite((0, 0, 0), "low", 0.3)
        archive.add_elite((1, 1, 1), "mid", 0.6)
        archive.add_elite((2, 2, 2), "high", 0.9)
        best = archive.highest_fitness_elite()
        assert best.hypothesis_id == "high"

    def test_random_empty_coords(self, archive):
        archive.add_elite((0, 0, 0), "h1", 0.5)
        coords = archive.random_empty_coords()
        assert coords is not None
        assert coords != (0, 0, 0)

    def test_random_empty_coords_full(self, archive):
        shape = archive.grid_shape
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    archive.add_elite((x, y, z), f"h{x}_{y}_{z}", 0.5)
        assert archive.random_empty_coords() is None

    def test_elites_by_island(self, archive):
        archive.add_elite((0, 0, 0), "h1", 0.5, island_id="a")
        archive.add_elite((1, 0, 0), "h2", 0.6, island_id="a")
        archive.add_elite((2, 0, 0), "h3", 0.7, island_id="b")
        a_elites = archive.elites_by_island("a")
        assert len(a_elites) == 2
        b_elites = archive.elites_by_island("b")
        assert len(b_elites) == 1

    def test_all_elites(self, archive):
        archive.add_elite((0, 0, 0), "h1", 0.5)
        archive.add_elite((1, 1, 1), "h2", 0.6)
        assert len(archive.all_elites()) == 2


# ── Mutation operator tests ──


class TestMutationOperators:
    @pytest.fixture
    def archive(self):
        arch = EliteArchive()
        arch.add_elite((0, 0, 0), "h1", 0.5)
        arch.add_elite((1, 1, 1), "h2", 0.6)
        return arch

    def test_reactant_substitution(self, archive):
        rng = random.Random(42)
        result = mutate_reactant_substitution(archive, rng)
        assert result is not None
        assert result["mutation"] == "reactant_substitution"
        assert "parent_id" in result

    def test_condition_optimization(self, archive):
        rng = random.Random(42)
        result = mutate_condition_optimization(archive, rng)
        assert result is not None
        assert result["mutation"] == "condition_optimization"

    def test_reaction_type_crossover(self, archive):
        rng = random.Random(42)
        result = mutate_reaction_type_crossover(archive, rng)
        assert result is not None
        assert result["mutation"] == "reaction_type_crossover"
        assert "crossover_parent_id" in result

    def test_scaffold_hopping(self, archive):
        rng = random.Random(42)
        result = mutate_scaffold_hopping(archive, rng)
        assert result is not None
        assert result["mutation"] == "scaffold_hopping"

    def test_insight_guided(self, archive):
        rng = random.Random(42)
        result = mutate_insight_guided(archive, rng)
        assert result is not None
        assert result["mutation"] == "insight_guided"

    def test_crossover_needs_two_elites(self):
        archive_single = EliteArchive()
        archive_single.add_elite((0, 0, 0), "h1", 0.5)
        rng = random.Random(42)
        assert mutate_reaction_type_crossover(archive_single, rng) is None

    def test_empty_archive_mutations(self):
        empty = EliteArchive()
        rng = random.Random(42)
        assert mutate_reactant_substitution(empty, rng) is None
        assert mutate_condition_optimization(empty, rng) is None
        assert mutate_scaffold_hopping(empty, rng) is None
        assert mutate_insight_guided(empty, rng) is None


# ── select_mutator tests ──


class TestSelectMutator:
    def test_weighted_selection(self):
        ops = [
            MutationOperator("a", 0.9, lambda a, r: None),
            MutationOperator("b", 0.1, lambda a, r: None),
        ]
        rng = random.Random(42)
        counts = {"a": 0, "b": 0}
        for _ in range(1000):
            selected = select_mutator(ops, rng)
            counts[selected.name] += 1
        assert counts["a"] > counts["b"]
        assert counts["a"] > 700  # ~90% chance

    def test_default_mutators_all_weights(self):
        total = sum(op.weight for op in DEFAULT_MUTATORS)
        assert abs(total - 1.0) < 0.01


# ── Island tests ──


class TestIsland:
    @pytest.fixture
    def island(self):
        config = IslandConfig(id="test", name="Test Island", metric_weights={"fitness": 1.0})
        return Island(config=config, archive=EliteArchive())

    def test_best_fitness_initial(self, island):
        assert island.best_fitness == 0.0

    def test_record_generation(self, island):
        island.archive.add_elite((0, 0, 0), "h1", 0.5)
        island.record_generation()
        assert island.best_fitness == 0.5
        assert island.generation == 1
        assert island.stagnation_count == 0

    def test_stagnation_detection(self, island):
        island.archive.add_elite((0, 0, 0), "h1", 0.5)
        island.record_generation()
        island.record_generation()  # no improvement
        assert island.stagnation_count == 1

        island.archive.add_elite((1, 0, 0), "h2", 0.7)
        island.record_generation()
        assert island.stagnation_count == 0  # reset on improvement


# ── Migration tests ──


class TestMigration:
    def test_migration_between_islands(self):
        a = EliteArchive()
        b = EliteArchive()
        a.add_elite((0, 0, 0), "h1", 0.5)
        a.add_elite((1, 1, 1), "h2", 0.6)
        a.add_elite((2, 2, 2), "h3", 0.7)

        island_a = Island(
            config=IslandConfig(id="a", name="A", metric_weights={}),
            archive=a,
        )
        island_b = Island(
            config=IslandConfig(id="b", name="B", metric_weights={}),
            archive=b,
        )

        moved = migrate_between_islands([island_a, island_b], random.Random(42))
        assert moved > 0
        assert b.size > 0

    def test_migration_single_island_noop(self):
        a = EliteArchive()
        a.add_elite((0, 0, 0), "h1", 0.5)
        island_a = Island(
            config=IslandConfig(id="a", name="A", metric_weights={}),
            archive=a,
        )
        moved = migrate_between_islands([island_a], random.Random(42))
        assert moved == 0

    def test_migration_empty_source(self):
        a = EliteArchive()
        b = EliteArchive()
        island_a = Island(
            config=IslandConfig(id="a", name="A", metric_weights={}),
            archive=a,
        )
        island_b = Island(
            config=IslandConfig(id="b", name="B", metric_weights={}),
            archive=b,
        )
        moved = migrate_between_islands([island_a, island_b], random.Random(42))
        assert moved == 0


# ── MapElitesOrchestrator integration tests ──


class TestMapElitesOrchestrator:
    def test_single_island_run(self):
        config = IslandConfig(id="d", name="Diversity", metric_weights={})
        island = Island(config=config, archive=EliteArchive())
        orch = MapElitesOrchestrator(
            islands=[island],
            max_generations=3,
            mutants_per_generation=5,
            seed=42,
        )
        stats = orch.run()
        assert stats["generations_completed"] == 3
        assert stats["total_elites_added"] > 0
        assert 0 < stats["final_coverage"] <= 1.0

    def test_multi_island_run(self):
        configs = [
            IslandConfig(id="a", name="A", metric_weights={}),
            IslandConfig(id="b", name="B", metric_weights={}),
        ]
        islands = [Island(config=c, archive=EliteArchive()) for c in configs]
        orch = MapElitesOrchestrator(
            islands=islands,
            max_generations=5,
            mutants_per_generation=5,
            seed=42,
        )
        stats = orch.run()
        assert stats["generations_completed"] == 5

    def test_stagnation_convergence(self):
        config = IslandConfig(id="s", name="Single", metric_weights={})
        island = Island(config=config, archive=EliteArchive())
        orch = MapElitesOrchestrator(
            islands=[island],
            max_generations=100,
            mutants_per_generation=2,
            max_stagnation=4,
            seed=123,
        )
        stats = orch.run()
        assert stats["generations_completed"] < 100

    def test_deterministic_reproducibility(self):
        config = IslandConfig(id="d", name="D", metric_weights={})

        def run_once():
            island = Island(config=config, archive=EliteArchive())
            orch = MapElitesOrchestrator(
                islands=[island],
                max_generations=5,
                mutants_per_generation=5,
                seed=42,
            )
            return orch.run()

        stats1 = run_once()
        stats2 = run_once()
        assert stats1["generations_completed"] == stats2["generations_completed"]
        assert abs(stats1["total_elites_added"] - stats2["total_elites_added"]) <= 5
        assert abs(stats1["final_coverage"] - stats2["final_coverage"]) < 0.01

    def test_custom_evaluator(self):
        def evaluator(mutation: dict) -> float:
            return 0.85  # always great

        config = IslandConfig(id="e", name="E", metric_weights={})
        island = Island(config=config, archive=EliteArchive())
        orch = MapElitesOrchestrator(
            islands=[island],
            max_generations=3,
            mutants_per_generation=5,
            seed=42,
        )
        stats = orch.run(evaluator=evaluator)
        assert stats["total_elites_added"] > 0
