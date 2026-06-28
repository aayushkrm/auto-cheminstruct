"""MAP-Elites quality-diversity optimization for Auto-ChemInstruct.

Multi-island evolutionary search with 3D behavior grid, five mutation
operators, migration, lineage tracking, and convergence detection.

Behavior space: (reaction_type_bin × molecular_weight_bin × fitness_bin).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from loguru import logger

# ── Behavior space binning ──


@dataclass
class BehaviorDimension:
    name: str
    bins: int
    bounds: tuple[float, float]
    binning: str = "linear"

    def bin_index(self, value: float) -> int:
        lo, hi = self.bounds
        if value <= lo:
            return 0
        if value >= hi:
            return self.bins - 1
        if self.binning == "log":
            import math

            val = max(value, lo + 1e-10)
            return int((math.log(val) - math.log(lo)) / (math.log(hi) - math.log(lo)) * self.bins)
        index = int((value - lo) / (hi - lo) * self.bins)
        return min(index, self.bins - 1)


DEFAULT_DIMS = [
    BehaviorDimension(name="reaction_type", bins=26, bounds=(0, 25)),
    BehaviorDimension(name="molecular_weight", bins=10, bounds=(50, 900), binning="log"),
    BehaviorDimension(name="fitness", bins=10, bounds=(0.0, 1.0)),
]


def _compute_coords(
    dims: list[BehaviorDimension],
    reaction_type_idx: int,
    molecular_weight: float,
    fitness: float,
) -> tuple[int, int, int]:
    return (
        dims[0].bin_index(reaction_type_idx),
        dims[1].bin_index(molecular_weight),
        dims[2].bin_index(fitness),
    )


# ── Elite cell ──


@dataclass
class MapCell:
    dims: tuple[int, int, int]
    hypothesis_id: str
    fitness: float
    generation: int = 0
    island_id: str = "default"
    parent_id: str | None = None
    mutation_name: str | None = None


@dataclass
class EliteArchive:
    """MAP-Elites grid archive with coverage tracking and elitism."""

    dimensions: list[BehaviorDimension] = field(default_factory=lambda: DEFAULT_DIMS)
    grid: dict[tuple[int, int, int], MapCell] = field(default_factory=dict)

    def offers_improvement(self, coords: tuple[int, int, int], fitness: float) -> bool:
        existing = self.grid.get(coords)
        return existing is None or fitness > existing.fitness

    def add_elite(
        self,
        coords: tuple[int, int, int],
        hypothesis_id: str,
        fitness: float,
        generation: int = 0,
        island_id: str = "default",
        parent_id: str | None = None,
        mutation_name: str | None = None,
    ) -> bool:
        if not self.offers_improvement(coords, fitness):
            return False
        self.grid[coords] = MapCell(
            dims=coords,
            hypothesis_id=hypothesis_id,
            fitness=fitness,
            generation=generation,
            island_id=island_id,
            parent_id=parent_id,
            mutation_name=mutation_name,
        )
        return True

    @property
    def coverage(self) -> float:
        max_cells = self.dimensions[0].bins * self.dimensions[1].bins * self.dimensions[2].bins
        return len(self.grid) / max_cells if max_cells > 0 else 0.0

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (self.dimensions[0].bins, self.dimensions[1].bins, self.dimensions[2].bins)

    @property
    def size(self) -> int:
        return len(self.grid)

    def random_empty_coords(self, rng: random.Random | None = None) -> tuple[int, int, int] | None:
        r = rng or random
        max_cells = self.dimensions[0].bins * self.dimensions[1].bins * self.dimensions[2].bins
        if len(self.grid) >= max_cells:
            return None
        for _ in range(100):
            coords = (
                r.randint(0, self.dimensions[0].bins - 1),
                r.randint(0, self.dimensions[1].bins - 1),
                r.randint(0, self.dimensions[2].bins - 1),
            )
            if coords not in self.grid:
                return coords
        return None

    def random_elite(self, rng: random.Random | None = None) -> MapCell | None:
        if not self.grid:
            return None
        r = rng or random
        return r.choice(list(self.grid.values()))

    def highest_fitness_elite(self) -> MapCell | None:
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda c: c.fitness)

    def elites_by_island(self, island_id: str) -> list[MapCell]:
        return [c for c in self.grid.values() if c.island_id == island_id]

    def all_elites(self) -> list[MapCell]:
        return list(self.grid.values())


# ── Mutation operators ──


@dataclass
class MutationOperator:
    name: str
    weight: float
    fn: Callable[[EliteArchive, random.Random], dict | None]


def mutate_reactant_substitution(archive: EliteArchive, rng: random.Random) -> dict | None:
    elite = archive.random_elite(rng)
    if elite is None:
        return None
    return {
        "mutation": "reactant_substitution",
        "parent_id": elite.hypothesis_id,
        "parent_fitness": elite.fitness,
        "instructions": (
            "Replace one substituent on the reactant scaffold with a bioisostere. "
            "Consider F→OH, Cl→CH3, or OMe→NH2 replacements."
        ),
    }


def mutate_condition_optimization(archive: EliteArchive, rng: random.Random) -> dict | None:
    elite = archive.random_elite(rng)
    if elite is None:
        return None
    mods = [
        "Increase reaction temperature by 15-30°C",
        "Switch to a polar aprotic solvent (DMSO, DMF, or acetonitrile)",
        "Add a phase-transfer catalyst",
        "Reduce concentration to favor intramolecular cyclization",
    ]
    return {
        "mutation": "condition_optimization",
        "parent_id": elite.hypothesis_id,
        "parent_fitness": elite.fitness,
        "instructions": rng.choice(mods),
    }


def mutate_reaction_type_crossover(archive: EliteArchive, rng: random.Random) -> dict | None:
    cells = list(archive.grid.values())
    if len(cells) < 2:
        return None
    a, b = rng.sample(cells, 2)
    return {
        "mutation": "reaction_type_crossover",
        "parent_id": a.hypothesis_id,
        "crossover_parent_id": b.hypothesis_id,
        "parent_fitness": max(a.fitness, b.fitness),
        "instructions": (
            "Combine the reaction framework from the first parent with functional "
            "group chemistry from the second parent. Think retrosynthetically."
        ),
    }


def mutate_scaffold_hopping(archive: EliteArchive, rng: random.Random) -> dict | None:
    elite = archive.random_elite(rng)
    if elite is None:
        return None
    hops = [
        "phenyl → pyridyl ring swap",
        "benzene → thiophene scaffold",
        "5-membered → 6-membered ring expansion",
        "linear alkane → cyclohexane replacement",
        "ester → amide bioisostere",
    ]
    return {
        "mutation": "scaffold_hopping",
        "parent_id": elite.hypothesis_id,
        "parent_fitness": elite.fitness,
        "instructions": rng.choice(hops),
    }


def mutate_insight_guided(archive: EliteArchive, rng: random.Random) -> dict | None:
    elite = archive.random_elite(rng)
    if elite is None:
        return None
    return {
        "mutation": "insight_guided",
        "parent_id": elite.hypothesis_id,
        "parent_fitness": elite.fitness,
        "instructions": (
            "Using the reflection traces from failed hypotheses, apply a targeted fix: "
            "address steric clashes by removing bulky ortho substituents, balance "
            "electronic effects with EDG/EWG tuning, or optimize leaving groups."
        ),
    }


DEFAULT_MUTATORS = [
    MutationOperator("reactant_substitution", 0.25, mutate_reactant_substitution),
    MutationOperator("condition_optimization", 0.20, mutate_condition_optimization),
    MutationOperator("reaction_type_crossover", 0.15, mutate_reaction_type_crossover),
    MutationOperator("scaffold_hopping", 0.15, mutate_scaffold_hopping),
    MutationOperator("insight_guided", 0.25, mutate_insight_guided),
]


def select_mutator(operators: list[MutationOperator], rng: random.Random) -> MutationOperator:
    total = sum(op.weight for op in operators)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for op in operators:
        cumulative += op.weight
        if r <= cumulative:
            return op
    return operators[-1]


# ── Island configuration ──


@dataclass
class IslandConfig:
    id: str
    name: str
    metric_weights: dict[str, float]
    max_size: int = 100
    migration_interval: int = 10
    max_migrants: int = 3


@dataclass
class Island:
    config: IslandConfig
    archive: EliteArchive
    generation: int = 0
    stagnation_count: int = 0
    best_fitness_history: list[float] = field(default_factory=list)

    @property
    def best_fitness(self) -> float:
        return self.best_fitness_history[-1] if self.best_fitness_history else 0.0

    def record_generation(self) -> None:
        best = max((c.fitness for c in self.archive.grid.values()), default=0.0)
        self.best_fitness_history.append(best)
        if len(self.best_fitness_history) > 1 and best <= self.best_fitness_history[-2]:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        self.generation += 1


# ── Migration ──


def migrate_between_islands(
    islands: list[Island],
    rng: random.Random,
    max_migrants: int = 3,
) -> int:
    if len(islands) < 2:
        return 0

    total_moved = 0
    for src in islands:
        if src.archive.size < 2:
            continue
        elites = src.archive.all_elites()
        rng.shuffle(elites)
        donors = elites[:max_migrants]
        for dst in islands:
            if dst is src:
                continue
            for cell in donors:
                added = dst.archive.add_elite(
                    coords=cell.dims,
                    hypothesis_id=cell.hypothesis_id,
                    fitness=cell.fitness,
                    generation=dst.generation,
                    island_id=dst.config.id,
                    parent_id=cell.parent_id,
                    mutation_name="migration",
                )
                if added:
                    total_moved += 1
    return total_moved


# ── Evolution orchestrator ──


@dataclass
class MapElitesOrchestrator:
    islands: list[Island]
    operators: list[MutationOperator] = field(default_factory=lambda: DEFAULT_MUTATORS)
    max_generations: int = 50
    mutants_per_generation: int = 20
    min_coverage: float = 0.05
    max_stagnation: int = 10
    seed: int = 42
    seed_programs: int = 10

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def run(
        self,
        evaluator: Callable[[dict], float] | None = None,
    ) -> dict:
        """Run the MAP-Elites evolution loop.

        Args:
            evaluator: Optional fitness function(mutation_dict) -> float.
                When None, uses simulated fitness jitter.
        """
        evaluate = evaluator or (lambda m: _simulate_fitness(m, self._rng))

        logger.info(
            "MAP-Elites: {} islands, {} gens, {} mutants/gen, seed={}",
            len(self.islands),
            self.max_generations,
            self.mutants_per_generation,
            self.seed,
        )

        stats: dict[str, Any] = {
            "generations_completed": 0,
            "total_elites_added": 0,
            "final_coverage": 0.0,
            "generation_history": [],
            "convergence_reason": "max_generations",
        }

        self._seed_archives()

        for gen in range(self.max_generations):
            gen_stats = self._evolve_generation(gen, evaluate)
            stats["generation_history"].append(gen_stats)

            if gen > 0 and gen % self.islands[0].config.migration_interval == 0:
                migrated = migrate_between_islands(self.islands, self._rng)
                if migrated:
                    logger.debug("Migration transferred {} elites", migrated)

            max_stag = max(island.stagnation_count for island in self.islands)
            min_cov = min(island.archive.coverage for island in self.islands)

            if max_stag >= self.max_stagnation:
                stats["convergence_reason"] = "stagnation"
                logger.info("Converged: stagnation limit at generation {}", gen)
                break
            if min_cov >= self.min_coverage and gen >= 10:
                stats["convergence_reason"] = "coverage"
                logger.info("Converged: coverage threshold at generation {}", gen)
                break

        stats["generations_completed"] = len(stats["generation_history"])
        stats["final_coverage"] = max(island.archive.coverage for island in self.islands)
        stats["total_elites_added"] = sum(island.archive.size for island in self.islands)

        logger.info(
            "Evolution done: {} gens, {} elites, {:.1%} coverage ({})",
            stats["generations_completed"],
            stats["total_elites_added"],
            stats["final_coverage"],
            stats["convergence_reason"],
        )
        return stats

    def _seed_archives(self) -> None:
        """Seed each island's archive with random initial elites."""
        for island in self.islands:
            for _ in range(self.seed_programs):
                fitness = self._rng.uniform(0.3, 0.8)
                coords = _compute_coords(
                    island.archive.dimensions,
                    reaction_type_idx=self._rng.randint(0, 23),
                    molecular_weight=self._rng.uniform(100, 800),
                    fitness=fitness,
                )
                if island.archive.offers_improvement(coords, fitness):
                    island.archive.add_elite(
                        coords=coords,
                        hypothesis_id=str(uuid4()),
                        fitness=fitness,
                        generation=0,
                        island_id=island.config.id,
                    )
        logger.debug("Seeded archives with {} programs each", self.seed_programs)

    def _evolve_generation(self, gen: int, evaluator: Callable[[dict], float]) -> dict:
        gen_stats: dict[str, dict] = {}
        for island in self.islands:
            added = 0
            for _ in range(self.mutants_per_generation):
                op = select_mutator(self.operators, self._rng)
                mutation = op.fn(island.archive, self._rng)
                if mutation is None:
                    continue
                fitness = evaluator(mutation)

                coords = _compute_coords(
                    island.archive.dimensions,
                    reaction_type_idx=self._rng.randint(0, 23),
                    molecular_weight=self._rng.uniform(100, 800),
                    fitness=fitness,
                )

                if island.archive.offers_improvement(coords, fitness):
                    island.archive.add_elite(
                        coords=coords,
                        hypothesis_id=str(uuid4()),
                        fitness=fitness,
                        generation=gen,
                        island_id=island.config.id,
                        parent_id=mutation.get("parent_id"),
                        mutation_name=mutation.get("mutation"),
                    )
                    added += 1

            island.record_generation()
            gen_stats[island.config.id] = {
                "generation": gen,
                "added": added,
                "coverage": island.archive.coverage,
                "best_fitness": island.best_fitness,
                "size": island.archive.size,
            }

        return gen_stats


def _simulate_fitness(mutation: dict, rng: random.Random) -> float:
    base = mutation.get("parent_fitness", 0.5)
    jitter = rng.uniform(-0.15, 0.15)
    return max(0.0, min(1.0, base + jitter))
