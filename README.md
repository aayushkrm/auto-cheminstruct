# Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry**

[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/aayushkrm/autochem-instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-230%2F230-brightgreen)](https://github.com/aayushkrm/auto-cheminstruct)

> Internship: TSU Laboratory of AI in Chemistry & Molecular Engineering × AIRI Institute
>
> Target: NeurIPS Datasets & Benchmarks Track

---

An autonomous multi-agent pipeline that generates **physically-validated instruction data** for chemistry DSLMs. Reaction hypotheses are structurally verified (RDKit), energetically validated (MMFF94), causally analyzed on failure via 4-step reasoning chains, and refined through MAP-Elites evolutionary search — producing DPO preference pairs without human annotation.

## Architecture

```
                         ┌─────────────────────────────────┐
                         │  MAP-Elites Evolutionary Search  │
                         │  26×10×10 grid, 5 mutation ops  │
                         │  4 specialist islands            │
                         └──────────────┬──────────────────┘
                                        │
    Hypothesis Agent ──→ Verify (RDKit+MMFF94) ──PASS→ Compile → DPO Pairs
          ↑                       │
          │                       └──FAIL→ CARL Reflection Chain
          │                           Steric → Electronic → Thermo → Synthesis
          │                                       │
          └─────────── LearningContext ←──────────┘
                (Self-Bootstrapping Loop)
```

## Implementation Phases

| Phase | Commit | Tests | Key Deliverables |
|-------|--------|-------|------------------|
| **I: Foundation** | `df5a2ca` | 140 | Redis state layer, 5 Hydra YAML configs, problem directory (5 seed reactions, validate.py, metrics.yaml) |
| **II: DAG Engine** | `a522fb5` | 163 | Async DAGPipeline, Kahn topological sort, semaphore-bounded parallelism, 8 Pydantic I/O models, 4 agent factory functions |
| **III: MAP-Elites** | `1467e7e` | 202 | 2,600-cell behavior grid, 5 weighted mutation operators, 4 specialist islands with migration, seeding + stagnation convergence, deterministic RNG |
| **IV: CARL Chains** | `f5d3dcf` | 219 | 4-step parallel DAG reflection (StericAnalysis → ElectronicAnalysis → ThermodynamicAnalysis → CausalSynthesis), CARLReflectionAgent, batch filtering |
| **V: Ablation + Paper** | `1834c45` | **230** | 7-variant evolution ablation, CLI `--mode evolution`, updated NeurIPS paper with evolution results |

## What's Built

### Core Pipeline (4 Agents)

| Component | Description |
|---|---|
| **Hypothesis Agent** | LLM-based generation of diverse chemical reactions (19 named types) with RAG enrichment |
| **Verification Agent** | RDKit structural validation, MMFF94 energetics, chemical feasibility filtering |
| **Reflection Agent** | CARL 4-step causal decomposition (steric/electronic/thermodynamic/causal synthesis) |
| **Compilation Agent** | DPO preference pairs (chosen=passed, rejected=failed+reflection), 6-dim quality scoring |
| **Pipeline Orchestrator** | SQLite checkpoints, bootstrap loop, temperature cosine annealing (1.0→0.3) |

### MAP-Elites Evolutionary Search

Custom implementation following the GigaEvo (AIRI, arXiv:2511.17592) architecture pattern:

- **Behavior grid**: 26 reaction types × 10 MW bins × 10 fitness bins = 2,600 cells
- **5 mutation operators**: Reactant substitution, condition optimization, reaction crossover, scaffold hopping, insight-guided
- **4 specialist islands**: Diversity, Quality, Novelty, Yield — with conservative migration
- **Convergence**: Stagnation detection (10 gens) + coverage threshold (5%)

### CARL Reasoning Chains

Custom implementation following the Maestro CARL (AIRI Institute) Event-Action-Result pattern:

- **Step 1**: Steric analysis (van der Waals, Baldwin's rules, eclipsing)
- **Step 2**: Electronic analysis (HOMO-LUMO, electrophilicity, HSAB)
- **Step 3**: Thermodynamic & kinetic analysis (enthalpy/entropy, competing pathways)
- **Step 4**: Causal synthesis (merged explanation + actionable fix)
- Steps 1–3 run in parallel via DAG engine, Step 4 synthesizes

### Dataset (v1.0)

- **110 DPO pairs** (81 train / 6 val / 23 test), published on HuggingFace
- 187 unique molecules, 87.9% Tanimoto diversity
- 13 distinct reaction types, 72.8% causal reflection coverage
- Avg quality score: 0.636 (6-dimension rubric)

### Benchmarks

- **Pipeline ablation** (4 variants): Bootstrap → 2.5× pairs, Reflection → +15.4% quality, RAG → +7.5pp pass rate
- **Evolution ablation** (7 variants): Full-System achieves 3.5× elites, +1.0% coverage, +17% pass rate vs Baseline
- **ChemCoTBench comparison**: Match molecular diversity of human-annotated 1,495-sample dataset
- **230/230 tests passing**

### Paper

NeurIPS LaTeX scaffold with ablation tables, ChemCoTBench comparison, GigaEvo + AlphaEvolve + MAP-Elites citations, 6-dim quality scoring analysis.

## Quick Start

```bash
# Install
cd auto-cheminstruct && uv sync

# Run pipeline (full system, 5 hypotheses, 3 bootstrap iterations)
uv run python -m src.cli.main pipeline -n 5 -B 3

# Run pipeline-level ablation (4 variants)
uv run python -m src.cli.main ablation -n 8 -o benchmarks

# Run evolution-level ablation (7 variants)
uv run python -m src.cli.main ablation -m evolution -o benchmarks

# Compare against ChemCoTBench
uv run python -m src.cli.main chemcot -d datasets/autochem-merged/

# Run tests
uv run pytest

# View config
uv run python -m src.cli.main config-cmd
```

## Dependencies

- **Python 3.13+**, Pydantic v2
- **RDKit** for structural chemistry (validation, descriptors, MMFF94)
- **NetworkX** for chemical knowledge graph
- **Loguru** for logging, **OmegaConf + Hydra** for configuration
- **Fireworks AI** (OpenAI-compatible) — model `accounts/fireworks/models/deepseek-v3p2`
- **Redis** (optional, for distributed state)

## Research Framework Alignment

Based on 27 research documents analyzing AIRI ecosystem tools:

| Framework | AIRI Lab | Our Implementation |
|-----------|----------|-------------------|
| **GigaEvo** (evolutionary search) | `AIRI-Institute/gigaevo-core` | `src/evolution/map_elites.py` — 2,600-cell MAP-Elites, 5 mutation ops, 4 islands |
| **Maestro CARL** (reasoning chains) | `AIRI-Institute/maestro-core` / `mmar-carl` | `src/carl/chain.py` — 4-step parallel DAG reflection, Event-Action-Result pattern |
| **DAG Engine** | GigaEvo async pipeline | `src/evolution/dag.py` — Kahn sort, asyncio semaphore, failure propagation |
| **Redis Store** | GigaEvo persistence | `src/evolution/redis_store.py` — JSON storage, atomic counters, in-memory fallback |

Detailed research: `docs/maestro-gigachain-gigaevo-research.md`, `docs/gigaevo-integration-blueprint.md`

## Citation

```bibtex
@misc{auto-cheminstruct-2026,
  author = {Kumar, Aayush},
  title = {Auto-ChemInstruct: Agent-Driven Synthesization of RLHF Data
           for Domain-Specific Language Models in Chemistry},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/aayushkrm/auto-cheminstruct}
}
```
