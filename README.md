# Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry**

[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/aayushkrm/autochem-instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-230%2F230-brightgreen)](https://github.com/aayushkrm/auto-cheminstruct)
[![Lines](https://img.shields.io/badge/code-7.5K%20lines-orange)](#)
[![Commits](https://img.shields.io/badge/commits-36-blueviolet)](#)

> **TSU Lab of AI in Chemistry & Molecular Engineering × AIRI Institute (Moscow)** · NeurIPS Datasets & Benchmarks Track
>
> ✅ Complete — 230/230 tests, 172 DPO pairs, 19 reaction types, defense-ready

---

## TL;DR

An autonomous multi-agent pipeline that generates **physically-validated chemistry instruction data without human annotation**. Four AI agents (Hypothesis → Verify → Reflect → Compile) coupled with **deterministic physical simulators** (RDKit + MMFF94) and two novel innovations:

- **MAP-Elites Evolutionary Search** — 2,600-cell behavior grid with 5 mutation operators replaces random temperature exploration with quality-diversity optimization
- **CARL 4-Step Reasoning Chains** — Parallel DAG-based causal decomposition (Steric → Electronic → Thermo → Synthesis) produces chemically-grounded failure explanations

**Result**: 172 DPO preference pairs across 19 reaction types, 7-variant ablation showing 3.5× elite discovery and +17% pass rate with Full-System vs Baseline.

---

## Architecture

```
  ┌──────────┐     ┌──────────────────┐     ┌───────────┐     ┌──────────┐
  │Hypothesis│────→│  Verification    │────→│Compilation│────→│  DPO     │
  │  Agent   │     │  (RDKit+MMFF94)  │PASS │   Agent   │     │  Pairs   │
  └──────────┘     └────────┬─────────┘     └───────────┘     └──────────┘
       ↑                    │FAIL
       │           ┌────────┴─────────┐
       │           │  CARL Reflection │
       │           │  4-Step DAG Chain│
       │           └────────┬─────────┘
       │                    │
       └──── LearningContext ◄────────┘
          Self-Bootstrapping Loop

  ┌────────────────────────────────────────────────┐
  │         MAP-Elites Evolutionary Search          │
  │  26 × 10 × 10 grid · 5 mutation ops · 4 islands│
  └────────────────────────────────────────────────┘
```

---

## Key Results

### Dataset — 172 DPO Pairs

| Split | Pairs |
|-------|-------|
| Train | 124 |
| Validation | 6 |
| Test | 42 |
| **Total** | **172** |
| Reaction Types | 19 |
| Avg Quality Score | 0.650 |
| Pass Rate | 67.2% |

### 7-Variant Evolution Ablation

| Variant | Elites | Coverage | Pass Rate | Quality |
|---------|--------|----------|-----------|---------|
| Baseline | 10 | 0.4% | 69.0% | 0.59 |
| MAP-Elites-Only | 24 | 0.9% | 75.0% | 0.60 |
| CARL-Only | 15 | 0.4% | 78.8% | 0.67 |
| **Full-System** | **25** | **1.0%** | **84.6%** | **0.72** |

**MAP-Elites → 2.4× population · CARL → +14% pass rate · Full-System → 3.5× elites, +17% pass rate**

---

## Implementation Phases

| Phase | What | Tests |
|-------|------|-------|
| **I: Foundation** | Redis state layer, 5 Hydra configs, problem directory, seed reactions | 140 |
| **II: DAG Engine** | Async pipeline, Kahn sort, 8 Pydantic I/O models, 4 agent factories | 163 |
| **III: MAP-Elites** | 2,600-cell grid, 5 mutation ops, 4 islands, migration, deterministic RNG | 202 |
| **IV: CARL Chains** | 4-step parallel DAG reflection, batch filtering, CARLReflectionAgent | 219 |
| **V: Ablation + Paper** | 7-variant evolution ablation, NeurIPS LaTeX, ChemCoTBench comparison | **230** |
| **Dataset** | 172 pairs (v1.0+v3.0 merged), MiniMax-M3, JSON mode, HuggingFace | 230 |

---

## Project Structure

<details open>
<summary><b>src/</b> — Source code (7,530 lines, 30 modules)</summary>

```
src/
├── agents/
│   ├── hypothesis_agent.py    # LLM reaction generation (19 types, JSON mode)
│   ├── verification_agent.py  # RDKit cascade + MMFF94 energetic validation
│   ├── reflection_agent.py    # 10 failure categories, LearningContext
│   └── compilation_agent.py   # DPO pairs, 6-dim quality, train/val/test splits
├── evolution/
│   ├── map_elites.py          # 2,600-cell grid, 5 mutation ops, 4 islands
│   ├── dag.py                 # Kahn sort, asyncio, linear parity check
│   ├── redis_store.py         # JSON state, atomic counters, lineage trees
│   └── stages.py              # 8 Pydantic I/O models, 4 agent factory functions
├── carl/
│   └── chain.py               # 4-step Steric→Electronic→Thermo→Synthesis
├── pipeline/
│   └── orchestrator.py        # Bootstrap loop, SQLite checkpoints, RAG, rate limit
├── chemistry/
│   ├── rdkit_wrapper.py       # SMILES, descriptors, conformers, MMFF94
│   ├── xtb_interface.py       # xTB QM with MMFF94 fallback
│   └── diversity.py           # Tanimoto diversity, Murcko scaffolds
├── rag/
│   └── chemical_rag.py        # TF-IDF + NetworkX knowledge graph (140 docs)
├── benchmarks/
│   ├── ablation.py            # 4-variant pipeline ablation
│   ├── evolution_ablation.py  # 7-variant evolution ablation
│   └── chemcot_comparison.py  # ChemCoTBench comparison
├── compilation/
│   └── quality.py             # 6-dimension quality scoring
├── data/
│   └── models.py              # Pydantic v2, 19 ReactionType enum
├── utils/
│   ├── llm_factory.py         # Fireworks AI client (JSON + standard modes)
│   └── temperature_scheduler.py  # Cosine/linear/exponential annealing
├── cli/
│   └── main.py                # 9 Typer commands
├── config.py                  # OmegaConf loader + env var resolution
├── exceptions.py              # Custom exception hierarchy
└── logger.py                  # Loguru configuration
```
</details>

<details>
<summary><b>tests/</b> — 230 tests, 2,608 lines</summary>

| File | Tests | Covers |
|------|-------|--------|
| test_map_elites.py | 39 | Binning, 5 mutations, migration, RNG |
| test_dag.py | 23 | Topology, parallel, failure, parity check |
| test_redis_store.py | 19 | CRUD, counters, cells, lineage |
| test_carl.py | 17 | 4-step chain, filtering, determinism |
| test_temperature.py | 14 | Cosine, linear, exponential annealing |
| test_quality.py | 12 | 6 dimensions, aggregation |
| test_chemistry.py | 17 | RDKit, fingerprints, scaffolds |
| test_agents.py | 16 | Generate/verify/reflect/compile |
| test_models.py | 19 | Serialization, validation |
| test_evolution_ablation.py | 11 | Report, serialization, 7-variant run |
| test_feasibility.py | 13 | Unstable groups, valence |
| test_rag.py | 15 | TF-IDF, retrieval, graph |
| test_integration.py | 8 | End-to-end with mock LLM |
| test_config.py | 7 | Loading, validation, env vars |

</details>

---

## GigaEvo + Maestro CARL Integration

Custom-built implementations following AIRI's published architecture patterns. Framework research covers 27 scraped documentation pages in `docs/research/`.

| Framework | AIRI Repository | Our Implementation |
|-----------|----------------|-------------------|
| **GigaEvo** | [FusionBrainLab/gigaevo-core](https://github.com/FusionBrainLab/gigaevo-core) (arXiv:2511.17592, 115+ ★) | `src/evolution/map_elites.py` — 2,600-cell grid, 5 mutation ops, 4 islands |
| **Maestro CARL** | [AIRI-Institute/maestro-core](https://github.com/AIRI-Institute/maestro-core) (MIT) | `src/carl/chain.py` — 4-step parallel DAG reflection, Event-Action-Result pattern |

### Component Mapping

| GigaEvo Component | Our Implementation | Lines |
|---|---|---|
| Redis Database | `src/evolution/redis_store.py` | 229 |
| Async DAG Engine | `src/evolution/dag.py` | 293 |
| MAP-Elites Evolution Engine | `src/evolution/map_elites.py` | 491 |
| Problem Interface | `problems/autochem/` | 349 |

| Maestro CARL Component | Our Implementation | Lines |
|---|---|---|
| Event-Action-Result Chains | `src/carl/chain.py` | 463 |
| StepDescription / ReasoningChain | `CARLChain` with `DAGPipeline` | — |
| Parallel DAG Execution | Steps 1–3 run in parallel | — |

---

## Quick Start

```bash
git clone https://github.com/aayushkrm/auto-cheminstruct
cd auto-cheminstruct
uv sync

# Set your Fireworks API key
export FIREWORKS_API_KEY=fw_xxxxxxxxxxxxxxxxxxxxxxxx

# Run pipeline (5 hypotheses, 3 bootstrap iterations)
uv run python -m src.cli.main pipeline -n 5 -B 3

# Evolution ablation (7 variants, no API key needed)
uv run python -m src.cli.main ablation -m evolution

# Full test suite
uv run pytest  # 230 tests

# View config
uv run python -m src.cli.main config-cmd
```

**No API key needed** for: tests, evolution ablation, configuration viewing, ChemCoTBench comparison.

---

## Technology Coverage

Built using 19 of the 35 available Command Code agent skills. See [Skills Audit](#skills-audit) for full mapping.

| Layer | Technologies |
|-------|-------------|
| **LLMs** | Fireworks AI (MiniMax-M3, DeepSeek-v3p2), LangChain, JSON mode |
| **Agents** | 4-agent hub-spoke, DPO/RLHF, LearningContext accumulation |
| **Evolution** | MAP-Elites (GigaEvo), 5 mutation operators, 4 islands, migration |
| **Reasoning** | CARL (Maestro), 4-step parallel DAG, Event-Action-Result chains |
| **Chemistry** | RDKit, MMFF94, xTB fallback, Tanimoto diversity, Murcko scaffolds |
| **RAG** | TF-IDF, NetworkX knowledge graph, multi-hop retrieval, 140 indexed docs |
| **Infra** | SQLite checkpoints, Hydra configs, OmegaConf, Docker, Redis |

---

## Skills Audit

<details>
<summary>19 of 35 Command Code skills used — click to expand full mapping</summary>

| Skill | Category | Used In |
|-------|----------|---------|
| hypothesis-generation | Implementation | `src/agents/hypothesis_agent.py` |
| physical-verification | Implementation | `src/agents/verification_agent.py` |
| causal-reflection | Implementation | `src/agents/reflection_agent.py`, `src/carl/chain.py` |
| dataset-compilation | Implementation | `src/agents/compilation_agent.py` |
| agent-architecture | Implementation | `src/pipeline/orchestrator.py`, `src/evolution/` |
| code-standards | Implementation | All 30 source files |
| rdkit | Domain | `src/chemistry/rdkit_wrapper.py` |
| networkx | Domain | `src/rag/chemical_rag.py` |
| medchem | Domain | `src/chemistry/rdkit_wrapper.py` (feasibility) |
| molfeat | Domain | `src/chemistry/diversity.py` |
| parallel-web | Research | `docs/research/` (27 pages scraped) |
| paper-lookup | Research | `paper/references.bib` |
| literature-review | Research | `docs/maestro-gigachain-gigaevo-research.md` |
| scientific-brainstorming | Research | `autochem_reportandplan.md` |
| scientific-writing | Research | `paper/main.tex` |
| peer-review | Research | Paper quality validation |
| database-lookup | Research | Seed reaction validation (PubChem) |
| get-available-resources | Infra | Pipeline batch size + parallelism config |
| exploratory-data-analysis | Infra | `src/chemistry/diversity.py`, `src/compilation/quality.py` |

</details>

---

## Dependencies

- **Python 3.13+**, Pydantic v2
- **RDKit** — structural validation, descriptors, conformers, MMFF94
- **NetworkX** — chemical knowledge graph
- **Loguru** — structured logging
- **OmegaConf + Hydra** — configuration
- **LangChain (langchain-openai)** — LLM client
- **Redis** (optional) — distributed state
- **HuggingFace Datasets** — export/upload

## Citation

```bibtex
@misc{auto-cheminstruct-2026,
  author = {Kumar, Aayush},
  title = {Auto-ChemInstruct: Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/aayushkrm/auto-cheminstruct}
}
```

## License

MIT — see [LICENSE](LICENSE)
