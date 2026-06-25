# Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry**

[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/aayushkrm/autochem-instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-121%2F121-brightgreen)](https://github.com/aayushkrm/auto-cheminstruct)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)

> Internship: TSU Laboratory of AI in Chemistry & Molecular Engineering × AIRI Institute
>
> Target: NeurIPS Datasets & Benchmarks Track

---

An autonomous multi-agent pipeline that generates **physically-validated instruction data** for chemistry domain-specific language models. Each reaction hypothesis is structurally verified (RDKit), energetically validated (MMFF94 force fields), and causally analyzed on failure — producing RLHF/DPO preference pairs with rich reasoning traces.

## Architecture

```
Hypothesis Agent → Verification Agent (RDKit + MMFF94) ──PASS→ Compilation Agent → DPO Pairs
        ↑                         │                                          │
        │                         └──FAIL→ Reflection Agent ─────────────────┘
        │                                      │
        └────── Learning Context ←─────────────┘
              (Self-Bootstrapping Loop)
```

### Core Innovation: Self-Bootstrapping Learning Loop

```
Generate → Verify → Reflect → Accumulate → Repeat (with learned constraints)
```

Failed reactions are analyzed for root causes (10 failure categories), which are accumulated into a LearningContext and injected as constraints back into the hypothesis agent's prompt. Temperature cosine annealing (1.0→0.3) drives exploration→exploitation across bootstrap iterations.

## What's Completed

### Core Pipeline (4 Agents)

| Component | Description | Status |
|---|---|---|
| **Hypothesis Agent** | LLM-based generation of diverse chemical reactions (19 named types) | ✅ |
| **Verification Agent** | RDKit structural validation + MMFF94 energetic validation + feasibility checks | ✅ |
| **Reflection Agent** | 10 failure categories, structured causal analysis, LearningContext injection | ✅ |
| **Compilation Agent** | DPO preference pairs (chosen=passed, rejected=failed+reflection), 6-dim quality scoring | ✅ |
| **Orchestrator** | SQLite checkpoints, full state machine, bootstrap loop, RAG enrichment | ✅ |

### Dataset (v1.0)

- **110 DPO pairs** (81 train / 6 val / 23 test), published on HuggingFace
- 187 unique molecules, 87.9% Tanimoto diversity, 33.2% scaffold diversity
- 13 distinct reaction types, 72.8% causal reflection coverage
- Avg quality score: 0.636 (6-dimension rubric)

### Benchmarking

- **Ablation study** (N=8/variant): Full-System vs No-Bootstrap vs No-Reflection vs No-RAG
  - Bootstrap → 2.5× more pairs (volume driver)
  - Reflection → +15.4% quality improvement (grounding driver)
  - RAG → +7.5 pp pass rate (accuracy driver)
- **ChemCoTBench comparison** module for standardized evaluation
- **121/121 tests passing** (unit + integration + RAG + quality + feasibility + temperature)

### Paper (NeurIPS LaTeX)

Complete scaffold with:
- Real ablation results in tables
- ChemCoTBench comparison
- Related work with ChemCrow, Coscientist, GigaEvo, AlphaEvolve, MAP-Elites citations
- arXiv submission package ready (`arxiv_submission.tar.gz`)

### External Tool Evaluations

27 research documents covering integration analysis of AIRI ecosystem tools:

| Tool | Repository | Assessment | Docs |
|---|---|---|---|
| **GigaEvo** | [AIRI-Institute/gigaevo-core](https://github.com/AIRI-Institute/gigaevo-core) | ✅ Integrate — replaces simple bootstrap with MAP-Elites evolutionary search | `docs/research/gigaevo_*.md` (12 files) |
| **Maestro (CARL)** | [AIRI-Institute/maestro-core](https://github.com/AIRI-Institute/maestro-core) | ✅ Integrate CARL — structures Reflection Agent as formal reasoning chains | `docs/research/maestro_*.md` (5 files) |
| **GigaChain** | [ai-forever/gigachain](https://github.com/ai-forever/gigachain) | ❌ Skip — GigaChat-specific, no alignment with Fireworks LLM | `docs/research/langchain_gigachat*.md` (3 files) |
| **ChemCrow** | [ur-whitelab/chemcrow-public](https://github.com/ur-whitelab/chemcrow-public) | 📄 Cite only — single-agent, no reflection/verification | `docs/research/chemcrow_public.md` |
| **AiZynthFinder** | [MolecularAI/aizynthfinder](https://github.com/MolecularAI/aizynthfinder) | ❌ Skip — retrosynthesis direction (product→reactants), opposite of our forward generation | `docs/research/aizynthfinder.md` |

## Quick Start

```bash
# Install
cd auto-cheminstruct && uv sync

# Run pipeline (full system, 5 hypotheses, 3 bootstrap iterations)
uv run python -m src.cli.main pipeline -n 5 -B 3

# Run scaled pipeline (20 hypotheses)
uv run python -m src.cli.main pipeline -n 20

# Run ablation study (all 4 variants)
uv run python -m src.cli.main ablation -n 8 -o benchmarks

# Compare against ChemCoTBench
uv run python -m src.cli.main chemcot -d datasets/autochem-merged/

# Run tests
uv run pytest

# Type check
uv run mypy src/

# Docker
docker build -t auto-cheminstruct .
```

## In Progress / Next Up

### Current Integration Sprint

1. **GigaEvo Integration** — Replace simple temperature-scheduled bootstrap with proper MAP-Elites evolutionary search:
   - Wrapping our 4 agents as GigaEvo DAG stages
   - Chemistry-specific behavior grid (pass rate × diversity × reaction type richness)
   - LLM-driven mutation operators with insight generation
   - Multi-island parallel evolution

2. **Maestro CARL Integration** — Restructure Reflection Agent as a CARL ReasoningChain:
   - 5–8 structured causal analysis steps (symptom → root cause → constraint → learning)
   - DAG-parallelized reasoning with RAG context extraction
   - Per-step LLM configuration

### Pipeline

- [ ] GigaEvo DAG stage wrappers (GenerateStage, VerifyStage, ReflectStage, ScoreStage)
- [ ] Chemistry-specific MAP-Elites behavior space definition
- [ ] Multi-island evolution strategy (pass rate island / diversity island / coverage island)
- [ ] CARL-based Reflection Agent with structured reasoning chains
- [ ] Run scaled experiments (100+ generations) and compare against current bootstrap baseline
- [ ] Update paper with GigaEvo integration results
- [ ] Submit to arXiv + Zenodo archive

## Dependencies

- **Python 3.13+**, Pydantic v2 for all data models
- **RDKit** for structural chemistry (validation, descriptors, conformers, MMFF94 force fields)
- **NetworkX** for chemical knowledge graph
- **Loguru** for logging, **OmegaConf** for configuration
- **Fireworks AI** (OpenAI-compatible) — model `accounts/fireworks/models/deepseek-v3p2`
- **Tavily** for web research, **Firecrawl** for web scraping

## Links

- 🤗 [HuggingFace Dataset](https://huggingface.co/datasets/aayushkrm/autochem-instruct)
- 🏗️ [GitHub Repository](https://github.com/aayushkrm/auto-cheminstruct)
- 📖 [Research Docs](docs/research/) (GigaEvo, Maestro, GigaChain, ChemCrow evaluations)

## Citation

```bibtex
@misc{auto-cheminstruct-2025,
  author = {Kumar, Aayush and ...},
  title = {Auto-ChemInstruct: Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/aayushkrm/auto-cheminstruct}
}
```
