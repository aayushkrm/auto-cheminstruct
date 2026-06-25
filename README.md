# Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry**

[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/aayushkrm/autochem-instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-121%2F121-brightgreen)](https://github.com/aayushkrm/auto-cheminstruct)

An autonomous, self-verifying multi-agent pipeline that generates physically-validated instruction datasets for chemistry domain-specific language models. Targets the NeurIPS Datasets & Benchmarks track.

## Pipeline

```
Hypothesis Agent → Verification Agent (RDKit + MMFF94) ──PASS→ Compilation Agent → DPO Pairs
        ↑                         │                                          │
        │                         └──FAIL→ Reflection Agent ─────────────────┘
        │                                      │
        └────── Learning Context ←─────────────┘
              (Self-Bootstrapping Loop)
```

## Key Features

- **4-agent hub-spoke system**: Hypothesis → Verify → Reflect → Compile
- **Self-bootstrapping loop**: Failed reactions → causal analysis → learned constraints → better hypotheses
- **Temperature cosine annealing**: 1.0→0.3 across bootstrap iterations for exploration→exploitation
- **MMFF94 energetic validation**: Real computed energies via RDKit force fields — no external binary needed
- **TF-IDF chemical RAG**: Lightweight retrieval + NetworkX knowledge graph — offline, zero API keys
- **6-dimension quality scoring**: Structural validity, QED, reflection depth, yield, scaffold diversity, reaction specificity
- **19 named reaction types** with alias matching (Diels-Alder → diels_alder, Suzuki → suzuki_coupling, etc.)

## Ablation Results (N=8/variant)

| Variant | Pairs | Pass% | Diversity | Quality |
|---|---|---|---|---|
| **Full-System** | 15 | 78.9% | 0.877 | **0.653** |
| No-Bootstrap | 6 | 100% | **0.901** | 0.576 |
| No-Reflection | 17 | 89.5% | 0.878 | 0.568 |
| No-RAG | 15 | 71.4% | 0.876 | 0.660 |

**Key Findings:**
- **Bootstrap** → 2.5× more pairs (volume driver)
- **Reflection** → +15.4% quality improvement (grounding driver)
- **RAG** → +7.5 pp pass rate (accuracy driver)

## Dataset (v1.0)

| Metric | Value |
|---|---|
| Total pairs | **110** (81 train / 6 val / 23 test) |
| Reaction types | 13 distinct |
| Unique molecules | 187 |
| Tanimoto diversity | 0.879 |
| Scaffold diversity | 33.2% |
| Avg quality score | 0.636 |
| Reflection coverage | 72.8% |

📦 **Download:** `datasets.load_dataset("aayushkrm/autochem-instruct")`

## Quick Start

```bash
# Install
cd auto-cheminstruct && uv sync --extra chemistry

# Configure (edit configs/default.yaml or set env vars)
# LLM: Fireworks AI (OpenAI-compatible), model deepseek-v3p2

# Run pipeline (full system, 10 hypotheses, 3 bootstrap iterations)
uv run python -m src.cli.main pipeline -n 10 -B 3

# Run ablation study (4 variants)
uv run python -m src.cli.main ablation -n 8 -o benchmarks

# Compare against ChemCoTBench
uv run python -m src.cli.main chemcot -d datasets/autochem-merged/

# Run tests
uv run pytest  # 121/121 passing

# View config
uv run python -m src.cli.main config
```
https://github.com/aayushkrm/auto-cheminstruct

## Links

- 🤗 [HuggingFace Dataset](https://huggingface.co/datasets/aayushkrm/autochem-instruct)
- 🏗️ [GitHub Repository](https://github.com/aayushkrm/auto-cheminstruct)
