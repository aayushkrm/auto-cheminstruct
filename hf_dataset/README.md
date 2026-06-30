---
license: mit
language:
- en
tags:
- chemistry
- rlhf
- dpo
- multi-agent
- preference-pairs
- cheminformatics
- molecular-reactions
- scientific-ai
- instruction-dataset
pretty_name: Auto-ChemInstruct
size_categories:
- 100<n<1K
task_categories:
- text-generation
- question-answering
viewer: true
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
  - split: test
    path: test.jsonl
---

# Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry**

An autonomous, self-verifying multi-agent pipeline that generates physically-validated DPO/RHLF preference pairs for chemistry instruction tuning.

## Dataset Stats

| Metric | Value |
|---|---|
| Total pairs | **181** |
| Splits | 130 train / 6 val / 45 test |
| Reaction types | 19 distinct |
| Avg quality score | 0.650 (6-dim rubric) |
| Pass rate | 71.5% (253 total hypotheses) |
| Self-bootstrapping | 3 iterations, cosine annealing (1.0→0.3) |

## Ablation Results (7-variant evolution study)

| Variant | Elites | Coverage | Pass Rate | Quality |
|---------|--------|----------|-----------|---------|
| **Full-System** (ME + CARL + Bootstrap + RAG + Reflection) | 25 | 1.0% | **84.6%** | **0.72** |
| Baseline | 10 | 0.4% | 69.0% | 0.59 |
| MAP-Elites-Only | 24 | 0.9% | 75.0% | 0.60 |
| CARL-Only | 15 | 0.4% | 78.8% | 0.67 |
| No-Reflection | 10 | 0.4% | 66.3% | 0.59 |
| No-RAG | 13 | 0.4% | 68.0% | 0.65 |
| Bootstrap-Only | 11 | 0.4% | 68.9% | 0.61 |

- **MAP-Elites** → 2.5× more elites (population)
- **CARL** → +14% pass rate (quality)
- **Full-System** → 3.5× elites, +17% pass rate vs Baseline

## Data Format

Each preference pair contains:
- `prompt`: Instruction for the model (e.g., "Propose a Diels-Alder reaction...")
- `chosen`: Passed reaction with mechanism, conditions, and yield
- `rejected`: Failed reaction with causal failure analysis and reflection trace
- `reaction_type`: One of 19 distinct named reaction types
- `quality_score`: 6-dimensional composite score (0-1)

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("aayushkrm/autochem-instruct")
print(dataset["train"][0]["prompt"])
print(dataset["train"][0]["chosen"])
print(dataset["train"][0]["rejected"])
```

## How It Was Generated

Auto-ChemInstruct uses a 4-agent pipeline with MAP-Elites evolutionary search and CARL reasoning chains:

1. **Hypothesis Agent** — LLM generates creative reaction pathways (MiniMax-M3, JSON mode)
2. **Verification Agent** — RDKit structural validation + MMFF94 force-field energetics
3. **Reflection Agent** — CARL 4-step causal analysis (Steric→Electronic→Thermo→Synthesis)
4. **Compilation Agent** — DPO preference pairs with 6-dim quality scoring

The pipeline includes a self-bootstrapping loop with cosine temperature annealing (1.0→0.3) and MAP-Elites evolutionary optimization across a 2,600-cell behavior grid.

## Citation

```bibtex
@software{autochem_instruct,
  title = {Auto-ChemInstruct: Agent-Driven Synthesization of RLHF Data for Chemistry DSLMs},
  author = {Stepanov, Deyan and Kundu, Akash},
  year = {2026},
  url = {https://github.com/aayushkrm/auto-cheminstruct},
}
```

## Links

- 🏗️ [GitHub](https://github.com/aayushkrm/auto-cheminstruct)
- 📄 [Paper (NeurIPS 2026 submission)](https://github.com/aayushkrm/auto-cheminstruct/tree/main/paper)
