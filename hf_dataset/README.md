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
| Total pairs | **110** |
| Splits | 81 train / 6 val / 23 test |
| Reaction types | 13 distinct |
| Unique molecules | 187 |
| Tanimoto diversity | 0.879 |
| Scaffold diversity | 33.2% |
| Avg quality score | 0.636 (6-dim rubric) |
| Reflection coverage | 72.8% |

## Ablation Results

| Variant | Pairs | Pass% | Diversity | Quality |
|---|---|---|---|---|
| **Full-System** | 15 | 78.9% | 0.877 | **0.653** |
| No-Bootstrap | 6 | 100% | **0.901** | 0.576 |
| No-Reflection | 17 | 89.5% | 0.878 | 0.568 |
| No-RAG | 15 | 71.4% | 0.876 | 0.660 |

- **Bootstrap** → 2.5× more pairs
- **Reflection** → +15.4% quality
- **RAG** → +7.5 pp pass rate

## Data Format

Each preference pair contains:
- `prompt`: Instruction for the model (e.g., "Propose a Diels-Alder reaction...")
- `chosen`: Passed reaction with mechanism, conditions, and yield
- `rejected`: Failed reaction with causal failure analysis and reflection trace
- `reaction_type`: One of 13 distinct named reaction types
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

Auto-ChemInstruct uses a 4-agent pipeline:

1. **Hypothesis Agent** — LLM generates creative reaction pathways (deepseek-v3p2, T=0.9)
2. **Verification Agent** — RDKit structural validation + MMFF94 force-field energetics
3. **Reflection Agent** — Causal failure analysis with chemical principles
4. **Compilation Agent** — DPO preference pairs with quality scoring

The pipeline includes a self-bootstrapping loop where failed reactions inform subsequent generations, and cosine temperature annealing (1.0→0.3) for exploration→exploitation.

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
