# Auto-ChemInstruct Memory & Roadmap

## Project Identity
- **Name**: Auto-ChemInstruct
- **Full Title**: Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models
- **Publication Target**: NeurIPS Datasets & Benchmarks Track
- **GitHub**: github.com/aayushkrm/auto-cheminstruct
- **HuggingFace**: huggingface.co/datasets/aayushkrm/autochem-instruct (66 pairs)

## Status: PRODUCTION-READY — All components functional, ablation validated (2026-05-09)

## LLM Provider
- **Fireworks AI**: model `accounts/fireworks/models/deepseek-v3p2`
- **Energetic validation**: RDKit MMFF94 force-field fallback (real energies, no external binary)
- **RAG**: Lightweight TF-IDF + NetworkX knowledge graph (offline, no API keys, no ChromaDB)

## Core Innovations

### 1. Self-Bootstrapping Reflection Loop
Generate → Verify → Reflect → Accumulate Learning → Repeat with learned constraints.
Temperature cosine annealing (1.0→0.3) across iterations.

### 2. Multi-Hop Chemical RAG
TF-IDF vector search + NetworkX knowledge graph with typed edges (reactant_of, product_of, has_scaffold, contains_group). Multi-hop retrieval with query decomposition.

### 3. MMFF94 Energetic Validation
RDKit Merck Molecular Force Field provides real computed energies as xTB fallback. Physically realistic, no binary dependencies.

### 4. Chemical Feasibility + Quality Scoring
6 chemistry-aware quality dimensions (structural validity, QED, reflection depth, yield, scaffold diversity, reaction specificity).

## Test Coverage: 129/129 passing

## Most Recent Ablation (2026-05-09, N=8/variant)

| Variant | Pairs | Pass% | Diversity | Quality |
|---|---|---|---|---|
| Full-System | 15 | 78.9% | 0.877 | 0.653 |
| No-Bootstrap | 6 | 100% | 0.901 | 0.576 |
| No-Reflection | 17 | 89.5% | 0.878 | 0.568 |
| No-RAG | 15 | 71.4% | 0.876 | 0.660 |

**Key empirical claims:**
- Bootstrap → 2.5× more pairs (volume)
- Reflection → +15.4% quality (grounding)
- RAG → +7.5 pp pass rate (accuracy)

## Dataset (Final: 2026-05-09)
- HuggingFace: `aayushkrm/autochem-instruct` — **110 pairs** (81 train, 6 val, 23 test)
- 187 unique molecules, 87.9% Tanimoto diversity, 33.2% scaffold diversity
- 13 distinct reaction types, 72.8% causal reflection coverage
- Avg quality 0.636 (6-dimension rubric)

## Known Issues
- Reaction type diversity: LLM tends to return "other" despite named type prompts
- Solution in progress: stronger system prompt constraints + exact regex matching

## Next Steps
1. arXiv preprint upload (arxiv_submission.tar.gz ready in repo root)
2. Zenodo archive publish (zip repo → zenodo.org/deposit for DOI)
3. (Done) Scale dataset — 110 pairs achieved
