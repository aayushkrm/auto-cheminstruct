# Auto-ChemInstruct Memory & Status

## Project Identity
- **Name**: Auto-ChemInstruct
- **Full Title**: Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models
- **Publication Target**: NeurIPS Datasets & Benchmarks Track
- **GitHub**: github.com/aayushkrm/auto-cheminstruct
- **HuggingFace**: huggingface.co/datasets/aayushkrm/autochem-instruct (181 pairs)

## Status: COMPLETE — 230/230 tests, defense-ready (2026-06-30)

## LLM Provider
- **Fireworks AI**: model `accounts/fireworks/models/minimax-m3`
- **API Key**: `FIREWORKS_API_KEY` environment variable (no hardcoded keys)
- **JSON Mode**: `response_format=json_object` for reliable parsing
- **Rate Limiting**: 8s delay between LLM calls (configurable)

## Implementation Summary

### Phase I: Foundation (df5a2ca)
- Redis state layer with in-memory fallback
- 5 Hydra YAML hierarchical configs
- Problem directory (5 seed reactions, validate.py, metrics.yaml)
- 140 tests

### Phase II: DAG Engine (a522fb5)
- Async DAGPipeline with Kahn topological sort
- Semaphore-bounded parallelism, failure propagation
- 8 Pydantic I/O models, 4 agent factory functions
- linear_parity_check() for DAG vs sequential comparison
- 163 tests

### Phase III: MAP-Elites (1467e7e)
- 2,600-cell behavior grid (26×10×10)
- 5 mutation operators: reactant substitution, condition optimization, reaction crossover, scaffold hopping, insight-guided
- 4 specialist islands (diversity, quality, novelty, yield) with conservative migration
- Seeding, stagnation convergence (10 gens), deterministic RNG
- 202 tests

### Phase IV: CARL Chains (f5d3dcf)
- 4-step parallel DAG reflection: StericAnalysis → ElectronicAnalysis → ThermodynamicAnalysis → CausalSynthesis
- CARLReflectionAgent with batch filtering
- Chemistry-specific LLM prompts for each step
- 219 tests

### Phase V: Ablation + Paper (1834c45)
- 7-variant evolution ablation (Baseline, Bootstrap, CARL, MAP-Elites, No-Reflection, No-RAG, Full-System)
- CLI --mode evolution flag
- Updated NeurIPS paper with evolution results
- 230 tests

### Dataset v2.0 (906fb9b)
- Merged dataset: 181 DPO pairs (136 train + 45 test)
- 19 reaction types, avg quality 0.650
- v2.0: 71 pairs at 82.6% pass rate (MiniMax-M3)
- HuggingFace updated

### Final Cleanup (d9dab54)
- API key to env var, .env.example added
- Stale deps removed, old checkpoints/datasets cleaned
- Reflection agent JSON mode
- 230 tests

## GigaEvo + Maestro CARL Integration

### Architecture Alignment

| GigaEvo Component | Our Implementation | Lines | Tests |
|---|---|---|---|
| Redis Database | src/evolution/redis_store.py | 229 | 19 |
| Async DAG Engine | src/evolution/dag.py | 293 | 23 |
| MAP-Elites Engine | src/evolution/map_elites.py | 491 | 39 |
| Problem Interface | problems/autochem/ | 349 | — |

| Maestro CARL Component | Our Implementation | Lines | Tests |
|---|---|---|---|
| Event-Action-Result Chains | src/carl/chain.py | 463 | 17 |
| StepDescription/ReasoningChain | CARLChain with DAGPipeline | — | — |
| Parallel DAG Execution | Steps 1-3 run in parallel | — | — |

Both are custom-built from scratch following AIRI's published architecture patterns. No external GigaEvo or Maestro packages exist on PyPI.

## Dataset

| Metric | v1.0 | v2.0 | Merged |
|--------|------|------|--------|
| Pairs | 110 | 71 | 181 |
| Reaction Types | 13 | 18 | 19 |
| Pass Rate | 65.9% | 82.6% | 71.5% |
| Avg Quality | 0.636 | 0.671 | 0.650 |
| Splits | 81/6/23 | 49/0/22 | 136/45 test |

## Key Results
- Self-bootstrapping: 2.5× more pairs
- Reflection: +15.4% quality improvement
- RAG: +7.5pp pass rate
- Full-System evolution: 3.5× elites, +17% pass rate over Baseline

## Known Limitations
- Reflection traces not embedded in v2.0 output pairs (MiniMax-M3 reflection JSON parsing issue — fixed in final cleanup with llm_json)
- MiniMax-M3 occasional invalid SMILES (e.g., Grignard reagents)
- xTB binary not installed (MMFF94 fallback used instead)
- No arXiv/Zenodo submission yet

## Next Steps
- [ ] arXiv preprint upload
- [ ] Zenodo archive publish (DOI)
- [ ] Presentation/defense preparation

## Command Code Skills Audit

The project was built using Command Code's agent skills system (`.commandcode/skills/` — 35 skill modules).
19 were directly used; 16 were available but not needed.

### Skills → Code Implementation Map

| Skill | Usage | Deliverable |
|-------|-------|-------------|
| hypothesis-generation | Prompt engineering, 8 templates, JSON extraction | src/agents/hypothesis_agent.py |
| physical-verification | 7-step validation cascade, xTB/MMFF94 | src/agents/verification_agent.py, src/chemistry/ |
| causal-reflection | 10 categories, CARL 4-step chain | src/agents/reflection_agent.py, src/carl/chain.py |
| dataset-compilation | DPO pairs, 6-dim quality, HF export | src/agents/compilation_agent.py, src/compilation/quality.py |
| agent-architecture | Hub-spoke topology, DAG, state machine | src/pipeline/orchestrator.py, src/evolution/ |
| code-standards | Types, Pydantic, Loguru, pre-commit | All 30 source files |
| rdkit | SMILES, descriptors, MMFF94, scaffolds | src/chemistry/rdkit_wrapper.py |
| networkx | Chemical knowledge graph, multi-hop RAG | src/rag/chemical_rag.py |
| medchem | Drug-likeness, PAINS, structural alerts | src/chemistry/rdkit_wrapper.py (feasibility) |
| molfeat | ECFP/MACCS fingerprints | src/chemistry/diversity.py |
| parallel-web | AIRI framework docs (27 pages scraped) | docs/research/ |
| paper-lookup | GigaEvo, AlphaEvolve, ChemCoTBench citations | paper/references.bib |
| literature-review | AIRI ecosystem systematic analysis | docs/maestro-gigachain-gigaevo-research.md |
| scientific-brainstorming | 4 architecture proposals, Auto-ChemInstruct selected | autochem_reportandplan.md |
| scientific-writing | NeurIPS LaTeX, IMRAD, ablation tables | paper/main.tex |
| peer-review | Paper methodology validation | Quality assurance |
| database-lookup | PubChem reference data for seeds | problems/autochem/initial_programs/ |
| get-available-resources | CPU/memory/disk → batch size config | Pipeline configuration |
| exploratory-data-analysis | Diversity metrics, quality distributions | src/chemistry/diversity.py |
