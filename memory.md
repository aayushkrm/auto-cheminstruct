# Auto-ChemInstruct Memory & Roadmap

## Project Identity
- **Name**: Auto-ChemInstruct
- **Full Title**: Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models
- **Domain**: AI Cheminformatics, Multi-Agent Systems, RAG
- **Publication Target**: NeurIPS Datasets & Benchmarks Track
- **GitHub**: github.com/aayushkrm/auto-cheminstruct
- **HuggingFace**: huggingface.co/datasets/aayushkrm/autochem-instruct

## Status: PRODUCTION-READY — All components functional, ablation validated

## LLM Provider
- **Fireworks AI**: model `accounts/fireworks/models/deepseek-v3p2`
- **Energetic validation**: RDKit MMFF94 force-field fallback (real energies, no external binary needed)
- **RAG**: Lightweight TF-IDF + NetworkX knowledge graph (no API keys, offline)
- Config: `configs/default.yaml`

## Repository Structure
```
auto-cheminstruct/
├── AGENTS.md, memory.md, tasks.md   # Project documentation
├── pyproject.toml                   # uv-managed deps
├── configs/default.yaml             # Fireworks AI + scheduler config
├── .pre-commit-config.yaml          # Pre-commit hooks (ruff, mypy)
├── .zenodo.json                     # Zenodo DOI metadata
├── Dockerfile                       # Reproducibility
├── docs/
│   ├── architecture.md              # Full system design
│   └── configuration.md             # Config reference
├── paper/
│   ├── main.tex                     # NeurIPS LaTeX (filled with 66-pair stats)
│   └── references.bib               # Bibliography
├── benchmarks/                      # Ablation + ChemCoT comparison outputs
├── tests/                           # 129 tests (9 test files)
│   ├── test_models.py (20)          # Core data models
│   ├── test_agents.py (18)          # Agent logic
│   ├── test_chemistry.py (17)       # RDKit, feasibility
│   ├── test_rag.py (23)             # RAG + knowledge graph
│   ├── test_config.py (4)           # Config loading
│   ├── test_integration.py (8)      # End-to-end pipeline
│   ├── test_temperature.py (13)     # Temperature scheduling
│   ├── test_feasibility.py (13)     # Chemical feasibility filter
│   └── test_quality.py (13)         # Quality scoring module
├── src/
│   ├── agents/                      # 4 agents
│   │   ├── hypothesis_agent.py
│   │   ├── verification_agent.py
│   │   ├── reflection_agent.py
│   │   └── compilation_agent.py
│   ├── pipeline/orchestrator.py     # Self-bootstrapping coordinator
│   ├── rag/chemical_rag.py          # Multi-hop RAG + knowledge graph
│   ├── chemistry/                   # RDKit, xTB, diversity, feasibility
│   ├── benchmarks/                  # Ablation + ChemCoT comparison
│   │   ├── ablation.py
│   │   └── chemcot_comparison.py
│   ├── compilation/quality.py       # Standalone quality scoring module
│   ├── data/models.py               # All Pydantic models
│   ├── config.py                    # OmegaConf + Pydantic config
│   ├── utils/
│   │   ├── llm_factory.py
│   │   └── temperature_scheduler.py
│   └── cli/main.py                  # Typer CLI (all commands functional)
└── datasets/                        # Local JSONL output (gitignored)
```

## Implemented Innovations

### 1. Self-Bootstrapping Reflection Loop
- `LearningContext` accumulates failure patterns across bootstrap iterations
- Causal reflection traces analyzed → constraints injected into hypothesis prompt
- Pipeline: Generate → Verify → Reflect → Accumulate → Repeat
- CLI: `-B N` flag for bootstrap iterations

### 2. Temperature Scheduling
- Cosine annealing (1.0→0.3) across bootstrap iterations
- Configurable: cosine, linear, exponential schedules
- Exploration (high temp) → Exploitation (low temp)

### 3. Multi-Hop Chemical RAG
- ChromaDB vector store + NetworkX chemical knowledge graph
- Typed edges: reactant_of, product_of, has_scaffold, contains_group
- Multi-hop retrieval with query decomposition
- Scaffold indexing (Murcko) + functional group detection (SMARTS)

### 4. Chemical Feasibility Filter
- Unstable group detection (peroxides, azides, diazo, acid chlorides, sulfonyl chlorides)
- Hypervalent atom checks, ring strain analysis
- Atom count limits

### 5. Enhanced Quality Scoring
- 6 chemistry-aware dimensions
- Standalone module: `src/compilation/quality.py`
- Weighted composite score (0-1)

## Test Coverage
- **129/129 passing** (59 unit + 8 integration + 23 RAG + 39 quality/feasibility/temperature)
- No failures, 0 skipped

## CLI Commands (All Functional)
```bash
pipeline -n N -b BATCH -B BOOTSTRAP  # Full pipeline
verify -s SESSION_ID --list          # Verify from checkpoint
reflect -s SESSION_ID --list         # Reflect from checkpoint
compile -s SESSION_ID --list         # Compile from checkpoint
status -s SESSION_ID --list          # Session status
ablation -n N -o DIR                 # 4-variant ablation
chemcot -d DIR -o FILE               # ChemCoTBench comparison
config                               # Show configuration
```

## Dataset
- **HuggingFace**: `aayushkrm/autochem-instruct` — 66 pairs (51 train, 6 val, 9 test)
- 125 unique molecules, 87.4% Tanimoto diversity, 34.4% scaffold diversity
- 100% causal reflection coverage, avg quality score 0.596

## Bugs Fixed
- reflection_agent: `trace.failure_category` → iterate `failure_categories` list
- orchestrator: `h.reaction_name` → `h.reaction_type.value`
- models.py: `datetime.utcnow()` → `datetime.now(timezone.utc)`
- ablation.py: PreferencePair `.get()` → `getattr()` (Pydantic model, not dict)
- .gitignore: `benchmarks/` → `/benchmarks/` (was blocking src/benchmarks/)

## Key Decisions Log
- 2026-05-08: Switched LLM from deepseek-v4-pro to deepseek-v3p2 (Fireworks billing)
- Cosine temperature scheduling — paper's exploration-exploitation mechanism
- NetworkX for chemical knowledge graph — lightweight, no external DB needed
- RDKit-only validation active; xTB code-complete, pending binary

## Next Steps
1. arXiv preprint upload (compile LaTeX → upload)
2. Zenodo archive publish (zip repo + dataset → zenodo.org for DOI)
3. Scale dataset to 100+ pairs with improved reaction type diversity
4. Fix reaction type diversity (LLM not following named types — prompt refinement needed)
