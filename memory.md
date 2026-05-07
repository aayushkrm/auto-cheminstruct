# Auto-ChemInstruct Memory & Roadmap

## Project Identity
- **Name**: Auto-ChemInstruct
- **Full Title**: Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models
- **Domain**: AI Cheminformatics, Multi-Agent Systems, RAG
- **Publication Target**: NeurIPS Datasets & Benchmarks Track

## Status: CODE-COMPLETE — Ready for scale-up + paper writing

## Repository Structure
```
auto-cheminstruct/
├── AGENTS.md, memory.md, tasks.md   # Documentation
├── pyproject.toml                   # uv-managed deps
├── configs/default.yaml             # Fireworks AI + scheduler config
├── docs/
│   ├── architecture.md              # Full system design
│   └── configuration.md             # Config reference
├── paper/
│   ├── main.tex                     # NeurIPS LaTeX scaffold
│   └── references.bib               # Bibliography
├── benchmarks/                      # Ablation + comparison outputs
├── tests/                           # 90 tests (5 test files)
│   └── test_rag.py                  # 23 RAG tests
├── src/
│   ├── agents/                      # 4 agents (hypothesis, verify, reflect, compile)
│   ├── pipeline/orchestrator.py     # Self-bootstrapping coordinator
│   ├── rag/chemical_rag.py          # Multi-hop RAG + knowledge graph
│   ├── chemistry/                   # RDKit, xTB, diversity, feasibility
│   ├── benchmarks/                  # Ablation + ChemCoT comparison
│   ├── data/models.py               # All Pydantic models
│   ├── config.py                    # OmegaConf + Pydantic config
│   ├── utils/
│   │   ├── llm_factory.py
│   │   └── temperature_scheduler.py
│   └── cli/main.py                  # Typer CLI (pipeline, ablation, chemcot, config)
└── datasets/                        # JSONL + HuggingFace output
```

## Implemented Innovations

### 1. Self-Bootstrapping Reflection Loop
- `LearningContext` accumulates failure patterns across bootstrap iterations
- Causal reflection traces analyzed → constraints injected into hypothesis prompt
- Pipeline: Generate → Verify → Reflect → Accumulate → Repeat (with learned constraints)
- CLI: `-B N` flag for bootstrap iterations

### 2. Temperature Scheduling
- Cosine annealing (1.0→0.3) across bootstrap iterations
- Configurable: cosine, linear, exponential schedules
- Exploration (high temp) → Exploitation (low temp) as learning accumulates

### 3. Multi-Hop Chemical RAG
- ChromaDB vector store + NetworkX chemical knowledge graph
- Typed edges: reactant_of, product_of, has_scaffold, contains_group
- Multi-hop retrieval: extract SMILES → formulate query → repeat
- Scaffold indexing (Murcko decomposition) + functional group detection (SMARTS)
- Wired into orchestrator for prompt enrichment and post-verification indexing

### 4. Chemical Feasibility Filter
- Unstable group detection (peroxides, azides, diazo)
- Hypervalent atom checks, ring strain analysis
- Atom count limits, heavy atom detection

### 5. Enhanced Preference Pair Quality Scoring
- 6 chemistry-aware dimensions: structural validity, QED, reflection depth, yield, scaffold diversity, reaction specificity
- Weighted composite score (0-1)

## Test Coverage
- **90/90 passing** (59 unit + 8 integration + 23 RAG)
- No failures, 0 skipped

## CLI Commands
```bash
pipeline -n N -B M      # Run pipeline with bootstrap
ablation -n N -o DIR    # 4-variant ablation study
chemcot -d DIR -o FILE  # ChemCoTBench comparison
config                  # Show current configuration
```

## Key Decisions Log
- Fireworks AI deepseek-v4-pro (reasoning model) — batch_size=1 optimal
- Cosine temperature scheduling — paper's exploration-exploitation mechanism
- NetworkX for chemical knowledge graph — lightweight, no external DB needed
- RDKit-only validation active; xTB code-complete, pending binary

## Next Steps
1. Scale to 100+ hypotheses (needs ~2.5 hours of LLM calls)
2. Run full ablation study with N=20+ per variant
3. Fill paper results with real ablation data
4. Install xTB binary for energetic validation
5. Docker image + HuggingFace upload + arXiv preprint
