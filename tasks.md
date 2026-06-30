# Auto-ChemInstruct Task Checklist

## Phase I: Infrastructure & Environment ✅
- [x] Repository setup, pyproject.toml, .gitignore, .env.example
- [x] venv via uv (Python 3.13), RDKit verified
- [x] NetworkX installed for chemical knowledge graph
- [x] LLM provider: Fireworks AI (deepseek-v3p2 → minimax-m3)
- [x] All core imports verified
- [x] Initialize git repository, pre-commit hooks (ruff, ruff-format)
- [x] Docker build + docker-compose with Redis

## Phase II: Core Data Models ✅
- [x] ChemicalEntity, ReactionHypothesis, VerificationResult, ReflectionTrace
- [x] PreferencePair (chosen/rejected), PipelineStatus, SessionState
- [x] LearningContext (self-bootstrapping knowledge accumulator)
- [x] 19 ReactionType enum values

## Phase III: Core Modules ✅

### 3.1 Chemistry Utilities ✅
- [x] RDKit wrapper (validate, conformers, descriptors, steric clash)
- [x] Chemical feasibility filter (unstable groups, ring strain, hypervalent atoms)
- [x] xTB interface (MMFF94 fallback active, xTB binary optional)
- [x] Molecular diversity metrics (Tanimoto, scaffold analysis)

### 3.2 Hypothesis Generation Agent ✅
- [x] System prompt for 19 reaction types
- [x] JSON extraction with markdown code block handling
- [x] Learning context injection (self-bootstrapping feedback)
- [x] JSON mode support (response_format=json_object) for MiniMax-M3

### 3.3 Verification Agent ✅
- [x] RDKit structural validation cascade
- [x] Chemical feasibility check
- [x] Descriptor computation (SA, QED, LogP, MW, TPSA)
- [x] MMFF94 energetic validation (xTB graceful fallback)

### 3.4 Reflection Agent ✅
- [x] 10 failure categories
- [x] Structured JSON output with raw text fallback
- [x] accumulate_learning() for self-bootstrapping context
- [x] JSON mode support for MiniMax-M3

### 3.5 Compilation Agent ✅
- [x] DPO preference pair construction
- [x] Train/val/test stratified splitting
- [x] Deduplication (MD5 hash)
- [x] Enhanced quality scoring (6 chemistry-aware dimensions)
- [x] JSONL export + HuggingFace datasets format

## Phase IV: Multi-Agent Pipeline ✅

### 4.1 Orchestrator ✅
- [x] Session management with SQLite checkpoints
- [x] Full state machine (IDLE → GENERATING → VERIFYING → REFLECTING → COMPILING → COMPLETED)
- [x] Self-bootstrapping iteration loop (3 iterations)
- [x] Temperature scheduling integration (cosine 1.0→0.3)
- [x] RAG initialization, indexing, and prompt enrichment
- [x] Rate limiting (8s delay between LLM calls)

### 4.2 Self-Bootstrapping Reflection Loop ✅
- [x] LearningContext data model
- [x] Accumulate learning across iterations
- [x] Inject into hypothesis agent system prompt
- [x] build_context_prompt()

### 4.3 Temperature Scheduling ✅
- [x] Cosine annealing scheduler (1.0 → 0.3)
- [x] Linear and exponential schedule variants
- [x] Integrated into bootstrap loop
- [x] Configurable via default.yaml

### 4.4 Multi-Hop Chemical RAG ✅
- [x] TF-IDF vector store + NetworkX knowledge graph
- [x] Multi-hop iterative retrieval with query decomposition
- [x] Typed knowledge graph edges
- [x] Scaffold relation indexing (Murcko)
- [x] Functional group detection (SMARTS)
- [x] Graph traversal (neighbors, paths)
- [x] 140 docs indexed

## Phase V: CLI & Interface ✅
- [x] pipeline: Full pipeline with --bootstrap
- [x] verify: Resume session and rerun verification
- [x] reflect: Resume session and rerun reflection
- [x] compile: Resume session and rerun compilation
- [x] status: Check session state from checkpoint
- [x] ablation: 4-variant pipeline + 7-variant evolution (--mode evolution)
- [x] chemcot: ChemCoTBench comparison
- [x] config-cmd: Show configuration
- [x] Rich-formatted summary tables

## Phase VI: Testing & Benchmarking ✅
- [x] 230/230 tests passing
- [x] Standalone quality scoring module (src/compilation/quality.py)
- [x] Pipeline ablation framework (4 variants)
- [x] Evolution ablation framework (7 variants)
- [x] ChemCoT comparison module
- [x] Dataset v1.0 (110 pairs)
- [x] Dataset v2.0 (181 pairs, HuggingFace updated)

## Phase VII: Paper & Release ✅ (in-progress for arXiv/Zenodo)
- [x] Architecture overview (docs/architecture.md)
- [x] Configuration reference (docs/configuration.md)
- [x] LaTeX paper scaffold (paper/main.tex + references.bib)
- [x] Paper updated with real numbers (181 pairs, 19 types, ablation results)
- [x] Evolution ablation table (7 variants)
- [x] GigaEvo/AlphaEvolve/MAP-Elites/ChemCrow/ChemCoTBench citations
- [x] Dockerfile updated for reproducibility
- [x] GitHub repo: github.com/aayushkrm/auto-cheminstruct
- [x] HuggingFace dataset: aayushkrm/autochem-instruct (181 pairs)
- [x] README defense-ready
- [ ] arXiv preprint
- [ ] Zenodo archive publish

## Evolution Layer Implementation ✅ (Post Phase V)
- [x] GigaEvo-style MAP-Elites (src/evolution/map_elites.py)
- [x] Async DAG engine (src/evolution/dag.py)
- [x] Redis state layer (src/evolution/redis_store.py)
- [x] 5 mutation operators with selection weights
- [x] 4 specialist islands with migration
- [x] Deterministic RNG throughout
- [x] Maestro CARL-style reasoning chains (src/carl/chain.py)
- [x] 4 Pydantic step models (Steric/Electronic/Thermodynamic/Causal)
- [x] Parallel DAG execution for reflection
- [x] CLI integration (--mode evolution)

## GigaEvo + Maestro CARL Framework Alignment ✅
- [x] Full research analysis (27 docs in docs/research/)
- [x] Integration blueprint documented
- [x] Custom implementation following published architecture patterns
- [x] All components tested and validated
- [x] Paper acknowledgment of AIRI ecosystem
