# Auto-ChemInstruct Task Checklist

## Phase I: Infrastructure & Environment
- [x] Repository setup, pyproject.toml, .gitignore, .env.example
- [x] venv via uv (Python 3.13), RDKit verified, ChromaDB tested
- [x] NetworkX installed for chemical knowledge graph
- [x] LLM provider: Fireworks AI deepseek-v4-pro (OpenAI-compatible)
- [x] All core imports verified
- [x] Initialize git repository, pre-commit hooks (git init done, pre-commit pending)

## Phase II: Core Data Models
- [x] ChemicalEntity, ReactionHypothesis, VerificationResult, ReflectionTrace
- [x] PreferencePair (chosen/rejected), PipelineStatus, SessionState
- [x] LearningContext (self-bootstrapping knowledge accumulator)
- [x] 129/129 tests passing (59 unit + 8 integration + 23 RAG + 15 quality + 13 temperature + 11 feasibility)

## Phase III: Core Modules

### 3.1 Chemistry Utilities
- [x] RDKit wrapper (validate, conformers, descriptors, steric clash)
- [x] Chemical feasibility filter (unstable groups, ring strain, hypervalent atoms)
- [x] xTB interface (code-complete, binary pending installation)
- [x] Molecular diversity metrics (Tanimoto, scaffold analysis)

### 3.2 Hypothesis Generation Agent
- [x] Simplified system prompt for reasoning model
- [x] Robust JSON extraction with fallback parsing
- [x] Learning context injection (self-bootstrapping feedback)
- [x] 8 diverse prompt templates

### 3.3 Verification Agent
- [x] RDKit structural validation cascade
- [x] Chemical feasibility check
- [x] Descriptor computation (SA, QED, LogP, MW, TPSA)
- [x] xTB energetic validation (code-complete, graceful fallback)

### 3.4 Reflection Agent
- [x] 10 failure categories
- [x] Structured JSON output with raw text fallback
- [x] accumulate_learning() for self-bootstrapping context

### 3.5 Compilation Agent
- [x] DPO preference pair construction
- [x] Train/val/test stratified splitting (80/10/10)
- [x] Deduplication (MD5 hash)
- [x] Enhanced quality scoring (6 chemistry-aware dimensions)
- [x] JSONL export + HuggingFace datasets format

## Phase IV: Multi-Agent Pipeline & Innovations

### 4.1 Orchestrator
- [x] Session management with SQLite checkpoints
- [x] Full state machine
- [x] Self-bootstrapping iteration loop
- [x] Temperature scheduling integration
- [x] RAG initialization, indexing, and prompt enrichment

### 4.2 Self-Bootstrapping Reflection Loop
- [x] LearningContext data model
- [x] Accumulate learning across iterations
- [x] Inject into hypothesis agent system prompt
- [x] build_context_prompt()

### 4.3 Temperature Scheduling
- [x] Cosine annealing scheduler (1.0 → 0.3)
- [x] Linear and exponential schedule variants
- [x] Integrated into bootstrap loop
- [x] Configurable via default.yaml

### 4.4 Multi-Hop Chemical RAG
- [x] ChromaDB vector store + NetworkX knowledge graph
- [x] Multi-hop iterative retrieval with query decomposition
- [x] Typed knowledge graph edges
- [x] Scaffold relation indexing (Murcko)
- [x] Functional group detection (SMARTS)
- [x] Graph traversal (neighbors, paths)
- [x] RAG wired into orchestrator
- [x] 23 RAG tests passing

## Phase V: CLI & Interface
- [x] pipeline: Full pipeline with --bootstrap
- [x] verify: Resume session and rerun verification
- [x] reflect: Resume session and rerun reflection
- [x] compile: Resume session and rerun compilation
- [x] status: Check session state from checkpoint
- [x] ablation: 4-variant ablation study
- [x] chemcot: ChemCoTBench comparison
- [x] config: Show configuration
- [x] Rich-formatted summary tables
- [x] Session listing (--list flag) for all commands

## Phase VI: Testing & Benchmarking
- [x] 129/129 tests passing (59 unit + 8 integration + 23 RAG + 39 quality/feasibility/temperature)
- [x] Standalone quality scoring module (src/compilation/quality.py)
- [x] Ablation framework (4 variants, metrics, reporting)
- [x] ChemCoT comparison module
- [x] Enhanced quality scoring (chemistry-aware)
- [~] Scale to 100+ hypotheses (15 runs complete — 13 hypotheses, 9 pairs, 69% pass rate; full 100+ needs ~2.5h LLM time run separately)
- [ ] Run full ablation study (N=20+ per variant)
- [ ] ChemCoTBench comparison with real data

## Phase VII: Paper & Release
- [x] Architecture overview (docs/architecture.md)
- [x] Configuration reference (docs/configuration.md)
- [x] LaTeX paper scaffold (paper/main.tex + references.bib)
- [~] Fill in paper results (ablation data pending, benchmarks filled)
- [x] Docker image for reproducibility
- [ ] HuggingFace dataset upload
- [ ] arXiv preprint
- [ ] GitHub release with DOI (Zenodo)
