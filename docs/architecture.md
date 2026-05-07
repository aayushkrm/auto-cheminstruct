# Auto-ChemInstruct: Architecture Overview

## System Design

Auto-ChemInstruct is an autonomous multi-agent pipeline that generates physically-validated instruction datasets for chemistry domain-specific language models. The system operates without human annotation — it generates, validates, and compiles DPO/RLHF preference pairs entirely autonomously.

## Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │        PipelineOrchestrator         │
                    │  ┌───────────────────────────────┐  │
                    │  │   Self-Bootstrapping Loop     │  │
                    │  │                               │  │
                    │  │  Generate → Verify → Reflect  │  │
                    │  │     ↑                    │    │  │
                    │  │     └── LearningContext ←─┘    │  │
                    │  └───────────────────────────────┘  │
                    │                                     │
                    │  ┌───────────────────────────────┐  │
                    │  │   Temperature Scheduler       │  │
                    │  │   Cosine annealing 1.0→0.3    │  │
                    │  └───────────────────────────────┘  │
                    │                                     │
                    │  ┌───────────────────────────────┐  │
                    │  │   Multi-Hop Chemical RAG      │  │
                    │  │   Vector store + KG traversal │  │
                    │  └───────────────────────────────┘  │
                    └─────────────────────────────────────┘
```

## Core Components

### 1. Hypothesis Generation Agent
**File**: `src/agents/hypothesis_agent.py`

Generates diverse chemical reaction hypotheses using LLM. Supports:
- **Self-bootstrapping**: Injects accumulated failure knowledge as system prompt constraints
- **RAG enrichment**: Uses multi-hop chemical knowledge graph for context
- **8 prompt templates**: Substitutions, cycloadditions, cross-couplings, rearrangements, etc.
- **Robust parsing**: JSON extraction with fallback for reasoning model quirks

### 2. Verification Agent
**File**: `src/agents/verification_agent.py`

Validates hypotheses against physical constraints:
1. **SMILES parsing** → RDKit molecule construction
2. **Valence checks** → Atom valence + sanitization
3. **Steric analysis** → ETKDGv3 conformer + van der Waals overlap detection
4. **Chemical feasibility** → Unstable groups, hypervalent atoms, ring strain
5. **Descriptor computation** → SA score, QED, LogP, TPSA, molecular weight
6. **Energetic validation** → xTB semi-empirical QM (code-complete, pending binary)

**Validation flow**:
```
SMILES → Parse → Valence → Conformer → Steric → Feasibility → Descriptors → [xTB]
                                                                              ↓
                                                                    Gibbs free energy
                                                                    HOMO/LUMO energies
                                                                    Dipole moment
```

### 3. Reflection Agent
**File**: `src/agents/reflection_agent.py`

Generates causal failure analysis for failed reactions:
- **10 failure categories**: Steric clash, electronic mismatch, thermodynamic impossibility, etc.
- **Structured output**: Root cause, causal chain, fix suggestion
- **Learning accumulator**: Compiles insights across iterations into `LearningContext`

### 4. Compilation Agent
**File**: `src/agents/compilation_agent.py`

Builds DPO preference pairs from verified reactions:
- **Chosen**: Passed verification (valid reaction)
- **Rejected**: Failed verification + causal reflection trace
- **Output**: JSONL (train/val/test) + HuggingFace datasets format
- **Quality scoring**: Yield bonus, confidence weighting, mechanism depth

## Self-Bootstrapping Innovation

The core paper contribution — the system learns from its own failures:

```
Iteration 1 (T=1.0, exploration):
  Generate → Verify → 3/5 failed → Reflect → LearningContext
                                            ↓
Iteration 2 (T=0.65, transition):    "Avoid azides", "Check sterics",
  Generate ← enriched prompt ←      "Prefer amide coupling"
  Verify → 4/5 passed → Reflect...
```

**LearningContext accumulation**:
- `key_lessons`: Causal insights from reflection traces
- `recurring_failures`: Patterns appearing across multiple iterations
- `successful_strategies`: Reaction types with high pass rates
- `failure_category_counts`: Distribution of failure modes

## Temperature Scheduling

Cosine annealing controls the exploration-exploitation trade-off:

```
T(iter) = T_min + (T_max - T_min) × 0.5 × (1 + cos(π × iter / N))
```

- **Early iterations**: High temperature (1.0) → maximum reaction diversity
- **Later iterations**: Low temperature (0.3) → exploit learned constraints

Configurable schedules: cosine, linear, exponential.

## Multi-Hop Chemical RAG

**File**: `src/rag/chemical_rag.py`

Extends standard RAG with chemical knowledge graph traversal:

1. **Vector retrieval** (ChromaDB): Semantic search over indexed reaction templates
2. **Knowledge graph** (NetworkX): Molecules, scaffolds, functional groups, reaction types
3. **Multi-hop traversal**: Retrieve → extract entities → formulate query → repeat
4. **Scaffold indexing**: Murcko scaffold decomposition links molecules by core structure
5. **Functional group detection**: SMARTS-based group identification

**Graph topology**:
```
Molecule ──reactant_of──→ Reaction ──product_of──→ Molecule
    │                                                    │
    ├──has_scaffold──→ Scaffold                           │
    │                    │                                │
    └──contains_group──→ Functional Group ←───────────────┘
```

## Data Flow

```
User CLI → PipelineOrchestrator
               │
               ├→ HypothesisAgent.generate()
               │   ├→ RAG.enrich_prompt() (multi-hop context)
               │   └→ LLM.generate() → parse JSON
               │
               ├→ VerificationAgent.verify()
               │   ├→ RDKit (structural)
               │   └→ xTB (energetic, optional)
               │
               ├→ [ReflectionAgent.reflect()]
               │   └→ accumulate → LearningContext
               │
               ├→ CompilationAgent.compile()
               │   ├→ Build preference pairs
               │   ├→ Stratified split (80/10/10)
               │   └→ Save JSONL + HF datasets
               │
               └→ Index into RAG (for next iterations)
```

## Configuration

Default configuration: `configs/default.yaml`
- LLM: Fireworks AI deepseek-v4-pro (OpenAI-compatible)
- Pipeline: batch_size=1, checkpoint_dir=.checkpoints
- Temperature: schedule=cosine, max=1.0, min=0.3
- RAG: ChromaDB persistence, multi-hop enabled

## Output Artifacts

| Artifact | Format | Location |
|----------|--------|----------|
| Pipeline log | Plain text | `logs/autochem.log` |
| Session state | SQLite | `.checkpoints/<session-id>.db` |
| Preference pairs | JSONL | `datasets/autochem-<id>/train.jsonl` |
| HF dataset | Arrow | `datasets/autochem-<id>/hf_dataset/` |
| Ablation results | JSON | `benchmarks/ablation_report.json` |
| Dataset metadata | JSON | `datasets/autochem-<id>/metadata.json` |

## Key Design Decisions

1. **Single-hypothesis prompts** (not batches): Reasoning model (deepseek-v4-pro) works best with focused tasks
2. **Chemistry-first validation**: Physical constraints take priority over LLM confidence
3. **Graceful degradation**: xTB, RAG, and embeddings all fail gracefully
4. **Checkpoint resilience**: SQLite state allows session resumption after interruption
5. **Modular agents**: Each agent is independently testable with mock LLMs
