# Complete Integration Blueprint: Maestro + GigaEvo for Auto-ChemInstruct

## Research Phase Complete — All Documentation Scraped

**Date:** 2026-05-11
**Tools Used:** Crawl4AI (JS-rendered page scraping), ClawHub (skill registry), Firecrawl (web search)
**Documents Scraped:** 15+ pages across all three frameworks
**Status:** Research complete — all APIs, architecture, and code examples obtained

---

## 1. Framework Verdicts (Updated with Deep Research)

### GigaChain — STILL DEPRECATED

After scraping the `langchain-gigachat` repo and examples, confirmed:
- `gigachain` PyPI package says: "deprecated, upgrade to langchain + langchain_gigachat"
- `langchain-gigachat` is a lightweight provider plugin (~200 lines)
- It just adds `ChatGigaChat` and `GigaChatEmbeddings` to LangChain
- All other functionality is identical to vanilla LangChain

**Decision: Permanent skip. Not worth a single line of code.**

### Maestro — Agent Orchestration Platform

After scraping the full English docs and instruction page:

**Architecture:**
```
Gateway (FastAPI) → Chat Manager → Tracks → Components (gRPC services)
                        ↓
                   LLM Hub (unified LLM access)
                        ↓
                   Moderation (FLAME)
                        ↓
                   Reasoning Chains (CARL)
```

**Key Libraries:**

| Library | Purpose | Directly Useful? |
|---------|---------|-----------------|
| `mmar-mapi` | Session management, file storage, chat models, service APIs | Maybe |
| `mmar-ptag` | Type-safe gRPC (`ptag_client`, `deploy_server`) | No (Docker-only) |
| `mmar-flame` | Content moderation (n-gram + rules, 98.7% precision) | Yes — could validate reaction outputs |
| `mmar-carl` | CARL reasoning chains (Event-Action-Result triads) | **YES** — replaces our Reflection Agent |
| `mmar-llm` | LLM accessor (GigaChat + OpenAI + OpenRouter) | No (we use LangChain) |

**Agent Creation Pattern (from instruction page):**
```python
# 1. Define a component (gRPC service)
class MetadataExtractor(ContentInterpreterAPI):
    def interpret(self, resource_id, trace_id):
        # Extract metadata from document
        ...

# 2. Register as gRPC service
deploy_server(config=config, service=MetadataExtractor())

# 3. Use in a track (agent pipeline)
class MetadataExtraction(SimpleTrack):
    def __init__(self, config):
        self.document_extractor = ptag_client(DocumentExtractorAPI, ...)
        self.metadata_extractor = ptag_client(ContentInterpreterAPI, ...)
    
    def generate_response(self, query):
        # Chain of agent calls
        text_file = self.document_extractor.extract(pdf_file)
        metadata = self.metadata_extractor.interpret(text_file)
        answer = self.metadata_extractor.interpret(question, metadata)
        return answer
```

**CARL Reasoning Chains (the killer feature):**
```python
from mmar_carl import ReasoningChain, StepDescription, ReasoningContext

CLINICAL_REASONING = [
    StepDescription(
        number=1,
        title="Analyze reaction validity",
        aim="Check if reaction passes RDKit validation",
        reasoning_questions="What structural features cause failure?",
        step_context_queries=["SMILES validity", "atom conservation", "valence"],
        stage_action="Identify structural issues",
        dependencies=[]
    ),
    StepDescription(
        number=2,
        title="Compute energetics",
        aim="Estimate thermodynamic feasibility",
        reasoning_questions="Is the reaction energetically favorable?",
        dependencies=[1],
        step_context_queries=["MMFF94 energy", "strain", "barrier"],
        stage_action="Quantify energetic profile"
    ),
    StepDescription(
        number=3,
        title="Generate causal reflection",
        aim="Explain WHY the reaction failed in chemical terms",
        reasoning_questions="What is the root cause of failure?",
        dependencies=[1, 2],
        step_context_queries=["steric hindrance", "electronic effects", "stability"],
        stage_action="Synthesize failure explanation"
    ),
]

chain = ReasoningChain(steps=CLINICAL_REASONING)
result = chain.execute(context)
failure_analysis = result.get_final_output()
```

**Decision: CARL is the value-add. The full Maestro Docker stack is overkill. We should extract CARL's pattern (Event-Action-Result chains) and implement it directly without Docker/gRPC.**

### GigaEvo — Evolutionary Optimization (The Real Deal)

After scraping all 6 docs + paper + problem examples + config system:

**Exact API for Custom Problems:**

```python
# problems/autochem/validate.py
def validate(hypothesis: dict) -> dict:
    """
    Args: hypothesis dict with 'reactants', 'products', 'conditions'
    Returns: dict of metrics (must include 'fitness')
    """
    from rdkit import Chem, AllChem
    from src.chemistry.rdkit_wrapper import compute_molecule_energy
    from src.compilation.quality import score_hypothesis
    
    metrics = {}
    
    # 1. SMILES validity
    products = [Chem.MolFromSmiles(s) for s in hypothesis['products']]
    metrics['smiles_valid'] = float(all(m is not None for m in products))
    
    if not metrics['smiles_valid']:
        metrics['fitness'] = 0.0
        return metrics
    
    # 2. MMFF94 energy
    energies = [compute_molecule_energy(m) for m in products]
    metrics['energy'] = sum(e for e in energies if e)
    
    # 3. Quality score (6-dim rubric)
    metrics['quality'] = score_hypothesis(hypothesis)
    
    # 4. Fitness = weighted composite
    metrics['fitness'] = (
        0.4 * metrics['quality'] +
        0.3 * (1.0 / (1.0 + abs(metrics['energy']))) +
        0.3 * metrics['smiles_valid']
    )
    
    return metrics

# problems/autochem/metrics.yaml
specs:
  fitness:
    is_primary: true
    higher_is_better: true
    description: "Composite reaction quality"
    lower_bound: 0.0
    upper_bound: 1.0
    significant_change: 0.01
  quality:
    higher_is_better: true
    description: "6-dim quality rubric score"
  energy:
    higher_is_better: false
    description: "MMFF94 energy (kcal/mol)"
  smiles_valid:
    higher_is_better: true
    description: "SMILES structural validity"

# problems/autochem/task_description.txt
**OBJECTIVE**: Generate a novel, physically valid chemical reaction pathway.
[Full prompt template from our existing system...]
```

**MAP-Elites Behavioral Space for Chemistry:**
```python
BehaviorSpace(
    feature_bounds={
        "reaction_type": (0, 13),      # 13 types → enumerated bins
        "molecular_weight": (0, 500),   # MW range
        "fitness": (0.0, 1.0),          # Quality score
    },
    resolution={
        "reaction_type": 13,   # One bin per type
        "molecular_weight": 10,  # 50 Da bins
        "fitness": 20,           # 0.05 bins
    }
)
# → 13 × 10 × 20 = 2600 cells → MAP-Elites fills them ALL
```

**DAG Stages for Auto-ChemInstruct:**
```python
# Stage 1: SMILES Validation (cheap, cacheable)
class SmilesValidationStage(Stage):
    InputsModel = SmilesInputs  # requires: hypothesis
    OutputModel = SmilesOutput  # produces: valid_smiles, molecules
    cacheable = True
    
    async def compute(self, program):
        hypothesis = program.get_stage_output("generate")
        mols = [Chem.MolFromSmiles(s) for s in hypothesis.products]
        return SmilesOutput(valid=all(m is not None for m in mols), mols=mols)

# Stage 2: MMFF94 Energy (expensive, cacheable)  
class EnergyStage(Stage):
    InputsModel = EnergyInputs  # requires: mols from Stage 1
    OutputModel = EnergyOutput  # produces: energy, strain
    cacheable = True
    
    async def compute(self, program):
        mols = program.get_stage_output("validate_smiles").mols
        energies = [compute_energy(m) for m in mols]
        return EnergyOutput(total=sum(energies), per_molecule=energies)

# Stage 3: Quality Scoring (post-validation, cacheable)
class QualityStage(Stage):
    InputsModel = QualityInputs  # requires: hypothesis + energy + mols
    OutputModel = QualityOutput  # produces: quality_score, diversity
    cacheable = True

# Stage 4: Reflection/Insight (LLM, non-cacheable — depends on context)
class ReflectionStage(Stage):
    InputsModel = ReflectionInputs  # requires: hypothesis + quality + energy
    OutputModel = ReflectionOutput  # produces: insight_text, failure_cause
    cacheable = False  # Must re-execute — depends on evolutionary context

# Stage 5: Lineage Update (always runs, non-cacheable)
class LineageStage(Stage):
    InputsModel = LineageInputs  # requires: program_id + parent_id
    OutputModel = LineageOutput  # produces: lineage_graph
    cacheable = False
```

**DAG Pipeline Configuration:**
```yaml
# config/pipeline/autochem.yaml
pipeline:
  stages:
    - name: validate_smiles
      type: SmilesValidationStage
      timeout: 10s
      dependencies: []
    - name: compute_energy
      type: EnergyStage
      timeout: 30s
      dependencies:
        - source: validate_smiles
          input: mols
      precondition: "validate_smiles.valid == true"  # Skip if invalid
    - name: score_quality
      type: QualityStage
      timeout: 20s
      dependencies:
        - source: validate_smiles
          input: hypothesis
        - source: compute_energy
          input: energy
    - name: generate_insight
      type: ReflectionStage
      timeout: 60s
      dependencies:
        - source: score_quality
          input: metrics
```

**Evolution Run Command:**
```bash
python run.py problem.name=autochem \
    max_generations=20 \
    island_max_size=100 \
    model_name=openai/deepseek-v3p2 \
    redis.db=1
```

---

## 2. Concrete Integration Architecture

### What We Keep From Our Current Pipeline
- RDKit wrapper (`chemistry/rdkit_wrapper.py`)
- 6-dim quality scoring (`compilation/quality.py`)
- Data models (`data/models.py`) — Pydantic v2
- LLM factory (`utils/llm_factory.py`) — Fireworks API
- DPO compilation (`agents/compilation_agent.py`)
- Config system (`configs/default.yaml`)

### What Gets Replaced
| Old Component | New Component | Framework |
|---|---|---|
| `PipelineOrchestrator._generate_hypotheses()` | GigaEvo Mutation Operator | GigaEvo |
| `PipelineOrchestrator._verify_hypotheses()` | GigaEvo DAG Stages 1-2 | GigaEvo |
| `PipelineOrchestrator._reflect_on_failures()` | GigaEvo Insight Generation | GigaEvo |
| `PipelineOrchestrator._accumulate_learning()` | GigaEvo MAP-Elites Archive | GigaEvo |
| Simple temperature annealing | MAP-Elites selection + genealogy | GigaEvo |

### What Gets Added
| New Capability | Source |
|---|---|
| CARL-style causal reasoning chains | Maestro `mmar-carl` (reimplemented) |
| FLAME-style output moderation | Maestro `mmar-flame` (reimplemented) |
| Population-based parallel evolution | GigaEvo |
| Quality-diversity optimization | GigaEvo MAP-Elites |
| Lineage tracking | GigaEvo Redis |

---

## 3. Phased Implementation Plan

### Phase 0: Setup (Day 1)
```bash
brew install redis
cd auto-cheminstruct
# GigaEvo lives alongside our code
git clone https://github.com/AIRI-Institute/gigaevo-core vendor/gigaevo
cd vendor/gigaevo && pip install -e .
# Test
python run.py problem.name=toy_example max_generations=3
```

### Phase 1: Auto-ChemInstruct as GigaEvo Problem (Days 2-5)
1. Create `problems/autochem/` directory
2. Write `validate.py` — ports our RDKit + MMFF94 + quality scoring
3. Write `task_description.txt` — ports our hypothesis prompt
4. Write `metrics.yaml` — defines fitness, quality, energy, validity
5. Create `initial_programs/` — 10 seed hypotheses as Python functions
6. Write `config/pipeline/autochem.yaml` — DAG stage definitions
7. Run small evolution: `python run.py problem.name=autochem max_generations=5`

### Phase 2: MAP-Elites Behavioral Space (Days 6-8)
8. Define behavioral space: reaction_type × molecular_weight × fitness
9. Configure multi-island evolution (fitness island + diversity island)
10. Run medium evolution: `max_generations=10, island_max_size=50`

### Phase 3: CARL Reflection Chains (Days 9-11)
11. Implement CARL Event-Action-Result pattern (not the library — the pattern)
12. Map to our chemistry domain:
    - Event: "Reaction Z failed RDKit validation"
    - Action: "Analyze SMILES, compute energy, check sterics"
    - Result: "Nucleophilic attack blocked by tert-butyl at C4"
13. Plug into GigaEvo's insight generation stage

### Phase 4: Ablation Study (Days 12-14)
14. Run 4 variants:
    - Full GigaEvo (MAP-Elites + CARL + insight)
    - No MAP-Elites (single-objective only)
    - No CARL (no structured reflection)
    - Baseline (our original pipeline, same N)
15. Measure: diversity (types filled), quality (avg score), efficiency (pairs/LLM-call)

### Phase 5: Paper & Dataset (Days 15-17)
16. Export GigaEvo archive → DPO pairs
17. Upload to HuggingFace
18. Rewrite paper framing: "GigaEvo-Powered Evolutionary Hypothesis Generation"
19. Ablation results as main empirical contribution

---

## 4. ClawHub Resources

Found these skills via `clawhub search`:
- `maestro-skill` — Maestro API integration
- `maestro-sdk` — Maestro SDK wrapper

Can be used to bootstrap Maestro integration faster if needed.

---

## 5. What We DON'T Need

| Framework/Component | Why Skip |
|---|---|
| GigaChain (gigachain) | Deprecated — replaced by langchain-gigachat |
| langchain-gigachat | Just a provider plugin; we use Fireworks, not GigaChat |
| Maestro Docker stack | gRPC + PostgreSQL overkill for a research pipeline |
| Maestro mmar-ptag | Type-safe gRPC — designed for microservices, not needed |
| Maestro mmar-mapi | Session management — we use LangChain chat models |
| Maestro AuthService | JWT + OAuth — we don't have users |
| Scrapling | API changed since v0.3; Crawl4AI works perfectly |

## 6. What We EXTRACT (Not Install)

| Pattern | Source | Implementation |
|---|---|---|
| CARL Event-Action-Result chains | `mmar-carl` | Reimplement as Python classes with Pydantic |
| FLAME n-gram moderation | `mmar-flame` | Lightweight regex + keyword filter |
| GigaEvo problem structure | GigaEvo core | Adapt our code, embed in their runner |
| GigaEvo MAP-Elites | GigaEvo core | Use directly (their implementation) |
| GigaEvo DAG stages | GigaEvo core | Use directly (their implementation) |

---

## 7. Total Feasible Scope

| Deliverable | Effort | Risk |
|---|---|---|
| GigaEvo problem definition | 3 days | Low |
| Custom DAG stages (RDKit + MMFF94) | 2 days | Low |
| CARL reflection chains | 2 days | Low |
| MAP-Elites behavioral space | 1 day | Low |
| Evolution runs (small + medium) | 1 day (compute) | Medium (LLM costs ~$20-40) |
| Ablation study | 1 day (compute) | Low |
| Paper rewrite | 2 days | Low |
| **Total** | **~12 days** | |

**LLM Cost Estimate:** ~$25-50 for all evolution runs (Fireworks DeepSeek-v3p2).
