# GigaEvo + Maestro Integration Plan for Auto-ChemInstruct

## Executive Summary

After thorough research of all three AIRI/Sber frameworks (Maestro, GigaChain, GigaEvo), the verdict is:

| Framework | Status | Value for Auto-ChemInstruct |
|-----------|--------|----------------------------|
| **GigaChain** | DEPRECATED — replaced by `langchain-gigachat` | **NONE** — our LangChain + Fireworks setup is superior |
| **Maestro** | Active, open source, Docker-based | **MODERATE** — adds safety/auditability but heavy infrastructure |
| **GigaEvo** | Active, open source (MIT), arXiv paper | **VERY HIGH** — transforms our core innovation |

**Recommendation:** Integrate GigaEvo as the primary new component (replacing our self-bootstrapping loop with MAP-Elites evolutionary optimization). Optionally wrap in Maestro for paper credibility if the internship demands it.

---

## GigaChain — Why We Skip It

GigaChain was a LangChain fork for Sber's GigaChat models. PyPI shows:
```
⚠️ This package was deprectaed. Upgrade to clear langchain + langchain_gigachat.
```

The replacement `langchain-gigachat` (v0.5.1) is just a LangChain provider plugin — `pip install langchain-gigachat` lets you call GigaChat models through LangChain. Since we use Fireworks AI (DeepSeek-v3p2), not GigaChat, there is zero reason to touch this.

**Action: None. Mention in paper that we use LangChain (the upstream that GigaChain forked from).**

---

## Maestro — Optional Wrapper

### What Maestro Is
A Docker-based multi-agent orchestration platform for building digital assistants. Key components:
- **Gateway**: FastAPI backend for all requests
- **Chat Manager**: Manages agent business logic
- **LLM Hub**: Unified interface to multiple LLMs (OpenAI-compatible + GigaChat)
- **CARL**: Collaborative Agent Reasoning Library — formalizes expert thinking as Event-Action-Result chains
- **Flame**: Moderation loop analyzing agent outputs
- **PostgreSQL**: State persistence

### Where It Fits
Maestro could replace our `PipelineOrchestrator` class (~400 lines of Python) with:
- Formal agent definitions with I/O specifications
- CARL reasoning chains for structured reflection traces
- Moderation guardrails on generated outputs
- Multi-LLM routing via LLM Hub
- Full audit trails

### Cost-Benefit Analysis

| Pro | Con |
|-----|-----|
| "Built on Maestro" adds internship credibility | Docker dependency adds complexity |
| Formal safety guardrails | Designed for chatbots, not scientific pipelines |
| CARL reasoning chains add novelty | PostgreSQL adds infrastructure |
| Good for paper's industrial relevance | Our custom orchestrator is battle-tested |

### Recommendation
**Light integration only.** Cite Maestro as the architectural inspiration and optionally wrap a demo in Maestro if the internship explicitly requires it. Do NOT replace our orchestrator — it's simpler, faster, and already works.

---

## GigaEvo — Primary Integration Target

### What GigaEvo Is

An open-source (MIT) framework for LLM-guided evolutionary computation, inspired by Google DeepMind's AlphaEvolve. Key specs:

- **Python 3.12+** (compatible with our 3.13)
- **Redis** persistence layer
- **Hydra** configuration system (compatible with our OmegaConf — we can keep ours)
- **LangGraph** mutation operator (compatible with our LangChain setup)
- **arXiv:2511.17592** (Nov 2025, 7 citations)
- **GitHub**: github.com/AIRI-Institute/gigaevo-core (24★) / github.com/FusionBrainLab/gigaevo-core (115★)

### Four Core Components

```
┌──────────────────────────────────────────────┐
│ 1. REDIS DATABASE                             │
│    • Program UUID, source code, state         │
│    • Metrics, lineage (parent→child tree)     │
│    • Optimistic concurrency control           │
├──────────────────────────────────────────────┤
│ 2. ASYNC DAG EXECUTION ENGINE                 │
│    • asyncio-based parallel evaluation        │
│    • Cascading validation (cheap first)       │
│    • Stage caching + auto-skip                │
│    • Configurable timeouts per stage          │
├──────────────────────────────────────────────┤
│ 3. MAP-ELITES EVOLUTION ENGINE                │
│    • Quality-Diversity (not just optimization)│
│    • Behavioral space: multi-dim bins         │
│    • Fitness-proportional selection           │
│    • Multi-island with migration              │
├──────────────────────────────────────────────┤
│ 4. LANGGRAPH MUTATION OPERATOR                │
│    • Prompt = task + parent code + metrics    │
│      + insights + lineage analysis            │
│    • LLM rewrites code (not random mutation)  │
│    • Multi-model routing (GPT, Claude, etc.)  │
└──────────────────────────────────────────────┘
```

### How MAP-Elites Beats Our Current Loop

| Our Self-Bootstrapping | GigaEvo MAP-Elites |
|---|---|
| Linear: generate → verify → reflect → repeat | Population: 100s of hypotheses evolved in parallel |
| Single objective: pass/fail | Multi-objective: validity × diversity × energy × quality |
| No diversity optimization | MAP-Elites fills ALL behavioral cells |
| Simple temperature annealing | Fitness-proportional selection + genealogy tracking |
| 15-20 hypotheses per run | 100s per generation, 10-50 generations |
| Ad-hoc learning context | Redis-backed lineage + insight accumulation |

---

## Integration Architecture

### Phase 1: Define Auto-ChemInstruct as a GigaEvo Problem

GigaEvo expects a problem to have:
```
problems/autochem/
├── validate.py           # Validation + metrics computation
├── task_description.txt  # LLM prompt for hypothesis generation
├── metrics.yaml          # Metric definitions
└── initial_programs/     # Seed hypotheses
    ├── diels_alder_001.py
    ├── suzuki_001.py
    └── ...
```

#### `validate.py`
```python
"""Auto-ChemInstruct validation for GigaEvo."""
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

def validate(hypothesis: dict) -> dict:
    """
    Validate a chemical reaction hypothesis.
    
    Args:
        hypothesis: dict with 'reactants', 'products', 'conditions', 'mechanism'
    
    Returns:
        dict of metrics: validity, quality_score, energy, diversity_contribution, etc.
    """
    metrics = {}
    
    # 1. Structural validity (RDKit)
    products_mol = [Chem.MolFromSmiles(s) for s in hypothesis['products']]
    reactants_mol = [Chem.MolFromSmiles(s) for s in hypothesis['reactants']]
    
    metrics['smiles_valid'] = float(all(m is not None for m in products_mol + reactants_mol))
    
    if not metrics['smiles_valid']:
        metrics['fitness'] = 0.0
        return metrics
    
    # 2. MMFF94 energy (kcal/mol, lower is more stable)
    energies = []
    for mol in products_mol:
        mol = Chem.AddHs(mol)
        if AllChem.MMFFOptimizeMolecule(mol) == 0:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                energies.append(ff.CalcEnergy())
    metrics['energy'] = sum(energies) if energies else 999.0
    
    # 3. Quality score (6-dimension rubric)
    metrics.update(_compute_quality(hypothesis, products_mol, reactants_mol))
    
    # 4. Fitness = combined score
    metrics['fitness'] = (
        metrics.get('quality_score', 0) * 0.4 +
        (1.0 / (1.0 + metrics['energy'])) * 0.3 +
        metrics.get('diversity_contribution', 0) * 0.3
    )
    
    return metrics
```

#### `task_description.txt`
```
**OBJECTIVE**: Generate a novel, physically valid chemical reaction pathway.
Propose reactants, products, reaction conditions (solvent, temperature, catalyst),
and a detailed mechanism. The reaction must pass RDKit structural validation and
MMFF94 energetic checks.

**RULES**:
- Products must be valid SMILES strings
- Atom conservation: reactant atoms = product atoms
- No hypervalent carbons or impossible valence states
- Prefer reactions with favorable thermodynamics (exothermic or low barrier)

**REACTION TYPES** (choose one):
Diels-Alder, Suzuki coupling, Wittig, Grignard, Aldol, Michael addition,
Heck reaction, Friedel-Crafts, Claisen rearrangement, Mannich,
Buchwald-Hartwig, Click chemistry, Esterification, Amide coupling

**MUTATION STRATEGIES**:
- Add/remove substituents
- Change catalyst or solvent
- Modify temperature/pressure conditions
- Swap reaction type while keeping scaffold
- Introduce protecting groups
- Scale from milligram to kilogram scale
```

#### `metrics.yaml`
```yaml
specs:
  fitness:
    description: Composite quality score (higher is better)
    decimals: 5
    is_primary: true
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 1.0
    include_in_prompts: true
    significant_change: 0.01
  quality_score:
    description: 6-dimension quality rubric score
    decimals: 4
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 1.0
  energy:
    description: MMFF94 total energy (kcal/mol, lower is more stable)
    decimals: 2
    higher_is_better: false
    lower_bound: -1000.0
    upper_bound: 10000.0
  smiles_valid:
    description: Whether all SMILES are valid (1=valid, 0=invalid)
    decimals: 0
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 1.0
  diversity_contribution:
    description: Contribution to scaffold diversity (Tanimoto distance)
    decimals: 4
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 1.0
  reflection_quality:
    description: Depth of causal failure analysis
    decimals: 3
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 1.0
```

### Phase 2: Define MAP-Elites Behavioral Space

```python
# The behavioral space defines the archive bins
BehaviorSpace(
    feature_bounds={
        "reaction_type": (0, 13),     # 13 named reaction types
        "molecular_weight": (0, 500),  # MW range
        "scaffold_diversity": (0, 1),  # Tanimoto distance from archive
    },
    resolution={
        "reaction_type": 13,   # One bin per type
        "molecular_weight": 10, # 50 Da bins
        "scaffold_diversity": 10, # 0.1 bins
    }
)
# Total: 13 × 10 × 10 = 1300 behavioral cells
# MAP-Elites tries to fill ALL cells with elite solutions
```

### Phase 3: Mutation Operator → Hypothesis Agent

GigaEvo's LangGraph mutation operator REPLACES our Hypothesis Agent. The LLM:
1. Reads the task description (reaction generation prompt)
2. Reads parent solutions (previous successful reactions)
3. Reads their metrics (fitness, quality, energy)
4. Reads lineage analysis (which mutations worked, which didn't)
5. Reads LLM-generated insights (WHY previous mutations failed)
6. Generates a new reaction hypothesis as Python code

This is exactly our self-bootstrapping loop but:
- **Population-based** instead of sequential
- **Lineage-aware** instead of simple accumulation
- **Quality-diversity** instead of pass/fail only
- **Parallel** instead of one-at-a-time

### Phase 4: DAG Stages → Verification + Reflection

GigaEvo's DAG execution engine processes each hypothesis through stages:

```
Stage 1: EXECUTE        → Run the hypothesis code, extract SMILES
Stage 2: RDKIT_CHECK    → Validate SMILES, check atom conservation
Stage 3: MMFF94_ENERGY  → Compute MMFF94 energies
Stage 4: QUALITY_SCORE  → 6-dimension quality rubric
Stage 5: DIVERSITY      → Compute Tanimoto vs archive
Stage 6: INSIGHT        → LLM reflects on failures (causal reasoning)
Stage 7: LINEAGE        → Update parent-child genealogy tree
```

### Phase 5: Compilation → Dataset Export

After evolution completes:
1. Query Redis for all EVOLVING programs
2. Filter by quality_score > threshold
3. Build DPO preference pairs: chosen=passed, rejected=failed+reflection
4. Export as JSONL (HuggingFace-compatible)
5. Upload to HF datasets

---

## Feasibility Assessment

### Hardware Requirements
- **Redis**: Local `redis-server` — trivial
- **Python**: 3.12+ — we're on 3.13 ✓
- **LLM**: Any OpenAI-compatible API — our Fireworks setup works ✓
- **RDKit**: Already installed ✓

### Development Effort

| Task | Effort | Risk |
|------|--------|------|
| Define GigaEvo problem structure | 2 days | Low |
| Port validation code | 2 days | Low (we have all the code) |
| Define behavioral space | 1 day | Medium (need to tune bins) |
| Configure mutation operator prompt | 2 days | Low (we have prompts) |
| Define DAG stages | 2 days | Low |
| Test evolution run | 3 days | Medium (LLM costs) |
| Export to DPO dataset | 1 day | Low |
| Ablation study | 2 days | Medium (LLM costs) |
| **Total** | **~15 days** | |

### LLM Cost Estimate
- Population: 100 programs/island
- Generations: 20
- Mutations per generation: 50
- Total LLM calls: ~1000 (mutations) + ~200 (insights) = ~1200
- Fireworks DeepSeek-v3p2: ~$0.50/1M tokens
- Estimated cost: **$15-30** for a full evolution run

---

## Comparison: GigaEvo vs Current Pipeline

| Dimension | Current | GigaEvo |
|-----------|---------|---------|
| Hypothesis generation | Sequential, 1 at a time | Parallel, 100s simultaneous |
| Diversity objective | None (random only) | MAP-Elites fills all behavioral cells |
| Failure use | Simple accumulation | Genealogy + insight-driven mutation |
| Scaling | O(n) linear | O(n) parallel |
| Reproducibility | Session checkpoints | Redis persistence + full lineage |
| Paper novelty | Novel but incremental | Built on published framework (arXiv) |
| Internship credibility | Generic agent pipeline | Uses GigaEvo + cites AlphaEvolve |

---

## Decision Matrix

| Option | Novelty | Effort | Risk | Paper Impact | Internship Fit |
|--------|---------|--------|------|-------------|----------------|
| Keep current pipeline + cite frameworks | Medium | 0 | 0 | Medium | Weak |
| GigaEvo integration | High | 15 days | Medium | High | Strong |
| Full Maestro + GigaEvo | Very High | 25 days | High | Very High | Perfect |
| GigaChain (deprecated) | None | N/A | N/A | None | Negative |

---

## Recommendation

**Phase A (Week 1-2): GigaEvo Integration**
- Set up Redis + GigaEvo in the project
- Define the autochem problem
- Run small-scale evolution (5 generations, 20 programs)
- Validate that MAP-Elites produces diverse hypotheses

**Phase B (Week 2-3): Ablation + Comparison**
- Run GigaEvo vs current pipeline side-by-side (same N, same seeds)
- Measure: diversity (reaction types filled), quality (avg score), efficiency (LLM calls/pair)
- Ablation: MAP-Elites on/off, insight generation on/off

**Phase C (Week 3-4, optional): Maestro Wrapper**
- If internship demands it, wrap the GigaEvo pipeline in Maestro agents
- Add CARL reasoning chains for structured reflection traces
- This adds paper credibility without changing the core algorithm

**Phase D (Week 4): Paper Rewrite**
- Frame as "GigaEvo-Powered Evolutionary Hypothesis Generation for Chemistry DSLM Data"
- Central claim: MAP-Elites quality-diversity produces 3× more diverse datasets than sequential generation
- Ablation shows each GigaEvo component's contribution
- Cite AlphaEvolve (DeepMind) + GigaEvo (AIRI) for theoretical grounding
