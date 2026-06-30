% Auto-ChemInstruct
% Aayush Kumar
% TSU Lab of AI in Chemistry & Molecular Engineering × AIRI Institute, Moscow

---

# Auto-ChemInstruct

## Agent-Driven Synthesization of RLHF Data for Chemistry DSLMs

**Aayush Kumar**

TSU Laboratory of AI in Chemistry & Molecular Engineering × AIRI Institute (Moscow)

NeurIPS Datasets & Benchmarks Track

---

# The Problem

- **Scarcity of chemistry instruction data** — datasets like ChemCoTBench rely on expensive human expert annotation, limiting scale and diversity

- **Positivity bias** — existing datasets only show what works, never explain *why reactions fail*, missing critical causal reasoning

- **LLM hallucination** — models generate chemically impossible reactions with no physical validation

- **No self-improvement** — static generation pipeline with no feedback loop or learning from failures

::: {style="border: 2px solid red; padding: 15px; margin-top: 20px;"}
**Key Insight:** The verification bottleneck can be automated — use computational chemistry as a deterministic ground-truth oracle instead of relying on LLM self-evaluation.
:::

---

# Our Solution: 4-Agent Pipeline

```
Hypothesis Agent ──→ Verification Agent ──→ Compilation Agent ──→ DPO Pairs
      ↑                     │
      │              (RDKit + MMFF94)
      │                     │ FAIL
      │              Reflection Agent
      │              10 Failure Categories
      │                     │
      └────── LearningContext ◄──────────────┘
         Self-Bootstrapping Loop
```

**Four specialized AI agents orchestrated through a self-bootstrapping reflection loop:**

- **Hypothesis Agent** — LLM generates diverse chemical reactions (19 types, T=0.9)
- **Verification Agent** — RDKit structural cascade + MMFF94 energetic validation — physical ground truth
- **Reflection Agent** — Causal failure analysis with 10 categories, CARL 4-step decomposition
- **Compilation Agent** — DPO preference pairs, 6-dimension quality scoring

---

# Core Innovation: Physics-Grounded Self-Bootstrapping

::: {style="text-align: center; font-size: 1.5em; margin: 20px 0;"}
**Generate → Verify → Reflect → Accumulate → Repeat**
:::

- Unlike LLM self-evaluation (which **hallucinates**), our system couples generative models with **deterministic physical simulators** as ground-truth oracles

- Failed reactions are **not discarded** — they become structured learning signals that improve future generations

- **Cosine temperature annealing** (1.0 → 0.3) drives exploration → exploitation across 3 bootstrap iterations

> **Example Reflection Trace:**
>
> "The proposed nucleophilic attack is **blocked by severe steric hindrance** from the adjacent tert-butyl group. The transition state requires an impossible geometry due to van der Waals clashes. **Fix**: Use a less bulky electrophile or switch to SN1 conditions."

---

# MAP-Elites Evolutionary Search (GigaEvo)

**Behavior Grid: 26 reaction types × 10 MW bins × 10 fitness bins = 2,600 cells**

| Mutation Operator | Weight |
|---|---|
| Reactant Substitution (bioisostere replacement) | 0.25 |
| Condition Optimization (solvent, temperature, catalyst) | 0.20 |
| Reaction Type Crossover (combine two parent frameworks) | 0.15 |
| Scaffold Hopping (core structure variation) | 0.15 |
| Insight-Guided (CARL reflection trace-driven) | 0.25 |

| Specialist Island | Optimization Objective |
|---|---|
| Diversity | Chemical reaction variety |
| Quality | High fitness scores |
| Novelty | Unique molecular scaffolds |
| Yield | Maximum pass rate |

**Convergence**: Stagnation detection (10 generations no improvement) | **Migration**: 3 elites every 10 generations | **RNG**: Deterministic seed for full reproducibility

---

# CARL Structured Reasoning Chains (Maestro)

```
Step 1: Steric Analysis ──────────────┐
Step 2: Electronic Analysis ──────────┤── Step 4: Causal Synthesis
Step 3: Thermodynamic & Kinetic ──────┘

     Steps 1-3 run in PARALLEL via async DAG engine
```

| Step | Analysis Performed | Chemistry Checks |
|---|---|---|
| **Steric** | van der Waals clashes, transition state geometry accessibility | Baldwin's rules, eclipsing interactions |
| **Electronic** | Frontier orbital compatibility (HOMO-LUMO), electrophilicity/nucleophilicity | HSAB theory, charge distribution |
| **Thermodynamic** | Enthalpy/entropy changes, competing reaction pathways | Activation barrier feasibility |
| **Synthesis** | Merged causal explanation with actionable fix suggestion | Confidence scoring, chemical principles |

---

# Implementation: 5 Phases

| Phase | Key Deliverables | Tests |
|---|---|---|
| **I: Foundation** | Redis state layer, Hydra configs, problem directory | 140 |
| **II: DAG Engine** | Async pipeline, Kahn topological sort | 163 |
| **III: MAP-Elites** | 2,600-cell grid, 5 mutation ops, 4 islands | 202 |
| **IV: CARL Chains** | 4-step parallel DAG reflection, batch filtering | 219 |
| **V: Ablation + Paper** | 7-variant evolution ablation, NeurIPS LaTeX | **230** |

**Technology Stack**

| Layer | Tools & Frameworks |
|---|---|
| LLM | Fireworks AI (MiniMax-M3 + DeepSeek-v3p2), LangChain |
| Agents | Pydantic v2, 4-agent hub-spoke topology |
| Chemistry | RDKit, MMFF94 force fields (physics ground truth) |
| RAG | TF-IDF + NetworkX knowledge graph, 140 indexed docs |
| Evolution | Custom MAP-Elites (GigaEvo architecture pattern) |
| Infrastructure | SQLite checkpoints, Hydra/OmegaConf, Docker, Redis |
| Testing | 230/230 tests, 7,530 source LOC, 2,608 test LOC |

---

# Dataset Results — 172 DPO Pairs

| Split | Pairs |
|---|---|
| **Train** | 124 |
| **Validation** | 6 |
| **Test** | 42 |
| **Total** | **172** |

| Metric | v1.0 (DeepSeek) | v3.0 (MiniMax-M3) | Merged |
|---|---|---|---|
| **Pairs** | 110 | 62 | **172** |
| **Hypotheses** | 167 | 89 | 256 |
| **Pass Rate** | 65.9% | 69.7% | **67.2%** |
| **Reaction Types** | 13 | 18 | **19** |
| **Avg Quality Score** | 0.636 | 0.671 | **0.650** |

Published on HuggingFace: `aayushkrm/autochem-instruct`

---

# Ablation: 7-Variant Evolution Study

| Variant | ME | CARL | Elites | Pass Rate | Quality |
|---|---|---|---|---|---|
| Baseline | — | — | 10 | 69.0% | 0.59 |
| Bootstrap-Only | — | — | 11 | 68.9% | 0.61 |
| **CARL-Only** | — | ✓ | 15 | **78.8%** | 0.67 |
| **MAP-Elites-Only** | ✓ | — | **24** | 75.0% | 0.60 |
| No-Reflection | — | — | 10 | 66.3% | 0.59 |
| No-RAG | — | — | 13 | 68.0% | 0.65 |
| **Full-System** | ✓ | ✓ | **25** | **84.6%** | **0.72** |

**Key Findings:**

- **MAP-Elites scales population** — 2.4× more elite discoveries (24 vs 10)
- **CARL improves quality** — +14% pass rate (78.8% vs 69.0%)
- **Combined (Full-System)** — 3.5× elites, +17% pass rate vs Baseline

---

# GigaEvo + Maestro CARL Integration

**GigaEvo** — GitHub: [FusionBrainLab/gigaevo-core](https://github.com/FusionBrainLab/gigaevo-core), arXiv:2511.17592

| GigaEvo Component | Our Implementation | LOC |
|---|---|---|
| Redis Database | src/evolution/redis_store.py | 229 |
| Async DAG Engine | src/evolution/dag.py | 293 |
| MAP-Elites Evolution | src/evolution/map_elites.py | 491 |
| Problem Interface | problems/autochem/ | 349 |

**Maestro CARL** — GitHub: [AIRI-Institute/maestro-core](https://github.com/AIRI-Institute/maestro-core)

| CARL Component | Our Implementation | LOC |
|---|---|---|
| Event-Action-Result Chains | src/carl/chain.py | 463 |
| StepDescription / ReasoningChain | CARLChain with DAGPipeline | — |
| Parallel DAG Execution | Steps 1-3 parallel | — |

Custom implementations following AIRI's published architecture patterns. 27 scraped research documents in `docs/research/`.

---

# Conclusion & Future Work

## What We Achieved

- 4-agent autonomous pipeline for chemistry data generation
- Self-bootstrapping loop with physical ground truth
- MAP-Elites evolution across 2,600-cell behavior grid
- CARL 4-step causal reasoning chains (parallel DAG)
- 172 DPO pairs across 19 reaction types
- 7-variant ablation study with component impact analysis
- 230/230 tests, 7,530 source LOC, MIT licensed

## Future Directions

- Full xTB energetic validation for reaction barrier estimation
- Distributed MAP-Elites execution for 1000+ hypotheses
- Fine-tune chemistry DSLMs on the resulting dataset
- Replace simulated evaluators with real LLM feedback in evolution loop
- Extend framework to materials science and drug discovery domains

**Auto-ChemInstruct demonstrates that autonomous, physics-grounded data generation is feasible at scale.**

---

# Thank You

## Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Chemistry DSLMs**

🤗 [huggingface.co/datasets/aayushkrm/autochem-instruct](https://huggingface.co/datasets/aayushkrm/autochem-instruct)

🏗️ [github.com/aayushkrm/auto-cheminstruct](https://github.com/aayushkrm/auto-cheminstruct)

<br>

**Questions?**
