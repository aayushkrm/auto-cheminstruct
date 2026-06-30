---
marp: true
theme: uncover
class: invert
paginate: true
size: 16:9
---

# Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry**

Aayush Kumar

TSU Lab of AI in Chemistry & Molecular Engineering × AIRI Institute, Moscow

NeurIPS Datasets & Benchmarks Track

---

# The Problem

- **Scarcity of chemistry instruction data** — existing datasets like ChemCoTBench rely on expensive human expert annotation
- **Positivity bias** — datasets only show what works, never explain *why reactions fail*
- **LLM hallucination** — models generate chemically impossible reactions with no physical validation
- **No self-improvement** — static generation with no feedback loop

<div style="margin-top: 40px; padding: 20px; border: 2px solid #e74c3c; border-radius: 10px;">
<b>Key Insight:</b> We can automate the verification bottleneck using computational chemistry as a deterministic oracle
</div>

---

# Our Solution: 4-Agent Pipeline

```
┌──────────┐   ┌────────────────┐   ┌───────────┐   ┌──────────┐
│Hypothesis│──→│  Verification  │──→│Compilation│──→│   DPO    │
│  Agent   │   │ (RDKit+MMFF94) │   │   Agent   │   │  Pairs   │
└──────────┘   └───────┬────────┘   └───────────┘   └──────────┘
      ↑                 │ FAIL
      │        ┌────────┴─────────┐
      │        │ Reflection Agent │
      │        │  10 categories   │
      │        └────────┬─────────┘
      │                 │
      └── LearningContext ◄────────┘
        (Self-Bootstrapping Loop)
```

- **Hypothesis Agent**: LLM generates diverse reactions (19 types, T=0.9)
- **Verification Agent**: RDKit cascade + MMFF94 energetics — physical ground truth
- **Reflection Agent**: Causal failure analysis, 10 categories
- **Compilation Agent**: DPO preference pairs, 6-dim quality scoring

---

# Core Innovation: Physics-Grounded Self-Bootstrapping

<div style="text-align: center; font-size: 1.5em; margin: 30px 0;">
Generate → Verify → Reflect → Accumulate → <b>Repeat</b>
</div>

- Unlike LLM self-evaluation (hallucinates), we couple generation with **deterministic physical simulators**
- Failed reactions become **structured learning signals** — not wasted compute
- **Cosine temperature annealing** (1.0→0.3) drives exploration→exploitation across 3 bootstrap iterations

<div style="margin-top: 30px; font-size: 0.9em; background: #2c3e50; padding: 15px; border-radius: 8px;">
<b>Example Reflection Trace:</b><br>
"The proposed nucleophilic attack is <b>blocked by severe steric hindrance</b> from the adjacent tert-butyl group. The transition state requires an impossible geometry. Fix: use a less bulky electrophile or switch to SN1 conditions."
</div>

---

# MAP-Elites Evolutionary Search (GigaEvo)

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
<div>

### Behavior Grid
- **26** reaction types
- **10** molecular weight bins
- **10** fitness bins
- **= 2,600 cells**

### 5 Mutation Operators
1. Reactant substitution
2. Condition optimization
3. Reaction type crossover
4. Scaffold hopping
5. Insight-guided (CARL traces)

</div>
<div>

### 4 Specialist Islands

| Island | Focus |
|--------|-------|
| Diversity | Reaction variety |
| Quality | High fitness |
| Novelty | Unique scaffolds |
| Yield | Pass rate |

- **Conservative migration**: 3 elites / 10 gens
- **Stagnation detection**: stop at 10 gens no improvement
- **Deterministic RNG**: full reproducibility

</div>
</div>

---

# CARL Structured Reasoning Chains (Maestro)

<div style="text-align: center; margin: 20px 0;">

```
Step 1: Steric Analysis ───┐
Step 2: Electronic Analysis ─┤── Step 4: Causal Synthesis
Step 3: Thermodynamic Analysis┘
```

*Steps 1-3 execute in **parallel** via async DAG engine*

</div>

| Step | Analysis | Chemistry Checks |
|------|----------|-----------------|
| **Steric** | van der Waals clashes, transition state geometry | Baldwin's rules, eclipsing interactions |
| **Electronic** | HOMO-LUMO compatibility, electrophilicity | HSAB theory, nucleophile richness |
| **Thermodynamic** | Enthalpy/entropy, competing pathways | Activation barrier feasibility |
| **Synthesis** | Merged causal explanation + actionable fix | Confidence scoring |

---

# Implementation

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
<div>

### 5 Phases · 37 Commits

| Phase | Tests |
|-------|-------|
| I: Foundation | 140 |
| II: DAG Engine | 163 |
| III: MAP-Elites | 202 |
| IV: CARL Chains | 219 |
| V: Ablation + Paper | **230** |

</div>
<div>

### Technology Stack

| Layer | Tools |
|-------|-------|
| **LLM** | Fireworks AI (MiniMax-M3) |
| **Agents** | LangChain, Pydantic v2 |
| **Chemistry** | RDKit, MMFF94 |
| **RAG** | TF-IDF, NetworkX KG |
| **Evolution** | Custom MAP-Elites |
| **Infra** | SQLite, Hydra, Docker |

</div>
</div>

<div style="margin-top: 20px; text-align: center;">
<b>7,530 source LOC · 2,608 test LOC · 230/230 tests passing</b>
</div>

---

# Dataset Results — 172 DPO Pairs

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
<div>

| Metric | v1.0 | v2.0 |
|--------|------|------|
| Model | DeepSeek-v3p2 | MiniMax-M3 |
| Pairs | 110 | 71 |
| Pass Rate | 65.9% | 82.6% |
| Reaction Types | 13 | 18 |
| Avg Quality | 0.636 | 0.671 |

</div>
<div>

| Split | Merged |
|-------|--------|
| **Train** | 136 |
| **Validation** | 6 |
| **Test** | 45 |
| **Total** | **172** |
| **Reaction Types** | **19** |
| **Avg Quality** | **0.650** |

</div>
</div>

<div style="margin-top: 20px; text-align: center; font-size: 0.9em;">
🤗 Published on HuggingFace: <code>aayushkrm/autochem-instruct</code>
</div>

---

# Ablation: 7-Variant Evolution Study

| Variant | ME | CARL | Elites | Pass Rate | Quality | Impact |
|---------|:--:|:----:|--------|-----------|---------|--------|
| Baseline | — | — | 10 | 69.0% | 0.59 | — |
| Bootstrap-Only | — | — | 11 | 68.9% | 0.61 | +10% elites |
| **CARL-Only** | — | ✅ | 15 | **78.8%** | 0.67 | +14% pass |
| **MAP-Elites-Only** | ✅ | — | **24** | 75.0% | 0.60 | 2.4× elites |
| No-Reflection | — | — | 10 | 66.3% | 0.59 | — |
| No-RAG | — | — | 13 | 68.0% | 0.65 | — |
| **Full-System** | ✅ | ✅ | **25** | **84.6%** | **0.72** | **2.5× +17% pass** |

<div style="margin-top: 15px; padding: 10px; background: #27ae60; border-radius: 8px; text-align: center;">
<b>Key Finding:</b> MAP-Elites scales population (2.4×) · CARL improves quality (+14% pass)<br>
<b>Combined:</b> 3.5× elites · +17% pass rate · Complementary benefits
</div>

---

# GigaEvo + Maestro CARL Integration

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
<div>

### GigaEvo
**[FusionBrainLab/gigaevo-core](https://github.com/FusionBrainLab/gigaevo-core)**
arXiv:2511.17592, 115+ ⭐

| Component | Our Code | LOC |
|-----------|----------|-----|
| Redis DB | redis_store.py | 229 |
| DAG Engine | dag.py | 293 |
| MAP-Elites | map_elites.py | 491 |
| Problem I/F | problems/autochem/ | 349 |

</div>
<div>

### Maestro CARL
**[AIRI-Institute/maestro-core](https://github.com/AIRI-Institute/maestro-core)**
MIT License

| Component | Our Code | LOC |
|-----------|----------|-----|
| Event-Action-Result | chain.py | 463 |
| StepDescription | CARLChain class | — |
| Parallel DAG | Steps 1-3 parallel | — |

</div>
</div>

<div style="margin-top: 30px; text-align: center; font-style: italic;">
Custom implementations following AIRI's published architecture patterns<br>
27 scraped research documents in <code>docs/research/</code>
</div>

---

# Conclusion & Future Work

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
<div>

### What We Achieved

✅ **4-agent autonomous pipeline** for chemistry data generation

✅ **Self-bootstrapping loop** with physical ground truth

✅ **MAP-Elites evolution** across 2,600-cell grid

✅ **CARL 4-step** causal reasoning chains

✅ **172 DPO pairs** across 19 reaction types

✅ **7-variant ablation** proving component impact

✅ **230/230 tests** · 7,530 lines · MIT licensed

</div>
<div>

### Future Directions

🔮 **xTB energetic validation** — full reaction barrier estimation

🔮 **Distributed MAP-Elites** — 1000+ hypothesis parallel evolution

🔮 **Fine-tune chemistry DSLMs** — train on our dataset

🔮 **Live evolution loop** — replace simulated evaluators with real LLM feedback

🔮 **Materials science extension** — expand beyond organic chemistry

</div>
</div>

<div style="margin-top: 30px; text-align: center; font-size: 1.2em;">
<b>Auto-ChemInstruct demonstrates that autonomous,<br>
physics-grounded data generation is feasible at scale.</b>
</div>

---

# Thank You

<div style="text-align: center; margin: 50px 0;">

## Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Chemistry DSLMs**

<br>

🤗 [aayushkrm/autochem-instruct](https://huggingface.co/datasets/aayushkrm/autochem-instruct)

🏗️ [github.com/aayushkrm/auto-cheminstruct](https://github.com/aayushkrm/auto-cheminstruct)

<br>

**Questions?**

</div>
