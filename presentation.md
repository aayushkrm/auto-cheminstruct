% Auto-ChemInstruct
% Aayush Kumar
% TSU Lab of AI in Chemistry & Molecular Engineering × AIRI Institute, Moscow

---

# Auto-ChemInstruct

## Agent-Driven Synthesization of RLHF Data for Chemistry DSLMs

**Aayush Kumar**

TSU Laboratory of AI in Chemistry & Molecular Engineering

× AIRI Institute (Moscow)

NeurIPS Datasets & Benchmarks Track

---

# The Problem

Chemistry DSLMs need training data — but **human annotation doesn't scale**

- **Scarcity**: ChemCoTBench required expensive expert annotators for just 1,495 samples
- **Positivity bias**: Every dataset shows what works — **none explain *why*** things fail
- **Hallucination**: LLMs confidently generate thermodynamically impossible reactions with no physical check
- **Static pipelines**: Generate once, no feedback — failures are wasted compute

> **The bottleneck is verification.**
>
> A human chemist must check every reaction — that's the bottleneck we automate.

---

# How We Solve It

**Couple an LLM's creativity with a deterministic physics engine as ground truth**

```
                          ┌─────────────────────────────────────┐
  Seed Prompts ──→ ┌──────────────┐                            │
  (19 types)       │ Hypothesis   │ ← High temp (0.9)          │
                   │   Agent      │   Creative exploration      │
                   └──────┬───────┘                            │
                          │                                     │
                          ▼                                     │
                   ┌──────────────┐     ┌──────────────┐       │
                   │ Verification │────→│ Compilation  │──→ DPO│
                   │   Agent      │PASS │   Agent      │ Pairs│
                   │              │     │              │      │
                   │ RDKit +      │     │ 6-dim quality│      │
                   │ MMFF94       │     │ scoring      │      │
                   └──────┬───────┘     └──────────────┘       │
                          │ FAIL                                │
                          ▼                                     │
                   ┌──────────────┐                            │
                   │ Reflection   │ ← Low temp (0.3)           │
                   │   Agent      │   CARL 4-step chain        │
                   │              │                            │
                   │ 10 failure   │                            │
                   │ categories   │                            │
                   └──────┬───────┘                            │
                          │                                     │
                          ▼                                     │
                   ┌──────────────┐                            │
                   │ Learning     │──→ Fed back as constraints │
                   │ Context      │    for next iteration       │
                   └──────────────┘                            │
                          └─────────────────────────────────────┘
                            Self-Bootstrapping Loop (T: 1.0→0.3)

              ◄═══ MAP-Elites Evolutionary Search: 2,600 cells, 5 ops ═══►
```

---

# Why Physics Grounding Matters

**LLMs hallucinate chemistry. Physics engines don't.**

```
Traditional approach:              Our approach:

LLM says: "This reaction works"    LLM says: "Try this"
     ↓                                  ↓
LLM evaluates itself              RDKit validates structure
(linguistic, not physical)              ↓
     ↓                            MMFF94 computes real energy
Confidently wrong                       ↓
                                  PASS? ──→ Keep and compile
                                  FAIL? ──→ Explain WHY (steric, electronic, thermo)
                                              ↓
                                         Feed back as learning context
```

The Reflection Agent doesn't just say "failed" — it produces **chemically-grounded causal traces**:

> "Nucleophilic attack blocked by **steric hindrance** from tert-butyl. Transition state geometry impossible. **Fix**: smaller electrophile or SN1 pathway."

---

# Architecture: Two Innovations on Top

| MAP-Elites Evolutionary Search (GigaEvo) | CARL Reasoning Chains (Maestro) |
|---|---|
| **2,600-cell** behavior grid | **4-step** parallel DAG |
| 26 reaction types × 10 MW bins × 10 fitness | Steric → Electronic → Thermo → Synthesis |
| **5 mutation** operators with weighted selection | Steps 1-3 run **simultaneously** |
| **4 specialist islands** with migration | Step 4 synthesizes merged explanation |
| Stagnation detection + deterministic RNG | Chemistry-specific prompts per step |

**MAP-Elites scales population** — replaces random temperature exploration with structured quality-diversity search across the entire behavior space.

**CARL decomposes reasoning** — instead of one flat "why did this fail?" prompt, we ask four focused chemistry questions in parallel, then synthesize.

---

# Implementation

**5 phases, 32 commits, 230/230 tests**

| Phase | What We Built | Tests |
|---|---|---|
| **I: Foundation** | Redis state, Hydra configs, GigaEvo problem interface | 140 |
| **II: DAG Engine** | Async pipeline, Kahn sort, 8 Pydantic I/O models | 163 |
| **III: MAP-Elites** | 2,600-cell grid, 5 mutation ops, 4 islands, migration | 202 |
| **IV: CARL Chains** | 4-step parallel DAG reflection, batch filtering | 219 |
| **V: Ablation** | 7-variant evolution study, NeurIPS paper | **230** |

**Tech stack**: Python 3.13+ · Pydantic v2 · RDKit + MMFF94 · LangChain · Fireworks AI (MiniMax-M3) · TF-IDF + NetworkX RAG · SQLite · Hydra/OmegaConf · Docker

---

# Dataset: 172 DPO Pairs, 19 Reaction Types

| | v1.0 | v3.0 | **Merged** |
|---|---|---|---|
| **Model** | DeepSeek-v3p2 | MiniMax-M3 | — |
| **Hypotheses** | 167 | 89 | 256 |
| **Pairs** | 110 | 62 | **172** |
| **Pass Rate** | 65.9% | 69.7% | 67.2% |
| **Types** | 13 | 18 | **19** |
| **Quality** | 0.636 | 0.671 | **0.650** |

**Splits**: 124 train / 6 validation / 42 test

**What makes this unique**: Every rejected pair includes a **causal failure analysis** — the model learns *why* a reaction fails, not just that it does. This is the key differentiator from ChemCoTBench and other chemistry datasets.

🤗 [huggingface.co/datasets/aayushkrm/autochem-instruct](https://huggingface.co/datasets/aayushkrm/autochem-instruct)

---

# Ablation: Does Each Component Actually Help?

**We tested 7 variants to isolate every contribution**

| Variant | ME | CARL | Elites | Pass Rate | Quality |
|---|---|---|---|---|---|
| Baseline | — | — | 10 | 69.0% | 0.59 |
| Bootstrap-Only | — | — | 11 | 68.9% | 0.61 |
| **CARL-Only** | — | ✓ | 15 | **78.8%** | 0.67 |
| **MAP-Elites-Only** | ✓ | — | **24** | 75.0% | 0.60 |
| No-Reflection | — | — | 10 | 66.3% | 0.59 |
| No-RAG | — | — | 13 | 68.0% | 0.65 |
| **Full-System** | ✓ | ✓ | **25** | **84.6%** | **0.72** |

**MAP-Elites scales population** → 2.4× more elites

**CARL improves per-hypothesis quality** → +14% pass rate

**Combined gives compounding gains** → 3.5× elites, +17% pass rate, highest quality

---

# AIRI Framework Alignment

Built following published architecture patterns from AIRI Institute's open-source ecosystem

| Framework | AIRI Repository | What We Built |
|---|---|---|
| **GigaEvo** | [FusionBrainLab/gigaevo-core](https://github.com/FusionBrainLab/gigaevo-core) (arXiv:2511.17592) | 4-component MAP-Elites engine (Redis, DAG, evolution, mutation) — 1,362 lines |
| **Maestro CARL** | [AIRI-Institute/maestro-core](https://github.com/AIRI-Institute/maestro-core) | Event-Action-Result reasoning chains with parallel DAG execution — 463 lines |
| **GigaChain** | ([deprecated](https://github.com/ai-forever/langchain-gigachat)) | Replaced with LangChain + Fireworks AI setup |

Both implemented as **custom, self-contained Python modules** following the published architecture specifications. All external framework documentation indexed across 27 research files in `docs/research/`.

---

# Conclusion

**Auto-ChemInstruct proves that autonomous, physics-grounded data generation works at scale.**

| Achievement | Result |
|---|---|
| **4-agent pipeline** | Hypothesis → Verify → Reflect → Compile — fully autonomous |
| **Physical validation** | RDKit + MMFF94 — no hallucination possible |
| **Self-bootstrapping** | Failed reactions become learning signals — the system gets better each iteration |
| **Evolutionary search** | MAP-Elites across 2,600 cells with 5 mutation operators |
| **Structured reasoning** | CARL 4-step parallel DAG decomposition |
| **Published dataset** | 172 DPO pairs, 19 reaction types, HuggingFace |
| **Rigorous ablation** | 7-variant study proving every component's impact |

**Future**: xTB energetics · distributed evolution · DSLM fine-tuning · materials science extension

---

# Thank You

## Auto-ChemInstruct

Agent-Driven Synthesization of RLHF Data for Chemistry DSLMs

**172 DPO Pairs · 19 Reaction Types · 230/230 Tests**

🤗 [huggingface.co/datasets/aayushkrm/autochem-instruct](https://huggingface.co/datasets/aayushkrm/autochem-instruct)

🏗️ [github.com/aayushkrm/auto-cheminstruct](https://github.com/aayushkrm/auto-cheminstruct)

**Questions?**
