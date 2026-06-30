% Auto-ChemInstruct
% Aayush Kumar · TSU × AIRI Institute, Moscow
% NeurIPS Datasets & Benchmarks Track

---

# Auto-ChemInstruct

## Teaching AI to Understand Chemistry — Without Human Help

**Aayush Kumar**

TSU Laboratory of AI in Chemistry × AIRI Institute, Moscow

---

# Why This Matters

AI models need training data to learn chemistry. Today, we create that data by **hiring expert chemists** to manually annotate reactions. This is slow, expensive, and simply doesn't scale.

**But there's a bigger problem.** Existing datasets only show what works. They never explain **why** a reaction fails — why the atoms can't fit together, why the energy barrier is too high. An AI trained only on successes learns nothing about failure.

And if we just ask a language model to generate reactions? **It hallucinates.** It confidently proposes molecules that violate basic physics, because it's a text predictor, not a chemist.

> **Our idea**: What if we could automate the hardest part — checking whether a reaction is physically valid — using computational chemistry?

---

# What We Built

We created Auto-ChemInstruct: **four AI agents working together** to generate, verify, and compile chemistry training data — with zero human annotation.

**How it works:**

- **Agent 1 — Hypothesis Generator**: An LLM creates creative chemical reaction proposals (19 different reaction types)

- **Agent 2 — Physical Verifier**: RDKit and MMFF94 check every reaction against real physics — are the atoms arranged correctly? Is the energy realistic? This is a **real computation**, not the AI guessing.

- **Agent 3 — Causal Reflector**: When a reaction fails, this agent explains **why** — is it steric hindrance blocking the atoms? Is the reaction thermodynamically impossible? It produces a detailed explanation and suggests a fix.

- **Agent 4 — Dataset Compiler**: Builds training pairs — successful reactions paired with failed ones, including the causal explanation of why the failure happened.

**And here's the key**: all of this runs in a **self-improving loop**. Failed reactions are fed back as learning signals. The system gets better with each iteration.

---

# The Physics Difference

This is what separates our system from everything else.

| Traditional AI Pipeline | Auto-ChemInstruct |
|---|---|
| AI generates a reaction | AI generates a reaction |
| AI checks its own work (guesses) | **RDKit + MMFF94 check the physics** |
| Confidently wrong about impossible molecules | **Real energy computed, real geometry checked** |
| Failures are thrown away | **Failures become learning signals** |

**Example**: The AI proposes a nucleophilic attack reaction. Our physics engine detects that a bulky tert-butyl group is **physically blocking** the reactive site — the transition state is geometrically impossible. The Reflection Agent explains this in chemical terms and suggests using a smaller electrophile. The next iteration avoids this mistake.

---

# The Data: What We Actually Produced

Our pipeline generated **172 DPO preference pairs** — each pair teaches AI both what works and what fails.

**What a training pair looks like:**

| Component | Example |
|---|---|
| **Prompt** | "Propose a Suzuki coupling reaction to synthesize a biaryl compound" |
| **Chosen (correct)** | Full reaction: SMILES, conditions (Pd catalyst, 80°C, THF), mechanism steps, 85% yield estimate |
| **Rejected (incorrect)** | Failed reaction + **causal analysis**: "Nucleophilic attack blocked by steric hindrance from tert-butyl group. Transition state impossible. Fix: smaller electrophile or SN1." |

**Dataset stats:**

| | |
|---|---|
| Training pairs | 124 |
| Validation pairs | 6 |
| Test pairs | 42 |
| **Total** | **172** |
| Reaction types | 19 (Suzuki, Diels-Alder, Heck, Wittig, etc.) |
| Quality score | 0.650 average on 6-dimension rubric |
| Pass rate | 67% across 256 total hypotheses |

**Why this data is useful:**

Every existing chemistry dataset is **pass-only** — showing correct reactions but never explaining failures. Our data is different. Each rejected pair includes a **causal explanation** of exactly why the reaction fails and how to fix it. When you train a chemistry language model on this data, it learns:

- What a valid reaction looks like (structure, conditions, yield)
- **Why** a proposed reaction fails (steric, electronic, thermodynamic reasons)
- How to **avoid** failures in future proposals
- The chemical principles behind the success or failure

This is exactly the format needed for RLHF and DPO training — the same techniques used to align ChatGPT and Claude, but applied to chemistry.

**Live on HuggingFace:** `aayushkrm/autochem-instruct`

---

# Two Key Innovations

## MAP-Elites Evolutionary Search

Instead of random trial-and-error, we use an evolutionary algorithm across a **2,600-cell behavior grid** (26 reaction types × 10 molecular weight ranges × 10 quality levels).

Five different mutation strategies evolve the best reactions. Four specialist "islands" optimize different goals — diversity, quality, novelty, and yield — sharing their best discoveries with each other.

**Result**: 2.4× more high-quality reactions discovered.

## CARL Structured Reasoning

When a reaction fails, instead of one vague question ("why did this fail?"), we ask **four focused chemistry questions in parallel**:

1. Is there steric interference? (shape)
2. Are the electrons compatible? (reactivity)
3. Is it thermodynamically favorable? (energy)
4. What's the combined explanation? (synthesis)

**Result**: 14% higher verification pass rate.

---

# What We Achieved

**The Pipeline** — 4 AI agents working autonomously. 230 automated tests. 7,500 lines of code. Zero human annotation needed.

**The Dataset** — 172 training pairs covering 19 reaction types. Every failure includes a causal explanation — not just "this didn't work" but "this failed because of steric hindrance at the C4 position, try a smaller electrophile." Published on HuggingFace, ready for anyone to use for RLHF or DPO training.

**The Proof** — 7 different configurations tested to prove every component matters:

| Configuration | Results |
|---|---|
| **Basic system only** | 69% pass rate |
| **+ CARL reasoning** | 79% pass rate (+14%) |
| **+ MAP-Elites evolution** | 2.4× more discoveries |
| **Full system (everything)** | **85% pass rate, 3.5× discoveries** |

**The Technology** — Built on AIRI Institute's GigaEvo and Maestro CARL frameworks. Python, RDKit, MMFF94 physics engine, LangChain, Fireworks AI.

GitHub: `github.com/aayushkrm/auto-cheminstruct`

---

# Key Takeaway

**We proved that autonomous, physics-grounded data generation for scientific AI is not just theoretically possible — it's practical, reproducible, and works.**

A system that:

- generates creative chemical hypotheses
- validates them against real physics
- explains its failures in chemical terms
- learns from those failures
- and produces training data that teaches AI both what works and why things fail

**All without a single human annotator.**

---

# Thank You

**Auto-ChemInstruct**

Agent-Driven Synthesization of RLHF Data for Chemistry DSLMs

172 Training Pairs · 19 Reaction Types · 230 Automated Tests

`github.com/aayushkrm/auto-cheminstruct`

`huggingface.co/datasets/aayushkrm/autochem-instruct`

**Questions?**
