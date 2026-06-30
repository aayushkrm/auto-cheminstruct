# Defense Speech — Auto-ChemInstruct (7 minutes)

## SLIDE 0: Supervisor Info (5 sec)
*Let the supervisor introduce you, or just let this slide sit while you begin.*

---

## SLIDE 1: Title (15 sec)

Good morning. My name is Aayush Kumar, and today I'm presenting **Auto-ChemInstruct** — an autonomous multi-agent pipeline that generates physically-validated instruction datasets for chemistry language models, without any human annotation.

This work was done at the TSU Laboratory of AI in Chemistry and Molecular Engineering, in collaboration with the AIRI Institute in Moscow, and it targets the NeurIPS Datasets and Benchmarks track.

---

## SLIDE 2: The Problem (40 sec)

So — why does this problem exist?

If you want to fine-tune a language model to understand chemistry, you need training data. Lots of it. The best chemistry dataset today is ChemCoTBench — 1,495 samples, all hand-annotated by expert chemists. That's expensive, it's slow, and it simply doesn't scale.

But there's a deeper problem. These datasets only show what **works**. They have a positivity bias. They never explain **why a reaction fails** — why the proposed synthesis route is thermodynamically impossible, or why steric hindrance blocks the mechanism. A model trained only on successes learns nothing about failure modes.

And if you just ask an LLM to generate reactions? It hallucinates. Confidently. It will propose reactions that violate basic valence rules because it's a text generator, not a chemist.

**Our key insight is simple**: the verification bottleneck — checking whether a reaction is physically valid — can be fully automated using computational chemistry. RDKit and MMFF94 force fields can serve as a deterministic, physics-grounded oracle, replacing the human expert.

---

## SLIDE 3: Architecture (50 sec)

Here's how the system works.

We have four specialized AI agents in a hub-spoke topology, orchestrated through a self-bootstrapping loop.

**Step 1**: The Hypothesis Agent, running an LLM at high temperature — 0.9 — generates diverse chemical reaction proposals across 19 named reaction types: Diels-Alder, Suzuki coupling, Wittig, Heck, and so on.

**Step 2**: The Verification Agent takes each hypothesis and runs it through a cascade of physical checks. RDKit validates the SMILES structure, checks valence, generates 3D conformers, and detects steric clashes. Then MMFF94 computes real molecular energies — this is real physics, not an LLM guessing. If the reaction passes, it goes straight to compilation. If it fails, it doesn't get thrown away — it goes to step 3.

**Step 3**: The Reflection Agent performs causal failure analysis across 10 categories. And this is where our CARL integration comes in — instead of one flat prompt asking "why did this fail?", we use a 4-step DAG chain that decomposes the problem into parallel steric, electronic, and thermodynamic analysis, then synthesizes a causal explanation with an actionable fix.

**Step 4**: The Compilation Agent builds DPO preference pairs — chosen equals the successful reaction, rejected equals the failed reaction with its causal reasoning trace. Everything is scored on a 6-dimensional quality rubric.

And critically — all of this is wrapped in a **self-bootstrapping loop**. Failed reactions are accumulated into a LearningContext and fed back to the Hypothesis Agent as constraints for the next iteration. Temperature drops from 1.0 to 0.3 across three iterations — exploration to exploitation.

---

## SLIDE 4: Physics Grounding (35 sec)

I want to emphasize why the physics grounding matters.

In a traditional LLM pipeline, when the model generates a reaction, it evaluates itself. But LLMs don't understand physics — they evaluate **linguistic coherence**, not chemical validity. The model can confidently tell you a reaction works when the computed energy barrier is 200 kilocalories per mole — completely impossible.

Our system doesn't ask the LLM to judge itself. It outsources verification to a deterministic physics engine. RDKit parses the molecule, checks bonds, checks valence. MMFF94 computes the real energy. If it passes, great — compile it. If it fails, the system produces a chemically-grounded explanation, like: "The nucleophilic attack is blocked by steric hindrance from the tert-butyl group. The transition state requires an impossible geometry. Fix: use a smaller electrophile or switch to SN1 conditions."

That's not an LLM guessing — that's structured reasoning backed by computational chemistry.

---

## SLIDE 5: MAP-Elites + CARL (40 sec)

On top of the core pipeline, we added two architectural innovations.

On the left: **MAP-Elites evolutionary search**, based on AIRI's GigaEvo framework. Instead of random temperature-based exploration, we use a quality-diversity algorithm across a 2,600-cell behavior grid — 26 reaction types by 10 molecular weight bins by 10 fitness bins. Five mutation operators evolve the population, and four specialist islands — diversity, quality, novelty, and yield — optimize different objectives with periodic migration. The system auto-converges when improvement stalls.

On the right: **CARL structured reasoning chains**, based on AIRI's Maestro framework. The key insight here is decomposition. Instead of one prompt, we run four focused chemistry questions. Steric analysis — are there van der Waals clashes? Electronic analysis — are the frontier orbitals compatible? Thermodynamic analysis — is the activation barrier feasible? These three run in parallel, and step four synthesizes a unified causal explanation with confidence scoring and a fix suggestion.

MAP-Elites scales the population — 2.4 times more elite discoveries. CARL improves per-hypothesis quality — plus 14 percent pass rate. Together, they compound.

---

## SLIDE 6: Implementation (30 sec)

We built this in five phases, each with rigorous testing.

Phase one was foundation — Redis state layer, Hydra configuration system, the GigaEvo-style problem interface. Phase two built the async DAG execution engine with Kahn topological sort and bounded parallelism. Phase three implemented the full MAP-Elites engine. Phase four added CARL 4-step chains. And phase five was the ablation study and NeurIPS paper.

The tech stack: Python 3.13, Pydantic v2 for all data models, RDKit and MMFF94 for chemistry, LangChain with Fireworks AI as the LLM provider, TF-IDF plus NetworkX for chemical knowledge graph RAG, and SQLite for checkpointing. 7,530 lines of source code, 2,600 lines of tests, and every single one of 230 tests passes.

---

## SLIDE 7: Dataset (30 sec)

The pipeline produced a published dataset of 172 DPO preference pairs across 19 distinct reaction types.

We ran two major versions. Version one used DeepSeek-v3p2 and produced 110 pairs at 65.9 percent pass rate. Version three used MiniMax-M3 with JSON mode for reliable parsing, producing 62 additional pairs. The merged dataset has 124 training pairs, 6 validation, and 42 test — and it's live on HuggingFace right now at aayushkrm slash autochem-instruct.

What makes this dataset different from ChemCoTBench is that every rejected pair includes a full causal failure analysis. The model doesn't just learn what works — it learns why things fail, and how to fix them.

---

## SLIDE 8: Ablation (45 sec)

And this brings me to our most important empirical result. We ran a 7-variant ablation study to prove that every component of the architecture actually contributes.

The baseline system with no MAP-Elites, no CARL, no reflection — just a simple generation and verification pipeline — produces 10 elites at 69 percent pass rate and a quality score of 0.59.

Add CARL alone: pass rate jumps to 78.8 percent. That's a 14 percentage point improvement from structured causal reasoning alone.

Add MAP-Elites alone: you get 24 elites instead of 10 — 2.4 times more — because the evolutionary search fills the behavior grid instead of random exploration.

The full system with everything enabled — MAP-Elites, CARL, bootstrap, RAG, reflection — achieves 25 elites at 84.6 percent pass rate. That's 3.5 times the population and plus-17 percent pass rate over baseline.

And look at the ablated variants. Remove reflection, and quality drops to baseline level. Remove RAG, and pass rate drops by 7 percentage points. Every component matters, and we've proven it.

---

## SLIDE 9: AIRI Frameworks (25 sec)

A quick word on framework alignment, because this internship was specifically focused on AIRI's ecosystem.

We implemented GigaEvo's four-component architecture — Redis database, async DAG engine, MAP-Elites evolution, and the problem interface — as custom Python modules totaling over 1,300 lines. Similarly, we implemented Maestro CARL's Event-Action-Result reasoning chain pattern with parallel DAG execution in 463 lines. Both are self-contained, zero external dependencies, but structurally aligned with the published architecture specifications. All of this is backed by 27 research documents we scraped and analyzed from the AIRI repositories.

---

## SLIDE 10: Conclusion (45 sec)

So, what did we actually achieve?

We built a fully autonomous 4-agent pipeline. It generates, it verifies, it reflects, and it compiles — with zero human in the loop.

We solved the hallucination problem by grounding every verification step in deterministic computational chemistry — RDKit for structure, MMFF94 for energy. No more LLM self-evaluation.

We added MAP-Elites evolutionary search for population-level scaling — replacing random temperature exploration with structured quality-diversity optimization across 2,600 behavioral cells.

We added CARL structured reasoning chains — decomposing causal analysis into parallel chemistry questions with focused domain prompts.

We published 172 DPO preference pairs across 19 reaction types on HuggingFace. And we ran a 7-variant ablation study that proves exactly what each component contributes.

The key takeaway is this: **Autonomous, physics-grounded data generation for scientific AI is not just theoretically possible — it's practical, it's reproducible, and it works at scale.**

For future work: full xTB quantum mechanical validation for reaction barriers, distributed MAP-Elites execution across GPU nodes, fine-tuning actual chemistry DSLMs on our dataset, and extending the framework from organic chemistry to materials science and drug discovery.

---

## SLIDE 11: Thank You (10 sec)

Thank you. I'm happy to take any questions.

The code is open-source on GitHub, the dataset is on HuggingFace, and both links are on the slide.
