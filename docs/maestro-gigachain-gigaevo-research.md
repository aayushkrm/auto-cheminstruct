# Maestro, GigaChain, GigaEvo — Research Findings

## Research Date: 2026-05-11

---

## 1. GigaChain — DEPRECATED

| Aspect | Detail |
|--------|--------|
| **What it is** | A LangChain fork adapted for the Russian AI ecosystem (Sber's GigaChat models) |
| **PyPI** | `gigachain` v100.0.4 — **marked as DEPRECATED** |
| **Deprecation message** | "This package was deprecated. Upgrade to clear langchain + langchain_gigachat. Uninstall all gigachain-* packages!" |
| **Replacement** | `langchain-gigachat` v0.5.1 — just a LangChain provider plugin for GigaChat models |
| **GitHub** | https://github.com/ai-forever/langchain-gigachat |
| **Relevance to us** | **ZERO.** It's literally LangChain with GigaChat models. We use Fireworks AI's DeepSeek-v3p2. We gain nothing by switching. |

**Verdict: DO NOT USE.** GigaChain was abandoned. The ecosystem moved to `langchain-gigachat` which is just `pip install langchain-gigachat` alongside regular LangChain. Our LangChain setup is superior.

---

## 2. Maestro — Multi-Agent Orchestration

| Aspect | Detail |
|--------|--------|
| **What it is** | Modular framework + orchestrator for building multi-agent digital assistants |
| **Developer** | AIRI research group "Multimodal AI Architectures" |
| **Published** | Dec 25, 2025 (Medium article) |
| **Docs** | https://airi-institute.github.io/maestro-cover/ |
| **GitHub** | https://github.com/AIRI-Institute/maestro-core (public, MIT) |
| **Architecture** | Docker-based, TOML config, modular services: gateway, chat-manager, LLM-hub, document-extractor |
| **Key features** | • Strict task decomposition (each agent solves ONE atomic business task) • Centralized moderation loop (analyzes agent outputs, not just user inputs) • Input/output data specs for hard behavioral boundaries • Multi-LLM support (OpenAI-compatible + GigaChat) • Open documentation |

### Architecture

```
Gateway → Chat Manager → [Agent₁, Agent₂, ...] → LLM Hub
              ↕                              ↕
         Moderation Loop              External Tools/APIs
```

### Installation

```bash
git clone https://github.com/AIRI-Institute/maestro-core
make build setup-env up  # Docker-based
```

### Relevance to Auto-ChemInstruct

| Our current orchestrator | Maestro |
|---|---|
| Single Python class (~400 lines) | Dockerized service mesh |
| Sequential agent calls | Service-based agent management |
| SQLite checkpoints | Centralized state management |
| Direct Python imports | TOML config + LLM hub routing |
| No moderation | Built-in safety guardrails |
| Ad-hoc error handling | Formal input/output specifications |

**Potential value:** Maestro would add formal safety guardrails, input/output validation at each step, and distributed agent deployment. However, it's designed for **customer-facing digital assistants** (chatbots, document describers), not **scientific computation pipelines**. The Docker-based deployment adds significant complexity for a research project.

**Verdict: MODERATE VALUE.** The formal agent boundaries and safety guardrails would improve robustness, but the infrastructure overhead is high. We'd gain the "Maestro badge" for the paper but lose development velocity. Can be cited as the theoretical framework while using our lighter orchestrator.

---

## 3. GigaEvo — LLM-Powered Evolutionary Optimization

| Aspect | Detail |
|--------|--------|
| **What it is** | Open-source framework for LLM-guided evolutionary computation |
| **Paper** | arXiv:2511.17592 (Nov 2025, cited by 7) |
| **GitHub** | https://github.com/AIRI-Institute/gigaevo-core (MIT, 24 ★) |
| **Alt GitHub** | https://github.com/FusionBrainLab/gigaevo-core (115 ★) |
| **Python** | 3.12+ |
| **Dependencies** | Redis, Hydra, LangGraph, OpenRouter/OpenAI-compatible LLMs |
| **Inspired by** | Google DeepMind's AlphaEvolve (Novikov et al., 2025) |

### Four Core Components

```
┌─────────────────────────────────────────────────┐
│ 1. Redis Database                                │
│    • Stores evolutionary units (UUID, code,      │
│      lifecycle state, metrics, lineage)          │
│    • Optimistic concurrency control              │
│    • Real-time bidirectional lineage tracking    │
├─────────────────────────────────────────────────┤
│ 2. Async DAG Execution Engine                    │
│    • Based on Python asyncio                     │
│    • Concurrent program processing               │
│    • Cascading validation (cheap→expensive)      │
│    • Stage caching + automatic skip              │
├─────────────────────────────────────────────────┤
│ 3. Evolution Engine                              │
│    • MAP-Elites quality-diversity algorithm      │
│    • Behavioral cell mapping (fitness × validity)│
│    • Fitness-proportional elite selection        │
│    • Multi-island with periodic migration        │
├─────────────────────────────────────────────────┤
│ 4. Mutation Operator (LangGraph-based)           │
│    • Constructs prompts from: task description   │
│      + parent code + metrics + insights +        │
│      lineage analysis                            │
│    • LLM generates offspring programs            │
│    • Rewrite-based and diff-based mutation modes │
│    • Multi-model routing support                 │
└─────────────────────────────────────────────────┘
```

### Program Lifecycle

```
FRESH → DAG_PROCESSING_STARTED → DAG_PROCESSING_COMPLETED → EVOLVING → FRESH ...
                                            ↓
                                    (if rejected) → DISCARDED
```

### Quick Start

```bash
pip install -e .
redis-server
python run.py problem.name=heilbron max_generations=10 model_name=openai/gpt-4o
```

### Key Innovation: Insight Generation

The mutation operator doesn't just mutate randomly — the LLM generates **insights** explaining *why* previous iterations failed, then uses those insights to guide future mutations. This is exactly the "Reflection Agent" pattern in Auto-ChemInstruct, but embedded in an evolutionary loop.

### MAP-Elites vs Our Current Approach

| Our Self-Bootstrapping | GigaEvo MAP-Elites |
|---|---|
| Sequential: generate → verify → reflect → repeat | Population-based: 100s of programs evolved in parallel |
| No fitness function (pass/fail only) | Multi-objective fitness (validity × energy × diversity) |
| Temperature annealing only | Fitness-proportional selection + behavioral diversity |
| No diversity objective | MAP-Elites fills ALL behavioral cells |
| Simple learning context | Redis-backed genealogy + insight accumulation |
| Scale: 15-20 hypotheses/run | Scale: 100s-1000s per generation |

### Benchmark Results (from paper)

GigaEvo reproduced AlphaEvolve results on:
- **Heilbronn triangle problem**: Found competitive solutions
- **Circle packing in squares**: Reproduced published results
- **Kissing numbers in high dimensions**: Matched known bounds
- **Prompt and agent evolution**: Evolved prompts that outperform hand-crafted ones

**Verdict: VERY HIGH VALUE.** This is the framework that could genuinely transform Auto-ChemInstruct. Our self-bootstrapping loop is effectively a primitive version of what GigaEvo does properly. Integrating it would:
1. Add MAP-Elites quality-diversity (currently we don't optimize for diversity)
2. Enable parallel hypothesis evolution at scale (100s per generation vs 15-20)
3. Add proper genealogy tracking (currently just linear accumulation)
4. Make the paper about "GigaEvo-powered evolutionary hypothesis generation" rather than "a pipeline that generates data"
