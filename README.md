# Auto-ChemInstruct

**Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry**

An autonomous, self-verifying multi-agent pipeline that synthetically generates, physically validates, and systematically compiles massive instructional datasets for complex chemical reasoning tasks.

## Overview

Auto-ChemInstruct addresses the critical bottleneck in training domain-specific language models for chemistry: the lack of high-quality, physically-validated instruction datasets. The system employs a sophisticated multi-agent architecture:

- **Hypothesis Agent**: Generates creative molecular reaction pathways
- **Verification Agent**: Validates reactions using RDKit and semi-empirical QM (xTB)
- **Reflection Agent**: Produces causal reasoning traces for failed reactions
- **Compilation Agent**: Builds RLHF/DPO-formatted preference pairs

## Architecture

```
Hypothesis Agent → Verification Agent ──PASS→ Compilation Agent → Dataset
                         │                                      │
                         └──FAIL→ Reflection Agent ─────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[chemistry,dev]"

# Configure environment
cp .env.example .env
# Edit .env with your LLM API keys

# Generate a dataset
auto-chem pipeline --num-hypotheses 1000 --batch-size 50
```

## Requirements

- Python 3.11+
- RDKit (for molecular validation)
- xTB (optional, for quantum mechanical verification)
- LLM API access (OpenAI, OpenRouter, or compatible)

## Project Structure

```
auto-cheminstruct/
├── src/
│   ├── agents/       # Hypothesis, Verification, Reflection, Compilation agents
│   ├── chemistry/    # RDKit wrappers, xTB interface, molecular utilities
│   ├── pipeline/     # Orchestrator, state management, RAG module
│   ├── data/         # Data models, schemas, dataset compilation
│   ├── cli/          # Typer CLI interface
│   └── utils/        # Logging, config, helpers
├── configs/          # YAML configuration files
├── tests/            # Test suite
└── docs/             # Documentation
```

## Publication

This research targets the NeurIPS Datasets & Benchmarks track, contributing:
1. A novel methodology for physics-verified synthetic data generation
2. An open-source dataset of chemistry instruction pairs with causal reasoning traces
3. A reusable agentic pipeline for domain-specific instruction data generation
