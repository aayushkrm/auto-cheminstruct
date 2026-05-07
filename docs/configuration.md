# Auto-ChemInstruct Configuration Reference

## Configuration File

Default: `configs/default.yaml`

## Pipeline Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pipeline.batch_size` | int | 1 | Hypotheses per generation batch |
| `pipeline.seed` | int | 42 | Random seed for reproducibility |
| `pipeline.checkpoint_dir` | str | `.checkpoints` | SQLite checkpoint directory |
| `pipeline.temperature_schedule` | str | `cosine` | Schedule type: cosine/linear/exponential |
| `pipeline.temperature_max` | float | 1.0 | Starting temperature (exploration) |
| `pipeline.temperature_min` | float | 0.3 | Ending temperature (exploitation) |
| `pipeline.bootstrap_iterations` | int | 1 | Self-bootstrapping iterations (1=off) |

## LLM Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.provider` | str | `fireworks` | LLM provider |
| `llm.model` | str | `accounts/fireworks/models/deepseek-v4-pro` | Model identifier |
| `llm.base_url` | str | `https://api.fireworks.ai/inference/v1` | API base URL |
| `llm.temperature` | float | 0.8 | Default sampling temperature |
| `llm.max_tokens` | int | 4096 | Maximum output tokens |
| `llm.request_timeout` | int | 180 | API timeout (seconds) |
| `llm.top_p` | float | 0.95 | Nucleus sampling parameter |

## Agent Settings

### Hypothesis Agent
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hypothesis_agent.temperature` | float | 0.8 | Generation temperature |
| `hypothesis_agent.top_p` | float | 0.95 | Nucleus sampling |
| `hypothesis_agent.max_tokens` | int | 4096 | Max output tokens |
| `hypothesis_agent.num_generations_per_prompt` | int | 1 | Hypotheses per call |

### Verification Agent
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `verification_agent.enable_xtb` | bool | false | Enable xTB energetic validation |
| `verification_agent.energy_barrier_threshold` | float | 50.0 | Max acceptable energy barrier (kcal/mol) |
| `verification_agent.sa_score_min` | float | 1.0 | Minimum SA score |
| `verification_agent.sa_score_max` | float | 10.0 | Maximum SA score |
| `verification_agent.qed_min` | float | 0.0 | Minimum QED score |
| `verification_agent.max_atoms` | int | 100 | Maximum atoms per molecule |

### Reflection Agent
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `reflection_agent.temperature` | float | 0.3 | Low temp for factual analysis |
| `reflection_agent.max_tokens` | int | 2048 | Max output tokens |

### Compilation Agent
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `compilation_agent.output_format` | str | `jsonl` | Output format |
| `compilation_agent.train_split` | float | 0.8 | Train fraction |
| `compilation_agent.val_split` | float | 0.1 | Validation fraction |
| `compilation_agent.test_split` | float | 0.1 | Test fraction |
| `compilation_agent.deduplicate` | bool | true | Remove duplicate pairs |

## RAG Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rag.enabled` | bool | true | Enable RAG system |
| `rag.embedding_model` | str | `text-embedding-3-small` | Embedding model |
| `rag.chroma_persist_dir` | str | `.chromadb` | ChromaDB persistence |
| `rag.retrieval_k` | int | 5 | Documents per query |
| `rag.use_reranker` | bool | false | Enable reranker (future) |

## CLI Commands

```bash
# Run pipeline with self-bootstrapping
python -m src.cli.main pipeline -n 100 -B 3

# Run ablation study
python -m src.cli.main ablation -n 20 -o benchmarks

# View current config
python -m src.cli.main config

# Generate 1 hypothesis (quick test)
python -m src.cli.main pipeline -n 1
```
