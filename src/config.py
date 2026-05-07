"""Centralized configuration management using OmegaConf and Pydantic."""

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field, field_validator

from src.exceptions import ConfigurationError

CONFIG_DIR = Path(__file__).parent.parent / "configs"


class LLMConfig(BaseModel):
    """Single-provider LLM configuration via Fireworks AI (OpenAI-compatible)."""

    provider: str = "fireworks"
    model: str = "accounts/fireworks/models/deepseek-v4-pro"
    base_url: str = "https://api.fireworks.ai/inference/v1"
    api_key: Optional[str] = None
    temperature: float = 0.8
    max_tokens: int = 4096
    request_timeout: int = 180
    top_p: float = 0.95


class HypothesisAgentConfig(BaseModel):
    name: str = "HypothesisGenerator"
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 2048
    num_generations_per_prompt: int = 5
    prompt_template_version: str = "v1"
    max_smiles_length: int = 500


class VerificationAgentConfig(BaseModel):
    name: str = "PhysicalVerifier"
    enable_xtb: bool = True
    xtb_method: str = "GFN2-xTB"
    xtb_timeout: int = 300
    xtb_max_atoms: int = 100
    energy_barrier_threshold: float = 40.0
    sa_score_min: float = 1.0
    sa_score_max: float = 8.0
    qed_min: float = 0.0


class ReflectionAgentConfig(BaseModel):
    name: str = "CausalReflector"
    temperature: float = 0.3
    max_tokens: int = 1024
    prompt_template_version: str = "v1"


class CompilationAgentConfig(BaseModel):
    name: str = "DatasetCompiler"
    output_format: str = "dpo"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_pairs_per_reaction_type: int = 10
    deduplicate: bool = True


class RAGConfig(BaseModel):
    enabled: bool = True
    embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = ".chromadb"
    retrieval_k: int = 5
    use_reranker: bool = False


class DatabaseConfig(BaseModel):
    sqlite_path: str = ".state/autochem.db"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/autochem.log"
    rotation: str = "100 MB"
    retention: str = "30 days"


class PipelineConfig(BaseModel):
    batch_size: int = 50
    max_iterations: int = 10000
    max_retries_per_hypothesis: int = 3
    checkpoint_dir: str = ".checkpoints"
    checkpoint_interval: int = 10
    seed: int = 42
    temperature_schedule: str = "cosine"
    temperature_max: float = 1.0
    temperature_min: float = 0.3
    bootstrap_iterations: int = 1

    @field_validator("batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v


class AutoChemConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    hypothesis_agent: HypothesisAgentConfig = Field(default_factory=HypothesisAgentConfig)
    verification_agent: VerificationAgentConfig = Field(default_factory=VerificationAgentConfig)
    reflection_agent: ReflectionAgentConfig = Field(default_factory=ReflectionAgentConfig)
    compilation_agent: CompilationAgentConfig = Field(default_factory=CompilationAgentConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str | None = None) -> AutoChemConfig:
    """Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to YAML config file. Defaults to configs/default.yaml.

    Returns:
        Validated AutoChemConfig instance.

    Raises:
        ConfigurationError: If config is invalid or file not found.
    """
    if config_path is None:
        config_path = str(CONFIG_DIR / "default.yaml")

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        available = list(CONFIG_DIR.glob("*.yaml"))
        raise ConfigurationError(
            f"Config file not found: {config_path}. Available: {available}"
        )

    try:
        omega_conf = OmegaConf.load(config_path)
        dict_conf = OmegaConf.to_container(omega_conf, resolve=True)
        return AutoChemConfig.model_validate(dict_conf)
    except Exception as e:
        raise ConfigurationError(f"Failed to load config from {config_path}: {e}") from e
