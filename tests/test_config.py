"""Tests for configuration and pipeline modules."""

import pytest
from pathlib import Path

from src.config import load_config, AutoChemConfig, PipelineConfig
from src.exceptions import ConfigurationError


class TestConfig:
    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, AutoChemConfig)
        assert config.llm.provider == "fireworks"
        assert config.llm.model != ""

    def test_pipeline_config_validation(self):
        config = load_config()
        assert config.pipeline.batch_size > 0
        assert 0 < config.pipeline.seed

    def test_batch_size_positive(self):
        cfg = PipelineConfig(batch_size=10)
        assert cfg.batch_size == 10

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(batch_size=0)

    def test_llm_providers_both_configured(self):
        config = load_config()
        assert config.llm.provider == "fireworks"
        assert config.llm.base_url.startswith("https://api.fireworks.ai")

    def test_provider_config_switch(self):
        config = load_config()
        assert config.llm.model == "accounts/fireworks/models/deepseek-v3p2"

    def test_missing_config_file(self):
        with pytest.raises(ConfigurationError):
            load_config("/nonexistent/config.yaml")
