"""Tests for model configuration."""

import tempfile
from pathlib import Path

import pytest
import yaml

from henrycli.models.config import (
    ContextConfig,
    HardwareConfig,
    ModelConfig,
    PerformanceConfig,
    TierConfig,
)


class TestTierConfig:
    """Tests for TierConfig."""

    def test_create_tier_config(self):
        """Test creating a tier config."""
        tier = TierConfig(
            models=["model-1", "model-2"],
            default="model-1",
            purpose="testing",
            vram_gb=4.5,
            resident=False,
        )
        assert len(tier.models) == 2
        assert tier.default == "model-1"
        assert tier.purpose == "testing"
        assert tier.vram_gb == 4.5
        assert tier.resident is False


class TestModelConfig:
    """Tests for ModelConfig."""

    @pytest.fixture
    def temp_config_path(self):
        """Create a temporary config file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "config.yaml"

    def test_default_config_has_all_tiers(self):
        """Test default config has all tiers."""
        config = ModelConfig()

        assert "T1" in config.tiers
        assert "T2" in config.tiers
        assert "T3" in config.tiers
        assert "T4" in config.tiers

    def test_default_t1_is_resident(self):
        """Test T1 tier is resident by default."""
        config = ModelConfig()
        assert config.tiers["T1"].resident is True

    def test_get_tier(self):
        """Test getting a tier."""
        config = ModelConfig()
        tier = config.get_tier("T2")

        assert tier is not None
        assert tier.purpose == "general"

    def test_get_nonexistent_tier(self):
        """Test getting nonexistent tier."""
        config = ModelConfig()
        tier = config.get_tier("T99")
        assert tier is None

    def test_get_default_model(self):
        """Test getting default model for tier."""
        config = ModelConfig()
        model = config.get_default_model("T1")

        assert model is not None
        assert "phi-3" in model or "qwen2.5-3b" in model or "llama-3.2" in model

    def test_get_all_models(self):
        """Test getting all configured models."""
        config = ModelConfig()
        models = config.get_all_models()

        assert len(models) > 0
        # Should have models from all tiers
        assert any("phi-3" in m or "qwen2.5-3b" in m for m in models)

    def test_get_tier_for_model(self):
        """Test getting tier for a model."""
        config = ModelConfig()
        # Get a known model from T1
        t1_default = config.tiers["T1"].default
        tier = config.get_tier_for_model(t1_default)

        assert tier == "T1"

    def test_get_resident_models(self):
        """Test getting resident models."""
        config = ModelConfig()
        resident = config.get_resident_models()

        # T1 should be resident
        assert len(resident) >= 1
        assert config.tiers["T1"].default in resident

    def test_get_model_vram(self):
        """Test getting VRAM for a model."""
        config = ModelConfig()
        t1_default = config.tiers["T1"].default
        vram = config.get_model_vram(t1_default)

        assert vram > 0
        assert vram <= 3.0  # T1 should be small

    def test_needs_cpu_offload(self):
        """Test checking if model needs CPU offload."""
        config = ModelConfig()

        # T1 should not need offload
        t1_default = config.tiers["T1"].default
        assert config.needs_cpu_offload(t1_default) is False

        # T4 should need offload (16GB > 8GB VRAM)
        t4_default = config.tiers["T4"].default
        assert config.needs_cpu_offload(t4_default) is True

    def test_load_custom_config(self, temp_config_path):
        """Test loading custom configuration."""
        custom_config = {
            "tiers": {
                "T1": {
                    "models": ["custom-model"],
                    "default": "custom-model",
                    "purpose": "routing",
                    "vram_gb": 2.0,
                    "resident": True,
                }
            },
            "hardware": {
                "vram_gb": 16,
                "ram_gb": 64,
            },
        }

        with open(temp_config_path, "w") as f:
            yaml.dump(custom_config, f)

        config = ModelConfig(config_path=temp_config_path)

        assert config.get_default_model("T1") == "custom-model"
        assert config.hardware.vram_gb == 16
        assert config.hardware.ram_gb == 64

    def test_save_config(self, temp_config_path):
        """Test saving configuration."""
        config = ModelConfig(config_path=temp_config_path)
        config.save_config()

        assert temp_config_path.exists()

        # Load and verify
        with open(temp_config_path) as f:
            saved = yaml.safe_load(f)

        assert "tiers" in saved
        assert "hardware" in saved

    def test_hardware_config_defaults(self, temp_config_path):
        """Test hardware config defaults."""
        # Use temp path to avoid pollution from other tests
        config = ModelConfig(config_path=temp_config_path)

        assert config.hardware.vram_gb == 8
        assert config.hardware.ram_gb == 32
        assert "RTX 4060" in config.hardware.gpu

    def test_context_config_defaults(self):
        """Test context config defaults."""
        config = ModelConfig()

        assert config.context.compression_threshold == 0.85
        assert config.context.offload_token_limit == 20000
        assert config.context.keep_recent_messages == 10

    def test_performance_config_defaults(self):
        """Test performance config defaults."""
        config = ModelConfig()

        assert config.performance.model_switch_timeout_sec == 60
        assert config.performance.model_switch_poll_interval_ms == 500
        assert config.performance.inference_timeout_sec == 300
