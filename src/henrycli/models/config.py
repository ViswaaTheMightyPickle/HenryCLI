"""Model configuration and tier management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TierConfig:
    """Configuration for a model tier."""

    models: list[str]
    default: str
    purpose: str
    vram_gb: float = 0.0
    resident: bool = False
    cpu_offload: bool = False


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    vram_gb: int = 8
    ram_gb: int = 32
    gpu: str = "RTX 4060 Laptop"
    gpu_layers_max: int = 99
    cpu_offload_threshold: float = 0.9
    # Context length settings by model size
    context_length_small: int = 4096  # For models < 7B
    context_length_medium: int = 8192  # For models 7B-20B
    context_length_large: int = 16384  # For models > 20B


@dataclass
class ContextConfig:
    """Context management configuration."""

    compression_threshold: float = 0.85
    offload_token_limit: int = 20000
    keep_recent_messages: int = 10


@dataclass
class PerformanceConfig:
    """Performance configuration."""

    model_switch_timeout_sec: int = 60
    model_switch_poll_interval_ms: int = 500
    inference_timeout_sec: int = 300


class ModelConfig:
    """
    Model configuration manager.
    
    Loads and manages configuration for model tiers and hardware.
    """

    DEFAULT_CONFIG = {
        "tiers": {
            "T1": {
                "models": [
                    "phi-3-mini-4k-instruct-q4_k_m",
                    "qwen2.5-3b-instruct-q4_k_m",
                    "llama-3.2-3b-instruct-q4_k_m",
                ],
                "default": "phi-3-mini-4k-instruct-q4_k_m",
                "purpose": "routing",
                "vram_gb": 2.5,
                "resident": True,
            },
            "T2": {
                "models": [
                    "qwen2.5-7b-instruct-q4_k_m",
                    "llama-3.1-8b-instruct-q4_k_m",
                    "mistral-7b-instruct-v0.3-q4_k_m",
                ],
                "default": "qwen2.5-7b-instruct-q4_k_m",
                "purpose": "general",
                "vram_gb": 4.5,
                "resident": False,
            },
            "T3": {
                "models": [
                    "deepseek-coder-6.7b-instruct-q4_k_m",
                    "codellama-13b-instruct-q4_k_m",
                    "starcoder2-7b-q4_k_m",
                ],
                "default": "deepseek-coder-6.7b-instruct-q4_k_m",
                "purpose": "code",
                "vram_gb": 8.0,
                "resident": False,
            },
            "T4": {
                "models": [
                    "qwen2.5-32b-instruct-q4_k_m",
                    "command-r-q4_k_m",
                    "yi-34b-chat-q4_k_m",
                ],
                "default": "qwen2.5-32b-instruct-q4_k_m",
                "purpose": "reasoning",
                "vram_gb": 16.0,
                "resident": False,
                "cpu_offload": True,
            },
        },
        "hardware": {
            "vram_gb": 8,
            "ram_gb": 32,
            "gpu": "RTX 4060 Laptop",
            "gpu_layers_max": 99,
            "cpu_offload_threshold": 0.9,
        },
        "context": {
            "compression_threshold": 0.85,
            "offload_token_limit": 20000,
            "keep_recent_messages": 10,
        },
        "performance": {
            "model_switch_timeout_sec": 60,
            "model_switch_poll_interval_ms": 500,
            "inference_timeout_sec": 300,
        },
    }

    def __init__(self, config_path: Path | None = None):
        """
        Initialize model configuration.

        Args:
            config_path: Path to config file (default: ~/.henrycli/models/config.yaml)
        """
        if config_path is None:
            config_path = Path.home() / ".henrycli" / "models" / "config.yaml"

        self.config_path = config_path
        self.tiers: dict[str, TierConfig] = {}
        self.hardware = HardwareConfig()
        self.context = ContextConfig()
        self.performance = PerformanceConfig()

        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        import copy
        config_data = copy.deepcopy(self.DEFAULT_CONFIG)

        # Try to load user config
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        self._merge_config(config_data, user_config)
            except Exception:
                # Use defaults on error
                pass

        # Parse tiers
        for tier_id, tier_data in config_data.get("tiers", {}).items():
            self.tiers[tier_id] = TierConfig(
                models=tier_data.get("models", []),
                default=tier_data.get("default", ""),
                purpose=tier_data.get("purpose", ""),
                vram_gb=tier_data.get("vram_gb", 0.0),
                resident=tier_data.get("resident", False),
                cpu_offload=tier_data.get("cpu_offload", False),
            )

        # Parse hardware
        hw = config_data.get("hardware", {})
        self.hardware = HardwareConfig(
            vram_gb=hw.get("vram_gb", 8),
            ram_gb=hw.get("ram_gb", 32),
            gpu=hw.get("gpu", "RTX 4060 Laptop"),
            gpu_layers_max=hw.get("gpu_layers_max", 99),
            cpu_offload_threshold=hw.get("cpu_offload_threshold", 0.9),
            context_length_small=hw.get("context_length_small", 4096),
            context_length_medium=hw.get("context_length_medium", 8192),
            context_length_large=hw.get("context_length_large", 16384),
        )

        # Parse context
        ctx = config_data.get("context", {})
        self.context = ContextConfig(
            compression_threshold=ctx.get("compression_threshold", 0.85),
            offload_token_limit=ctx.get("offload_token_limit", 20000),
            keep_recent_messages=ctx.get("keep_recent_messages", 10),
        )

        # Parse performance
        perf = config_data.get("performance", {})
        self.performance = PerformanceConfig(
            model_switch_timeout_sec=perf.get("model_switch_timeout_sec", 60),
            model_switch_poll_interval_ms=perf.get(
                "model_switch_poll_interval_ms", 500
            ),
            inference_timeout_sec=perf.get("inference_timeout_sec", 300),
        )

    def _merge_config(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> None:
        """Recursively merge override config into base."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get_tier(self, tier_id: str) -> TierConfig | None:
        """Get configuration for a tier."""
        return self.tiers.get(tier_id)

    def get_default_model(self, tier_id: str) -> str | None:
        """Get default model for a tier."""
        tier = self.tiers.get(tier_id)
        return tier.default if tier else None

    def get_all_models(self) -> list[str]:
        """Get all configured models."""
        models = []
        for tier in self.tiers.values():
            models.extend(tier.models)
        return models

    def get_tier_for_model(self, model_id: str) -> str | None:
        """Get tier ID for a model."""
        for tier_id, tier in self.tiers.items():
            if model_id in tier.models:
                return tier_id
        return None

    def get_resident_models(self) -> list[str]:
        """Get list of resident models that should always be loaded."""
        return [
            tier.default for tier in self.tiers.values() if tier.resident
        ]

    def get_model_vram(self, model_id: str) -> float:
        """Get VRAM requirement for a model."""
        for tier in self.tiers.values():
            if model_id in tier.models:
                return tier.vram_gb
        return 0.0

    def get_context_length_for_model(self, model_id: str) -> int:
        """
        Get appropriate context length for a model based on its size.

        Args:
            model_id: Model identifier

        Returns:
            Context length in tokens
        """
        model_lower = model_id.lower()

        # Small models (< 7B)
        if any(p in model_lower for p in ["1b", "2b", "3b", "4b", "0.5b", "phi-3-mini", "phi-2"]):
            return self.hardware.context_length_small

        # Large models (> 20B)
        if any(p in model_lower for p in ["30b", "32b", "34b", "40b", "65b", "70b", "20b"]):
            return self.hardware.context_length_large

        # Medium models (7B - 20B) - default
        return self.hardware.context_length_medium

    def needs_cpu_offload(self, model_id: str) -> bool:
        """Check if a model needs CPU offloading."""
        for tier in self.tiers.values():
            if model_id in tier.models:
                return tier.cpu_offload or tier.vram_gb > self.hardware.vram_gb
        return False

    def save_config(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "tiers": {
                tier_id: {
                    "models": tier.models,
                    "default": tier.default,
                    "purpose": tier.purpose,
                    "vram_gb": tier.vram_gb,
                    "resident": tier.resident,
                    "cpu_offload": tier.cpu_offload,
                }
                for tier_id, tier in self.tiers.items()
            },
            "hardware": {
                "vram_gb": self.hardware.vram_gb,
                "ram_gb": self.hardware.ram_gb,
                "gpu": self.hardware.gpu,
                "gpu_layers_max": self.hardware.gpu_layers_max,
                "cpu_offload_threshold": self.hardware.cpu_offload_threshold,
            },
            "context": {
                "compression_threshold": self.context.compression_threshold,
                "offload_token_limit": self.context.offload_token_limit,
                "keep_recent_messages": self.context.keep_recent_messages,
            },
            "performance": {
                "model_switch_timeout_sec": self.performance.model_switch_timeout_sec,
                "model_switch_poll_interval_ms": self.performance.model_switch_poll_interval_ms,
                "inference_timeout_sec": self.performance.inference_timeout_sec,
            },
        }

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)
