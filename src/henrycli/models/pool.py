"""Model pool manager with switching logic."""

from dataclasses import dataclass, field
from typing import Any

from ..lmstudio import LMStudioClient
from ..model_switcher import ModelSwitcher, ModelSwitchStatus
from .config import ModelConfig, TierConfig


@dataclass
class ModelInfo:
    """Information about a model in the pool."""

    model_id: str
    tier: str
    is_loaded: bool = False
    is_resident: bool = False
    vram_gb: float = 0.0
    last_used: float = 0.0
    load_count: int = 0
    instance_id: str | None = None  # LM Studio instance ID


class ModelPool:
    """
    Manages a pool of models with intelligent switching.
    
    Handles:
    - Model loading/unloading via LM Studio API
    - VRAM management
    - Hot caching for frequently-used models
    - Fallback to smaller models on failure
    - Auto-tier classification for unknown models
    """

    def __init__(
        self,
        client: LMStudioClient,
        config: ModelConfig | None = None,
    ):
        """
        Initialize model pool.

        Args:
            client: LM Studio client
            config: Model configuration (uses default if None)
        """
        self.client = client
        self.config = config or ModelConfig()
        self.switcher = ModelSwitcher(
            client,
            poll_interval=self.config.performance.model_switch_poll_interval_ms
            / 1000.0,
            timeout=self.config.performance.model_switch_timeout_sec,
            extended_timeout=self.config.performance.model_switch_timeout_sec * 2,
        )

        self.models: dict[str, ModelInfo] = {}
        self.current_model: str | None = None
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize model pool from configuration."""
        for tier_id, tier in self.config.tiers.items():
            for model_id in tier.models:
                self.models[model_id] = ModelInfo(
                    model_id=model_id,
                    tier=tier_id,
                    is_resident=tier.resident,
                    vram_gb=tier.vram_gb,
                )

    async def refresh_model_status(self) -> None:
        """Refresh loaded status for all models."""
        try:
            model_list = await self.client.get_models()
            loaded_ids = model_list.model_ids()

            for model_id, info in self.models.items():
                info.is_loaded = model_id in loaded_ids

            # Update current model
            for model_id, info in self.models.items():
                if info.is_loaded and not info.is_resident:
                    # Non-resident loaded model is likely current
                    if self.current_model is None:
                        self.current_model = model_id
        except Exception:
            # LM Studio not available, mark all as unloaded
            for info in self.models.values():
                if not info.is_resident:
                    info.is_loaded = False

    def get_model_for_tier(self, tier_id: str) -> str | None:
        """
        Get the best available model for a tier.

        Args:
            tier_id: Tier ID (T1-T4)

        Returns:
            Model ID or None if tier not found
        """
        tier = self.config.get_tier(tier_id)
        if not tier:
            return None

        # Prefer already loaded models in tier
        for model_id in tier.models:
            if model_id in self.models and self.models[model_id].is_loaded:
                return model_id

        # Return default
        return tier.default

    def get_tier_for_task(self, task_type: str, complexity: str) -> str:
        """
        Get recommended tier for a task.

        Args:
            task_type: Type of task (code, research, writing, reasoning)
            complexity: Complexity (simple, moderate, complex)

        Returns:
            Tier ID
        """
        # Base tier by task type
        type_to_tier = {
            "code": "T3",
            "research": "T2",
            "writing": "T2",
            "reasoning": "T4",
        }

        base_tier = type_to_tier.get(task_type, "T2")
        tier_order = ["T1", "T2", "T3", "T4"]

        if base_tier not in tier_order:
            return "T2"

        tier_index = tier_order.index(base_tier)

        # Adjust by complexity
        if complexity == "simple":
            tier_index = max(0, tier_index - 1)
        elif complexity == "complex":
            tier_index = min(3, tier_index + 1)

        return tier_order[tier_index]

    async def switch_to_model(
        self,
        model_id: str,
        force: bool = False,
        context_length: int | None = None,
    ) -> bool:
        """
        Switch to a specific model.

        Args:
            model_id: Model to switch to
            force: Force switch even if already loaded
            context_length: Optional context length (auto-detected if None)

        Returns:
            True if successful
        """
        # Check if already current
        if self.current_model == model_id and not force:
            return True

        # Check if model exists in pool
        if model_id not in self.models:
            # Try to find tier for this model
            tier = self.config.get_tier_for_model(model_id)
            if tier:
                self.models[model_id] = ModelInfo(
                    model_id=model_id,
                    tier=tier,
                    vram_gb=self.config.get_model_vram(model_id),
                )
            else:
                return False

        # Determine if extended timeout needed
        use_extended = self.models[model_id].vram_gb > self.config.hardware.vram_gb

        # Get context length if not specified
        if context_length is None:
            context_length = self.config.get_context_length_for_model(model_id)

        # Wait for model to be loaded (user or external process)
        result = await self.switcher.switch_model(
            model_id,
            use_extended_timeout=use_extended,
            context_length=context_length,
        )

        if result.status in (
            ModelSwitchStatus.SUCCESS,
            ModelSwitchStatus.ALREADY_LOADED,
        ):
            self.current_model = model_id
            self.models[model_id].is_loaded = True
            self.models[model_id].load_count += 1
            return True

        return False

    def get_fallback_model(self, current_tier: str) -> str | None:
        """
        Get fallback model (one tier lower).

        Args:
            current_tier: Current tier ID

        Returns:
            Fallback model ID or None
        """
        tier_order = ["T1", "T2", "T3", "T4"]
        if current_tier not in tier_order:
            return None

        current_index = tier_order.index(current_tier)
        if current_index == 0:
            # Already at lowest tier
            return self.config.get_default_model("T1")

        fallback_tier = tier_order[current_index - 1]
        return self.get_model_for_tier(fallback_tier)

    async def switch_with_fallback(
        self,
        model_id: str,
        max_fallbacks: int = 2,
    ) -> tuple[bool, str]:
        """
        Switch to model with automatic fallback on failure.

        Args:
            model_id: Target model
            max_fallbacks: Maximum number of fallback attempts

        Returns:
            Tuple of (success, actual_model_used)
        """
        current_attempt = model_id
        fallbacks_used = 0

        while fallbacks_used <= max_fallbacks:
            # Get tier for current attempt
            tier = self.models.get(current_attempt, ModelInfo("", "")).tier
            if not tier:
                tier = self.config.get_tier_for_model(current_attempt) or "T2"

            if await self.switch_to_model(current_attempt):
                return True, current_attempt

            # Fallback to smaller model
            next_model = self.get_fallback_model(tier)
            if next_model is None:
                break

            current_attempt = next_model
            fallbacks_used += 1

        return False, model_id

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded models."""
        return [
            info.model_id for info in self.models.values() if info.is_loaded
        ]

    def get_vram_usage(self) -> dict[str, float]:
        """
        Estimate current VRAM usage.

        Returns:
            Dict with used, available, and total VRAM
        """
        used = sum(
            info.vram_gb for info in self.models.values() if info.is_loaded
        )
        total = self.config.hardware.vram_gb

        return {
            "used_gb": used,
            "available_gb": total - used,
            "total_gb": total,
        }

    def can_load_model(self, model_id: str) -> bool:
        """
        Check if a model can be loaded with current VRAM.

        Args:
            model_id: Model to check

        Returns:
            True if model can be loaded
        """
        if model_id not in self.models:
            return False

        vram_usage = self.get_vram_usage()
        model_vram = self.models[model_id].vram_gb

        # Resident models always have priority
        if self.models[model_id].is_resident:
            return True

        return model_vram <= vram_usage["available_gb"]

    async def unload_non_resident(self) -> list[str]:
        """
        Prompt to unload non-resident models.

        Note: LM Studio doesn't have direct unload API.
        This returns a list of models that should be unloaded.

        Returns:
            List of model IDs to unload
        """
        to_unload = []
        for info in self.models.values():
            if info.is_loaded and not info.is_resident:
                to_unload.append(info.model_id)
        return to_unload

    def get_model_stats(self) -> list[dict[str, Any]]:
        """
        Get statistics for all models.

        Returns:
            List of model stats
        """
        stats = []
        for info in self.models.values():
            stats.append(
                {
                    "model_id": info.model_id,
                    "tier": info.tier,
                    "is_loaded": info.is_loaded,
                    "is_resident": info.is_resident,
                    "vram_gb": info.vram_gb,
                    "load_count": info.load_count,
                }
            )
        return stats

    async def auto_load_model(
        self,
        model_key: str,
        gpu_layers: str = "auto",
        unload_current: bool = True,
    ) -> tuple[bool, str]:
        """
        Automatically load a model via LM Studio API.

        Args:
            model_key: Model identifier
            gpu_layers: GPU layers setting
            unload_current: Whether to unload current model first (saves VRAM)

        Returns:
            Tuple of (success, message)
        """
        try:
            # Unload current model if requested and not resident
            if unload_current and self.current_model:
                current_info = self.models.get(self.current_model)
                if current_info and not current_info.is_resident:
                    try:
                        await self.client.unload_model(self.current_model)
                        self.models[self.current_model].is_loaded = False
                    except Exception:
                        pass  # Model might already be unloaded
                self.current_model = None

            # Get appropriate context length for model
            context_length = self.config.get_context_length_for_model(model_key)

            # Load new model with optimized context length
            result = await self.client.load_model(
                model_key=model_key,
                gpu_layers=gpu_layers,
                context_length=context_length,
            )

            instance_id = result.get("instance_id", "")
            self.current_model = model_key

            # Update model info
            if model_key in self.models:
                self.models[model_key].is_loaded = True
                self.models[model_key].instance_id = instance_id
                self.models[model_key].load_count += 1
            else:
                # Add new model to pool
                self.models[model_key] = ModelInfo(
                    model_id=model_key,
                    tier="T2",  # Default tier
                    is_loaded=True,
                    instance_id=instance_id,
                    load_count=1,
                )

            return True, f"Loaded {model_key}"

        except Exception as e:
            return False, f"Failed to load {model_key}: {e}"

    async def auto_unload_all(self) -> list[str]:
        """
        Unload all non-resident models.

        Returns:
            List of unloaded model IDs
        """
        unloaded = []
        try:
            models = await self.client.get_models()
            for model in models.data:
                # Skip resident models
                if model.id in self.models and self.models[model.id].is_resident:
                    continue

                try:
                    await self.client.unload_model(model.id)
                    unloaded.append(model.id)
                    if model.id in self.models:
                        self.models[model.id].is_loaded = False
                except Exception:
                    pass

            self.current_model = None
        except Exception:
            pass

        return unloaded

    async def discover_and_classify_models(self) -> dict[str, Any]:
        """
        Discover local models and auto-classify them into tiers.

        Returns:
            Tier configuration from auto-classification
        """
        from ..auto_tier import AutoTierClassifier

        classifier = AutoTierClassifier(
            hardware_vram_gb=self.config.hardware.vram_gb
        )

        try:
            local_models = await self.client.list_local_models()
            config = classifier.generate_tier_config(local_models)
            return config
        except Exception:
            return {"T1": [], "T2": [], "T3": [], "T4": []}
