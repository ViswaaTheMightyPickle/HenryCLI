"""
Model Manager - Algorithmic model loading and unloading.

Provides automated model lifecycle management:
- Load T1 router for task analysis
- Unload all models before specialist execution
- Load specialist model with optimized context
- Reload T1 for result verification
"""

import asyncio
from typing import Any

from .lmstudio import LMStudioClient
from .models.config import ModelConfig
from .models.pool import ModelPool


class ModelManager:
    """Manages model loading/unloading lifecycle for task execution."""

    def __init__(self, client: LMStudioClient, config: ModelConfig):
        self.client = client
        self.config = config
        self.pool = ModelPool(client, config)

    async def initialize(self) -> None:
        """Initialize by refreshing model status."""
        await self.pool.refresh_model_status()

    async def load_router(self) -> str | None:
        """
        Load T1 router model for task analysis.

        Returns:
            Router model ID or None if failed
        """
        router_model = self.config.get_default_model("T1")
        if not router_model:
            return None

        # Check if already loaded
        if self.pool.current_model == router_model:
            return router_model

        # Unload all first to free VRAM
        await self.unload_all()

        # Load router with optimized context
        context_length = self.config.get_context_length_for_model(router_model)
        try:
            result = await self.client.load_model(
                model_key=router_model,
                gpu_layers="auto",
                context_length=context_length,
            )
            if result.get("instance_id"):
                self.pool.current_model = router_model
                self.pool.models[router_model].is_loaded = True
                return router_model
        except Exception:
            pass

        # Fallback without context length
        try:
            result = await self.client.load_model(
                model_key=router_model,
                gpu_layers="auto",
            )
            if result.get("instance_id"):
                self.pool.current_model = router_model
                self.pool.models[router_model].is_loaded = True
                return router_model
        except Exception:
            pass

        return None

    async def load_specialist(self, model_id: str) -> bool:
        """
        Load specialist model for task execution.

        Args:
            model_id: Specialist model to load

        Returns:
            True if successful
        """
        # Unload all first (including router)
        await self.unload_all()

        # Get optimized context length
        context_length = self.config.get_context_length_for_model(model_id)

        # Try with context length first
        try:
            result = await self.client.load_model(
                model_key=model_id,
                gpu_layers="auto",
                context_length=context_length,
            )
            if result.get("instance_id"):
                self.pool.current_model = model_id
                if model_id in self.pool.models:
                    self.pool.models[model_id].is_loaded = True
                return True
        except Exception:
            pass

        # Fallback without context length
        try:
            result = await self.client.load_model(
                model_key=model_id,
                gpu_layers="auto",
            )
            if result.get("instance_id"):
                self.pool.current_model = model_id
                if model_id in self.pool.models:
                    self.pool.models[model_id].is_loaded = True
                return True
        except Exception:
            pass

        return False

    async def reload_router(self) -> str | None:
        """
        Reload T1 router for result verification.

        Returns:
            Router model ID or None if failed
        """
        return await self.load_router()

    async def unload_all(self) -> list[str]:
        """
        Unload all non-resident models.

        Returns:
            List of unloaded model IDs
        """
        return await self.pool.auto_unload_all()

    async def get_current_model(self) -> str | None:
        """Get currently loaded model ID."""
        return self.pool.current_model

    async def get_available_models(self) -> dict[str, list[str]]:
        """
        Get available models by tier from local system.

        Returns:
            Dict mapping tier IDs to list of model IDs
        """
        return await self.pool.discover_and_classify_models()
