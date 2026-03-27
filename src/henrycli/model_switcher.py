"""Model polling mechanism for LM Studio model switching."""

import asyncio
from dataclasses import dataclass
from enum import Enum


class ModelSwitchStatus(Enum):
    """Status of a model switch operation."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ALREADY_LOADED = "already_loaded"
    FAILED = "failed"


@dataclass
class ModelSwitchResult:
    """Result of a model switch operation."""

    status: ModelSwitchStatus
    model_id: str
    elapsed_seconds: float
    message: str = ""


class ModelSwitcher:
    """Handles model switching with polling."""

    def __init__(
        self,
        lmstudio_client,
        poll_interval: float = 0.5,
        timeout: float = 60.0,
        extended_timeout: float = 120.0,
    ):
        """
        Initialize model switcher.

        Args:
            lmstudio_client: LMStudioClient instance
            poll_interval: Seconds between polls
            timeout: Normal timeout in seconds
            extended_timeout: Extended timeout for large models
        """
        self.client = lmstudio_client
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.extended_timeout = extended_timeout

    async def switch_model(
        self,
        target_model: str,
        use_extended_timeout: bool = False,
    ) -> ModelSwitchResult:
        """
        Wait for target model to be loaded.

        Note: LM Studio doesn't have a direct model load API.
        This method polls for the model to appear, assuming the user
        or external process loads it.

        Args:
            target_model: Model ID to wait for
            use_extended_timeout: Use extended timeout for large models

        Returns:
            ModelSwitchResult with status and timing
        """
        import time

        start_time = time.monotonic()
        timeout = self.extended_timeout if use_extended_timeout else self.timeout

        # Check if already loaded
        if await self.client.is_model_loaded(target_model):
            return ModelSwitchResult(
                status=ModelSwitchStatus.ALREADY_LOADED,
                model_id=target_model,
                elapsed_seconds=0.0,
                message="Model already loaded",
            )

        # Poll for model to be loaded
        max_polls = int(timeout / self.poll_interval)
        for i in range(max_polls):
            await asyncio.sleep(self.poll_interval)

            try:
                if await self.client.is_model_loaded(target_model):
                    elapsed = time.monotonic() - start_time
                    return ModelSwitchResult(
                        status=ModelSwitchStatus.SUCCESS,
                        model_id=target_model,
                        elapsed_seconds=elapsed,
                        message=f"Model loaded in {elapsed:.1f}s",
                    )
            except Exception as e:
                # Continue polling on transient errors
                if i == max_polls - 1:
                    elapsed = time.monotonic() - start_time
                    return ModelSwitchResult(
                        status=ModelSwitchStatus.FAILED,
                        model_id=target_model,
                        elapsed_seconds=elapsed,
                        message=f"Error checking model status: {e}",
                    )

        elapsed = time.monotonic() - start_time
        return ModelSwitchResult(
            status=ModelSwitchStatus.TIMEOUT,
            model_id=target_model,
            elapsed_seconds=elapsed,
            message=f"Timeout waiting for model after {elapsed:.1f}s",
        )

    async def wait_for_model_unload(
        self,
        model_id: str,
        timeout: float | None = None,
    ) -> bool:
        """
        Wait for a model to be unloaded.

        Args:
            model_id: Model ID to wait for unloading
            timeout: Timeout in seconds (uses default if None)

        Returns:
            True if unloaded, False if timeout
        """
        if timeout is None:
            timeout = self.timeout

        max_polls = int(timeout / self.poll_interval)
        for _ in range(max_polls):
            if not await self.client.is_model_loaded(model_id):
                return True
            await asyncio.sleep(self.poll_interval)

        return False
