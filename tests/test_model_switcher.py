"""Tests for model switcher."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from henrycli.model_switcher import (
    ModelSwitchResult,
    ModelSwitchStatus,
    ModelSwitcher,
)


class TestModelSwitchResult:
    """Tests for ModelSwitchResult."""

    def test_success_result(self):
        """Test successful switch result."""
        result = ModelSwitchResult(
            status=ModelSwitchStatus.SUCCESS,
            model_id="test-model",
            elapsed_seconds=5.0,
            message="Loaded",
        )
        assert result.status == ModelSwitchStatus.SUCCESS
        assert result.model_id == "test-model"
        assert result.elapsed_seconds == 5.0


class TestModelSwitcher:
    """Tests for ModelSwitcher."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LM Studio client."""
        client = MagicMock()
        client.is_model_loaded = AsyncMock(return_value=False)
        return client

    @pytest.fixture
    def switcher(self, mock_client):
        """Create ModelSwitcher with mock client."""
        return ModelSwitcher(
            mock_client,
            poll_interval=0.1,
            timeout=0.5,
        )

    @pytest.mark.asyncio
    async def test_switch_already_loaded(self, mock_client, switcher):
        """Test switch returns immediately if model already loaded."""
        mock_client.is_model_loaded.return_value = True

        result = await switcher.switch_model("test-model")

        assert result.status == ModelSwitchStatus.ALREADY_LOADED
        assert result.model_id == "test-model"
        assert result.elapsed_seconds == 0.0

    @pytest.mark.asyncio
    async def test_switch_timeout(self, mock_client, switcher):
        """Test switch times out when model never loads."""
        mock_client.is_model_loaded.return_value = False

        result = await switcher.switch_model("test-model")

        assert result.status == ModelSwitchStatus.TIMEOUT
        assert result.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_switch_success(self, mock_client):
        """Test switch succeeds when model loads."""
        # Model not loaded first check, then loaded on second check
        call_count = 0

        async def check_model(model_id):
            nonlocal call_count
            call_count += 1
            return call_count > 1

        mock_client.is_model_loaded.side_effect = check_model

        switcher = ModelSwitcher(mock_client, poll_interval=0.1, timeout=2.0)
        result = await switcher.switch_model("test-model")

        assert result.status == ModelSwitchStatus.SUCCESS
        assert result.model_id == "test-model"
        assert result.elapsed_seconds > 0

    @pytest.mark.asyncio
    async def test_wait_for_unload_success(self, mock_client):
        """Test waiting for model unload."""
        # Model loaded first, then unloaded
        call_count = 0

        async def check_model(model_id):
            nonlocal call_count
            call_count += 1
            return call_count == 1

        mock_client.is_model_loaded.side_effect = check_model

        switcher = ModelSwitcher(mock_client, poll_interval=0.1, timeout=2.0)
        result = await switcher.wait_for_model_unload("test-model")

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_unload_timeout(self, mock_client):
        """Test waiting for unload times out."""
        mock_client.is_model_loaded.return_value = True

        switcher = ModelSwitcher(mock_client, poll_interval=0.1, timeout=0.3)
        result = await switcher.wait_for_model_unload("test-model")

        assert result is False

    @pytest.mark.asyncio
    async def test_extended_timeout(self, mock_client):
        """Test extended timeout for large models."""
        mock_client.is_model_loaded.return_value = False

        switcher = ModelSwitcher(
            mock_client,
            poll_interval=0.1,
            timeout=0.2,
            extended_timeout=1.0,
        )

        # Use extended timeout
        result = await switcher.switch_model("large-model", use_extended_timeout=True)

        # Should take longer than normal timeout
        assert result.status == ModelSwitchStatus.TIMEOUT
        assert result.elapsed_seconds > 0.2
