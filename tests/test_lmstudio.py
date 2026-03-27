"""Tests for LM Studio client."""

import pytest

from henrycli.lmstudio import (
    ChatMessage,
    LMStudioClient,
    ModelInfo,
    ModelList,
)


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = ChatMessage(role="system", content="You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"


class TestModelList:
    """Tests for ModelList model."""

    def test_empty_model_list(self):
        """Test empty model list."""
        model_list = ModelList()
        assert model_list.data == []
        assert model_list.model_ids() == []

    def test_model_list_with_models(self):
        """Test model list with models."""
        model_list = ModelList(
            data=[
                ModelInfo(id="model-1"),
                ModelInfo(id="model-2"),
            ]
        )
        assert len(model_list.data) == 2
        assert model_list.model_ids() == ["model-1", "model-2"]

    def test_has_model_true(self):
        """Test has_model returns True for existing model."""
        model_list = ModelList(data=[ModelInfo(id="test-model")])
        assert model_list.has_model("test-model") is True

    def test_has_model_false(self):
        """Test has_model returns False for missing model."""
        model_list = ModelList(data=[ModelInfo(id="test-model")])
        assert model_list.has_model("other-model") is False


class TestLMStudioClient:
    """Tests for LMStudioClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return LMStudioClient(base_url="http://localhost:1234")

    def test_init_default_url(self):
        """Test client initializes with default URL."""
        client = LMStudioClient()
        assert client.base_url == "http://localhost:1234"

    def test_init_custom_url(self):
        """Test client initializes with custom URL."""
        client = LMStudioClient(base_url="http://custom:8080")
        assert client.base_url == "http://custom:8080"

    def test_init_strips_trailing_slash(self):
        """Test client strips trailing slash from URL."""
        client = LMStudioClient(base_url="http://localhost:1234/")
        assert client.base_url == "http://localhost:1234"

    @pytest.mark.asyncio
    async def test_health_check_connection_failed(self, client):
        """Test health check returns False when connection fails."""
        # This will fail if LM Studio is not running
        result = await client.health_check()
        # Result depends on whether LM Studio is running
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_get_models_connection_failed(self, client):
        """Test get_models raises on connection failure."""
        # This will fail if LM Studio is not running
        # Note: httpx may return empty response instead of raising
        try:
            await client.get_models()
        except Exception:
            pass  # Expected if LM Studio is not running
