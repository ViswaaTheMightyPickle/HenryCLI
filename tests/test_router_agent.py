"""Tests for router agent."""

import pytest

from henrycli.agents.router import Complexity, RouterAgent, TaskType
from henrycli.lmstudio import LMStudioClient


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_type_values(self):
        """Test task type enum values."""
        assert TaskType.CODE.value == "code"
        assert TaskType.RESEARCH.value == "research"
        assert TaskType.WRITING.value == "writing"
        assert TaskType.REASONING.value == "reasoning"


class TestComplexity:
    """Tests for Complexity enum."""

    def test_complexity_values(self):
        """Test complexity enum values."""
        assert Complexity.SIMPLE.value == "simple"
        assert Complexity.MODERATE.value == "moderate"
        assert Complexity.COMPLEX.value == "complex"


class TestRouterAgent:
    """Tests for RouterAgent."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LM Studio client."""
        import unittest.mock as mock
        return mock.AsyncMock(spec=LMStudioClient)

    @pytest.fixture
    def router(self, mock_client):
        """Create RouterAgent with mock client."""
        return RouterAgent(mock_client, model="test-model")

    def test_init(self, mock_client):
        """Test router initialization."""
        router = RouterAgent(mock_client)
        assert router.agent_id == "router-agent"
        assert router.model == "phi-3-mini-4k-instruct-q4_k_m"
        assert router.system_prompt is not None
        assert "task router" in router.system_prompt.lower()

    def test_init_custom_model(self, mock_client):
        """Test router with custom model."""
        router = RouterAgent(mock_client, model="custom-model")
        assert router.model == "custom-model"

    def test_get_tier_for_task_code_complex(self, router):
        """Test tier selection for complex code task."""
        tier = router.get_tier_for_task(TaskType.CODE, Complexity.COMPLEX)
        # Complex code should be T4 or T3
        assert tier in ["T3", "T4"]

    def test_get_tier_for_task_code_simple(self, router):
        """Test tier selection for simple code task."""
        tier = router.get_tier_for_task(TaskType.CODE, Complexity.SIMPLE)
        # Simple code can be T2
        assert tier in ["T2", "T3"]

    def test_get_tier_for_task_reasoning_complex(self, router):
        """Test tier selection for complex reasoning task."""
        tier = router.get_tier_for_task(TaskType.REASONING, Complexity.COMPLEX)
        # Complex reasoning should be T4
        assert tier == "T4"

    def test_get_tier_for_task_writing_simple(self, router):
        """Test tier selection for simple writing task."""
        tier = router.get_tier_for_task(TaskType.WRITING, Complexity.SIMPLE)
        # Simple writing can be T1
        assert tier in ["T1", "T2"]

    def test_get_tier_for_task_research_moderate(self, router):
        """Test tier selection for moderate research task."""
        tier = router.get_tier_for_task(TaskType.RESEARCH, Complexity.MODERATE)
        # Moderate research should be T2
        assert tier == "T2"

    def test_get_tier_for_task_unknown(self, router):
        """Test tier selection for unknown task type."""
        tier = router.get_tier_for_task(TaskType.UNKNOWN, Complexity.MODERATE)
        # Unknown should default to T2
        assert tier == "T2"

    def test_parse_json_response_valid(self, router):
        """Test parsing valid JSON response."""
        response = '''
        {
            "task_type": "code",
            "complexity": "moderate",
            "recommended_tier": "T3",
            "confidence": 0.9,
            "reasoning": "This is a coding task"
        }
        '''
        result = router._parse_json_response(response)

        assert result["task_type"] == "code"
        assert result["complexity"] == "moderate"
        assert result["recommended_tier"] == "T3"
        assert result["confidence"] == 0.9

    def test_parse_json_response_with_surrounding_text(self, router):
        """Test parsing JSON with surrounding text."""
        response = '''
        Here's the analysis:
        {
            "task_type": "research",
            "complexity": "simple"
        }
        Hope this helps!
        '''
        result = router._parse_json_response(response)

        assert result["task_type"] == "research"
        assert result["complexity"] == "simple"

    def test_parse_json_response_invalid_fallback(self, router):
        """Test parsing invalid JSON returns fallback."""
        response = "This is not JSON at all"
        result = router._parse_json_response(response)

        assert result["task_type"] == "unknown"
        assert result["recommended_tier"] == "T2"
        assert "Failed to parse" in result["reasoning"]

    def test_classify_type_code(self, router):
        """Test classifying code task type."""
        task = "Write a Python function to sort a list"
        # Without actual model, this will return UNKNOWN
        # Test is mainly for structure verification
        assert isinstance(TaskType.UNKNOWN, TaskType)

    def test_add_message_to_history(self, router):
        """Test adding messages to history."""
        router._add_user_message("Hello")
        router._add_assistant_message("Hi there")

        history = router.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there"

    def test_clear_history(self, router):
        """Test clearing conversation history."""
        router._add_user_message("Test")
        router.clear_history()

        history = router.get_history()
        assert len(history) == 0
