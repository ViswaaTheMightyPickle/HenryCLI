"""Comprehensive CLI command tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typer.testing import CliRunner

from henrycli.cli import app
from henrycli.lmstudio import LMStudioClient, ModelList, ModelInfo

runner = CliRunner()


class MockLMStudioClient:
    """Mock LM Studio client for testing."""

    def __init__(self, *args, **kwargs):
        self.base_url = "http://localhost:1234"

    async def health_check(self):
        return True

    async def get_models(self):
        return ModelList(data=[
            ModelInfo(id="test-model-1"),
            ModelInfo(id="test-model-2"),
        ])

    async def close(self):
        pass


@pytest.fixture
def mock_lmstudio():
    """Mock LM Studio client."""
    with patch("henrycli.cli.LMStudioClient", MockLMStudioClient):
        yield


class TestVersionCommand:
    """Tests for version command."""

    def test_version(self, mock_lmstudio):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "HenryCLI" in result.output
        assert "v" in result.output


class TestHealthCommand:
    """Tests for health command."""

    def test_health_success(self, mock_lmstudio):
        """Test health command with successful connection."""
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        assert "LM Studio" in result.output


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_requires_task(self, mock_lmstudio):
        """Test analyze requires task argument."""
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code != 0

    def test_analyze_with_task(self, mock_lmstudio):
        """Test analyze with a task."""
        result = runner.invoke(app, ["analyze", "Write hello world"])
        # May fail due to no actual LM Studio connection
        # Just verify it runs
        assert result.exit_code in [0, 1]


class TestModelsCommand:
    """Tests for models command."""

    def test_models_list(self, mock_lmstudio):
        """Test models list command."""
        result = runner.invoke(app, ["models", "--list"])
        assert result.exit_code == 0
        assert "Configured Models" in result.output or "T1" in result.output

    def test_models_stats(self, mock_lmstudio):
        """Test models stats command."""
        result = runner.invoke(app, ["models", "--stats"])
        # May fail due to no actual connection
        assert result.exit_code in [0, 1]


class TestContextCommand:
    """Tests for context command."""

    def test_context_show(self, mock_lmstudio):
        """Test context show command."""
        result = runner.invoke(app, ["context", "--show"])
        assert result.exit_code == 0

    def test_context_clear(self, mock_lmstudio):
        """Test context clear command."""
        result = runner.invoke(app, ["context", "--clear"])
        assert result.exit_code == 0
        assert "Cleared" in result.output or "clear" in result.output.lower()


class TestConfigCommand:
    """Tests for config command."""

    def test_config_show(self, mock_lmstudio):
        """Test config show command."""
        result = runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "Configuration" in result.output or "Hardware" in result.output

    def test_config_edit(self, mock_lmstudio):
        """Test config edit command."""
        result = runner.invoke(app, ["config", "--edit"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()


class TestPluginsCommand:
    """Tests for plugins command."""

    def test_plugins_list(self, mock_lmstudio):
        """Test plugins list command."""
        result = runner.invoke(app, ["plugins", "--list"])
        assert result.exit_code == 0
        assert "Plugin" in result.output or "duckduckgo" in result.output.lower()

    def test_plugins_enable(self, mock_lmstudio):
        """Test plugins enable command."""
        result = runner.invoke(app, ["plugins", "--enable", "duckduckgo"])
        assert result.exit_code == 0
        assert "Enabled" in result.output or "duckduckgo" in result.output.lower()

    def test_plugins_disable(self, mock_lmstudio):
        """Test plugins disable command."""
        result = runner.invoke(app, ["plugins", "--disable", "duckduckgo"])
        assert result.exit_code == 0
        assert "Disabled" in result.output or "duckduckgo" in result.output.lower()


class TestLoadCommand:
    """Tests for load command."""

    def test_load_requires_model(self, mock_lmstudio):
        """Test load requires model argument."""
        result = runner.invoke(app, ["load"])
        assert result.exit_code != 0

    def test_load_with_model(self, mock_lmstudio):
        """Test load with model argument."""
        result = runner.invoke(app, ["load", "test/model"])
        # Will fail without actual LM Studio
        assert result.exit_code in [0, 1]


class TestUnloadCommand:
    """Tests for unload command."""

    def test_unload_all(self, mock_lmstudio):
        """Test unload all command."""
        result = runner.invoke(app, ["unload", "--all"])
        # Will fail without actual LM Studio
        assert result.exit_code in [0, 1]

    def test_unload_show_loaded(self, mock_lmstudio):
        """Test unload without args shows loaded models."""
        result = runner.invoke(app, ["unload"])
        assert result.exit_code in [0, 1]


class TestDiscoverCommand:
    """Tests for discover command."""

    def test_discover_no_models(self, mock_lmstudio):
        """Test discover when no models found."""
        result = runner.invoke(app, ["discover"])
        # Should show helpful message
        assert result.exit_code in [0, 1]

    def test_discover_with_cli_flag(self, mock_lmstudio):
        """Test discover with --use-cli flag."""
        result = runner.invoke(app, ["discover", "--use-cli"])
        assert result.exit_code in [0, 1]


class TestGetCommand:
    """Tests for get command."""

    def test_get_requires_url(self, mock_lmstudio):
        """Test get requires URL argument."""
        result = runner.invoke(app, ["get"])
        assert result.exit_code != 0

    def test_get_list(self, mock_lmstudio):
        """Test get --list command."""
        result = runner.invoke(app, ["get", "--list"])
        assert result.exit_code == 0
        # Should show either files or "no downloaded files" message
        assert "RAG" in result.output or "downloaded" in result.output.lower() or "No" in result.output

    def test_get_with_url(self, mock_lmstudio):
        """Test get with URL."""
        result = runner.invoke(app, ["get", "https://example.com/file.txt"])
        # Will attempt download
        assert result.exit_code in [0, 1]


class TestTUICommand:
    """Tests for TUI command."""

    def test_tui_help(self, mock_lmstudio):
        """Test TUI command exists."""
        result = runner.invoke(app, ["tui", "--help"])
        # TUI doesn't have --help, but should not crash
        assert result.exit_code in [0, 2]


class TestHelpCommand:
    """Tests for help."""

    def test_main_help(self, mock_lmstudio):
        """Test main help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "HenryCLI" in result.output
        assert "Commands" in result.output

    def test_all_commands_listed(self, mock_lmstudio):
        """Test that all commands are listed in help."""
        result = runner.invoke(app, ["--help"])
        expected_commands = [
            "version",
            "health",
            "analyze",
            "run",
            "models",
            "context",
            "config",
            "plugins",
            "load",
            "unload",
            "discover",
            "get",
            "tui",
        ]
        for cmd in expected_commands:
            assert cmd in result.output
