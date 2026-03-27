"""Tests for plugin manager."""

import pytest

from henrycli.plugins import PluginManager, ToolType


class TestPluginManager:
    """Tests for PluginManager."""

    @pytest.fixture
    def plugin_manager(self):
        """Create plugin manager."""
        return PluginManager()

    def test_init_default_tools(self, plugin_manager):
        """Test initialization with default tools."""
        assert "duckduckgo" in plugin_manager.tools
        assert "visit_website" in plugin_manager.tools
        assert "big_rag" in plugin_manager.tools
        assert "image_view" in plugin_manager.tools

    def test_enable_tool(self, plugin_manager):
        """Test enabling a tool."""
        # Disable first
        plugin_manager.disable_tool("duckduckgo")
        assert plugin_manager.is_tool_enabled("duckduckgo") is False

        # Enable
        result = plugin_manager.enable_tool("duckduckgo")
        assert result is True
        assert plugin_manager.is_tool_enabled("duckduckgo") is True

    def test_disable_tool(self, plugin_manager):
        """Test disabling a tool."""
        result = plugin_manager.disable_tool("duckduckgo")
        assert result is True
        assert plugin_manager.is_tool_enabled("duckduckgo") is False

    def test_enable_unknown_tool(self, plugin_manager):
        """Test enabling unknown tool."""
        result = plugin_manager.enable_tool("unknown_plugin")
        assert result is False

    def test_get_enabled_tools(self, plugin_manager):
        """Test getting enabled tools."""
        # All tools should be enabled by default (except big_rag)
        enabled = plugin_manager.get_enabled_tools()
        assert "duckduckgo" in enabled
        assert "visit_website" in enabled
        assert "image_view" in enabled

    def test_get_tool_parameters(self, plugin_manager):
        """Test getting tool parameters."""
        params = plugin_manager.get_tool_parameters("duckduckgo")

        assert params is not None
        assert "max_results_per_page" in params
        assert "safe_search" in params

    def test_set_tool_parameter(self, plugin_manager):
        """Test setting tool parameter."""
        result = plugin_manager.set_tool_parameter(
            "duckduckgo", "max_results_per_page", 20
        )
        assert result is True

        params = plugin_manager.get_tool_parameters("duckduckgo")
        assert params["max_results_per_page"] == 20

    def test_set_unknown_tool_parameter(self, plugin_manager):
        """Test setting parameter for unknown tool."""
        result = plugin_manager.set_tool_parameter(
            "unknown", "key", "value"
        )
        assert result is False

    def test_configure_rag(self, plugin_manager):
        """Test configuring BigRAG plugin."""
        result = plugin_manager.configure_rag(
            documents_dir="/home/user/docs",
            vector_store_dir="/home/user/rag-db",
            retrieval_limit=10,
        )

        assert result is True

        params = plugin_manager.get_tool_parameters("big_rag")
        assert params["documents_directory"] == "/home/user/docs"
        assert params["vector_store_directory"] == "/home/user/rag-db"
        assert params["retrieval_limit"] == 10

    def test_get_tool_definitions(self, plugin_manager):
        """Test getting tool definitions for API."""
        # Enable all tools
        plugin_manager.enable_tool("duckduckgo")
        plugin_manager.enable_tool("visit_website")

        definitions = plugin_manager.get_tool_definitions()

        assert len(definitions) >= 2

        # Check DuckDuckGo definition
        ddg_def = next(
            (d for d in definitions if d["function"]["name"] == "duckduckgo_search"),
            None,
        )
        assert ddg_def is not None
        assert "query" in ddg_def["function"]["parameters"]["required"]

        # Check Visit Website definition
        vw_def = next(
            (d for d in definitions if d["function"]["name"] == "visit_website"),
            None,
        )
        assert vw_def is not None
        assert "url" in vw_def["function"]["parameters"]["required"]

    def test_get_tool_definitions_excludes_disabled(self, plugin_manager):
        """Test that disabled tools are excluded from definitions."""
        plugin_manager.disable_tool("duckduckgo")
        plugin_manager.disable_tool("visit_website")

        definitions = plugin_manager.get_tool_definitions()

        # Should not include disabled tools
        ddg_def = next(
            (d for d in definitions if d["function"]["name"] == "duckduckgo_search"),
            None,
        )
        assert ddg_def is None

    def test_list_plugins(self, plugin_manager):
        """Test listing all plugins."""
        plugins = plugin_manager.list_plugins()

        assert len(plugins) >= 4

        # Check structure
        for plugin in plugins:
            assert "name" in plugin
            assert "type" in plugin
            assert "enabled" in plugin
            assert "parameters" in plugin

    def test_big_rag_default_disabled(self, plugin_manager):
        """Test that BigRAG is disabled by default."""
        # BigRAG requires configuration, should be disabled by default
        assert plugin_manager.is_tool_enabled("big_rag") is False

    def test_tool_type_enum(self):
        """Test ToolType enum values."""
        assert ToolType.WEB_SEARCH.value == "web_search"
        assert ToolType.WEBSITE_VISIT.value == "website_visit"
        assert ToolType.RAG.value == "rag"
        assert ToolType.IMAGE_VIEW.value == "image_view"
