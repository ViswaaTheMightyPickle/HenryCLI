"""LM Studio plugin and tool support."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolType(Enum):
    """Type of tool/plugin."""

    WEB_SEARCH = "web_search"
    WEBSITE_VISIT = "website_visit"
    RAG = "rag"
    IMAGE_VIEW = "image_view"


@dataclass
class ToolConfig:
    """Configuration for a tool."""

    name: str
    tool_type: ToolType
    enabled: bool = True
    parameters: dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class PluginManager:
    """
    Manages LM Studio plugins and tools.
    
    Supports:
    - DuckDuckGo search
    - Visit website
    - BigRAG (document retrieval)
    - Image viewing
    """

    DEFAULT_TOOLS = {
        "duckduckgo": ToolConfig(
            name="duckduckgo",
            tool_type=ToolType.WEB_SEARCH,
            enabled=True,
            parameters={
                "max_results_per_page": 10,
                "safe_search": "moderate",
            },
        ),
        "visit_website": ToolConfig(
            name="visit_website",
            tool_type=ToolType.WEBSITE_VISIT,
            enabled=True,
            parameters={},
        ),
        "big_rag": ToolConfig(
            name="big_rag",
            tool_type=ToolType.RAG,
            enabled=False,
            parameters={
                "documents_directory": "",
                "vector_store_directory": "",
                "retrieval_limit": 5,
                "affinity_threshold": 0.5,
                "chunk_size": 512,
                "chunk_overlap": 100,
                "max_concurrent_files": 1,
                "enable_ocr": True,
            },
        ),
        "image_view": ToolConfig(
            name="image_view",
            tool_type=ToolType.IMAGE_VIEW,
            enabled=True,
            parameters={},
        ),
    }

    def __init__(self):
        """Initialize plugin manager."""
        self.tools: dict[str, ToolConfig] = {}
        self._initialize_default_tools()

    def _initialize_default_tools(self) -> None:
        """Initialize default tools."""
        for name, config in self.DEFAULT_TOOLS.items():
            self.tools[name] = ToolConfig(
                name=config.name,
                tool_type=config.tool_type,
                enabled=config.enabled,
                parameters=config.parameters.copy(),
            )

    def enable_tool(self, name: str) -> bool:
        """
        Enable a tool.

        Args:
            name: Tool name

        Returns:
            True if tool exists and was enabled
        """
        if name in self.tools:
            self.tools[name].enabled = True
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """
        Disable a tool.

        Args:
            name: Tool name

        Returns:
            True if tool exists and was disabled
        """
        if name in self.tools:
            self.tools[name].enabled = False
            return True
        return False

    def is_tool_enabled(self, name: str) -> bool:
        """
        Check if a tool is enabled.

        Args:
            name: Tool name

        Returns:
            True if tool exists and is enabled
        """
        tool = self.tools.get(name)
        return tool.enabled if tool else False

    def get_enabled_tools(self) -> list[str]:
        """Get list of enabled tool names."""
        return [name for name, tool in self.tools.items() if tool.enabled]

    def get_tool_parameters(self, name: str) -> dict[str, Any] | None:
        """
        Get parameters for a tool.

        Args:
            name: Tool name

        Returns:
            Tool parameters or None if tool doesn't exist
        """
        tool = self.tools.get(name)
        return tool.parameters if tool else None

    def set_tool_parameter(
        self,
        name: str,
        key: str,
        value: Any,
    ) -> bool:
        """
        Set a parameter for a tool.

        Args:
            name: Tool name
            key: Parameter key
            value: Parameter value

        Returns:
            True if tool exists and parameter was set
        """
        if name in self.tools:
            self.tools[name].parameters[key] = value
            return True
        return False

    def configure_rag(
        self,
        documents_dir: str,
        vector_store_dir: str,
        **kwargs: Any,
    ) -> bool:
        """
        Configure BigRAG plugin.

        Args:
            documents_dir: Directory containing documents
            vector_store_dir: Directory for vector database
            **kwargs: Additional parameters

        Returns:
            True if configuration was successful
        """
        if "big_rag" not in self.tools:
            return False

        params = self.tools["big_rag"].parameters
        params["documents_directory"] = documents_dir
        params["vector_store_directory"] = vector_store_dir

        # Update additional parameters
        for key, value in kwargs.items():
            if key in params:
                params[key] = value

        return True

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get tool definitions for LM Studio API.

        Returns:
            List of tool definitions in OpenAI format
        """
        definitions = []

        if self.is_tool_enabled("duckduckgo"):
            definitions.append({
                "type": "function",
                "function": {
                    "name": "duckduckgo_search",
                    "description": "Search the web using DuckDuckGo",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results to return",
                            },
                        },
                        "required": ["query"],
                    },
                },
            })

        if self.is_tool_enabled("visit_website"):
            definitions.append({
                "type": "function",
                "function": {
                    "name": "visit_website",
                    "description": "Visit and extract content from a website",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to visit",
                            },
                        },
                        "required": ["url"],
                    },
                },
            })

        if self.is_tool_enabled("big_rag"):
            definitions.append({
                "type": "function",
                "function": {
                    "name": "rag_search",
                    "description": "Search indexed documents using RAG",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results to return",
                            },
                        },
                        "required": ["query"],
                    },
                },
            })

        return definitions

    def list_plugins(self) -> list[dict[str, Any]]:
        """
        List all available plugins with status.

        Returns:
            List of plugin information
        """
        return [
            {
                "name": name,
                "type": config.tool_type.value,
                "enabled": config.enabled,
                "parameters": config.parameters,
            }
            for name, config in self.tools.items()
        ]
