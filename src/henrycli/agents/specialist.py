"""Specialist agents for different task types."""

import asyncio
from pathlib import Path
from typing import Any

from .base import BaseAgent, TaskResult
from ..lmstudio import LMStudioClient
from ..tools import FileSystemTools


class CodeAgent(BaseAgent):
    """
    Specialist agent for coding tasks.
    
    Handles:
    - Code generation
    - Debugging
    - Refactoring
    - Code review
    """

    SYSTEM_PROMPT = """You are an expert software engineer specializing in code generation, debugging, and refactoring.

Your capabilities:
- Write clean, efficient, and well-documented code
- Debug and fix issues
- Refactor code for better maintainability
- Follow best practices and design patterns
- Write tests when appropriate

When writing code:
- Use meaningful variable names
- Add comments for complex logic
- Handle errors appropriately
- Follow the project's existing style

Output your response with code in markdown code blocks."""

    def __init__(self, client: LMStudioClient, model: str = "qwen2.5-7b-instruct-q4_k_m"):
        super().__init__(
            agent_id="code-agent",
            model=model,
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
        )
        self.fs_tools = FileSystemTools()
        self.file_store: dict[str, str] = {}

    async def execute(self, task: str) -> TaskResult:
        """Execute a coding task."""
        try:
            self._add_user_message(task)
            response = await self._call_model(temperature=0.3, max_tokens=4000)
            
            # Extract any code blocks as artifacts
            artifacts = self._extract_code_blocks(response)
            
            return TaskResult(
                success=True,
                output=response,
                artifacts=artifacts,
                metadata={"agent": "code", "artifacts_count": len(artifacts)},
            )
        except Exception as e:
            return TaskResult(
                success=False,
                output="",
                error=str(e),
            )

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract code blocks from response."""
        import re
        blocks = re.findall(r'```[\w]*\n(.*?)```', text, re.DOTALL)
        return blocks

    def read_file(self, path: str) -> str:
        """Read a file into the agent's context."""
        result = self.fs_tools.read_file(path)
        if result["success"]:
            content = result["content"]
            self.file_store[path] = content
            self._add_user_message(f"--- Content of {path} ---\n{content}")
            return content
        else:
            error_msg = f"Error reading {path}: {result.get('error', 'Unknown error')}"
            self._add_user_message(error_msg)
            return ""

    def write_file(self, path: str, content: str) -> bool:
        """Write content to a file."""
        result = self.fs_tools.write_file(path, content)
        if result["success"]:
            self.file_store[path] = content
            return True
        return False

    def list_directory(self, path: str = ".") -> list[str]:
        """List files in a directory."""
        result = self.fs_tools.list_directory(path)
        if result["success"]:
            files = result["files"]
            self._add_user_message(f"--- Files in {path} ---\n" + "\n".join(files))
            return files
        return []

    def search_files(self, pattern: str, recursive: bool = True) -> list[str]:
        """Search for files matching a pattern."""
        result = self.fs_tools.search_files(pattern, recursive=recursive)
        if result["success"]:
            files = result["files"]
            self._add_user_message(f"--- Files matching '{pattern}' ---\n" + "\n".join(files))
            return files
        return []


class ResearchAgent(BaseAgent):
    """
    Specialist agent for research tasks.
    
    Handles:
    - File analysis
    - Pattern recognition
    - Information gathering
    - Documentation review
    """

    SYSTEM_PROMPT = """You are an expert researcher specializing in analyzing codebases, documentation, and technical information.

Your capabilities:
- Analyze file structures and patterns
- Review documentation and extract key information
- Identify architectural patterns
- Summarize technical content
- Find connections between different parts of a codebase

When analyzing:
- Be thorough and systematic
- Cite specific files or sections when relevant
- Provide clear summaries
- Highlight important findings"""

    def __init__(self, client: LMStudioClient, model: str = "qwen2.5-7b-instruct-q4_k_m"):
        super().__init__(
            agent_id="research-agent",
            model=model,
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
        )
        self.fs_tools = FileSystemTools()
        self.analyzed_files: list[str] = []

    async def execute(self, task: str) -> TaskResult:
        """Execute a research task."""
        try:
            self._add_user_message(task)
            response = await self._call_model(temperature=0.2, max_tokens=4000)
            
            return TaskResult(
                success=True,
                output=response,
                artifacts=self.analyzed_files,
                metadata={"agent": "research", "files_analyzed": len(self.analyzed_files)},
            )
        except Exception as e:
            return TaskResult(
                success=False,
                output="",
                error=str(e),
            )

    def read_file(self, path: str) -> str:
        """Read a file for analysis."""
        result = self.fs_tools.read_file(path, max_lines=200)
        if result["success"]:
            content = result["content"]
            self.analyzed_files.append(path)
            self._add_user_message(f"--- Content of {path} ({result['lines']} lines) ---\n{content}")
            return content
        else:
            error_msg = f"Error reading {path}: {result.get('error', 'Unknown error')}"
            self._add_user_message(error_msg)
            return ""

    def list_directory(self, path: str = ".", recursive: bool = False) -> list[str]:
        """List files in a directory."""
        result = self.fs_tools.list_directory(path, recursive=recursive)
        if result["success"]:
            files = result["files"]
            self._add_user_message(f"--- Files in {path} ({result['count']} files) ---\n" + "\n".join(files[:50]))
            if len(files) > 50:
                self._add_user_message(f"... and {len(files) - 50} more files")
            return files
        return []

    def search_files(self, pattern: str, recursive: bool = True) -> list[str]:
        """Search for files matching a pattern."""
        result = self.fs_tools.search_files(pattern, recursive=recursive)
        if result["success"]:
            files = result["files"]
            self._add_user_message(f"--- Files matching '{pattern}' ({result['count']} found) ---\n" + "\n".join(files[:50]))
            if len(files) > 50:
                self._add_user_message(f"... and {len(files) - 50} more files")
            return files
        return []


class WritingAgent(BaseAgent):
    """
    Specialist agent for writing tasks.
    
    Handles:
    - Documentation
    - Explanations
    - Content creation
    - Technical writing
    """

    SYSTEM_PROMPT = """You are an expert technical writer specializing in clear, concise, and well-structured documentation.

Your capabilities:
- Write technical documentation
- Create tutorials and guides
- Write clear explanations
- Edit and improve existing content
- Adapt tone for different audiences

When writing:
- Use clear, simple language
- Structure content logically
- Use examples when helpful
- Include headings and formatting for readability
- Be concise but thorough"""

    def __init__(self, client: LMStudioClient, model: str = "qwen2.5-7b-instruct-q4_k_m"):
        super().__init__(
            agent_id="writing-agent",
            model=model,
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
        )
        self.fs_tools = FileSystemTools()

    async def execute(self, task: str) -> TaskResult:
        """Execute a writing task."""
        try:
            self._add_user_message(task)
            response = await self._call_model(temperature=0.5, max_tokens=4000)
            
            return TaskResult(
                success=True,
                output=response,
                metadata={"agent": "writing"},
            )
        except Exception as e:
            return TaskResult(
                success=False,
                output="",
                error=str(e),
            )

    def read_context(self, path: str) -> str:
        """Read context from a file."""
        result = self.fs_tools.read_file(path)
        if result["success"]:
            content = result["content"]
            self._add_user_message(f"--- Context from {path} ---\n{content}")
            return content
        else:
            error_msg = f"Error reading {path}: {result.get('error', 'Unknown error')}"
            self._add_user_message(error_msg)
            return ""

    def write_file(self, path: str, content: str) -> bool:
        """Write content to a file."""
        result = self.fs_tools.write_file(path, content)
        return result["success"]


class ReasoningAgent(BaseAgent):
    """
    Specialist agent for complex reasoning tasks.
    
    Handles:
    - Architecture decisions
    - Security analysis
    - Complex problem solving
    - Mathematical reasoning
    """

    SYSTEM_PROMPT = """You are an expert problem solver specializing in complex reasoning tasks.

Your capabilities:
- Analyze complex problems systematically
- Make architectural decisions
- Perform security analysis
- Solve mathematical problems
- Provide step-by-step reasoning

When reasoning:
- Break down problems into smaller parts
- Show your reasoning step-by-step
- Consider multiple approaches
- Evaluate trade-offs
- Provide clear conclusions"""

    def __init__(self, client: LMStudioClient, model: str = "qwen2.5-32b-instruct-q4_k_m"):
        super().__init__(
            agent_id="reasoning-agent",
            model=model,
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
        )

    async def execute(self, task: str) -> TaskResult:
        """Execute a reasoning task."""
        try:
            self._add_user_message(task)
            response = await self._call_model(temperature=0.2, max_tokens=6000)
            
            return TaskResult(
                success=True,
                output=response,
                metadata={"agent": "reasoning"},
            )
        except Exception as e:
            return TaskResult(
                success=False,
                output="",
                error=str(e),
            )


def create_agent_for_type(
    task_type: str,
    client: LMStudioClient,
    model: str | None = None,
) -> BaseAgent:
    """
    Create appropriate agent for task type.
    
    Args:
        task_type: Type of task (code, research, writing, reasoning)
        client: LM Studio client
        model: Optional model override
    
    Returns:
        Specialist agent instance
    """
    # Default models for each type
    defaults = {
        "code": "qwen2.5-7b-instruct-q4_k_m",
        "research": "qwen2.5-7b-instruct-q4_k_m",
        "writing": "qwen2.5-7b-instruct-q4_k_m",
        "reasoning": "qwen2.5-32b-instruct-q4_k_m",
    }
    
    selected_model = model or defaults.get(task_type, "qwen2.5-7b-instruct-q4_k_m")
    
    agents = {
        "code": CodeAgent(client, selected_model),
        "research": ResearchAgent(client, selected_model),
        "writing": WritingAgent(client, selected_model),
        "reasoning": ReasoningAgent(client, selected_model),
    }
    
    return agents.get(task_type, WritingAgent(client, selected_model))
