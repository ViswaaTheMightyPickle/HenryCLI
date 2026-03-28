"""Agentic agent with ReAct-style tool use."""

import re
from typing import Any, Callable

from .base import BaseAgent, TaskResult
from ..lmstudio import LMStudioClient
from ..tools import FileSystemTools


class Tool:
    """A tool that an agent can use."""
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., dict[str, Any]],
        parameters: dict[str, str] | None = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}
    
    def __call__(self, **kwargs) -> dict[str, Any]:
        """Execute the tool."""
        return self.func(**kwargs)
    
    def to_prompt(self) -> str:
        """Generate tool description for prompt."""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        if params:
            # Use concatenation to avoid format() conflicts
            return "- " + self.name + "(" + params + "): " + self.description
        return "- " + self.name + "(): " + self.description


class AgenticAgent(BaseAgent):
    """
    Agentic agent with ReAct-style tool use.
    
    Uses a thought-action-observation loop:
    1. Think about what to do
    2. Act using a tool
    3. Observe the result
    4. Repeat until task is complete
    """

    SYSTEM_PROMPT = """You are an AI assistant that can use tools to complete tasks.

You have access to the following tools:
{tools}

To use a tool, respond with this format:
Thought: <your reasoning about what to do>
Action: <tool_name>
Action Input: <tool parameters as JSON or simple text>

When you have completed the task, respond with:
Thought: I have completed the task
Final Answer: <your final response to the user>

Important rules:
1. Always start with a Thought
2. Use only one tool at a time
3. Wait for the observation before taking the next action
4. Use Final Answer when the task is complete
5. If you make a mistake, acknowledge it and try a different approach

Begin!"""

    def __init__(
        self,
        client: LMStudioClient,
        model: str = "qwen2.5-7b-instruct-q4_k_m",
        max_iterations: int = 10,
    ):
        super().__init__(
            agent_id="agentic-agent",
            model=model,
            client=client,
            system_prompt="",  # Will be set after tools are registered
        )
        self.tools: dict[str, Tool] = {}
        self.fs_tools = FileSystemTools()
        self.max_iterations = max_iterations
        self._register_default_tools()
        self._update_system_prompt()

    def _register_default_tools(self) -> None:
        """Register default tools."""
        self.register_tool(
            Tool(
                name="read_file",
                description="Read the contents of a file",
                func=self._read_file_tool,
                parameters={"path": "str - Path to the file"},
            )
        )
        self.register_tool(
            Tool(
                name="write_file",
                description="Write content to a file (creates if doesn't exist)",
                func=self._write_file_tool,
                parameters={
                    "path": "str - Path to the file",
                    "content": "str - Content to write",
                },
            )
        )
        self.register_tool(
            Tool(
                name="list_directory",
                description="List files in a directory",
                func=self._list_directory_tool,
                parameters={"path": "str - Path to the directory (default: current)"},
            )
        )
        self.register_tool(
            Tool(
                name="search_files",
                description="Search for files matching a pattern",
                func=self._search_files_tool,
                parameters={
                    "pattern": "str - Glob pattern (e.g., '*.py')",
                    "recursive": "bool - Search recursively (default: True)",
                },
            )
        )
        self.register_tool(
            Tool(
                name="run_command",
                description="Run a shell command and return output",
                func=self._run_command_tool,
                parameters={"command": "str - Shell command to execute"},
            )
        )

    def register_tool(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        self._update_system_prompt()

    def _update_system_prompt(self) -> None:
        """Update system prompt with available tools."""
        tools_str = "\n".join(tool.to_prompt() for tool in self.tools.values())
        self.system_prompt = self.SYSTEM_PROMPT.format(tools=tools_str)

    def _read_file_tool(self, path: str) -> dict[str, Any]:
        """Read file tool implementation."""
        return self.fs_tools.read_file(path)

    def _write_file_tool(self, path: str, content: str) -> dict[str, Any]:
        """Write file tool implementation."""
        return self.fs_tools.write_file(path, content)

    def _list_directory_tool(self, path: str = ".") -> dict[str, Any]:
        """List directory tool implementation."""
        return self.fs_tools.list_directory(path)

    def _search_files_tool(self, pattern: str, recursive: bool = True) -> dict[str, Any]:
        """Search files tool implementation."""
        return self.fs_tools.search_files(pattern, recursive=recursive)

    def _run_command_tool(self, command: str) -> dict[str, Any]:
        """Run command tool implementation."""
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out (60s limit)",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _parse_action(self, response: str) -> dict[str, str] | None:
        """Parse action from model response."""
        # Look for Action: and Action Input: with more flexible patterns
        action_match = re.search(r"Action:\s*([a-zA-Z_]+)", response)
        
        # Try multiple patterns for action input
        input_patterns = [
            r"Action Input:\s*(\{.+?\})",  # JSON format
            r"Action Input:\s*(.+?)(?=\n|$)",  # Text format until newline
        ]
        
        input_match = None
        for pattern in input_patterns:
            input_match = re.search(pattern, response, re.DOTALL)
            if input_match:
                break
        
        if action_match:
            action_input = input_match.group(1).strip() if input_match else ""
            
            # Clean up common issues
            if action_input.startswith('"') and action_input.endswith('"'):
                action_input = action_input[1:-1]
            
            return {
                "action": action_match.group(1).strip(),
                "input": action_input,
            }
        return None

    def _has_final_answer(self, response: str) -> tuple[bool, str | None]:
        """Check if response contains final answer."""
        if "Final Answer:" in response:
            # Extract final answer
            match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
            if match:
                return True, match.group(1).strip()
        return False, None

    async def execute(self, task: str) -> TaskResult:
        """Execute task using ReAct loop."""
        try:
            # Clear history and add task
            self.clear_history()
            self._add_user_message(f"Task: {task}")
            
            artifacts = []
            iteration = 0
            thoughts = []
            
            while iteration < self.max_iterations:
                iteration += 1
                
                # Get model response
                response = await self._call_model(temperature=0.2, max_tokens=1000)
                self._add_assistant_message(response)
                
                # Extract thought
                thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", response, re.DOTALL)
                if thought_match:
                    thoughts.append(thought_match.group(1).strip())
                
                # Check for final answer
                has_answer, final_answer = self._has_final_answer(response)
                if has_answer and final_answer:
                    return TaskResult(
                        success=True,
                        output=final_answer,
                        artifacts=artifacts,
                        metadata={
                            "agent": "agentic",
                            "iterations": iteration,
                            "thoughts": thoughts,
                        },
                    )
                
                # Parse and execute action
                action = self._parse_action(response)
                if not action:
                    # No action found and no final answer - model may be confused
                    self._add_user_message(
                        "Error: Could not parse action. Please use the format:\n"
                        "Thought: <reasoning>\n"
                        "Action: <tool_name>\n"
                        "Action Input: <parameters>"
                    )
                    continue
                
                # Execute tool
                tool_name = action["action"]
                tool_input = action["input"]
                
                if tool_name not in self.tools:
                    observation = f"Error: Unknown tool '{tool_name}'"
                else:
                    # Parse tool input
                    try:
                        # Try to parse as JSON first
                        import json
                        input_dict = json.loads(tool_input)
                    except (json.JSONDecodeError, ValueError):
                        # Fall back to simple string parsing
                        input_dict = {"path": tool_input} if tool_input else {}
                    
                    # Execute tool
                    try:
                        tool_result = self.tools[tool_name](**input_dict)
                        
                        # Format observation
                        if tool_result.get("success"):
                            if "content" in tool_result:
                                observation = f"Success:\n{tool_result['content'][:2000]}"
                            elif "files" in tool_result:
                                observation = f"Success: Found {tool_result['count']} files\n{chr(10).join(tool_result['files'][:20])}"
                            elif "path" in tool_result:
                                observation = f"Success: {tool_result['path']}"
                            else:
                                observation = f"Success: {tool_result}"
                            
                            # Track artifacts
                            if "path" in tool_result and tool_name == "write_file":
                                artifacts.append(tool_result["path"])
                        else:
                            observation = f"Error: {tool_result.get('error', 'Unknown error')}"
                    except Exception as e:
                        observation = f"Error executing tool: {e}"
                
                # Add observation to history
                self._add_user_message(f"Observation: {observation}")
            
            # Max iterations reached
            return TaskResult(
                success=False,
                output="",
                error=f"Max iterations ({self.max_iterations}) reached",
                metadata={
                    "agent": "agentic",
                    "iterations": iteration,
                    "thoughts": thoughts,
                },
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                output="",
                error=str(e),
            )
