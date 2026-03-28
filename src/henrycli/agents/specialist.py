"""Specialist agents for different task types."""

import asyncio
from pathlib import Path
from typing import Any

from .base import BaseAgent, TaskResult
from .agentic import AgenticAgent
from ..lmstudio import LMStudioClient
from ..tools import FileSystemTools


class CodeAgent(AgenticAgent):
    """
    Specialist agent for coding tasks.
    
    Handles:
    - Code generation
    - Debugging
    - Refactoring
    - Code review
    
    Uses ReAct-style loop to actually write files and run commands.
    """

    SYSTEM_PROMPT = """You are an expert software engineer that completes coding tasks by actually writing files and running commands.

You have access to these tools:
{tools}

You MUST use this EXACT format for EVERY response:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <tool parameters>

OR when the task is complete:

Thought: I have completed the task
Final Answer: <your final response>

IMPORTANT RULES:
1. You MUST write files to actually create code - do NOT just describe code in markdown
2. Use write_file to create code files with the actual code content
3. Use run_command to test your code (e.g., "python hello.py")
4. Always wait for the Observation after each action
5. Only use Final Answer when the task is truly complete

Example for creating a hello world program:
Thought: I need to create a Python file that prints Hello World
Action: write_file
Action Input: {"path": "hello.py", "content": "print('Hello, World!')"}

Then after you see the file was created:
Thought: Now I should test the program
Action: run_command
Action Input: {"command": "python hello.py"}

Then after you see the output:
Thought: I have completed the task
Final Answer: Created and tested hello.py which prints "Hello, World!"

Begin!"""

    def __init__(self, client: LMStudioClient, model: str = "qwen2.5-7b-instruct-q4_k_m"):
        super().__init__(
            client=client,
            model=model,
            max_iterations=15,  # More iterations for complex coding tasks
        )
        self.agent_id = "code-agent"


class ResearchAgent(AgenticAgent):
    """
    Specialist agent for research tasks.
    
    Handles:
    - File analysis
    - Pattern recognition
    - Information gathering
    - Documentation review
    
    Uses ReAct-style loop to read files and explore the codebase.
    """

    SYSTEM_PROMPT = """You are an expert researcher that analyzes codebases by reading files and exploring directories.

You have access to these tools:
{tools}

To analyze a codebase:
1. Start by listing directories to understand the structure
2. Read relevant files to understand the code
3. Search for patterns or specific code
4. Synthesize your findings
5. Provide a Final Answer with your analysis

Always actually read the files - don't make assumptions!
Use Final Answer when you have completed your analysis.

Begin!"""

    def __init__(self, client: LMStudioClient, model: str = "qwen2.5-7b-instruct-q4_k_m"):
        super().__init__(
            client=client,
            model=model,
            max_iterations=20,  # More iterations for thorough research
        )
        self.agent_id = "research-agent"


class WritingAgent(AgenticAgent):
    """
    Specialist agent for writing tasks.
    
    Handles:
    - Documentation
    - Explanations
    - Content creation
    - Technical writing
    
    Uses ReAct-style loop to write documentation files.
    """

    SYSTEM_PROMPT = """You are an expert technical writer that creates documentation by actually writing files.

You have access to these tools:
{tools}

To complete a writing task:
1. Read any existing context files if needed
2. Write the documentation/content to a file
3. Review what you wrote
4. Provide a Final Answer with the location and summary

Always actually write the documentation files!
Use Final Answer when the writing is complete.

Begin!"""

    def __init__(self, client: LMStudioClient, model: str = "qwen2.5-7b-instruct-q4_k_m"):
        super().__init__(
            client=client,
            model=model,
            max_iterations=10,
        )
        self.agent_id = "writing-agent"


class ReasoningAgent(AgenticAgent):
    """
    Specialist agent for complex reasoning tasks.
    
    Handles:
    - Architecture decisions
    - Security analysis
    - Complex problem solving
    - Mathematical reasoning
    
    Uses ReAct-style loop to read context and write analysis files.
    """

    SYSTEM_PROMPT = """You are an expert problem solver that completes complex reasoning tasks.

You have access to these tools:
{tools}

To complete a reasoning task:
1. Read any relevant context files
2. Think through the problem step-by-step
3. Write your analysis/recommendations to a file
4. Provide a Final Answer with your conclusions

Show your reasoning clearly and provide actionable recommendations.
Use Final Answer when you have completed your analysis.

Begin!"""

    def __init__(self, client: LMStudioClient, model: str = "qwen2.5-32b-instruct-q4_k_m"):
        super().__init__(
            client=client,
            model=model,
            max_iterations=15,
        )
        self.agent_id = "reasoning-agent"


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
        model: Optional model override (if None, uses agent's default)
    
    Returns:
        Specialist agent instance
    """
    # Default models for each type - these are MINIMUM sizes for agentic behavior
    # Smaller models (<7B) often fail to follow ReAct format reliably
    defaults = {
        "code": "qwen2.5-7b-instruct-q4_k_m",  # 7B minimum for code generation
        "research": "qwen2.5-7b-instruct-q4_k_m",  # 7B for analysis
        "writing": "qwen2.5-7b-instruct-q4_k_m",  # 7B for coherent writing
        "reasoning": "qwen2.5-32b-instruct-q4_k_m",  # 32B for complex reasoning
    }
    
    # Use provided model or default
    # If provided model is too small, use the default instead
    selected_model = model or defaults.get(task_type, "qwen2.5-7b-instruct-q4_k_m")
    
    # Check if model is likely too small for agentic work
    # Models < 7B often fail at ReAct-style tool use
    model_too_small = any(
        small in selected_model.lower()
        for small in ["1b", "2b", "3b", "4b", "0.5b", "phi-3-mini", "phi-2"]
    )
    
    if model_too_small and not model:  # Only override if using default
        selected_model = defaults.get(task_type, "qwen2.5-7b-instruct-q4_k_m")
    
    agents = {
        "code": CodeAgent(client, selected_model),
        "research": ResearchAgent(client, selected_model),
        "writing": WritingAgent(client, selected_model),
        "reasoning": ReasoningAgent(client, selected_model),
    }
    
    return agents.get(task_type, WritingAgent(client, selected_model))
