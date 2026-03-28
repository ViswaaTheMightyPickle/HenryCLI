"""Specialist agents for different task types."""

import asyncio
import warnings
from pathlib import Path
from typing import Any

# Suppress spurious RuntimeWarning about unawaited coroutines
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*coroutine.*never awaited')

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
Action Input: {{"path": "hello.py", "content": "print('Hello, World!')"}}

Then after you see the file was created:
Thought: Now I should test the program
Action: run_command
Action Input: {{"command": "python hello.py"}}

Then after you see the output:
Thought: I have completed the task
Final Answer: Created and tested hello.py which prints "Hello, World!"

Begin!"""

    # Models that are too small for reliable ReAct behavior
    SMALL_MODEL_PATTERNS = ["1b", "2b", "3b", "4b", "0.5b", "0.6b", "phi-3-mini", "phi-2", "phi-3-small", "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b"]
    # Preferred models for coding tasks (in priority order)
    PREFERRED_MODELS = ["magnum-v4-9b", "qwen3.5-9b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "nemotron"]

    def __init__(self, client: LMStudioClient, model: str | None = None):
        # Select appropriate model
        selected_model = self._select_model(client, model, "code")
        
        super().__init__(
            client=client,
            model=selected_model,
            max_iterations=15,
        )
        self.agent_id = "code-agent"
    
    def _select_model(self, client: LMStudioClient, model: str | None, task_type: str) -> str:
        """Select appropriate model, avoiding small models that fail at ReAct."""
        import asyncio

        # If no model specified, use default
        if model is None:
            return "qwen2.5-7b-instruct-q4_k_m"

        # Check if model is too small
        model_lower = model.lower()
        if any(pattern in model_lower for pattern in self.SMALL_MODEL_PATTERNS):
            # Need to find a better model from loaded models
            try:
                loop = asyncio.get_event_loop()

                # Check if loop is already running (we're inside async context)
                if loop.is_running():
                    # Can't use run_until_complete, but we can try to get models synchronously
                    # by checking if client has a sync method or cached models
                    # For now, return a preferred model directly
                    # The CLI should have loaded an appropriate model
                    return "qwen2.5-7b-instruct-q4_k_m"

                # Loop not running, safe to use run_until_complete
                loaded = loop.run_until_complete(client.get_models())

                # Find best match from preferred models
                for preferred in self.PREFERRED_MODELS:
                    for mdl in loaded.data:
                        if preferred in mdl.id.lower():
                            return mdl.id

                # Fall back to any model >= 7B
                for mdl in loaded.data:
                    mdl_lower = mdl.id.lower()
                    if any(size in mdl_lower for size in ["7b", "8b", "9b", "14b", "20b", "30b", "32b"]):
                        return mdl.id

                # Last resort: use any non-small model
                for mdl in loaded.data:
                    if not any(p in mdl.id.lower() for p in self.SMALL_MODEL_PATTERNS):
                        return mdl.id

            except Exception as e:
                pass  # Fall through to original model

        return model


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

You MUST use this EXACT format for EVERY response:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <tool parameters>

OR when the task is complete:

Thought: I have completed the task
Final Answer: <your final response>

To analyze a codebase:
1. Start by listing directories to understand the structure
2. Read relevant files to understand the code
3. Search for patterns or specific code
4. Synthesize your findings
5. Provide a Final Answer with your analysis

Always actually read the files - don't make assumptions!
Use Final Answer when you have completed your analysis.

Begin!"""

    SMALL_MODEL_PATTERNS = ["1b", "2b", "3b", "4b", "0.5b", "0.6b", "phi-3-mini", "phi-2", "phi-3-small"]
    PREFERRED_MODELS = ["qwen2.5-7b", "qwen2.5-14b", "magnum-v4-9b", "qwen3.5-9b", "ministral"]

    def __init__(self, client: LMStudioClient, model: str | None = None):
        selected_model = self._select_model(client, model)
        super().__init__(
            client=client,
            model=selected_model,
            max_iterations=20,
        )
        self.agent_id = "research-agent"
    
    def _select_model(self, client: LMStudioClient, model: str | None) -> str:
        """Select appropriate model, avoiding small models."""
        import asyncio
        
        if model is None:
            return "qwen2.5-7b-instruct-q4_k_m"
        
        model_lower = model.lower()
        if any(pattern in model_lower for pattern in self.SMALL_MODEL_PATTERNS):
            try:
                loop = asyncio.get_event_loop()
                loaded = loop.run_until_complete(client.get_models())
                
                for preferred in self.PREFERRED_MODELS:
                    for mdl in loaded.data:
                        if preferred in mdl.id.lower():
                            return mdl.id
                
                for mdl in loaded.data:
                    mdl_lower = mdl.id.lower()
                    if any(size in mdl_lower for size in ["7b", "8b", "9b", "14b", "20b", "30b", "32b"]):
                        return mdl.id
                        
            except Exception:
                pass
        
        return model


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

You MUST use this EXACT format for EVERY response:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <tool parameters>

OR when the task is complete:

Thought: I have completed the task
Final Answer: <your final response>

To complete a writing task:
1. Read any existing context files if needed
2. Write the documentation/content to a file
3. Review what you wrote
4. Provide a Final Answer with the location and summary

Always actually write the documentation files!
Use Final Answer when the writing is complete.

Begin!"""

    SMALL_MODEL_PATTERNS = ["1b", "2b", "3b", "4b", "0.5b", "0.6b", "phi-3-mini", "phi-2", "phi-3-small"]
    PREFERRED_MODELS = ["qwen2.5-7b", "qwen2.5-14b", "magnum-v4-9b", "qwen3.5-9b"]

    def __init__(self, client: LMStudioClient, model: str | None = None):
        selected_model = self._select_model(client, model)
        super().__init__(
            client=client,
            model=selected_model,
            max_iterations=10,
        )
        self.agent_id = "writing-agent"
    
    def _select_model(self, client: LMStudioClient, model: str | None) -> str:
        """Select appropriate model, avoiding small models."""
        import asyncio
        
        if model is None:
            return "qwen2.5-7b-instruct-q4_k_m"
        
        model_lower = model.lower()
        if any(pattern in model_lower for pattern in self.SMALL_MODEL_PATTERNS):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return model  # Can't query during running loop
                
                loaded = loop.run_until_complete(client.get_models())
                
                for preferred in self.PREFERRED_MODELS:
                    for mdl in loaded.data:
                        if preferred in mdl.id.lower():
                            return mdl.id
                
                for mdl in loaded.data:
                    if any(size in mdl.id.lower() for size in ["7b", "8b", "9b", "14b", "20b", "30b", "32b"]):
                        return mdl.id
                        
            except Exception:
                pass
        
        return model


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

You MUST use this EXACT format for EVERY response:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <tool parameters>

OR when the task is complete:

Thought: I have completed the task
Final Answer: <your final response>

To complete a reasoning task:
1. Read any relevant context files
2. Think through the problem step-by-step
3. Write your analysis/recommendations to a file
4. Provide a Final Answer with your conclusions

Show your reasoning clearly and provide actionable recommendations.
Use Final Answer when you have completed your analysis.

Begin!"""

    SMALL_MODEL_PATTERNS = ["1b", "2b", "3b", "4b", "0.5b", "0.6b", "phi-3-mini", "phi-2", "phi-3-small"]
    PREFERRED_MODELS = ["qwen2.5-32b", "qwen3-coder-30b", "gpt-oss-20b", "qwen2.5-14b"]

    def __init__(self, client: LMStudioClient, model: str | None = None):
        selected_model = self._select_model(client, model)
        super().__init__(
            client=client,
            model=selected_model,
            max_iterations=15,
        )
        self.agent_id = "reasoning-agent"
    
    def _select_model(self, client: LMStudioClient, model: str | None) -> str:
        """Select appropriate model, avoiding small models."""
        import asyncio

        if model is None:
            return "qwen2.5-32b-instruct-q4_k_m"

        model_lower = model.lower()
        if any(pattern in model_lower for pattern in self.SMALL_MODEL_PATTERNS):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return model

                loaded = loop.run_until_complete(client.get_models())

                for preferred in self.PREFERRED_MODELS:
                    for mdl in loaded.data:
                        if preferred in mdl.id.lower():
                            return mdl.id

                # For reasoning, prefer larger models (20B+)
                for mdl in loaded.data:
                    if any(size in mdl.id.lower() for size in ["20b", "30b", "32b", "34b", "70b"]):
                        return mdl.id

                # Fall back to any 7B+
                for mdl in loaded.data:
                    if any(size in mdl.id.lower() for size in ["7b", "8b", "9b", "14b"]):
                        return mdl.id

            except Exception:
                pass

        return model


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
    # Patterns that indicate small models (< 7B) - these fail at ReAct format
    small_model_patterns = ["1b", "2b", "3b", "4b", "0.5b", "0.6b", "phi-3-mini", "phi-2", "phi-3-small"]
    
    # Pass model selection to agent - it will handle async model discovery
    agents = {
        "code": CodeAgent(client, model),
        "research": ResearchAgent(client, model),
        "writing": WritingAgent(client, model),
        "reasoning": ReasoningAgent(client, model),
    }
    
    agent = agents.get(task_type, WritingAgent(client, model))
    
    # If model was too small, agent will have selected a better one
    return agent
