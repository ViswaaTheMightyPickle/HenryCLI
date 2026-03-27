"""Router Agent for task classification and routing."""

import json
import re
from dataclasses import dataclass
from enum import Enum

from ..lmstudio import ChatMessage, LMStudioClient
from .base import BaseAgent, TaskResult


class TaskType(Enum):
    """Type of task."""

    CODE = "code"
    RESEARCH = "research"
    WRITING = "writing"
    REASONING = "reasoning"
    UNKNOWN = "unknown"


class Complexity(Enum):
    """Task complexity level."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


@dataclass
class TaskAnalysis:
    """Analysis result from router."""

    task_type: TaskType
    complexity: Complexity
    subtasks: list[dict[str, str]]
    recommended_tier: str
    confidence: float
    reasoning: str


class RouterAgent(BaseAgent):
    """
    Router Agent - analyzes tasks and routes to appropriate agents.
    
    Uses a small model (4B) to:
    - Classify task type
    - Estimate complexity
    - Decompose into subtasks
    - Recommend agent tier
    """

    SYSTEM_PROMPT = """You are a task router for a multi-agent LLM system. Analyze the input task and provide a structured analysis.

Task Types:
- code: Programming, debugging, refactoring, code generation
- research: Information gathering, file analysis, pattern recognition
- writing: Documentation, explanations, content creation
- reasoning: Complex multi-step reasoning, architecture decisions, security analysis

Complexity Levels:
- simple: Single step, straightforward, < 5 minutes
- moderate: Multiple steps, some complexity, 5-30 minutes
- complex: Many steps, high complexity, > 30 minutes

Agent Tiers:
- T1: 4B models (routing, simple Q&A)
- T2: 7B models (general tasks, writing, research)
- T3: 13B models (code, debugging, architecture)
- T4: 30B models (deep reasoning, security, math)

Output JSON with this exact structure:
{
    "task_type": "<code|research|writing|reasoning>",
    "complexity": "<simple|moderate|complex>",
    "subtasks": [{"description": "...", "type": "...", "complexity": "..."}],
    "recommended_tier": "<T1|T2|T3|T4>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}

Be concise in your reasoning. Output ONLY valid JSON, no other text."""

    def __init__(self, client: LMStudioClient, model: str = "phi-3-mini-4k-instruct-q4_k_m"):
        """
        Initialize router agent.

        Args:
            client: LM Studio client
            model: Model to use (default: 4B model)
        """
        super().__init__(
            agent_id="router-agent",
            model=model,
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
        )

    async def analyze(self, task: str) -> TaskAnalysis:
        """
        Analyze a task and return structured analysis.

        Args:
            task: Task description

        Returns:
            TaskAnalysis with classification and routing info
        """
        self._add_user_message(f"Analyze this task: {task}")
        response = await self._call_model(temperature=0.3, max_tokens=1000)
        self._add_assistant_message(response)

        # Parse JSON response
        analysis_data = self._parse_json_response(response)

        return TaskAnalysis(
            task_type=TaskType(analysis_data.get("task_type", "unknown")),
            complexity=Complexity(analysis_data.get("complexity", "unknown")),
            subtasks=analysis_data.get("subtasks", []),
            recommended_tier=analysis_data.get("recommended_tier", "T2"),
            confidence=analysis_data.get("confidence", 0.5),
            reasoning=analysis_data.get("reasoning", ""),
        )

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from model response."""
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse as JSON directly
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Last resort: return default structure
        return {
            "task_type": "unknown",
            "complexity": "moderate",
            "subtasks": [],
            "recommended_tier": "T2",
            "confidence": 0.3,
            "reasoning": "Failed to parse model response",
        }

    async def execute(self, task: str) -> TaskResult:
        """
        Execute routing analysis on a task.

        Args:
            task: Task description

        Returns:
            TaskResult with analysis output
        """
        try:
            analysis = await self.analyze(task)

            output = (
                f"Task Analysis:\n"
                f"  Type: {analysis.task_type.value}\n"
                f"  Complexity: {analysis.complexity.value}\n"
                f"  Recommended Tier: {analysis.recommended_tier}\n"
                f"  Confidence: {analysis.confidence:.0%}\n"
                f"  Reasoning: {analysis.reasoning}\n"
            )

            if analysis.subtasks:
                output += f"\nSubtasks ({len(analysis.subtasks)}):\n"
                for i, subtask in enumerate(analysis.subtasks, 1):
                    output += f"  {i}. {subtask.get('description', 'N/A')} "
                    output += f"[{subtask.get('type', '?')}, {subtask.get('complexity', '?')}]\n"

            return TaskResult(
                success=True,
                output=output,
                metadata={
                    "task_type": analysis.task_type.value,
                    "complexity": analysis.complexity.value,
                    "recommended_tier": analysis.recommended_tier,
                    "confidence": analysis.confidence,
                    "subtasks": analysis.subtasks,
                },
            )

        except Exception as e:
            return TaskResult(
                success=False,
                output="",
                error=str(e),
            )

    async def classify_type(self, task: str) -> TaskType:
        """
        Quickly classify task type only.

        Args:
            task: Task description

        Returns:
            TaskType enum value
        """
        prompt = f"""Classify this task into ONE category: code, research, writing, or reasoning.
Task: {task}

Output ONLY the category name (one word)."""

        self._add_user_message(prompt)
        response = await self._call_model(temperature=0.1, max_tokens=20)
        self._add_assistant_message(response)

        response = response.strip().lower()
        try:
            return TaskType(response)
        except ValueError:
            return TaskType.UNKNOWN

    async def estimate_complexity(self, task: str) -> Complexity:
        """
        Quickly estimate task complexity only.

        Args:
            task: Task description

        Returns:
            Complexity enum value
        """
        prompt = f"""Estimate the complexity of this task: {task}

Output ONE word: simple, moderate, or complex.
- simple: Single step, straightforward
- moderate: Multiple steps
- complex: Many steps, high complexity"""

        self._add_user_message(prompt)
        response = await self._call_model(temperature=0.1, max_tokens=20)
        self._add_assistant_message(response)

        response = response.strip().lower()
        try:
            return Complexity(response)
        except ValueError:
            return Complexity.UNKNOWN

    def get_tier_for_task(
        self,
        task_type: TaskType,
        complexity: Complexity,
    ) -> str:
        """
        Get recommended tier based on task type and complexity.

        Args:
            task_type: Type of task
            complexity: Complexity level

        Returns:
            Tier string (T1-T4)
        """
        # Base tier by task type
        type_to_tier = {
            TaskType.CODE: "T3",
            TaskType.RESEARCH: "T2",
            TaskType.WRITING: "T2",
            TaskType.REASONING: "T4",
            TaskType.UNKNOWN: "T2",
        }

        base_tier = type_to_tier.get(task_type, "T2")

        # Adjust by complexity
        tier_order = ["T1", "T2", "T3", "T4"]
        tier_index = tier_order.index(base_tier)

        if complexity == Complexity.SIMPLE:
            tier_index = max(0, tier_index - 1)
        elif complexity == Complexity.COMPLEX:
            tier_index = min(3, tier_index + 1)

        return tier_order[tier_index]
