"""Base agent class for HenryCLI."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..lmstudio import ChatMessage, LMStudioClient


@dataclass
class AgentState:
    """State of an agent."""

    agent_id: str
    model: str
    task: str = ""
    conversation_history: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a task execution."""

    success: bool
    output: str
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality for model interaction and state management.
    """

    def __init__(
        self,
        agent_id: str,
        model: str,
        client: LMStudioClient,
        system_prompt: str = "",
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique agent identifier
            model: Model ID to use
            client: LM Studio client
            system_prompt: System prompt for the agent
        """
        self.agent_id = agent_id
        self.model = model
        self.client = client
        self.system_prompt = system_prompt
        self.state = AgentState(agent_id=agent_id, model=model)

    def _get_messages(self) -> list[ChatMessage]:
        """Get messages for API call including system prompt."""
        messages = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        messages.extend(self.state.conversation_history)
        return messages

    async def _call_model(
        self,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """
        Call the model with current conversation history.

        Args:
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Model response content
        """
        response = await self.client.chat_completion(
            model=self.model,
            messages=self._get_messages(),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _add_user_message(self, content: str) -> None:
        """Add a user message to conversation history."""
        self.state.conversation_history.append(
            ChatMessage(role="user", content=content)
        )

    def _add_assistant_message(self, content: str) -> None:
        """Add an assistant message to conversation history."""
        self.state.conversation_history.append(
            ChatMessage(role="assistant", content=content)
        )

    @abstractmethod
    async def execute(self, task: str) -> TaskResult:
        """
        Execute a task.

        Args:
            task: Task description

        Returns:
            TaskResult with output and artifacts
        """
        pass

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.state.conversation_history = []

    def get_history(self) -> list[ChatMessage]:
        """Get conversation history."""
        return self.state.conversation_history.copy()
