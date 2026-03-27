"""Context manager with dual-stream architecture."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .filestore import FileStore


@dataclass
class SemanticStream:
    """
    Semantic stream - compressed conversation in LLM context.
    
    This contains the information needed for immediate reasoning,
    kept small enough to fit within context limits.
    """

    conversation_summary: str = ""
    current_step: str = ""
    next_steps: list[str] = field(default_factory=list)
    recent_messages: list[dict[str, str]] = field(default_factory=list)
    task_intent: str = ""
    key_decisions: list[str] = field(default_factory=list)
    artifacts_created: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_summary": self.conversation_summary,
            "current_step": self.current_step,
            "next_steps": self.next_steps,
            "task_intent": self.task_intent,
            "key_decisions": self.key_decisions,
            "artifacts_created": self.artifacts_created,
            "recent_messages": self.recent_messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticStream":
        """Create from dictionary."""
        return cls(
            conversation_summary=data.get("conversation_summary", ""),
            current_step=data.get("current_step", ""),
            next_steps=data.get("next_steps", []),
            task_intent=data.get("task_intent", ""),
            key_decisions=data.get("key_decisions", []),
            artifacts_created=data.get("artifacts_created", []),
            recent_messages=data.get("recent_messages", []),
        )


@dataclass
class RuntimeStream:
    """
    Runtime stream - full state in filesystem with file references.
    
    This contains the complete history and large artifacts,
    stored on disk and referenced by path.
    """

    full_history_path: str = ""
    artifacts: list[str] = field(default_factory=list)
    file_references: dict[str, str] = field(default_factory=dict)
    context_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "full_history_path": self.full_history_path,
            "artifacts": self.artifacts,
            "file_references": self.file_references,
            "context_metadata": self.context_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeStream":
        """Create from dictionary."""
        return cls(
            full_history_path=data.get("full_history_path", ""),
            artifacts=data.get("artifacts", []),
            file_references=data.get("file_references", {}),
            context_metadata=data.get("context_metadata", {}),
        )


@dataclass
class ContextState:
    """Complete context state for an agent."""

    agent_id: str
    model: str
    task: str
    semantic_stream: SemanticStream = field(default_factory=SemanticStream)
    runtime_stream: RuntimeStream = field(default_factory=RuntimeStream)
    context_usage: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pending_subtasks: list[dict[str, Any]] = field(default_factory=list)
    completed_subtasks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "task": self.task,
            "semantic_stream": self.semantic_stream.to_dict(),
            "runtime_stream": self.runtime_stream.to_dict(),
            "context_usage": self.context_usage,
            "timestamp": self.timestamp,
            "pending_subtasks": self.pending_subtasks,
            "completed_subtasks": self.completed_subtasks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextState":
        """Create from dictionary."""
        return cls(
            agent_id=data.get("agent_id", ""),
            model=data.get("model", ""),
            task=data.get("task", ""),
            semantic_stream=SemanticStream.from_dict(
                data.get("semantic_stream", {})
            ),
            runtime_stream=RuntimeStream.from_dict(
                data.get("runtime_stream", {})
            ),
            context_usage=data.get("context_usage", {}),
            timestamp=data.get("timestamp", ""),
            pending_subtasks=data.get("pending_subtasks", []),
            completed_subtasks=data.get("completed_subtasks", []),
        )


class ContextManager:
    """
    Manages dual-stream context for agents.
    
    The dual-stream architecture separates:
    - Semantic Stream: Compressed, in-context information
    - Runtime Stream: Full state stored on filesystem
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        compression_threshold: float = 0.85,
        offload_token_limit: int = 20000,
        keep_recent_messages: int = 10,
    ):
        """
        Initialize context manager.

        Args:
            base_dir: Base directory for context storage
            compression_threshold: Threshold (0-1) to trigger compression
            offload_token_limit: Token limit before offloading
            keep_recent_messages: Number of recent messages to keep in context
        """
        if base_dir is None:
            base_dir = Path.home() / ".henrycli" / "contexts"
        self.base_dir = base_dir
        self.compression_threshold = compression_threshold
        self.offload_token_limit = offload_token_limit
        self.keep_recent_messages = keep_recent_messages

        self.filestore = FileStore(base_dir.parent / "filestore")
        self.active_dir = self.base_dir / "active"
        self.completed_dir = self.base_dir / "completed"

        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)

        self._current_state: ContextState | None = None

    def create_context(
        self,
        agent_id: str,
        model: str,
        task: str,
    ) -> ContextState:
        """
        Create a new context for an agent.

        Args:
            agent_id: Unique agent identifier
            model: Model being used
            task: Task description

        Returns:
            New ContextState
        """
        state = ContextState(
            agent_id=agent_id,
            model=model,
            task=task,
            semantic_stream=SemanticStream(
                task_intent=task,
                current_step="Starting task",
            ),
        )
        self._current_state = state
        return state

    def get_current_state(self) -> ContextState | None:
        """Get the current context state."""
        return self._current_state

    async def save_state(
        self,
        state: ContextState | None = None,
    ) -> str:
        """
        Save context state to filesystem.

        Args:
            state: State to save (uses current if None)

        Returns:
            Path to saved context
        """
        if state is None:
            state = self._current_state
        if state is None:
            raise ValueError("No state to save")

        agent_dir = self.active_dir / state.agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Save full conversation history
        history_path = agent_dir / "history.json"
        history_path.write_text(
            json.dumps(
                state.semantic_stream.recent_messages,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # Update runtime stream
        state.runtime_stream.full_history_path = str(
            history_path.relative_to(self.base_dir)
        )

        # Save context state
        context_path = agent_dir / "context.json"
        context_path.write_text(
            json.dumps(state.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        return str(context_path)

    async def restore_state(self, agent_id: str) -> ContextState:
        """
        Restore context state from filesystem.

        Args:
            agent_id: Agent ID to restore

        Returns:
            Restored ContextState
        """
        context_path = self.active_dir / agent_id / "context.json"
        if not context_path.exists():
            raise FileNotFoundError(f"Context not found: {context_path}")

        data = json.loads(context_path.read_text(encoding="utf-8"))
        state = ContextState.from_dict(data)
        self._current_state = state
        return state

    def archive_context(self, agent_id: str) -> str:
        """
        Move context from active to completed.

        Args:
            agent_id: Agent ID to archive

        Returns:
            New path in completed directory
        """
        active_path = self.active_dir / agent_id
        completed_path = self.completed_dir / agent_id

        if active_path.exists():
            completed_path.mkdir(parents=True, exist_ok=True)
            for file in active_path.iterdir():
                if file.is_file():
                    file.rename(completed_path / file.name)
            active_path.rmdir()

        return str(completed_path)

    def update_semantic_stream(
        self,
        summary: str | None = None,
        current_step: str | None = None,
        next_steps: list[str] | None = None,
        key_decision: str | None = None,
        artifact: str | None = None,
    ) -> None:
        """
        Update the semantic stream.

        Args:
            summary: New conversation summary
            current_step: Current step description
            next_steps: List of next steps
            key_decision: A key decision to record
            artifact: An artifact that was created
        """
        if self._current_state is None:
            raise ValueError("No active context")

        stream = self._current_state.semantic_stream

        if summary:
            stream.conversation_summary = summary
        if current_step:
            stream.current_step = current_step
        if next_steps is not None:
            stream.next_steps = next_steps
        if key_decision:
            stream.key_decisions.append(key_decision)
        if artifact:
            stream.artifacts_created.append(artifact)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        if self._current_state is None:
            raise ValueError("No active context")

        stream = self._current_state.semantic_stream
        stream.recent_messages.append({"role": role, "content": content})

        # Trim to keep only recent messages
        if len(stream.recent_messages) > self.keep_recent_messages * 2:
            stream.recent_messages = stream.recent_messages[
                -self.keep_recent_messages :
            ]

    def offload_content(self, content: str, prefix: str = "") -> str:
        """
        Offload large content to filesystem.

        Args:
            content: Content to offload
            prefix: Optional prefix for filename

        Returns:
            File reference path
        """
        return self.filestore.offload(content, prefix)

    def get_file_preview(self, ref: str, lines: int = 10) -> str:
        """
        Get preview of offloaded content.

        Args:
            ref: File reference path
            lines: Number of lines to preview

        Returns:
            Preview of content
        """
        return self.filestore.load_preview(ref, lines)

    def load_offloaded_content(self, ref: str) -> str:
        """
        Load full offloaded content.

        Args:
            ref: File reference path

        Returns:
            Full content
        """
        return self.filestore.load(ref)

    def get_context_usage_ratio(self, tokens_used: int, tokens_limit: int) -> float:
        """
        Calculate context usage ratio.

        Args:
            tokens_used: Current tokens used
            tokens_limit: Maximum tokens allowed

        Returns:
            Usage ratio (0-1)
        """
        if tokens_limit == 0:
            return 0.0
        ratio = tokens_used / tokens_limit
        if self._current_state:
            self._current_state.context_usage = {
                "tokens_used": tokens_used,
                "tokens_limit": tokens_limit,
                "ratio": ratio,
            }
        return ratio

    def needs_compression(self, tokens_used: int, tokens_limit: int) -> bool:
        """
        Check if context needs compression.

        Args:
            tokens_used: Current tokens used
            tokens_limit: Maximum tokens allowed

        Returns:
            True if compression is needed
        """
        ratio = self.get_context_usage_ratio(tokens_used, tokens_limit)
        return ratio >= self.compression_threshold

    async def compress_context(
        self,
        summarizer_callable=None,
    ) -> str:
        """
        Compress the current context.

        Args:
            summarizer_callable: Async function to generate summary

        Returns:
            Summary text
        """
        if self._current_state is None:
            raise ValueError("No active context")

        # Save full history to filestore
        messages = self._current_state.semantic_stream.recent_messages
        history_ref = self.filestore.offload_json(
            messages, prefix="conversation_history"
        )
        self._current_state.runtime_stream.file_references[
            "conversation_history"
        ] = history_ref

        # Generate summary if callable provided
        summary = ""
        if summarizer_callable:
            summary = await summarizer_callable(messages)
            self._current_state.semantic_stream.conversation_summary = summary

        # Keep only recent messages
        self._current_state.semantic_stream.recent_messages = messages[
            -self.keep_recent_messages :
        ]

        return summary
