"""LM Studio API client with OpenAI compatibility."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

import httpx
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""

    id: str
    model: str
    choices: list[Choice]
    usage: Optional[Usage] = None


class Choice(BaseModel):
    """A choice from chat completion."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "lmstudio"


class ModelList(BaseModel):
    """List of loaded models."""

    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=list)

    def model_ids(self) -> list[str]:
        """Get list of model IDs."""
        return [model.id for model in self.data]

    def has_model(self, model_id: str) -> bool:
        """Check if a model is loaded."""
        return model_id in self.model_ids()


class LMStudioClient:
    """Client for LM Studio API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_models(self) -> ModelList:
        """Get list of loaded models."""
        client = await self._get_client()
        response = await client.get("/v1/models")
        response.raise_for_status()
        return ModelList(**response.json())

    async def is_model_loaded(self, model_id: str) -> bool:
        """Check if a specific model is loaded."""
        models = await self.get_models()
        return models.has_model(model_id)

    async def chat_completion(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatCompletionResponse:
        """Get chat completion (non-streaming)."""
        client = await self._get_client()
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        response = await client.post(
            "/v1/chat/completions",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ChatCompletionResponse(**response.json())

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion."""
        client = await self._get_client()
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=request.model_dump(),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json

                        parsed = json.loads(data)
                        if parsed.get("choices"):
                            delta = parsed["choices"][0].get("delta", {})
                            if content := delta.get("content"):
                                yield content
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        """Check if LM Studio is available."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            # Fallback: try to get models
            try:
                await self.get_models()
                return True
            except httpx.HTTPError:
                return False

    async def load_model(
        self,
        model_key: str,
        gpu_layers: str | None = None,
        context_length: int | None = None,
        identifier: str | None = None,
    ) -> dict[str, Any]:
        """
        Load a model via LM Studio REST API.

        Args:
            model_key: Model identifier (e.g., "TheBloke/phi-3-mini-4k-instruct-GGUF")
            gpu_layers: GPU layers setting ("max", "auto", or "0.0-1.0")
            context_length: Context length
            identifier: Custom identifier for the loaded model

        Returns:
            Response with instance_id
        """
        client = await self._get_client()
        payload: dict[str, Any] = {"model_key": model_key}

        if gpu_layers:
            payload["gpu_layers"] = gpu_layers
        if context_length:
            payload["context_length"] = context_length
        if identifier:
            payload["identifier"] = identifier

        response = await client.post(
            "/api/v1/models/load",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def unload_model(self, instance_id: str) -> dict[str, Any]:
        """
        Unload a model via LM Studio REST API.

        Args:
            instance_id: Instance ID of the model to unload

        Returns:
            Response with unloaded instance_id
        """
        client = await self._get_client()
        response = await client.post(
            "/api/v1/models/unload",
            json={"instance_id": instance_id},
        )
        response.raise_for_status()
        return response.json()

    async def unload_all_models(self) -> list[dict[str, Any]]:
        """
        Unload all loaded models.

        Returns:
            List of unloaded instance IDs
        """
        models = await self.get_models()
        results = []
        for model in models.data:
            try:
                result = await self.unload_model(model.id)
                results.append(result)
            except Exception:
                pass
        return results

    async def download_model(self, model_key: str) -> dict[str, Any]:
        """
        Download a model via LM Studio CLI API.

        Args:
            model_key: Model identifier (e.g., "TheBloke/phi-3-mini-4k-instruct-GGUF")

        Returns:
            Download status
        """
        client = await self._get_client()
        response = await client.post(
            "/api/v1/models/download",
            json={"model_key": model_key},
        )
        response.raise_for_status()
        return response.json()

    async def list_local_models(self) -> list[dict[str, Any]]:
        """
        List all local models via LM Studio REST API.

        Returns:
            List of local models
        """
        client = await self._get_client()
        response = await client.get("/api/v1/models/local")
        response.raise_for_status()
        return response.json().get("models", [])
