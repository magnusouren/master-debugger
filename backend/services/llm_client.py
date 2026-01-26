"""
LLM Client Interface and Implementations

Provides an abstract interface for LLM calls.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from openai import AsyncOpenAI



@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    usage: Dict[str, int]  # tokens used
    latency_ms: float
    success: bool
    error: Optional[str] = None


class LLMClient(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the client is properly configured."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name being used."""
        pass


class OpenAIClient(LLMClient):
    """
    OpenAI API client implementation.
    
    Requires: pip install openai>=1.0.0
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model
        self._base_url = base_url
        self._client: Optional[Any] = None

        if self._api_key:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        try:

            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncOpenAI(**kwargs)
        except ImportError:
            print("[OpenAIClient] openai package not installed")
            self._client = None

    def is_configured(self) -> bool:
        return self._client is not None and self._api_key is not None

    def get_model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        import time

        if not self.is_configured():
            return LLMResponse(
                content="",
                model=self._model,
                usage={},
                latency_ms=0,
                success=False,
                error="OpenAI client not configured",
            )

        start = time.time()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant that provides concise feedback. Hints should increase learning without giving away full solutions."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            return LLMResponse(
                content=content,
                model=self._model,
                usage=usage,
                latency_ms=(time.time() - start) * 1000,
                success=True,
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self._model,
                usage={},
                latency_ms=(time.time() - start) * 1000,
                success=False,
                error=str(e),
            )


def create_llm_client(
    provider: Optional[str] = None, # from config: llm_provider
    api_key: Optional[str] = None, # from config: llm_api_key
    model: Optional[str] = None, # from config: llm_model
) -> Optional[LLMClient]:
    """
    Factory function to create the appropriate LLM client.
    
    Args:
        provider: "openai" or None
        api_key: API key for the provider
        model: Model name to use
    
    Returns:
        Configured LLMClient instance or None if not configured
    """
    if provider == "openai":
        client = OpenAIClient(
            api_key=api_key,
            model=model or "gpt-4o-mini",
        )
        if client.is_configured():
            print(f"[LLMClient] OpenAI client configured with model: {client.get_model_name()}")
            return client
        else:
            print("[LLMClient] OpenAI not configured")
            return None

    print("[LLMClient] No provider specified, or unsupported provider. value:", provider)
    return None

