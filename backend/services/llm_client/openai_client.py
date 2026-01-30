"""
OpenAI LLM Client Adapter

Implementation of LLM client for OpenAI API.
"""

import os
import time
from typing import Optional, Dict, Any

from openai import AsyncOpenAI

from backend.services.llm_client.base import LLMClient, LLMResponse
from backend.services.logger_service import get_logger


class OpenAILLMClient(LLMClient):
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
        """
        Initialize the OpenAI client adapter.
        
        Args:
            api_key: OpenAI API key. If None, will try to read from OPENAI_API_KEY env var.
            model: Model name to use (default: gpt-4o-mini).
            base_url: Optional base URL for API calls (for proxies or custom endpoints).
        """
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
            logger = get_logger()
            logger.system(
                "openai_package_not_installed",
                {},
                level="WARNING",
            )
            self._client = None

    def is_configured(self) -> bool:
        """
        Check if the client is properly configured.
        
        Returns:
            True if client is initialized and has API key, False otherwise.
        """
        return self._client is not None and self._api_key is not None

    def get_model_name(self) -> str:
        """
        Return the model name being used.
        
        Returns:
            String identifier of the OpenAI model.
        """
        return self._model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Generate a response from the OpenAI LLM.
        
        Args:
            prompt: The input prompt for the LLM.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 to 1.0).
            
        Returns:
            LLMResponse containing the generated content and metadata.
        """
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
