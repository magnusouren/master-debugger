"""
Base LLM Client Adapter Protocol

Defines the interface that all LLM client adapters must implement.
This allows the system to work with any LLM provider without direct
dependencies on specific SDKs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict


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
    """
    Abstract base class for LLM client adapters.
    
    This interface allows the system to communicate with
    different LLM providers (OpenAI, Anthropic, etc.) without
    knowing the specific implementation details.
    
    All adapters must implement this interface to ensure consistent
    behavior across different providers.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt for the LLM.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 to 1.0).
            
        Returns:
            LLMResponse containing the generated content and metadata.
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """
        Check if the client is properly configured.
        
        Returns:
            True if the client has all necessary configuration (API keys, etc.),
            False otherwise.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model name being used.
        
        Returns:
            String identifier of the model (e.g., "gpt-4o-mini").
        """
        pass
