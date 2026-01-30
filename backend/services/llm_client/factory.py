"""
LLM Client Adapter Factory

Creates and configures LLM client adapters based on system configuration.
"""

from typing import Optional

from backend.services.llm_client.base import LLMClient
from backend.services.llm_client.openai_client import OpenAILLMClient
from backend.services.llm_client.development_client import DevelopmentLLMClient
from backend.services.logger_service import get_logger


def create_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[LLMClient]:
    """
    Create an LLM client adapter based on configuration.
    
    Args:
        provider: LLM provider name ("openai", "development", or None).
        api_key: API key for the provider.
        model: Model name to use.
    
    Returns:
        Configured LLMClient instance or None if not configured/unsupported.
        
    Raises:
        None - returns None for invalid configurations instead of raising.
    """
    logger = get_logger()
    
    if provider == "openai":
        client = OpenAILLMClient(
            api_key=api_key,
            model=model or "gpt-4o-mini",
        )
        if client.is_configured():
            logger.system(
                "llm_client_created",
                {"provider": "openai", "model": client.get_model_name()},
                level="DEBUG",
            )
            return client
        else:
            logger.system(
                "llm_client_not_configured",
                {"provider": "openai"},
                level="WARNING",
            )
            return None
    elif provider == "development":
        logger.system(
            "llm_client_development_mode",
            {"provider": "development"},
            level="INFO",
        )
        return DevelopmentLLMClient()

    logger.system(
        "llm_provider_unsupported",
        {"provider": provider},
        level="WARNING",
    )
    return None
