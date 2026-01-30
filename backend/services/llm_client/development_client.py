"""
Development LLM Client Adapter

Mock implementation for development and testing purposes.
"""

from backend.services.llm_client.base import LLMClient, LLMResponse


class DevelopmentLLMClient(LLMClient):
    """
    Development mode LLM client that does not make any real API calls.
    
    Returns mock responses suitable for testing and development.
    """

    def is_configured(self) -> bool:
        """
        Check if the client is properly configured.
        
        Returns:
            Always True for development client.
        """
        return True

    def get_model_name(self) -> str:
        """
        Return the model name being used.
        
        Returns:
            "development-mode" identifier.
        """
        return "development-mode"

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Generate a mock response.
        
        Args:
            prompt: The input prompt (ignored in development mode).
            max_tokens: Maximum number of tokens (ignored in development mode).
            temperature: Sampling temperature (ignored in development mode).
            
        Returns:
            LLMResponse containing mock feedback data.
        """
        # Return a valid JSON string matching the requested schema
        content = (
            '{\n'
            '  "items": [\n'
            '    {\n'
            '      "title": "Mocked Response",\n'
            '      "message": "This is a mocked response for development purposes.",\n'
            '      "type": "explanation",\n'
            '      "priority": "low",\n'
            '      "code_range": {\n'
            '        "start": {"line": 1, "character": 0},\n'
            '        "end": {"line": 2, "character": 0}\n'
            '      },\n'
            '      "confidence": 1.0,\n'
            '      "dismissible": true,\n'
            '      "actionable": false,\n'
            '      "action_label": null\n'
            '    }\n'
            '  ]\n'
            '}'
        )
        return LLMResponse(
            content=content,
            model="development-mode",
            usage={},
            latency_ms=0,
            success=True,
        )
