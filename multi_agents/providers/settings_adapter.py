"""
Settings adapter for converting universal settings to provider-specific formats.
"""

from typing import Dict, Any
from .base import ProviderSettings


class SettingsAdapter:
    """Adapts universal settings to provider-specific formats"""
    
    @staticmethod
    def to_openai(settings: ProviderSettings) -> Dict[str, Any]:
        """
        Convert universal settings to OpenAI API parameters.
        
        Args:
            settings: Universal provider settings
            
        Returns:
            Dictionary of OpenAI-specific parameters
        """
        return {
            'temperature': settings.temperature,
            'max_completion_tokens': settings.max_tokens,
            'top_p': settings.top_p,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stop': settings.stop_sequences,
            'timeout': settings.timeout,
        }
    
    @staticmethod
    def to_anthropic(settings: ProviderSettings) -> Dict[str, Any]:
        """
        Convert universal settings to Anthropic API parameters.
        
        Args:
            settings: Universal provider settings
            
        Returns:
            Dictionary of Anthropic-specific parameters
        """
        return {
            'temperature': settings.temperature,
            'max_tokens': settings.max_tokens,
            'top_p': settings.top_p,
            'top_k': 10,  # Anthropic-specific parameter
            'stop_sequences': settings.stop_sequences or [],
        }
    
    @staticmethod
    def get_timeout(settings: ProviderSettings) -> int:
        """
        Get appropriate timeout for requests.
        
        Args:
            settings: Provider settings
            
        Returns:
            Timeout in seconds
        """
        return settings.timeout
