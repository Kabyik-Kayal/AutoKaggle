"""
Multi-provider LLM abstraction layer for AutoKaggle.

Supports:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude models)

Usage:
    from multi_agents.providers import ProviderFactory
    
    # Get a provider for a specific model
    provider = ProviderFactory.get_llm_provider('gpt-4o')
    
    # Or use Anthropic
    provider = ProviderFactory.get_llm_provider('claude-3-opus')
    
    # Generate text
    from multi_agents.providers import ProviderSettings
    settings = ProviderSettings(max_tokens=1000, temperature=0.7)
    response = provider.generate(messages, settings)
"""

from .base import LLMProvider, EmbeddingProvider, ProviderSettings
from .openai_provider import OpenAILLMProvider, OpenAIEmbeddingProvider
from .anthropic_provider import AnthropicLLMProvider, AnthropicEmbeddingProvider
from .factory import ProviderFactory, load_api_credentials
from .settings_adapter import SettingsAdapter

__all__ = [
    'LLMProvider',
    'EmbeddingProvider',
    'ProviderSettings',
    'OpenAILLMProvider',
    'OpenAIEmbeddingProvider',
    'AnthropicLLMProvider',
    'AnthropicEmbeddingProvider',
    'ProviderFactory',
    'SettingsAdapter',
    'load_api_credentials',
]

__version__ = '1.0.0'
