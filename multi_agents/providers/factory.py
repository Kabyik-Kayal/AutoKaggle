"""
Provider factory for selecting and creating the appropriate LLM provider.
"""

import os
import logging
from typing import Tuple, Optional
from dotenv import load_dotenv

from .base import LLMProvider, EmbeddingProvider, ProviderSettings
from .openai_provider import OpenAILLMProvider, OpenAIEmbeddingProvider
from .anthropic_provider import AnthropicLLMProvider, AnthropicEmbeddingProvider

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def load_api_credentials() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Load API credentials from environment variables or .env file.
    
    Environment variables (in priority order):
    1. OPENAI_API_KEY - OpenAI API key (starts with sk-proj- or sk-)
    2. OPENAI_BASE_URL - OpenAI base URL (optional, defaults to API endpoint)
    3. ANTHROPIC_API_KEY - Anthropic API key (starts with sk-ant-)
    
    Fallback: Reads from api_key.txt if .env not found (backward compatibility)
    
    Returns:
        Tuple of (openai_key, openai_base_url, anthropic_key)
        
    Raises:
        ValueError: If no API keys are configured
    """
    # First, try to load from environment variables
    openai_key = os.getenv('OPENAI_API_KEY')
    openai_base_url = os.getenv('OPENAI_BASE_URL')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    # If keys found via environment variables, return them
    if openai_key or anthropic_key:
        logger.info("✓ API credentials loaded from environment variables")
        return openai_key, openai_base_url, anthropic_key
    
    # Fallback: Try to load from api_key.txt for backward compatibility
    api_key_file = 'api_key.txt'
    if os.path.exists(api_key_file):
        try:
            logger.warning("⚠ Loading credentials from api_key.txt (legacy method). "
                          "Consider using .env file for better security.")
            
            with open(api_key_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Parse credentials from lines
            for line in lines:
                # Skip comments
                if line.startswith('#'):
                    continue
                
                # OpenAI base URL (contains https)
                if 'https://' in line:
                    openai_base_url = line
                # Anthropic key (starts with sk-ant-)
                elif 'sk-ant-' in line:
                    anthropic_key = line.split()[-1]
                # OpenAI key (starts with sk-)
                elif line.startswith('sk-proj-') or line.startswith('sk-'):
                    openai_key = line.split()[-1]
            
            if openai_key or anthropic_key:
                return openai_key, openai_base_url, anthropic_key
        
        except Exception as e:
            logger.error(f"Error reading api_key.txt: {e}")
    
    # No credentials found
    raise ValueError(
        "No API credentials configured. Please set up one of:\n"
        "  1. Environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY\n"
        "  2. Create .env file in project root with credentials\n"
        "  3. Legacy: api_key.txt (not recommended for security reasons)\n"
        "\nSee .env.template for .env file format."
        
    except FileNotFoundError as e:
        raise ValueError(
            f"API key file not found. Create 'api_key.txt' with credentials:\n"
            f"sk-your-openai-key\n"
            f"https://api.openai.com/v1  # optional\n"
            f"sk-ant-your-anthropic-key  # optional"
        ) from e


class ProviderFactory:
    """Factory for creating and caching LLM providers"""
    
    # Model to provider mapping
    MODEL_PROVIDER_MAP = {
        # OpenAI models
        'gpt-4o': 'openai',
        'gpt-4o-mini': 'openai',
        'gpt-4-turbo': 'openai',
        'gpt-4': 'openai',
        'gpt-3.5-turbo': 'openai',
        'o1-mini': 'openai',
        'o1-preview': 'openai',
        
        # Anthropic models (latest 2025 versions)
        # 'claude-opus-4-7': 'anthropic',          # NOTE: Disabled due to high API costs; use claude-sonnet-4-6 instead
        'claude-sonnet-4-6': 'anthropic',         # Best balance of speed and intelligence (recommended)
        'claude-haiku-4-5-20251001': 'anthropic', # Fastest with near-frontier intelligence
        
        # Anthropic models (legacy)
        'claude-3-opus': 'anthropic',
        'claude-3-opus-20240229': 'anthropic',
        'claude-3-sonnet': 'anthropic',
        'claude-3-sonnet-20240229': 'anthropic',
        'claude-3-haiku': 'anthropic',
        'claude-3-haiku-20240307': 'anthropic',
        'claude-2.1': 'anthropic',
        'claude-2': 'anthropic',
        'claude-instant-1.2': 'anthropic',
    }
    
    # Cache providers to avoid recreating them
    _llm_provider_cache = {}
    _embedding_provider_cache = {}
    _credentials_cache = None
    
    @classmethod
    def get_llm_provider(cls, model: str) -> LLMProvider:
        """
        Get LLM provider for given model name.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'claude-3-opus')
            
        Returns:
            LLMProvider instance appropriate for the model
            
        Raises:
            ValueError: If model not recognized or credentials missing
        """
        
        # Check cache first
        if model in cls._llm_provider_cache:
            return cls._llm_provider_cache[model]
        
        # Determine which provider this model belongs to
        provider_name = cls.MODEL_PROVIDER_MAP.get(model, 'openai')
        
        # Load credentials
        openai_key, openai_url, anthropic_key = cls._load_credentials()
        
        # Create appropriate provider
        if provider_name == 'anthropic':
            if not anthropic_key:
                raise ValueError(
                    f"Model '{model}' requires Anthropic API key.\n"
                    f"Add your Anthropic key to api_key.txt:\n"
                    f"sk-ant-your-key-here"
                )
            logger.info(f"Creating Anthropic provider for model: {model}")
            provider = AnthropicLLMProvider(model, anthropic_key)
            
        else:  # Default to OpenAI
            if not openai_key:
                raise ValueError(
                    f"Model '{model}' requires OpenAI API key.\n"
                    f"Add your OpenAI key to api_key.txt:\n"
                    f"sk-your-key-here"
                )
            logger.info(f"Creating OpenAI provider for model: {model}")
            provider = OpenAILLMProvider(model, openai_key, openai_url)
        
        # Cache and return
        cls._llm_provider_cache[model] = provider
        return provider
    
    @classmethod
    def get_embedding_provider(cls) -> EmbeddingProvider:
        """
        Get embedding provider (currently OpenAI only).
        
        Returns:
            EmbeddingProvider instance
            
        Raises:
            ValueError: If OpenAI API key not configured
        """
        
        cache_key = 'embedding'
        if cache_key in cls._embedding_provider_cache:
            return cls._embedding_provider_cache[cache_key]
        
        # Load credentials
        openai_key, openai_url, _ = cls._load_credentials()
        
        if not openai_key:
            raise ValueError(
                "OpenAI API key required for embeddings.\n"
                "Add to api_key.txt:\n"
                "sk-your-openai-key"
            )
        
        logger.info("Creating OpenAI embedding provider")
        provider = OpenAIEmbeddingProvider(openai_key, openai_url)
        cls._embedding_provider_cache[cache_key] = provider
        return provider
    
    @classmethod
    def _load_credentials(cls) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Load and cache credentials.
        
        Returns:
            Tuple of (openai_key, openai_base_url, anthropic_key)
        """
        if cls._credentials_cache is not None:
            return cls._credentials_cache
        
        cls._credentials_cache = load_api_credentials()
        return cls._credentials_cache
    
    @classmethod
    def validate_configuration(cls) -> None:
        """
        Validate that at least one provider is configured.
        
        Raises:
            ValueError: If no providers configured
        """
        try:
            openai_key, _, anthropic_key = cls._load_credentials()
            
            if not openai_key and not anthropic_key:
                raise ValueError(
                    "No API keys configured in api_key.txt.\n"
                    "Add at least one:\n"
                    "- OpenAI: sk-...\n"
                    "- Anthropic: sk-ant-..."
                )
            
            if openai_key:
                logger.info("✓ OpenAI API configured")
            if anthropic_key:
                logger.info("✓ Anthropic API configured")
                
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear provider cache (useful for testing)"""
        cls._llm_provider_cache.clear()
        cls._embedding_provider_cache.clear()
        cls._credentials_cache = None
