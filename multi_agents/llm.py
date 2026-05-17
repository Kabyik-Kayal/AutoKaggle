"""
Language Model abstraction layer supporting multiple providers.
Provides unified interface for LLM generation and embeddings.
"""

import os
import sys
import logging

sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Tuple

from .providers import ProviderFactory, ProviderSettings

logger = logging.getLogger(__name__)


class LLM:
    """
    Language Model abstraction using provider pattern.
    Supports multiple providers (OpenAI, Anthropic, etc.)
    """
    
    def __init__(self, model: str, type: str = 'api'):
        """
        Initialize LLM with specified model.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'claude-3-opus')
            type: 'api' for API-based (only option currently)
            
        Raises:
            ValueError: If type is not 'api' or model not recognized
        """
        if type == 'api':
            self.provider = ProviderFactory.get_llm_provider(model)
            self.model = model
        elif type == 'local':
            raise NotImplementedError("Local models not yet supported")
        else:
            raise ValueError(f"Unknown LLM type: {type}")
        
        logger.info(f'LLM initialized with model: {model}')
    
    def generate(self, prompt: str, history: list = None, 
                max_completion_tokens: int = 4096) -> Tuple[str, list]:
        """
        Generate response using the provider.
        
        Args:
            prompt: User prompt
            history: Conversation history (list of message dicts)
            max_completion_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (response_text, updated_history)
        """
        if history is None:
            history = []
        
        # Build message list
        messages = history + [{'role': 'user', 'content': prompt}]
        
        # Create settings
        settings = ProviderSettings(max_tokens=max_completion_tokens)
        
        # Generate using provider
        reply = self.provider.generate(messages, settings)
        
        # Update history
        history.append({'role': 'user', 'content': prompt})
        history.append({'role': 'assistant', 'content': reply})
        
        return reply, history


class Embeddings:
    """
    Embeddings abstraction (currently OpenAI only).
    ChromaDB uses this for semantic search.
    """
    
    def __init__(self):
        """Initialize embeddings provider"""
        self.provider = ProviderFactory.get_embedding_provider()
        logger.info(f'Embeddings provider initialized: {self.provider.model}')
    
    def encode(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self.provider.encode(text)
    
    def num_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.provider.num_tokens(text)


# Backward compatibility alias
OpenaiEmbeddings = Embeddings
