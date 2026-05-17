"""
Abstract base classes for LLM and Embedding providers.
Defines the interface that all providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class ProviderSettings:
    """Universal settings that map to provider-specific parameters"""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None
    
    @property
    def timeout(self) -> int:
        """Calculate timeout based on token count"""
        return (self.max_tokens // 1000 + 1) * 30


class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, model: str):
        """
        Initialize provider with model name.
        
        Args:
            model: Model identifier (e.g., 'gpt-4o', 'claude-3-opus')
        """
        self.model = model
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], 
                settings: ProviderSettings) -> str:
        """
        Generate text response from messages.
        
        Args:
            messages: Conversation history in OpenAI message format
                     [{"role": "user"/"assistant"/"system", "content": "..."}, ...]
            settings: Provider-agnostic settings
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If API call fails after retries
        """
        pass
    
    @abstractmethod
    def process_image(self, prompt: str, image_base64: str, 
                     image_type: str = "image/png") -> str:
        """
        Process image with prompt (for vision capabilities).
        
        Args:
            prompt: Text prompt about the image
            image_base64: Base64 encoded image
            image_type: MIME type of image (image/png, image/jpeg, etc.)
            
        Returns:
            Analysis/description of image
            
        Raises:
            NotImplementedError: If provider doesn't support vision
        """
        pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, model: str):
        """
        Initialize embedding provider.
        
        Args:
            model: Model identifier for embeddings
        """
        self.model = model
    
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def num_tokens(self, text: str) -> int:
        """
        Count approximate number of tokens in text.
        
        Args:
            text: Input text to count
            
        Returns:
            Number of tokens
        """
        pass
