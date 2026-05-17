"""
Anthropic provider implementation for Claude models.
"""

import anthropic
import time
import logging
from typing import List, Dict, Any

from .base import LLMProvider, EmbeddingProvider, ProviderSettings
from .settings_adapter import SettingsAdapter

logger = logging.getLogger(__name__)


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude models provider (claude-3-opus, claude-3-sonnet, etc.)"""
    
    def __init__(self, model: str, api_key: str):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Model name (e.g., 'claude-3-opus-20240229')
            api_key: Anthropic API key
        """
        super().__init__(model)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_attempts = 5
        self.retry_delay = 30
    
    def generate(self, messages: List[Dict[str, str]], 
                settings: ProviderSettings) -> str:
        """Generate response using Anthropic API with retry logic"""
        
        adapted_settings = SettingsAdapter.to_anthropic(settings)
        
        # Separate system message from regular messages
        system_message = ""
        regular_messages = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_message = msg.get('content', '')
            else:
                regular_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', '')
                })
        
        for attempt in range(self.max_attempts):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=adapted_settings['max_tokens'],
                    system=system_message if system_message else None,
                    messages=regular_messages,
                    temperature=adapted_settings['temperature'],
                    top_p=adapted_settings['top_p'],
                    stop_sequences=adapted_settings.get('stop_sequences'),
                )
                
                if response.content and len(response.content) > 0:
                    if response.content[0].type == "text":
                        return response.content[0].text
                else:
                    return "Error: Wrong response format."
                    
            except anthropic.RateLimitError as e:
                if attempt < self.max_attempts - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{self.max_attempts}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max attempts reached. Rate limit error: {e}")
                    raise
                    
            except anthropic.APIError as e:
                error_str = str(e).lower()
                if "overloaded" in error_str or "temporarily unavailable" in error_str:
                    if attempt < self.max_attempts - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"API overloaded (attempt {attempt + 1}/{self.max_attempts}). "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max attempts reached. API error: {e}")
                        raise
                else:
                    logger.error(f"API Error: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt < self.max_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        return "Error: Max attempts reached"
    
    def process_image(self, prompt: str, image_base64: str,
                     image_type: str = "image/png") -> str:
        """
        Process image using Claude vision capabilities.
        
        Args:
            prompt: Text prompt about the image
            image_base64: Base64 encoded image
            image_type: MIME type of image
            
        Returns:
            Analysis of the image
        """
        # Map common image types to Anthropic media types
        media_type_map = {
            "image/png": "image/png",
            "image/jpeg": "image/jpeg",
            "image/jpg": "image/jpeg",
            "image/gif": "image/gif",
            "image/webp": "image/webp",
        }
        
        media_type = media_type_map.get(image_type, "image/png")
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        
        if message.content and len(message.content) > 0:
            return message.content[0].text
        else:
            return "Error: No response from image processing"


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """
    Placeholder for Anthropic embeddings.
    Anthropic doesn't currently offer embeddings API.
    Users should use OpenAI embeddings instead.
    """
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        """Initialize (note: Anthropic doesn't support embeddings yet)"""
        super().__init__(model)
        logger.warning(
            "⚠️ Anthropic embeddings not yet available. "
            "ChromaDB will use OpenAI embeddings instead. "
            "This is normal and working as intended."
        )
    
    def encode(self, text: str) -> List[float]:
        """
        Anthropic doesn't offer embeddings API yet.
        
        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "Anthropic embeddings API not yet available. "
            "ChromaDB will automatically fall back to OpenAI embeddings. "
            "If you see this error, check that OpenAI API key is configured."
        )
    
    def num_tokens(self, text: str) -> int:
        """
        Approximate token count for Claude models.
        Claude uses roughly 1 token per 4 characters.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Approximate token count
        """
        return max(1, len(text) // 4)
