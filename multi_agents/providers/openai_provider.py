"""
OpenAI provider implementation for GPT models.
"""

import openai
import httpx
import time
import logging
from typing import List, Dict, Any

from .base import LLMProvider, EmbeddingProvider, ProviderSettings
from .settings_adapter import SettingsAdapter

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    """OpenAI GPT models provider (gpt-4o, gpt-4o-mini, etc.)"""
    
    def __init__(self, model: str, api_key: str, base_url: str = None):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name (e.g., 'gpt-4o')
            api_key: OpenAI API key
            base_url: Optional custom base URL for OpenAI API
        """
        super().__init__(model)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(verify=False)
        )
        self.max_attempts = 5
        self.retry_delay = 30
    
    def generate(self, messages: List[Dict[str, str]], 
                settings: ProviderSettings) -> str:
        """Generate response using OpenAI API with retry logic"""
        
        adapted_settings = SettingsAdapter.to_openai(settings)
        
        # Special case: o1-mini models require temperature=1
        if 'o1' in self.model:
            adapted_settings['temperature'] = 1.0
        
        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    **adapted_settings
                )
                
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content
                else:
                    return "Error: Wrong response format."
                    
            except openai.BadRequestError as e:
                error_str = str(e).lower()
                if "string too long" in error_str or "context length" in error_str:
                    logger.warning(f"Message too long, truncating... (attempt {attempt + 1})")
                    messages = self._truncate_messages(messages)
                    continue
                else:
                    logger.error(f"BadRequestError: {e}")
                    raise
                    
            except (openai.RateLimitError, openai.APIError, 
                   openai.APIConnectionError, TimeoutError) as e:
                if attempt < self.max_attempts - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_attempts} failed. "
                        f"Retrying in {wait_time}s... Error: {type(e).__name__}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max attempts reached. Last error: {e}")
                    raise
        
        return "Error: Max attempts reached"
    
    def process_image(self, prompt: str, image_base64: str,
                     image_type: str = "image/png") -> str:
        """
        Process image using GPT-4 vision capabilities.
        
        Args:
            prompt: Text prompt about the image
            image_base64: Base64 encoded image
            image_type: MIME type of image
            
        Returns:
            Analysis of the image
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=4096,
            timeout=300
        )
        
        return response.choices[0].message.content
    
    def _truncate_messages(self, messages: List[Dict[str, str]], 
                          max_length: int = 100000) -> List[Dict[str, str]]:
        """
        Truncate message history to fit within length limits.
        
        Args:
            messages: Message history
            max_length: Maximum total character length
            
        Returns:
            Truncated message history
        """
        total_length = sum(len(m.get('content', '')) for m in messages)
        
        if total_length <= max_length:
            return messages
        
        # Keep all but last message
        truncated = messages[:-1]
        last_message = messages[-1]
        
        # Calculate available space for last message
        current_length = sum(len(m.get('content', '')) for m in truncated)
        available_length = max_length - current_length
        
        if available_length > 100:
            content = last_message.get('content', '')[:available_length-3] + "..."
            truncated.append({"role": last_message['role'], "content": content})
        
        return truncated


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding models provider (text-embedding-3-large, etc.)"""
    
    def __init__(self, api_key: str, base_url: str = None, 
                model: str = 'text-embedding-3-large'):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key
            base_url: Optional custom base URL
            model: Embedding model name
        """
        super().__init__(model)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def encode(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format='float'
        )
        return response.data[0].embedding
    
    def num_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding('cl100k_base')
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens with tiktoken: {e}. Using character estimate.")
            return len(text) // 4  # Rough estimate: 1 token per 4 characters
