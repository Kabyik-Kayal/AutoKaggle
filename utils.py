import os
from api_handler import APIHandler, APISettings
import base64
import subprocess
import re
import shutil
import json
import logging

DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX_STRONG_BASELINE = f'{DIR}/strong_baseline/competition'
PREFIX_WEAK_BASELINE = f'{DIR}/weak_baseline/competition'
PREFIX_MULTI_AGENTS = f'{DIR}/multi_agents'
SEPERATOR_TEMPLATE = '-----------------------------------{step_name}-----------------------------------'

logger = logging.getLogger(__name__)

def load_config(file_path: str):
    assert file_path.endswith('json'), "The configuration file should be in JSON format."
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def read_file(file_path: str):
    """
    Read the content of a file and return it as a string.
    """
    if file_path.endswith('txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    if file_path.endswith('csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    
def multi_chat(api_handler: APIHandler, prompt, history=None, max_completion_tokens=4096):
    """
    Multi-round chat with the assistant.
    
    Backward compatibility wrapper - uses APIHandler if provided.
    For new code, use the LLM class directly with provider abstraction.
    """
    if history is None:
        history = []

    messages = history + [{'role': 'user', 'content': prompt}]

    settings = APISettings(max_completion_tokens=max_completion_tokens)
    reply = api_handler.get_output(messages=messages, settings=settings)
    history.append({'role': 'user', 'content': prompt})
    history.append({'role': 'assistant', 'content': reply})
    
    return reply, history

def read_image(prompt, image_path, model: str = 'gpt-4o'):
    """
    Read and analyze image with specified model.
    
    Args:
        prompt: Text prompt about the image
        image_path: Path to image file
        model: Model to use for analysis (defaults to gpt-4o)
        
    Returns:
        Analysis/description of the image
    """
    def encode_image(image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Determine image type from file extension
    image_type = 'image/png'
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        image_type = 'image/jpeg'
    elif image_path.lower().endswith('.gif'):
        image_type = 'image/gif'
    elif image_path.lower().endswith('.webp'):
        image_type = 'image/webp'
    
    # Get provider and process image
    try:
        from multi_agents.providers import ProviderFactory
        provider = ProviderFactory.get_llm_provider(model)
        reply = provider.process_image(prompt, base64_image, image_type)
        return reply
    except Exception as e:
        logger.error(f"Image processing failed with {model}: {e}")
        # Fallback to APIHandler for backward compatibility
        api_handler = APIHandler(model)
        messages = [
            {"role": "system", "content": "You are a professional data analyst."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64_image}"}}
                ]
            }
        ]
        settings = APISettings(max_completion_tokens=4096)
        reply = api_handler.get_output(messages=messages, settings=settings, response_type='image')
        return reply