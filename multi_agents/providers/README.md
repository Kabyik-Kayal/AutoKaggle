# Multi-Provider LLM Abstraction

This directory contains the multi-provider abstraction layer for AutoKaggle, enabling seamless integration with multiple LLM providers including OpenAI and Anthropic.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  High-Level Code (Agents, SOP, Framework, etc.)│
│              (No changes needed)                 │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
        ┌─────────────────────┐
        │  Provider Factory   │
        │  (Detects provider  │
        │   based on model)   │
        └────────┬────────────┘
                 │
        ┌────────┴──────────┐
        ↓                   ↓
   ┌─────────────┐    ┌──────────────┐
   │ OpenAI      │    │ Anthropic    │
   │ Provider    │    │ Provider     │
   └─────────────┘    └──────────────┘
        ↓                   ↓
    OpenAI API          Anthropic API
```

## Files Overview

### Core Provider Classes
- **base.py**: Abstract base classes defining provider interface
  - `LLMProvider`: Abstract LLM provider interface
  - `EmbeddingProvider`: Abstract embedding provider interface
  - `ProviderSettings`: Universal settings object

- **openai_provider.py**: OpenAI implementation
  - `OpenAILLMProvider`: Handles GPT models (gpt-4o, gpt-4o-mini, etc.)
  - `OpenAIEmbeddingProvider`: Handles text embeddings

- **anthropic_provider.py**: Anthropic implementation
  - `AnthropicLLMProvider`: Handles Claude models
  - `AnthropicEmbeddingProvider`: Placeholder (Anthropic doesn't offer embeddings yet)

### Support Classes
- **factory.py**: Provider factory and credential management
  - `ProviderFactory`: Creates providers based on model name
  - `load_api_credentials()`: Loads credentials from api_key.txt

- **settings_adapter.py**: Settings conversion utilities
  - `SettingsAdapter`: Converts universal settings to provider-specific formats

## Supported Models

### OpenAI
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-4
- gpt-3.5-turbo
- o1-mini
- o1-preview

### Anthropic
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307
- claude-2.1
- claude-2

## Usage Examples

### Basic Text Generation

```python
from multi_agents.providers import ProviderFactory, ProviderSettings

# Get provider (automatically selected based on model name)
provider = ProviderFactory.get_llm_provider('gpt-4o')

# Create settings
settings = ProviderSettings(max_tokens=1000, temperature=0.7)

# Generate text
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
response = provider.generate(messages, settings)
print(response)
```

### Using Anthropic

```python
# Same code, just different model
provider = ProviderFactory.get_llm_provider('claude-3-opus-20240229')
response = provider.generate(messages, settings)
```

### Image Processing

```python
# OpenAI vision
provider = ProviderFactory.get_llm_provider('gpt-4o')
response = provider.process_image(
    prompt="What's in this image?",
    image_base64="iVBORw0KGgo...",  # base64 encoded image
    image_type="image/png"
)

# Anthropic vision (same interface)
provider = ProviderFactory.get_llm_provider('claude-3-opus-20240229')
response = provider.process_image(
    prompt="What's in this image?",
    image_base64="iVBORw0KGgo...",
    image_type="image/png"
)
```

### Embeddings

```python
# Get embedding provider (currently OpenAI)
embedding_provider = ProviderFactory.get_embedding_provider()

# Generate embedding
embedding = embedding_provider.encode("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")

# Count tokens
num_tokens = embedding_provider.num_tokens("Hello, world!")
print(f"Tokens: {num_tokens}")
```

## Configuration

### API Credentials (api_key.txt)

```
# Line 1: OpenAI API key
sk-proj-your-openai-key

# Line 2: OpenAI base URL (optional)
https://api.openai.com/v1

# Line 3+: Anthropic API key
sk-ant-your-anthropic-key
```

Both providers are optional. At least one must be configured.

### Model Mapping (config.json)

```json
{
  "agent_model_mapping": {
    "Reader": "gpt-4o-mini",
    "Planner": "gpt-4o",
    "Developer": "gpt-4o",
    "Reviewer": "gpt-4o-mini",
    "Summarizer": "gpt-4o-mini"
  }
}
```

Edit this to use different models for each agent.

## Cost Optimization

### Model Comparison (per 1M tokens)

| Model | Input Cost | Output Cost | Speed | Quality |
|-------|-----------|------------|-------|---------|
| gpt-4o | $5 | $15 | Fast | Excellent |
| gpt-4o-mini | $0.15 | $0.60 | Very Fast | Good |
| claude-3-opus | $15 | $75 | Moderate | Excellent |
| claude-3-sonnet | $3 | $15 | Fast | Very Good |
| claude-3-haiku | $0.80 | $4 | Very Fast | Good |

### Cost-Optimized Configuration

```json
{
  "agent_model_mapping": {
    "Reader": "claude-3-haiku",
    "Planner": "claude-3-sonnet",
    "Developer": "gpt-4o",
    "Reviewer": "claude-3-haiku",
    "Summarizer": "claude-3-haiku"
  }
}
```

**Expected savings: 40-60% cost reduction**

## Error Handling

The providers include built-in error handling:

- **Automatic retries**: Up to 5 attempts with exponential backoff
- **Rate limiting**: Handles 429 errors gracefully
- **API errors**: Specific handling for each provider
- **Message truncation**: Automatically truncates long messages (OpenAI)

## Testing

```bash
# Run provider tests
python -m pytest tests/test_providers.py

# Test specific provider
python -m pytest tests/test_providers.py -k openai

# Validate configuration
python -c "from multi_agents.providers import ProviderFactory; ProviderFactory.validate_configuration()"
```

## Troubleshooting

### Issue: "API key file not found"
**Solution**: Create `api_key.txt` in project root with your API keys.

### Issue: "Model not found in provider mapping"
**Solution**: Add the model to `MODEL_PROVIDER_MAP` in `factory.py`

### Issue: Rate limiting errors
**Solution**: 
- Use cheaper models (claude-3-haiku instead of gpt-4o)
- Increase retry_delay in provider implementation
- Implement request queuing

### Issue: Different outputs between providers
**Solution**: This is expected - different models have different strengths. Tune settings and prompts per provider.

### Issue: "Embeddings not available"
**Solution**: This is normal when using Anthropic. ChromaDB automatically falls back to OpenAI embeddings.

## Design Decisions

### Provider Pattern
Each provider implements the same interface, allowing high-level code to remain provider-agnostic.

### Factory Pattern
Providers are created by a factory that automatically selects based on model name and caches instances.

### Settings Adapter
Universal settings are mapped to provider-specific parameters, abstracting API differences.

### Lazy Loading
Providers are created only when needed and cached for reuse.

## Future Enhancements

- [ ] Add Cohere provider
- [ ] Add Mistral provider
- [ ] Add local LLM support (Ollama, LM Studio)
- [ ] Add Anthropic embeddings when available
- [ ] Add structured output support
- [ ] Add streaming support
- [ ] Add function calling for multiple providers

## References

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Provider Factory Pattern](https://en.wikipedia.org/wiki/Factory_method_pattern)
