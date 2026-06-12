"""
Unified LLM client interface for multiple providers.

This module provides a provider-agnostic interface for interacting with various
large language models (LLMs). It supports multiple backend providers including
Gemini, Claude, OpenAI, and Ollama with a consistent API.

The design isolates provider-specific logic and allows for easy addition of
new providers in the future.

Example:
    >>> from llm_client import initialize_client, generate_response
    >>> initialize_client(provider="gemini")
    >>> response = generate_response("What is machine learning?")
    >>> print(response)
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Global client instance
_client_instance: Optional['BaseProvider'] = None


# ============================================================================
# Provider Base Class
# ============================================================================

class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all provider implementations must follow.
    Subclasses should implement provider-specific initialization and request logic.
    """
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the base provider.
        
        Args:
            api_key: API key for the provider service
            model_name: Name of the model to use
            
        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError(f"API key cannot be empty for {self.__class__.__name__}")
        
        self.api_key = api_key
        self.model_name = model_name
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt to guide model behavior
            
        Returns:
            Generated response as a string
            
        Raises:
            RuntimeError: If API call fails
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of available model names
        """
        pass


# ============================================================================
# Gemini Provider Implementation
# ============================================================================

class GeminiProvider(BaseProvider):
    """
    Google Gemini API provider implementation.
    
    This provider uses the google-generativeai library to interact with
    Google's Gemini models.
    """
    
    AVAILABLE_MODELS = [
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google API key from environment
            model_name: Gemini model to use (default: gemini-2.5-flash)
            
        Raises:
            ImportError: If google-generativeai is not installed
            ValueError: If api_key is invalid
        """
        super().__init__(api_key, model_name)
        
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
            raise ImportError("google-generativeai library required for Gemini provider")
        
        # Configure the Gemini API
        self.genai.configure(api_key=api_key)
        self.client = self.genai.GenerativeModel(model_name)
        logger.debug(f"Gemini provider configured with model: {model_name}")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using Gemini API.
        
        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt to guide model behavior
            
        Returns:
            Generated response as a string
            
        Raises:
            RuntimeError: If API call fails
        """
        try:
            # Build the message content
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            logger.debug(f"Generating response with Gemini: {prompt[:50]}...")
            response = self.client.generate_content(full_prompt)
            
            if response.text:
                logger.debug("Response generated successfully")
                return response.text
            else:
                logger.warning("Gemini returned empty response")
                return ""
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise RuntimeError(f"Gemini API error: {str(e)}") from e
    
    def get_available_models(self) -> List[str]:
        """
        Get available Gemini models.
        
        Returns:
            List of available model names
        """
        return self.AVAILABLE_MODELS


# ============================================================================
# Claude Provider Implementation (Stub for Future)
# ============================================================================

class ClaudeProvider(BaseProvider):
    """
    Anthropic Claude API provider (placeholder for future implementation).
    
    This provider will use the anthropic library to interact with Claude models.
    """
    
    AVAILABLE_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20250219",
    ]
    
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Claude provider.
        
        Args:
            api_key: Anthropic API key
            model_name: Claude model to use
            
        Raises:
            NotImplementedError: Claude support not yet implemented
        """
        super().__init__(api_key, model_name)
        raise NotImplementedError("Claude provider not yet implemented. Use Gemini for now.")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with Claude (not implemented)."""
        raise NotImplementedError("Claude provider not yet implemented")
    
    def get_available_models(self) -> List[str]:
        """Get available Claude models."""
        return self.AVAILABLE_MODELS


# ============================================================================
# OpenAI Provider Implementation (Stub for Future)
# ============================================================================

class OpenAIProvider(BaseProvider):
    """
    OpenAI API provider (placeholder for future implementation).
    
    This provider will use the openai library to interact with GPT models.
    """
    
    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model_name: OpenAI model to use
            
        Raises:
            NotImplementedError: OpenAI support not yet implemented
        """
        super().__init__(api_key, model_name)
        raise NotImplementedError("OpenAI provider not yet implemented. Use Gemini for now.")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with OpenAI (not implemented)."""
        raise NotImplementedError("OpenAI provider not yet implemented")
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        return self.AVAILABLE_MODELS


# ============================================================================
# Ollama Provider Implementation
# ============================================================================

class OllamaProvider(BaseProvider):
    """
    Ollama local LLM provider implementation.
    
    This provider uses the ollama library for local model inference, enabling
    running open-source models like Qwen, Llama, and Mistral locally without
    requiring API keys or external services.
    
    Ollama must be running as a service (typically on localhost:11434) for
    this provider to work.
    """
    
    AVAILABLE_MODELS = [
        "qwen3:8b",
        "qwen2:7b",
        "llama2",
        "mistral",
        "neural-chat",
        "phi",
        "openchat",
    ]
    
    def __init__(self, api_key: str = "", model_name: str = "qwen3:8b"):
        """
        Initialize Ollama provider.
        
        Ollama does not require an API key as it runs locally. The api_key
        parameter is accepted for compatibility with the BaseProvider interface
        but is not used.
        
        Args:
            api_key: Not required for Ollama (local execution). Ignored.
            model_name: Ollama model to use (default: qwen3:8b)
            
        Raises:
            ImportError: If ollama library is not installed
            RuntimeError: If Ollama service is not accessible
            
        Example:
            >>> provider = OllamaProvider(model_name="qwen3:8b")
            >>> response = provider.generate_response("Hello")
        """
        # Ollama doesn't require an API key, use "local" as placeholder
        super().__init__(api_key or "local", model_name)
        
        try:
            from ollama import chat
            self.chat = chat
            logger.debug("ollama library imported successfully")
        except ImportError:
            logger.error("ollama library not installed. Install with: pip install ollama")
            raise ImportError("ollama library required for Ollama provider. Install with: pip install ollama")
        
        # Verify connection to Ollama service
        self._verify_connection()
        logger.info(f"Ollama provider ready. Model: {model_name}")
    
    def _verify_connection(self) -> None:
        """
        Verify that Ollama service is running and accessible.
        
        Raises:
            RuntimeError: If Ollama service is not reachable
        """
        try:
            logger.debug("Verifying Ollama connection...")
            # Test connection with a minimal call
            self.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                stream=False,
            )
            logger.debug("Ollama connection verified")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama service: {str(e)}")
            raise RuntimeError(
                f"Cannot connect to Ollama service. Ensure Ollama is running "
                f"(typically on localhost:11434). Error: {str(e)}"
            ) from e
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using Ollama local model.
        
        Args:
            prompt: User prompt/query to send to the model
            system_prompt: Optional system prompt to guide model behavior
            
        Returns:
            Generated response as a string
            
        Raises:
            RuntimeError: If model inference fails
            
        Example:
            >>> provider = OllamaProvider(model_name="qwen3:8b")
            >>> response = provider.generate_response(
            ...     prompt="What is Python?",
            ...     system_prompt="You are a helpful programmer."
            ... )
            >>> print(response)
        """
        try:
            logger.debug(f"Generating response with Ollama ({self.model_name}): {prompt[:50]}...")
            
            # Build messages list
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Call Ollama model
            response = self.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
            )
            
            # Extract response text
            response_text = response.get("message", {}).get("content", "").strip()
            
            if not response_text:
                logger.warning("Ollama returned empty response")
                return ""
            
            logger.debug("Response generated successfully")
            return response_text
            
        except Exception as e:
            logger.error(f"Ollama inference error: {str(e)}")
            raise RuntimeError(
                f"Failed to generate response with Ollama: {str(e)}"
            ) from e
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Ollama models.
        
        Returns:
            List of known Ollama model names
            
        Note:
            This returns a predefined list. To get models actually installed
            on your Ollama instance, use `ollama list` command.
        """
        return self.AVAILABLE_MODELS


# ============================================================================
# Provider Factory
# ============================================================================

PROVIDER_REGISTRY: Dict[str, type] = {
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def _get_provider_class(provider: str) -> type:
    """
    Get the provider class for the given provider name.
    
    Args:
        provider: Provider name (lowercase)
        
    Returns:
        Provider class
        
    Raises:
        ValueError: If provider is not recognized
    """
    provider_lower = provider.lower()
    if provider_lower not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider: {provider}. Available providers: {available}"
        )
    return PROVIDER_REGISTRY[provider_lower]


# ============================================================================
# Public API Functions
# ============================================================================

def load_api_keys() -> Dict[str, Optional[str]]:
    """
    Load API keys from environment variables.
    
    Expected environment variables:
        - GEMINI_API_KEY: Google Gemini API key
        - CLAUDE_API_KEY: Anthropic Claude API key
        - OPENAI_API_KEY: OpenAI API key
        - OLLAMA_API_URL: Ollama service URL (optional, for local inference only)
    
    Note:
        Ollama does not require an API key as it runs locally. If OLLAMA_API_URL
        is not set, Ollama assumes the service is running on localhost:11434.
    
    Returns:
        Dictionary mapping provider names to their API keys
        
    Example:
        >>> keys = load_api_keys()
        >>> if keys['gemini']:
        ...     print("Gemini API key found")
        >>> if keys['ollama']:
        ...     print("Ollama API URL found")
    """
    keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "claude": os.getenv("CLAUDE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "ollama": os.getenv("OLLAMA_API_URL"),  # Optional
    }
    
    logger.info("API keys loaded from environment")
    
    # Log which providers are available (without exposing keys)
    available_providers = [k for k, v in keys.items() if v]
    if available_providers:
        logger.debug(f"Available providers: {', '.join(available_providers)}")
    else:
        logger.warning("No API keys found in environment (Ollama can run without API key)")
    
    return keys


def initialize_client(
    provider: str = "gemini",
    model_name: Optional[str] = None
) -> BaseProvider:
    """
    Initialize and cache an LLM client for the specified provider.
    
    This function creates a global client instance that can be reused across
    multiple function calls, improving efficiency.
    
    Args:
        provider: LLM provider to use (default: "gemini")
        model_name: Specific model to use (uses provider default if None)
        
    Returns:
        Initialized provider instance
        
    Raises:
        ValueError: If provider is unknown or API key is missing (for providers that require it)
        ImportError: If required library for provider is not installed
        RuntimeError: If provider connection fails
        
    Example:
        >>> client = initialize_client(provider="gemini")
        >>> print(client.model_name)
        gemini-2.5-flash
        
        >>> client = initialize_client(provider="ollama", model_name="qwen3:8b")
        >>> print(client.model_name)
        qwen3:8b
    """
    global _client_instance
    
    provider_class = _get_provider_class(provider)
    provider_lower = provider.lower()
    
    # Ollama doesn't require an API key
    if provider_lower == "ollama":
        logger.info(f"Initializing {provider} provider (local, no API key required)")
        api_key = ""
    else:
        # Get API key from environment for other providers
        api_key_env_var = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_env_var)
        
        if not api_key:
            raise ValueError(
                f"Missing API key for {provider}. "
                f"Please set the {api_key_env_var} environment variable."
            )
    
    # Set default model names if not provided
    if model_name is None:
        model_defaults = {
            "gemini": "gemini-2.5-flash",
            "claude": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o",
            "ollama": "qwen3:8b",
        }
        model_name = model_defaults.get(provider_lower, "default")
    
    # Create provider instance
    _client_instance = provider_class(api_key, model_name)
    logger.info(f"LLM client initialized: {provider} ({model_name})")
    
    return _client_instance


def generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None
) -> str:
    """
    Generate a response from the specified LLM provider.
    
    This function handles initialization if needed and caches the client
    for efficiency. If provider/model differs from cached client, a new
    client will be initialized.
    
    Args:
        prompt: User prompt/query to send to the LLM
        system_prompt: Optional system prompt to guide model behavior
        provider: LLM provider to use (default: "gemini")
        model_name: Specific model to use (uses provider default if None)
        
    Returns:
        Generated response as a string
        
    Raises:
        ValueError: If provider is unknown or API key is missing
        RuntimeError: If LLM API call fails
        
    Example:
        >>> response = generate_response(
        ...     prompt="What is machine learning?",
        ...     system_prompt="You are an AI expert.",
        ...     provider="gemini"
        ... )
        >>> print(response)
    """
    global _client_instance
    
    # Initialize client if needed or if provider changed
    if _client_instance is None or \
       _client_instance.__class__.__name__ != f"{provider.capitalize()}Provider":
        _client_instance = initialize_client(provider, model_name)
    
    # Generate response
    logger.info(f"Generating response with {provider}")
    response = _client_instance.generate_response(prompt, system_prompt)
    
    return response


def answer_with_context(
    context: str,
    question: str,
    system_prompt: str,
    provider: str = "gemini",
    model_name: Optional[str] = None
) -> str:
    """
    Answer a question given specific context using the LLM.
    
    This function combines a context string and question into a prompt,
    then generates a response. Useful for RAG (Retrieval-Augmented Generation)
    scenarios where context from a knowledge base is provided.
    
    Args:
        context: Contextual information (e.g., from vector database)
        question: Question to answer based on context
        system_prompt: System prompt to guide model behavior
        provider: LLM provider to use (default: "gemini")
        model_name: Specific model to use (uses provider default if None)
        
    Returns:
        Answer based on the provided context
        
    Raises:
        ValueError: If provider is unknown or API key is missing
        RuntimeError: If LLM API call fails
        
    Example:
        >>> context = "Machine learning is a subset of AI..."
        >>> question = "What is machine learning?"
        >>> answer = answer_with_context(
        ...     context=context,
        ...     question=question,
        ...     system_prompt="You are helpful.",
        ...     provider="gemini",
        ...     model_name="gemini-2.5-flash"
        ... )
        >>> print(answer)
    """
    # Construct context-aware prompt
    prompt = f"""Context:
{context}

Question: {question}

Please answer based on the provided context."""
    
    logger.debug(f"Answering with context (provider: {provider}, model: {model_name or 'default'})")
    
    return generate_response(
        prompt=prompt,
        system_prompt=system_prompt,
        provider=provider,
        model_name=model_name
    )


def get_available_models(provider: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get available models for the specified provider(s).
    
    Args:
        provider: Specific provider to query (all providers if None)
        
    Returns:
        Dictionary mapping provider names to their available models
        
    Raises:
        ValueError: If provider is unknown
        
    Example:
        >>> models = get_available_models("gemini")
        >>> print(models)
        {'gemini': ['gemini-2.5-flash', 'gemini-1.5-pro', ...]}
        
        >>> all_models = get_available_models()
        >>> print(all_models.keys())
        dict_keys(['gemini', 'claude', 'openai', 'ollama'])
    """
    if provider:
        # Get models for specific provider
        provider_class = _get_provider_class(provider)
        return {provider.lower(): provider_class.AVAILABLE_MODELS}
    else:
        # Get models for all providers
        models = {}
        for provider_name, provider_class in PROVIDER_REGISTRY.items():
            models[provider_name] = provider_class.AVAILABLE_MODELS
        return models


# ============================================================================
# Utility Functions
# ============================================================================

def reset_client() -> None:
    """
    Reset the global client instance.
    
    Useful for testing or switching between providers.
    
    Example:
        >>> reset_client()
        >>> initialize_client(provider="claude")
    """
    global _client_instance
    _client_instance = None
    logger.info("LLM client instance reset")


if __name__ == "__main__":
    # Example usage
    logger.info("LLM Client Module - Example Usage")
    
    # Load API keys
    keys = load_api_keys()
    
    # Initialize and generate response
    try:
        initialize_client(provider="gemini")
        response = generate_response(
            prompt="What is a research paper?",
            system_prompt="You are a helpful academic assistant."
        )
        print(f"Response: {response}")
    except ValueError as e:
        logger.warning(f"Cannot run example: {e}")
    
    # Show available models
    print("\nAvailable Models:")
    all_models = get_available_models()
    for provider_name, models in all_models.items():
        print(f"  {provider_name}: {', '.join(models)}")
