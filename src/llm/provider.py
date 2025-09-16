"""
LLM provider module supporting ONLY Gemini and Mistral with Auto fallback.

Provides chat and embedding models with automatic provider selection and fallback.
Supports live API integrations with comprehensive error handling and retries.
"""

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, environment variables may be set directly
    pass

# Configure structured logging
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


def choose_provider(explicit: str | None) -> str:
    """
    Choose provider based on explicit setting or environment.
    
    Args:
        explicit: Explicit provider name ("auto", "gemini", "mistral") or None
        
    Returns:
        Provider name: "auto", "gemini", or "mistral"
    """
    if explicit is not None:
        return explicit.lower()
    
    return os.getenv("LLM_PROVIDER", "auto").lower()


def require_env_for(provider: str) -> None:
    """
    Check that required environment variables are set for the given provider.
    
    Args:
        provider: Provider name ("gemini" or "mistral")
        
    Raises:
        RuntimeError: If required API keys are missing
    """
    if provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required for Gemini provider. "
                "Please set it in your .env file or environment."
            )
    elif provider == "mistral":
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError(
                "MISTRAL_API_KEY environment variable is required for Mistral provider. "
                "Please set it in your .env file or environment."
            )
    else:
        raise RuntimeError(f"Unknown provider: {provider}")


def _test_chat_model(model: BaseChatModel) -> bool:
    """
    Test a chat model with a simple healthcheck request using tenacity.
    
    Args:
        model: Chat model to test
        
    Returns:
        True if model responds successfully, False otherwise
    """
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def _ping_model():
        """Perform a one-shot health check with retries."""
        response = model.invoke("respond with 'ok' only")
        if not response or not response.content:
            raise LLMProviderError("Empty response from model")
        return response.content.strip().lower()
    
    try:
        result = _ping_model()
        return "ok" in result
    except Exception as e:
        logger.warning(f"Chat model healthcheck failed after 3 retries: {e}")
        return False


def _test_embedding_model(embeddings: Embeddings) -> bool:
    """
    Test an embedding model with a simple healthcheck request using tenacity.
    
    Args:
        embeddings: Embedding model to test
        
    Returns:
        True if model responds successfully, False otherwise
    """
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def _ping_embeddings():
        """Perform a one-shot embedding health check with retries."""
        result = embeddings.embed_query("test")
        if not result or len(result) == 0:
            raise LLMProviderError("Empty embedding result")
        return result
    
    try:
        _ping_embeddings()
        return True
    except Exception as e:
        logger.warning(f"Embedding model healthcheck failed after 3 retries: {e}")
        return False


def _create_gemini_chat(model_name: str | None, temperature: float, timeout: int) -> BaseChatModel:
    """Create Gemini chat model."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai package is required. Install with: pip install langchain-google-genai"
        )
    
    # Try preferred model first, fallback to stable model
    if model_name is None:
        preferred_models = ["gemini-2.0-flash", "gemini-1.5-pro"]
        for model in preferred_models:
            try:
                chat_model = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    timeout=timeout,
                    max_retries=3,
                    google_api_key=os.getenv("GEMINI_API_KEY")
                )
                if _test_chat_model(chat_model):
                    logger.info(f"Using Gemini model: {model}")
                    return chat_model
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini model {model}: {e}")
                continue
        
        # If all preferred models fail, raise error
        raise LLMProviderError("Failed to initialize any Gemini chat model")
    else:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            max_retries=3,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )


def _create_mistral_chat(model_name: str | None, temperature: float, timeout: int) -> BaseChatModel:
    """Create Mistral chat model."""
    try:
        from langchain_mistralai import ChatMistralAI
    except ImportError:
        raise ImportError(
            "langchain-mistralai package is required. Install with: pip install langchain-mistralai"
        )
    
    final_model = model_name or os.getenv("MISTRAL_MODEL") or "mistral-large-latest"
    
    return ChatMistralAI(
        model=final_model,
        temperature=temperature,
        timeout=timeout,
        max_retries=3,
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _test_and_return_chat_model(provider: str, model_name: str | None, temperature: float, timeout: int) -> BaseChatModel:
    """
    Create and test a chat model with retries.
    
    Args:
        provider: Provider name ("gemini" or "mistral")
        model_name: Model name or None for default
        temperature: Model temperature
        timeout: Request timeout in seconds
        
    Returns:
        Tested chat model
        
    Raises:
        LLMProviderError: If model creation or testing fails
    """
    require_env_for(provider)
    
    if provider == "gemini":
        model = _create_gemini_chat(model_name, temperature, timeout)
    elif provider == "mistral":
        model = _create_mistral_chat(model_name, temperature, timeout)
    else:
        raise LLMProviderError(f"Unsupported provider: {provider}")
    
    # Test the model with a simple healthcheck
    if not _test_chat_model(model):
        raise LLMProviderError(f"Chat model healthcheck failed for {provider}")
    
    return model


def get_chat_model(
    provider: str | None, 
    model_name: str | None, 
    temperature: float = 0.2, 
    timeout: int = 60
) -> BaseChatModel:
    """
    Get a chat model with automatic provider selection and fallback.
    
    Args:
        provider: Provider name ("auto", "gemini", "mistral") or None for auto
        model_name: Model name override or None for defaults
        temperature: Model temperature (default: 0.2)
        timeout: Request timeout in seconds (default: 60)
        
    Returns:
        Live chat model instance
        
    Raises:
        RuntimeError: If no providers are available or API keys are missing
    """
    selected_provider = choose_provider(provider)
    
    if selected_provider == "auto":
        # Try Gemini first, fallback to Mistral
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))
        has_mistral = bool(os.getenv("MISTRAL_API_KEY"))
        
        if not has_gemini and not has_mistral:
            raise RuntimeError(
                "At least one API key is required for auto mode. "
                "Please set GEMINI_API_KEY or MISTRAL_API_KEY in your .env file."
            )
        
        # Try Gemini first with health check
        if has_gemini:
            try:
                require_env_for("gemini")
                model = _create_gemini_chat(model_name, temperature, timeout)
                logger.info("Testing Gemini chat model health...")
                
                if _test_chat_model(model):
                    logger.info("Auto mode: Using Gemini chat model (health check passed)")
                    return model
                else:
                    logger.warning("Gemini chat model failed health check, switching to Mistral")
            except Exception as e:
                logger.warning(f"Gemini chat initialization failed, trying Mistral: {e}")
        
        # Fallback to Mistral with health check
        if has_mistral:
            try:
                require_env_for("mistral")
                model = _create_mistral_chat(model_name, temperature, timeout)
                logger.info("Testing Mistral chat model health...")
                
                if _test_chat_model(model):
                    logger.info("Auto mode: Using Mistral chat model (health check passed)")
                    return model
                else:
                    logger.error("Mistral chat model also failed health check")
                    raise RuntimeError("All chat providers failed health checks in auto mode")
            except Exception as e:
                logger.error(f"Mistral chat also failed: {e}")
                raise RuntimeError("All chat providers failed in auto mode")
        
        raise RuntimeError("No valid chat providers available")
    
    else:
        # Use explicit provider with health check
        require_env_for(selected_provider)
        
        if selected_provider == "gemini":
            model = _create_gemini_chat(model_name, temperature, timeout)
        elif selected_provider == "mistral":
            model = _create_mistral_chat(model_name, temperature, timeout)
        else:
            raise LLMProviderError(f"Unsupported provider: {selected_provider}")
        
        logger.info(f"Testing {selected_provider} chat model health...")
        if not _test_chat_model(model):
            raise LLMProviderError(f"Chat model healthcheck failed for {selected_provider}")
        
        logger.info(f"Using {selected_provider} chat model")
        return model


def _create_gemini_embeddings(embed_model_name: str | None) -> Embeddings:
    """Create Gemini embeddings."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError:
        raise ImportError(
            "langchain-google-genai package is required. Install with: pip install langchain-google-genai"
        )
    
    final_model = embed_model_name or os.getenv("GEMINI_EMBED_MODEL") or "text-embedding-004"
    
    return GoogleGenerativeAIEmbeddings(
        model=final_model,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )


def _create_mistral_embeddings(embed_model_name: str | None) -> Embeddings:
    """Create Mistral embeddings."""
    try:
        from langchain_mistralai import MistralAIEmbeddings
    except ImportError:
        raise ImportError(
            "langchain-mistralai package is required. Install with: pip install langchain-mistralai"
        )
    
    final_model = embed_model_name or os.getenv("MISTRAL_EMBED_MODEL") or "mistral-embed"
    
    return MistralAIEmbeddings(
        model=final_model,
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )


def _create_offline_embeddings() -> Embeddings:
    """Create offline sentence-transformers embeddings."""
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        raise ImportError(
            "sentence-transformers package is required for offline embeddings. "
            "Install with: pip install sentence-transformers"
        )


def get_embedding_model(provider: str | None, embed_model_name: str | None) -> Embeddings:
    """
    Get an embedding model with automatic provider selection and fallback.
    
    Args:
        provider: Provider name ("auto", "gemini", "mistral") or None for auto
        embed_model_name: Embedding model name or None for defaults
        
    Returns:
        Embeddings instance
        
    Raises:
        RuntimeError: If no providers are available or API keys are missing
    """
    selected_provider = choose_provider(provider)
    
    if selected_provider == "auto":
        # Try Gemini embeddings first, fallback to Mistral
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))
        has_mistral = bool(os.getenv("MISTRAL_API_KEY"))
        
        if not has_gemini and not has_mistral:
            # Check for offline fallback
            if os.getenv("ALLOW_OFFLINE_EMBEDDINGS", "false").lower() == "true":
                logger.info("Auto mode: Using offline embeddings (sentence-transformers)")
                return _create_offline_embeddings()
            else:
                raise RuntimeError(
                    "At least one API key is required for auto embeddings mode. "
                    "Please set GEMINI_API_KEY or MISTRAL_API_KEY in your .env file. "
                    "Or set ALLOW_OFFLINE_EMBEDDINGS=true for offline mode."
                )
        
        # Try Gemini first with health check
        if has_gemini:
            try:
                require_env_for("gemini")
                embeddings = _create_gemini_embeddings(embed_model_name)
                logger.info("Testing Gemini embeddings health...")
                
                if _test_embedding_model(embeddings):
                    logger.info("Auto mode: Using Gemini embeddings (health check passed)")
                    return embeddings
                else:
                    logger.warning("Gemini embeddings failed health check, switching to Mistral")
            except Exception as e:
                logger.warning(f"Gemini embeddings failed, trying Mistral: {e}")
        
        # Fallback to Mistral with health check
        if has_mistral:
            try:
                require_env_for("mistral")
                embeddings = _create_mistral_embeddings(embed_model_name)
                logger.info("Testing Mistral embeddings health...")
                
                if _test_embedding_model(embeddings):
                    logger.info("Auto mode: Using Mistral embeddings (health check passed)")
                    return embeddings
                else:
                    logger.error("Mistral embeddings also failed health check")
                    
                    # Final fallback to offline if allowed
                    if os.getenv("ALLOW_OFFLINE_EMBEDDINGS", "false").lower() == "true":
                        logger.info("Auto mode: Falling back to offline embeddings")
                        return _create_offline_embeddings()
                    else:
                        raise RuntimeError("All embedding providers failed health checks in auto mode")
            except Exception as e:
                logger.error(f"Mistral embeddings also failed: {e}")
                
                # Final fallback to offline if allowed
                if os.getenv("ALLOW_OFFLINE_EMBEDDINGS", "false").lower() == "true":
                    logger.info("Auto mode: Falling back to offline embeddings")
                    return _create_offline_embeddings()
                else:
                    raise RuntimeError("All embedding providers failed in auto mode")
        
        raise RuntimeError("No valid embedding providers available")
    
    elif selected_provider == "gemini":
        require_env_for("gemini")
        embeddings = _create_gemini_embeddings(embed_model_name)
        logger.info("Testing Gemini embeddings health...")
        
        if not _test_embedding_model(embeddings):
            raise LLMProviderError(f"Embedding model healthcheck failed for {selected_provider}")
        
        logger.info("Using Gemini embeddings")
        return embeddings
    
    elif selected_provider == "mistral":
        require_env_for("mistral")
        embeddings = _create_mistral_embeddings(embed_model_name)
        logger.info("Testing Mistral embeddings health...")
        
        if not _test_embedding_model(embeddings):
            raise LLMProviderError(f"Embedding model healthcheck failed for {selected_provider}")
        
        logger.info("Using Mistral embeddings")
        return embeddings
    
    else:
        raise RuntimeError(f"Unsupported provider: {selected_provider}")
