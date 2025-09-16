"""
Environment validation module for skill gap analyst.

Enforces strict API key requirements and configuration validation.
No fallbacks or mock modes allowed - requires live internet and valid API keys.
"""

import os
import logging
from typing import Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


class EnvironmentValidationError(Exception):
    """Raised when environment validation fails."""
    pass


class EnvironmentValidator:
    """Validates required environment variables and configuration."""
    
    # Required environment variables that must be present
    REQUIRED_VARS = {
        "OPENAI_API_KEY": "OpenAI API key for LLM and embedding services",
        "TAVILY_API_KEY": "Tavily API key for web search functionality"
    }
    
    # Optional environment variables with defaults
    OPTIONAL_VARS = {
        "LLM_PROVIDER": ("openai", "LLM provider (only 'openai' supported in live mode)"),
        "OPENAI_MODEL": ("gpt-4o-mini", "OpenAI model name for chat completions"),
        "OPENAI_EMBED_MODEL": ("text-embedding-3-small", "OpenAI embedding model"),
        "TAVILY_API_BASE": (None, "Tavily API base URL (optional)"),
        "ALLOW_OFFLINE_EMBEDDINGS": ("false", "Allow offline embeddings (must be 'false' for live mode)"),
        "SEARCH_PROVIDER": ("tavily", "Search provider (only 'tavily' supported in live mode)")
    }
    
    @classmethod
    def validate_environment(cls) -> Dict[str, str]:
        """
        Validate all required environment variables are present and configured correctly.
        
        Returns:
            Dict[str, str]: Dictionary of validated environment variables
            
        Raises:
            EnvironmentValidationError: If any required variables are missing or invalid
        """
        logger.info("🔍 Validating environment configuration...")
        
        env_vars = {}
        missing_vars = []
        
        # Check required variables
        for var_name, description in cls.REQUIRED_VARS.items():
            value = os.getenv(var_name)
            if not value or not value.strip():
                missing_vars.append(f"{var_name}: {description}")
            else:
                env_vars[var_name] = value.strip()
                logger.info(f"✅ {var_name}: Present")
        
        if missing_vars:
            error_msg = (
                "❌ Missing required environment variables:\n" +
                "\n".join(f"  - {var}" for var in missing_vars) +
                "\n\nPlease set these in your .env file or environment."
            )
            logger.error(error_msg)
            raise EnvironmentValidationError(error_msg)
        
        # Check optional variables and set defaults
        for var_name, (default_value, description) in cls.OPTIONAL_VARS.items():
            value = os.getenv(var_name, default_value)
            if value is not None:
                env_vars[var_name] = value.strip()
                logger.info(f"🔧 {var_name}: {value or 'None'}")
        
        # Validate specific configurations for live mode
        cls._validate_live_mode_config(env_vars)
        
        logger.info("✅ Environment validation complete - Live mode enabled")
        return env_vars
    
    @classmethod
    def _validate_live_mode_config(cls, env_vars: Dict[str, str]) -> None:
        """
        Validate configuration specific to live mode requirements.
        
        Args:
            env_vars: Dictionary of environment variables
            
        Raises:
            EnvironmentValidationError: If configuration is invalid for live mode
        """
        # Check LLM provider is supported
        llm_provider = env_vars.get("LLM_PROVIDER", "openai").lower()
        if llm_provider != "openai":
            raise EnvironmentValidationError(
                f"❌ LLM_PROVIDER '{llm_provider}' not supported in live mode. "
                "Only 'openai' is supported."
            )
        
        # Check search provider is supported
        search_provider = env_vars.get("SEARCH_PROVIDER", "tavily").lower()
        if search_provider != "tavily":
            raise EnvironmentValidationError(
                f"❌ SEARCH_PROVIDER '{search_provider}' not supported in live mode. "
                "Only 'tavily' is supported."
            )
        
        # Check offline embeddings is disabled
        allow_offline = env_vars.get("ALLOW_OFFLINE_EMBEDDINGS", "false").lower()
        if allow_offline not in ("false", "f", "0", "no", "n"):
            raise EnvironmentValidationError(
                "❌ ALLOW_OFFLINE_EMBEDDINGS must be 'false' in live mode. "
                "No mock or offline modes are allowed."
            )
        
        # Validate API key formats (basic sanity checks)
        openai_key = env_vars.get("OPENAI_API_KEY", "")
        if not openai_key.startswith("sk-"):
            logger.warning("⚠️ OPENAI_API_KEY does not start with 'sk-' - may be invalid")
        
        tavily_key = env_vars.get("TAVILY_API_KEY", "")
        if not tavily_key.startswith("tvly-"):
            logger.warning("⚠️ TAVILY_API_KEY does not start with 'tvly-' - may be invalid")
    
    @classmethod
    def ensure_live_mode(cls) -> Dict[str, str]:
        """
        Ensure environment is properly configured for live mode operation.
        This method should be called at the start of all applications.
        
        Returns:
            Dict[str, str]: Validated environment configuration
            
        Raises:
            EnvironmentValidationError: If environment is not suitable for live mode
        """
        try:
            return cls.validate_environment()
        except EnvironmentValidationError:
            logger.error("🚫 Live mode validation failed")
            logger.error("💡 This application requires:")
            logger.error("   - Valid OPENAI_API_KEY")
            logger.error("   - Valid TAVILY_API_KEY") 
            logger.error("   - Active internet connection")
            logger.error("   - No mock or offline modes")
            raise


def validate_env_on_import() -> Dict[str, str]:
    """
    Convenience function to validate environment on module import.
    Can be called from application entry points.
    
    Returns:
        Dict[str, str]: Validated environment configuration
    """
    return EnvironmentValidator.ensure_live_mode()


# Auto-validation when imported (can be disabled by setting SKIP_ENV_VALIDATION=1)
if os.getenv("SKIP_ENV_VALIDATION") != "1":
    _env_config = None  # Will be set on first validation call