"""
Global configuration for live integrations with robust error handling.

Provides centralized timeout, retry, and reliability settings for all external API calls.
Enforces production-ready defaults with comprehensive observability.
"""

import logging
import time
from functools import wraps
from typing import Callable, Any, Dict, Optional, TypeVar, Union
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import httpx
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global timeout settings (in seconds)
TIMEOUTS = {
    "connect": 15.0,      # Connection timeout - 15s
    "read": 30.0,         # Read timeout - 30s  
    "total": 45.0,        # Total request timeout - 45s
    "pool": 60.0,         # Connection pool timeout - 60s
}

# Global retry settings
RETRY_CONFIG = {
    "max_attempts": 3,                    # Maximum retry attempts
    "min_wait": 1.0,                     # Minimum wait between retries (seconds)
    "max_wait": 10.0,                    # Maximum wait between retries (seconds)
    "multiplier": 2.0,                   # Exponential backoff multiplier
    "jitter": True,                      # Add random jitter to prevent thundering herd
}

# Retryable HTTP status codes
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

# Retryable exception types
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectTimeout, 
    httpx.ReadTimeout,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)

F = TypeVar('F', bound=Callable[..., Any])


def get_http_client(
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs
) -> httpx.Client:
    """
    Create HTTP client with standardized timeout and reliability settings.
    
    Args:
        base_url: Base URL for requests
        headers: Default headers for requests
        **kwargs: Additional httpx.Client arguments
    
    Returns:
        Configured httpx.Client with production timeouts
    """
    default_headers = {
        "User-Agent": "SkillGapAnalyst/1.0 (Live-Mode-Only)",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    
    if headers:
        default_headers.update(headers)
    
    client_kwargs = {
        "base_url": base_url,
        "headers": default_headers,
        "timeout": httpx.Timeout(
            connect=TIMEOUTS["connect"],
            read=TIMEOUTS["read"],
            pool=TIMEOUTS["pool"]
        ),
        "limits": httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0
        ),
        "follow_redirects": True,
        **kwargs
    }
    
    return httpx.Client(**client_kwargs)


def with_retries(
    max_attempts: Optional[int] = None,
    min_wait: Optional[float] = None, 
    max_wait: Optional[float] = None,
    exceptions: Optional[tuple] = None,
    reraise_on_failure: bool = True,
    operation_name: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator that adds robust retry logic to functions making external API calls.
    
    Args:
        max_attempts: Maximum retry attempts (default: global config)
        min_wait: Minimum wait between retries in seconds
        max_wait: Maximum wait between retries in seconds  
        exceptions: Tuple of exception types to retry on
        reraise_on_failure: Whether to reraise exception after all retries fail
        operation_name: Human-readable operation name for logging
    
    Returns:
        Decorated function with retry logic
    """
    # Use global defaults if not specified
    attempts = max_attempts or RETRY_CONFIG["max_attempts"]
    min_wait_time = min_wait or RETRY_CONFIG["min_wait"]
    max_wait_time = max_wait or RETRY_CONFIG["max_wait"]
    retry_exceptions = exceptions or RETRYABLE_EXCEPTIONS
    
    def decorator(func: F) -> F:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=RETRY_CONFIG["multiplier"],
                min=min_wait_time,
                max=max_wait_time
            ),
            retry=retry_if_exception_type(retry_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO)
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                logger.info(
                    "Starting operation",
                    operation=op_name,
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys())
                )
                
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(
                    "Operation completed successfully",
                    operation=op_name,
                    duration_seconds=round(duration, 3)
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Operation failed after retries",
                    operation=op_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_seconds=round(duration, 3)
                )
                
                if reraise_on_failure:
                    raise
                return None
                
        return wrapper
    return decorator


def http_status_retry_condition(exception: Exception) -> bool:
    """
    Determine if an HTTP exception should trigger a retry based on status code.
    
    Args:
        exception: Exception to evaluate
        
    Returns:
        True if the exception indicates a retryable HTTP error
    """
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in RETRYABLE_HTTP_CODES
    return False


class OperationTimer:
    """Context manager for timing operations with structured logging."""
    
    def __init__(self, operation_name: str, **context):
        self.operation_name = operation_name
        self.context = context
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(
            "Operation started",
            operation=self.operation_name,
            **self.context
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            logger.info(
                "Operation completed",
                operation=self.operation_name,
                duration_seconds=round(duration, 3),
                **self.context
            )
        else:
            logger.error(
                "Operation failed",
                operation=self.operation_name,
                duration_seconds=round(duration, 3),
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context
            )


# Pre-configured decorators for common use cases
api_retry = with_retries(
    operation_name="API Call",
    exceptions=RETRYABLE_EXCEPTIONS + (httpx.HTTPStatusError,)
)

embedding_retry = with_retries(
    max_attempts=2,  # Fewer retries for expensive embedding calls
    operation_name="Embedding Generation"
)

search_retry = with_retries(
    max_attempts=3,
    min_wait=2.0,  # Longer wait for search APIs
    operation_name="Web Search"
)


def log_api_call(
    provider: str, 
    endpoint: str, 
    request_size: Optional[int] = None,
    **context
) -> OperationTimer:
    """
    Create an operation timer specifically for API calls with provider context.
    
    Args:
        provider: API provider name (e.g., "OpenAI", "Tavily")
        endpoint: API endpoint being called
        request_size: Size of request data (e.g., token count, text length)
        **context: Additional context for logging
        
    Returns:
        OperationTimer context manager
    """
    timer_context = {
        "provider": provider,
        "endpoint": endpoint,
        **context
    }
    
    if request_size is not None:
        timer_context["request_size"] = request_size
        
    return OperationTimer(f"{provider} API Call", **timer_context)


# Export key components
__all__ = [
    "TIMEOUTS",
    "RETRY_CONFIG", 
    "get_http_client",
    "with_retries",
    "OperationTimer",
    "api_retry",
    "embedding_retry", 
    "search_retry",
    "log_api_call",
    "logger"
]