"""
Embeddings utilities for text similarity and vector operations.

Provides text embedding functionality with:
- Gemini and Mistral embeddings (live API required)
- Optional offline fallback via ALLOW_OFFLINE_EMBEDDINGS env var
- In-memory caching with SHA1 hashing
- Cosine similarity matrix calculations
"""

import hashlib
import os
from typing import Dict, List
import numpy as np

from src.llm.provider import get_embedding_model

# Configure structured logging
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Global in-memory cache for embeddings
_embedding_cache: Dict[str, np.ndarray] = {}


def _get_text_hash(text: str) -> str:
    """Generate SHA1 hash for text caching."""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def _embed_with_provider(texts: List[str], provider_preference: str | None) -> np.ndarray:
    """Embed texts using the configured LLM provider (Gemini/Mistral)."""
    try:
        # Get embedding model from provider
        embedding_model = get_embedding_model(provider_preference, None)
        
        # Embed all texts
        embeddings = embedding_model.embed_documents(texts)
        
        # Convert to numpy array
        return np.array(embeddings, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Provider embedding failed: {e}")
        raise RuntimeError(f"Provider embeddings not available: {e}")


def _embed_with_offline_fallback(texts: List[str]) -> np.ndarray:
    """Embed texts using offline sentence-transformers fallback."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use a lightweight model for offline embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        logger.info(f"Using offline embeddings for {len(texts)} texts")
        return embeddings.astype(np.float32)
        
    except ImportError:
        raise RuntimeError(
            "Offline embeddings require sentence-transformers package. "
            "Install with: pip install sentence-transformers"
        )
    except Exception as e:
        logger.error(f"Offline embedding failed: {e}")
        raise RuntimeError(f"Offline embeddings failed: {e}")


def embed_texts(texts: list[str], provider_preference: str | None = None) -> np.ndarray:
    """
    Embed a list of texts into dense vector representations.
    
    By default, uses live Gemini/Mistral embeddings. Set ALLOW_OFFLINE_EMBEDDINGS=true
    to enable sentence-transformers fallback if live embeddings fail.
    
    Supports in-memory caching with SHA1 text hashing.
    
    Args:
        texts: List of text strings to embed
        provider_preference: Preferred provider ("auto", "gemini", "mistral") or None for auto
        
    Returns:
        np.ndarray: Array of shape (len(texts), embedding_dim) with embeddings
        
    Raises:
        ValueError: If texts list is empty
        RuntimeError: If provider embeddings fail and offline not allowed
    """
    if not texts:
        raise ValueError("Cannot embed empty list of texts")
    
    # Collect embeddings from cache and identify missing ones
    cached_embeddings = []
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        text_hash = _get_text_hash(text)
        
        # Check in-memory cache
        if text_hash in _embedding_cache:
            embedding = _embedding_cache[text_hash]
            cached_embeddings.append((i, embedding))
            logger.debug(f"Found in memory cache: {text_hash}")
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # If all texts are cached, return cached results
    if not uncached_texts:
        result = np.zeros((len(texts), cached_embeddings[0][1].shape[0]), dtype=np.float32)
        for idx, embedding in cached_embeddings:
            result[idx] = embedding
        logger.debug(f"All {len(texts)} texts found in cache")
        return result
    
    # Generate embeddings for uncached texts
    logger.info(f"Generating embeddings for {len(uncached_texts)} new texts")
    
    # Check if offline fallback is allowed
    allow_offline = os.getenv("ALLOW_OFFLINE_EMBEDDINGS", "false").lower() in ("true", "t", "1", "yes", "y")
    
    # Try provider embeddings first
    new_embeddings = None
    try:
        new_embeddings = _embed_with_provider(uncached_texts, provider_preference)
        logger.info(f"Generated {len(uncached_texts)} embeddings using provider")
        
    except Exception as provider_error:
        if not allow_offline:
            raise RuntimeError(
                f"Provider embeddings failed and ALLOW_OFFLINE_EMBEDDINGS is not set. "
                f"Either provide valid GEMINI_API_KEY or MISTRAL_API_KEY or set ALLOW_OFFLINE_EMBEDDINGS=true. "
                f"Provider error: {provider_error}"
            )
        
        # Try offline fallback
        logger.warning(f"Provider failed, attempting offline fallback: {provider_error}")
        try:
            new_embeddings = _embed_with_offline_fallback(uncached_texts)
        except Exception as offline_error:
            raise RuntimeError(
                f"Both provider and offline embeddings failed. "
                f"Provider error: {provider_error}. "
                f"Offline error: {offline_error}"
            )
    
    # Cache the new embeddings
    for i, text in enumerate(uncached_texts):
        text_hash = _get_text_hash(text)
        embedding = new_embeddings[i]
        
        # Cache in memory
        _embedding_cache[text_hash] = embedding
    
    # Combine cached and new embeddings in original order
    if cached_embeddings:
        # We have some cached embeddings to merge
        embedding_dim = (cached_embeddings[0][1].shape[0] if cached_embeddings 
                        else new_embeddings.shape[1])
        result = np.zeros((len(texts), embedding_dim), dtype=np.float32)
        
        # Fill in cached embeddings
        for idx, embedding in cached_embeddings:
            result[idx] = embedding
            
        # Fill in new embeddings
        for i, orig_idx in enumerate(uncached_indices):
            result[orig_idx] = new_embeddings[i]
            
        return result
    else:
        # All embeddings are new
        return new_embeddings


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of embeddings.
    
    Args:
        A: Array of shape (n_samples_A, embedding_dim)
        B: Array of shape (n_samples_B, embedding_dim)
        
    Returns:
        np.ndarray: Cosine similarity matrix of shape (n_samples_A, n_samples_B)
                   Values range from -1 (opposite) to 1 (identical)
                   
    Raises:
        ValueError: If input arrays have incompatible shapes
    """
    # Validate inputs
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Embedding dimensions must match: A has {A.shape[1]}, B has {B.shape[1]}"
        )
    
    # Normalize vectors to unit length
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(A_norm, B_norm.T)
    
    # Clip to handle numerical precision issues
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    
    return similarity_matrix


def clear_embedding_cache() -> int:
    """
    Clear the in-memory embedding cache.
    
    Returns:
        int: Number of cached embeddings that were cleared
    """
    global _embedding_cache
    
    memory_count = len(_embedding_cache)
    _embedding_cache.clear()
    
    logger.info(f"Cleared {memory_count} memory cache embeddings")
    return memory_count


def get_cache_info() -> Dict[str, any]:
    """
    Get information about the embedding cache.
    
    Returns:
        dict: Cache statistics including size, memory usage, offline status
    """
    memory_size = len(_embedding_cache)
    
    # Memory cache info
    memory_info = {"size": memory_size, "memory_mb": 0.0}
    if memory_size > 0:
        # Estimate memory usage (rough calculation)
        sample_embedding = next(iter(_embedding_cache.values()))
        embedding_size_bytes = sample_embedding.nbytes
        total_memory_bytes = memory_size * embedding_size_bytes
        memory_mb = total_memory_bytes / (1024 * 1024)
        
        memory_info.update({
            "memory_mb": round(memory_mb, 2),
            "embedding_dim": sample_embedding.shape[0]
        })
    
    allow_offline = os.getenv("ALLOW_OFFLINE_EMBEDDINGS", "false").lower() in ("true", "t", "1", "yes", "y")
    
    return {
        "memory": memory_info,
        "offline_allowed": allow_offline,
        "provider_available": _check_provider_available()
    }


def _check_provider_available() -> bool:
    """Check if provider embeddings are available."""
    try:
        get_embedding_model("auto", None)
        return True
    except Exception:
        return False
