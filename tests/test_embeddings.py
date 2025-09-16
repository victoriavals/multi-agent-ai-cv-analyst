"""
Tests for embeddings functionality.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.tools.embeddings import (
    embed_texts,
    cosine_sim_matrix,
    clear_embedding_cache,
    get_cache_info,
    _get_text_hash,
    _embedding_cache
)


class TestEmbedTexts:
    """Test the embed_texts function."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_embedding_cache()
    
    def test_embed_texts_empty_list(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot embed empty list"):
            embed_texts([])
    
    @patch('src.tools.embeddings.get_embedding_model')
    def test_embed_texts_with_provider_success(self, mock_get_model):
        """Test successful embedding using provider."""
        # Mock the embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_get_model.return_value = mock_embedding_model
        
        texts = ["hello world", "test text"]
        result = embed_texts(texts)
        
        # Check result shape and values
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result, 
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        )
        
        # Check that model was called correctly
        mock_embedding_model.embed_documents.assert_called_once_with(texts)
    
    @patch('src.tools.embeddings.get_embedding_model')
    @patch('src.tools.embeddings._get_fallback_model')
    def test_embed_texts_provider_fails_fallback_succeeds(self, mock_fallback, mock_get_model):
        """Test fallback to sentence-transformers when provider fails."""
        # Mock provider failure
        mock_get_model.side_effect = Exception("Provider unavailable")
        
        # Mock successful fallback
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.array([
            [0.7, 0.8, 0.9],
            [0.1, 0.2, 0.3]
        ], dtype=np.float32)
        mock_fallback.return_value = mock_st_model
        
        texts = ["test1", "test2"]
        result = embed_texts(texts)
        
        # Check result
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        
        # Check that fallback was called
        mock_st_model.encode.assert_called_once_with(
            texts, convert_to_numpy=True, dtype=np.float32
        )
    
    @patch('src.tools.embeddings.get_embedding_model')
    @patch('src.tools.embeddings._get_fallback_model')
    def test_embed_texts_both_fail(self, mock_fallback, mock_get_model):
        """Test that RuntimeError is raised when both provider and fallback fail."""
        # Mock both failures
        mock_get_model.side_effect = Exception("Provider error")
        mock_fallback.side_effect = Exception("Fallback error")
        
        with pytest.raises(RuntimeError, match="Both provider and fallback embedding failed"):
            embed_texts(["test"])
    
    def test_embed_texts_single_text_fallback_shape(self):
        """Test that single text gets proper shape with fallback."""
        with patch('src.tools.embeddings.get_embedding_model') as mock_get_model, \
             patch('src.tools.embeddings._get_fallback_model') as mock_fallback:
            
            # Mock provider failure
            mock_get_model.side_effect = Exception("Provider unavailable")
            
            # Mock fallback returning 1D array (single text)
            mock_st_model = Mock()
            mock_st_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            mock_fallback.return_value = mock_st_model
            
            result = embed_texts(["single text"])
            
            # Should be reshaped to 2D
            assert result.shape == (1, 3)
            assert result.dtype == np.float32
    
    @patch('src.tools.embeddings.get_embedding_model')
    def test_embed_texts_caching(self, mock_get_model):
        """Test that embeddings are cached correctly."""
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_get_model.return_value = mock_embedding_model
        
        texts = ["hello", "world"]
        
        # First call - should hit provider
        result1 = embed_texts(texts)
        assert mock_embedding_model.embed_documents.call_count == 1
        
        # Second call with same texts - should use cache
        result2 = embed_texts(texts)
        assert mock_embedding_model.embed_documents.call_count == 1  # No additional calls
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        
        # Cache should contain both texts
        cache_info = get_cache_info()
        assert cache_info["size"] == 2
    
    @patch('src.tools.embeddings.get_embedding_model')
    def test_embed_texts_partial_caching(self, mock_get_model):
        """Test embedding with some cached and some new texts."""
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_get_model.return_value = mock_embedding_model
        
        # First call - cache "hello"
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2]]
        embed_texts(["hello"])
        
        # Second call - "hello" cached, "world" new
        mock_embedding_model.embed_documents.return_value = [[0.3, 0.4]]
        result = embed_texts(["hello", "world"])
        
        # Should only call provider for "world"
        assert mock_embedding_model.embed_documents.call_count == 2
        assert mock_embedding_model.embed_documents.call_args_list[1][0][0] == ["world"]
        
        # Result should have correct shape and values
        assert result.shape == (2, 2)
        expected = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestCosineSimilarityMatrix:
    """Test cosine similarity matrix computation."""
    
    def test_cosine_sim_matrix_basic(self):
        """Test basic cosine similarity computation."""
        A = np.array([[1, 0], [0, 1]], dtype=np.float32)
        B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        
        result = cosine_sim_matrix(A, B)
        
        assert result.shape == (2, 3)
        
        # Check specific values
        assert abs(result[0, 0] - 1.0) < 1e-6  # Same vectors
        assert abs(result[0, 1] - 0.0) < 1e-6  # Orthogonal vectors
        assert abs(result[1, 1] - 1.0) < 1e-6  # Same vectors
        assert abs(result[1, 0] - 0.0) < 1e-6  # Orthogonal vectors
    
    def test_cosine_sim_matrix_identical_vectors(self):
        """Test similarity of identical vectors."""
        A = np.array([[3, 4]], dtype=np.float32)
        B = np.array([[6, 8]], dtype=np.float32)  # Same direction, different magnitude
        
        result = cosine_sim_matrix(A, B)
        
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 1.0) < 1e-6  # Should be 1.0 for same direction
    
    def test_cosine_sim_matrix_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        A = np.array([[1, 0]], dtype=np.float32)
        B = np.array([[-1, 0]], dtype=np.float32)
        
        result = cosine_sim_matrix(A, B)
        
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - (-1.0)) < 1e-6  # Should be -1.0 for opposite direction
    
    def test_cosine_sim_matrix_wrong_dimensions(self):
        """Test error handling for incompatible dimensions."""
        A = np.array([[1, 2]], dtype=np.float32)
        B = np.array([[1, 2, 3]], dtype=np.float32)  # Different embedding dim
        
        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            cosine_sim_matrix(A, B)
    
    def test_cosine_sim_matrix_wrong_shape(self):
        """Test error handling for wrong array shapes."""
        A = np.array([1, 2, 3], dtype=np.float32)  # 1D array
        B = np.array([[1, 2, 3]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Input arrays must be 2-dimensional"):
            cosine_sim_matrix(A, B)
    
    def test_cosine_sim_matrix_zero_vectors(self):
        """Test handling of zero vectors."""
        A = np.array([[0, 0]], dtype=np.float32)
        B = np.array([[1, 1]], dtype=np.float32)
        
        result = cosine_sim_matrix(A, B)
        
        # Zero vector should have similarity close to 0 (due to epsilon)
        assert result.shape == (1, 1)
        assert abs(result[0, 0]) < 0.1  # Small value due to epsilon


class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_embedding_cache()
    
    def test_get_text_hash(self):
        """Test text hashing function."""
        hash1 = _get_text_hash("hello world")
        hash2 = _get_text_hash("hello world")
        hash3 = _get_text_hash("hello world!")
        
        # Same text should produce same hash
        assert hash1 == hash2
        
        # Different text should produce different hash
        assert hash1 != hash3
        
        # Hash should be valid SHA1 (40 hex characters)
        assert len(hash1) == 40
        assert all(c in "0123456789abcdef" for c in hash1)
    
    def test_clear_embedding_cache(self):
        """Test cache clearing."""
        # Ensure cache is empty first
        clear_embedding_cache()
        
        # Add something to cache
        _embedding_cache["test"] = np.array([1, 2, 3])
        
        assert len(_embedding_cache) == 1
        
        count = clear_embedding_cache()
        
        assert count == 1
        assert len(_embedding_cache) == 0
    
    def test_get_cache_info_empty(self):
        """Test cache info when empty."""
        clear_embedding_cache()
        
        info = get_cache_info()
        
        assert info["size"] == 0
        assert info["memory_mb"] == 0.0
    
    def test_get_cache_info_with_data(self):
        """Test cache info with data."""
        clear_embedding_cache()
        
        # Add test data to cache (larger arrays for measurable memory)
        _embedding_cache["test1"] = np.array([1.0] * 1000, dtype=np.float32)
        _embedding_cache["test2"] = np.array([4.0] * 1000, dtype=np.float32)
        
        info = get_cache_info()
        
        assert info["size"] == 2
        assert info["embedding_dim"] == 1000
        assert info["memory_mb"] > 0


class TestFallbackModel:
    """Test sentence-transformers fallback functionality."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_fallback_model_success(self, mock_st):
        """Test successful fallback model loading."""
        from src.tools.embeddings import _get_fallback_model
        
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        # Clear global model
        import src.tools.embeddings
        src.tools.embeddings._fallback_model = None
        
        result = _get_fallback_model()
        
        assert result == mock_model
        mock_st.assert_called_once_with('all-MiniLM-L6-v2')
    
    def test_get_fallback_model_import_error(self):
        """Test fallback model import error handling."""
        from src.tools.embeddings import _get_fallback_model
        
        # Clear global model
        import src.tools.embeddings
        src.tools.embeddings._fallback_model = None
        
        with patch('builtins.__import__', side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="sentence-transformers package is required"):
                _get_fallback_model()


if __name__ == "__main__":
    pytest.main([__file__])