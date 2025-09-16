"""
Unit tests for evaluation utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.utils.eval import (
    cosine_similarity,
    classify_priority,
    prioritize_gaps
)


class TestCosineSimilarity:
    """Test cosine similarity calculation."""
    
    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2, 3])
        result = cosine_similarity(vec1, vec2)
        assert abs(result - 1.0) < 1e-10
    
    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = np.array([1, 0])
        vec2 = np.array([0, 1])
        result = cosine_similarity(vec1, vec2)
        assert abs(result - 0.0) < 1e-10
    
    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([-1, -2, -3])
        result = cosine_similarity(vec1, vec2)
        assert abs(result - (-1.0)) < 1e-10
    
    def test_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 2, 3])
        result = cosine_similarity(vec1, vec2)
        assert result == 0.0
        
        # Both zero vectors
        result = cosine_similarity(vec1, vec1)
        assert result == 0.0
    
    def test_known_similarity(self):
        """Test cosine similarity with known expected result."""
        vec1 = np.array([1, 0, 1])
        vec2 = np.array([1, 1, 0])
        expected = 1 / (np.sqrt(2) * np.sqrt(2))  # 0.5
        result = cosine_similarity(vec1, vec2)
        assert abs(result - expected) < 1e-10


class TestClassifyPriority:
    """Test priority classification."""
    
    def test_high_priority(self):
        """Test high priority classification."""
        assert classify_priority(0.8) == "high"
        assert classify_priority(0.7) == "high"  # Boundary case
        assert classify_priority(1.0) == "high"
    
    def test_medium_priority(self):
        """Test medium priority classification."""
        assert classify_priority(0.6) == "medium"
        assert classify_priority(0.4) == "medium"  # Boundary case
        assert classify_priority(0.5) == "medium"
    
    def test_low_priority(self):
        """Test low priority classification."""
        assert classify_priority(0.3) == "low"
        assert classify_priority(0.0) == "low"
        assert classify_priority(0.39) == "low"
    
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        assert classify_priority(0.6, high_threshold=0.5, medium_threshold=0.3) == "high"
        assert classify_priority(0.4, high_threshold=0.5, medium_threshold=0.3) == "medium"
        assert classify_priority(0.2, high_threshold=0.5, medium_threshold=0.3) == "low"


class TestPrioritizeGaps:
    """Test skill gap prioritization."""
    
    @pytest.fixture
    def mock_embedding_func(self):
        """Create a mock embedding function with predictable outputs."""
        def embedding_func(skill: str):
            # Create deterministic embeddings based on skill name
            embeddings = {
                "Python": [1.0, 0.0, 0.0],
                "SQL": [0.0, 1.0, 0.0],
                "Machine Learning": [0.8, 0.6, 0.0],  # Similar to Python
                "TensorFlow": [0.7, 0.0, 0.7],  # Somewhat similar to Python
                "JavaScript": [0.0, 0.0, 1.0],  # Orthogonal to others
                "Data Science": [0.6, 0.8, 0.0],  # Similar to SQL
                "Deep Learning": [0.9, 0.4, 0.2],  # Similar to Python/ML
            }
            return embeddings.get(skill, [0.1, 0.1, 0.1])  # Default for unknown skills
        return embedding_func
    
    def test_empty_market_skills(self, mock_embedding_func):
        """Test with empty market skills list."""
        result = prioritize_gaps(["Python", "SQL"], [], mock_embedding_func)
        assert result == []
    
    def test_empty_candidate_skills(self, mock_embedding_func):
        """Test with empty candidate skills list."""
        market_skills = ["Machine Learning", "TensorFlow", "Machine Learning"]
        result = prioritize_gaps([], market_skills, mock_embedding_func)
        
        assert len(result) == 2  # Two unique skills
        
        # Check that Machine Learning has higher score due to frequency
        ml_item = next(item for item in result if item['skill'] == 'Machine Learning')
        tf_item = next(item for item in result if item['skill'] == 'TensorFlow')
        
        assert ml_item['score'] > tf_item['score']  # ML appears twice, TF once
        assert all(item['priority'] in ['high', 'medium', 'low'] for item in result)
    
    def test_basic_gap_detection(self, mock_embedding_func):
        """Test basic gap detection with known similarities."""
        candidate_skills = ["Python"]
        market_skills = ["Machine Learning", "JavaScript"]
        
        result = prioritize_gaps(candidate_skills, market_skills, mock_embedding_func)
        
        assert len(result) == 2
        
        # JavaScript should have higher gap score (more orthogonal to Python)
        js_item = next(item for item in result if item['skill'] == 'JavaScript')
        ml_item = next(item for item in result if item['skill'] == 'Machine Learning')
        
        assert js_item['score'] > ml_item['score']
    
    def test_frequency_weighting(self, mock_embedding_func):
        """Test that frequency weighting works correctly."""
        candidate_skills = ["Python"]
        market_skills = ["JavaScript", "TensorFlow", "JavaScript", "JavaScript"]  # JS appears 3x
        
        result = prioritize_gaps(candidate_skills, market_skills, mock_embedding_func)
        
        js_item = next(item for item in result if item['skill'] == 'JavaScript')
        tf_item = next(item for item in result if item['skill'] == 'TensorFlow')
        
        # JavaScript should have higher score due to frequency weighting
        assert js_item['score'] > tf_item['score']
    
    def test_deterministic_sorting(self, mock_embedding_func):
        """Test that output is consistently sorted."""
        candidate_skills = ["Python"]
        market_skills = ["A", "B", "C"]  # Skills with same embeddings should sort by name
        
        # Run multiple times to ensure consistent ordering
        results = []
        for _ in range(5):
            result = prioritize_gaps(candidate_skills, market_skills, mock_embedding_func)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0]
        
        # Results should be sorted by score (desc) then skill name (asc)
        result = results[0]
        for i in range(len(result) - 1):
            current = result[i]
            next_item = result[i + 1]
            
            # Either higher score, or same score with earlier alphabetical order
            assert (current['score'] > next_item['score'] or
                   (current['score'] == next_item['score'] and current['skill'] <= next_item['skill']))
    
    def test_score_precision(self, mock_embedding_func):
        """Test that scores are rounded to 3 decimal places."""
        candidate_skills = ["Python"]
        market_skills = ["Machine Learning"]
        
        result = prioritize_gaps(candidate_skills, market_skills, mock_embedding_func)
        
        assert len(str(result[0]['score']).split('.')[-1]) <= 3
    
    def test_all_skills_covered(self, mock_embedding_func):
        """Test that candidate already has all market skills."""
        candidate_skills = ["Python", "SQL", "Machine Learning"]
        market_skills = ["Python", "SQL"]  # All skills candidate already has
        
        result = prioritize_gaps(candidate_skills, market_skills, mock_embedding_func)
        
        # Should still return results, but with low scores
        assert len(result) == 2
        assert all(item['score'] < 0.5 for item in result)  # Low gap scores
    
    def test_embedding_error_handling(self):
        """Test handling of embedding function errors."""
        def failing_embed_func(skill):
            if skill == "BadSkill":
                raise ValueError("Cannot embed this skill")
            return [1.0, 0.0, 0.0]
        
        candidate_skills = ["Python"]
        market_skills = ["GoodSkill", "BadSkill"]
        
        result = prioritize_gaps(candidate_skills, market_skills, failing_embed_func)
        
        # Should only include skills that can be embedded
        assert len(result) == 1
        assert result[0]['skill'] == "GoodSkill"
    
    def test_real_world_scenario(self, mock_embedding_func):
        """Test with realistic skill gap scenario."""
        candidate_skills = ["Python", "SQL"]
        market_skills = [
            "Machine Learning", "Deep Learning", "TensorFlow",
            "Python", "Data Science", "Machine Learning",  # ML appears twice
            "JavaScript", "SQL"
        ]
        
        result = prioritize_gaps(candidate_skills, market_skills, mock_embedding_func)
        
        # Should identify gaps (not Python/SQL which candidate has)
        gap_skills = {item['skill'] for item in result}
        
        # Python and SQL should have very low scores since candidate already has them
        python_items = [item for item in result if item['skill'] == "Python"]
        sql_items = [item for item in result if item['skill'] == "SQL"]
        
        if python_items:
            assert python_items[0]['score'] < 0.1  # Very low gap
        if sql_items:
            assert sql_items[0]['score'] < 0.1  # Very low gap
        
        # JavaScript should be high priority (most orthogonal)
        js_items = [item for item in result if item['skill'] == 'JavaScript']
        if js_items:
            assert js_items[0]['priority'] == 'high'
            assert js_items[0]['score'] > 0.8
        
        # Machine Learning should have some score due to frequency but low due to similarity to Python
        ml_items = [item for item in result if item['skill'] == 'Machine Learning']
        if ml_items:
            assert 0.2 <= ml_items[0]['score'] <= 0.3  # Some gap but not huge due to Python similarity
            # ML should score higher than Deep Learning due to frequency (appears twice)
            dl_items = [item for item in result if item['skill'] == 'Deep Learning']
            if dl_items:
                assert ml_items[0]['score'] > dl_items[0]['score']
        
        # Should have proper structure
        for item in result:
            assert 'skill' in item
            assert 'priority' in item
            assert 'score' in item
            assert item['priority'] in ['high', 'medium', 'low']
            assert 0.0 <= item['score'] <= 1.0