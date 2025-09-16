"""
Tests for skill taxonomy functionality.
"""

import numpy as np
import pytest
from unittest.mock import Mock

from src.tools.taxonomy import (
    canonicalize_skills,
    closest_market_skills,
    get_skill_categories,
    get_canonical_skills,
    suggest_skills,
    _clean_skill_text,
    CANONICAL_SKILLS,
    ALIAS_TO_CANONICAL
)


class TestCleanSkillText:
    """Test skill text cleaning functionality."""
    
    def test_clean_basic_text(self):
        """Test basic text cleaning."""
        assert _clean_skill_text("Python") == "python"
        assert _clean_skill_text("  TENSORFLOW  ") == "tensorflow"
        assert _clean_skill_text("Machine Learning") == "machine learning"
    
    def test_clean_with_noise(self):
        """Test cleaning text with noise patterns."""
        assert _clean_skill_text("Python programming experience") == "python"
        assert _clean_skill_text("5 years of TensorFlow") == "tensorflow"
        assert _clean_skill_text("Strong knowledge in ML") == "machine learning"
        assert _clean_skill_text("Proficient with Docker containers") == "docker containers"
    
    def test_clean_special_characters(self):
        """Test handling of special characters."""
        assert _clean_skill_text("C++") == "c++"
        assert _clean_skill_text("C#") == "c#" 
        assert _clean_skill_text("Node.js") == "javascript"  # Normalized
        assert _clean_skill_text("AWS/GCP") == "aws/gcp"
    
    def test_clean_empty_and_invalid(self):
        """Test handling of empty and invalid inputs."""
        assert _clean_skill_text("") == ""
        assert _clean_skill_text(None) == ""
        assert _clean_skill_text("   ") == ""
    
    def test_clean_abbreviations(self):
        """Test expansion of common abbreviations."""
        assert _clean_skill_text("ML") == "machine learning"
        assert _clean_skill_text("AI") == "artificial intelligence"
        assert _clean_skill_text("DS") == "data science"
        assert _clean_skill_text("DE") == "data engineering"


class TestCanonicalizeSkills:
    """Test skill canonicalization functionality."""
    
    def test_canonicalize_empty_list(self):
        """Test canonicalization of empty list."""
        assert canonicalize_skills([]) == []
        assert canonicalize_skills([None, "", "  "]) == []
    
    def test_canonicalize_exact_matches(self):
        """Test canonicalization with exact alias matches."""
        skills = ["Python", "TensorFlow", "tf", "PYTHON"]
        result = canonicalize_skills(skills)
        assert "python" in result
        assert "tensorflow" in result
        assert len([s for s in result if s == "python"]) == 1  # No duplicates
    
    def test_canonicalize_rag_variations(self):
        """Test RAG and related variations."""
        skills = ["RAG", "retrieval augmented generation", "Retrieval-Augmented Generation"]
        result = canonicalize_skills(skills)
        assert "rag" in result
        assert len(result) == 1  # All should map to same canonical
    
    def test_canonicalize_vector_db_variations(self):
        """Test vector database variations."""
        skills = ["vector db", "ChromaDB", "Qdrant", "vector database"]
        result = canonicalize_skills(skills)
        expected = {"vector databases", "chroma", "qdrant"}
        assert expected.issubset(set(result))
    
    def test_canonicalize_observability_variations(self):
        """Test observability and monitoring variations."""
        skills = ["observability", "tracing", "prompt telemetry", "monitoring"]
        result = canonicalize_skills(skills)
        expected = {"observability", "prompt telemetry"}
        assert expected.issubset(set(result))
    
    def test_canonicalize_with_noise(self):
        """Test canonicalization with noisy input."""
        skills = [
            "5 years Python programming experience",
            "Strong knowledge in TensorFlow",
            "Working with Docker containers",
            "Proficient in ML algorithms"
        ]
        result = canonicalize_skills(skills)
        expected = {"python", "tensorflow", "docker", "machine learning"}
        assert expected.issubset(set(result))
    
    def test_canonicalize_unknown_skills(self):
        """Test canonicalization with unknown skills."""
        skills = ["Python", "SomeUnknownFramework", "TensorFlow"]
        result = canonicalize_skills(skills)
        assert "python" in result
        assert "tensorflow" in result
        assert "someunknownframework" in result  # Unknown skills are kept as cleaned
    
    def test_canonicalize_partial_matches(self):
        """Test canonicalization with partial matches."""
        skills = ["pytorch lightning", "tensorflow keras", "huggingface transformers"]
        result = canonicalize_skills(skills)
        # Should match pytorch, tensorflow, huggingface
        expected = {"pytorch", "tensorflow", "huggingface"}
        assert expected.issubset(set(result))
    
    def test_canonicalize_deduplication(self):
        """Test that duplicates are properly removed."""
        skills = ["Python", "python", "PYTHON", "py", "python3"]
        result = canonicalize_skills(skills)
        assert result.count("python") == 1
        assert len(result) == 1


class TestClosestMarketSkills:
    """Test market skill matching functionality."""
    
    def create_mock_embed_func(self, similarity_matrix: np.ndarray):
        """Create a mock embedding function that returns predictable results."""
        def mock_embed(texts):
            # Return embeddings that will produce the given similarity matrix
            n = len(texts)
            return np.random.rand(n, 384).astype(np.float32)
        return mock_embed
    
    def test_closest_market_skills_empty_inputs(self):
        """Test with empty inputs."""
        embed_func = lambda x: np.array([])
        
        assert closest_market_skills([], ["python"], embed_func) == {}
        assert closest_market_skills(["python"], [], embed_func) == {}
        assert closest_market_skills([], [], embed_func) == {}
    
    def test_closest_market_skills_exact_matches(self):
        """Test with exact skill matches."""
        def mock_embed(texts):
            # Create embeddings where identical texts have similarity 1.0
            embeddings = []
            for text in texts:
                if text == "python":
                    embeddings.append([1.0, 0.0, 0.0])
                elif text == "tensorflow":
                    embeddings.append([0.0, 1.0, 0.0])
                else:
                    embeddings.append([0.0, 0.0, 1.0])
            return np.array(embeddings, dtype=np.float32)
        
        candidate_skills = ["python", "tensorflow"]
        market_skills = ["python", "machine learning"]
        
        result = closest_market_skills(candidate_skills, market_skills, mock_embed)
        
        assert "python" in result
        assert result["python"]["match"] == "python"
        assert result["python"]["similarity"] > 0.9  # Should be high similarity
        assert "strong match" in result["python"]["evidence"]
    
    def test_closest_market_skills_partial_matches(self):
        """Test with partial/similar skill matches."""
        def mock_embed(texts):
            # Create embeddings with moderate similarities
            embeddings = []
            for text in texts:
                if "python" in text.lower():
                    embeddings.append([1.0, 0.1, 0.1])
                elif "machine learning" in text.lower() or "tensorflow" in text.lower():
                    embeddings.append([0.1, 1.0, 0.1])
                else:
                    embeddings.append([0.1, 0.1, 1.0])
            return np.array(embeddings, dtype=np.float32)
        
        candidate_skills = ["python programming", "tensorflow"]
        market_skills = ["python", "machine learning frameworks"]
        
        result = closest_market_skills(candidate_skills, market_skills, mock_embed)
        
        assert len(result) == 2
        assert "python" in result
        assert "machine learning frameworks" in result or "machine learning" in result
    
    def test_closest_market_skills_with_noise(self):
        """Test robustness to noisy input."""
        def mock_embed(texts):
            # Simple mock that returns random but consistent embeddings
            np.random.seed(42)  # For reproducibility
            return np.random.rand(len(texts), 10).astype(np.float32)
        
        candidate_skills = [
            "5 years Python programming experience",
            "Strong knowledge in TensorFlow",
            "Working with Docker containers"
        ]
        market_skills = [
            "Python development",
            "Machine Learning frameworks",
            "DevOps tools"
        ]
        
        result = closest_market_skills(candidate_skills, market_skills, mock_embed)
        
        # Should return some mapping for each market skill
        assert len(result) > 0
        for market_skill, match_info in result.items():
            assert "match" in match_info
            assert "similarity" in match_info
            assert "evidence" in match_info
            assert isinstance(match_info["similarity"], float)
    
    def test_closest_market_skills_embedding_failure(self):
        """Test handling of embedding function failures."""
        def failing_embed(texts):
            raise Exception("Embedding failed")
        
        candidate_skills = ["python"]
        market_skills = ["python"]
        
        result = closest_market_skills(candidate_skills, market_skills, failing_embed)
        assert result == {}  # Should return empty dict on failure


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_skill_categories(self):
        """Test skill categories function."""
        categories = get_skill_categories()
        
        assert isinstance(categories, dict)
        assert "programming_languages" in categories
        assert "ml_frameworks" in categories
        assert "llm_nlp" in categories
        assert "vector_databases" in categories
        
        # Check that python is in programming languages
        assert "python" in categories["programming_languages"]
        assert "tensorflow" in categories["ml_frameworks"]
        assert "rag" in categories["llm_nlp"]
        assert "chroma" in categories["vector_databases"]
    
    def test_get_canonical_skills(self):
        """Test canonical skills retrieval."""
        skills = get_canonical_skills()
        
        assert isinstance(skills, dict)
        assert len(skills) > 0
        assert "python" in skills
        assert "tensorflow" in skills
        assert "rag" in skills
        
        # Should be a copy, not reference
        skills["test"] = ["test"]
        assert "test" not in CANONICAL_SKILLS
    
    def test_suggest_skills_empty_input(self):
        """Test skill suggestions with empty input."""
        assert suggest_skills("") == []
        assert suggest_skills("a") == []  # Too short
        assert suggest_skills(None) == []
    
    def test_suggest_skills_exact_start(self):
        """Test skill suggestions with exact start matches."""
        suggestions = suggest_skills("py")
        assert "python" in suggestions
        
        suggestions = suggest_skills("tensor")
        assert "tensorflow" in suggestions
    
    def test_suggest_skills_partial_match(self):
        """Test skill suggestions with partial matches."""
        suggestions = suggest_skills("learn")
        assert any("learning" in s for s in suggestions)
        
        suggestions = suggest_skills("vector")
        assert "vector databases" in suggestions
    
    def test_suggest_skills_limit(self):
        """Test skill suggestions limit."""
        suggestions = suggest_skills("data", limit=3)
        assert len(suggestions) <= 3
        
        suggestions = suggest_skills("ma", limit=5)
        assert len(suggestions) <= 5


class TestTaxonomyRobustness:
    """Test robustness to various edge cases and noise."""
    
    def test_mixed_case_and_spacing(self):
        """Test handling of mixed case and spacing."""
        skills = ["  PyTorch  ", "TENSOR FLOW", "node.JS", "c++"]
        result = canonicalize_skills(skills)
        
        expected = {"pytorch", "tensorflow", "javascript", "c++"}
        assert expected.issubset(set(result))
    
    def test_special_characters_and_numbers(self):
        """Test handling of special characters and numbers."""
        skills = ["Python3.9", "TensorFlow 2.0", "C#", "R-language", "Node.js v16"]
        result = canonicalize_skills(skills)
        
        # Should still recognize core technologies
        assert "python" in result
        assert "tensorflow" in result
        assert "javascript" in result  # Node.js should map to javascript
        assert "r" in result
    
    def test_long_descriptive_text(self):
        """Test handling of long descriptive text."""
        skills = [
            "Advanced proficiency in Python programming with 5+ years experience",
            "Deep learning using TensorFlow and PyTorch frameworks",
            "Extensive knowledge of RAG (Retrieval-Augmented Generation) systems",
            "Expert in vector databases including ChromaDB and Qdrant"
        ]
        result = canonicalize_skills(skills)
        
        expected = {"python", "tensorflow", "pytorch", "rag", "chroma", "qdrant"}
        assert expected.issubset(set(result))
    
    def test_acronyms_and_abbreviations(self):
        """Test handling of acronyms and abbreviations."""
        skills = ["ML", "AI", "NLP", "CV", "MLOps", "LLM", "RAG", "API"]
        result = canonicalize_skills(skills)
        
        # Should expand some abbreviations
        assert "machine learning" in result  # ML
        assert "artificial intelligence" in result  # AI
        assert "nlp" in result
        assert "rag" in result
        assert "large language models" in result  # LLM
    
    def test_typos_and_variations(self):
        """Test handling of common typos and variations."""
        skills = ["Pythong", "Tensorflow", "Pytorch", "Langchain"]
        result = canonicalize_skills(skills)
        
        # Even with typos, some should be recognized through partial matching
        # At minimum, cleaned versions should be preserved
        assert len(result) > 0
        assert all(isinstance(skill, str) for skill in result)


class TestIntegrationWithEmbeddings:
    """Test integration with the embeddings module."""
    
    def test_with_mock_embeddings(self):
        """Test closest_market_skills with mock embeddings that simulate real behavior."""
        # Create a more realistic mock embedding function
        def realistic_embed(texts):
            # Simulate embeddings where similar terms have higher similarity
            embeddings = []
            for text in texts:
                # Create different embedding patterns for different skill types
                if "python" in text.lower():
                    base = [1.0, 0.0, 0.0, 0.0]
                elif any(term in text.lower() for term in ["tensorflow", "pytorch", "machine learning"]):
                    base = [0.2, 1.0, 0.0, 0.0]
                elif any(term in text.lower() for term in ["docker", "kubernetes", "devops"]):
                    base = [0.0, 0.0, 1.0, 0.0]
                else:
                    base = [0.0, 0.0, 0.0, 1.0]
                
                # Add some noise
                embedding = np.array(base) + np.random.normal(0, 0.1, 4)
                embeddings.append(embedding)
            
            return np.array(embeddings, dtype=np.float32)
        
        candidate_skills = [
            "Python development",
            "TensorFlow deep learning",
            "Docker containerization"
        ]
        market_skills = [
            "Python programming",
            "Machine learning frameworks",
            "DevOps tools"
        ]
        
        result = closest_market_skills(candidate_skills, market_skills, realistic_embed)
        
        # Should have mappings for all market skills
        assert len(result) == 3
        
        # Python should map to Python with high similarity
        python_match = next((v for k, v in result.items() if "python" in k.lower()), None)
        assert python_match is not None
        assert "python" in python_match["match"].lower()
        
        # Check that all results have required fields
        for market_skill, match_info in result.items():
            assert "match" in match_info
            assert "similarity" in match_info
            assert "evidence" in match_info
            assert 0 <= match_info["similarity"] <= 1


if __name__ == "__main__":
    pytest.main([__file__])