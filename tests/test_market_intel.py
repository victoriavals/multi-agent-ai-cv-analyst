"""Tests for market intelligence agent functionality."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from src.agents.market_intel import (
    gather_market_summary,
    build_market_queries,
    _analyze_rule_based,
    _prepare_search_summary
)
from src.graph.state import MarketSummary


def test_build_market_queries():
    """Test market query building."""
    queries = build_market_queries("Senior AI Engineer", "Global")
    
    # Should return multiple queries
    assert len(queries) > 0
    assert len(queries) <= 8  # Should be limited
    
    # Should include role variations
    query_text = ' '.join(queries).lower()
    assert 'ai engineer' in query_text
    assert 'skills' in query_text
    assert 'requirements' in query_text


def test_build_market_queries_regional():
    """Test query building with specific region."""
    queries = build_market_queries("Data Scientist", "US")
    
    query_text = ' '.join(queries).lower()
    assert 'data scientist' in query_text
    assert 'us' in query_text or 'united states' in query_text


def test_prepare_search_summary():
    """Test search results summary preparation."""
    search_results = [
        {
            'title': 'Senior AI Engineer Job Requirements',
            'url': 'https://example.com/job1',
            'content': 'Required skills: Python, TensorFlow, Kubernetes\nExperience with machine learning models'
        },
        {
            'title': 'AI Engineer Hiring Trends 2024',
            'url': 'https://example.com/trends',
            'content': 'Top technologies: Docker, AWS, SQL\nFrameworks: PyTorch, React'
        }
    ]
    
    summary = _prepare_search_summary(search_results, "AI Engineer", "Global")
    
    assert 'AI Engineer' in summary
    assert 'Global' in summary
    assert 'Senior AI Engineer Job Requirements' in summary
    assert 'Python' in summary
    assert 'TensorFlow' in summary


def test_analyze_rule_based():
    """Test rule-based market analysis."""
    search_results = [
        {
            'title': 'AI Engineer Jobs',
            'content': 'Required: Python, TensorFlow, AWS, Docker, Kubernetes, SQL'
        },
        {
            'title': 'ML Engineer Requirements',
            'content': 'Skills needed: PyTorch, React, PostgreSQL, Git, machine learning'
        }
    ]
    
    analysis = _analyze_rule_based(search_results, "AI Engineer")
    
    # Should have all required categories
    assert 'in_demand_skills' in analysis
    assert 'common_tools' in analysis
    assert 'frameworks' in analysis
    assert 'nice_to_have' in analysis
    
    # Should find common skills
    in_demand = [s.lower() for s in analysis['in_demand_skills']]
    assert 'python' in in_demand
    assert 'machine learning' in in_demand
    
    # Should categorize tools
    tools = [s.lower() for s in analysis['common_tools']]
    assert 'aws' in tools
    
    # Should find frameworks
    frameworks = [s.lower() for s in analysis['frameworks']]
    assert any(fw in frameworks for fw in ['tensorflow', 'pytorch'])


def test_analyze_rule_based_no_content():
    """Test rule-based analysis with no content (uses defaults)."""
    analysis = _analyze_rule_based([], "AI Engineer")
    
    # Should still return valid structure with defaults
    assert len(analysis['in_demand_skills']) > 0
    assert 'python' in [s.lower() for s in analysis['in_demand_skills']]
    assert 'machine learning' in [s.lower() for s in analysis['in_demand_skills']]


def test_gather_market_summary_no_llm():
    """Test market summary gathering without LLM."""
    with patch('src.agents.market_intel.search_web') as mock_search:
        mock_search.return_value = [
            {
                'title': 'Senior ML Engineer Job',
                'url': 'https://example.com/job',
                'content': 'Requirements: Python, TensorFlow, AWS, Docker, Kubernetes'
            }
        ]
        
        summary = gather_market_summary("Senior ML Engineer", "Global", llm=None)
        
        # Should be valid MarketSummary
        assert isinstance(summary, MarketSummary)
        assert summary.role == "Senior ML Engineer"
        assert summary.region == "Global"
        
        # Should have non-empty skills
        assert len(summary.in_demand_skills) > 0
        
        # Should have sources
        assert len(summary.sources_sample) > 0
        assert 'Senior ML Engineer Job' in summary.sources_sample[0]


def test_gather_market_summary_with_mock_llm():
    """Test market summary with mocked LLM."""
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = '''
    ```json
    {
        "in_demand_skills": ["python", "machine learning", "tensorflow", "kubernetes"],
        "common_tools": ["aws", "docker", "git", "jenkins"],
        "frameworks": ["tensorflow", "pytorch", "react"],
        "nice_to_have": ["mlflow", "airflow", "kafka"]
    }
    ```
    '''
    mock_llm.invoke.return_value = mock_response
    
    with patch('src.agents.market_intel.search_web') as mock_search:
        mock_search.return_value = [
            {
                'title': 'AI Engineer Market Analysis',
                'url': 'https://example.com/analysis',
                'content': 'Market trends for AI engineers...'
            }
        ]
        
        # Mock file reading for market prompt
        market_prompt = "You are a market intelligence analyst..."
        with patch('builtins.open', mock_open(read_data=market_prompt)):
            summary = gather_market_summary("AI Engineer", "Global", llm=mock_llm)
        
        # Should use LLM results
        assert 'python' in [s.lower() for s in summary.in_demand_skills]
        assert 'machine learning' in [s.lower() for s in summary.in_demand_skills]
        assert 'aws' in [s.lower() for s in summary.common_tools]
        assert 'tensorflow' in [s.lower() for s in summary.frameworks]


def test_gather_market_summary_llm_fallback():
    """Test fallback when LLM fails."""
    mock_llm = Mock()
    mock_llm.invoke.side_effect = Exception("LLM failed")
    
    with patch('src.agents.market_intel.search_web') as mock_search:
        mock_search.return_value = [
            {
                'title': 'Python Developer Jobs',
                'content': 'Skills: Python, SQL, AWS, Docker, Flask'
            }
        ]
        
        with patch('builtins.open', mock_open(read_data="prompt")):
            summary = gather_market_summary("Python Developer", "Global", llm=mock_llm)
        
        # Should still return valid summary using rule-based fallback
        assert isinstance(summary, MarketSummary)
        assert len(summary.in_demand_skills) > 0


def test_gather_market_summary_search_failure():
    """Test handling when search fails."""
    with patch('src.agents.market_intel.search_web') as mock_search:
        mock_search.side_effect = Exception("Search failed")
        
        summary = gather_market_summary("Data Engineer", "US", llm=None)
        
        # Should still return valid summary with defaults
        assert isinstance(summary, MarketSummary)
        assert summary.role == "Data Engineer"
        assert summary.region == "US"
        assert len(summary.in_demand_skills) > 0  # Should have defaults


def test_gather_market_summary_invalid_inputs():
    """Test error handling with invalid inputs."""
    with pytest.raises(ValueError, match="Role cannot be empty"):
        gather_market_summary("", "Global")
    
    with pytest.raises(ValueError, match="Region cannot be empty"):
        gather_market_summary("Engineer", "")
    
    with pytest.raises(ValueError, match="Role cannot be empty"):
        gather_market_summary(None, "Global")


def test_market_summary_canonicalization():
    """Test that skills are canonicalized through taxonomy."""
    with patch('src.agents.market_intel.search_web') as mock_search:
        mock_search.return_value = [
            {
                'title': 'Job Requirements',
                'content': 'Technologies: Python, JavaScript, Node.js, ML, AI'
            }
        ]
        
        # Mock canonicalize_skills to verify it's called
        with patch('src.agents.market_intel.canonicalize_skills') as mock_canon:
            mock_canon.side_effect = lambda x: [s.lower() for s in x]  # Simple canonicalization
            
            summary = gather_market_summary("Full Stack Engineer", "Global", llm=None)
            
            # Should call canonicalize_skills for each category
            assert mock_canon.call_count >= 4  # Once for each skill category


def test_market_queries_role_variations():
    """Test that query building handles role variations correctly."""
    # Test senior role variations
    queries = build_market_queries("Senior Software Engineer", "Global")
    query_text = ' '.join(queries).lower()
    assert any(variant in query_text for variant in ['sr software', 'lead software'])
    
    # Test AI role variations
    queries = build_market_queries("AI Engineer", "Global")
    query_text = ' '.join(queries).lower()
    assert 'machine learning' in query_text
    
    # Test engineer variations
    queries = build_market_queries("Backend Engineer", "Europe")
    query_text = ' '.join(queries).lower()
    assert 'backend developer' in query_text or 'developer' in query_text


def test_search_content_extraction():
    """Test that relevant content is properly extracted from search results."""
    search_results = [
        {
            'title': 'Job Posting: Senior Developer',
            'content': '''
            About the role:
            We are looking for a senior developer.
            
            Required skills:
            - Python programming
            - SQL database experience
            - Cloud platform knowledge
            
            Nice to have:
            - Kubernetes experience
            - Machine learning background
            '''
        }
    ]
    
    summary = _prepare_search_summary(search_results, "Senior Developer", "Global")
    
    # Should extract skill-related content
    assert 'Required skills' in summary or 'python' in summary.lower()
    assert 'Senior Developer' in summary


if __name__ == '__main__':
    pytest.main([__file__])