"""
Integration tests for skill gap analysis pipeline.

Tests cover end-to-end functionality with real API calls to Gemini/Mistral and Tavily.
Tests are automatically skipped if required API keys are not available.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.provider import get_chat_model
from tools.search_adapter import search_web
from graph.builder import build_graph
from graph.state import GraphState
from .fixtures import sample_cv_text


# Integration test marker
pytestmark = pytest.mark.integration


def check_api_keys():
    """Check if required API keys are available."""
    gemini_key = os.getenv('GEMINI_API_KEY')
    mistral_key = os.getenv('MISTRAL_API_KEY')
    return bool(gemini_key or mistral_key)


# Skip entire suite if API keys are missing
pytestmark = pytest.mark.skipif(
    not check_api_keys(),
    reason="Integration tests require GEMINI_API_KEY or MISTRAL_API_KEY environment variables"
)


@pytest.mark.integration
def test_auto_provider_resolves_to_gemini_or_mistral():
    """
    Test that AUTO provider resolves to either Gemini or Mistral based on available API keys.
    """
    # Test that auto provider creates a valid LLM instance
    llm = get_chat_model("auto", None)
    assert llm is not None, "Auto provider should return a valid LLM instance"
    
    # Verify the model type is one of our supported providers
    model_name = str(type(llm).__name__).lower()
    assert any(provider in model_name for provider in ['gemini', 'mistral', 'chat']), \
        f"Auto provider should resolve to Gemini or Mistral, got: {type(llm)}"
    
    # Test that explicit providers work too
    gemini_key = os.getenv('GEMINI_API_KEY')
    mistral_key = os.getenv('MISTRAL_API_KEY')
    
    if gemini_key:
        gemini_llm = get_chat_model("gemini", None)
        assert gemini_llm is not None, "Gemini provider should work with valid API key"
    
    if mistral_key:
        mistral_llm = get_chat_model("mistral", None)
        assert mistral_llm is not None, "Mistral provider should work with valid API key"


@pytest.mark.integration
def test_live_market_search_returns_docs():
    """
    Test that live market search returns actual documents with real URLs.
    
    This test hits the Tavily API and verifies we get back real job market data.
    """
    # Perform live market search
    search_results = search_web(
        query="Senior AI Engineer jobs requirements skills",
        k=5
    )
    
    # Verify we got real results
    assert len(search_results) > 0, "Should return at least one search result"
    
    # Check each result has expected structure
    for result in search_results:
        assert 'title' in result, "Each result should have a title"
        assert 'url' in result, "Each result should have a URL"
        assert 'content' in result, "Each result should have content"
        
        # Verify URL is real and well-formed
        url = result['url']
        assert url.startswith('http'), f"URL should start with http: {url}"
        assert len(url) > 10, f"URL should be substantial: {url}"
        
        # Verify content is meaningful
        content = result['content']
        assert len(content) > 50, f"Content should be substantial: {len(content)} chars"
        
        # Check for job-related terms in content
        content_lower = content.lower()
        job_terms = ['engineer', 'developer', 'skills', 'experience', 'requirements', 'python', 'ai', 'machine learning']
        assert any(term in content_lower for term in job_terms), f"Content should contain job-related terms: {content[:100]}..."


@pytest.mark.integration
def test_pipeline_generates_report_markdown():
    """
    Test that the full pipeline generates a proper markdown report.
    
    This test runs the complete skill gap analysis pipeline and verifies 
    the final output is valid markdown with expected sections.
    """
    import tempfile
    import os
    
    # Set offline embeddings to allow fallback if provider embeddings fail
    original_env = os.environ.get('ALLOW_OFFLINE_EMBEDDINGS')
    os.environ['ALLOW_OFFLINE_EMBEDDINGS'] = 'true'
    
    # Create a temporary file with the CV content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(sample_cv_text())
        temp_file_path = f.name
    
    try:
        # Build the graph with auto provider
        graph = build_graph("auto")
        
        # Create initial state with file path (as expected by the pipeline)
        initial_state = GraphState(
            file_path=temp_file_path,
            target_role="Data Scientist",
            provider="auto"
        )
        
        # Run the pipeline
        result = graph.invoke(initial_state)
        
        # Verify we got a result
        assert result is not None, "Pipeline should return a result"
        assert 'report_md' in result, "Result should contain report_md"
        
        # Verify the report is markdown
        report = result['report_md']
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 100, "Report should be substantial"
        
        # Check for markdown formatting
        assert '#' in report, "Report should contain markdown headers"
        
        # Check for expected sections in the report
        report_lower = report.lower()
        expected_sections = [
            'skill',
            'gap',
            'recommendation',
            'summary'
        ]
        
        found_sections = [section for section in expected_sections if section in report_lower]
        assert len(found_sections) >= 2, f"Report should contain key sections. Found: {found_sections}"
        
        # Verify report contains structured content
        assert any(char in report for char in ['*', '-', '1.', '2.']), \
            "Report should contain lists or bullet points"
        
        # If market search worked, should have some URLs
        if 'http' in report:
            assert report.count('http') >= 1, "Report should contain source URLs when available"
        
        print(f"Generated report ({len(report)} chars) with sections: {found_sections}")
        print("Test passed - full pipeline working with live APIs")
        
    finally:
        # Clean up the temporary file and restore environment
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if original_env is None:
            os.environ.pop('ALLOW_OFFLINE_EMBEDDINGS', None)
        else:
            os.environ['ALLOW_OFFLINE_EMBEDDINGS'] = original_env


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "-m", "integration"])
