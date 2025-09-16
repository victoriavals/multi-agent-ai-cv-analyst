"""
Tests for search adapter functionality.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tools.search_adapter import (
    ContentExtractor,
    TavilySearchProvider,
    SerpAPISearchProvider, 
    MockSearchProvider,
    get_search_provider,
    search_web
)


class TestContentExtractor:
    """Test content extraction functionality."""
    
    def test_extract_content_with_trafilatura_success(self):
        """Test successful content extraction using trafilatura."""
        extractor = ContentExtractor()
        
        with patch('trafilatura.extract') as mock_trafilatura, \
             patch.object(extractor.client, 'get') as mock_get:
            
            # Mock HTTP response
            mock_response = Mock()
            mock_response.text = '<html><body><p>Test content</p></body></html>'
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Mock trafilatura success - returns None (short content triggers fallback)
            mock_trafilatura.return_value = None
            
            with patch('src.tools.search_adapter.Document') as mock_document:
                # Mock readability success
                mock_doc_instance = Mock()
                mock_doc_instance.summary.return_value = '<p>Test content</p>'
                mock_document.return_value = mock_doc_instance
                
                result = extractor.extract_content("https://example.com")
                
                assert "Test content" in result
    
    def test_extract_content_fallback_to_readability(self):
        """Test fallback to readability when trafilatura fails."""
        extractor = ContentExtractor()
        
        with patch('trafilatura.extract') as mock_trafilatura, \
             patch('src.tools.search_adapter.Document') as mock_document, \
             patch.object(extractor.client, 'get') as mock_get:
            
            # Mock HTTP response
            mock_response = Mock()
            mock_response.text = '<html><body><p>Test content</p></body></html>'
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Mock trafilatura failure (returns short content)
            mock_trafilatura.return_value = "Short"
            
            # Mock readability success
            mock_doc_instance = Mock()
            mock_doc_instance.summary.return_value = '<p>Readability extracted content</p>'
            mock_document.return_value = mock_doc_instance
            
            result = extractor.extract_content("https://example.com")
            
            assert "Readability extracted content" in result
    
    def test_extract_content_handles_http_error(self):
        """Test handling of HTTP errors during content extraction."""
        extractor = ContentExtractor()
        
        with patch.object(extractor.client, 'get') as mock_get:
            mock_get.side_effect = Exception("HTTP Error")
            
            result = extractor.extract_content("https://example.com")
            
            assert "Content extraction failed" in result


class TestMockSearchProvider:
    """Test mock search provider."""
    
    def test_mock_search_returns_ai_engineer_content(self):
        """Test that mock provider returns relevant AI engineer content."""
        provider = MockSearchProvider()
        
        results = provider.search("AI engineer skills", 5)
        
        assert len(results) == 5
        for result in results:
            assert 'title' in result
            assert 'url' in result  
            assert 'content' in result
            assert 'AI' in result['content'] or 'engineer' in result['content'].lower()
    
    def test_mock_search_respects_num_results_limit(self):
        """Test that mock provider respects num_results parameter."""
        provider = MockSearchProvider()
        
        results = provider.search("test query", 3)
        
        assert len(results) == 3
        
        results = provider.search("test query", 10)
        
        # Should return all available mock results (5)
        assert len(results) == 5


class TestTavilySearchProvider:
    """Test Tavily search provider."""
    
    @patch('tavily.TavilyClient')
    def test_tavily_search_success(self, mock_tavily_client):
        """Test successful Tavily search."""
        # Mock Tavily client
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        
        # Mock search response
        mock_response = {
            'results': [
                {
                    'title': 'Test Article',
                    'url': 'https://example.com/article',
                    'content': 'This is a test article with sufficient content for testing purposes.'
                }
            ]
        }
        mock_client_instance.search.return_value = mock_response
        
        provider = TavilySearchProvider("test-api-key")
        results = provider.search("test query", 1)
        
        assert len(results) == 1
        assert results[0]['title'] == 'Test Article'
        assert results[0]['url'] == 'https://example.com/article'
        assert results[0]['content'] == 'This is a test article with sufficient content for testing purposes.'
    
    @patch('tavily.TavilyClient')
    def test_tavily_search_extracts_content_when_missing(self, mock_tavily_client):
        """Test that Tavily provider extracts content when not provided."""
        # Mock Tavily client
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        
        # Mock search response with short/missing content
        mock_response = {
            'results': [
                {
                    'title': 'Test Article',
                    'url': 'https://example.com/article',
                    'content': 'Short'  # Too short, should trigger extraction
                }
            ]
        }
        mock_client_instance.search.return_value = mock_response
        
        provider = TavilySearchProvider("test-api-key")
        
        # Mock content extraction
        with patch.object(provider.content_extractor, 'extract_content') as mock_extract:
            mock_extract.return_value = "Extracted longer content from the URL"
            
            results = provider.search("test query", 1)
            
            assert results[0]['content'] == "Extracted longer content from the URL"
            mock_extract.assert_called_once_with('https://example.com/article')


class TestSerpAPISearchProvider:
    """Test SerpAPI search provider."""
    
    @patch('serpapi.GoogleSearch')
    def test_serpapi_search_success(self, mock_google_search):
        """Test successful SerpAPI search."""
        # Mock GoogleSearch instance
        mock_search_instance = Mock()
        mock_google_search.return_value = mock_search_instance
        
        # Mock search response
        mock_response = {
            'organic_results': [
                {
                    'title': 'Test Result',
                    'link': 'https://example.com/page',
                    'snippet': 'This is a test snippet'
                }
            ]
        }
        mock_search_instance.get_dict.return_value = mock_response
        
        provider = SerpAPISearchProvider("test-api-key")
        
        # Mock content extraction
        with patch.object(provider.content_extractor, 'extract_content') as mock_extract:
            mock_extract.return_value = "Extracted content from the page"
            
            results = provider.search("test query", 1)
            
            assert len(results) == 1
            assert results[0]['title'] == 'Test Result'
            assert results[0]['url'] == 'https://example.com/page'
            assert 'This is a test snippet' in results[0]['content']
            assert 'Extracted content from the page' in results[0]['content']


class TestSearchProviderSelection:
    """Test search provider selection logic."""
    
    def test_get_provider_tavily_with_key(self):
        """Test provider selection returns Tavily when configured."""
        with patch.dict(os.environ, {
            'SEARCH_PROVIDER': 'tavily',
            'TAVILY_API_KEY': 'test-key'
        }):
            with patch('tavily.TavilyClient'):
                provider = get_search_provider()
                assert isinstance(provider, TavilySearchProvider)
    
    def test_get_provider_serpapi_with_key(self):
        """Test provider selection returns SerpAPI when configured."""
        with patch.dict(os.environ, {
            'SEARCH_PROVIDER': 'serpapi', 
            'SERPAPI_API_KEY': 'test-key'
        }):
            with patch('serpapi.GoogleSearch'):
                provider = get_search_provider()
                assert isinstance(provider, SerpAPISearchProvider)
    
    def test_get_provider_falls_back_to_mock_missing_key(self):
        """Test provider selection falls back to mock when API key missing."""
        with patch.dict(os.environ, {
            'SEARCH_PROVIDER': 'tavily'
        }, clear=True):
            provider = get_search_provider()
            assert isinstance(provider, MockSearchProvider)
    
    def test_get_provider_falls_back_to_mock_unknown_provider(self):
        """Test provider selection falls back to mock for unknown provider."""
        with patch.dict(os.environ, {
            'SEARCH_PROVIDER': 'unknown_provider'
        }):
            provider = get_search_provider()
            assert isinstance(provider, MockSearchProvider)


class TestSearchWebFunction:
    """Test main search_web function."""
    
    def test_search_web_with_mock_provider(self):
        """Test search_web function with mock provider."""
        with patch('src.tools.search_adapter.get_search_provider') as mock_get_provider:
            mock_provider = MockSearchProvider()
            mock_get_provider.return_value = mock_provider
            
            results = search_web("AI engineer skills", 3)
            
            assert len(results) == 3
            assert all('title' in r and 'url' in r and 'content' in r for r in results)
    
    def test_search_web_handles_provider_error(self):
        """Test search_web handles provider errors gracefully."""
        with patch('src.tools.search_adapter.get_search_provider') as mock_get_provider:
            # Mock provider that raises an error
            mock_provider = Mock()
            mock_provider.search.side_effect = Exception("API Error")
            mock_get_provider.return_value = mock_provider
            
            # Should fall back to mock results
            results = search_web("test query", 2)
            
            assert len(results) == 2
            # Results should be from MockSearchProvider fallback
            assert all('AI' in r['content'] or 'engineer' in r['content'].lower() for r in results)
    
    def test_search_web_default_parameters(self):
        """Test search_web with default parameters."""
        with patch('src.tools.search_adapter.get_search_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.search.return_value = [
                {'title': f'Result {i}', 'url': f'https://example.com/{i}', 'content': f'Content {i}'}
                for i in range(8)
            ]
            mock_get_provider.return_value = mock_provider
            
            results = search_web("test query")  # No k parameter
            
            # Should call with default k=8
            mock_provider.search.assert_called_once_with("test query", 8)
            assert len(results) == 8


def test_import_fallback_handling():
    """Test that import errors are handled gracefully."""
    # Test TavilySearchProvider import handling
    with patch('tavily.TavilyClient', side_effect=ImportError("tavily not found")):
        with pytest.raises(ImportError, match="tavily-python package is required"):
            TavilySearchProvider("test-key")
    
    # Test SerpAPISearchProvider import handling  
    with patch('serpapi.GoogleSearch', side_effect=ImportError("serpapi not found")):
        with pytest.raises(ImportError, match="google-search-results package is required"):
            SerpAPISearchProvider("test-key")


if __name__ == "__main__":
    pytest.main([__file__])