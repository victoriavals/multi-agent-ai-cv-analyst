"""
Tavily search adapter with live web search and content extraction.

Provides real-time web search using Tavily API with full page content extraction.
No mock implementations - requires valid TAVILY_API_KEY and live internet connection.
"""

import logging
import os
from typing import Dict, List
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import trafilatura
from readability import Document

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, environment variables may be set directly
    pass

logger = logging.getLogger(__name__)

# Type alias for search results
SearchResult = Dict[str, str]  # {title, url, content}

# HTTP client configuration for robust page fetching
HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=30.0)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 SkillGapAnalyst/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Retryable exceptions for HTTP requests
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)


def require_tavily_api_key() -> str:
    """
    Check that TAVILY_API_KEY is available and return it.
    
    Returns:
        str: The Tavily API key
        
    Raises:
        RuntimeError: If TAVILY_API_KEY is missing
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY environment variable is required for web search. "
            "Please set it in your .env file or environment. "
            "Get your API key from: https://tavily.com/ "
            "No mock implementations available - live API key required."
        )
    return api_key


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS)
)
def fetch_page_content(url: str) -> str:
    """
    Fetch and extract readable content from a web page.
    
    Args:
        url: URL to fetch content from
        
    Returns:
        str: Extracted readable text content
        
    Raises:
        Exception: If page fetching fails after retries
    """
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS) as client:
            logger.debug(f"Fetching content from: {url}")
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            html_content = response.text
            
            # Try trafilatura first (best for article extraction)
            try:
                extracted_text = trafilatura.extract(
                    html_content,
                    include_comments=False,
                    include_tables=True,
                    include_formatting=False,
                    deduplicate=True
                )
                
                if extracted_text and len(extracted_text.strip()) > 100:
                    logger.debug(f"Trafilatura extracted {len(extracted_text)} chars from {url}")
                    return extracted_text.strip()
                    
            except Exception as e:
                logger.debug(f"Trafilatura failed for {url}: {e}")
            
            # Fallback to readability + BeautifulSoup
            try:
                doc = Document(html_content)
                readable_html = doc.summary()
                
                # Extract text using BeautifulSoup
                soup = BeautifulSoup(readable_html, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = '\n'.join(chunk for chunk in chunks if chunk)
                
                if clean_text and len(clean_text.strip()) > 100:
                    logger.debug(f"Readability extracted {len(clean_text)} chars from {url}")
                    return clean_text.strip()
                    
            except Exception as e:
                logger.debug(f"Readability fallback failed for {url}: {e}")
            
            # Final fallback: basic text extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            logger.debug(f"Basic extraction got {len(clean_text)} chars from {url}")
            return clean_text.strip() if clean_text else ""
            
    except Exception as e:
        logger.error(f"Failed to fetch content from {url}: {e}")
        # Don't raise - return empty string to allow other results to succeed
        return ""


def validate_results(docs: List[SearchResult]) -> None:
    """
    Validate that search results meet minimum quality requirements.
    
    Args:
        docs: List of search result dictionaries
        
    Raises:
        RuntimeError: If results are insufficient
    """
    if not docs:
        raise RuntimeError(
            "No search results obtained from Tavily API. "
            "Check your TAVILY_API_KEY and internet connection."
        )
    
    valid_docs = [
        doc for doc in docs 
        if doc.get('content', '').strip() and len(doc.get('content', '').strip()) > 50
    ]
    
    if len(valid_docs) < 3:
        raise RuntimeError(
            f"Insufficient live results: got {len(valid_docs)} valid docs, need at least 3. "
            f"Total docs: {len(docs)}. "
            "This may indicate network issues, blocked content, or API rate limits."
        )
    
    logger.info(f"Validation passed: {len(valid_docs)} valid docs out of {len(docs)} total")


def search_web(query: str, k: int = 10) -> List[SearchResult]:
    """
    Search the web using Tavily API and extract full page content.
    
    Args:
        query: Search query string
        k: Number of results to return (default: 10, max: 20)
        
    Returns:
        List of dictionaries with keys: title, url, content
        Content is readable text extracted from full web pages (min 500 chars when possible)
        
    Raises:
        RuntimeError: If TAVILY_API_KEY is missing or insufficient results obtained
        ImportError: If tavily-python package is not installed
    """
    # Validate environment and get API key
    api_key = require_tavily_api_key()
    
    # Import Tavily client
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "tavily-python package is required. Install with: pip install tavily-python"
        )
    
    # Limit k to reasonable bounds
    k = min(max(k, 1), 20)
    
    try:
        # Initialize Tavily client
        client = TavilyClient(api_key=api_key)
        
        logger.info(f"Searching Tavily for: '{query}' (max_results={k})")
        
        # Perform search with advanced depth
        search_response = client.search(
            query=query,
            include_answer=False,  # We don't need the AI-generated answer
            max_results=k,
            search_depth="advanced"  # Get more comprehensive results
        )
        
        # Extract results from response
        raw_results = search_response.get('results', [])
        
        if not raw_results:
            logger.warning(f"No results returned from Tavily for query: {query}")
            return []
        
        logger.info(f"Tavily returned {len(raw_results)} raw results")
        
        # Process each result to extract full content
        processed_results = []
        
        for i, result in enumerate(raw_results):
            title = result.get('title', 'No Title')
            url = result.get('url', '')
            snippet = result.get('content', '')
            
            if not url:
                logger.warning(f"Result {i+1} missing URL, skipping")
                continue
            
            # Validate URL
            try:
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    logger.warning(f"Invalid URL {url}, skipping")
                    continue
            except Exception as e:
                logger.warning(f"URL parsing failed for {url}: {e}")
                continue
            
            # Fetch full page content
            try:
                full_content = fetch_page_content(url)
                
                # Use full content if substantial, otherwise fallback to snippet
                if full_content and len(full_content) >= 500:
                    content = full_content
                    logger.debug(f"Using full content ({len(content)} chars) for {url}")
                elif full_content and len(full_content) >= 100:
                    # Combine snippet and partial content
                    content = f"{snippet}\n\n{full_content}" if snippet else full_content
                    logger.debug(f"Using combined content ({len(content)} chars) for {url}")
                elif snippet:
                    content = snippet
                    logger.debug(f"Using snippet only ({len(content)} chars) for {url}")
                else:
                    content = f"Content from {url} (extraction failed)"
                    logger.warning(f"No content extracted from {url}")
                
            except Exception as e:
                logger.error(f"Content extraction failed for {url}: {e}")
                content = snippet if snippet else f"Content from {url} (fetch failed)"
            
            processed_results.append({
                'title': title,
                'url': url,
                'content': content
            })
            
            logger.debug(f"Processed result {i+1}/{len(raw_results)}: {title[:50]}...")
        
        # Validate results meet minimum requirements
        validate_results(processed_results)
        
        logger.info(f"Search completed successfully: {len(processed_results)} results for '{query}'")
        return processed_results
        
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        raise RuntimeError(
            f"Web search failed using Tavily API. "
            f"Ensure TAVILY_API_KEY is valid and internet connection is active. "
            f"Error: {e}"
        )
