"""
Market Intelligence Agent

This module provides functionality to gather market intelligence about technical roles,
including in-demand skills, common tools, frameworks, and nice-to-have skills.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.graph.state import MarketSummary
from src.tools.search_adapter import search_web
from src.tools.taxonomy import canonicalize_skills
from src.llm.provider import get_chat_model


logger = logging.getLogger(__name__)


def gather_market_summary(
    role: str, 
    region: str, 
    provider_choice: str = "auto",
    model_name: Optional[str] = None
) -> MarketSummary:
    """
    Gather live market intelligence summary for a specific role and region.
    
    Enforces live-only mode - no fallbacks or simulated data.
    
    Steps:
    1) Build 3-5 diverse queries for the role and region
    2) For each query, call search_adapter.search_web(query, k=6) and accumulate docs
    3) Deduplicate by URL
    4) Validate with validate_results, require at least 5 unique docs total
    5) Concatenate trimmed contents (max 2000 tokens worth) and summarize with LLM
    6) Parse normalized JSON with arrays: in_demand_skills, common_tools, frameworks, nice_to_have
    7) If LLM returns invalid JSON, retry once with "Return valid JSON only" instruction
    8) Fill sources_sample with top 6 titles and URLs
    
    Args:
        role: Target job role (e.g., "Senior AI Engineer")
        region: Target region (e.g., "Global", "US", "Europe")
        provider_choice: LLM provider choice ("auto", "gemini", "mistral")
        model_name: Optional specific model name to use
        
    Returns:
        MarketSummary: Market intelligence with skills, tools, frameworks
        
    Raises:
        ValueError: If role or region is empty
        RuntimeError: If search returns too few results or internet/API keys missing, or provider fails
    """
    if not role or not role.strip():
        raise ValueError("Role cannot be empty")
    if not region or not region.strip():
        raise ValueError("Region cannot be empty")

    # Get LLM for content analysis
    try:
        llm = get_chat_model(provider_choice, model_name)
        logger.info(f"Using {provider_choice} chat model for market intelligence")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM provider '{provider_choice}': {e}")
    
    logger.info(f"Gathering LIVE market intelligence for {role} in {region}")
    
    # Step 1: Build 3-5 diverse queries for the role and region
    queries = [
        f"{role} required skills {region}",
        f"{role} responsibilities {region}",
        f"{role} hiring requirements {region}",
        f"{role} interview requirements {region}",
        f"{role} tech stack {region}"
    ]
    
    logger.info(f"Built {len(queries)} diverse search queries")
    
    # Step 2: For each query, call search_adapter.search_web(query, k=6) and accumulate docs
    all_search_results = []
    failed_queries = 0
    
    for i, query in enumerate(queries):
        try:
            logger.info(f"Executing search {i+1}/{len(queries)}: {query}")
            results = search_web(query, k=6)
            all_search_results.extend(results)
            logger.info(f"Got {len(results)} results for query: {query}")
        except Exception as e:
            failed_queries += 1
            logger.error(f"Search failed for query '{query}': {e}")
            if failed_queries >= len(queries):
                raise RuntimeError(
                    f"All search queries failed. Check your TAVILY_API_KEY and internet connection. "
                    f"Last error: {e}"
                )
    
    logger.info(f"Collected {len(all_search_results)} total search results")
    
    # Step 3: Deduplicate by URL
    seen_urls = set()
    unique_results = []
    
    for result in all_search_results:
        url = result.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    logger.info(f"After deduplication: {len(unique_results)} unique results")
    
    # Step 4: Validate results - require at least 5 unique docs total
    def validate_results(docs: List[Dict[str, str]]) -> None:
        if len(docs) < 5:
            raise RuntimeError(
                f"Insufficient live results: got {len(docs)} unique docs, need at least 5. "
                f"This may indicate network issues, blocked content, or API rate limits. "
                "No simulated or fallback static data available."
            )
        
        valid_docs = [
            doc for doc in docs 
            if doc.get('content', '').strip() and len(doc.get('content', '').strip()) > 50
        ]
        
        if len(valid_docs) < 5:
            raise RuntimeError(
                f"Insufficient quality results: got {len(valid_docs)} valid docs with content, need at least 5. "
                f"Total docs: {len(docs)}. "
                "No simulated or fallback static data available."
            )
        
        logger.info(f"Validation passed: {len(valid_docs)} valid docs out of {len(docs)} total")
    
    validate_results(unique_results)
    
    # Step 5: Concatenate trimmed contents (max 2000 tokens worth) and summarize with LLM
    content_summary = _prepare_content_for_llm(unique_results, role, region, max_tokens=2000)
    
    # Step 6: Parse normalized JSON with arrays
    market_data = _analyze_with_llm_strict(content_summary, role, region, llm)
    
    # Step 7: Canonicalize skills via taxonomy  
    in_demand_skills = canonicalize_skills(market_data.get('in_demand_skills', []))
    common_tools = canonicalize_skills(market_data.get('common_tools', []))
    frameworks = canonicalize_skills(market_data.get('frameworks', []))
    nice_to_have = canonicalize_skills(market_data.get('nice_to_have', []))
    
    # Ensure we have non-empty results for common roles
    if not in_demand_skills:
        raise RuntimeError(
            f"No in-demand skills extracted for {role} in {region}. "
            "This indicates content extraction or LLM analysis failure. "
            "No fallback data available in live mode."
        )
    
    logger.info(f"Canonicalized skills: {len(in_demand_skills)} in-demand, "
                f"{len(common_tools)} tools, {len(frameworks)} frameworks, "
                f"{len(nice_to_have)} nice-to-have")
    
    # Step 8: Fill sources_sample with top 6 titles and URLs
    sources_sample = []
    for result in unique_results[:6]:
        if result.get('title') and result.get('url'):
            sources_sample.append(f"{result['title']} - {result['url']}")
    
    # Build MarketSummary
    try:
        market_summary = MarketSummary(
            role=role.strip(),
            region=region.strip(),
            in_demand_skills=in_demand_skills[:20],
            common_tools=common_tools[:15],
            frameworks=frameworks[:15],
            nice_to_have=nice_to_have[:15],
            sources_sample=sources_sample
        )
        
        logger.info(f"Built market summary with {len(market_summary.in_demand_skills)} in-demand skills")
        return market_summary
        
    except ValidationError as e:
        raise RuntimeError(f"MarketSummary validation failed: {e}. No fallback data available in live mode.")


def _prepare_content_for_llm(
    search_results: List[Dict[str, str]], 
    role: str, 
    region: str,
    max_tokens: int = 2000
) -> str:
    """
    Prepare search results content for LLM analysis with token limit.
    
    Args:
        search_results: List of search results with title, url, content
        role: Target job role
        region: Target region
        max_tokens: Maximum tokens to include (approximate, using char count)
        
    Returns:
        str: Formatted content summary for LLM analysis
    """
    # Rough estimation: 1 token ≈ 4 characters for English text
    max_chars = max_tokens * 4
    
    summary_parts = [f"Market analysis for: {role} in {region}\n"]
    current_chars = len(summary_parts[0])
    
    for i, result in enumerate(search_results):
        title = result.get('title', f'Source {i+1}')
        content = result.get('content', '').strip()
        url = result.get('url', '')
        
        # Format the result entry
        entry_header = f"\n{i+1}. {title}"
        if url:
            entry_header += f" ({url})"
        entry_header += "\n"
        
        # Check if we have room for this entry
        if current_chars + len(entry_header) > max_chars:
            summary_parts.append(f"\n[Content truncated at {max_chars} characters to fit token limit]")
            break
        
        summary_parts.append(entry_header)
        current_chars += len(entry_header)
        
        if content:
            # Extract relevant lines mentioning skills, requirements, etc.
            content_lines = content.split('\n')
            relevant_lines = []
            
            for line in content_lines:
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                    
                # Look for lines mentioning job-relevant keywords
                if any(keyword in line.lower() for keyword in [
                    'skill', 'requirement', 'experience', 'qualification', 
                    'technology', 'tool', 'framework', 'language', 'platform',
                    'must have', 'should have', 'proficiency', 'knowledge',
                    'expertise', 'familiar', 'tech stack', 'technologies'
                ]):
                    # Check if we have room for this line
                    line_with_bullet = f"• {line}\n"
                    if current_chars + len(line_with_bullet) > max_chars:
                        summary_parts.append("[Content truncated to fit token limit]")
                        break
                    
                    relevant_lines.append(line_with_bullet)
                    current_chars += len(line_with_bullet)
                    
                    # Limit lines per result
                    if len(relevant_lines) >= 5:
                        break
            
            if relevant_lines:
                summary_parts.extend(relevant_lines)
            else:
                # Fallback to first few lines if no keywords found
                for line in content_lines[:3]:
                    line = line.strip()
                    if line and len(line) > 10:
                        line_with_bullet = f"• {line}\n"
                        if current_chars + len(line_with_bullet) > max_chars:
                            summary_parts.append("[Content truncated to fit token limit]")
                            break
                        summary_parts.append(line_with_bullet)
                        current_chars += len(line_with_bullet)
                        break
        
        # Add spacing between results
        summary_parts.append("")
        
        # Break if we're approaching the limit
        if current_chars > max_chars * 0.9:
            break
    
    return ''.join(summary_parts)


def _analyze_with_llm_strict(
    content_summary: str,
    role: str, 
    region: str, 
    llm: BaseChatModel
) -> Dict[str, List[str]]:
    """
    Analyze content using LLM with strict JSON validation and retry logic.
    
    Args:
        content_summary: Prepared content summary for analysis
        role: Target job role
        region: Target region
        llm: LLM model for analysis
        
    Returns:
        Dict[str, List[str]]: Market data with in_demand_skills, tools, frameworks, nice_to_have
        
    Raises:
        RuntimeError: If LLM analysis fails after retry
    """
    # Load market prompt
    prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "market.md"
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            market_prompt = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Market prompt not found at {prompt_path}. Required for live mode.")
    
    # Replace placeholders in prompt
    market_prompt = market_prompt.replace('{role}', role).replace('{region}', region)
    
    def attempt_llm_analysis(retry_instruction: str = "") -> Dict[str, List[str]]:
        """Single attempt at LLM analysis."""
        # Create messages
        system_message = SystemMessage(content=market_prompt + retry_instruction)
        human_message = HumanMessage(content=content_summary)
        
        # Get LLM response
        try:
            response = llm.invoke([system_message, human_message])
            response_text = response.content
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    raise ValueError("No JSON found in LLM response")
            
            # Parse JSON
            market_data = json.loads(json_text)
            
            # Validate structure
            if not isinstance(market_data, dict):
                raise ValueError("LLM response is not a dictionary")
            
            # Ensure required keys exist and are lists
            required_keys = ['in_demand_skills', 'common_tools', 'frameworks', 'nice_to_have']
            for key in required_keys:
                if key not in market_data:
                    raise ValueError(f"Missing required key: {key}")
                if not isinstance(market_data[key], list):
                    raise ValueError(f"Key '{key}' must be a list, got {type(market_data[key])}")
            
            # Return validated structure
            return {
                'in_demand_skills': market_data['in_demand_skills'],
                'common_tools': market_data['common_tools'],
                'frameworks': market_data['frameworks'],
                'nice_to_have': market_data['nice_to_have']
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise ValueError(f"LLM analysis failed: {e}")
    
    # First attempt
    try:
        return attempt_llm_analysis()
    except ValueError as e:
        logger.warning(f"First LLM attempt failed: {e}. Retrying with explicit JSON instruction.")
    
    # Second attempt with retry instruction
    try:
        retry_instruction = (
            "\n\nIMPORTANT: Your previous response was invalid. "
            "Return ONLY valid JSON with the exact structure shown in the example. "
            "Do not include any explanatory text before or after the JSON."
        )
        return attempt_llm_analysis(retry_instruction)
    except ValueError as e:
        raise RuntimeError(
            f"LLM failed to return valid JSON after 2 attempts. "
            f"Last error: {e}. "
            "No fallback data available in live mode."
        )


# Legacy class for backwards compatibility
class MarketIntel:
    """Legacy Market Intelligence class - use gather_market_summary function instead."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
    
    def analyze(self, role: str, region: str) -> MarketSummary:
        """Analyze market for role and region."""
        return gather_market_summary(role, region, self.llm)
