"""
CV Parsing & Normalization Agent

This module provides functionality to parse CV/resume text and extract structured information
using LLM-based parsing with rule-based fallbacks for robustness.
"""

import re
import json
import logging
from typing import Optional, Any, Dict, List
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.graph.state import CVStruct
from src.llm.provider import get_chat_model


logger = logging.getLogger(__name__)


def pre_clean_text(cv_text: str) -> str:
    """
    Pre-clean CV text by normalizing spaces and removing repeated headers/footers.
    
    Args:
        cv_text: Raw CV text to clean
        
    Returns:
        str: Cleaned CV text
    """
    if not cv_text or not cv_text.strip():
        return ""
    
    # Normalize line endings
    text = cv_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace but preserve paragraph structure
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
    
    # Remove common repeated headers/footers
    # Pattern for common CV headers that repeat (case insensitive)
    repeated_patterns = [
        r'^.*confidential.*$',
        r'^.*page \d+ of \d+.*$',
        r'^.*resume.*$',
        r'^.*curriculum vitae.*$',
        r'^.*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*$',  # Date lines
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append('')
            continue
            
        # Check if line matches any repeated pattern
        is_repeated = False
        for pattern in repeated_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_repeated = True
                break
        
        if not is_repeated:
            cleaned_lines.append(line)
    
    # Remove duplicate consecutive lines (common in poorly formatted PDFs)
    final_lines = []
    prev_line = None
    
    for line in cleaned_lines:
        if line != prev_line or line == '':  # Keep empty lines for structure
            final_lines.append(line)
        prev_line = line
    
    # Join and final cleanup
    cleaned_text = '\n'.join(final_lines)
    
    # Remove excessive leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def parse_cv_text(cv_text: str, provider_choice: str = "auto", model_name: Optional[str] = None) -> CVStruct:
    """
    Parse CV text and extract structured information using live LLM only.
    
    Enforces deterministic live-only parsing with strict JSON validation.
    
    Steps:
    1. Pre-clean text (drop repeated headers/footers, normalize spaces)
    2. Use provider.get_chat_model() with temperature=0 for deterministic parsing
    3. Use prompts/parser.md and ask for STRICT JSON only
    4. Validate with CVStruct; on validation error, try one repair pass
    5. If LLM fails twice, keep minimal regex fallback only for headings (no invented data)
    
    Args:
        cv_text: Raw CV text to parse
        provider_choice: LLM provider choice ("auto", "gemini", "mistral")
        model_name: Optional specific model name to use
        
    Returns:
        CVStruct: Structured CV data
        
    Raises:
        ValueError: If CV text is empty or invalid
        RuntimeError: If provider initialization fails
    """
    if not cv_text or not cv_text.strip():
        raise ValueError("CV text cannot be empty")

    # Step 1: Pre-clean text
    cleaned_text = pre_clean_text(cv_text)
    logger.info(f"Pre-cleaned CV text: {len(cleaned_text)} characters")

    # Step 2: Get LLM with deterministic settings (temperature=0)
    try:
        llm = get_chat_model(provider_choice, model_name, temperature=0)
        logger.info(f"Using {provider_choice} chat model for deterministic parsing")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM provider '{provider_choice}': {e}")

    # Step 3: Try LLM-based parsing with strict JSON
    try:
        cv_data = _parse_with_llm_strict(cleaned_text, llm)
        
        # Step 4: Validate with CVStruct
        try:
            cv_struct = CVStruct(**cv_data)
            logger.info(f"Successfully parsed CV with LLM: {len(cv_struct.skills)} skills extracted")
            return cv_struct
        except ValidationError as e:
            logger.warning(f"CVStruct validation failed: {e}. Attempting repair pass.")
            
            # Try one repair pass
            repaired_data = _repair_json_with_llm(cleaned_text, cv_data, str(e), llm)
            
            try:
                cv_struct = CVStruct(**repaired_data)
                logger.info(f"Successfully repaired and parsed CV: {len(cv_struct.skills)} skills extracted")
                return cv_struct
            except ValidationError as e2:
                logger.error(f"Repair attempt failed: {e2}. Falling back to minimal extraction.")
                raise RuntimeError(
                    f"LLM failed to produce valid CVStruct after repair attempt. "
                    f"Validation errors: {e2}. No content fabrication fallbacks available."
                )
                
    except Exception as e:
        logger.error(f"LLM parsing completely failed: {e}. Attempting minimal regex extraction.")
        
        # Step 5: Minimal regex fallback for headings only (no invented data)
        try:
            cv_data = _minimal_regex_extraction(cleaned_text)
            cv_struct = CVStruct(**cv_data)
            logger.warning(f"Used minimal regex extraction: {len(cv_struct.skills)} items found")
            return cv_struct
        except Exception as e2:
            logger.error(f"Even minimal extraction failed: {e2}")
            raise RuntimeError(
                f"CV parsing failed completely. LLM error: {e}. "
                f"Minimal extraction error: {e2}. "
                "No content fabrication available in live mode."
            )


def _parse_with_llm_strict(text: str, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Parse CV text using LLM with strict JSON validation and deterministic settings.
    
    Args:
        text: Cleaned CV text
        llm: LLM model to use for parsing (configured for temperature=0)
        
    Returns:
        Dict[str, Any]: Parsed CV data
        
    Raises:
        Exception: If LLM parsing fails
    """
    # Load parser prompt
    prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "parser.md"
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            parser_prompt = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Parser prompt not found at {prompt_path}. Required for live mode.")
    
    # Add strict JSON instruction to the prompt
    strict_instruction = (
        "\n\nIMPORTANT: Return ONLY valid JSON with the exact structure shown. "
        "Do not include any explanatory text before or after the JSON. "
        "Extract only information explicitly present in the CV text. "
        "Do not fabricate or invent any information not found in the original text."
    )
    
    # Create messages
    system_message = SystemMessage(content=parser_prompt + strict_instruction)
    human_message = HumanMessage(content=text)

    # Get LLM response (temperature=0 already set during model creation)
    try:
        response = llm.invoke([system_message, human_message])
        response_text = response.content
        
        # Extract JSON from response with improved pattern matching
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
        cv_data = json.loads(json_text)
        
        # Ensure required structure exists
        if not isinstance(cv_data, dict):
            raise ValueError("LLM response is not a dictionary")
        
        # Validate that we have the expected top-level keys
        required_keys = ['basics', 'skills', 'experience', 'projects', 'education', 'certifications']
        for key in required_keys:
            if key not in cv_data:
                cv_data[key] = [] if key != 'basics' else {}
        
        return cv_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"LLM invocation failed: {e}")


def _repair_json_with_llm(
    original_text: str, 
    broken_data: Dict[str, Any], 
    validation_error: str, 
    llm: BaseChatModel
) -> Dict[str, Any]:
    """
    Attempt to repair broken JSON using LLM with specific error information.
    
    Args:
        original_text: Original CV text
        broken_data: Broken CV data that failed validation
        validation_error: Specific validation error message
        llm: LLM model for repair
        
    Returns:
        Dict[str, Any]: Repaired CV data
        
    Raises:
        Exception: If repair fails
    """
    # Create repair prompt
    repair_prompt = f"""
You are fixing a JSON parsing error. The CV parser produced invalid JSON that failed validation.

VALIDATION ERROR: {validation_error}

BROKEN JSON:
{json.dumps(broken_data, indent=2)}

Fix the JSON to match the CVStruct schema:
- basics: dict (can be empty {{}})
- skills: list of strings
- experience: list of dicts
- projects: list of dicts  
- education: list of dicts
- certifications: list of dicts

Return ONLY the corrected JSON with proper field names and types. Do not add any explanatory text.
"""
    
    # Create messages for repair
    system_message = SystemMessage(content="You fix JSON validation errors by correcting field names and types.")
    human_message = HumanMessage(content=repair_prompt)
    
    try:
        response = llm.invoke([system_message, human_message])
        response_text = response.content
        
        # Extract JSON from response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                raise ValueError("No JSON found in repair response")
        
        # Parse repaired JSON
        repaired_data = json.loads(json_text)
        
        if not isinstance(repaired_data, dict):
            raise ValueError("Repaired response is not a dictionary")
        
        return repaired_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse repair response as JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"JSON repair failed: {e}")


def _minimal_regex_extraction(text: str) -> Dict[str, Any]:
    """
    Minimal regex-based extraction for headings only - no content fabrication.
    
    Only extracts what can be clearly identified without inventing data.
    
    Args:
        text: Cleaned CV text
        
    Returns:
        Dict[str, Any]: Minimal CV data with only identifiable sections
    """
    cv_data = {
        'basics': {},
        'skills': [],
        'experience': [],
        'projects': [],
        'education': [],
        'certifications': []
    }
    
    lines = text.split('\n')
    
    # Extract basic information from first few lines (name/title only)
    potential_name = None
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) > 2 and len(line) < 100:
            # Very conservative name detection - avoid headers
            if not re.match(r'^[A-Z\s]+:.*$', line) and not '@' in line:
                potential_name = line
                break
    
    if potential_name:
        cv_data['basics']['name'] = potential_name
    
    # Extract email if clearly identifiable
    email_match = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)
    if email_match:
        cv_data['basics']['email'] = email_match.group()
    
    # Extract skills only if there's a clear skills section
    skills_section_found = False
    skills_content = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        
        # Look for explicit skills headers
        if re.match(r'^(skills?|technical skills?|technologies):?\s*$', line_stripped):
            skills_section_found = True
            # Collect next few lines until another section
            for j in range(i+1, min(i+10, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                # Stop if we hit another section header
                if re.match(r'^[A-Z][A-Z\s]+:?\s*$', next_line) and len(next_line) > 10:
                    break
                skills_content.append(next_line)
            break
    
    if skills_section_found and skills_content:
        # Extract skills conservatively
        for line in skills_content:
            # Look for comma-separated or bullet-pointed skills
            if ',' in line:
                potential_skills = [s.strip() for s in line.split(',')]
            elif line.startswith(('•', '-', '*')):
                potential_skills = [line[1:].strip()]
            else:
                potential_skills = [line.strip()]
            
            for skill in potential_skills:
                skill = skill.strip()
                if skill and len(skill) > 1 and len(skill) < 50:
                    # Only add clearly technical terms
                    if re.match(r'^[A-Za-z][A-Za-z0-9\+\#\.\-_]*(?:\s+[A-Za-z][A-Za-z0-9\+\#\.\-_]*)*$', skill):
                        cv_data['skills'].append(skill)
    
    # Note: We do NOT extract experience, projects, education, or certifications
    # unless they're extremely clearly formatted, to avoid fabricating content
    
    logger.warning(
        "Used minimal regex extraction - only basic info and explicit skills extracted. "
        "No content fabrication performed."
    )
    
    return cv_data


# Legacy class for backwards compatibility
class CVParser:
    """Legacy CV Parser class - use parse_cv_text function instead."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
    
    def parse(self, cv_text: str) -> CVStruct:
        """Parse CV text using the main parsing function."""
        return parse_cv_text(cv_text, self.llm)
