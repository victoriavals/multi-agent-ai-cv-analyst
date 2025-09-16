"""
Specialized Skill Analyst Agent

This module provides functionality to analyze CV data and build comprehensive skill profiles
including explicit, implicit, transferable skills and seniority signals.
"""

import json
import re
import logging
from typing import Optional, Any, Dict, List, Callable
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.graph.state import CVStruct, SkillProfile
from src.tools.taxonomy import canonicalize_skills
from src.llm.provider import get_chat_model


logger = logging.getLogger(__name__)


def extract_explicit_skills(cv: CVStruct) -> List[str]:
    """
    Extract explicit skills from CV skills list and project technologies.
    
    Args:
        cv: Structured CV data
        
    Returns:
        List[str]: Combined explicit skills from all sources
    """
    explicit_skills = []
    
    # Add skills from cv.skills
    if cv.skills:
        explicit_skills.extend(cv.skills)
    
    # Extract technologies from projects
    if cv.projects:
        for project in cv.projects:
            if isinstance(project, dict):
                # Handle both "tech" and "technologies" fields
                tech_list = project.get('tech', []) or project.get('technologies', [])
                if tech_list:
                    explicit_skills.extend(tech_list)
    
    # Extract technologies from experience entries
    if cv.experience:
        for exp in cv.experience:
            if isinstance(exp, dict):
                # Look for technologies in description or separate field
                tech_list = exp.get('technologies', []) or exp.get('tech', [])
                if tech_list:
                    explicit_skills.extend(tech_list)
                
                # Extract technologies mentioned in bullet points
                bullets = exp.get('bullets', [])
                description = exp.get('description', '')
                
                # Combine all text from this experience
                text_content = f"{description} {' '.join(bullets)}" if bullets else description
                
                if text_content:
                    # Extract common technology patterns from text
                    tech_patterns = [
                        r'\b(Python|Java|JavaScript|TypeScript|SQL|R|Go|Rust|C\+\+|C#)\b',
                        r'\b(TensorFlow|PyTorch|Scikit-learn|Keras|XGBoost)\b',
                        r'\b(AWS|Azure|GCP|Docker|Kubernetes|Git)\b',
                        r'\b(React|Angular|Vue|Node\.js|Django|Flask)\b',
                        r'\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch)\b',
                        r'\b(MLflow|Airflow|Jenkins|CI/CD|Terraform)\b'
                    ]
                    
                    for pattern in tech_patterns:
                        matches = re.findall(pattern, text_content, re.IGNORECASE)
                        explicit_skills.extend(matches)
    
    # Clean and deduplicate
    cleaned_skills = []
    seen = set()
    
    for skill in explicit_skills:
        if isinstance(skill, str):
            skill = skill.strip()
            if skill and skill.lower() not in seen:
                cleaned_skills.append(skill)
                seen.add(skill.lower())
    
    return cleaned_skills


def build_skill_profile(
    cv: CVStruct, 
    provider_choice: str = "auto", 
    model_name: Optional[str] = None,
    embed_f: Optional[Callable] = None
) -> SkillProfile:
    """
    Build comprehensive skill profile from CV data with real and reproducible inference.
    
    Makes implicit skill inference real using LLM with temperature=0.2 and evidence grounding.
    
    Steps:
    1. Extract explicit skills from cv.skills and project tech
    2. Use provider.get_chat_model() with temperature=0.2 for reproducible inference
    3. Infer implicit and transferable skills grounded by quotes/bullet references from CV
    4. Include evidence snippets temporarily but store only skill names in SkillProfile
    5. Canonicalize via taxonomy and remove duplicates
    6. Do not hallucinate - keep arrays small if CV lacks evidence
    
    Args:
        cv: Structured CV data
        provider_choice: LLM provider choice ("auto", "gemini", "mistral")
        model_name: Optional specific model name to use
        embed_f: Optional embedding function for similarity calculations
        
    Returns:
        SkillProfile: Comprehensive skill profile with grounded inferences
        
    Raises:
        ValueError: If CV is None or invalid
        RuntimeError: If provider initialization fails
    """
    if cv is None:
        raise ValueError("CV cannot be None")

    logger.info("Building skill profile from CV data with live LLM inference")
    
    # Step 1: Extract explicit skills from cv.skills and project tech
    explicit_skills = extract_explicit_skills(cv)
    logger.info(f"Extracted {len(explicit_skills)} explicit skills")

    # Step 2: Get LLM with temperature=0.2 for reproducible inference
    try:
        llm = get_chat_model(provider_choice, model_name, temperature=0.2)
        logger.info(f"Using {provider_choice} chat model for skill inference with temperature=0.2")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM provider '{provider_choice}': {e}")

    # Step 3: Use LLM for grounded analysis
    try:
        llm_analysis = _analyze_with_llm_grounded(cv, explicit_skills, llm)
        implicit_skills = llm_analysis.get('implicit', [])
        transferable_skills = llm_analysis.get('transferable', [])
        seniority_signals = llm_analysis.get('seniority_signals', [])
        
        logger.info(f"LLM inference: {len(implicit_skills)} implicit, "
                   f"{len(transferable_skills)} transferable, {len(seniority_signals)} seniority signals")
        
    except Exception as e:
        logger.error(f"LLM inference failed: {e}. No fabrication fallbacks available.")
        raise RuntimeError(
            f"Skill inference failed using LLM. "
            f"No rule-based fallbacks that fabricate content are available. "
            f"Error: {e}"
        )
    
    # Step 3: Canonicalize via taxonomy and remove duplicates
    canonical_explicit = canonicalize_skills(explicit_skills)
    canonical_implicit = canonicalize_skills(implicit_skills)
    canonical_transferable = canonicalize_skills(transferable_skills)
    
    # Remove duplicates between categories
    implicit_filtered = [skill for skill in canonical_implicit if skill not in canonical_explicit]
    transferable_filtered = [skill for skill in canonical_transferable 
                           if skill not in canonical_explicit and skill not in implicit_filtered]
    
    logger.info(f"After canonicalization and deduplication: "
                f"{len(canonical_explicit)} explicit, {len(implicit_filtered)} implicit, "
                f"{len(transferable_filtered)} transferable")
    
    # Step 4: Build coverage_map placeholder
    coverage_map = {}
    # TODO: This will be implemented when market data is available
    
    # Create and return SkillProfile (no fallback - fail if validation fails)
    try:
        skill_profile = SkillProfile(
            explicit=canonical_explicit,
            implicit=implicit_filtered,
            transferable=transferable_filtered,
            seniority_signals=seniority_signals,
            coverage_map=coverage_map
        )
        
        total_skills = len(skill_profile.explicit) + len(skill_profile.implicit) + len(skill_profile.transferable)
        logger.info(f"Built skill profile with {total_skills} total skills, "
                   f"{len(skill_profile.seniority_signals)} seniority signals")
        return skill_profile
        
    except ValidationError as e:
        logger.error(f"SkillProfile validation failed: {e}")
        raise RuntimeError(f"Failed to create valid SkillProfile: {e}. No fabrication fallbacks available.")


def _analyze_with_llm_grounded(
    cv: CVStruct, 
    explicit_skills: List[str], 
    llm: BaseChatModel
) -> Dict[str, Any]:
    """
    Analyze CV using LLM with temperature=0.2 for grounded skill inference.
    
    Derives explicit skills from cv.skills and project tech, then asks LLM to infer 
    implicit and transferable skills grounded by quotes or bullet references from CV.
    
    Args:
        cv: Structured CV data
        explicit_skills: Already extracted explicit skills
        llm: LLM model for analysis (configured with temperature=0.2)
        
    Returns:
        Dict[str, Any]: Analysis results with implicit, transferable, seniority_signals
        
    Raises:
        Exception: If LLM analysis fails
    """
    # Load analyst prompt
    prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / "analyst.md"
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            analyst_prompt = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Analyst prompt not found at {prompt_path}. Required for live mode.")

    # Prepare detailed CV content with evidence grounding
    cv_content = _prepare_cv_content_with_evidence(cv, explicit_skills)

    # Add grounding instruction to prompt
    grounding_instruction = (
        "\n\nIMPORTANT: For each skill you infer, you MUST cite specific evidence from the CV. "
        "Include the exact quote, bullet point, or project description that supports your inference. "
        "Format: skill_name: 'quoted evidence from CV'. "
        "If there is insufficient evidence in the CV text, keep the arrays small. "
        "Do not fabricate skills without clear supporting evidence. "
        "In your final JSON response, include only the skill names - evidence is for validation only."
    )
    
    # Create messages
    system_message = SystemMessage(content=analyst_prompt + grounding_instruction)
    human_message = HumanMessage(content=cv_content)
    
    # Get LLM response with error handling and retry
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
        analysis_data = json.loads(json_text)
        
        # Validate structure
        if not isinstance(analysis_data, dict):
            raise ValueError("LLM response is not a dictionary")
        
        # Ensure required keys exist and are lists
        required_keys = ['implicit', 'transferable', 'seniority_signals']
        for key in required_keys:
            if key not in analysis_data:
                analysis_data[key] = []
            if not isinstance(analysis_data[key], list):
                raise ValueError(f"Key '{key}' must be a list, got {type(analysis_data[key])}")
        
        # Validate that results are grounded (not excessive)
        total_cv_length = len(cv_content)
        max_implicit = min(15, total_cv_length // 500)  # Scale with CV content
        max_transferable = min(10, total_cv_length // 700)
        max_seniority = min(8, total_cv_length // 600)
        
        result = {
            'implicit': analysis_data['implicit'][:max_implicit],
            'transferable': analysis_data['transferable'][:max_transferable], 
            'seniority_signals': analysis_data['seniority_signals'][:max_seniority]
        }
        
        # Log grounding validation
        logger.info(f"LLM inference grounded by CV length {total_cv_length}: "
                   f"{len(result['implicit'])} implicit, {len(result['transferable'])} transferable, "
                   f"{len(result['seniority_signals'])} seniority signals")
        
        return result
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"LLM skill inference failed: {e}")


def _prepare_cv_content_with_evidence(cv: CVStruct, explicit_skills: List[str]) -> str:
    """
    Prepare detailed CV content with evidence grounding for LLM analysis.
    
    Args:
        cv: Structured CV data
        explicit_skills: Already extracted explicit skills
        
    Returns:
        str: Detailed CV content formatted for evidence-based analysis
    """
    content_parts = []
    
    # Start with explicit skills context
    content_parts.append("=== EXPLICIT SKILLS IDENTIFIED ===")
    if explicit_skills:
        content_parts.append(f"Skills from CV: {', '.join(explicit_skills[:20])}")
    else:
        content_parts.append("No explicit skills found")
    content_parts.append("")
    
    # Basic information
    if cv.basics:
        content_parts.append("=== BASIC INFORMATION ===")
        name = cv.basics.get('name', 'Unknown')
        title = cv.basics.get('title', 'Unknown')
        content_parts.append(f"Name: {name}")
        content_parts.append(f"Professional Title: {title}")
        content_parts.append("")
    
    # Experience with detailed context
    if cv.experience:
        content_parts.append("=== WORK EXPERIENCE ===")
        for i, exp in enumerate(cv.experience, 1):
            if isinstance(exp, dict):
                title = exp.get('title', exp.get('position', 'Unknown Position'))
                company = exp.get('company', 'Unknown Company')
                dates = exp.get('dates', exp.get('duration', 'Unknown Duration'))
                
                content_parts.append(f"{i}. {title} at {company} ({dates})")
                
                # Include all available details for evidence
                bullets = exp.get('bullets', [])
                description = exp.get('description', '')
                tech = exp.get('tech', []) or exp.get('technologies', [])
                
                if tech:
                    content_parts.append(f"   Technologies used: {', '.join(tech)}")
                
                if bullets:
                    content_parts.append("   Key achievements/responsibilities:")
                    for bullet in bullets:
                        content_parts.append(f"   • {bullet}")
                elif description:
                    content_parts.append("   Description:")
                    for line in description.split('\n'):
                        if line.strip():
                            content_parts.append(f"   • {line.strip()}")
                
                content_parts.append("")
    
    # Projects with detailed context
    if cv.projects:
        content_parts.append("=== PROJECTS ===")
        for i, project in enumerate(cv.projects, 1):
            if isinstance(project, dict):
                name = project.get('name', f'Project {i}')
                tech = project.get('tech', []) or project.get('technologies', [])
                bullets = project.get('bullets', [])
                description = project.get('description', '')
                
                content_parts.append(f"{i}. {name}")
                
                if tech:
                    content_parts.append(f"   Technologies: {', '.join(tech)}")
                
                if bullets:
                    content_parts.append("   Details:")
                    for bullet in bullets:
                        content_parts.append(f"   • {bullet}")
                elif description and description != name:
                    content_parts.append(f"   Description: {description}")
                
                content_parts.append("")
    
    # Education
    if cv.education:
        content_parts.append("=== EDUCATION ===")
        for edu in cv.education:
            if isinstance(edu, dict):
                degree = edu.get('degree', edu.get('institution', 'Unknown'))
                school = edu.get('school', edu.get('institution', ''))
                year = edu.get('year', '')
                field = edu.get('field', '')
                
                content_parts.append(f"• {degree}")
                if school and school != degree:
                    content_parts.append(f"  Institution: {school}")
                if field:
                    content_parts.append(f"  Field: {field}")
                if year:
                    content_parts.append(f"  Year: {year}")
        content_parts.append("")
    
    # Certifications
    if cv.certifications:
        content_parts.append("=== CERTIFICATIONS ===")
        for cert in cv.certifications:
            if isinstance(cert, dict):
                name = cert.get('name', 'Unknown Certification')
                issuer = cert.get('org', cert.get('issuer', ''))
                year = cert.get('year', '')
                
                content_parts.append(f"• {name}")
                if issuer:
                    content_parts.append(f"  Issuer: {issuer}")
                if year:
                    content_parts.append(f"  Year: {year}")
        content_parts.append("")
    
    # Add analysis instruction
    content_parts.append("=== ANALYSIS TASK ===")
    content_parts.append("Based on the above CV content, infer:")
    content_parts.append("1. IMPLICIT skills that can be reasonably inferred from the roles and responsibilities")
    content_parts.append("2. TRANSFERABLE skills like communication, leadership, mentoring based on evidence")
    content_parts.append("3. SENIORITY SIGNALS with specific quotes or quantified achievements")
    content_parts.append("")
    content_parts.append("IMPORTANT: Only infer skills with strong supporting evidence from the CV text above.")
    
    return '\n'.join(content_parts)


# Legacy class for backwards compatibility
class SkillAnalyst:
    """Legacy Skill Analyst class - use build_skill_profile function instead."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None, embed_f: Optional[Callable] = None):
        self.llm = llm
        self.embed_f = embed_f
    
    def analyze(self, cv: CVStruct) -> SkillProfile:
        """Analyze CV to build skill profile."""
        return build_skill_profile(cv, self.llm, self.embed_f)
