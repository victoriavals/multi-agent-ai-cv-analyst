"""
LangGraph node implementations for the skill gap analysis pipeline.

This module provides node functions that can be connected in a LangGraph to perform
end-to-end skill gap analysis from CV text to comprehensive reports.
"""

import logging
from typing import Dict, Any
import structlog

from agents.cv_parser import parse_cv_text
from agents.skill_analyst import build_skill_profile
from agents.market_intel import gather_market_summary
from agents.reporter import synthesize_report
from graph.state import GraphState, CVStruct, SkillProfile, MarketSummary
from tools.embeddings import embed_texts
from utils.io import read_text_auto, save_text
from llm.provider import get_chat_model

logger = structlog.get_logger(__name__)


def ingest_cv(state: GraphState, **kwargs) -> GraphState:
    """
    Node: Load CV file content from filepath.
    
    Args:
        state: Graph state containing file_path
        **kwargs: Additional arguments
        
    Returns:
        Updated state with cv_text
    """
    try:
        logger.info("Starting CV ingestion", file_path=state.file_path)
        
        if not state.file_path:
            raise ValueError("No file path provided")
        
        # Load CV text using the file utility
        cv_text = read_text_auto(state.file_path)
        
        logs = state.logs + [
            f"✅ CV file loaded: {state.file_path}",
            f"   - Size: {len(cv_text)} characters"
        ]
        
        logger.info("CV ingestion completed", 
                   file_path=state.file_path, 
                   text_length=len(cv_text))
        
        return state.model_copy(update={
            "cv_text": cv_text,
            "logs": logs
        })
        
    except Exception as e:
        error_msg = f"❌ CV file loading failed: {str(e)}"
        logger.error("CV ingestion failed", error=str(e), file_path=state.file_path)
        logs = state.logs + [error_msg]
        return state.model_copy(update={
            "logs": logs,
            "error": str(e)
        })


def parse_cv(state: GraphState, **kwargs) -> GraphState:
    """
    Node: Parse CV text into structured format.
    
    Args:
        state: Graph state containing cv_text
        **kwargs: Additional arguments including provider_choice
        
    Returns:
        Updated state with cv_struct
    """
    try:
        logger.info("Starting CV parsing", text_length=len(state.cv_text) if state.cv_text else 0)
        
        if not state.cv_text:
            raise ValueError("No CV text to parse")
        
        # Get provider choice from kwargs or fallback to state
        provider_choice = kwargs.get("provider_choice", state.provider)
        
        # Parse CV using the CV parser agent with deterministic settings
        cv_struct = parse_cv_text(state.cv_text, provider_choice, model_name=None)
        
        logs = state.logs + [
            f"✅ CV parsed successfully",
            f"   - Skills: {len(cv_struct.skills)} extracted",
            f"   - Experience: {len(cv_struct.experience)} entries",
            f"   - Projects: {len(cv_struct.projects)} entries"
        ]
        
        logger.info("CV parsing completed",
                   skills_count=len(cv_struct.skills),
                   experience_count=len(cv_struct.experience),
                   projects_count=len(cv_struct.projects))
        
        return state.model_copy(update={
            "cv_struct": cv_struct.model_dump(),
            "logs": logs
        })
        
        return state.model_copy(update={
            "cv_struct": cv_struct.model_dump(),
            "logs": logs
        })
        
    except Exception as e:
        error_msg = f"❌ CV parsing failed: {str(e)}"
        logger.error("CV parsing failed", error=str(e))
        logs = state.logs + [error_msg]
        return state.model_copy(update={
            "logs": logs,
            "error": str(e)
        })


def analyze_skills(state: GraphState, **kwargs) -> GraphState:
    """
    Node: Analyze CV structure to build comprehensive skill profile.
    
    Args:
        state: Graph state containing cv_struct
        **kwargs: Additional arguments including provider_choice
        
    Returns:
        Updated state with skill_profile
    """
    try:
        logger.info("Starting skill analysis")
        
        if not state.cv_struct:
            raise ValueError("No CV structure to analyze")
        
        # Handle both dict and CVStruct object formats
        if isinstance(state.cv_struct, dict):
            cv_struct = CVStruct(**state.cv_struct)
        else:
            cv_struct = state.cv_struct
        
        # Get provider choice from kwargs or fallback to state
        provider_choice = kwargs.get("provider_choice", state.provider)
        
        # Build skill profile using the skill analyst agent
        skill_profile = build_skill_profile(cv_struct, provider_choice, model_name=None)
        
        total_skills = len(skill_profile.explicit) + len(skill_profile.implicit) + len(skill_profile.transferable)
        
        logs = state.logs + [
            f"✅ Skill analysis completed",
            f"   - Explicit skills: {len(skill_profile.explicit)}",
            f"   - Implicit skills: {len(skill_profile.implicit)}",
            f"   - Transferable skills: {len(skill_profile.transferable)}",
            f"   - Seniority signals: {len(skill_profile.seniority_signals)}",
            f"   - Total skills identified: {total_skills}"
        ]
        
        logger.info("Skill analysis completed",
                   explicit_skills=len(skill_profile.explicit),
                   implicit_skills=len(skill_profile.implicit),
                   transferable_skills=len(skill_profile.transferable),
                   seniority_signals=len(skill_profile.seniority_signals),
                   total_skills=total_skills)
        
        return state.model_copy(update={
            "skill_profile": skill_profile.model_dump(),
            "logs": logs
        })
        
        logs = state.logs + [
            f"✅ Skill analysis completed",
            f"   - Explicit skills: {len(skill_profile.explicit)}",
            f"   - Implicit skills: {len(skill_profile.implicit)}",
            f"   - Transferable skills: {len(skill_profile.transferable)}",
            f"   - Seniority signals: {len(skill_profile.seniority_signals)}",
            f"   - Total skills identified: {total_skills}"
        ]
        
        logger.info("Skill analysis completed",
                   explicit_skills=len(skill_profile.explicit),
                   implicit_skills=len(skill_profile.implicit),
                   transferable_skills=len(skill_profile.transferable),
                   seniority_signals=len(skill_profile.seniority_signals),
                   total_skills=total_skills)
        
        return state.model_copy(update={
            "skill_profile": skill_profile.model_dump(),
            "logs": logs
        })
        
    except Exception as e:
        error_msg = f"❌ Skill analysis failed: {str(e)}"
        logger.error("Skill analysis failed", error=str(e))
        logs = state.logs + [error_msg]
        return state.model_copy(update={
            "logs": logs,
            "error": str(e)
        })


def market_scan(state: GraphState, **kwargs) -> GraphState:
    """
    Node: Gather market intelligence for the target role and region.
    
    Args:
        state: Graph state containing target_role and market_region
        **kwargs: Additional arguments including provider_choice
        
    Returns:
        Updated state with market_summary
    """
    try:
        logger.info("Starting market scan", 
                   target_role=state.target_role, 
                   market_region=state.market_region)
        
        # Get provider choice from kwargs or fallback to state
        provider_choice = kwargs.get("provider_choice", state.provider)
        
        # Gather market intelligence using the market intel agent
        market_summary = gather_market_summary(
            role=state.target_role,
            region=state.market_region,
            provider_choice=provider_choice,
            model_name=None
        )
        
        total_skills = (
            len(market_summary.in_demand_skills) + 
            len(market_summary.common_tools) + 
            len(market_summary.frameworks) + 
            len(market_summary.nice_to_have)
        )
        
        logs = state.logs + [
            f"✅ Market scan completed for {state.target_role} in {state.market_region}",
            f"   - In-demand skills: {len(market_summary.in_demand_skills)}",
            f"   - Common tools: {len(market_summary.common_tools)}",
            f"   - Frameworks: {len(market_summary.frameworks)}",
            f"   - Nice-to-have: {len(market_summary.nice_to_have)}",
            f"   - Total market skills: {total_skills}",
            f"   - Sources analyzed: {len(market_summary.sources_sample)}"
        ]
        
        logger.info("Market scan completed",
                   target_role=state.target_role,
                   market_region=state.market_region,
                   in_demand_skills=len(market_summary.in_demand_skills),
                   common_tools=len(market_summary.common_tools),
                   frameworks=len(market_summary.frameworks),
                   nice_to_have=len(market_summary.nice_to_have),
                   total_skills=total_skills,
                   sources_count=len(market_summary.sources_sample))
        
        return state.model_copy(update={
            "market_summary": market_summary.model_dump(),
            "logs": logs
        })
        
    except Exception as e:
        error_msg = f"❌ Market scan failed: {str(e)}"
        logger.error("Market scan failed", error=str(e), 
                    target_role=state.target_role, 
                    market_region=state.market_region)
        logs = state.logs + [error_msg]
        return state.model_copy(update={
            "logs": logs,
            "error": str(e)
        })


def synthesize_report_node(state: GraphState, **kwargs) -> GraphState:
    """
    Node: Synthesize comprehensive skill gap analysis report.
    
    Args:
        state: Graph state containing skill_profile, market_summary, and lang
        **kwargs: Additional arguments including provider_choice
        
    Returns:
        Updated state with report_md
    """
    try:
        logger.info("Starting report synthesis", lang=state.lang)
        
        if not state.skill_profile:
            raise ValueError("No skill profile available for report generation")
        
        if not state.market_summary:
            raise ValueError("No market summary available for report generation")
        
        # Handle both dict and object formats
        if isinstance(state.skill_profile, dict):
            skill_profile = SkillProfile(**state.skill_profile)
        else:
            skill_profile = state.skill_profile
            
        if isinstance(state.market_summary, dict):
            market_summary = MarketSummary(**state.market_summary)
        else:
            market_summary = state.market_summary
        
        # Get provider choice from kwargs or fallback to state
        provider_choice = kwargs.get("provider_choice", state.provider)
        
        # Generate comprehensive report using the reporter agent
        report_md = synthesize_report(
            profile=skill_profile,
            market=market_summary,
            provider_choice=provider_choice,
            lang=state.lang
        )
        
        logs = state.logs + [
            f"✅ Report synthesized successfully",
            f"   - Report length: {len(report_md)} characters",
            f"   - Language: {state.lang}",
            f"   - Analysis complete for {market_summary.role} in {market_summary.region}"
        ]
        
        logger.info("Report synthesis completed",
                   report_length=len(report_md),
                   lang=state.lang,
                   target_role=market_summary.role,
                   market_region=market_summary.region)
        
        return state.model_copy(update={
            "report_md": report_md,
            "logs": logs
        })
        
    except Exception as e:
        error_msg = f"❌ Report synthesis failed: {str(e)}"
        logger.error("Report synthesis failed", error=str(e))
        logs = state.logs + [error_msg]
        return state.model_copy(update={
            "logs": logs,
            "error": str(e)
        })


def save_report(state: GraphState, **kwargs) -> GraphState:
    """
    Node: Save the generated report to a file.
    
    Args:
        state: Graph state containing report_md and output_path
        **kwargs: Additional arguments (can include output_path override)
        
    Returns:
        Updated state with save confirmation
    """
    try:
        # Allow kwargs to override output path
        output_path = kwargs.get("output_path", state.output_path)
        
        logger.info("Starting report save", output_path=output_path)
        
        if not state.report_md:
            raise ValueError("No report content to save")
        
        if not output_path:
            raise ValueError("No output path specified")
        
        # Save report using the file utility
        save_text(output_path, state.report_md)
        
        logs = state.logs + [
            f"✅ Report saved successfully: {output_path}",
            f"   - Report size: {len(state.report_md)} characters"
        ]
        
        logger.info("Report save completed", 
                   output_path=output_path,
                   report_size=len(state.report_md))
        
        return state.model_copy(update={
            "output_path": output_path,
            "logs": logs
        })
        
    except Exception as e:
        error_msg = f"❌ Report save failed: {str(e)}"
        logger.error("Report save failed", error=str(e), output_path=kwargs.get("output_path", state.output_path))
        logs = state.logs + [error_msg]
        return state.model_copy(update={
            "logs": logs,
            "error": str(e)
        })
