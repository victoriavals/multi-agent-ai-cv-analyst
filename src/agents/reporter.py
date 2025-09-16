"""
Recommendation & Report Agent

This module provides functionality to synthesize comprehensive skill gap analysis reports
including strengths, gaps, development plans, and appendices.
"""

import logging
from typing import Dict, List, Callable, Optional, Any
from collections import Counter

import numpy as np

from src.graph.state import SkillProfile, MarketSummary
from src.utils.eval import cosine_similarity, classify_priority
from src.tools.search_adapter import search_web
from src.tools.embeddings import embed_texts
from src.utils.md import (
    render_report_header, render_strengths, render_gaps, render_plan,
    render_section, render_bullet_list, render_skills_table, render_horizontal_rule
)


logger = logging.getLogger(__name__)


def synthesize_report(
    profile: SkillProfile, 
    market: MarketSummary, 
    provider_choice: str = "auto",
    lang: str = "en"
) -> str:
    """
    Synthesize comprehensive skill gap analysis report.
    
    Steps:
    1) Compute strengths, near-miss, and gaps using cosine similarity
    2) Build a 30-60-90 day plan focusing on top gaps with concrete tasks and resources
    3) Render markdown using utils.md helpers; include "Appendix: Coverage Map" and "Sources"
    4) Language default English; if lang != "en", keep headings English, body localized lightly
    
    Args:
        profile: Skill profile with explicit, implicit, transferable skills and seniority signals
        market: Market summary with in-demand skills, tools, frameworks, nice-to-have
        provider_choice: LLM provider choice ("auto", "gemini", "mistral") for embeddings
        lang: Language code (default "en")
        
    Returns:
        str: Complete markdown report
        
    Raises:
        ValueError: If profile or market is None
    """
    if profile is None:
        raise ValueError("SkillProfile cannot be None")
    if market is None:
        raise ValueError("MarketSummary cannot be None")
    
    logger.info(f"Synthesizing report for {market.role} in {market.region} (lang: {lang})")
    
    # Step 1: Compute strengths, near-misses, and gaps using cosine similarity
    analysis = _analyze_skill_gaps(profile, market, provider_choice)
    
    # Step 2: Build 30-60-90 day plan
    # Use gaps for planning, fall back to near misses if no gaps
    plan_gaps = analysis['gaps'] if analysis['gaps'] else analysis['near_misses'][:3]  # Use top 3 near misses
    development_plan = _build_development_plan(plan_gaps, lang)
    
    # Step 3: Render markdown report
    report_sections = []
    
    # Header
    report_sections.append(render_report_header(market.role, market.region))
    
    # Executive Summary
    summary_content = _render_executive_summary(profile, market, analysis, lang)
    report_sections.append(render_section("Executive Summary", summary_content, level=2))
    
    # Strengths
    if analysis['strengths']:
        strengths_list = [f"{strength['skill']} (similarity: {strength['similarity']:.2f})" for strength in analysis['strengths']]
        report_sections.append(render_strengths(strengths_list))
    
    # Near Misses (skills that are close matches)
    if analysis['near_misses']:
        near_miss_content = _render_near_misses(analysis['near_misses'], lang)
        report_sections.append(render_section("🎯 Near Misses", near_miss_content, level=2))
    
    # Skill Gaps
    if analysis['gaps']:
        gaps_list = [f"{gap['skill']} ({gap['priority']} priority)" for gap in analysis['gaps'][:10]]
        report_sections.append(render_gaps(gaps_list))
    
    # Development Plan
    report_sections.append(render_plan(development_plan))
    
    # Transferable Skills
    if profile.transferable:
        transferable_content = _render_transferable_skills(profile.transferable, lang)
        report_sections.append(render_section("🔄 Transferable Skills", transferable_content, level=2))
    
    # Seniority Signals
    if profile.seniority_signals:
        seniority_content = _render_seniority_signals(profile.seniority_signals, lang)
        report_sections.append(render_section("👑 Seniority Signals", seniority_content, level=2))
    
    # Add horizontal rule before appendix
    report_sections.append(render_horizontal_rule())
    
    # Step 4: Appendices
    # Coverage Map Appendix
    coverage_content = _render_coverage_map(analysis, lang)
    report_sections.append(render_section("Appendix: Coverage Map", coverage_content, level=2))
    
    # Sources Appendix (include both market and learning resources)
    learning_resources = development_plan.get('resources', {}) if isinstance(development_plan, dict) else {}
    sources_content = _render_sources(market.sources_sample, learning_resources, lang)
    report_sections.append(render_section("Appendix: Sources", sources_content, level=2))
    
    # Join all sections
    final_report = "\n".join(report_sections)
    
    logger.info(f"Generated report with {len(final_report)} characters")
    return final_report


def _analyze_skill_gaps(
    profile: SkillProfile, 
    market: MarketSummary, 
    provider_choice: str
) -> Dict[str, Any]:
    """
    Analyze skill gaps using real embeddings and cosine similarity.
    
    Categories:
    - Strengths: similarity >= 0.75
    - Near-miss: 0.5 to < 0.75  
    - Gaps: < 0.5 and present in market
    
    Args:
        profile: Candidate skill profile
        market: Market skill requirements
        provider_choice: LLM provider choice for embeddings
        
    Returns:
        Dict[str, Any]: Analysis with strengths, near_misses, gaps
        
    Raises:
        ValueError: If embedding generation fails
    """
    # Combine all candidate skills
    all_candidate_skills = list(set(
        profile.explicit + 
        profile.implicit + 
        profile.transferable
    ))
    
    # Combine all market skills (deduplicated)
    all_market_skills = list(set(
        market.in_demand_skills + 
        market.common_tools + 
        market.frameworks + 
        market.nice_to_have
    ))
    
    if not all_candidate_skills:
        logger.warning("No candidate skills found - all market skills will be gaps")
        gaps = [{'skill': skill, 'priority': 'high', 'score': 0.0} for skill in all_market_skills]
        return {
            'strengths': [],
            'near_misses': [],
            'gaps': gaps,
            'all_gaps': gaps
        }
    
    if not all_market_skills:
        logger.warning("No market skills found")
        return {
            'strengths': [],
            'near_misses': [],
            'gaps': [],
            'all_gaps': []
        }
    
    logger.info(f"Analyzing {len(all_candidate_skills)} candidate skills vs {len(all_market_skills)} market skills")
    
    # Get embeddings for all skills using the provider system
    try:
        # Generate embeddings for all skills in batch
        all_skills = all_candidate_skills + all_market_skills
        all_embeddings_array = embed_texts(all_skills, provider_preference=provider_choice)
        
        # Split back into candidate and market embeddings
        num_candidate = len(all_candidate_skills)
        candidate_embeddings = {
            skill: all_embeddings_array[i] 
            for i, skill in enumerate(all_candidate_skills)
        }
        market_embeddings = {
            skill: all_embeddings_array[num_candidate + i] 
            for i, skill in enumerate(all_market_skills)
        }
        
        logger.info(f"Generated embeddings for {len(all_skills)} skills using {provider_choice} provider")
        
    except Exception as e:
        raise ValueError(f"Embedding generation failed: {e}. No fallback scoring available.")
    
    # Analyze each market skill
    strengths = []
    near_misses = []
    gaps = []
    
    for market_skill in all_market_skills:
        market_embedding = market_embeddings[market_skill]
        
        # Find maximum similarity to any candidate skill
        max_similarity = 0.0
        best_match = None
        
        for candidate_skill in all_candidate_skills:
            candidate_embedding = candidate_embeddings[candidate_skill]
            similarity = cosine_similarity(candidate_embedding, market_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = candidate_skill
        
        # Categorize based on similarity thresholds
        if max_similarity >= 0.75:
            strengths.append({
                'skill': market_skill,
                'similarity': round(max_similarity, 3),
                'match': best_match
            })
        elif max_similarity >= 0.5:
            near_misses.append({
                'skill': market_skill,
                'similarity': round(max_similarity, 3),
                'match': best_match,
                'priority': classify_priority(max_similarity),
                'score': round(max_similarity, 3)
            })
        else:
            # This is a gap
            gap_score = 1.0 - max_similarity  # Invert for gap scoring
            gaps.append({
                'skill': market_skill,
                'similarity': round(max_similarity, 3),
                'match': best_match,
                'priority': classify_priority(gap_score),
                'score': round(gap_score, 3)
            })
    
    # Sort by relevance/similarity
    strengths.sort(key=lambda x: -x['similarity'])
    near_misses.sort(key=lambda x: -x['similarity'])
    gaps.sort(key=lambda x: -x['score'])  # Sort gaps by gap score (higher = bigger gap)
    
    logger.info(f"Analysis complete: {len(strengths)} strengths, {len(near_misses)} near-misses, {len(gaps)} gaps")
    
    return {
        'strengths': strengths[:10],  # Top 10 strengths
        'near_misses': near_misses[:8],  # Top 8 near misses
        'gaps': gaps[:12],  # Top 12 gaps
        'all_gaps': gaps  # Full list for appendix
    }


def _gather_learning_resources(gaps: List[Dict[str, Any]], k: int = 3) -> Dict[str, List[Dict[str, str]]]:
    """
    Gather learning resources for top gaps using Tavily search.
    
    Args:
        gaps: List of skill gaps
        k: Number of resources to gather per gap
        
    Returns:
        Dict[str, List[Dict]]: Mapping of skill -> list of resources with title and url
        
    Raises:
        ValueError: If Tavily search fails for all gaps
    """
    resources = {}
    
    # Focus on top high-priority gaps
    top_gaps = [gap for gap in gaps if gap.get('priority') == 'high'][:5]
    if not top_gaps:
        # Fall back to top gaps regardless of priority
        top_gaps = gaps[:3]
    
    logger.info(f"Gathering learning resources for {len(top_gaps)} top gaps")
    
    for gap in top_gaps:
        skill = gap['skill']
        query = f"{skill} tutorial learning guide documentation"
        
        try:
            search_results = search_web(query, k=k)
            
            skill_resources = []
            for result in search_results:
                if result.get('url') and result.get('title'):
                    skill_resources.append({
                        'title': result['title'],
                        'url': result['url']
                    })
            
            if skill_resources:
                resources[skill] = skill_resources
                logger.info(f"Found {len(skill_resources)} resources for {skill}")
            else:
                logger.warning(f"No valid resources found for {skill}")
                
        except Exception as e:
            logger.error(f"Resource search failed for {skill}: {e}")
            # Continue with other skills rather than failing completely
    
    return resources


def _build_development_plan(
    gaps: List[Dict[str, Any]], 
    lang: str = "en"
) -> Dict[str, Any]:
    """
    Build 30-60-90 day development plan with real learning resources.
    
    Args:
        gaps: Prioritized skill gaps
        lang: Language code
        
    Returns:
        Dict[str, Any]: Development plan with activities and resources
    """
    if not gaps:
        return {"30": [], "60": [], "90": [], "resources": {}}
    
    # Gather real learning resources first
    try:
        resources = _gather_learning_resources(gaps, k=3)
        logger.info(f"Gathered resources for {len(resources)} skills")
    except Exception as e:
        logger.warning(f"Resource gathering failed: {e}")
        resources = {}
    
    # Take top gaps for planning
    top_gaps = gaps[:6]  # Focus on top 6 gaps
    
    # Plan templates by skill category
    plan_templates = {
        'programming': {
            '30': 'Complete online course in {skill} fundamentals',
            '60': 'Build 2-3 practice projects using {skill}',
            '90': 'Contribute to open source {skill} project or deploy production application'
        },
        'tool': {
            '30': 'Set up {skill} environment and complete tutorials',
            '60': 'Use {skill} in personal project or work task',
            '90': 'Achieve proficiency with {skill} for production use'
        },
        'framework': {
            '30': 'Learn {skill} basics through documentation and tutorials',
            '60': 'Build demo application using {skill}',
            '90': 'Optimize and deploy {skill}-based solution'
        },
        'domain': {
            '30': 'Study {skill} fundamentals and key concepts',
            '60': 'Apply {skill} knowledge to practical problems',
            '90': 'Lead project or initiative involving {skill}'
        }
    }
    
    def categorize_skill(skill: str) -> str:
        """Categorize skill for planning template."""
        skill_lower = skill.lower()
        
        # Programming languages
        if any(lang in skill_lower for lang in ['python', 'java', 'javascript', 'typescript', 'go', 'rust', 'c++', 'sql']):
            return 'programming'
        
        # Tools and platforms
        if any(tool in skill_lower for tool in ['aws', 'docker', 'kubernetes', 'git', 'jenkins', 'terraform', 'grafana']):
            return 'tool'
        
        # Frameworks and libraries
        if any(fw in skill_lower for fw in ['react', 'django', 'tensorflow', 'pytorch', 'flask', 'express', 'spring']):
            return 'framework'
        
        # Default to domain knowledge
        return 'domain'
    
    plan = {"30": [], "60": [], "90": []}
    
    # Distribute gaps across time periods with resource links
    for i, gap in enumerate(top_gaps):
        skill = gap['skill']
        category = categorize_skill(skill)
        templates = plan_templates.get(category, plan_templates['domain'])
        
        # Alternate between time periods for variety
        if i < 2:  # First 2 gaps start in 30 days
            periods = ['30', '60', '90']
        elif i < 4:  # Next 2 gaps start in 60 days
            periods = ['60', '90']
        else:  # Remaining gaps in 90 days
            periods = ['90']
        
        for period in periods:
            if period in templates:
                task = templates[period].format(skill=skill)
                
                # Add resource links if available
                if skill in resources and resources[skill]:
                    resource_links = []
                    for resource in resources[skill][:2]:  # Top 2 resources per task
                        resource_links.append(f"[{resource['title']}]({resource['url']})")
                    
                    if resource_links:
                        task += f" - Resources: {', '.join(resource_links)}"
                
                if task not in plan[period]:  # Avoid duplicates
                    plan[period].append(task)
    
    # Add general improvement tasks
    if lang == "en":
        plan['30'].append("Set up learning tracking system and skill development goals")
        plan['60'].append("Join relevant professional communities and attend industry events")
        plan['90'].append("Update portfolio with new projects and seek feedback from peers")
    else:
        # Light localization for non-English
        plan['30'].append("Establecer sistema de seguimiento del aprendizaje y objetivos de desarrollo")
        plan['60'].append("Unirse a comunidades profesionales relevantes y asistir a eventos de la industria")
        plan['90'].append("Actualizar portafolio con nuevos proyectos y buscar feedback de pares")
    
    # Include resources for reference
    plan['resources'] = resources
    
    return plan


def _render_executive_summary(
    profile: SkillProfile, 
    market: MarketSummary, 
    analysis: Dict[str, Any], 
    lang: str = "en"
) -> str:
    """
    Render executive summary section.
    
    Args:
        profile: Skill profile
        market: Market summary
        analysis: Gap analysis results
        lang: Language code
        
    Returns:
        str: Executive summary content
    """
    total_skills = len(profile.explicit) + len(profile.implicit)
    strengths_count = len(analysis['strengths'])
    gaps_count = len(analysis['gaps'])
    
    if lang == "en":
        lines = [
            f"**Candidate Skills:** {total_skills} total skills ({len(profile.explicit)} explicit, {len(profile.implicit)} implicit)",
            f"**Market Alignment:** {strengths_count} strengths identified, {gaps_count} priority gaps to address",
            f"**Target Role:** {market.role} in {market.region}",
            "",
            "This analysis provides a comprehensive view of your current skill set compared to market demands, "
            "with actionable recommendations for professional development."
        ]
    else:
        lines = [
            f"**Habilidades del Candidato:** {total_skills} habilidades totales ({len(profile.explicit)} explícitas, {len(profile.implicit)} implícitas)",
            f"**Alineación con el Mercado:** {strengths_count} fortalezas identificadas, {gaps_count} brechas prioritarias a abordar",
            f"**Rol Objetivo:** {market.role} en {market.region}",
            "",
            "Este análisis proporciona una visión integral de tu conjunto de habilidades actual comparado con las demandas del mercado, "
            "con recomendaciones accionables para el desarrollo profesional."
        ]
    
    return "\n".join(lines)


def _render_near_misses(near_misses: List[Dict[str, Any]], lang: str = "en") -> str:
    """
    Render near misses section.
    
    Args:
        near_misses: List of near miss skills
        lang: Language code
        
    Returns:
        str: Near misses content
    """
    if not near_misses:
        return "*No near misses identified.*" if lang == "en" else "*No se identificaron coincidencias cercanas.*"
    
    intro = ("These skills are moderately aligned with your background and could be developed with focused effort:" 
             if lang == "en" else 
             "Estas habilidades están moderadamente alineadas con tu experiencia y podrían desarrollarse con esfuerzo enfocado:")
    
    lines = [intro, ""]
    for miss in near_misses:
        lines.append(f"- **{miss['skill']}** (score: {miss['score']:.2f})")
    
    return "\n".join(lines)


def _render_transferable_skills(transferable: List[str], lang: str = "en") -> str:
    """
    Render transferable skills section.
    
    Args:
        transferable: List of transferable skills
        lang: Language code
        
    Returns:
        str: Transferable skills content
    """
    if not transferable:
        return "*No transferable skills identified.*" if lang == "en" else "*No se identificaron habilidades transferibles.*"
    
    intro = ("These skills from your background apply broadly across technical roles:" 
             if lang == "en" else 
             "Estas habilidades de tu experiencia se aplican ampliamente en roles técnicos:")
    
    lines = [intro, ""]
    for skill in transferable:
        lines.append(f"- {skill}")
    
    return "\n".join(lines)


def _render_seniority_signals(signals: List[str], lang: str = "en") -> str:
    """
    Render seniority signals section.
    
    Args:
        signals: List of seniority signals
        lang: Language code
        
    Returns:
        str: Seniority signals content
    """
    if not signals:
        return "*No seniority signals identified.*" if lang == "en" else "*No se identificaron señales de seniority.*"
    
    intro = ("Evidence of leadership and advanced capabilities in your background:" 
             if lang == "en" else 
             "Evidencia de liderazgo y capacidades avanzadas en tu experiencia:")
    
    lines = [intro, ""]
    for signal in signals:
        lines.append(f"- {signal}")
    
    return "\n".join(lines)


def _render_coverage_map(analysis: Dict[str, Any], lang: str = "en") -> str:
    """
    Render coverage map appendix.
    
    Args:
        analysis: Gap analysis results with similarity-based categorization
        lang: Language code
        
    Returns:
        str: Coverage map content
    """
    intro = ("Detailed skill coverage analysis showing alignment with market requirements:" 
             if lang == "en" else 
             "Análisis detallado de cobertura de habilidades mostrando alineación con requisitos del mercado:")
    
    lines = [intro, ""]
    
    # Strengths table (similarity >= 0.75)
    if analysis['strengths']:
        lines.append("### ✅ Strengths (Similarity ≥ 0.75)")
        lines.append("")
        lines.append("| Skill | Similarity | Best Match |")
        lines.append("|-------|------------|------------|")
        for strength in analysis['strengths'][:10]:
            match = strength.get('match', 'N/A')
            lines.append(f"| {strength['skill']} | {strength['similarity']:.3f} | {match} |")
        lines.append("")
    
    # Near misses (0.5 <= similarity < 0.75)
    if analysis['near_misses']:
        lines.append("### 🎯 Near Misses (0.5 ≤ Similarity < 0.75)")
        lines.append("")
        lines.append("| Skill | Similarity | Best Match | Priority |")
        lines.append("|-------|------------|------------|----------|")
        for near_miss in analysis['near_misses'][:8]:
            match = near_miss.get('match', 'N/A')
            priority = near_miss.get('priority', 'unknown')
            lines.append(f"| {near_miss['skill']} | {near_miss['similarity']:.3f} | {match} | {priority.title()} |")
        lines.append("")
    
    # Gaps (similarity < 0.5)
    if analysis['all_gaps']:
        lines.append("### ❌ Skill Gaps (Similarity < 0.5)")
        lines.append("")
        lines.append("| Skill | Gap Score | Similarity | Priority |")
        lines.append("|-------|-----------|------------|----------|")
        
        for gap in analysis['all_gaps'][:15]:  # Show top 15
            similarity = gap.get('similarity', 0.0)
            priority = gap.get('priority', 'unknown')
            lines.append(f"| {gap['skill']} | {gap['score']:.3f} | {similarity:.3f} | {priority.title()} |")
        
        lines.append("")
    
    return "\n".join(lines)


def _render_sources(
    market_sources: List[str], 
    learning_resources: Dict[str, List[Dict[str, str]]], 
    lang: str = "en"
) -> str:
    """
    Render sources appendix with both market intelligence and learning resources.
    
    Args:
        market_sources: List of market intelligence source references
        learning_resources: Dict mapping skills to learning resources with titles and URLs
        lang: Language code
        
    Returns:
        str: Sources content with real URLs
    """
    lines = []
    
    # Market Intelligence Sources
    if market_sources:
        intro = ("Market intelligence gathered from the following sources:" 
                 if lang == "en" else 
                 "Inteligencia de mercado recopilada de las siguientes fuentes:")
        
        lines.extend([intro, ""])
        for i, source in enumerate(market_sources, 1):
            lines.append(f"{i}. {source}")
        lines.append("")
    
    # Learning Resources
    if learning_resources:
        resources_intro = ("Learning resources for skill development:" 
                          if lang == "en" else 
                          "Recursos de aprendizaje para desarrollo de habilidades:")
        
        lines.extend([resources_intro, ""])
        
        for skill, resources in learning_resources.items():
            lines.append(f"**{skill}:**")
            for resource in resources:
                if resource.get('url') and resource.get('title'):
                    lines.append(f"- [{resource['title']}]({resource['url']})")
                else:
                    lines.append(f"- {resource.get('title', 'Unknown resource')}")
            lines.append("")
    
    if not market_sources and not learning_resources:
        return ("*No sources available.*" if lang == "en" else "*No hay fuentes disponibles.*")
    
    return "\n".join(lines)


# Legacy class for backwards compatibility
class Reporter:
    """Legacy Reporter class - use synthesize_report function instead."""
    
    def __init__(self, provider_choice: str = "auto"):
        self.provider_choice = provider_choice
    
    def generate_report(
        self, 
        profile: SkillProfile, 
        market: MarketSummary, 
        lang: str = "en"
    ) -> str:
        """Generate comprehensive report."""
        return synthesize_report(profile, market, self.provider_choice, lang)
