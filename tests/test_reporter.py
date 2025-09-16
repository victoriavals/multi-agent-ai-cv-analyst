"""Tests for recommendation and report agent functionality."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.agents.reporter import (
    synthesize_report,
    _analyze_skill_gaps,
    _build_development_plan,
    _prioritize_gaps_fallback,
    _render_executive_summary,
    _render_coverage_map
)
from src.graph.state import SkillProfile, MarketSummary


@pytest.fixture
def sample_profile():
    """Sample skill profile for testing."""
    return SkillProfile(
        explicit=['Python', 'SQL', 'Git', 'Linux'],
        implicit=['API Design', 'Database Design', 'Testing'],
        transferable=['Leadership', 'Project Management', 'Communication'],
        seniority_signals=['Led team of 5', 'Improved performance by 25%'],
        coverage_map={}
    )


@pytest.fixture
def sample_market():
    """Sample market summary for testing."""
    return MarketSummary(
        role='Senior Python Developer',
        region='Global',
        in_demand_skills=['Python', 'Machine Learning', 'Kubernetes', 'AWS'],
        common_tools=['Docker', 'Git', 'Jenkins', 'Terraform'],
        frameworks=['Django', 'FastAPI', 'TensorFlow'],
        nice_to_have=['GraphQL', 'Redis', 'Microservices'],
        sources_sample=[
            'Python Developer Jobs 2024 - https://example.com/jobs',
            'Tech Skills Survey 2024 - https://example.com/survey'
        ]
    )


@pytest.fixture
def mock_embed_function():
    """Mock embedding function that returns deterministic vectors."""
    def embed_f(text):
        # Create embeddings that ensure clear gaps for testing
        # Known candidate skills get specific embeddings
        candidate_skills = {
            'python': [0.9, 0.1, 0.1],
            'sql': [0.1, 0.9, 0.1], 
            'git': [0.1, 0.1, 0.9],
            'linux': [0.8, 0.2, 0.1],
            'api design': [0.7, 0.3, 0.2],
            'database design': [0.2, 0.8, 0.3],
            'testing': [0.3, 0.2, 0.8],
            'leadership': [0.1, 0.1, 0.1],
            'project management': [0.2, 0.2, 0.2],
            'communication': [0.3, 0.3, 0.3]
        }
        
        # High-priority market skills that should show as gaps
        high_priority_gaps = {
            'machine learning': [0.0, 0.0, 0.1],  # Very different from candidate skills
            'kubernetes': [0.0, 0.1, 0.0],
            'aws': [0.1, 0.0, 0.0]
        }
        
        text_lower = text.lower()
        if text_lower in candidate_skills:
            return np.array(candidate_skills[text_lower])
        elif text_lower in high_priority_gaps:
            return np.array(high_priority_gaps[text_lower])
        else:
            # Other market skills get moderate difference - should be near misses
            hash_val = hash(text_lower) % 50 + 25  # Range 25-75
            return np.array([0.5, 0.5, hash_val / 100.0])
    return embed_f


def test_analyze_skill_gaps(sample_profile, sample_market, mock_embed_function):
    """Test skill gap analysis."""
    analysis = _analyze_skill_gaps(sample_profile, sample_market, mock_embed_function)
    
    # Should have all required keys
    assert 'strengths' in analysis
    assert 'near_misses' in analysis
    assert 'gaps' in analysis
    assert 'all_gaps' in analysis
    
    # Should find overlapping skills as strengths
    assert 'Python' in analysis['strengths']
    assert 'Git' in analysis['strengths']
    
    # Should have identified gaps
    assert len(analysis['gaps']) > 0
    
    # Gaps should have required structure
    for gap in analysis['gaps']:
        assert 'skill' in gap
        assert 'priority' in gap
        assert 'score' in gap


def test_analyze_skill_gaps_embedding_failure(sample_profile, sample_market):
    """Test skill gap analysis fallback when embeddings fail."""
    def failing_embed_f(text):
        raise Exception("Embedding failed")
    
    analysis = _analyze_skill_gaps(sample_profile, sample_market, failing_embed_f)
    
    # Should still return valid analysis using fallback
    assert 'strengths' in analysis
    assert 'gaps' in analysis
    assert len(analysis['strengths']) > 0  # Should find Python and Git overlaps


def test_prioritize_gaps_fallback(sample_profile, sample_market):
    """Test fallback gap prioritization."""
    candidate_skills = sample_profile.explicit + sample_profile.implicit
    market_skills = sample_market.in_demand_skills + sample_market.common_tools
    
    gaps = _prioritize_gaps_fallback(candidate_skills, market_skills)
    
    # Should return gaps for skills not in candidate list
    gap_skills = [gap['skill'] for gap in gaps]
    assert 'Machine Learning' in gap_skills
    assert 'Kubernetes' in gap_skills
    assert 'Docker' in gap_skills
    
    # Should not include skills the candidate already has
    assert 'Python' not in gap_skills
    assert 'Git' not in gap_skills


def test_build_development_plan():
    """Test development plan generation."""
    gaps = [
        {'skill': 'Machine Learning', 'priority': 'high', 'score': 0.8},
        {'skill': 'Kubernetes', 'priority': 'high', 'score': 0.7},
        {'skill': 'Docker', 'priority': 'medium', 'score': 0.6},
        {'skill': 'TensorFlow', 'priority': 'medium', 'score': 0.5}
    ]
    
    plan = _build_development_plan(gaps, lang="en")
    
    # Should have all time periods
    assert '30' in plan
    assert '60' in plan
    assert '90' in plan
    
    # Should have activities for each period
    assert len(plan['30']) > 0
    assert len(plan['60']) > 0
    assert len(plan['90']) > 0
    
    # Should reference top gap skills
    plan_text = ' '.join(plan['30'] + plan['60'] + plan['90']).lower()
    assert 'machine learning' in plan_text


def test_build_development_plan_spanish():
    """Test development plan generation in Spanish."""
    gaps = [
        {'skill': 'Python', 'priority': 'high', 'score': 0.8}
    ]
    
    plan = _build_development_plan(gaps, lang="es")
    
    # Should have Spanish content in general tasks
    plan_text = ' '.join(plan['30'] + plan['60'] + plan['90'])
    assert 'Establecer' in plan_text or 'Unirse' in plan_text or 'Actualizar' in plan_text


def test_render_executive_summary(sample_profile, sample_market):
    """Test executive summary rendering."""
    analysis = {
        'strengths': ['Python', 'Git'],
        'gaps': [{'skill': 'ML', 'priority': 'high'}],
        'near_misses': []
    }
    
    summary = _render_executive_summary(sample_profile, sample_market, analysis, lang="en")
    
    assert 'Senior Python Developer' in summary
    assert 'Global' in summary
    assert str(len(sample_profile.explicit)) in summary
    assert 'strengths' in summary.lower()


def test_render_executive_summary_spanish(sample_profile, sample_market):
    """Test executive summary in Spanish."""
    analysis = {
        'strengths': ['Python', 'Git'],
        'gaps': [{'skill': 'ML', 'priority': 'high'}],
        'near_misses': []
    }
    
    summary = _render_executive_summary(sample_profile, sample_market, analysis, lang="es")
    
    # Should have Spanish content
    assert 'Habilidades del Candidato' in summary
    assert 'Alineación con el Mercado' in summary


def test_render_coverage_map():
    """Test coverage map rendering."""
    analysis = {
        'strengths': ['Python', 'Git'],
        'all_gaps': [
            {'skill': 'Machine Learning', 'priority': 'high', 'score': 0.8},
            {'skill': 'Kubernetes', 'priority': 'medium', 'score': 0.6}
        ]
    }
    
    coverage = _render_coverage_map(analysis, lang="en")
    
    assert '✅ Covered Skills' in coverage
    assert '❌ Skill Gaps' in coverage
    assert 'Python' in coverage
    assert 'Machine Learning' in coverage
    assert 'Priority' in coverage  # Table header


def test_synthesize_report_complete(sample_profile, sample_market, mock_embed_function):
    """Test complete report synthesis."""
    report = synthesize_report(sample_profile, sample_market, mock_embed_function, lang="en")
    
    # Should be a substantial markdown report
    assert len(report) > 1000  # Should be long
    
    # Should contain all major sections
    assert '# Skill Gap Analysis Report' in report
    assert 'Executive Summary' in report
    assert '🔍 Skill Gaps' in report or '💪 Strengths' in report or '🎯 Key Strengths' in report
    assert '📈 Development Plan' in report
    assert '🎯 30 Days' in report or '30 Days' in report  # May or may not have emoji
    assert '🚀 60 Days' in report or '60 Days' in report
    assert '🏆 90 Days' in report or '90 Days' in report
    assert 'Appendix: Coverage Map' in report
    assert 'Appendix: Sources' in report
    
    # Should include role and region
    assert 'Senior Python Developer' in report
    assert 'Global' in report


def test_synthesize_report_with_all_sections(sample_profile, sample_market, mock_embed_function):
    """Test that report includes all possible sections."""
    # Ensure profile has all types of content
    profile = SkillProfile(
        explicit=['Python', 'SQL'],
        implicit=['API Design'],
        transferable=['Leadership', 'Communication'],
        seniority_signals=['Led team of 5', 'Improved performance by 25%'],
        coverage_map={}
    )
    
    report = synthesize_report(profile, sample_market, mock_embed_function, lang="en")
    
    # Should include transferable skills section
    assert '🔄 Transferable Skills' in report
    assert 'Leadership' in report
    
    # Should include seniority signals section
    assert '👑 Seniority Signals' in report
    assert 'Led team of 5' in report


def test_synthesize_report_spanish(sample_profile, sample_market, mock_embed_function):
    """Test report synthesis in Spanish."""
    report = synthesize_report(sample_profile, sample_market, mock_embed_function, lang="es")
    
    # Should have English headings
    assert '# Skill Gap Analysis Report' in report
    assert 'Executive Summary' in report
    assert 'Appendix: Coverage Map' in report
    
    # Should have Spanish content in body
    assert 'Habilidades del Candidato' in report
    assert 'desarrollo profesional' in report.lower()


def test_synthesize_report_invalid_inputs(mock_embed_function):
    """Test error handling with invalid inputs."""
    with pytest.raises(ValueError, match="SkillProfile cannot be None"):
        synthesize_report(None, MarketSummary(role="test", region="test", 
                         in_demand_skills=[], common_tools=[], frameworks=[], nice_to_have=[]), 
                         mock_embed_function)
    
    with pytest.raises(ValueError, match="MarketSummary cannot be None"):
        synthesize_report(SkillProfile(explicit=[], implicit=[], transferable=[], 
                         seniority_signals=[], coverage_map={}), 
                         None, mock_embed_function)


def test_development_plan_skill_categorization():
    """Test that skills are categorized correctly for planning."""
    gaps = [
        {'skill': 'Python', 'priority': 'high', 'score': 0.8},  # Programming
        {'skill': 'AWS', 'priority': 'high', 'score': 0.7},     # Tool
        {'skill': 'Django', 'priority': 'high', 'score': 0.6},  # Framework
        {'skill': 'Machine Learning', 'priority': 'high', 'score': 0.5}  # Domain
    ]
    
    plan = _build_development_plan(gaps, lang="en")
    
    plan_text = ' '.join(plan['30'] + plan['60'] + plan['90']).lower()
    
    # Should have different types of activities based on skill categories
    assert 'course' in plan_text or 'tutorial' in plan_text  # Learning activities
    assert 'project' in plan_text  # Practical application
    assert 'environment' in plan_text or 'setup' in plan_text  # Tool setup


def test_empty_profiles_and_markets(mock_embed_function):
    """Test handling of empty profiles and markets."""
    empty_profile = SkillProfile(
        explicit=[], implicit=[], transferable=[], seniority_signals=[], coverage_map={}
    )
    empty_market = MarketSummary(
        role="Test Role", region="Test Region",
        in_demand_skills=[], common_tools=[], frameworks=[], nice_to_have=[], sources_sample=[]
    )
    
    report = synthesize_report(empty_profile, empty_market, mock_embed_function)
    
    # Should still generate a valid report
    assert '# Skill Gap Analysis Report' in report
    assert 'Test Role' in report
    assert len(report) > 500  # Should still be substantial


def test_gap_analysis_with_no_overlaps():
    """Test gap analysis when there are no skill overlaps."""
    profile = SkillProfile(
        explicit=['Skill A', 'Skill B'], implicit=[], transferable=[], seniority_signals=[], coverage_map={}
    )
    market = MarketSummary(
        role="Test", region="Test", 
        in_demand_skills=['Skill C', 'Skill D'], common_tools=[], frameworks=[], nice_to_have=[]
    )
    
    def mock_embed_f(text):
        # Return different embeddings for different skills to ensure gaps are detected
        skill_map = {
            'skill a': np.array([1.0, 0.0, 0.0]),
            'skill b': np.array([0.0, 1.0, 0.0]),
            'skill c': np.array([0.0, 0.0, 1.0]),
            'skill d': np.array([0.5, 0.5, 0.0])
        }
        return skill_map.get(text.lower(), np.array([0.1, 0.2, 0.3]))
    
    analysis = _analyze_skill_gaps(profile, market, mock_embed_f)
    
    # Should have no strengths but should identify gaps
    assert len(analysis['strengths']) == 0
    assert len(analysis['gaps']) > 0


if __name__ == '__main__':
    pytest.main([__file__])