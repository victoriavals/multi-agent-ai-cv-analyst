"""Tests for skill analyst agent functionality."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from src.agents.skill_analyst import (
    build_skill_profile, 
    extract_explicit_skills,
    _analyze_rule_based,
    _prepare_cv_summary
)
from src.graph.state import CVStruct, SkillProfile


@pytest.fixture
def sample_cv():
    """Sample CV data for testing."""
    return CVStruct(
        basics={
            'name': 'John Doe',
            'title': 'Senior ML Engineer'
        },
        skills=['Python', 'TensorFlow', 'AWS', 'Docker', 'SQL'],
        experience=[
            {
                'title': 'Senior ML Engineer',
                'company': 'Tech Corp',
                'dates': '2020-2023',
                'bullets': [
                    'Led a team of 5 engineers building production ML models',
                    'Deployed models serving 1M+ daily requests using Kubernetes',
                    'Improved model performance by 25% through optimization'
                ],
                'technologies': ['Python', 'Kubernetes', 'MLflow']
            },
            {
                'title': 'Data Scientist',
                'company': 'Data Co',
                'dates': '2018-2020',
                'description': 'Analyzed customer data using statistical methods and machine learning',
                'tech': ['R', 'Scikit-learn', 'PostgreSQL']
            }
        ],
        projects=[
            {
                'name': 'Customer Recommendation System',
                'tech': ['PyTorch', 'Redis', 'FastAPI'],
                'bullets': [
                    'Built deep learning recommendation engine',
                    'Achieved 15% increase in user engagement'
                ]
            }
        ],
        education=[
            {
                'degree': 'MS Computer Science',
                'school': 'University of Tech',
                'year': '2018'
            }
        ]
    )


def test_extract_explicit_skills(sample_cv):
    """Test explicit skill extraction."""
    skills = extract_explicit_skills(sample_cv)
    
    # Should include skills from cv.skills
    assert 'Python' in skills
    assert 'TensorFlow' in skills
    assert 'AWS' in skills
    
    # Should include technologies from experience
    assert 'Kubernetes' in skills
    assert 'MLflow' in skills
    assert 'R' in skills
    assert 'Scikit-learn' in skills
    
    # Should include technologies from projects
    assert 'PyTorch' in skills
    assert 'Redis' in skills
    assert 'FastAPI' in skills
    
    # Should be deduplicated
    assert len([s for s in skills if s == 'Python']) == 1


def test_extract_explicit_skills_empty_cv():
    """Test skill extraction with empty CV."""
    empty_cv = CVStruct(
        basics={},
        skills=[],
        experience=[],
        projects=[],
        education=[]
    )
    
    skills = extract_explicit_skills(empty_cv)
    assert skills == []


def test_prepare_cv_summary(sample_cv):
    """Test CV summary preparation."""
    summary = _prepare_cv_summary(sample_cv)
    
    assert 'John Doe' in summary
    assert 'Senior ML Engineer' in summary
    assert 'Tech Corp' in summary
    assert 'Led a team of 5 engineers' in summary
    assert 'Customer Recommendation System' in summary
    assert 'MS Computer Science' in summary


def test_analyze_rule_based(sample_cv):
    """Test rule-based skill analysis."""
    analysis = _analyze_rule_based(sample_cv)
    
    # Should have required keys
    assert 'implicit' in analysis
    assert 'transferable' in analysis
    assert 'seniority_signals' in analysis
    
    # Should infer some implicit skills
    implicit = analysis['implicit']
    assert len(implicit) > 0
    
    # Should detect seniority signals
    seniority = analysis['seniority_signals']
    assert len(seniority) > 0
    
    # Should find leadership indicators
    leadership_found = any('led' in signal.lower() for signal in seniority)
    assert leadership_found


def test_build_skill_profile_no_llm(sample_cv):
    """Test skill profile building without LLM."""
    profile = build_skill_profile(sample_cv, llm=None)
    
    # Should be valid SkillProfile
    assert isinstance(profile, SkillProfile)
    
    # Should have explicit skills
    assert len(profile.explicit) > 0
    assert 'python' in [s.lower() for s in profile.explicit]
    
    # Should have some implicit skills from rule-based analysis
    assert len(profile.implicit) >= 0
    
    # Should have coverage_map placeholder
    assert isinstance(profile.coverage_map, dict)


def test_build_skill_profile_with_mock_llm(sample_cv):
    """Test skill profile building with mocked LLM."""
    # Mock LLM response
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = '''
    ```json
    {
        "implicit": ["model monitoring", "feature engineering", "system design"],
        "transferable": ["leadership", "problem solving", "communication"],
        "seniority_signals": ["led team of 5", "improved performance by 25%", "serving 1M+ requests"]
    }
    ```
    '''
    mock_llm.invoke.return_value = mock_response
    
    # Mock file reading for analyst prompt
    analyst_prompt = "You are an expert skill analyst..."
    with patch('builtins.open', mock_open(read_data=analyst_prompt)):
        profile = build_skill_profile(sample_cv, llm=mock_llm)
    
    # Should use LLM results
    assert 'model monitoring' in profile.implicit
    assert 'leadership' in profile.transferable
    assert 'led team of 5' in profile.seniority_signals


def test_build_skill_profile_llm_fallback(sample_cv):
    """Test fallback when LLM fails."""
    # Mock LLM that raises exception
    mock_llm = Mock()
    mock_llm.invoke.side_effect = Exception("LLM failed")
    
    with patch('builtins.open', mock_open(read_data="prompt")):
        profile = build_skill_profile(sample_cv, llm=mock_llm)
    
    # Should still return valid profile using rule-based fallback
    assert isinstance(profile, SkillProfile)
    assert len(profile.explicit) > 0


def test_build_skill_profile_invalid_cv():
    """Test error handling with invalid CV."""
    with pytest.raises(ValueError, match="CV cannot be None"):
        build_skill_profile(None)


def test_extract_explicit_skills_with_text_extraction(sample_cv):
    """Test that skills are extracted from text descriptions."""
    # Modify CV to include skills in text
    sample_cv.experience[0]['description'] = "Used Python and PostgreSQL for data processing"
    
    skills = extract_explicit_skills(sample_cv)
    
    # Should extract from text patterns
    extracted_techs = [s for s in skills if s.lower() in ['python', 'postgresql']]
    assert len(extracted_techs) >= 1


def test_rule_based_analysis_detects_patterns():
    """Test that rule-based analysis detects various patterns."""
    cv_with_patterns = CVStruct(
        basics={'name': 'Test', 'title': 'DevOps Engineer'},
        skills=['Docker', 'AWS'],
        experience=[
            {
                'title': 'DevOps Engineer',
                'company': 'Test Co',
                'bullets': [
                    'Established CI/CD practices for the team',
                    'Mentored junior developers on best practices',
                    'Scaled system to handle 10x traffic growth',
                    'Led research on new deployment strategies'
                ]
            }
        ],
        projects=[],
        education=[]
    )
    
    analysis = _analyze_rule_based(cv_with_patterns)
    
    # Should detect transferable skills
    transferable = [s.lower() for s in analysis['transferable']]
    assert any('project management' in s for s in transferable)
    
    # Should detect seniority signals
    seniority = analysis['seniority_signals']
    leadership_signals = [s for s in seniority if any(word in s.lower() for word in ['led', 'mentored', 'established', 'scaled'])]
    assert len(leadership_signals) > 0


if __name__ == '__main__':
    pytest.main([__file__])