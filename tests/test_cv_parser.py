"""
Unit tests for CV parsing and normalization agent.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.agents.cv_parser import (
    pre_clean_text,
    extract_skills_rule_based,
    parse_cv_text,
    _parse_with_llm,
    _parse_rule_based,
    CVParser
)
from src.graph.state import CVStruct


class TestPreCleanText:
    """Test CV text pre-cleaning functionality."""
    
    def test_empty_text(self):
        """Test handling of empty text."""
        assert pre_clean_text("") == ""
        assert pre_clean_text("   ") == ""
        assert pre_clean_text(None) == ""
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "John  Doe\t\tSenior   Engineer\n\n\nSkills"
        expected = "John Doe Senior Engineer\n\nSkills"
        assert pre_clean_text(text) == expected
    
    def test_remove_line_endings(self):
        """Test line ending normalization."""
        text = "John Doe\r\nSenior Engineer\rSkills"
        expected = "John Doe\nSenior Engineer\nSkills"
        assert pre_clean_text(text) == expected
    
    def test_remove_repeated_headers(self):
        """Test removal of repeated headers."""
        text = """John Doe
        CONFIDENTIAL
        Senior Engineer
        Page 1 of 2
        Skills: Python
        Resume - John Doe
        Experience"""
        
        result = pre_clean_text(text)
        assert "CONFIDENTIAL" not in result.upper()
        assert "PAGE 1 OF 2" not in result.upper()
        assert "John Doe" in result
        assert "Skills: Python" in result
    
    def test_remove_duplicate_lines(self):
        """Test removal of duplicate consecutive lines."""
        text = """John Doe
        John Doe
        Senior Engineer
        Senior Engineer
        Skills"""
        
        result = pre_clean_text(text)
        lines = result.split('\n')
        assert lines.count("John Doe") == 1
        assert lines.count("Senior Engineer") == 1


class TestExtractSkillsRuleBased:
    """Test rule-based skill extraction."""
    
    def test_skills_section_extraction(self):
        """Test extraction from skills section."""
        text = """John Doe
        SKILLS:
        Python, Java, SQL
        Machine Learning, TensorFlow
        
        EXPERIENCE:
        Software Engineer"""
        
        skills = extract_skills_rule_based(text)
        assert "Python" in skills
        assert "Java" in skills
        assert "SQL" in skills
        assert "Machine Learning" in skills
        assert "TensorFlow" in skills
        assert "Software Engineer" not in skills  # From different section
    
    def test_bullet_point_skills(self):
        """Test extraction of bullet point skills."""
        text = """TECHNICAL SKILLS:
        • Python
        • JavaScript
        - TensorFlow
        * AWS"""
        
        skills = extract_skills_rule_based(text)
        assert "Python" in skills
        assert "JavaScript" in skills
        assert "TensorFlow" in skills
        assert "AWS" in skills
    
    def test_common_tech_patterns(self):
        """Test extraction of common technology patterns."""
        text = """Experience with Python and TensorFlow.
        Built applications using React and Node.js.
        Deployed on AWS with Docker containers."""
        
        skills = extract_skills_rule_based(text)
        assert "Python" in skills
        assert "TensorFlow" in skills
        assert "React" in skills
        assert "Node.js" in skills
        assert "AWS" in skills
        assert "Docker" in skills
    
    def test_filter_non_skills(self):
        """Test filtering of non-skill terms."""
        text = """SKILLS:
        Python, 5 years, Java
        10 years experience, SQL"""
        
        skills = extract_skills_rule_based(text)
        assert "Python" in skills
        assert "Java" in skills
        assert "SQL" in skills
        assert "5 years" not in skills
        assert "10 years experience" not in skills
    
    def test_deduplication(self):
        """Test skill deduplication."""
        text = """SKILLS:
        Python, python, PYTHON
        Java, java"""
        
        skills = extract_skills_rule_based(text)
        python_count = sum(1 for skill in skills if skill.lower() == "python")
        java_count = sum(1 for skill in skills if skill.lower() == "java")
        
        assert python_count == 1
        assert java_count == 1


class TestParseCVText:
    """Test main CV parsing function."""
    
    def test_empty_cv_text(self):
        """Test handling of empty CV text."""
        with pytest.raises(ValueError, match="CV text cannot be empty"):
            parse_cv_text("")
        
        with pytest.raises(ValueError, match="CV text cannot be empty"):
            parse_cv_text("   ")
    
    def test_rule_based_parsing_only(self):
        """Test parsing without LLM (rule-based only)."""
        cv_text = """John Doe
        Senior AI Engineer
        john@email.com | +1-555-0123
        
        SKILLS:
        Python, TensorFlow, AWS
        
        EXPERIENCE:
        AI Engineer | TechCorp | 2020-Present
        • Developed ML models
        • Led team of 5 engineers"""
        
        result = parse_cv_text(cv_text, llm=None)
        
        assert isinstance(result, CVStruct)
        assert result.basics['name'] == "John Doe"
        assert result.basics['title'] == "Senior AI Engineer"
        assert "john@email.com" in result.basics['email']
        assert "Python" in result.skills
        assert "TensorFlow" in result.skills
        assert "AWS" in result.skills
        assert len(result.experience) > 0
    
    @patch('src.agents.cv_parser._parse_with_llm')
    def test_llm_parsing_success(self, mock_llm_parse):
        """Test successful LLM parsing."""
        mock_llm = Mock()
        
        mock_llm_parse.return_value = {
            'basics': {'name': 'John Doe', 'title': 'AI Engineer'},
            'skills': ['Python', 'TensorFlow'],
            'experience': [],
            'projects': [],
            'education': [],
            'certifications': []
        }
        
        cv_text = "John Doe\nAI Engineer\nPython, TensorFlow"
        result = parse_cv_text(cv_text, llm=mock_llm)
        
        assert isinstance(result, CVStruct)
        assert result.basics['name'] == 'John Doe'
        assert 'Python' in result.skills
        mock_llm_parse.assert_called_once()
    
    @patch('src.agents.cv_parser._parse_with_llm')
    @patch('src.agents.cv_parser._parse_rule_based')
    def test_llm_fallback_to_rule_based(self, mock_rule_parse, mock_llm_parse):
        """Test fallback to rule-based when LLM fails."""
        mock_llm = Mock()
        
        # LLM parsing fails
        mock_llm_parse.side_effect = Exception("LLM failed")
        
        # Rule-based parsing succeeds
        mock_rule_parse.return_value = {
            'basics': {'name': 'John Doe'},
            'skills': ['Python'],
            'experience': [],
            'projects': [],
            'education': [],
            'certifications': []
        }
        
        cv_text = "John Doe\nPython"
        result = parse_cv_text(cv_text, llm=mock_llm)
        
        assert isinstance(result, CVStruct)
        mock_llm_parse.assert_called_once()
        mock_rule_parse.assert_called_once()


class TestParseWithLLM:
    """Test LLM-based parsing."""
    
    @patch('builtins.open', new_callable=mock_open, read_data="System prompt for parsing")
    def test_successful_llm_parsing(self, mock_file):
        """Test successful LLM parsing with JSON response."""
        mock_llm = Mock()
        
        # Mock LLM response with JSON
        json_response = """{
            "basics": {"name": "John Doe", "title": "AI Engineer"},
            "skills": ["Python", "TensorFlow"],
            "experience": [],
            "projects": [],
            "education": [],
            "certifications": []
        }"""
        
        mock_llm.invoke.return_value = AIMessage(content=f"```json\n{json_response}\n```")
        
        result = _parse_with_llm("John Doe\nAI Engineer", mock_llm)
        
        assert result['basics']['name'] == 'John Doe'
        assert 'Python' in result['skills']
        mock_llm.invoke.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data="System prompt")
    def test_llm_json_without_code_blocks(self, mock_file):
        """Test LLM response without code blocks."""
        mock_llm = Mock()
        
        json_response = """{
            "basics": {"name": "Jane Doe"},
            "skills": ["Java"],
            "experience": [],
            "projects": [],
            "education": [],
            "certifications": []
        }"""
        
        mock_llm.invoke.return_value = AIMessage(content=json_response)
        
        result = _parse_with_llm("Jane Doe", mock_llm)
        
        assert result['basics']['name'] == 'Jane Doe'
        assert 'Java' in result['skills']
    
    @patch('builtins.open', new_callable=mock_open, read_data="System prompt")
    def test_llm_invalid_json(self, mock_file):
        """Test LLM response with invalid JSON."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Invalid response without JSON")
        
        with pytest.raises(Exception, match="No JSON found"):
            _parse_with_llm("John Doe", mock_llm)
    
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_missing_prompt_file(self, mock_file):
        """Test handling of missing prompt file."""
        mock_llm = Mock()
        
        with pytest.raises(Exception, match="Parser prompt not found"):
            _parse_with_llm("John Doe", mock_llm)


class TestParseRuleBased:
    """Test rule-based parsing."""
    
    def test_basic_info_extraction(self):
        """Test extraction of basic information."""
        text = """John Doe
        Senior AI Engineer
        john@email.com | +1-555-0123"""
        
        result = _parse_rule_based(text)
        
        assert result['basics']['name'] == 'John Doe'
        assert result['basics']['title'] == 'Senior AI Engineer'
        assert 'john@email.com' in result['basics']['email']
        assert '+1-555-0123' in result['basics']['phone']
    
    def test_experience_section(self):
        """Test extraction of experience section."""
        text = """EXPERIENCE:
        Senior Engineer | TechCorp | 2020-Present
        • Led development team
        • Built ML systems
        
        Junior Engineer | StartupXYZ | 2018-2020"""
        
        result = _parse_rule_based(text)
        
        assert len(result['experience']) >= 1
        # Should have parsed at least one job entry
    
    def test_projects_section(self):
        """Test extraction of projects section."""
        text = """PROJECTS:
        AI Chatbot System
        • Built using Python and TensorFlow
        
        Recommendation Engine"""
        
        result = _parse_rule_based(text)
        
        assert len(result['projects']) >= 1
    
    def test_education_section(self):
        """Test extraction of education section."""
        text = """EDUCATION:
        Master of Science in Computer Science
        Stanford University | 2016-2018
        
        Bachelor of Engineering
        UC Berkeley | 2012-2016"""
        
        result = _parse_rule_based(text)
        
        assert len(result['education']) >= 1
    
    def test_certifications_section(self):
        """Test extraction of certifications section."""
        text = """CERTIFICATIONS:
        AWS Certified Solutions Architect
        Google Cloud Professional ML Engineer
        Certified Kubernetes Administrator"""
        
        result = _parse_rule_based(text)
        
        assert len(result['certifications']) >= 1


class TestCVParserClass:
    """Test legacy CVParser class."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = CVParser()
        assert parser.llm is None
        
        mock_llm = Mock()
        parser_with_llm = CVParser(mock_llm)
        assert parser_with_llm.llm == mock_llm
    
    @patch('src.agents.cv_parser.parse_cv_text')
    def test_parser_parse_method(self, mock_parse):
        """Test parser parse method."""
        mock_parse.return_value = CVStruct()
        
        parser = CVParser()
        result = parser.parse("test cv text")
        
        mock_parse.assert_called_once_with("test cv text", None)
        assert isinstance(result, CVStruct)


class TestSampleCV:
    """Test parsing with sample CV file."""
    
    def test_sample_cv_parsing(self):
        """Test parsing of the sample CV file."""
        # Read the sample CV
        sample_path = Path(__file__).parent.parent.parent / "sample" / "sample_cv.txt"
        
        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_cv = f.read()
        except FileNotFoundError:
            pytest.skip("Sample CV file not found")
        
        # Parse without LLM (rule-based only)
        result = parse_cv_text(sample_cv, llm=None)
        
        # Validate results
        assert isinstance(result, CVStruct)
        assert result.basics is not None
        assert len(result.skills) > 0
        
        # Should extract at least some skills
        skill_names = [skill.lower() for skill in result.skills]
        assert any('python' in skill for skill in skill_names)
        
        # Should have some structure (not all empty)
        sections_with_content = 0
        if result.skills:
            sections_with_content += 1
        if result.experience:
            sections_with_content += 1
        if result.projects:
            sections_with_content += 1
        if result.education:
            sections_with_content += 1
        if result.certifications:
            sections_with_content += 1
        
        assert sections_with_content >= 2  # At least 2 sections should have content
    
    def test_sample_cv_non_empty_sections(self):
        """Test that sample CV parsing returns non-empty sections."""
        sample_path = Path(__file__).parent.parent.parent / "sample" / "sample_cv.txt"
        
        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_cv = f.read()
        except FileNotFoundError:
            pytest.skip("Sample CV file not found")
        
        result = parse_cv_text(sample_cv, llm=None)
        
        # According to acceptance criteria: "returns non-empty sections"
        assert len(result.skills) > 0, "Skills section should not be empty"
        
        # Check that we get meaningful data
        assert result.basics is not None, "Basics should be parsed"
        
        # At least one major section should have content
        has_content = (
            len(result.skills) > 0 or
            len(result.experience) > 0 or
            len(result.projects) > 0 or
            len(result.education) > 0
        )
        assert has_content, "At least one section should have content"