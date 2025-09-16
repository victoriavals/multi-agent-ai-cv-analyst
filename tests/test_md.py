"""
Tests for Markdown rendering utilities.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from src.utils.md import (
    render_report_header,
    render_strengths,
    render_gaps,
    render_plan,
    render_skills_table,
    render_section,
    render_bullet_list,
    render_numbered_list,
    render_code_block,
    render_quote,
    render_horizontal_rule,
    render_md
)


class TestRenderReportHeader:
    """Test report header rendering."""
    
    @patch('src.utils.md.datetime')
    def test_render_header_with_region(self, mock_datetime):
        """Test header rendering with role and region."""
        mock_datetime.now.return_value.strftime.return_value = "2024-03-15 10:30:45"
        
        result = render_report_header("AI Engineer", "San Francisco")
        
        expected = """# Skill Gap Analysis Report

**Role:** AI Engineer
**Region:** San Francisco
**Generated:** 2024-03-15 10:30:45

---
"""
        assert result == expected
    
    @patch('src.utils.md.datetime')
    def test_render_header_without_region(self, mock_datetime):
        """Test header rendering with only role."""
        mock_datetime.now.return_value.strftime.return_value = "2024-03-15 10:30:45"
        
        result = render_report_header("Data Scientist")
        
        expected = """# Skill Gap Analysis Report

**Role:** Data Scientist
**Generated:** 2024-03-15 10:30:45

---
"""
        assert result == expected
    
    @patch('src.utils.md.datetime')
    def test_render_header_with_none_region(self, mock_datetime):
        """Test header rendering with None region."""
        mock_datetime.now.return_value.strftime.return_value = "2024-03-15 10:30:45"
        
        result = render_report_header("ML Engineer", None)
        
        expected = """# Skill Gap Analysis Report

**Role:** ML Engineer
**Generated:** 2024-03-15 10:30:45

---
"""
        assert result == expected


class TestRenderStrengths:
    """Test strengths section rendering."""
    
    def test_render_strengths_with_items(self):
        """Test rendering strengths with multiple items."""
        items = ["Python programming", "Machine learning", "Data analysis"]
        
        result = render_strengths(items)
        
        expected = """## 🎯 Key Strengths

- Python programming
- Machine learning
- Data analysis
"""
        assert result == expected
    
    def test_render_strengths_empty_list(self):
        """Test rendering strengths with empty list."""
        result = render_strengths([])
        
        expected = "## 🎯 Key Strengths\n\n*No specific strengths identified.*\n"
        assert result == expected
    
    def test_render_strengths_single_item(self):
        """Test rendering strengths with single item."""
        result = render_strengths(["Deep learning expertise"])
        
        expected = """## 🎯 Key Strengths

- Deep learning expertise
"""
        assert result == expected


class TestRenderGaps:
    """Test gaps section rendering."""
    
    def test_render_gaps_with_items(self):
        """Test rendering gaps with multiple items."""
        items = ["LLM fine-tuning", "Vector databases", "MLOps"]
        
        result = render_gaps(items)
        
        expected = """## 🔍 Skill Gaps

- LLM fine-tuning
- Vector databases
- MLOps
"""
        assert result == expected
    
    def test_render_gaps_empty_list(self):
        """Test rendering gaps with empty list."""
        result = render_gaps([])
        
        expected = "## 🔍 Skill Gaps\n\n*No significant skill gaps identified.*\n"
        assert result == expected


class TestRenderPlan:
    """Test development plan rendering."""
    
    def test_render_plan_complete(self):
        """Test rendering complete 30-60-90 plan."""
        plan = {
            "30": ["Learn RAG fundamentals", "Set up vector database"],
            "60": ["Build RAG prototype", "Integrate with LLM"],
            "90": ["Deploy production system", "Optimize performance"]
        }
        
        result = render_plan(plan)
        
        expected = """## 📈 Development Plan

### 🎯 30 Days

- Learn RAG fundamentals
- Set up vector database

### 🚀 60 Days

- Build RAG prototype
- Integrate with LLM

### 🏆 90 Days

- Deploy production system
- Optimize performance
"""
        assert result == expected
    
    def test_render_plan_partial(self):
        """Test rendering partial plan (only some periods)."""
        plan = {
            "30": ["Start with basics"],
            "90": ["Advanced implementation"]
        }
        
        result = render_plan(plan)
        
        expected = """## 📈 Development Plan

### 🎯 30 Days

- Start with basics

### 🏆 90 Days

- Advanced implementation
"""
        assert result == expected
    
    def test_render_plan_empty(self):
        """Test rendering empty plan."""
        result = render_plan({})
        
        expected = "## 📈 Development Plan\n\n*No development plan provided.*\n"
        assert result == expected
    
    def test_render_plan_empty_periods(self):
        """Test rendering plan with empty periods."""
        plan = {
            "30": [],
            "60": ["Something"],
            "90": []
        }
        
        result = render_plan(plan)
        
        expected = """## 📈 Development Plan

### 🚀 60 Days

- Something
"""
        assert result == expected


class TestRenderSkillsTable:
    """Test skills table rendering."""
    
    def test_render_skills_table_full(self):
        """Test rendering skills table with all fields."""
        skills = [
            {"name": "Python", "level": "Expert", "match_score": 0.95},
            {"name": "TensorFlow", "level": "Intermediate", "match_score": 0.78}
        ]
        
        result = render_skills_table(skills, "Technical Skills")
        
        expected = """## Technical Skills

| Skill | Level | Match Score |
| ------- | ------- | ---------- |
| Python | Expert | 95% |
| TensorFlow | Intermediate | 78% |
"""
        assert result == expected
    
    def test_render_skills_table_names_only(self):
        """Test rendering skills table with names only."""
        skills = [
            {"name": "Python"},
            {"name": "JavaScript"}
        ]
        
        result = render_skills_table(skills)
        
        expected = """## Skills

| Skill |
| ------- |
| Python |
| JavaScript |
"""
        assert result == expected
    
    def test_render_skills_table_with_level(self):
        """Test rendering skills table with levels."""
        skills = [
            {"name": "Python", "level": "Expert"},
            {"name": "Go", "level": "Beginner"}
        ]
        
        result = render_skills_table(skills)
        
        expected = """## Skills

| Skill | Level |
| ------- | ------- |
| Python | Expert |
| Go | Beginner |
"""
        assert result == expected
    
    def test_render_skills_table_empty(self):
        """Test rendering empty skills table."""
        result = render_skills_table([], "My Skills")
        
        expected = "## My Skills\n\n*No my skills data available.*\n"
        assert result == expected
    
    def test_render_skills_table_score_formats(self):
        """Test different score formats in skills table."""
        skills = [
            {"name": "Skill1", "match_score": 0.95},  # Float < 1
            {"name": "Skill2", "match_score": 85},     # Int > 1
            {"name": "Skill3", "match_score": "High"}, # String
            {"name": "Skill4", "match_score": 1.0}     # Float = 1
        ]
        
        result = render_skills_table(skills)
        
        assert "| Skill1 | 95% |" in result
        assert "| Skill2 | 85% |" in result
        assert "| Skill3 | High |" in result
        assert "| Skill4 | 100% |" in result


class TestRenderSection:
    """Test generic section rendering."""
    
    def test_render_section_default_level(self):
        """Test rendering section with default heading level."""
        result = render_section("Summary", "This is the summary content.")
        
        expected = "## Summary\n\nThis is the summary content.\n"
        assert result == expected
    
    def test_render_section_custom_level(self):
        """Test rendering section with custom heading level."""
        result = render_section("Details", "Detailed information.", 3)
        
        expected = "### Details\n\nDetailed information.\n"
        assert result == expected
    
    def test_render_section_invalid_level(self):
        """Test rendering section with invalid heading level."""
        result = render_section("Title", "Content", 10)
        
        expected = "## Title\n\nContent\n"
        assert result == expected
    
    def test_render_section_level_1(self):
        """Test rendering section with level 1 heading."""
        result = render_section("Main Title", "Main content", 1)
        
        expected = "# Main Title\n\nMain content\n"
        assert result == expected


class TestRenderBulletList:
    """Test bullet list rendering."""
    
    def test_render_bullet_list_basic(self):
        """Test basic bullet list rendering."""
        items = ["Item 1", "Item 2", "Item 3"]
        
        result = render_bullet_list(items)
        
        expected = "- Item 1\n- Item 2\n- Item 3\n"
        assert result == expected
    
    def test_render_bullet_list_with_indent(self):
        """Test bullet list with indentation."""
        items = ["Nested 1", "Nested 2"]
        
        result = render_bullet_list(items, indent=2)
        
        expected = "  - Nested 1\n  - Nested 2\n"
        assert result == expected
    
    def test_render_bullet_list_empty(self):
        """Test empty bullet list."""
        result = render_bullet_list([])
        
        assert result == ""


class TestRenderNumberedList:
    """Test numbered list rendering."""
    
    def test_render_numbered_list_basic(self):
        """Test basic numbered list rendering."""
        items = ["First step", "Second step", "Third step"]
        
        result = render_numbered_list(items)
        
        expected = "1. First step\n2. Second step\n3. Third step\n"
        assert result == expected
    
    def test_render_numbered_list_with_indent(self):
        """Test numbered list with indentation."""
        items = ["Sub-step A", "Sub-step B"]
        
        result = render_numbered_list(items, indent=4)
        
        expected = "    1. Sub-step A\n    2. Sub-step B\n"
        assert result == expected
    
    def test_render_numbered_list_empty(self):
        """Test empty numbered list."""
        result = render_numbered_list([])
        
        assert result == ""


class TestRenderCodeBlock:
    """Test code block rendering."""
    
    def test_render_code_block_with_language(self):
        """Test code block with language specification."""
        code = "print('Hello, World!')"
        
        result = render_code_block(code, "python")
        
        expected = "```python\nprint('Hello, World!')\n```\n"
        assert result == expected
    
    def test_render_code_block_without_language(self):
        """Test code block without language specification."""
        code = "console.log('Hello');"
        
        result = render_code_block(code)
        
        expected = "```\nconsole.log('Hello');\n```\n"
        assert result == expected
    
    def test_render_code_block_multiline(self):
        """Test multiline code block."""
        code = "def hello():\n    print('Hello')\n    return True"
        
        result = render_code_block(code, "python")
        
        expected = "```python\ndef hello():\n    print('Hello')\n    return True\n```\n"
        assert result == expected


class TestRenderQuote:
    """Test blockquote rendering."""
    
    def test_render_quote_single_line(self):
        """Test single line quote."""
        result = render_quote("This is a quoted text.")
        
        expected = "> This is a quoted text.\n"
        assert result == expected
    
    def test_render_quote_multiline(self):
        """Test multiline quote."""
        text = "First line of quote.\nSecond line of quote."
        
        result = render_quote(text)
        
        expected = "> First line of quote.\n> Second line of quote.\n"
        assert result == expected


class TestRenderHorizontalRule:
    """Test horizontal rule rendering."""
    
    def test_render_horizontal_rule(self):
        """Test horizontal rule rendering."""
        result = render_horizontal_rule()
        
        expected = "---\n"
        assert result == expected


class TestLegacyCompatibility:
    """Test legacy function compatibility."""
    
    def test_render_md_passthrough(self):
        """Test legacy render_md function."""
        text = "# Some markdown text"
        
        result = render_md(text)
        
        assert result == text


class TestIntegration:
    """Test integrated markdown rendering."""
    
    @patch('src.utils.md.datetime')
    def test_complete_report_rendering(self, mock_datetime):
        """Test rendering a complete report."""
        mock_datetime.now.return_value.strftime.return_value = "2024-03-15 10:30:45"
        
        # Build complete report
        header = render_report_header("AI Engineer", "Remote")
        strengths = render_strengths(["Python", "Machine Learning"])
        gaps = render_gaps(["LLM fine-tuning", "Vector databases"])
        plan = render_plan({
            "30": ["Learn RAG basics"],
            "90": ["Build production system"]
        })
        
        complete_report = header + strengths + gaps + plan
        
        # Verify structure
        assert "# Skill Gap Analysis Report" in complete_report
        assert "## 🎯 Key Strengths" in complete_report
        assert "## 🔍 Skill Gaps" in complete_report
        assert "## 📈 Development Plan" in complete_report
        assert "### 🎯 30 Days" in complete_report
        assert "### 🏆 90 Days" in complete_report
        assert "- Python" in complete_report
        assert "- LLM fine-tuning" in complete_report
        assert "- Learn RAG basics" in complete_report
    
    def test_skills_table_integration(self):
        """Test skills table with mixed data."""
        skills = [
            {"name": "Python", "level": "Expert"},
            {"name": "TensorFlow", "match_score": 0.8},
            {"name": "Docker", "level": "Intermediate", "match_score": 0.6},
            {"name": "Kubernetes"}
        ]
        
        result = render_skills_table(skills, "Mixed Skills")
        
        # Should include all columns when any skill has that data
        assert "| Skill | Level | Match Score |" in result
        assert "| Python | Expert | N/A |" in result
        assert "| TensorFlow | N/A | 80% |" in result
        assert "| Docker | Intermediate | 60% |" in result
        assert "| Kubernetes | N/A | N/A |" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])