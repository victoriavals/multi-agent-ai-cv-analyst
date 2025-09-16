"""
Markdown rendering utilities for skill gap analysis reports.
Pure string builders with no external markdown library dependencies.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


def render_report_header(role: str, region: str = None) -> str:
    """
    Render report header with role, region, and timestamp.
    
    Args:
        role: Job role being analyzed (e.g., "AI Engineer", "ML Engineer")
        region: Geographic region (e.g., "San Francisco", "Remote")
        
    Returns:
        Formatted markdown header string
        
    Examples:
        >>> header = render_report_header("AI Engineer", "San Francisco")
        >>> print(header)
        # Skill Gap Analysis Report
        
        **Role:** AI Engineer  
        **Region:** San Francisco  
        **Generated:** 2024-03-15 10:30:45
        
        ---
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header_lines = [
        "# Skill Gap Analysis Report",
        "",
        f"**Role:** {role}"
    ]
    
    if region:
        header_lines.append(f"**Region:** {region}")
    
    header_lines.extend([
        f"**Generated:** {timestamp}",
        "",
        "---",
        ""
    ])
    
    return "\n".join(header_lines)


def render_strengths(items: List[str]) -> str:
    """
    Render strengths section as markdown bullet list.
    
    Args:
        items: List of strength descriptions
        
    Returns:
        Formatted markdown strengths section
        
    Examples:
        >>> strengths = ["Python programming", "Machine learning", "Data analysis"]
        >>> print(render_strengths(strengths))
        ## 🎯 Key Strengths
        
        - Python programming
        - Machine learning
        - Data analysis
    """
    if not items:
        return "## 🎯 Key Strengths\n\n*No specific strengths identified.*\n"
    
    lines = [
        "## 🎯 Key Strengths",
        ""
    ]
    
    for item in items:
        lines.append(f"- {item}")
    
    lines.append("")
    return "\n".join(lines)


def render_gaps(items: List[str]) -> str:
    """
    Render skill gaps section as markdown bullet list.
    
    Args:
        items: List of skill gap descriptions
        
    Returns:
        Formatted markdown gaps section
        
    Examples:
        >>> gaps = ["LLM fine-tuning", "Vector databases", "MLOps"]
        >>> print(render_gaps(gaps))
        ## 🔍 Skill Gaps
        
        - LLM fine-tuning
        - Vector databases
        - MLOps
    """
    if not items:
        return "## 🔍 Skill Gaps\n\n*No significant skill gaps identified.*\n"
    
    lines = [
        "## 🔍 Skill Gaps",
        ""
    ]
    
    for item in items:
        lines.append(f"- {item}")
    
    lines.append("")
    return "\n".join(lines)


def render_plan(plan306090: Dict[str, List[str]]) -> str:
    """
    Render 30-60-90 day development plan as structured markdown.
    
    Args:
        plan306090: Dictionary with keys "30", "60", "90" containing action items
        
    Returns:
        Formatted markdown development plan
        
    Examples:
        >>> plan = {
        ...     "30": ["Learn RAG fundamentals", "Set up vector database"],
        ...     "60": ["Build RAG prototype", "Integrate with LLM"],
        ...     "90": ["Deploy production system", "Optimize performance"]
        ... }
        >>> print(render_plan(plan))
        ## 📈 Development Plan
        
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
    if not plan306090:
        return "## 📈 Development Plan\n\n*No development plan provided.*\n"
    
    lines = [
        "## 📈 Development Plan",
        ""
    ]
    
    # Period configuration with emojis
    periods = [
        ("30", "🎯 30 Days"),
        ("60", "🚀 60 Days"), 
        ("90", "🏆 90 Days")
    ]
    
    for period_key, period_title in periods:
        if period_key in plan306090 and plan306090[period_key]:
            lines.extend([
                f"### {period_title}",
                ""
            ])
            
            for item in plan306090[period_key]:
                lines.append(f"- {item}")
            
            lines.append("")
    
    return "\n".join(lines)


def render_skills_table(skills: List[Dict[str, Any]], title: str = "Skills") -> str:
    """
    Render skills as markdown table with proficiency levels.
    
    Args:
        skills: List of skill dictionaries with 'name' and optionally 'level', 'match_score'
        title: Table section title
        
    Returns:
        Formatted markdown table
        
    Examples:
        >>> skills = [
        ...     {"name": "Python", "level": "Expert", "match_score": 0.95},
        ...     {"name": "TensorFlow", "level": "Intermediate", "match_score": 0.78}
        ... ]
        >>> print(render_skills_table(skills, "Technical Skills"))
        ## Technical Skills
        
        | Skill | Level | Match Score |
        |-------|-------|-------------|
        | Python | Expert | 95% |
        | TensorFlow | Intermediate | 78% |
    """
    if not skills:
        return f"## {title}\n\n*No {title.lower()} data available.*\n"
    
    lines = [
        f"## {title}",
        ""
    ]
    
    # Determine columns based on available data
    has_level = any('level' in skill for skill in skills)
    has_score = any('match_score' in skill for skill in skills)
    
    # Build header
    headers = ["Skill"]
    separators = ["-------"]
    
    if has_level:
        headers.append("Level")
        separators.append("-------")
    
    if has_score:
        headers.append("Match Score")
        separators.append("----------")
    
    lines.extend([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separators) + " |"
    ])
    
    # Build rows
    for skill in skills:
        row = [skill.get('name', 'Unknown')]
        
        if has_level:
            row.append(skill.get('level', 'N/A'))
        
        if has_score:
            score = skill.get('match_score')
            if score is not None:
                if isinstance(score, (int, float)):
                    row.append(f"{score:.0%}" if score <= 1 else f"{score:.0f}%")
                else:
                    row.append(str(score))
            else:
                row.append('N/A')
        
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    return "\n".join(lines)


def render_section(title: str, content: str, level: int = 2) -> str:
    """
    Render a generic markdown section with title and content.
    
    Args:
        title: Section title
        content: Section content
        level: Heading level (1-6)
        
    Returns:
        Formatted markdown section
        
    Examples:
        >>> section = render_section("Summary", "This is the summary content.", 2)
        >>> print(section)
        ## Summary
        
        This is the summary content.
    """
    if level < 1 or level > 6:
        level = 2
    
    heading_prefix = "#" * level
    
    return f"{heading_prefix} {title}\n\n{content}\n"


def render_bullet_list(items: List[str], indent: int = 0) -> str:
    """
    Render items as markdown bullet list with optional indentation.
    
    Args:
        items: List of items to render
        indent: Number of spaces to indent (for nested lists)
        
    Returns:
        Formatted markdown bullet list
        
    Examples:
        >>> items = ["Item 1", "Item 2", "Item 3"]
        >>> print(render_bullet_list(items))
        - Item 1
        - Item 2
        - Item 3
        
        >>> print(render_bullet_list(items, indent=2))
          - Item 1
          - Item 2
          - Item 3
    """
    if not items:
        return ""
    
    indent_str = " " * indent
    lines = []
    
    for item in items:
        lines.append(f"{indent_str}- {item}")
    
    return "\n".join(lines) + "\n"


def render_numbered_list(items: List[str], indent: int = 0) -> str:
    """
    Render items as markdown numbered list with optional indentation.
    
    Args:
        items: List of items to render
        indent: Number of spaces to indent
        
    Returns:
        Formatted markdown numbered list
        
    Examples:
        >>> items = ["First step", "Second step", "Third step"]
        >>> print(render_numbered_list(items))
        1. First step
        2. Second step
        3. Third step
    """
    if not items:
        return ""
    
    indent_str = " " * indent
    lines = []
    
    for i, item in enumerate(items, 1):
        lines.append(f"{indent_str}{i}. {item}")
    
    return "\n".join(lines) + "\n"


def render_code_block(code: str, language: str = "") -> str:
    """
    Render code block with optional language specification.
    
    Args:
        code: Code content
        language: Programming language for syntax highlighting
        
    Returns:
        Formatted markdown code block
        
    Examples:
        >>> code = "print('Hello, World!')"
        >>> print(render_code_block(code, "python"))
        ```python
        print('Hello, World!')
        ```
    """
    return f"```{language}\n{code}\n```\n"


def render_quote(text: str) -> str:
    """
    Render text as markdown blockquote.
    
    Args:
        text: Text to quote
        
    Returns:
        Formatted markdown blockquote
        
    Examples:
        >>> quote = render_quote("This is a quoted text.")
        >>> print(quote)
        > This is a quoted text.
    """
    lines = text.split('\n')
    quoted_lines = [f"> {line}" for line in lines]
    return "\n".join(quoted_lines) + "\n"


def render_horizontal_rule() -> str:
    """
    Render horizontal rule separator.
    
    Returns:
        Markdown horizontal rule
        
    Examples:
        >>> print(render_horizontal_rule())
        ---
    """
    return "---\n"


# Legacy compatibility
def render_md(md_text: str) -> str:
    """Legacy function - returns input as-is for backward compatibility."""
    return md_text
