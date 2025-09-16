"""
Tests for LangGraph builder functionality.

This module tests the graph builder's ability to create a valid graph
structure without executing the full pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from graph.builder import build_graph, create_default_state
from graph.utils import validate_graph_state, create_execution_summary


class TestGraphBuilder:
    """Test cases for graph builder functionality."""
    
    def test_build_graph_creates_valid_graph(self):
        """Test that build_graph creates a graph without errors."""
        # Should be able to build graph without execution
        graph = build_graph("auto")
        
        # Graph should be callable
        assert hasattr(graph, 'invoke')
        assert hasattr(graph, 'stream')
        assert hasattr(graph, 'get_state')
    
    def test_create_default_state(self):
        """Test default state creation."""
        cv_text = "John Doe\nSoftware Engineer with 5 years experience in Python..."
        
        state = create_default_state(
            cv_text=cv_text,
            target_role="Senior Python Developer",
            market_region="North America",
            lang="en"
        )
        
        assert state["cv_text"] == cv_text
        assert state["target_role"] == "Senior Python Developer"
        assert state["market_region"] == "North America"
        assert state["lang"] == "en"
        assert state["logs"] == []
    
    def test_validate_graph_state_valid(self):
        """Test graph state validation with valid input."""
        state = {
            "cv_text": "John Doe\nSoftware Engineer with extensive experience...",
            "target_role": "Senior AI Engineer",
            "market_region": "Global"
        }
        
        is_valid, error = validate_graph_state(state)
        assert is_valid
        assert error is None
    
    def test_validate_graph_state_missing_field(self):
        """Test graph state validation with missing field."""
        state = {
            "cv_text": "John Doe\nSoftware Engineer...",
            "target_role": "Senior AI Engineer"
            # Missing market_region
        }
        
        is_valid, error = validate_graph_state(state)
        assert not is_valid
        assert "Missing required field: market_region" in error
    
    def test_validate_graph_state_short_cv(self):
        """Test graph state validation with too short CV."""
        state = {
            "cv_text": "Short CV",  # Less than 50 characters
            "target_role": "Senior AI Engineer",
            "market_region": "Global"
        }
        
        is_valid, error = validate_graph_state(state)
        assert not is_valid
        assert "CV text too short" in error
    
    def test_create_execution_summary_success(self):
        """Test execution summary creation for successful run."""
        final_state = {
            "cv_struct": {"skills": ["Python", "SQL"], "experience": []},
            "skill_profile": {
                "explicit": ["Python", "SQL"],
                "implicit": ["API Design"],
                "transferable": ["Leadership"],
                "seniority_signals": ["Led team"]
            },
            "market_summary": {
                "in_demand_skills": ["Python", "ML"],
                "common_tools": ["Docker"],
                "frameworks": ["Django"],
                "nice_to_have": ["GraphQL"]
            },
            "report_md": "# Report\n\nThis is a test report...",
            "logs": ["✅ CV parsed", "✅ Skills analyzed"]
        }
        
        summary = create_execution_summary(final_state)
        
        assert summary["success"] is True
        assert summary["cv_parsed"] is True
        assert summary["skills_analyzed"] is True
        assert summary["market_scanned"] is True
        assert summary["report_generated"] is True
        assert summary["skills_found"]["explicit"] == 2
        assert summary["market_skills"]["in_demand"] == 2
        assert summary["report_length"] > 0
    
    def test_create_execution_summary_failure(self):
        """Test execution summary creation for failed run."""
        final_state = {
            "error": "CV parsing failed",
            "retry_count": 2,
            "logs": ["❌ CV parsing failed"]
        }
        
        summary = create_execution_summary(final_state)
        
        assert summary["success"] is False
        assert summary["error"] == "CV parsing failed"
        assert summary["retry_count"] == 2


class TestGraphImports:
    """Test that graph modules can be imported without execution."""
    
    def test_import_graph_builder(self):
        """Test that graph builder can be imported."""
        from graph.builder import build_graph, create_default_state
        
        assert callable(build_graph)
        assert callable(create_default_state)
    
    def test_import_graph_nodes(self):
        """Test that graph nodes can be imported."""
        from graph.nodes import (
            ingest_cv,
            parse_cv,
            analyze_skills,
            market_scan,
            synthesize_report_node
        )
        
        assert callable(ingest_cv)
        assert callable(parse_cv)
        assert callable(analyze_skills)
        assert callable(market_scan)
        assert callable(synthesize_report_node)
    
    def test_import_graph_utils(self):
        """Test that graph utilities can be imported."""
        from graph.utils import (
            validate_graph_state,
            create_execution_summary,
            stream_graph_execution
        )
        
        assert callable(validate_graph_state)
        assert callable(create_execution_summary)
        assert callable(stream_graph_execution)
    
    def test_import_graph_state(self):
        """Test that graph state classes can be imported."""
        from graph.state import GraphState, CVStruct, SkillProfile, MarketSummary
        
        assert GraphState is not None
        assert CVStruct is not None
        assert SkillProfile is not None
        assert MarketSummary is not None


def test_graph_node_signature():
    """Test that all graph nodes have consistent signatures."""
    from graph.nodes import (
        ingest_cv,
        parse_cv,
        analyze_skills,
        market_scan,
        synthesize_report_node
    )
    
    # All nodes should accept Dict[str, Any] and return Dict[str, Any]
    nodes = [ingest_cv, parse_cv, analyze_skills, market_scan, synthesize_report_node]
    
    for node in nodes:
        # Check function exists and is callable
        assert callable(node)
        # Check function has docstring
        assert node.__doc__ is not None
        assert len(node.__doc__.strip()) > 0


@pytest.mark.integration
def test_graph_builds_without_execution():
    """Integration test: Graph can be built without full execution."""
    # Building graph should not execute any agents
    graph = build_graph("auto", ":memory:")
    
    # Graph should have expected interface
    assert hasattr(graph, 'invoke')
    assert hasattr(graph, 'stream')
    assert hasattr(graph, 'get_state')
    
    # Should be able to create valid initial state
    state = create_default_state(
        cv_text="John Doe\nSoftware Engineer with 5+ years experience in Python and ML...",
        target_role="Senior AI Engineer",
        market_region="Global"
    )
    
    is_valid, error = validate_graph_state(state)
    assert is_valid, f"State validation failed: {error}"


if __name__ == "__main__":
    pytest.main([__file__])