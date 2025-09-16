"""
LangGraph builder for skill gap analysis pipeline.

This module provides a graph builder that creates a LangGraph workflow for
end-to-end skill gap analysis, connecting all the agent nodes with proper
retry policies and error handling.
"""

import logging
from typing import Dict, Any, Callable, Union

from langgraph.graph import StateGraph
from langgraph.constants import START, END

from graph.nodes import (
    ingest_cv,
    parse_cv,
    analyze_skills,
    market_scan,
    synthesize_report_node
)
from graph.state import GraphState

logger = logging.getLogger(__name__)


def _should_retry(state) -> bool:
    """
    Determine if a node should be retried based on state.
    
    Args:
        state: Current graph state (dict or GraphState object)
        
    Returns:
        True if retry is needed, False otherwise
    """
    if isinstance(state, dict):
        return "error" in state and state.get("retry_count", 0) < 2
    else:
        # GraphState object
        return hasattr(state, 'error') and state.error is not None and getattr(state, 'retry_count', 0) < 2


def _should_continue(state) -> str:
    """
    Determine the next node to execute based on state.
    
    Args:
        state: Current graph state (dict or GraphState object)
        
    Returns:
        Name of next node or END
    """
    if isinstance(state, dict):
        has_error = "error" in state
    else:
        # GraphState object
        has_error = hasattr(state, 'error') and state.error is not None
        
    if has_error:
        return END
    return "continue"


def _add_retry_wrapper(node_func: Callable, provider_choice: str = "auto") -> Callable:
    """
    Wrap a node function with retry logic and format conversion.
    
    Args:
        node_func: Original node function that expects GraphState and returns GraphState
        provider_choice: LLM provider choice to pass to the node
        
    Returns:
        Wrapped function that works with dict state for LangGraph
    """
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both dict and GraphState inputs
        if isinstance(state, dict):
            retry_count = state.get("retry_count", 0)
            state_dict = state
        else:
            # If it's a GraphState object, convert to dict and get retry count
            retry_count = getattr(state, 'retry_count', 0)
            state_dict = state.model_dump() if hasattr(state, 'model_dump') else dict(state)
        
        try:
            # Convert dict to GraphState for node function
            graph_state = GraphState(**state_dict)
            
            # Call the node function with provider_choice
            result_state = node_func(graph_state, provider_choice=provider_choice)
            
            # Convert GraphState back to dict
            result_dict = result_state.model_dump()
            
            # Reset retry count on success
            if "error" not in result_dict or result_dict.get("error") is None:
                result_dict["retry_count"] = 0
            
            return result_dict
            
        except Exception as e:
            error_msg = f"Node {node_func.__name__} failed (attempt {retry_count + 1}): {str(e)}"
            logger.error(error_msg)
            
            return {
                **state_dict,
                "error": str(e),
                "retry_count": retry_count + 1,
                "logs": state_dict.get("logs", []) + [error_msg]
            }
    
    wrapper.__name__ = f"{node_func.__name__}_with_retry"
    return wrapper


def build_graph(provider_choice: str = "auto", checkpointer_path: str = ":memory:"):
    """
    Build a LangGraph workflow for skill gap analysis.
    
    This creates a graph that coordinates the following pipeline:
    1. ingest_cv: Validate and ingest CV text
    2. parse_cv: Parse CV into structured format
    3. analyze_skills: Build comprehensive skill profile
    4. market_scan: Gather market intelligence
    5. synthesize_report: Generate final report
    
    Args:
        provider_choice: LLM provider to use ("auto", "gemini", "mistral")
        checkpointer_path: Path for SQLite checkpointer (":memory:" for in-memory)
        
    Returns:
        Compiled LangGraph that can be invoked with state
        
    Example:
        >>> graph = build_graph("auto")
        >>> result = graph.invoke({
        ...     "cv_text": "John Doe\\nSoftware Engineer...",
        ...     "target_role": "Senior Python Developer",
        ...     "market_region": "Global",
        ...     "provider": "auto"
        ... })
        >>> print(result["report_md"])
    """
    # Create state graph with GraphState schema
    workflow = StateGraph(GraphState)
    
    # Add nodes with retry wrappers that include provider_choice
    workflow.add_node("ingest_cv", _add_retry_wrapper(ingest_cv, provider_choice))
    workflow.add_node("parse_cv", _add_retry_wrapper(parse_cv, provider_choice))
    workflow.add_node("analyze_skills", _add_retry_wrapper(analyze_skills, provider_choice))
    workflow.add_node("market_scan", _add_retry_wrapper(market_scan, provider_choice))
    workflow.add_node("synthesize_report", _add_retry_wrapper(synthesize_report_node, provider_choice))
    
    # Define the pipeline flow with conditional edges only
    workflow.add_edge(START, "ingest_cv")
    
    # Add conditional edges for error handling and flow control
    workflow.add_conditional_edges(
        "ingest_cv",
        _should_continue,
        {
            "continue": "parse_cv",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "parse_cv",
        _should_continue,
        {
            "continue": "analyze_skills",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_skills", 
        _should_continue,
        {
            "continue": "market_scan",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "market_scan",
        _should_continue,
        {
            "continue": "synthesize_report",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "synthesize_report",
        _should_continue,
        {
            "continue": END,
            END: END
        }
    )
    
    # Compile the graph without checkpointing for now
    compiled_graph = workflow.compile()
    
    logger.info(f"Skill gap analysis graph built successfully with {provider_choice} provider")
    
    return compiled_graph


def create_default_state(
    cv_text: str,
    target_role: str = "Senior AI Engineer",
    market_region: str = "Global",
    lang: str = "en",
    provider: str = "auto"
) -> Dict[str, Any]:
    """
    Create a default state dictionary for graph execution.
    
    Args:
        cv_text: Raw CV text to analyze
        target_role: Target job role for market analysis
        market_region: Geographic region for market analysis  
        lang: Language for report generation
        provider: LLM provider choice for analysis
        
    Returns:
        State dictionary ready for graph execution
    """
    return {
        "cv_text": cv_text,
        "target_role": target_role,
        "market_region": market_region,
        "lang": lang,
        "provider": provider,
        "logs": []
    }
