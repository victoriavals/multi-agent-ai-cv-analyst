"""
Utility functions for LangGraph execution and monitoring.

This module provides helper functions for graph execution, logging,
and error handling in the skill gap analysis pipeline.
"""

import logging
import time
from typing import Dict, Any, Iterator, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def log_execution_start(state: Dict[str, Any]) -> None:
    """
    Log the start of graph execution.
    
    Args:
        state: Initial graph state
    """
    target_role = state.get("target_role", "Unknown Role")
    market_region = state.get("market_region", "Unknown Region")
    cv_length = len(state.get("cv_text", ""))
    
    logger.info(f"🚀 Starting skill gap analysis")
    logger.info(f"   Target Role: {target_role}")
    logger.info(f"   Market Region: {market_region}")
    logger.info(f"   CV Length: {cv_length} characters")


def log_execution_end(state: Dict[str, Any], duration: float) -> None:
    """
    Log the end of graph execution.
    
    Args:
        state: Final graph state
        duration: Execution duration in seconds
    """
    if "error" in state:
        logger.error(f"❌ Analysis failed after {duration:.2f}s: {state['error']}")
    else:
        report_length = len(state.get("report_md", ""))
        logger.info(f"✅ Analysis completed successfully in {duration:.2f}s")
        logger.info(f"   Report Generated: {report_length} characters")
        
        # Log execution summary
        logs = state.get("logs", [])
        if logs:
            logger.info("📊 Execution Summary:")
            for log_entry in logs:
                logger.info(f"   {log_entry}")


def stream_graph_execution(
    graph,
    initial_state: Dict[str, Any],
    thread_id: str = "default"
) -> Iterator[Dict[str, Any]]:
    """
    Execute graph with streaming updates and comprehensive logging.
    
    Args:
        graph: Compiled LangGraph instance
        initial_state: Initial state for execution
        thread_id: Thread ID for checkpointing
        
    Yields:
        State updates during graph execution
    """
    start_time = time.time()
    log_execution_start(initial_state)
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        for chunk in graph.stream(initial_state, config=config):
            # Extract the actual state from the chunk
            if isinstance(chunk, dict):
                for node_name, node_state in chunk.items():
                    if isinstance(node_state, dict):
                        # Log node completion
                        if "logs" in node_state:
                            latest_logs = node_state["logs"]
                            if latest_logs:
                                latest_log = latest_logs[-1]
                                logger.info(f"🔄 {node_name}: {latest_log}")
                        
                        yield {
                            "node": node_name,
                            "state": node_state,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Early termination on error
                        if "error" in node_state:
                            logger.error(f"🛑 {node_name} failed: {node_state['error']}")
                            break
            else:
                yield chunk
                
    except Exception as e:
        error_msg = f"Graph execution failed: {str(e)}"
        logger.error(error_msg)
        yield {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    
    finally:
        duration = time.time() - start_time
        # Try to get final state for logging
        try:
            final_state = graph.get_state(config).values
            log_execution_end(final_state, duration)
        except:
            logger.info(f"Graph execution completed in {duration:.2f}s")


def validate_graph_state(state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate that a graph state has required fields.
    
    Args:
        state: Graph state to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["cv_text", "target_role", "market_region"]
    
    for field in required_fields:
        if field not in state:
            return False, f"Missing required field: {field}"
        
        if not state[field] or (isinstance(state[field], str) and not state[field].strip()):
            return False, f"Empty or invalid value for field: {field}"
    
    # Additional validations
    if len(state["cv_text"]) < 50:
        return False, "CV text too short (minimum 50 characters required)"
    
    return True, None


def create_execution_summary(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a summary of graph execution results.
    
    Args:
        final_state: Final state after graph execution
        
    Returns:
        Execution summary dictionary
    """
    summary = {
        "success": "error" not in final_state,
        "timestamp": datetime.now().isoformat(),
    }
    
    if "error" in final_state:
        summary["error"] = final_state["error"]
        summary["retry_count"] = final_state.get("retry_count", 0)
    else:
        summary["cv_parsed"] = final_state.get("cv_struct") is not None
        summary["skills_analyzed"] = final_state.get("skill_profile") is not None
        summary["market_scanned"] = final_state.get("market_summary") is not None
        summary["report_generated"] = final_state.get("report_md") is not None
        
        if final_state.get("skill_profile"):
            profile = final_state["skill_profile"]
            summary["skills_found"] = {
                "explicit": len(profile.get("explicit", [])),
                "implicit": len(profile.get("implicit", [])),
                "transferable": len(profile.get("transferable", [])),
                "seniority_signals": len(profile.get("seniority_signals", []))
            }
        
        if final_state.get("market_summary"):
            market = final_state["market_summary"]
            summary["market_skills"] = {
                "in_demand": len(market.get("in_demand_skills", [])),
                "tools": len(market.get("common_tools", [])),
                "frameworks": len(market.get("frameworks", [])),
                "nice_to_have": len(market.get("nice_to_have", []))
            }
        
        if final_state.get("report_md"):
            summary["report_length"] = len(final_state["report_md"])
    
    summary["logs"] = final_state.get("logs", [])
    
    return summary