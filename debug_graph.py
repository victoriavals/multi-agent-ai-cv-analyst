#!/usr/bin/env python3
"""
Simple test to debug the GraphState issue.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph.builder import build_graph
from graph.state import GraphState

def test_graph():
    """Test the graph with minimal state."""
    print("Building graph...")
    graph = build_graph()
    print("✅ Graph built successfully")
    
    # Create minimal test state
    test_state = {
        "file_path": "sample/sample_cv.txt",
        "target_role": "Test Role",
        "market_region": "Global", 
        "lang": "en",
        "output_path": "test_output.md",
        "logs": []
    }
    
    print("Testing with dict state...")
    print(f"State type: {type(test_state)}")
    print(f"Keys: {list(test_state.keys())}")
    
    try:
        result = graph.invoke(test_state)
        print("✅ Graph execution completed")
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graph()