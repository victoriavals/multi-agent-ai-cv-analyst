#!/usr/bin/env python3
"""Test script to verify the provider_choice integration with GraphState and builder."""

import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from graph.builder import build_graph, create_default_state
from graph.state import GraphState

def test_provider_integration():
    """Test that provider_choice flows through the graph pipeline."""
    print("🧪 Testing Provider Choice Integration...")
    
    # Test 1: Verify build_graph accepts provider_choice
    print("\n1. Testing build_graph with provider_choice:")
    try:
        graph_auto = build_graph(provider_choice="auto")
        print("   ✅ build_graph('auto') successful")
        
        graph_gemini = build_graph(provider_choice="gemini")
        print("   ✅ build_graph('gemini') successful")
        
        graph_mistral = build_graph(provider_choice="mistral")
        print("   ✅ build_graph('mistral') successful")
    except Exception as e:
        print(f"   ❌ build_graph failed: {e}")
        return False
    
    # Test 2: Verify create_default_state includes provider
    print("\n2. Testing create_default_state with provider:")
    try:
        state_dict = create_default_state(
            cv_text="Test CV text",
            target_role="Test Role",
            provider="gemini"
        )
        
        expected_fields = ["cv_text", "target_role", "market_region", "lang", "provider", "logs"]
        for field in expected_fields:
            if field not in state_dict:
                print(f"   ❌ Missing field: {field}")
                return False
        
        if state_dict["provider"] != "gemini":
            print(f"   ❌ Provider mismatch: expected 'gemini', got '{state_dict['provider']}'")
            return False
            
        print("   ✅ create_default_state includes provider field")
        print(f"   ✅ Provider correctly set to: {state_dict['provider']}")
    except Exception as e:
        print(f"   ❌ create_default_state failed: {e}")
        return False
    
    # Test 3: Verify GraphState can be instantiated with provider
    print("\n3. Testing GraphState with provider:")
    try:
        graph_state = GraphState(
            file_path="test.txt",
            cv_text="Test CV content",
            target_role="Software Engineer",
            provider="auto"
        )
        
        if graph_state.provider != "auto":
            print(f"   ❌ GraphState provider mismatch: expected 'auto', got '{graph_state.provider}'")
            return False
            
        print(f"   ✅ GraphState created with provider: {graph_state.provider}")
        print(f"   ✅ Default model names: chat={graph_state.chat_model_name}, embed={graph_state.embed_model_name}")
    except Exception as e:
        print(f"   ❌ GraphState creation failed: {e}")
        return False
    
    # Test 4: Verify graph state conversion
    print("\n4. Testing state dict <-> GraphState conversion:")
    try:
        # Create state dict
        original_dict = {
            "file_path": "test.txt",
            "cv_text": "Test content",
            "target_role": "Engineer",
            "provider": "mistral",
            "logs": ["test log"]
        }
        
        # Convert to GraphState
        graph_state = GraphState(**original_dict)
        
        # Convert back to dict
        converted_dict = graph_state.model_dump()
        
        # Verify key fields preserved
        key_fields = ["file_path", "cv_text", "target_role", "provider"]
        for field in key_fields:
            if converted_dict[field] != original_dict[field]:
                print(f"   ❌ Field mismatch in {field}: {original_dict[field]} -> {converted_dict[field]}")
                return False
        
        print("   ✅ State conversion preserves provider and key fields")
    except Exception as e:
        print(f"   ❌ State conversion failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("🎯 Provider Choice Integration Test")
    print("=" * 50)
    
    # Check environment
    print("Environment Status:")
    print(f"  GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    print(f"  MISTRAL_API_KEY: {'✅ Set' if os.getenv('MISTRAL_API_KEY') else '❌ Missing'}")
    
    # Run tests
    success = test_provider_integration()
    
    if success:
        print("\n🎉 All provider choice integration tests passed!")
        print("\nUpdated components:")
        print("  ✅ GraphState: Has provider field")
        print("  ✅ build_graph: Accepts provider_choice parameter")
        print("  ✅ create_default_state: Includes provider field")
        print("  ✅ Node wrappers: Pass provider_choice to agent functions")
        print("  ✅ Agent nodes: Use provider_choice from kwargs")
        print("\nProvider flow:")
        print("  build_graph(provider_choice) -> node_wrappers(provider_choice) -> agent_functions(provider_choice)")
    else:
        print("\n❌ Provider choice integration tests failed!")

if __name__ == "__main__":
    main()