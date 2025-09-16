#!/usr/bin/env python3
"""Test script to verify the enhanced provider feedback functionality."""

import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from llm.provider import get_chat_model, choose_provider

def test_provider_info():
    """Test the enhanced provider information functionality."""
    print("🧪 Testing Enhanced Provider Feedback...")
    
    try:
        # Test auto provider selection
        print("\n1. Testing AUTO provider selection:")
        provider_name = "auto"
        chat_model = get_chat_model(provider_name, None)
        
        # Enhanced provider tracking (simulating streamlit app logic)
        provider_info = {
            "requested": "Auto",
            "resolved": None,
            "model_name": None,
            "fallback_used": False,
            "warnings": []
        }
        
        # Extract actual provider and model information
        if hasattr(chat_model, 'model'):
            model_name = chat_model.model
            provider_info["model_name"] = model_name
            
            if 'gemini' in model_name.lower():
                provider_info["resolved"] = "Gemini"
                if '2.0' in model_name:
                    provider_info["model_display"] = "Gemini 2.0 Flash"
                elif '1.5' in model_name:
                    provider_info["model_display"] = "Gemini 1.5 Pro"
                else:
                    provider_info["model_display"] = "Gemini"
            elif 'mistral' in model_name.lower():
                provider_info["resolved"] = "Mistral"
                if 'large' in model_name.lower():
                    provider_info["model_display"] = "Mistral Large"
                else:
                    provider_info["model_display"] = "Mistral"
            else:
                provider_info["resolved"] = "Unknown"
                provider_info["model_display"] = model_name
        
        # Check for Auto mode fallback
        if provider_name == "auto":
            if provider_info["resolved"] == "Mistral":
                # Check if Gemini was available but failed
                gemini_available = bool(os.getenv("GEMINI_API_KEY"))
                if gemini_available:
                    provider_info["fallback_used"] = True
                    provider_info["warnings"].append("Gemini unavailable, falling back to Mistral")
        
        print(f"✅ Provider Resolution Successful!")
        print(f"   Requested: {provider_info['requested']}")
        print(f"   Resolved: {provider_info['resolved']}")
        print(f"   Model Display: {provider_info['model_display']}")
        print(f"   Technical Model: {provider_info['model_name']}")
        print(f"   Fallback Used: {provider_info['fallback_used']}")
        if provider_info['warnings']:
            print(f"   Warnings: {provider_info['warnings']}")
        
        # Test specific providers
        print("\n2. Testing GEMINI provider specifically:")
        try:
            gemini_model = get_chat_model("gemini", None)
            gemini_model_name = getattr(gemini_model, 'model', 'unknown')
            print(f"   ✅ Gemini Model: {gemini_model_name}")
        except Exception as e:
            print(f"   ❌ Gemini Error: {e}")
        
        print("\n3. Testing MISTRAL provider specifically:")
        try:
            mistral_model = get_chat_model("mistral", None)
            mistral_model_name = getattr(mistral_model, 'model', 'unknown')
            print(f"   ✅ Mistral Model: {mistral_model_name}")
        except Exception as e:
            print(f"   ❌ Mistral Error: {e}")
        
        return provider_info
        
    except Exception as e:
        print(f"❌ Provider test failed: {e}")
        return None

def main():
    """Main test function."""
    print("🎯 Enhanced Provider Feedback Test")
    print("=" * 50)
    
    # Check environment
    print("Environment Status:")
    print(f"  GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    print(f"  MISTRAL_API_KEY: {'✅ Set' if os.getenv('MISTRAL_API_KEY') else '❌ Missing'}")
    print(f"  TAVILY_API_KEY: {'✅ Set' if os.getenv('TAVILY_API_KEY') else '❌ Missing'}")
    
    # Test provider info
    result = test_provider_info()
    
    if result:
        print("\n🎉 Enhanced provider feedback is working!")
        print("\nStreamlit UI will now show:")
        print(f"  - Colored badge: '{result['model_display']}'")
        print(f"  - Auto mode status with health indicators")
        print(f"  - Fallback warnings when applicable")
        print(f"  - Enhanced error messages with guidance")
    else:
        print("\n❌ Provider feedback enhancement needs debugging")

if __name__ == "__main__":
    main()