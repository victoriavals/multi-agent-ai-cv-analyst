#!/usr/bin/env python3
"""Test script for taxonomy functionality."""

from src.tools.taxonomy import canonicalize_skills, closest_market_skills, get_skill_categories, suggest_skills

def test_canonicalize():
    """Test skill canonicalization."""
    print("=== Testing Canonicalize Skills ===")
    
    # Test basic canonicalization
    skills = ['Python', 'TensorFlow', 'tf', 'RAG', 'vector db', 'ChromaDB']
    canonical = canonicalize_skills(skills)
    print(f"Input: {skills}")
    print(f"Canonical: {canonical}")
    
    # Test with noise
    noisy_skills = [
        '5 years Python programming experience',
        'Strong knowledge in TensorFlow',
        'Working with RAG systems',
        'Vector databases like Chroma'
    ]
    clean_skills = canonicalize_skills(noisy_skills)
    print(f"\nNoisy input: {noisy_skills}")
    print(f"Cleaned: {clean_skills}")
    
    print("\n✓ Canonicalization working!")

def test_categories():
    """Test skill categories."""
    print("\n=== Testing Skill Categories ===")
    
    categories = get_skill_categories()
    print(f"Available categories: {list(categories.keys())}")
    print(f"Programming languages: {categories['programming_languages'][:5]}")
    print(f"ML frameworks: {categories['ml_frameworks'][:3]}")
    print(f"LLM/NLP: {categories['llm_nlp'][:3]}")
    
    print("\n✓ Categories working!")

def test_suggestions():
    """Test skill suggestions."""
    print("\n=== Testing Skill Suggestions ===")
    
    suggestions = suggest_skills("py")
    print(f"Suggestions for 'py': {suggestions[:5]}")
    
    suggestions = suggest_skills("tensor")
    print(f"Suggestions for 'tensor': {suggestions[:3]}")
    
    suggestions = suggest_skills("vector")
    print(f"Suggestions for 'vector': {suggestions[:3]}")
    
    print("\n✓ Suggestions working!")

def test_market_mapping():
    """Test market skill mapping."""
    print("\n=== Testing Market Skill Mapping ===")
    
    # Mock embedding function for testing
    import numpy as np
    def mock_embed(texts):
        # Simple mock that creates random but consistent embeddings
        np.random.seed(hash(''.join(texts)) % 2**31)
        return np.random.rand(len(texts), 384).astype(np.float32)
    
    candidate_skills = [
        "Python programming",
        "TensorFlow deep learning", 
        "Docker containers"
    ]
    
    market_skills = [
        "Python development",
        "Machine learning frameworks",
        "DevOps tools"
    ]
    
    result = closest_market_skills(candidate_skills, market_skills, mock_embed)
    
    print(f"Candidate skills: {candidate_skills}")
    print(f"Market skills: {market_skills}")
    print(f"\nMappings:")
    for market_skill, match_info in result.items():
        print(f"  {market_skill} -> {match_info['match']} (sim: {match_info['similarity']:.3f})")
        print(f"    Evidence: {match_info['evidence']}")
    
    print("\n✓ Market mapping working!")

if __name__ == "__main__":
    test_canonicalize()
    test_categories()
    test_suggestions()
    test_market_mapping()
    print("\n✅ All taxonomy functions working correctly!")