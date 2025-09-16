"""
Evaluation utilities for skill gap analysis.

This module provides functions for evaluating and prioritizing skill gaps
based on market demand and candidate skills using embedding similarity.
"""

from typing import List, Dict, Callable, Any
from collections import Counter
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector as numpy array
        vec2: Second vector as numpy array
        
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    # Handle edge cases
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def classify_priority(score: float, high_threshold: float = 0.7, medium_threshold: float = 0.4) -> str:
    """
    Classify priority level based on score.
    
    Args:
        score: Similarity/relevance score (0.0 to 1.0)
        high_threshold: Threshold for high priority (default 0.7)
        medium_threshold: Threshold for medium priority (default 0.4)
        
    Returns:
        str: Priority level ("high", "medium", or "low")
    """
    if score >= high_threshold:
        return "high"
    elif score >= medium_threshold:
        return "medium"
    else:
        return "low"


def prioritize_gaps(
    candidate_skills: List[str], 
    market_skills: List[str], 
    embed_f: Callable[[str], Any]
) -> List[Dict[str, Any]]:
    """
    Prioritize skill gaps based on market demand and candidate skills.
    
    Uses cosine similarity to find gaps between candidate skills and market skills,
    with frequency weighting for market duplicates. Returns deterministic results
    through consistent sorting.
    
    Args:
        candidate_skills: List of skills the candidate currently has
        market_skills: List of skills in demand in the market (may contain duplicates)
        embed_f: Function that takes a string and returns an embedding vector
        
    Returns:
        List[Dict]: Sorted list of skill gaps with keys:
            - skill: str - The skill name
            - priority: str - Priority level ("high", "medium", "low")
            - score: float - Relevance score (0.0 to 1.0)
            
    Example:
        >>> candidate = ["Python", "SQL"]
        >>> market = ["Machine Learning", "Python", "TensorFlow", "Machine Learning"]
        >>> gaps = prioritize_gaps(candidate, market, embedding_function)
        >>> gaps[0]
        {'skill': 'Machine Learning', 'priority': 'high', 'score': 0.85}
    """
    if not market_skills:
        return []
    
    if not candidate_skills:
        # If candidate has no skills, all market skills are gaps
        # Use frequency weighting and return sorted by skill name for determinism
        market_counts = Counter(market_skills)
        unique_market_skills = list(set(market_skills))
        
        gaps = []
        for skill in unique_market_skills:
            frequency_weight = market_counts[skill] / len(market_skills)
            # Base score on frequency when no candidate skills to compare
            score = min(frequency_weight * 2, 1.0)  # Scale frequency, cap at 1.0
            
            gaps.append({
                'skill': skill,
                'priority': classify_priority(score),
                'score': round(score, 3)
            })
        
        # Sort by score (descending) then by skill name for determinism
        return sorted(gaps, key=lambda x: (-x['score'], x['skill']))
    
    # Get embeddings for candidate skills
    try:
        candidate_embeddings = [np.array(embed_f(skill)) for skill in candidate_skills]
    except Exception as e:
        raise ValueError(f"Failed to get embeddings for candidate skills: {e}")
    
    # Count market skill frequencies for weighting
    market_counts = Counter(market_skills)
    unique_market_skills = list(set(market_skills))
    
    gaps = []
    
    for market_skill in unique_market_skills:
        try:
            market_embedding = np.array(embed_f(market_skill))
        except Exception as e:
            # Skip skills that can't be embedded
            continue
            
        # Calculate maximum similarity to any candidate skill
        max_similarity = 0.0
        for candidate_embedding in candidate_embeddings:
            similarity = cosine_similarity(candidate_embedding, market_embedding)
            max_similarity = max(max_similarity, similarity)
        
        # Calculate gap score (inverse of similarity)
        gap_score = 1.0 - max_similarity
        
        # Apply frequency weighting
        frequency_weight = market_counts[market_skill] / len(market_skills)
        weighted_score = gap_score * (1.0 + frequency_weight)  # Boost by frequency
        
        # Normalize to [0, 1] range
        final_score = min(weighted_score, 1.0)
        
        gaps.append({
            'skill': market_skill,
            'priority': classify_priority(final_score),
            'score': round(final_score, 3)
        })
    
    # Sort by score (descending) then by skill name for deterministic output
    return sorted(gaps, key=lambda x: (-x['score'], x['skill']))


# Legacy function for backwards compatibility
def evaluate(data):
    """Legacy evaluation function - placeholder for backwards compatibility."""
    pass
