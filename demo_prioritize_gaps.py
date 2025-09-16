"""
Demonstration of the prioritize_gaps function for skill gap analysis.
"""

import numpy as np
from src.utils.eval import prioritize_gaps

def create_realistic_embedding_func():
    """
    Create a more realistic embedding function that simulates 
    actual semantic relationships between skills.
    """
    # Predefined embeddings that capture semantic relationships
    embeddings = {
        # Programming Languages
        "Python": [0.9, 0.1, 0.8, 0.2, 0.3],
        "JavaScript": [0.8, 0.2, 0.1, 0.9, 0.1],
        "Java": [0.8, 0.3, 0.6, 0.4, 0.2],
        "C++": [0.7, 0.2, 0.9, 0.1, 0.3],
        "SQL": [0.2, 0.9, 0.3, 0.1, 0.7],
        
        # Data Science & ML
        "Machine Learning": [0.8, 0.3, 0.7, 0.1, 0.8],
        "Deep Learning": [0.9, 0.2, 0.8, 0.1, 0.9],
        "Data Science": [0.7, 0.8, 0.5, 0.2, 0.8],
        "Statistics": [0.3, 0.9, 0.2, 0.1, 0.9],
        "TensorFlow": [0.8, 0.1, 0.9, 0.1, 0.7],
        "PyTorch": [0.9, 0.1, 0.8, 0.1, 0.7],
        "Pandas": [0.8, 0.7, 0.4, 0.2, 0.6],
        
        # Web Development
        "React": [0.7, 0.1, 0.2, 0.9, 0.2],
        "Node.js": [0.6, 0.2, 0.3, 0.8, 0.1],
        "HTML": [0.2, 0.1, 0.1, 0.9, 0.1],
        "CSS": [0.1, 0.1, 0.1, 0.9, 0.2],
        
        # Cloud & DevOps
        "AWS": [0.3, 0.2, 0.4, 0.3, 0.4],
        "Docker": [0.4, 0.1, 0.6, 0.2, 0.3],
        "Kubernetes": [0.4, 0.1, 0.7, 0.2, 0.3],
        "Git": [0.5, 0.2, 0.4, 0.4, 0.2],
        
        # Business Skills
        "Project Management": [0.1, 0.3, 0.2, 0.2, 0.3],
        "Communication": [0.1, 0.2, 0.1, 0.3, 0.2],
        "Leadership": [0.1, 0.2, 0.2, 0.2, 0.3],
    }
    
    def embedding_func(skill: str) -> list:
        # Return predefined embedding or generate a random one for unknown skills
        if skill in embeddings:
            return embeddings[skill]
        else:
            # Generate a deterministic random embedding based on skill name
            np.random.seed(hash(skill) % 2**32)
            return np.random.rand(5).tolist()
    
    return embedding_func

def demo_prioritize_gaps():
    """Demonstrate the prioritize_gaps function with different scenarios."""
    
    embedding_func = create_realistic_embedding_func()
    
    print("=== Skill Gap Prioritization Demo ===\n")
    
    # Scenario 1: Junior Python Developer looking at Data Science roles
    print("🐍 Scenario 1: Junior Python Developer → Data Science Roles")
    print("-" * 60)
    
    candidate_skills = ["Python", "SQL", "Git"]
    market_skills = [
        "Python", "Machine Learning", "Statistics", "Pandas",
        "Data Science", "TensorFlow", "Machine Learning",  # ML appears twice
        "SQL", "AWS", "Statistics"  # Statistics appears twice
    ]
    
    gaps = prioritize_gaps(candidate_skills, market_skills, embedding_func)
    
    print(f"Candidate Skills: {', '.join(candidate_skills)}")
    print(f"Market Demand: {', '.join(set(market_skills))}")
    print("\nPrioritized Skill Gaps:")
    
    for i, gap in enumerate(gaps[:7], 1):  # Show top 7
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[gap['priority']]
        print(f"{i:2d}. {priority_emoji} {gap['skill']:<20} Score: {gap['score']:<5} Priority: {gap['priority']}")
    
    print("\n" + "="*80 + "\n")
    
    # Scenario 2: Frontend Developer looking at Full-Stack roles  
    print("🌐 Scenario 2: Frontend Developer → Full-Stack Roles")
    print("-" * 60)
    
    candidate_skills = ["JavaScript", "React", "HTML", "CSS"]
    market_skills = [
        "JavaScript", "React", "Node.js", "Python", "SQL",
        "Docker", "AWS", "Git", "React", "Node.js",  # React and Node.js appear twice
        "Machine Learning", "TensorFlow"
    ]
    
    gaps = prioritize_gaps(candidate_skills, market_skills, embedding_func)
    
    print(f"Candidate Skills: {', '.join(candidate_skills)}")
    print(f"Market Demand: {', '.join(set(market_skills))}")
    print("\nPrioritized Skill Gaps:")
    
    for i, gap in enumerate(gaps[:7], 1):
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[gap['priority']]
        print(f"{i:2d}. {priority_emoji} {gap['skill']:<20} Score: {gap['score']:<5} Priority: {gap['priority']}")
    
    print("\n" + "="*80 + "\n")
    
    # Scenario 3: Complete career change (no relevant skills)
    print("🚀 Scenario 3: Career Change → AI/ML Engineer")
    print("-" * 60)
    
    candidate_skills = ["Project Management", "Communication"]  # Non-technical background
    market_skills = [
        "Python", "Machine Learning", "Deep Learning", "TensorFlow",
        "PyTorch", "Statistics", "Data Science", "SQL", "AWS",
        "Python", "Machine Learning", "Deep Learning"  # Some skills appear multiple times
    ]
    
    gaps = prioritize_gaps(candidate_skills, market_skills, embedding_func)
    
    print(f"Candidate Skills: {', '.join(candidate_skills)}")
    print(f"Market Demand: {', '.join(set(market_skills))}")
    print("\nPrioritized Skill Gaps:")
    
    for i, gap in enumerate(gaps, 1):
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[gap['priority']]
        print(f"{i:2d}. {priority_emoji} {gap['skill']:<20} Score: {gap['score']:<5} Priority: {gap['priority']}")
    
    print("\n" + "="*80 + "\n")
    
    # Scenario 4: Frequency weighting demonstration
    print("📊 Scenario 4: Frequency Weighting Effect")
    print("-" * 60)
    
    candidate_skills = ["Java"]
    market_skills_low_freq = ["Python", "JavaScript", "React"]  # Each appears once
    market_skills_high_freq = ["Python", "Python", "Python", "JavaScript", "React"]  # Python appears 3x
    
    gaps_low = prioritize_gaps(candidate_skills, market_skills_low_freq, embedding_func)
    gaps_high = prioritize_gaps(candidate_skills, market_skills_high_freq, embedding_func)
    
    print("Market Skills (Low Frequency):", market_skills_low_freq)
    print("Gaps (Low Frequency):")
    for gap in gaps_low:
        print(f"  {gap['skill']:<15} Score: {gap['score']}")
    
    print(f"\nMarket Skills (High Frequency): {market_skills_high_freq}")
    print("Gaps (High Frequency):")
    for gap in gaps_high:
        print(f"  {gap['skill']:<15} Score: {gap['score']}")
    
    python_low = next(g for g in gaps_low if g['skill'] == 'Python')['score']
    python_high = next(g for g in gaps_high if g['skill'] == 'Python')['score']
    
    print(f"\n📈 Python score increased from {python_low} to {python_high} due to frequency!")
    
    print("\n" + "="*80 + "\n")
    
    # Scenario 5: Edge case - candidate already has all market skills
    print("✅ Scenario 5: Overqualified Candidate")
    print("-" * 60)
    
    candidate_skills = ["Python", "Machine Learning", "TensorFlow", "AWS", "Docker"]
    market_skills = ["Python", "Machine Learning", "AWS"]  # Subset of candidate skills
    
    gaps = prioritize_gaps(candidate_skills, market_skills, embedding_func)
    
    print(f"Candidate Skills: {', '.join(candidate_skills)}")
    print(f"Market Demand: {', '.join(market_skills)}")
    print("\nGap Analysis (should show low scores):")
    
    for gap in gaps:
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[gap['priority']]
        print(f"  {priority_emoji} {gap['skill']:<20} Score: {gap['score']:<5} Priority: {gap['priority']}")
    
    print("\n💡 All gaps have low scores because candidate already possesses similar skills!")

if __name__ == "__main__":
    demo_prioritize_gaps()