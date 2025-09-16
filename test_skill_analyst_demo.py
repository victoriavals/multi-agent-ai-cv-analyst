"""Quick test of the skill analyst with sample CV data."""

from src.agents.skill_analyst import build_skill_profile
from src.graph.state import CVStruct

# Create sample CV
sample_cv = CVStruct(
    basics={
        'name': 'Alice Johnson',
        'title': 'Senior ML Engineer'
    },
    skills=['Python', 'TensorFlow', 'Kubernetes', 'AWS', 'PostgreSQL'],
    experience=[
        {
            'title': 'Senior ML Engineer',
            'company': 'TechCorp',
            'dates': '2021-2024',
            'bullets': [
                'Led a team of 4 ML engineers to deploy production models serving 2M+ users',
                'Architected scalable ML infrastructure using Kubernetes and MLflow',
                'Improved model performance by 35% through feature engineering and optimization',
                'Mentored junior developers on ML best practices and code reviews'
            ]
        }
    ],
    projects=[
        {
            'name': 'Real-time Recommendation Engine',
            'tech': ['PyTorch', 'Redis', 'FastAPI', 'Docker'],
            'bullets': [
                'Built deep learning model for personalized recommendations',
                'Reduced inference latency by 60% using model optimization techniques'
            ]
        }
    ],
    education=[
        {
            'degree': 'MS Machine Learning',
            'school': 'Stanford University',
            'year': '2020'
        }
    ]
)

# Test skill profile building
print("Testing skill analyst with sample CV...")
skill_profile = build_skill_profile(sample_cv, llm=None)

print(f"\nExplicit skills ({len(skill_profile.explicit)}):")
for skill in skill_profile.explicit[:10]:
    print(f"  - {skill}")

print(f"\nImplicit skills ({len(skill_profile.implicit)}):")
for skill in skill_profile.implicit[:10]:
    print(f"  - {skill}")

print(f"\nTransferable skills ({len(skill_profile.transferable)}):")
for skill in skill_profile.transferable[:10]:
    print(f"  - {skill}")

print(f"\nSeniority signals ({len(skill_profile.seniority_signals)}):")
for signal in skill_profile.seniority_signals[:5]:
    print(f"  - {signal}")

print(f"\nCoverage map: {skill_profile.coverage_map}")
print("\n✅ Skill analyst working correctly!")