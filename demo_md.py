"""
Demonstration of Markdown rendering utilities for skill gap analysis reports.
"""

from src.utils.md import (
    render_report_header,
    render_strengths,
    render_gaps,
    render_plan,
    render_skills_table,
    render_section,
    render_bullet_list,
    render_numbered_list,
    render_code_block,
    render_quote,
    render_horizontal_rule
)

def demo_markdown_utilities():
    """Demonstrate Markdown utilities for generating reports."""
    
    print("=== Markdown Utilities Demo ===\n")
    
    # 1. Report Header
    print("1. Report Header:")
    header = render_report_header("AI Engineer", "San Francisco")
    print(header)
    
    # 2. Strengths Section
    print("2. Strengths Section:")
    strengths = [
        "Strong Python programming skills",
        "Experience with machine learning frameworks (TensorFlow, PyTorch)",
        "Solid understanding of data structures and algorithms",
        "Familiarity with cloud platforms (AWS, GCP)"
    ]
    strengths_md = render_strengths(strengths)
    print(strengths_md)
    
    # 3. Skill Gaps Section  
    print("3. Skill Gaps Section:")
    gaps = [
        "Limited experience with LLM fine-tuning techniques",
        "No hands-on experience with vector databases (ChromaDB, Pinecone)",
        "Unfamiliar with RAG (Retrieval-Augmented Generation) systems",
        "Lacks MLOps experience (MLflow, model deployment pipelines)",
        "No experience with prompt engineering best practices"
    ]
    gaps_md = render_gaps(gaps)
    print(gaps_md)
    
    # 4. Development Plan
    print("4. Development Plan:")
    plan = {
        "30": [
            "Complete online course on RAG fundamentals",
            "Set up local ChromaDB instance and experiment with embeddings",
            "Build simple Q&A system using OpenAI API",
            "Learn prompt engineering basics and best practices"
        ],
        "60": [
            "Develop RAG-based chatbot prototype with document retrieval",
            "Integrate vector database with LLM for enhanced responses",
            "Experiment with different embedding models and chunk strategies",
            "Set up basic MLOps pipeline using MLflow for model tracking"
        ],
        "90": [
            "Deploy production-ready RAG system with monitoring",
            "Implement advanced prompt engineering techniques",
            "Build automated model evaluation and testing framework",
            "Contribute to open-source RAG/LLM projects for portfolio"
        ]
    }
    plan_md = render_plan(plan)
    print(plan_md)
    
    # 5. Skills Assessment Table
    print("5. Skills Assessment Table:")
    skills = [
        {"name": "Python", "level": "Expert", "match_score": 0.95},
        {"name": "TensorFlow", "level": "Intermediate", "match_score": 0.78},
        {"name": "Machine Learning", "level": "Intermediate", "match_score": 0.82},
        {"name": "LLM Fine-tuning", "level": "Beginner", "match_score": 0.25},
        {"name": "Vector Databases", "match_score": 0.15},
        {"name": "RAG Systems", "match_score": 0.20},
        {"name": "MLOps", "level": "Beginner", "match_score": 0.30},
        {"name": "Prompt Engineering", "match_score": 0.35}
    ]
    skills_table = render_skills_table(skills, "Technical Skills Assessment")
    print(skills_table)
    
    # 6. Additional Sections
    print("6. Additional Sections:")
    
    # Learning Resources
    resources = [
        "LangChain Documentation and Tutorials",
        "Pinecone Vector Database Getting Started Guide", 
        "OpenAI Cookbook - RAG Examples",
        "MLflow Tracking and Model Registry Documentation",
        "Hugging Face Transformers Course",
        "DeepLearning.AI Short Courses on LLMs and RAG"
    ]
    resources_section = render_section("Recommended Learning Resources", 
                                     render_bullet_list(resources), 3)
    print(resources_section)
    
    # Action Steps
    action_steps = [
        "Schedule learning time blocks in calendar",
        "Set up development environment with required tools",
        "Join relevant communities (LangChain Discord, ML Twitter)",
        "Create GitHub repository for RAG experiments",
        "Set measurable goals and track progress weekly"
    ]
    actions_section = render_section("Immediate Action Steps",
                                   render_numbered_list(action_steps), 3)
    print(actions_section)
    
    # Code Example
    print("7. Code Example:")
    code_example = '''# Simple RAG implementation example
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Initialize vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# Add documents
documents = ["Document text here..."]
vectorstore.add_texts(documents)

# Query similar documents
query = "What is RAG?"
similar_docs = vectorstore.similarity_search(query, k=3)'''
    
    code_section = render_section("Sample RAG Implementation",
                                render_code_block(code_example, "python"), 3)
    print(code_section)
    
    # Key Takeaway Quote
    print("8. Key Takeaway:")
    quote = render_quote("The gap between knowing and doing is closed by practice. "
                        "Focus on building real projects to accelerate your learning.")
    quote_section = render_section("Key Takeaway", quote, 3)
    print(quote_section)
    
    print("9. Complete Report Preview:")
    print("=" * 50)
    
    # Generate complete report
    complete_report = (
        header +
        strengths_md + 
        gaps_md +
        plan_md +
        skills_table +
        resources_section +
        actions_section +
        code_section +
        quote_section +
        render_horizontal_rule() +
        render_section("Report Generated", "This report was automatically generated using the skill-gap-analyst tool.", 4)
    )
    
    print(complete_report)

if __name__ == "__main__":
    demo_markdown_utilities()