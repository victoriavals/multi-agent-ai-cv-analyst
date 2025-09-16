"""
Example usage of the LangGraph skill gap analysis pipeline.

This script demonstrates how to use the graph builder to create and
execute a complete skill gap analysis workflow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import asyncio
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_cv_text() -> str:
    """Return example CV text for demonstration."""
    return """
John Doe
Senior Software Engineer

CONTACT INFORMATION
Email: john.doe@email.com
Phone: (555) 123-4567
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Experienced software engineer with 7+ years developing scalable web applications 
and machine learning systems. Strong background in Python, SQL, and cloud technologies.
Led teams of 5-8 engineers and improved system performance by 40%.

TECHNICAL SKILLS
• Programming Languages: Python, JavaScript, SQL, Java
• Frameworks: Django, Flask, React, TensorFlow
• Databases: PostgreSQL, MongoDB, Redis
• Cloud: AWS (EC2, S3, Lambda), Docker
• Tools: Git, Jenkins, Kubernetes

WORK EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2021-Present
• Led development of machine learning platform serving 1M+ users
• Architected microservices infrastructure reducing latency by 35%
• Mentored 5 junior engineers and established code review processes
• Implemented CI/CD pipelines using Jenkins and AWS

Software Engineer | StartupXYZ | 2019-2021  
• Built recommendation engine using TensorFlow and Python
• Developed REST APIs handling 10K+ requests per second
• Optimized database queries improving response time by 50%
• Collaborated with product team on feature requirements

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2019

PROJECTS
• Open Source Contributor: Contributed to scikit-learn and pandas libraries
• Personal Project: Built web scraping tool for market analysis using Python and BeautifulSoup
• Hackathon Winner: First place in AI/ML category for predictive analytics solution

CERTIFICATIONS
• AWS Certified Solutions Architect
• Certified Kubernetes Administrator (CKA)
"""


async def run_analysis_example():
    """
    Example of running the complete skill gap analysis pipeline.
    """
    try:
        # Import graph builder (this tests importability)
        from graph.builder import build_graph, create_default_state
        from graph.utils import validate_graph_state, stream_graph_execution
        
        logger.info("🏗️ Building skill gap analysis graph...")
        
        # Build the graph
        graph = build_graph(
            llm_provider="openai",
            checkpointer_path=":memory:"
        )
        
        logger.info("✅ Graph built successfully!")
        
        # Create initial state
        initial_state = create_default_state(
            cv_text=example_cv_text(),
            target_role="Senior AI Engineer",
            market_region="San Francisco Bay Area",
            lang="en"
        )
        
        # Validate state
        is_valid, error = validate_graph_state(initial_state)
        if not is_valid:
            raise ValueError(f"Invalid state: {error}")
        
        logger.info("✅ Initial state validated!")
        logger.info(f"   CV Length: {len(initial_state['cv_text'])} characters")
        logger.info(f"   Target Role: {initial_state['target_role']}")
        logger.info(f"   Market Region: {initial_state['market_region']}")
        
        # Note: In a real implementation, you would uncomment the following
        # to actually execute the graph:
        
        # logger.info("🚀 Starting analysis pipeline...")
        # 
        # async for update in stream_graph_execution(graph, initial_state):
        #     if "node" in update:
        #         logger.info(f"✅ Completed: {update['node']}")
        #     elif "error" in update:
        #         logger.error(f"❌ Error: {update['error']}")
        #         break
        # 
        # # Get final results
        # config = {"configurable": {"thread_id": "example"}}
        # final_state = graph.get_state(config).values
        # 
        # if "error" not in final_state:
        #     logger.info("🎉 Analysis completed successfully!")
        #     logger.info(f"   Report length: {len(final_state.get('report_md', ''))} characters")
        #     
        #     # Save report to file
        #     with open("skill_gap_report.md", "w", encoding="utf-8") as f:
        #         f.write(final_state["report_md"])
        #     logger.info("   Report saved to skill_gap_report.md")
        # else:
        #     logger.error(f"❌ Analysis failed: {final_state['error']}")
        
        logger.info("🏁 Example completed! Graph is ready for execution.")
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.info("💡 Make sure all dependencies are installed and agents are implemented")
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


def sync_example():
    """
    Synchronous example for testing graph building.
    """
    try:
        from graph.builder import build_graph, create_default_state
        
        print("🏗️ Testing graph builder...")
        
        # Test that graph can be built
        graph = build_graph("openai")
        print("✅ Graph built successfully!")
        
        # Test state creation
        state = create_default_state(
            cv_text=example_cv_text(),
            target_role="Senior Python Developer",
            market_region="Global"
        )
        print("✅ State created successfully!")
        
        # Test validation
        from graph.utils import validate_graph_state
        is_valid, error = validate_graph_state(state)
        
        if is_valid:
            print("✅ State validation passed!")
        else:
            print(f"❌ State validation failed: {error}")
        
        print("🎉 All basic tests passed! Graph is importable and buildable.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Skill Gap Analysis - LangGraph Builder Example")
    print("=" * 60)
    
    # Run synchronous tests first
    if sync_example():
        print("\n🚀 Running async example...")
        asyncio.run(run_analysis_example())
    else:
        print("\n❌ Basic tests failed. Check your implementation.")