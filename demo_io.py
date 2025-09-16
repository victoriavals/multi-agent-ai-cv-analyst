"""
Demonstration of I/O utilities for safe artifacts creation.
"""

from src.utils.io import (
    read_text_auto,
    save_json,
    save_text,
    load_json,
    timestamp_iso,
    timestamp_filename,
    timestamp_human,
    ensure_dir,
    safe_filename
)

def demo_io_utilities():
    """Demonstrate I/O utilities creating artifacts safely."""
    
    print("=== I/O Utilities Demo ===\n")
    
    # 1. Safe directory creation
    print("1. Creating directory structure...")
    artifacts_dir = ensure_dir("artifacts")
    outputs_dir = ensure_dir("outputs/reports")
    models_dir = ensure_dir("artifacts/models")
    print(f"✓ Created: {artifacts_dir}")
    print(f"✓ Created: {outputs_dir}")
    print(f"✓ Created: {models_dir}")
    
    # 2. Timestamp utilities
    print("\n2. Timestamp utilities...")
    iso_ts = timestamp_iso()
    filename_ts = timestamp_filename()
    human_ts = timestamp_human()
    print(f"✓ ISO timestamp: {iso_ts}")
    print(f"✓ Filename timestamp: {filename_ts}")
    print(f"✓ Human timestamp: {human_ts}")
    
    # 3. Safe filename generation
    print("\n3. Safe filename generation...")
    unsafe_names = [
        "Analysis: Q1 Results & Summary",
        "Model/Performance<Test>.json",
        "User Data: 2024\\Report",
        "Special chars: <>:\"/\\|?*"
    ]
    
    for unsafe_name in unsafe_names:
        safe_name = safe_filename(unsafe_name)
        print(f"✓ '{unsafe_name}' → '{safe_name}'")
    
    # 4. JSON saving with nested structure
    print("\n4. Saving JSON data...")
    analysis_data = {
        "analysis_id": "demo_001",
        "created_at": iso_ts,
        "timestamp_human": human_ts,
        "results": {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88
        },
        "metadata": {
            "model": "gpt-4",
            "dataset": "skill_analysis_v1",
            "version": "1.0.0"
        },
        "skills_analyzed": [
            "Python programming",
            "Machine learning",
            "Data analysis",
            "LLM fine-tuning"
        ]
    }
    
    json_path = f"artifacts/analysis_{filename_ts}.json"
    save_json(json_path, analysis_data)
    print(f"✓ Saved JSON: {json_path}")
    
    # 5. Text report generation
    print("\n5. Generating text report...")
    report_content = f"""# Skill Gap Analysis Report
    
Generated: {human_ts}
Analysis ID: {analysis_data['analysis_id']}

## Results Summary
- Accuracy: {analysis_data['results']['accuracy']:.1%}
- Precision: {analysis_data['results']['precision']:.1%}  
- Recall: {analysis_data['results']['recall']:.1%}

## Skills Analyzed
{chr(10).join(f'- {skill}' for skill in analysis_data['skills_analyzed'])}

## Model Information
- Model: {analysis_data['metadata']['model']}
- Dataset: {analysis_data['metadata']['dataset']}
- Version: {analysis_data['metadata']['version']}

---
Report generated automatically by skill-gap-analyst
"""
    
    report_path = f"outputs/reports/analysis_report_{filename_ts}.md"
    save_text(report_path, report_content)
    print(f"✓ Saved report: {report_path}")
    
    # 6. Configuration file
    print("\n6. Saving configuration...")
    config = {
        "app_name": "skill-gap-analyst",
        "version": "1.0.0",
        "last_updated": iso_ts,
        "paths": {
            "artifacts": "artifacts",
            "outputs": "outputs",
            "models": "artifacts/models"
        },
        "settings": {
            "max_skills": 100,
            "similarity_threshold": 0.8,
            "batch_size": 32
        }
    }
    
    config_path = "artifacts/config.json"
    save_json(config_path, config)
    print(f"✓ Saved config: {config_path}")
    
    # 7. Reading back and verifying
    print("\n7. Reading and verifying data...")
    loaded_analysis = load_json(json_path)
    loaded_config = load_json(config_path)
    loaded_report = read_text_auto(report_path)
    
    print(f"✓ Loaded analysis: {loaded_analysis['analysis_id']}")
    print(f"✓ Loaded config: {loaded_config['app_name']} v{loaded_config['version']}")
    print(f"✓ Loaded report: {len(loaded_report)} characters")
    
    # 8. Demonstrating text file reading
    print("\n8. Creating and reading text file...")
    sample_text_path = "artifacts/sample_data.txt"
    sample_text = """Sample skill descriptions for testing:

1. "5 years of Python programming experience with Django and Flask"
2. "Strong knowledge in TensorFlow 2.0 and PyTorch for deep learning"
3. "Proficient with Docker containerization and Kubernetes orchestration"
4. "Expert in RAG systems using ChromaDB and OpenAI embeddings"
5. "Advanced MLOps with MLflow, DVC, and CI/CD pipelines"
"""
    
    save_text(sample_text_path, sample_text)
    loaded_text = read_text_auto(sample_text_path)
    print(f"✓ Saved and loaded text file: {len(loaded_text)} characters")
    
    print("\n=== Demo Complete ===")
    print("All files created safely in artifacts/ and outputs/ directories!")
    print("\nFiles created:")
    
    import os
    from pathlib import Path
    
    for root, dirs, files in os.walk("artifacts"):
        for file in files:
            file_path = Path(root) / file
            print(f"  📄 {file_path}")
    
    for root, dirs, files in os.walk("outputs"):
        for file in files:
            file_path = Path(root) / file
            print(f"  📄 {file_path}")

if __name__ == "__main__":
    demo_io_utilities()