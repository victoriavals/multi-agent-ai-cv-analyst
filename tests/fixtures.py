"""
Test fixtures for integration tests.

Contains minimal sample data for live API integration testing.
"""


def sample_cv_text():
    """
    Sample CV text for integration testing.
    
    Returns a realistic CV text that can be used to test live parsing and analysis.
    """
    return """
ALICE CHEN
Senior AI Engineer
Email: alice.chen@email.com
Phone: (555) 123-4567
Location: San Francisco, CA

SUMMARY
Experienced AI Engineer with 8+ years developing machine learning systems and deep learning applications. Proven track record of leading AI initiatives at scale, implementing production ML pipelines, and mentoring engineering teams.

TECHNICAL SKILLS
• Programming Languages: Python, R, Java, C++, SQL
• Machine Learning: PyTorch, TensorFlow, Scikit-learn, Keras, MLflow
• Deep Learning: Computer Vision, NLP, Transformers, BERT, GPT
• Cloud Platforms: AWS (SageMaker, EC2, S3), Google Cloud Platform, Azure
• Data Engineering: Apache Spark, Kafka, Airflow, Docker, Kubernetes
• Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
• Tools: Git, Jenkins, Jupyter, Weights & Biases, Tableau

PROFESSIONAL EXPERIENCE

Senior AI Engineer | TechCorp Inc. | 2020 - Present
• Lead a team of 6 ML engineers developing recommendation systems serving 10M+ users
• Implemented real-time inference pipeline reducing latency by 40% using optimized PyTorch models
• Designed and deployed A/B testing framework for ML models improving conversion rates by 15%
• Built MLOps infrastructure using Kubernetes and MLflow for model versioning and deployment
• Mentored junior engineers and established ML best practices across the organization

Machine Learning Engineer | DataCorp Solutions | 2018 - 2020
• Developed computer vision models for autonomous vehicle perception achieving 98% accuracy
• Implemented end-to-end ML pipeline processing 1TB+ of sensor data daily
• Collaborated with product teams to integrate ML models into customer-facing applications
• Optimized model performance reducing inference time by 60% through quantization and pruning

AI Research Engineer | InnovateLab | 2016 - 2018
• Conducted research in natural language processing and published 5 peer-reviewed papers
• Developed novel transformer architectures for question-answering systems
• Prototyped conversational AI agents using BERT and GPT-based models
• Presented research findings at top-tier ML conferences (NeurIPS, ICML, ICLR)

EDUCATION
Ph.D. Computer Science (Machine Learning) | Stanford University | 2016
M.S. Computer Science | UC Berkeley | 2014
B.S. Computer Engineering | MIT | 2012

CERTIFICATIONS
• AWS Certified Machine Learning - Specialty (2023)
• TensorFlow Developer Certificate (2022)
• Google Cloud Professional ML Engineer (2021)

PROJECTS
Smart Home AI Assistant
• Built end-to-end voice assistant using transformer models and speech recognition
• Integrated with IoT devices for automated home control and energy optimization
• Technologies: PyTorch, Transformers, FastAPI, React, Raspberry Pi

Medical Image Analysis Platform
• Developed CNN models for medical imaging achieving diagnostic accuracy comparable to radiologists
• Deployed HIPAA-compliant solution processing 1000+ medical scans daily
• Technologies: TensorFlow, OpenCV, Flask, PostgreSQL, Docker

Open Source Contributions
• Core contributor to PyTorch Lightning with 50+ commits
• Maintainer of ML utilities library with 2000+ GitHub stars
• Regular speaker at PyData and MLOps conferences

PUBLICATIONS
• "Attention Mechanisms in Medical Image Analysis" - Nature Machine Intelligence (2023)
• "Scalable ML Pipeline Architecture" - ICML (2022)
• "Transformer Optimization for Production" - NeurIPS (2021)
• "Few-Shot Learning in Computer Vision" - CVPR (2020)
• "Neural Architecture Search for Edge Devices" - ICLR (2019)
"""
