"""
Lightweight skill taxonomy helper for AI/LLM/MLOps domains.

Provides skill canonicalization and market mapping functionality with:
- Comprehensive dictionaries of canonical skills and aliases
- Skill normalization and deduplication
- Similarity-based market skill matching
"""

import re
from typing import Dict, List, Set, Callable, Any, Tuple
import numpy as np

# Canonical skill categories for AI/LLM/MLOps
CANONICAL_SKILLS = {
    # Programming Languages
    "python": ["python", "python3", "py"],
    "r": ["r", "r programming", "r lang"],
    "sql": ["sql", "structured query language", "tsql", "mysql", "postgresql"],
    "javascript": ["javascript", "js", "node.js", "nodejs", "typescript", "ts"],
    "java": ["java", "openjdk"],
    "scala": ["scala"],
    "go": ["golang", "go"],
    "rust": ["rust"],
    "c++": ["cpp", "c++", "c plus plus"],
    
    # Machine Learning Frameworks
    "tensorflow": ["tensorflow", "tf", "keras"],
    "pytorch": ["pytorch", "torch"],
    "scikit-learn": ["sklearn", "scikit-learn", "scikit learn"],
    "xgboost": ["xgboost", "xgb"],
    "lightgbm": ["lightgbm", "lgb"],
    "catboost": ["catboost"],
    "huggingface": ["huggingface", "hf", "transformers", "hugging face"],
    
    # LLM and NLP
    "large language models": ["llm", "llms", "large language models", "language models"],
    "transformers": ["transformer", "transformers", "attention", "self-attention"],
    "bert": ["bert", "bidirectional encoder representations"],
    "gpt": ["gpt", "generative pre-trained transformer"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "prompt engineering": ["prompt engineering", "prompting", "prompt design", "prompt optimization"],
    "fine-tuning": ["fine-tuning", "finetuning", "model fine-tuning", "llm fine-tuning"],
    "embeddings": ["embeddings", "word embeddings", "sentence embeddings", "text embeddings"],
    "nlp": ["nlp", "natural language processing", "text processing"],
    "sentiment analysis": ["sentiment analysis", "sentiment classification"],
    "named entity recognition": ["ner", "named entity recognition", "entity extraction"],
    "text classification": ["text classification", "document classification"],
    
    # Vector Databases
    "vector databases": ["vector db", "vector database", "vector databases"],
    "chroma": ["chroma", "chromadb", "chroma db"],
    "pinecone": ["pinecone"],
    "qdrant": ["qdrant"],
    "weaviate": ["weaviate"],
    "faiss": ["faiss", "facebook ai similarity search"],
    "milvus": ["milvus"],
    "elasticsearch": ["elasticsearch", "elastic search", "es"],
    
    # Cloud Platforms
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud platform", "google cloud"],
    "sagemaker": ["sagemaker", "amazon sagemaker"],
    "azure ml": ["azure ml", "azure machine learning"],
    "vertex ai": ["vertex ai", "google vertex ai"],
    
    # MLOps Tools
    "mlflow": ["mlflow", "ml flow"],
    "wandb": ["wandb", "weights and biases", "weights & biases"],
    "dvc": ["dvc", "data version control"],
    "kubeflow": ["kubeflow", "kube flow"],
    "airflow": ["airflow", "apache airflow"],
    "prefect": ["prefect"],
    "dagster": ["dagster"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s", "k8"],
    "jenkins": ["jenkins"],
    "github actions": ["github actions", "gh actions"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    
    # Monitoring and Observability
    "observability": ["observability", "monitoring", "tracing"],
    "prometheus": ["prometheus"],
    "grafana": ["grafana"],
    "langsmith": ["langsmith", "lang smith"],
    "langfuse": ["langfuse", "lang fuse"],
    "prompt telemetry": ["prompt telemetry", "llm monitoring", "prompt monitoring"],
    "model monitoring": ["model monitoring", "ml monitoring", "drift detection"],
    
    # Data Engineering
    "spark": ["spark", "apache spark", "pyspark"],
    "kafka": ["kafka", "apache kafka"],
    "snowflake": ["snowflake"],
    "databricks": ["databricks"],
    "dbt": ["dbt", "data build tool"],
    "pandas": ["pandas", "pd"],
    "numpy": ["numpy", "np"],
    "dask": ["dask"],
    
    # Deep Learning Specializations
    "computer vision": ["computer vision", "cv", "image processing"],
    "convolutional neural networks": ["cnn", "convolutional neural networks", "conv nets"],
    "recurrent neural networks": ["rnn", "recurrent neural networks", "lstm", "gru"],
    "generative adversarial networks": ["gan", "gans", "generative adversarial networks"],
    "diffusion models": ["diffusion models", "stable diffusion", "ddpm"],
    "reinforcement learning": ["reinforcement learning", "rl", "deep rl"],
    
    # API and Integration
    "rest apis": ["rest api", "rest apis", "restful apis", "http apis"],
    "graphql": ["graphql", "graph ql"],
    "openai api": ["openai api", "openai"],
    "anthropic api": ["anthropic api", "claude api"],
    "langchain": ["langchain", "lang chain"],
    "llamaindex": ["llamaindex", "llama index"],
    
    # Databases
    "postgresql": ["postgresql", "postgres", "pg"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "neo4j": ["neo4j", "graph database"],
    
    # Statistics and Math
    "statistics": ["statistics", "statistical analysis", "stats"],
    "linear algebra": ["linear algebra", "linalg"],
    "calculus": ["calculus", "differential calculus"],
    "probability": ["probability", "probability theory"],
    "bayesian methods": ["bayesian", "bayesian methods", "bayesian statistics"],
}

# Build reverse lookup for aliases to canonical skills
ALIAS_TO_CANONICAL = {}
for canonical, aliases in CANONICAL_SKILLS.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical

# Common noise patterns to clean - more conservative approach
NOISE_PATTERNS = [
    r'\b\d+\+?\s*years?\s*(of\s*)?',  # "5 years of", "3+ years"
    r'\bexperience\b',
    r'\bproficiency\b', 
    r'\bproficient\b',
    r'\bknowledge\b',
    r'\bstrong\b',
    r'\badvanced\b',
    r'\bbasic\b',
    r'\bintermediate\b',
    r'\bexpert\b',
    r'\bextensive\b',
    r'\bworking\s*with\b',
    r'\busing\b',
    r'\bprogramming\b',
    r'\bin\b',  # Remove preposition "in"
    r'\bwith\b',  # Remove preposition "with"
    r'\balgorithms?\b',  # Remove generic "algorithm(s)"
]

def _clean_skill_text(text: str) -> str:
    """Clean and normalize skill text."""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    cleaned = text.lower().strip()
    
    # Handle common abbreviations BEFORE removing noise patterns
    cleaned = re.sub(r'\bml\b', 'machine learning', cleaned)
    cleaned = re.sub(r'\bai\b', 'artificial intelligence', cleaned)
    cleaned = re.sub(r'\bds\b', 'data science', cleaned)
    cleaned = re.sub(r'\bde\b', 'data engineering', cleaned)
    
    # Remove noise patterns but preserve important technical terms
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    
    # Clean up version numbers but preserve core technology names
    cleaned = re.sub(r'([a-zA-Z]+)\d+(\.\d+)*', r'\1', cleaned)  # Remove version numbers attached to words
    
    # Normalize common variations
    cleaned = re.sub(r'\btensorflow\s+keras\b', 'tensorflow', cleaned)
    cleaned = re.sub(r'\bpytorch\s+lightning\b', 'pytorch', cleaned)
    cleaned = re.sub(r'\bhuggingface\s+transformers?\b', 'huggingface', cleaned)
    cleaned = re.sub(r'\bnode\.?js\b', 'javascript', cleaned)
    cleaned = re.sub(r'\btensor\s+flow\b', 'tensorflow', cleaned)
    cleaned = re.sub(r'\br[-\s]language\b', 'r', cleaned)
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def canonicalize_skills(skills: List[str]) -> List[str]:
    """
    Canonicalize a list of skills using alias mapping, case folding, and deduplication.
    
    Args:
        skills: List of skill strings (may contain noise, variations, duplicates)
        
    Returns:
        List of canonical skill names, deduplicated and sorted
        
    Examples:
        >>> canonicalize_skills(["Python", "TensorFlow", "tf", "PYTHON"])
        ['python', 'tensorflow']
        
        >>> canonicalize_skills(["RAG", "Vector DB", "ChromaDB"])
        ['chroma', 'rag', 'vector databases']
    """
    if not skills:
        return []
    
    canonical_set: Set[str] = set()
    
    for skill in skills:
        if not skill or not isinstance(skill, str):
            continue
            
        # Clean the skill text
        cleaned = _clean_skill_text(skill)
        if not cleaned:
            continue
        
        # Try exact match first
        if cleaned in ALIAS_TO_CANONICAL:
            canonical_set.add(ALIAS_TO_CANONICAL[cleaned])
            continue
        
        # Try finding skills within the cleaned text
        found_matches = []
        words = cleaned.split()
        
        # Check for multi-word matches first
        for alias, canonical in ALIAS_TO_CANONICAL.items():
            alias_words = alias.split()
            if len(alias_words) > 1:
                # Multi-word alias - check if all words are present
                if all(word in cleaned for word in alias_words):
                    found_matches.append(canonical)
                # Or check if the alias is a substring
                elif alias in cleaned:
                    found_matches.append(canonical)
        
        # Check for single-word matches
        for word in words:
            if word in ALIAS_TO_CANONICAL:
                found_matches.append(ALIAS_TO_CANONICAL[word])
        
        # Add all found matches
        if found_matches:
            canonical_set.update(found_matches)
        else:
            # If no match found, add the cleaned version (for unknown skills)
            canonical_set.add(cleaned)
    
    return sorted(list(canonical_set))

def closest_market_skills(
    candidate_skills: List[str], 
    market_skills: List[str], 
    embed_func: Callable[[List[str]], np.ndarray]
) -> Dict[str, Dict[str, Any]]:
    """
    Map candidate skills to closest market skills using cosine similarity.
    
    Args:
        candidate_skills: List of candidate's skills
        market_skills: List of market/job requirement skills
        embed_func: Function that takes list of strings and returns embeddings
        
    Returns:
        Dictionary mapping each market skill to best candidate match:
        {
            market_skill: {
                'match': candidate_skill,
                'similarity': float,
                'evidence': str
            }
        }
        
    Examples:
        >>> embed_func = lambda x: np.random.rand(len(x), 384)  # Mock
        >>> candidate = ['python programming', 'tensorflow']
        >>> market = ['python', 'machine learning frameworks']
        >>> result = closest_market_skills(candidate, market, embed_func)
        >>> 'python' in result
        True
    """
    if not candidate_skills or not market_skills:
        return {}
    
    # Canonicalize both skill sets
    canonical_candidate = canonicalize_skills(candidate_skills)
    canonical_market = canonicalize_skills(market_skills)
    
    if not canonical_candidate or not canonical_market:
        return {}
    
    try:
        # Get embeddings for both skill sets
        candidate_embeddings = embed_func(canonical_candidate)
        market_embeddings = embed_func(canonical_market)
        
        # Ensure embeddings are 2D arrays
        if candidate_embeddings.ndim == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        if market_embeddings.ndim == 1:
            market_embeddings = market_embeddings.reshape(1, -1)
        
        # Compute cosine similarity matrix
        # Normalize embeddings
        candidate_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        market_norm = market_embeddings / (np.linalg.norm(market_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Similarity matrix: market_skills x candidate_skills
        similarity_matrix = np.dot(market_norm, candidate_norm.T)
        
        # Find best matches for each market skill
        result = {}
        for i, market_skill in enumerate(canonical_market):
            similarities = similarity_matrix[i]
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            best_candidate = canonical_candidate[best_idx]
            
            # Generate evidence string
            evidence = f"'{best_candidate}' matches '{market_skill}'"
            if best_similarity > 0.8:
                evidence += " (strong match)"
            elif best_similarity > 0.6:
                evidence += " (good match)"
            elif best_similarity > 0.4:
                evidence += " (moderate match)"
            else:
                evidence += " (weak match)"
            
            result[market_skill] = {
                'match': best_candidate,
                'similarity': float(best_similarity),
                'evidence': evidence
            }
        
        return result
        
    except Exception as e:
        # Return empty result if embedding fails
        return {}

def get_skill_categories() -> Dict[str, List[str]]:
    """
    Get skill categories for organizational purposes.
    
    Returns:
        Dictionary mapping category names to lists of canonical skills
    """
    categories = {
        'programming_languages': [
            'python', 'r', 'sql', 'javascript', 'java', 'scala', 'go', 'rust', 'c++'
        ],
        'ml_frameworks': [
            'tensorflow', 'pytorch', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'huggingface'
        ],
        'llm_nlp': [
            'large language models', 'transformers', 'bert', 'gpt', 'rag', 
            'prompt engineering', 'fine-tuning', 'embeddings', 'nlp'
        ],
        'vector_databases': [
            'vector databases', 'chroma', 'pinecone', 'qdrant', 'weaviate', 'faiss', 'milvus'
        ],
        'cloud_platforms': [
            'aws', 'azure', 'gcp', 'sagemaker', 'azure ml', 'vertex ai'
        ],
        'mlops_tools': [
            'mlflow', 'wandb', 'dvc', 'kubeflow', 'airflow', 'prefect', 'dagster',
            'docker', 'kubernetes', 'jenkins', 'github actions', 'ci/cd'
        ],
        'observability': [
            'observability', 'prometheus', 'grafana', 'langsmith', 'langfuse',
            'prompt telemetry', 'model monitoring'
        ],
        'data_engineering': [
            'spark', 'kafka', 'snowflake', 'databricks', 'dbt', 'pandas', 'numpy', 'dask'
        ]
    }
    return categories

def get_canonical_skills() -> Dict[str, List[str]]:
    """Get the complete canonical skills dictionary."""
    return CANONICAL_SKILLS.copy()

def suggest_skills(partial: str, limit: int = 10) -> List[str]:
    """
    Suggest canonical skills based on partial input.
    
    Args:
        partial: Partial skill name
        limit: Maximum number of suggestions
        
    Returns:
        List of suggested canonical skill names
    """
    if not partial or len(partial) < 2:
        return []
    
    partial_lower = partial.lower()
    suggestions = []
    
    # First, look for exact starts
    for canonical in CANONICAL_SKILLS.keys():
        if canonical.startswith(partial_lower):
            suggestions.append(canonical)
    
    # Then, look for contains
    for canonical in CANONICAL_SKILLS.keys():
        if partial_lower in canonical and canonical not in suggestions:
            suggestions.append(canonical)
    
    # Finally, look in aliases
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        if partial_lower in alias and canonical not in suggestions:
            suggestions.append(canonical)
    
    return sorted(suggestions)[:limit]
