"""
I/O utilities for safe file operations with automatic format detection and directory creation.
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json
    HAS_ORJSON = False

try:
    import pymupdf4llm
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def read_text_auto(path: Union[str, Path]) -> str:
    """
    Automatically read text from files, using PDF extraction for .pdf files.
    
    Args:
        path: Path to file (.pdf, .txt, or other text files)
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If PDF reading fails and no fallback available
        
    Examples:
        >>> content = read_text_auto("document.pdf")
        >>> content = read_text_auto("notes.txt")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix.lower() == '.pdf':
        return _read_pdf(path)
    else:
        # Read as text file with encoding detection
        return _read_text_file(path)


def _read_pdf(path: Path) -> str:
    """Extract text from PDF using available tools."""
    # Try pymupdf4llm first (best for LLM processing)
    if HAS_PYMUPDF:
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(str(path))
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}, trying pdf2txt...")
    
    # Fallback to pdf2txt command line tool
    try:
        result = subprocess.run(
            ["pdf2txt", str(path)], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=60
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise ValueError(f"Failed to extract text from PDF {path}: {e}")


def _read_text_file(path: Path) -> str:
    """Read text file with encoding detection."""
    # Try common encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    # Final fallback - read as binary and decode with errors='replace'
    with open(path, 'rb') as f:
        content = f.read()
    return content.decode('utf-8', errors='replace')


def save_json(path: Union[str, Path], obj: Any, **kwargs) -> None:
    """
    Save object as JSON with automatic parent directory creation.
    
    Args:
        path: Output file path
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dump (if orjson not available)
        
    Examples:
        >>> save_json("artifacts/results.json", {"score": 0.95})
        >>> save_json("outputs/data.json", data_list, indent=2)
    """
    path = Path(path)
    
    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if HAS_ORJSON:
        # Use orjson for better performance
        with open(path, 'wb') as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
    else:
        # Fallback to standard json
        kwargs.setdefault('indent', 2)
        kwargs.setdefault('ensure_ascii', False)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, **kwargs)


def save_text(path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """
    Save text content with automatic parent directory creation.
    
    Args:
        path: Output file path
        content: Text content to save
        encoding: Text encoding (default: utf-8)
        
    Examples:
        >>> save_text("artifacts/report.txt", "Analysis complete")
        >>> save_text("outputs/summary.md", markdown_content)
    """
    path = Path(path)
    
    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load JSON data from file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Examples:
        >>> data = load_json("artifacts/config.json")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    if HAS_ORJSON:
        with open(path, 'rb') as f:
            return orjson.loads(f.read())
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Timestamp helpers
def timestamp_iso() -> str:
    """
    Get current timestamp in ISO format with timezone.
    
    Returns:
        ISO timestamp string (e.g., "2024-03-15T10:30:45.123456+00:00")
        
    Examples:
        >>> ts = timestamp_iso()
        >>> filename = f"report_{ts}.json"
    """
    return datetime.now(timezone.utc).isoformat()


def timestamp_filename() -> str:
    """
    Get current timestamp formatted for safe filename usage.
    
    Returns:
        Filename-safe timestamp (e.g., "20240315_103045")
        
    Examples:
        >>> ts = timestamp_filename()
        >>> save_json(f"artifacts/results_{ts}.json", data)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_human() -> str:
    """
    Get current timestamp in human-readable format.
    
    Returns:
        Human-readable timestamp (e.g., "2024-03-15 10:30:45")
        
    Examples:
        >>> print(f"Analysis completed at {timestamp_human()}")
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating parent directories as needed.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
        
    Examples:
        >>> artifacts_dir = ensure_dir("artifacts/models")
        >>> outputs_dir = ensure_dir("outputs")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str, max_length: int = 255) -> str:
    """
    Convert string to safe filename by removing/replacing problematic characters.
    
    Args:
        name: Original name
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
        
    Examples:
        >>> safe_name = safe_filename("Analysis: Results & Summary")
        >>> # Returns: "Analysis_Results_Summary"
    """
    import re
    
    # Replace problematic characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    safe = re.sub(r'[^\w\-_\.]', '_', safe)
    safe = re.sub(r'_+', '_', safe)
    safe = safe.strip('_')
    
    # Truncate if too long
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip('_')
    
    return safe or "unnamed"


# Legacy compatibility
def read_file(path: Union[str, Path]) -> str:
    """Legacy function - use read_text_auto instead."""
    return read_text_auto(path)
