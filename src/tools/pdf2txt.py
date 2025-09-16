import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def pdf_to_text(path: str) -> str:
    """
    Extract text from PDF or read text file.
    
    Args:
        path: Path to PDF or text file
        
    Returns:
        Clean text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or extraction fails
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Handle text files directly
    if file_path.suffix.lower() == '.txt':
        return _read_text_file(file_path)
    
    # Handle PDF files
    if file_path.suffix.lower() == '.pdf':
        return _extract_pdf_text(file_path)
    
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _read_text_file(file_path: Path) -> str:
    """Read and clean text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return _clean_text(content)
    except UnicodeDecodeError:
        # Try with different encodings
        for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.warning(f"Read text file with {encoding} encoding")
                return _clean_text(content)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode text file: {file_path}")


def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF using multiple strategies."""
    # Strategy 1: Try pypdf first (faster, better for simple PDFs)
    try:
        text = _extract_with_pypdf(file_path)
        if text and text.strip():
            logger.info(f"Successfully extracted text with pypdf: {len(text)} chars")
            return _clean_text(text)
    except Exception as e:
        logger.warning(f"pypdf extraction failed: {e}")
    
    # Strategy 2: Fallback to pdfminer.six (more robust)
    try:
        text = _extract_with_pdfminer(file_path)
        if text and text.strip():
            logger.info(f"Successfully extracted text with pdfminer: {len(text)} chars")
            return _clean_text(text)
    except Exception as e:
        logger.warning(f"pdfminer extraction failed: {e}")
    
    raise ValueError(f"Failed to extract text from PDF: {file_path}")


def _extract_with_pypdf(file_path: Path) -> str:
    """Extract text using pypdf library."""
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf library not available")
    
    text_parts = []
    
    with open(file_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                continue
    
    return '\n\n'.join(text_parts)


def _extract_with_pdfminer(file_path: Path) -> str:
    """Extract text using pdfminer.six library."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        raise ImportError("pdfminer.six library not available")
    
    return extract_text(str(file_path))


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text or not text.strip():
        return ""
    
    # Remove null bytes and other control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Basic de-hyphenation: join words split across lines
    # Match word- followed by whitespace and newline, then another word
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    # Normalize whitespace
    # Replace multiple whitespace characters with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Clean up line breaks - preserve paragraph breaks but remove single line breaks within paragraphs
    # Replace single newlines with space, keep double newlines as paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    
    # Remove excessive spaces around punctuation
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([,.;:!?])\s+', r'\1 ', text)
    
    # Remove leading/trailing whitespace and empty lines
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]  # Remove empty lines
    
    result = '\n'.join(lines).strip()
    
    # Final cleanup - ensure we don't have excessive whitespace
    result = re.sub(r' +', ' ', result)
    
    return result


# Convenience function for testing
def is_text_extraction_working() -> bool:
    """Test if text extraction dependencies are available."""
    try:
        import pypdf
        pypdf_available = True
    except ImportError:
        pypdf_available = False
    
    try:
        from pdfminer.high_level import extract_text
        pdfminer_available = True
    except ImportError:
        pdfminer_available = False
    
    return pypdf_available or pdfminer_available
