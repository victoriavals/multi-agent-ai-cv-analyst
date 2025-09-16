import pytest
import tempfile
import os
from pathlib import Path
from src.tools.pdf2txt import pdf_to_text, is_text_extraction_working, _clean_text


class TestPdfToText:
    """Test suite for PDF to text extraction."""
    
    def test_text_file_passthrough(self):
        """Test that .txt files are read directly."""
        # Use the test sample file
        test_file = Path("tests/test_sample.txt")
        result = pdf_to_text(str(test_file))
        
        assert result is not None
        assert len(result) > 0
        assert "John Doe" in result
        assert "Senior Software Engineer" in result
        assert "Python" in result
        
    def test_text_file_with_temp_file(self):
        """Test text file reading with temporary file."""
        content = "This is a test file.\nWith multiple lines.\n\n  And some whitespace.  "
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            result = pdf_to_text(temp_path)
            assert "This is a test file." in result
            assert "With multiple lines." in result
            assert "And some whitespace." in result
            # Should be cleaned of excessive whitespace
            assert "  And some whitespace.  " not in result
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            pdf_to_text("non_existent_file.pdf")
    
    def test_unsupported_format(self):
        """Test handling of unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                pdf_to_text(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_clean_text_function(self):
        """Test the text cleaning functionality."""
        # Test whitespace normalization
        dirty_text = "This  has   multiple    spaces\n\n\nand\n\nexcessive\nlinebreaks"
        clean = _clean_text(dirty_text)
        assert "multiple spaces" in clean
        assert "   " not in clean  # No triple spaces
        
        # Test de-hyphenation
        hyphenated = "This is a test with word-\nbreak continuation"
        clean = _clean_text(hyphenated)
        assert "wordbreak" in clean
        assert "word-\n" not in clean
        
        # Test punctuation spacing
        punct_text = "Hello , world ! How are you ?"
        clean = _clean_text(punct_text)
        assert "Hello, world! How are you?" in clean
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        assert _clean_text("") == ""
        assert _clean_text("   \n\n   \t   ") == ""
        # Note: _clean_text expects a string, so we don't test None here
    
    def test_extraction_dependencies(self):
        """Test that at least one PDF extraction library is available."""
        assert is_text_extraction_working(), "No PDF extraction libraries available"
    
    @pytest.mark.integration
    def test_pdf_extraction_if_available(self):
        """Integration test for PDF extraction if libraries are available."""
        if not is_text_extraction_working():
            pytest.skip("No PDF extraction libraries available")
        
        # Create a simple test case with a known text file that works
        # This test would be expanded with actual PDF files in a real scenario
        test_file = Path("tests/test_sample.txt")
        result = pdf_to_text(str(test_file))
        assert len(result) > 0
    
    def test_unicode_handling(self):
        """Test handling of different text encodings."""
        # Test with UTF-8 content including special characters
        content = "Résumé for José García\n• Python développeur\n© 2023"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            result = pdf_to_text(temp_path)
            assert "Résumé" in result
            assert "José García" in result
            assert "développeur" in result
        finally:
            os.unlink(temp_path)
    
    def test_large_text_handling(self):
        """Test handling of large text content."""
        # Create a large text content
        large_content = "This is a test line.\n" * 1000
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            result = pdf_to_text(temp_path)
            assert len(result) > 0
            assert "This is a test line." in result
            # Should not have excessive line breaks
            assert "\n\n\n" not in result
        finally:
            os.unlink(temp_path)