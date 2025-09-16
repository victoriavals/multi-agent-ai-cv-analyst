"""
Tests for I/O utilities.
"""

import json
import tempfile
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.utils.io import (
    read_text_auto,
    save_json,
    save_text,
    load_json,
    timestamp_iso,
    timestamp_filename,
    timestamp_human,
    ensure_dir,
    safe_filename,
    _read_text_file,
    _read_pdf,
)


class TestReadTextAuto:
    """Test automatic text reading functionality."""
    
    def test_read_text_file(self, tmp_path):
        """Test reading regular text files."""
        test_file = tmp_path / "test.txt"
        content = "Hello, World!\nThis is a test file."
        test_file.write_text(content, encoding='utf-8')
        
        result = read_text_auto(test_file)
        assert result == content
    
    def test_read_text_file_with_encoding(self, tmp_path):
        """Test reading text files with different encodings."""
        test_file = tmp_path / "test_latin1.txt"
        content = "Café résumé naïve"
        test_file.write_text(content, encoding='latin1')
        
        result = read_text_auto(test_file)
        assert content in result  # Should handle encoding gracefully
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            read_text_auto("nonexistent.txt")
    
    @patch('src.utils.io.HAS_PYMUPDF', False)
    @patch('subprocess.run')
    def test_read_pdf_with_pdf2txt(self, mock_run, tmp_path):
        """Test PDF reading with pdf2txt fallback."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")
        
        mock_run.return_value.stdout = "Extracted PDF text"
        mock_run.return_value.returncode = 0
        
        result = read_text_auto(pdf_file)
        assert result == "Extracted PDF text"
        mock_run.assert_called_once()
    
    @patch('src.utils.io.HAS_PYMUPDF', True)
    def test_read_pdf_with_pymupdf(self, tmp_path):
        """Test PDF reading with pymupdf4llm."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")
        
        # Mock the import and function at the module level
        with patch.dict('sys.modules', {'pymupdf4llm': MagicMock()}):
            import sys
            sys.modules['pymupdf4llm'].to_markdown = MagicMock(return_value="# PDF Content\nExtracted text")
            
            result = read_text_auto(pdf_file)
            assert result == "# PDF Content\nExtracted text"
    
    @patch('src.utils.io.HAS_PYMUPDF', False)
    @patch('subprocess.run')
    def test_read_pdf_failure(self, mock_run, tmp_path):
        """Test PDF reading failure handling."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")
        
        mock_run.side_effect = FileNotFoundError("pdf2txt not found")
        
        with pytest.raises(ValueError, match="Failed to extract text from PDF"):
            read_text_auto(pdf_file)


class TestSaveJson:
    """Test JSON saving functionality."""
    
    def test_save_json_basic(self, tmp_path):
        """Test basic JSON saving."""
        data = {"name": "test", "value": 42}
        json_file = tmp_path / "test.json"
        
        save_json(json_file, data)
        
        assert json_file.exists()
        with open(json_file, 'r') as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_save_json_with_nested_dirs(self, tmp_path):
        """Test JSON saving with automatic directory creation."""
        data = {"nested": True}
        json_file = tmp_path / "artifacts" / "output" / "test.json"
        
        save_json(json_file, data)
        
        assert json_file.exists()
        assert json_file.parent.exists()
        with open(json_file, 'r') as f:
            loaded = json.load(f)
        assert loaded == data
    
    @patch('src.utils.io.HAS_ORJSON', False)
    def test_save_json_fallback(self, tmp_path):
        """Test JSON saving with fallback to standard json."""
        data = {"fallback": True, "unicode": "test"}
        json_file = tmp_path / "fallback.json"
        
        save_json(json_file, data, indent=4)
        
        assert json_file.exists()
        content = json_file.read_text()
        assert "fallback" in content
        assert "test" in content


class TestSaveText:
    """Test text saving functionality."""
    
    def test_save_text_basic(self, tmp_path):
        """Test basic text saving."""
        content = "Hello, World!\nThis is a test."
        text_file = tmp_path / "test.txt"
        
        save_text(text_file, content)
        
        assert text_file.exists()
        assert text_file.read_text() == content
    
    def test_save_text_with_nested_dirs(self, tmp_path):
        """Test text saving with automatic directory creation."""
        content = "Nested content"
        text_file = tmp_path / "outputs" / "reports" / "summary.txt"
        
        save_text(text_file, content)
        
        assert text_file.exists()
        assert text_file.parent.exists()
        assert text_file.read_text() == content
    
    def test_save_text_with_encoding(self, tmp_path):
        """Test text saving with specific encoding."""
        content = "Café résumé naïve"
        text_file = tmp_path / "unicode.txt"
        
        save_text(text_file, content, encoding='utf-8')
        
        assert text_file.exists()
        assert text_file.read_text(encoding='utf-8') == content


class TestLoadJson:
    """Test JSON loading functionality."""
    
    def test_load_json_basic(self, tmp_path):
        """Test basic JSON loading."""
        data = {"name": "test", "values": [1, 2, 3]}
        json_file = tmp_path / "test.json"
        
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        result = load_json(json_file)
        assert result == data
    
    def test_load_json_not_found(self):
        """Test loading non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            load_json("nonexistent.json")
    
    @patch('src.utils.io.HAS_ORJSON', False)
    def test_load_json_fallback(self, tmp_path):
        """Test JSON loading with fallback to standard json."""
        data = {"fallback": True}
        json_file = tmp_path / "fallback.json"
        
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        result = load_json(json_file)
        assert result == data


class TestTimestampHelpers:
    """Test timestamp utility functions."""
    
    def test_timestamp_iso(self):
        """Test ISO timestamp generation."""
        timestamp = timestamp_iso()
        
        # Should be valid ISO format with timezone
        assert "T" in timestamp
        assert "+" in timestamp or timestamp.endswith("Z")
        # Should be parseable
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    
    def test_timestamp_filename(self):
        """Test filename-safe timestamp generation."""
        timestamp = timestamp_filename()
        
        # Should be YYYYMMDD_HHMMSS format
        assert len(timestamp) == 15
        assert "_" in timestamp
        assert timestamp.replace("_", "").isdigit()
    
    def test_timestamp_human(self):
        """Test human-readable timestamp generation."""
        timestamp = timestamp_human()
        
        # Should be YYYY-MM-DD HH:MM:SS format
        assert len(timestamp) == 19
        assert timestamp.count("-") == 2
        assert timestamp.count(":") == 2
        assert " " in timestamp


class TestEnsureDir:
    """Test directory creation utility."""
    
    def test_ensure_dir_new(self, tmp_path):
        """Test creating new directory."""
        new_dir = tmp_path / "new_directory"
        
        result = ensure_dir(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
    
    def test_ensure_dir_nested(self, tmp_path):
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        
        result = ensure_dir(nested_dir)
        
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert result == nested_dir
    
    def test_ensure_dir_existing(self, tmp_path):
        """Test with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        result = ensure_dir(existing_dir)
        
        assert existing_dir.exists()
        assert result == existing_dir


class TestSafeFilename:
    """Test safe filename generation."""
    
    def test_safe_filename_basic(self):
        """Test basic filename sanitization."""
        result = safe_filename("simple_name.txt")
        assert result == "simple_name.txt"
    
    def test_safe_filename_with_problematic_chars(self):
        """Test filename with problematic characters."""
        result = safe_filename("Analysis: Results & Summary")
        assert result == "Analysis_Results_Summary"
    
    def test_safe_filename_with_path_chars(self):
        """Test filename with path-like characters."""
        result = safe_filename("folder/file<name>.txt")
        assert result == "folder_file_name_.txt"
    
    def test_safe_filename_max_length(self):
        """Test filename length limiting."""
        long_name = "a" * 300
        result = safe_filename(long_name, max_length=50)
        assert len(result) <= 50
    
    def test_safe_filename_empty(self):
        """Test empty filename handling."""
        result = safe_filename("")
        assert result == "unnamed"
        
        result = safe_filename("!!!")
        assert result == "unnamed"


class TestIntegration:
    """Test integrated functionality."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow with multiple functions."""
        # Create directory structure
        artifacts_dir = ensure_dir(tmp_path / "artifacts")
        outputs_dir = ensure_dir(tmp_path / "outputs")
        
        # Save some data
        data = {
            "timestamp": timestamp_iso(),
            "results": [1, 2, 3],
            "metadata": {"version": "1.0"}
        }
        
        # Use timestamp in filename
        ts = timestamp_filename()
        json_file = artifacts_dir / f"results_{ts}.json"
        save_json(json_file, data)
        
        # Save text report
        report = f"Analysis completed at {timestamp_human()}\nResults: {len(data['results'])} items"
        text_file = outputs_dir / "report.txt"
        save_text(text_file, report)
        
        # Load and verify
        loaded_data = load_json(json_file)
        loaded_text = read_text_auto(text_file)
        
        assert loaded_data == data
        assert "Analysis completed" in loaded_text
        assert json_file.exists()
        assert text_file.exists()
    
    def test_safe_artifacts_creation(self, tmp_path):
        """Test that artifacts are created safely in nested structure."""
        # Simulate user input with problematic characters
        analysis_name = "User Analysis: Q1 2024 <Final>"
        safe_name = safe_filename(analysis_name)
        
        # Create timestamped output
        timestamp = timestamp_filename()
        output_path = tmp_path / "artifacts" / "analyses" / f"{safe_name}_{timestamp}.json"
        
        data = {
            "analysis_name": analysis_name,
            "created_at": timestamp_iso(),
            "safe_filename": safe_name
        }
        
        save_json(output_path, data)
        
        # Verify safe creation
        assert output_path.exists()
        assert output_path.parent.exists()
        
        loaded = load_json(output_path)
        assert loaded["analysis_name"] == analysis_name
        assert loaded["safe_filename"] == safe_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])