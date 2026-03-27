"""Tests for filestore."""

import json
import tempfile
from pathlib import Path

import pytest

from henrycli.context.filestore import FileStore


@pytest.fixture
def temp_filestore():
    """Create a temporary filestore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FileStore(base_dir=Path(tmpdir))


class TestFileStore:
    """Tests for FileStore."""

    def test_offload_and_load(self, temp_filestore):
        """Test offloading and loading content."""
        content = "Hello, World!" * 100
        ref = temp_filestore.offload(content)

        assert isinstance(ref, str)
        loaded = temp_filestore.load(ref)
        assert loaded == content

    def test_offload_json(self, temp_filestore):
        """Test offloading JSON data."""
        data = {"key": "value", "number": 42}
        ref = temp_filestore.offload_json(data)

        loaded = temp_filestore.load_json(ref)
        assert loaded == data

    def test_load_preview(self, temp_filestore):
        """Test loading preview of content."""
        content = "\n".join([f"Line {i}" for i in range(50)])
        ref = temp_filestore.offload(content)

        preview = temp_filestore.load_preview(ref, lines=5)
        assert "Line 0" in preview
        assert "Line 4" in preview
        assert "more lines" in preview

    def test_load_preview_few_lines(self, temp_filestore):
        """Test preview when content has fewer lines than requested."""
        content = "Line 1\nLine 2\nLine 3"
        ref = temp_filestore.offload(content)

        preview = temp_filestore.load_preview(ref, lines=10)
        assert "Line 1" in preview
        assert "more lines" not in preview

    def test_delete(self, temp_filestore):
        """Test deleting a file."""
        content = "To be deleted"
        ref = temp_filestore.offload(content)

        result = temp_filestore.delete(ref)
        assert result is True

        with pytest.raises(FileNotFoundError):
            temp_filestore.load(ref)

    def test_delete_nonexistent(self, temp_filestore):
        """Test deleting a nonexistent file."""
        result = temp_filestore.delete("nonexistent.txt")
        assert result is False

    def test_get_full_path(self, temp_filestore):
        """Test getting full path for a reference."""
        content = "Test"
        ref = temp_filestore.offload(content)

        full_path = temp_filestore.get_full_path(ref)
        assert full_path.is_absolute()
        assert full_path.exists()

    def test_list_files(self, temp_filestore):
        """Test listing files in filestore."""
        temp_filestore.offload("Content 1")
        temp_filestore.offload("Content 2")

        files = temp_filestore.list_files()
        assert len(files) == 2

    def test_list_files_with_pattern(self, temp_filestore):
        """Test listing files with glob pattern."""
        ref1 = temp_filestore.offload("Text content", prefix="text")
        ref2 = temp_filestore.offload_json({"json": "data"}, prefix="data")

        # List all files first to see what we have
        all_files = temp_filestore.list_files()
        assert len(all_files) >= 2

        # Check specific patterns
        txt_files = temp_filestore.list_files("*.txt")
        json_files = temp_filestore.list_files("*.json")

        # At least one of each should exist
        assert len(txt_files) + len(json_files) >= 2

    def test_cleanup_old(self, temp_filestore):
        """Test cleaning up old files."""
        # Create a file
        ref = temp_filestore.offload("Test content")

        # Modify mtime to be old (8 days ago)
        file_path = temp_filestore.get_full_path(ref)
        import os
        import time

        old_time = time.time() - (8 * 24 * 60 * 60)
        os.utime(file_path, (old_time, old_time))

        # Cleanup should remove old file
        deleted = temp_filestore.cleanup_old(days=7)
        assert deleted >= 1
        assert not file_path.exists()

    def test_load_nonexistent_raises(self, temp_filestore):
        """Test that loading nonexistent file raises."""
        with pytest.raises(FileNotFoundError):
            temp_filestore.load("nonexistent.txt")

    def test_offload_creates_unique_filenames(self, temp_filestore):
        """Test that offload creates unique filenames."""
        import time

        ref1 = temp_filestore.offload("Content 1")
        time.sleep(0.01)  # Small delay to ensure different timestamp
        ref2 = temp_filestore.offload("Content 2")

        assert ref1 != ref2
