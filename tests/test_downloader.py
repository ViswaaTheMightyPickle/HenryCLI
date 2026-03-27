"""Tests for document downloader."""

import tempfile
from pathlib import Path

import pytest

from henrycli.downloader import DocumentDownloader


class TestDocumentDownloader:
    """Tests for DocumentDownloader."""

    @pytest.fixture
    def temp_downloader(self):
        """Create downloader with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DocumentDownloader(rag_directory=Path(tmpdir))

    def test_init_default_directory(self):
        """Test initialization with default directory."""
        downloader = DocumentDownloader()
        assert "henrycli" in str(downloader.get_rag_directory())
        assert "rag-docs" in str(downloader.get_rag_directory())

    def test_init_custom_directory(self):
        """Test initialization with custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = DocumentDownloader(rag_directory=Path(tmpdir))
            assert downloader.get_rag_directory() == Path(tmpdir)

    def test_generate_filename_from_url(self, temp_downloader):
        """Test filename generation from URL."""
        url = "https://example.com/document.pdf"
        filename = temp_downloader._generate_filename(url)
        assert filename == "document.pdf"

    def test_generate_filename_no_extension(self, temp_downloader):
        """Test filename generation for URL without extension."""
        url = "https://example.com/somepage"
        filename = temp_downloader._generate_filename(url)
        assert filename.endswith(".txt")
        assert "download_" in filename

    def test_generate_filename_sanitizes(self, temp_downloader):
        """Test filename sanitization."""
        url = "https://example.com/file with spaces & special!.pdf"
        filename = temp_downloader._generate_filename(url)
        # Special chars should be replaced with underscores
        assert " " not in filename
        assert "&" not in filename

    def test_convert_github_url_blob(self, temp_downloader):
        """Test GitHub blob URL conversion."""
        url = "https://github.com/user/repo/blob/main/file.py"
        converted = temp_downloader._convert_github_url(url)
        assert "raw.githubusercontent.com" in converted
        assert "user/repo/main/file.py" in converted

    def test_convert_github_url_tree(self, temp_downloader):
        """Test GitHub tree URL (directory)."""
        url = "https://github.com/user/repo/tree/main/src"
        converted = temp_downloader._convert_github_url(url)
        # Tree URLs are not converted (would need API call)
        assert converted == url

    def test_convert_non_github_url(self, temp_downloader):
        """Test non-GitHub URL passthrough."""
        url = "https://example.com/file.pdf"
        converted = temp_downloader._convert_github_url(url)
        assert converted == url

    def test_extract_arxiv_id_abs(self, temp_downloader):
        """Test arXiv ID extraction from abs URL."""
        url = "https://arxiv.org/abs/2301.12345"
        arxiv_id = temp_downloader._extract_arxiv_id(url)
        assert arxiv_id == "2301.12345"

    def test_extract_arxiv_id_pdf(self, temp_downloader):
        """Test arXiv ID extraction from pdf URL."""
        url = "https://arxiv.org/pdf/2301.12345.pdf"
        arxiv_id = temp_downloader._extract_arxiv_id(url)
        assert arxiv_id == "2301.12345"

    def test_extract_arxiv_id_invalid(self, temp_downloader):
        """Test arXiv ID extraction from invalid URL."""
        url = "https://example.com/not-arxiv"
        arxiv_id = temp_downloader._extract_arxiv_id(url)
        assert arxiv_id is None

    def test_has_supported_extension_true(self, temp_downloader):
        """Test supported extension check."""
        assert temp_downloader._has_supported_extension("file.pdf") is True
        assert temp_downloader._has_supported_extension("file.txt") is True
        assert temp_downloader._has_supported_extension("file.md") is True
        assert temp_downloader._has_supported_extension("file.py") is True

    def test_has_supported_extension_false(self, temp_downloader):
        """Test unsupported extension check."""
        assert temp_downloader._has_supported_extension("file.xyz") is False
        assert temp_downloader._has_supported_extension("file") is False

    def test_list_downloaded_empty(self, temp_downloader):
        """Test listing downloaded files when empty."""
        files = temp_downloader.list_downloaded()
        assert files == []

    def test_list_downloaded(self, temp_downloader):
        """Test listing downloaded files."""
        # Create test files
        (temp_downloader.get_rag_directory() / "file1.txt").write_text("content1")
        (temp_downloader.get_rag_directory() / "file2.pdf").write_text("content2")

        files = temp_downloader.list_downloaded()
        assert len(files) == 2

        filenames = [f["filename"] for f in files]
        assert "file1.txt" in filenames
        assert "file2.pdf" in filenames

    def test_delete_existing(self, temp_downloader):
        """Test deleting existing file."""
        # Create test file
        test_file = temp_downloader.get_rag_directory() / "to_delete.txt"
        test_file.write_text("content")

        result = temp_downloader.delete("to_delete.txt")
        assert result is True
        assert not test_file.exists()

    def test_delete_nonexistent(self, temp_downloader):
        """Test deleting nonexistent file."""
        result = temp_downloader.delete("nonexistent.txt")
        assert result is False

    @pytest.mark.asyncio
    async def test_download_batch(self, temp_downloader):
        """Test batch downloading (with mock URLs that will fail)."""
        urls = [
            "https://example.com/file1.txt",
            "https://example.com/file2.txt",
        ]

        results = await temp_downloader.download_batch(urls)
        assert len(results) == 2

        # All should fail (invalid URLs) but return proper structure
        for result in results:
            assert "success" in result
            assert "path" in result

    def test_rag_directory_creation(self):
        """Test that RAG directory is created on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag_dir = Path(tmpdir) / "new_rag"
            assert not rag_dir.exists()

            downloader = DocumentDownloader(rag_directory=rag_dir)
            assert rag_dir.exists()
            assert rag_dir.is_dir()
