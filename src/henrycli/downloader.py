"""File downloader for RAG documents."""

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import httpx


class DocumentDownloader:
    """
    Downloads documents to the RAG directory.
    
    Supports:
    - Direct URLs (PDF, TXT, MD, etc.)
    - GitHub URLs (files, repos)
    - ArXiv papers
    - Plain text from web pages
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".txt", ".md", ".markdown", ".rst",
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
        ".json", ".yaml", ".yml", ".xml", ".csv",
        ".html", ".htm",
    }

    def __init__(self, rag_directory: Path | None = None):
        """
        Initialize downloader.

        Args:
            rag_directory: Directory to store downloaded files
        """
        if rag_directory is None:
            rag_directory = Path.home() / ".henrycli" / "rag-docs"
        self.rag_directory = rag_directory
        self.rag_directory.mkdir(parents=True, exist_ok=True)

    async def download(
        self,
        url: str,
        filename: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Download a file from URL to RAG directory.

        Args:
            url: URL to download from
            filename: Optional filename (auto-generated if None)
            overwrite: Overwrite existing file

        Returns:
            Download result with path and status
        """
        # Handle special URL types
        if "github.com" in url:
            url = self._convert_github_url(url)
        elif "arxiv.org" in url or "ar5iv.org" in url:
            return await self._download_arxiv(url, filename, overwrite)

        # Generate filename
        if filename is None:
            filename = self._generate_filename(url)

        file_path = self.rag_directory / filename

        # Check if file exists
        if file_path.exists() and not overwrite:
            return {
                "success": False,
                "path": str(file_path),
                "message": f"File already exists: {filename}",
                "skipped": True,
            }

        # Download file
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Write file
                with open(file_path, "wb") as f:
                    f.write(response.content)

                return {
                    "success": True,
                    "path": str(file_path),
                    "filename": filename,
                    "size_bytes": len(response.content),
                    "url": url,
                }

        except httpx.HTTPError as e:
            return {
                "success": False,
                "path": str(file_path),
                "message": f"Download failed: {e}",
                "error": str(e),
            }

    async def download_batch(
        self,
        urls: list[str],
        overwrite: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Download multiple files concurrently.

        Args:
            urls: List of URLs to download
            overwrite: Overwrite existing files

        Returns:
            List of download results
        """
        tasks = [self.download(url, overwrite=overwrite) for url in urls]
        results = await asyncio.gather(*tasks)
        return list(results)

    def _convert_github_url(self, url: str) -> str:
        """
        Convert GitHub URL to raw content URL.

        Args:
            url: GitHub URL (e.g., github.com/user/repo/blob/main/file.py)

        Returns:
            Raw content URL
        """
        # Pattern: github.com/user/repo/blob/branch/path
        pattern = r"github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)"
        match = re.search(pattern, url)

        if match:
            user, repo, branch, path = match.groups()
            return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"

        # Pattern: github.com/user/repo/tree/branch/path (directory)
        pattern = r"github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)"
        match = re.search(pattern, url)

        if match:
            # For directories, we'd need to list files first
            # For now, return original URL with warning
            pass

        return url

    async def _download_arxiv(
        self,
        url: str,
        filename: str | None,
        overwrite: bool,
    ) -> dict[str, Any]:
        """
        Download arXiv paper.

        Args:
            url: arXiv URL
            filename: Optional filename
            overwrite: Overwrite existing file

        Returns:
            Download result
        """
        # Extract arXiv ID
        arxiv_id = self._extract_arxiv_id(url)
        if not arxiv_id:
            return {
                "success": False,
                "message": "Could not extract arXiv ID from URL",
            }

        # Use ar5iv for HTML/PDF conversion
        ar5iv_url = f"https://ar5iv.org/html/{arxiv_id}"

        if filename is None:
            filename = f"arxiv_{arxiv_id}.html"

        return await self.download(ar5iv_url, filename, overwrite)

    def _extract_arxiv_id(self, url: str) -> str | None:
        """
        Extract arXiv ID from URL.

        Args:
            url: arXiv URL

        Returns:
            arXiv ID or None
        """
        # Pattern: arxiv.org/abs/1234.56789
        pattern = r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)"
        match = re.search(pattern, url)

        if match:
            return match.group(1)

        return None

    def _generate_filename(self, url: str) -> str:
        """
        Generate a filename from URL.

        Args:
            url: URL to generate filename from

        Returns:
            Generated filename
        """
        # Extract path from URL
        path = url.split("?")[0]  # Remove query params
        filename = path.split("/")[-1]

        # If no extension, add .txt
        if "." not in filename or not self._has_supported_extension(filename):
            # Generate hash-based filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"download_{url_hash}.txt"

        # Sanitize filename
        filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

        return filename

    def _has_supported_extension(self, filename: str) -> bool:
        """
        Check if filename has a supported extension.

        Args:
            filename: Filename to check

        Returns:
            True if extension is supported
        """
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def list_downloaded(self) -> list[dict[str, Any]]:
        """
        List all downloaded files.

        Returns:
            List of file information
        """
        files = []
        for file_path in self.rag_directory.iterdir():
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                })
        return sorted(files, key=lambda x: x["modified"], reverse=True)

    def delete(self, filename: str) -> bool:
        """
        Delete a downloaded file.

        Args:
            filename: Filename to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self.rag_directory / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def get_rag_directory(self) -> Path:
        """
        Get the RAG directory path.

        Returns:
            Path to RAG directory
        """
        return self.rag_directory
