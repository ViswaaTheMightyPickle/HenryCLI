"""Filesystem-based filestore for offloading large content."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FileStore:
    """Manages filesystem storage for offloaded content."""

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize filestore.

        Args:
            base_dir: Base directory for filestore (default: ~/.henrycli/filestore)
        """
        if base_dir is None:
            base_dir = Path.home() / ".henrycli" / "filestore"
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, content: str) -> str:
        """Generate a unique filename based on content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{content_hash}.txt"

    def offload(self, content: str, prefix: str = "") -> str:
        """
        Store content to filesystem and return reference path.

        Args:
            content: Content to store
            prefix: Optional prefix for the filename

        Returns:
            File path relative to base_dir
        """
        filename = self._generate_filename(content)
        if prefix:
            filename = f"{prefix}_{filename}"

        file_path = self.base_dir / filename
        file_path.write_text(content, encoding="utf-8")

        return str(file_path.relative_to(self.base_dir))

    def offload_json(self, data: Any, prefix: str = "") -> str:
        """
        Store JSON data to filesystem and return reference path.

        Args:
            data: Data to serialize and store
            prefix: Optional prefix for the filename

        Returns:
            File path relative to base_dir
        """
        content = json.dumps(data, indent=2, default=str)
        filename = self._generate_filename(content)
        if prefix:
            filename = f"{prefix}_{filename}"

        file_path = self.base_dir / filename
        file_path.write_text(content, encoding="utf-8")

        return str(file_path.relative_to(self.base_dir))

    def load(self, ref: str) -> str:
        """
        Load content from filesystem reference.

        Args:
            ref: File path relative to base_dir

        Returns:
            Content as string
        """
        file_path = self.base_dir / ref
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path.read_text(encoding="utf-8")

    def load_json(self, ref: str) -> Any:
        """
        Load JSON data from filesystem reference.

        Args:
            ref: File path relative to base_dir

        Returns:
            Parsed JSON data
        """
        content = self.load(ref)
        return json.loads(content)

    def load_preview(self, ref: str, lines: int = 10) -> str:
        """
        Load a preview of content (first N lines).

        Args:
            ref: File path relative to base_dir
            lines: Number of lines to preview

        Returns:
            Preview of content
        """
        content = self.load(ref)
        content_lines = content.split("\n")
        preview = "\n".join(content_lines[:lines])
        if len(content_lines) > lines:
            preview += f"\n... ({len(content_lines) - lines} more lines)"
        return preview

    def delete(self, ref: str) -> bool:
        """
        Delete a file from the filestore.

        Args:
            ref: File path relative to base_dir

        Returns:
            True if deleted, False if not found
        """
        file_path = self.base_dir / ref
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def get_full_path(self, ref: str) -> Path:
        """
        Get full filesystem path for a reference.

        Args:
            ref: File path relative to base_dir

        Returns:
            Full Path object
        """
        return self.base_dir / ref

    def list_files(self, pattern: str = "*.txt") -> list[str]:
        """
        List files in the filestore.

        Args:
            pattern: Glob pattern to match

        Returns:
            List of file references
        """
        files = self.base_dir.glob(pattern)
        return [
            str(f.relative_to(self.base_dir))
            for f in files
            if f.is_file()
        ]

    def cleanup_old(self, days: int = 7) -> int:
        """
        Clean up files older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of files deleted
        """
        import time

        now = time.time()
        threshold = now - (days * 24 * 60 * 60)
        deleted = 0

        for file_path in self.base_dir.iterdir():
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                if mtime < threshold:
                    file_path.unlink()
                    deleted += 1

        return deleted
