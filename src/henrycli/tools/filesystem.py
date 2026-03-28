"""File system tools for agents."""

from pathlib import Path
from typing import Any


class FileSystemTools:
    """
    File system tools for agents.
    
    Provides methods for:
    - Reading files
    - Writing files
    - Listing directories
    - Searching files
    - Creating directories
    """

    def __init__(self, working_dir: str | None = None):
        """
        Initialize file system tools.
        
        Args:
            working_dir: Working directory (default: current directory)
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    def read_file(self, path: str, max_lines: int | None = None) -> dict[str, Any]:
        """
        Read a file and return content.
        
        Args:
            path: File path (absolute or relative to working_dir)
            max_lines: Maximum lines to read (None for all)
            
        Returns:
            Dict with success status and content/error
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            if not file_path.is_file():
                return {"success": False, "error": f"Not a file: {path}"}
            
            # Check file size (limit 1MB for safety)
            if file_path.stat().st_size > 1024 * 1024:
                return {"success": False, "error": "File too large (>1MB)"}
            
            content = file_path.read_text(encoding="utf-8")
            
            if max_lines:
                lines = content.splitlines()[:max_lines]
                content = "\n".join(lines)
                if len(lines) < len(content.splitlines()):
                    content += f"\n... (truncated, showing {max_lines} of {len(lines)} lines)"
            
            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "lines": len(content.splitlines()),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            path: File path (absolute or relative to working_dir)
            content: Content to write
            
        Returns:
            Dict with success status
        """
        try:
            file_path = self._resolve_path(path)
            
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_path.write_text(content, encoding="utf-8")
            
            return {
                "success": True,
                "path": str(file_path),
                "bytes": len(content.encode("utf-8")),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_directory(self, path: str = ".", recursive: bool = False) -> dict[str, Any]:
        """
        List files in a directory.
        
        Args:
            path: Directory path (default: working_dir)
            recursive: Whether to list recursively
            
        Returns:
            Dict with success status and file list
        """
        try:
            dir_path = self._resolve_path(path)
            
            if not dir_path.exists():
                return {"success": False, "error": f"Directory not found: {path}"}
            
            if not dir_path.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}
            
            if recursive:
                files = [str(f.relative_to(dir_path)) for f in dir_path.rglob("*") if f.is_file()]
            else:
                files = [str(f.name) for f in dir_path.iterdir() if f.is_file()]
            
            # Sort files
            files.sort()
            
            return {
                "success": True,
                "path": str(dir_path),
                "files": files,
                "count": len(files),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_directory(self, path: str) -> dict[str, Any]:
        """
        Create a directory.
        
        Args:
            path: Directory path
            
        Returns:
            Dict with success status
        """
        try:
            dir_path = self._resolve_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            return {
                "success": True,
                "path": str(dir_path),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_files(
        self,
        pattern: str,
        path: str = ".",
        recursive: bool = True,
    ) -> dict[str, Any]:
        """
        Search for files matching a pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.md")
            path: Search directory
            recursive: Whether to search recursively
            
        Returns:
            Dict with success status and matching files
        """
        try:
            dir_path = self._resolve_path(path)
            
            if not dir_path.exists():
                return {"success": False, "error": f"Directory not found: {path}"}
            
            if recursive:
                files = [str(f.relative_to(dir_path)) for f in dir_path.rglob(pattern)]
            else:
                files = [str(f.name) for f in dir_path.glob(pattern)]
            
            files.sort()
            
            return {
                "success": True,
                "pattern": pattern,
                "files": files,
                "count": len(files),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        file_path = self._resolve_path(path)
        return file_path.exists() and file_path.is_file()

    def directory_exists(self, path: str) -> bool:
        """Check if a directory exists."""
        dir_path = self._resolve_path(path)
        return dir_path.exists() and dir_path.is_dir()

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to working_dir if not absolute."""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.working_dir / file_path
        return file_path.resolve()


# Global instance for convenience
_fs_tools: FileSystemTools | None = None


def get_fs_tools(working_dir: str | None = None) -> FileSystemTools:
    """Get or create global FileSystemTools instance."""
    global _fs_tools
    if _fs_tools is None:
        _fs_tools = FileSystemTools(working_dir)
    return _fs_tools
