"""Local filesystem implementation of the raw document store."""

from __future__ import annotations

from pathlib import Path


class LocalRawDocumentStore:
    """Store raw filing bytes on the local filesystem."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)

    def exists(self, relative_path: str) -> bool:
        """Return whether the target file already exists."""

        return self.resolve(relative_path).exists()

    def save(self, relative_path: str, content: bytes) -> Path:
        """Persist bytes to the target path, creating parents as needed."""

        destination = self.resolve(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return destination.resolve()

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative object path under the store root."""

        path = Path(relative_path)
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()
        return (self._root_dir / path).resolve()
