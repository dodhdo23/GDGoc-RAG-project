"""Shared utility types and helpers for the retrieval baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any


@dataclass
class Document:
    """Represents a loaded source document."""

    text: str
    metadata: dict[str, Any]


@dataclass
class Chunk:
    """Represents a chunk split from a document."""

    text: str
    metadata: dict[str, Any]


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON data to disk with UTF-8 encoding."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """Load JSON data from disk."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Convert chunk dataclass into a JSON-serializable dictionary."""
    return asdict(chunk)


def chunk_from_dict(data: dict[str, Any]) -> Chunk:
    """Convert JSON dictionary back into Chunk."""
    return Chunk(text=data["text"], metadata=data["metadata"])
