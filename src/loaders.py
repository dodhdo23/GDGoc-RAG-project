"""Document loaders for txt and pdf files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from src.utils import Document


def _load_txt_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return Document(
        text=text,
        metadata={
            "source": path.name,
            "source_path": str(path.resolve()),
            "file_type": "txt",
        },
    )


def _load_pdf_file(path: Path) -> Document:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())

    return Document(
        text="\n\n".join(pages).strip(),
        metadata={
            "source": path.name,
            "source_path": str(path.resolve()),
            "file_type": "pdf",
            "num_pages": len(reader.pages),
        },
    )


def load_documents_from_folder(folder_path: str | Path) -> list[Document]:
    """
    Load .txt and .pdf files from a folder and return Document objects.

    Files that produce empty text are skipped.
    """
    base = Path(folder_path)
    if not base.exists():
        raise FileNotFoundError(f"Folder does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    files: Iterable[Path] = sorted(base.iterdir())
    documents: list[Document] = []

    for path in files:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".txt":
                doc = _load_txt_file(path)
            elif suffix == ".pdf":
                doc = _load_pdf_file(path)
            else:
                continue

            if doc.text.strip():
                documents.append(doc)
        except Exception as exc:
            print(f"[WARN] Failed to load {path.name}: {exc}")

    return documents
