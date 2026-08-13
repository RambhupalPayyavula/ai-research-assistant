"""
core/document_loader.py
==========================
Shared document loading + chunking logic. Used by both the standalone
ingestion script (phase_03_rag/04_multi_format_ingestion.py) and the
FastAPI /upload endpoint — single source of truth, no duplication.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument

MAX_FILE_SIZE_MB = 20
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def recursive_chunk(text: str, source: str, max_chars: int = 800, overlap_sentences: int = 1) -> list[Chunk]:
    sentences = split_into_sentences(text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(Chunk(" ".join(current), source, len(chunks)))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(Chunk(" ".join(current), source, len(chunks)))
    return chunks


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_md(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = re.sub(r"#{1,6}\s*", "", raw)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"`(.+?)`", r"\1", cleaned)
    return cleaned


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt, ".md": load_md}


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported format: {ext}")
    return LOADERS[ext](path)


def validate_file(path: Path) -> tuple[bool, str]:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported extension: {path.suffix}"
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)"
    return True, "OK"