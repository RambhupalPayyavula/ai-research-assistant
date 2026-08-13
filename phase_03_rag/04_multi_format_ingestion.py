"""
Phase 3 — Script 04: Multi-Format Document Ingestion
=========================================================
Goal: support the document types people actually deal with day to day —
PDF, DOCX, TXT, Markdown — behind one unified ingestion pipeline.
Extending to a new format later means adding one small function here,
nothing downstream needs to change.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from dataclasses import dataclass
from pypdf import PdfReader
from docx import Document as DocxDocument
import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.progress import track
from rich.table import Table

from core.document_loader import (
    Chunk, recursive_chunk, load_document, validate_file,
    SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB,
)

console = Console()

# ── Format-specific text extractors — each returns plain text, nothing else ──
def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_md(path: Path) -> str:
    # Markdown is just text for our purposes — we don't need to render it,
    # only extract the words. Strip the most common markdown syntax for cleaner chunks.
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = re.sub(r"#{1,6}\s*", "", raw)          # headers
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)  # bold
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)      # italic
    cleaned = re.sub(r"`(.+?)`", r"\1", cleaned)        # inline code
    return cleaned


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".md": load_md,
}


# ── The unified ingestion pipeline ────────────────────────────────────────
def ingest_folder(folder_path: str, collection):
    folder = Path(folder_path)
    all_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not all_files:
        console.print(f"[yellow]No supported files found in {folder_path}[/yellow]")
        console.print(f"[dim]Supported: {', '.join(SUPPORTED_EXTENSIONS)}[/dim]")
        return

    results = Table(title="Ingestion Results", show_header=True, header_style="bold")
    results.add_column("File")
    results.add_column("Type")
    results.add_column("Status")
    results.add_column("Chunks")

    total_chunks = 0
    for file_path in track(all_files, description="Ingesting..."):
        ok, reason = validate_file(file_path)
        if not ok:
            results.add_row(file_path.name, file_path.suffix, f"[red]REJECTED: {reason}[/red]", "-")
            continue

        try:
            text = load_document(file_path)
            if not text.strip():
                results.add_row(file_path.name, file_path.suffix, "[yellow]EMPTY (no extractable text)[/yellow]", "-")
                continue

            chunks = recursive_chunk(text, source=file_path.name)
            collection.upsert(
                ids=[f"{file_path.stem}_chunk_{c.chunk_index}" for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[{"source": c.source, "chunk_index": c.chunk_index, "file_type": file_path.suffix} for c in chunks],
            )
            results.add_row(file_path.name, file_path.suffix, "[green]OK[/green]", str(len(chunks)))
            total_chunks += len(chunks)

        except Exception as e:
            # ONE bad file never crashes the whole batch — logged and skipped
            results.add_row(file_path.name, file_path.suffix, f"[red]FAILED: {str(e)[:40]}[/red]", "-")

    console.print(results)
    console.print(f"\n[bold green]Total: {total_chunks} chunks ingested.[/bold green]")
    console.print(f"[dim]Collection now has {collection.count()} total chunks.[/dim]")


if __name__ == "__main__":
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="research_assistant_docs", embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"}
    )

    console.rule("[bold purple]Multi-Format Document Ingestion[/bold purple]")
    ingest_folder("./sample_documents", collection)