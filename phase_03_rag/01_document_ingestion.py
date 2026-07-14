"""
Phase 3 — Script 01: Document Ingestion
==========================================
Goal: recursive chunking with overlap — the real chunker your ChromaDB
collection will be filled with, instead of hand-typed sample chunks.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console

console = Console()


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for structured documents."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s]


def recursive_chunk(
    text: str,
    source: str,
    max_chars: int = 800,      # ~ 500 tokens; character approx for simplicity here
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Groups sentences into chunks up to max_chars, carrying the last
    `overlap_sentences` sentences forward into the next chunk so context
    isn't lost right at the boundary.
    """
    sentences = split_into_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(Chunk(text=" ".join(current), source=source, chunk_index=len(chunks)))
            # carry the overlap forward
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(Chunk(text=" ".join(current), source=source, chunk_index=len(chunks)))

    return chunks


if __name__ == "__main__":
    sample_doc = """
    TechCorp's Q3 2024 revenue was $2.4 billion, up 18% year-over-year. This growth was driven
    primarily by strong enterprise demand for the company's AI infrastructure products. Net income
    came in at $340 million, down 12% year-over-year due to increased R&D investment. The company
    launched two new products during the quarter: AI Analytics Suite and CloudSync Pro. Employee
    headcount grew 8% to 12,400 across all offices globally. The CEO stated in the earnings call
    that the company is doubling down on AI infrastructure investment heading into 2025. Analysts
    have responded positively to this strategic direction, citing strong market tailwinds.
    """

    console.rule("[bold purple]Recursive chunking with overlap[/bold purple]")
    chunks = recursive_chunk(sample_doc, source="Q3_2024_earnings.pdf", max_chars=250, overlap_sentences=1)

    for c in chunks:
        console.print(f"[bold]Chunk {c.chunk_index}[/bold] ({len(c.text)} chars):")
        console.print(f"  {c.text}\n")

    console.print(f"[dim]Notice consecutive chunks share their boundary sentence — that's the overlap.[/dim]")