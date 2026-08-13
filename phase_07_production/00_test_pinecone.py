"""
phase_07_production/00_test_pinecone.py
==========================================
Isolated smoke test for the Pinecone + LangChain backend, before wiring
it into the full application. Run this FIRST — small, fast feedback loop.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["VECTOR_STORE"] = "pinecone"  # force pinecone for this test

from core.vector_store import get_vector_store
from rich.console import Console

console = Console()

store = get_vector_store()
console.print(f"[green]Backend initialized:[/green] {type(store).__name__}")

console.print("[dim]Upserting test documents...[/dim]")
store.upsert(
    ids=["test_1", "test_2", "test_3"],
    documents=[
        "TechCorp's Q3 2024 revenue was $2.4 billion, up 18% year-over-year.",
        "TechCorp launched two new products: AI Analytics Suite and CloudSync Pro.",
        "Golden retrievers are friendly dogs, good with children.",
    ],
    metadatas=[
        {"source": "test.pdf", "chunk_index": 0},
        {"source": "test.pdf", "chunk_index": 1},
        {"source": "unrelated.pdf", "chunk_index": 0},
    ],
)

console.print("[dim]Waiting for Pinecone indexing to settle...[/dim]")
import time
time.sleep(5)  # Pinecone upserts are eventually consistent, not instant

console.print("\n[bold]Query: 'How much money did the company make?'[/bold]")
results = store.query("How much money did the company make?", top_k=3, min_relevance=0.0)
for r in results:
    print(f"[{r.relevance_score:.3f}] {r.text[:60]}")

console.print("\n[bold green]If the revenue chunk ranked first, the Pinecone + LangChain backend is working correctly.[/bold green]")