"""
Phase 2 — Script 02: Semantic Search with ChromaDB
=====================================================
Goal: stand up a real vector database, store document chunks with metadata,
and run semantic search — the exact retrieval mechanism Phase 3's RAG pipeline
will build on top of.
"""

import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.table import Table

console = Console()

# ── Set up ChromaDB with the SAME embedding model as script 01 ──────────────
# Using ChromaDB's built-in wrapper keeps embedding + storage consistent.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./chroma_db")  # persists to disk, gitignored

collection = client.get_or_create_collection(
    name="research_assistant_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # explicitly choose cosine similarity
)

# ── Seed with sample document chunks + metadata ──────────────────────────────
sample_chunks = [
    {
        "id": "chunk_001",
        "text": "TechCorp's Q3 2024 revenue was $2.4B, up 18% year-over-year.",
        "metadata": {"source": "Q3_2024_earnings.pdf", "category": "finance", "page": 1},
    },
    {
        "id": "chunk_002",
        "text": "TechCorp launched two new products: AI Analytics Suite and CloudSync Pro.",
        "metadata": {"source": "Q3_2024_earnings.pdf", "category": "product", "page": 2},
    },
    {
        "id": "chunk_003",
        "text": "Employee headcount grew 8% to 12,400 across all TechCorp offices.",
        "metadata": {"source": "Q3_2024_earnings.pdf", "category": "hr", "page": 3},
    },
    {
        "id": "chunk_004",
        "text": "The CEO stated the company is doubling down on AI infrastructure investment for 2025.",
        "metadata": {"source": "Q3_2024_earnings.pdf", "category": "strategy", "page": 4},
    },
    {
        "id": "chunk_005",
        "text": "Golden retrievers are known for being friendly and good with children.",
        "metadata": {"source": "unrelated_doc.pdf", "category": "misc", "page": 1},
    },
]

console.rule("[bold purple]Seeding ChromaDB collection[/bold purple]")
collection.upsert(
    ids=[c["id"] for c in sample_chunks],
    documents=[c["text"] for c in sample_chunks],
    metadatas=[c["metadata"] for c in sample_chunks],
)
console.print(f"Stored {len(sample_chunks)} chunks in ChromaDB.\n")


def search(query: str, top_k: int = 3, where_filter: dict = None):
    """The core retrieval function Phase 3's RAG pipeline will call directly."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter,  # metadata filtering — None means no filter
    )
    return results


def print_results(query, results):
    table = Table(title=f'Query: "{query}"', show_header=True, header_style="bold")
    table.add_column("Rank", width=5)
    table.add_column("Distance", width=10)
    table.add_column("Text", width=55)
    table.add_column("Source metadata", width=25)

    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    for i, (doc, dist, meta) in enumerate(zip(docs, dists, metas), 1):
        table.add_row(str(i), f"{dist:.3f}", doc, f"{meta['category']} / p.{meta['page']}")
    console.print(table)


# ── Experiment 1: Plain semantic search ──────────────────────────────────────
console.rule("[bold teal]Experiment 1: Semantic search — no keyword overlap needed[/bold teal]")
results = search("How much money did the company make?")
print_results("How much money did the company make?", results)
console.print(
    "\n[dim]Notice: the query says 'money' and 'company' — the top result says 'revenue' "
    "and 'TechCorp'. This is semantic matching, not keyword matching.[/dim]\n"
)

# ── Experiment 2: The irrelevant chunk gets correctly ranked low ────────────
console.rule("[bold blue]Experiment 2: Unrelated content should rank last[/bold blue]")
results = search("What are TechCorp's growth plans?", top_k=5)
print_results("What are TechCorp's growth plans?", results)
console.print("\n[dim]The golden retriever chunk should rank last — high distance = low relevance.[/dim]\n")

# ── Experiment 3: Metadata filtering — scope the search ─────────────────────
console.rule("[bold amber]Experiment 3: Metadata filtering[/bold amber]")
results = search("Tell me about the company", top_k=5, where_filter={"category": "finance"})
print_results("Tell me about the company (finance only)", results)
console.print("\n[dim]Filtered to ONLY the 'finance' category chunk, even though other chunks might be semantically closer.[/dim]")