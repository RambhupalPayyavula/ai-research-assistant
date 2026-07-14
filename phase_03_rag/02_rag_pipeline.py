"""
Phase 3 — Script 02: The Full RAG Pipeline
=============================================
Goal: wire together Phase 1 (grounded prompting) + Phase 2 (ChromaDB retrieval)
+ Phase 3 (chunking) into one working question-answering system.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # allow importing core/

import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.panel import Panel

from core.llm_client import LLMClient
from core.prompts import build_grounded_prompt, RetrievedChunk

console = Console()

# ── Set up ChromaDB (same collection style as Phase 2) ──────────────────────
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="research_assistant_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

llm = LLMClient()


def retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Phase 2's retrieval, wrapped to return the RetrievedChunk type Phase 1 expects."""
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = []
    for doc, meta, dist, doc_id in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0]
    ):
        chunks.append(RetrievedChunk(text=doc, source=meta.get("source", "unknown"), chunk_id=doc_id, relevance_score=1 - dist))
    return chunks


def answer_question(question: str, top_k: int = 5) -> dict:
    """The complete RAG loop: retrieve -> ground -> generate."""
    retrieved = retrieve(question, top_k=top_k)

    if not retrieved:
        return {"answer": "No documents have been ingested yet.", "sources": []}

    system_prompt = build_grounded_prompt(retrieved)
    answer = llm.simple(system=system_prompt, user_message=question, temperature=0.0)

    return {
        "answer": answer,
        "sources": [f"{c.source} (relevance: {c.relevance_score:.2f})" for c in retrieved],
    }


if __name__ == "__main__":
    # Assumes Phase 2's script already seeded the collection — re-run it if empty
    test_questions = [
        "What was TechCorp's revenue and how did it change?",
        "What products did TechCorp launch?",
        "What is TechCorp's stock price today?",  # should trigger refusal — not in context
    ]

    for q in test_questions:
        console.rule(f"[bold]{q}[/bold]")
        result = answer_question(q)
        console.print(Panel(result["answer"], title="Answer", border_style="cyan"))
        console.print(f"[dim]Sources: {', '.join(result['sources'])}[/dim]\n")