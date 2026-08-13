"""
Phase 6 — Script 03: Real RAGAS Evaluation
==============================================
Goal: replace the hand-built judge with the actual industry-standard
RAGAS framework, covering all four core metrics — not just faithfulness.

Run this with venv_eval activated (Python 3.11/3.12), not the main venv.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic
from rich.console import Console
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()
console = Console()

# ── RAGAS needs an LLM to act as the judge — wire it to Claude ──────────
judge_llm = LangchainLLMWrapper(ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
))
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="research_assistant_docs", embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"}
)

from core.llm_client import LLMClient
llm = LLMClient()

GOLDEN_SET = [
    {"question": "What was TechCorp's Q3 2024 revenue?", "ground_truth": "TechCorp's Q3 2024 revenue was $2.4 billion, up 18% year-over-year."},
    {"question": "What products did TechCorp launch?", "ground_truth": "TechCorp launched AI Analytics Suite and CloudSync Pro."},
    {"question": "How many employees does TechCorp have?", "ground_truth": "TechCorp has 12,400 employees, up 8%."},
]


def retrieve(question: str, top_k: int = 5) -> list[str]:
    results = collection.query(query_texts=[question], n_results=top_k)
    return results["documents"][0]


def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n---\n".join(chunks)
    system = f"Answer using ONLY this context.\nCONTEXT:\n{context}"
    return llm.simple(system=system, user_message=question, temperature=0.0)


def build_ragas_dataset():
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in GOLDEN_SET:
        chunks = retrieve(item["question"])
        answer = generate_answer(item["question"], chunks)
        rows["question"].append(item["question"])
        rows["answer"].append(answer)
        rows["contexts"].append(chunks)
        rows["ground_truth"].append(item["ground_truth"])
    return Dataset.from_dict(rows)


if __name__ == "__main__":
    console.print("[dim]Building dataset by running the RAG pipeline on the golden set...[/dim]")
    dataset = build_ragas_dataset()

    console.print("[dim]Scoring with RAGAS (faithfulness, relevancy, precision, recall)...[/dim]")
    result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=ragas_embeddings,   
)

    console.print("\n[bold]RAGAS Evaluation Results:[/bold]")
    console.print(result)

    df = result.to_pandas()
    df.to_csv("ragas_eval_results.csv", index=False)
    console.print("\n[dim]Saved to ragas_eval_results.csv[/dim]")