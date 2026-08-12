"""
Phase 6 — Script 02: Run Evaluation with LLM-as-Judge
=========================================================
Goal: run the golden set through the RAG pipeline, score faithfulness and
refusal-correctness, and produce a report you can compare across changes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import chromadb
from chromadb.utils import embedding_functions
from core.llm_client import LLMClient
from rich.console import Console
from rich.table import Table

console = Console()
llm = LLMClient()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="research_assistant_docs", embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"}
)

GOLDEN_SET = [
    {"question": "What was TechCorp's Q3 2024 revenue?", "expected_answerable": True},
    {"question": "How many employees does TechCorp have?", "expected_answerable": True},
    {"question": "What was TechCorp's financial performance and what are they investing in next?", "expected_answerable": True},
    {"question": "What products did TechCorp launch and how did headcount change?", "expected_answerable": True},
    {"question": "What is TechCorp's stock price today?", "expected_answerable": False},
    {"question": "What is TechCorp's mission statement?", "expected_answerable": False},
    {"question": "Who is TechCorp's main competitor?", "expected_answerable": False},
    {"question": "Tell me about the company", "expected_answerable": True},
    {"question": "What did the CEO say about 2025 plans?", "expected_answerable": True},
]


def retrieve(question: str, top_k: int = 5):
    results = collection.query(query_texts=[question], n_results=top_k)
    return results["documents"][0]


def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n---\n".join(chunks)
    system = f"""Answer using ONLY this context. If not in the context, say
"This information is not available in the provided documents."
CONTEXT:
{context}"""
    return llm.simple(system=system, user_message=question, temperature=0.0)

def parse_judge_json(raw: str) -> dict:
    """Try direct parse first, then strip common markdown wrapping as a fallback."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"score": None, "reason": f"judge output not parseable even after cleanup: {raw[:100]}"}
    
def judge_faithfulness(answer: str, chunks: list[str]) -> dict:
    context = "\n---\n".join(chunks)
    system = """You are an evaluation judge. Score the answer's faithfulness to the context
    on a scale of 0.0 to 1.0, where 1.0 means every claim is fully supported and
    0.0 means the answer is entirely unsupported/hallucinated.

    Respond with ONLY raw JSON, nothing else — no markdown code fences, no ```json
    tags, no explanation before or after. Your entire response must be parseable
    by json.loads() with no modification. Example of correct output:
    {"score": 0.9, "reason": "one sentence"}"""
    user_msg = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    result = llm.simple(system=system, user_message=user_msg, temperature=0.0)
    return parse_judge_json(result)


def is_refusal(answer: str) -> bool:
    return "not available in the provided documents" in answer.lower()


def run_evaluation():
    results = []
    for item in GOLDEN_SET:
        q = item["question"]
        chunks = retrieve(q)
        answer = generate_answer(q, chunks)
        refused = is_refusal(answer)

        refusal_correct = refused != item["expected_answerable"]  # refused XOR should-answer = correct
        faithfulness = judge_faithfulness(answer, chunks) if not refused else {"score": 1.0, "reason": "N/A (refusal)"}

        results.append({
            "question": q,
            "expected_answerable": item["expected_answerable"],
            "refused": refused,
            "refusal_correct": refusal_correct,
            "faithfulness_score": faithfulness.get("score"),
            "answer": answer,
        })
    return results


def print_report(results):
    table = Table(title="Phase 6 Evaluation Report", show_header=True, header_style="bold")
    table.add_column("Question", width=40)
    table.add_column("Refusal OK?", width=11)
    table.add_column("Faithfulness", width=12)
    table.add_column("Answer preview", width=40)

    for r in results:
        refusal_mark = "[green]YES[/green]" if r["refusal_correct"] else "[red]NO[/red]"
        faith = f"{r['faithfulness_score']:.2f}" if r["faithfulness_score"] is not None else "N/A"
        table.add_row(r["question"][:40], refusal_mark, faith, r["answer"][:40])

    console.print(table)

    n = len(results)
    refusal_acc = sum(r["refusal_correct"] for r in results) / n
    faith_scores = [r["faithfulness_score"] for r in results if r["faithfulness_score"] is not None]
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0

    console.print(f"\n[bold]Refusal accuracy:[/bold] {refusal_acc:.0%}")
    console.print(f"[bold]Average faithfulness:[/bold] {avg_faith:.2f}")
    console.print(f"[bold]Total eval cost:[/bold] ${llm.total_cost_usd:.4f}")


if __name__ == "__main__":
    results = run_evaluation()
    print_report(results)

    # Save for regression comparison later
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print("\n[dim]Saved to eval_results.json for future regression comparison.[/dim]")