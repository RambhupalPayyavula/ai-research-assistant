"""
Phase 5 — Script 01: Multi-Agent Workflow with LangGraph
============================================================
Goal: split Phase 4's single agent into three specialized roles —
Retriever, Fact-Checker, Summarizer — coordinated by a supervisor graph.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import TypedDict, List
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END
from core.llm_client import LLMClient
from rich.console import Console
from rich.panel import Panel

console = Console()
llm = LLMClient()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="research_assistant_docs", embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"}
)


# ── Shared state — flows through every node ──────────────────────────────
class AgentState(TypedDict):
    question: str
    retrieved_chunks: List[str]
    draft_answer: str
    fact_check_passed: bool
    fact_check_notes: str
    final_answer: str
    retrieval_attempts: int


# ── Node 1: Retriever ─────────────────────────────────────────────────────
def retrieve_node(state: AgentState) -> AgentState:
    results = collection.query(query_texts=[state["question"]], n_results=5)
    chunks = results["documents"][0]
    console.print(f"  [teal]retrieve_node:[/teal] found {len(chunks)} chunks")
    return {**state, "retrieved_chunks": chunks, "retrieval_attempts": state.get("retrieval_attempts", 0) + 1}


# ── Node 2: Draft answer (a focused, narrow generation step) ────────────
def draft_node(state: AgentState) -> AgentState:
    context = "\n---\n".join(state["retrieved_chunks"])
    system = f"""Answer the question using ONLY this context. Be concise.
CONTEXT:
{context}"""
    draft = llm.simple(system=system, user_message=state["question"], temperature=0.0)
    console.print(f"  [amber]draft_node:[/amber] drafted an answer")
    return {**state, "draft_answer": draft}


# ── Node 3: Summarizer — runs BEFORE fact-check, so fact-check verifies the polished output──────────────────
def summarize_node(state: AgentState) -> AgentState:
    system = """You are an editor. Rewrite the following answer to be clear,
    concise, plain prose (no markdown headers or bold), with citations like [Source N].
    You may ONLY use facts already present in the answer below — the source chunks
    are provided ONLY so you can determine correct [Source N] numbering. Do NOT
    add any fact, number, or claim that isn't already in the answer you're editing."""
    context = "\n---\n".join(state["retrieved_chunks"])
    user_msg = f"Source chunks:\n{context}\n\nAnswer to clean up:\n{state['draft_answer']}"
    final = llm.simple(system=system, user_message=user_msg, temperature=0.0)
    console.print(f"  [purple]summarize_node:[/purple] finalized with citation cleanup")
    return {**state, "final_answer": final}

# ── Node 4:   Fact-checker — a SEPARATE, adversarial pass ───
def fact_check_node(state: AgentState) -> AgentState:
    context = "\n---\n".join(state["retrieved_chunks"])
    system = f"""You are a skeptical fact-checker. Verify whether EVERY claim in the
    answer below is directly supported by the source context. Be strict.

    SOURCE CONTEXT:
    {context}

    ANSWER TO VERIFY:
    {state['final_answer']}

    Respond with exactly one line: "PASS" if every claim is supported, or "FAIL: <reason>" if not."""
    result = llm.simple(system=system, user_message="Verify the answer.", temperature=0.0)
    passed = result.strip().upper().startswith("PASS")
    console.print(f"  [blue]fact_check_node:[/blue] {'PASS' if passed else 'FAIL'} \u2014 {result[:80]}")
    return {**state, "fact_check_passed": passed, "fact_check_notes": result}


# ── Conditional edge: did fact-checking pass, and have we retried too much? ─
def route_after_fact_check(state: AgentState) -> str:
    if state["fact_check_passed"]:
        return "end"
    if state["retrieval_attempts"] >= 2:
        return "end"
    return "retrieve"



# ── Build the graph ─────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("draft", draft_node)
graph.add_node("summarize", summarize_node)
graph.add_node("fact_check", fact_check_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "draft")
graph.add_edge("draft", "summarize")
graph.add_edge("summarize", "fact_check")          
graph.add_conditional_edges("fact_check", route_after_fact_check, {
    "retrieve": "retrieve",
    "end": END,
})

app = graph.compile()


if __name__ == "__main__":
    questions = [
        "What was TechCorp's revenue and what products did they launch?",
    ]
    for q in questions:
        console.rule(f"[bold]{q}[/bold]")
        result = app.invoke({"question": q, "retrieval_attempts": 0})
        console.print(Panel(result["final_answer"], title="Final Answer (multi-agent)", border_style="cyan"))
        console.print(f"[dim]Fact-check attempts: {result['retrieval_attempts']} | Passed: {result['fact_check_passed']}[/dim]")