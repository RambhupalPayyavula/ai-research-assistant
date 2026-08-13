"""
Phase 4 — Script 02: ReAct Agent with Document Retrieval
===========================================================
Goal: a real agentic loop that decides whether to retrieve from documents,
possibly retrieves more than once, and only then answers — using Phase 3's
retrieve() function as an actual tool the model can choose to call.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.utils import embedding_functions
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

SEARCH_TOOL = {
    "name": "search_documents",
    "description": "Search the ingested document corpus for chunks relevant to a query. "
                    "Use this when the question may be answered by the uploaded documents. "
                    "You may call this more than once with different queries if the first "
                    "search doesn't return what you need. Do NOT use for general knowledge "
                    "questions unrelated to the documents.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    },
}


def search_documents(query: str, top_k: int = 5) -> str:
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return "No relevant documents found."

    formatted = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source", "unknown")
        formatted.append(f"[Source {i}] ({source})\n{doc}")

    return "\n\n---\n\n".join(formatted)



SYSTEM = """You are a research assistant with access to a document search tool.
For each user question:
1. Decide whether you need to search documents to answer accurately.
2. If you search and the results are insufficient, you may search again with a refined query.
3. Once you have enough information, give a final answer citing what you found using [Source N] notation.
4. If the documents don't contain the answer after searching, say so explicitly — never guess.

FORMAT RULES:
- Do NOT use markdown headers (##) or bold formatting (**text**) in your final answer.
- Write in plain prose, short paragraphs only.
- Every factual claim must cite its source using [Source N] notation.
"""

def run_agent(user_message: str, max_steps: int = 4):
    messages = [{"role": "user", "content": user_message}]
    source_counter = {"n": 0}  # shared across all searches in this conversation
    def search_documents_numbered(query: str, top_k: int = 5, min_relevance: float = 0.25) -> str:
        results = collection.query(query_texts=[query], n_results=top_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        
        formatted = []
        for doc, meta, dist in zip(docs, metas, dists):
            if (1 - dist) < min_relevance:
                continue
            source_counter["n"] += 1
            source = meta.get("source", "unknown")
            formatted.append(f"[Source {source_counter['n']}] ({source})\n{doc}")
        return "\n\n---\n\n".join(formatted) if formatted else "No sufficiently relevant documents found."

    for step in range(max_steps):
        response = llm.client.messages.create(
            model=llm.model, max_tokens=1024, system=SYSTEM, tools=[SEARCH_TOOL], messages=messages
        )

        if response.stop_reason != "tool_use":
            return response.content[0].text

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use_block in tool_use_blocks:
            query = tool_use_block.input["query"]
            console.print(f"  [dim]step {step+1}: searching for \"{query}\"[/dim]")
            result = search_documents_numbered(query)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "Reached max reasoning steps without a final answer."

if __name__ == "__main__":
    questions = [
        "What was TechCorp's financial performance and what are they investing in next?",
        "What is TechCorp's mission statement?",  # likely triggers a refined second search or refusal
    ]
    for q in questions:
        console.rule(f"[bold]{q}[/bold]")
        answer = run_agent(q)
        console.print(Panel(answer, title="Agent's Final Answer", border_style="cyan"))