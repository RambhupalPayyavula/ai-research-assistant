"""
Phase 2 — Script 01: Embed and Visualize
==========================================
Goal: generate real embeddings and SEE the semantic geometry with your own eyes,
before ChromaDB hides the mechanics behind a clean API in script 02.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# all-MiniLM-L6-v2: small (80MB), fast, free, runs locally on your M1.
# Produces 384-dimensional vectors — much smaller than Claude's internal
# embeddings, but plenty for demonstrating and even production small-scale RAG.
console.print("[dim]Loading embedding model (first run downloads ~80MB)...[/dim]")
model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Experiment 1: Embed a handful of related and unrelated sentences ────────
sentences = [
    "The cat sat on the mat.",
    "A kitten was resting on the rug.",
    "The stock market rallied today.",
    "Quarterly earnings exceeded expectations.",
    "Dogs are loyal companions.",
]

console.rule("[bold purple]Experiment 1: Embedding real sentences[/bold purple]")
embeddings = model.encode(sentences)
console.print(f"Embedding shape: {embeddings.shape}  (5 sentences x 384 dimensions)\n")

# ── Experiment 2: Similarity matrix — see the semantic clusters ─────────────
console.rule("[bold teal]Experiment 2: Pairwise similarity matrix[/bold teal]")
table = Table(show_header=True, header_style="bold")
table.add_column("", style="dim")
for i in range(len(sentences)):
    table.add_column(f"S{i+1}", justify="center")

for i, row_sent in enumerate(sentences):
    row = [f"S{i+1}: {row_sent[:22]}..."]
    for j in range(len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        row.append(f"{sim:.2f}")
    table.add_row(*row)

console.print(table)
console.print(
    "\n[dim]Notice: S1/S2 (cat/kitten) and S3/S4 (stock market) cluster high with each other,"
    "\nbut low with unrelated sentences — this is semantic geometry, not keyword matching.[/dim]"
)

# ── Experiment 3: The analogy test (recreating the King/Queen example) ──────
console.rule("[bold blue]Experiment 3: Vector arithmetic on real embeddings[/bold blue]")
words = ["king", "man", "woman", "queen"]
word_vecs = model.encode(words)
king, man, woman, queen = word_vecs

result_vec = king - man + woman
sim_to_queen = cosine_similarity(result_vec, queen)
console.print(f'vector("king") - vector("man") + vector("woman")  vs  vector("queen")')
console.print(f"Cosine similarity: [bold]{sim_to_queen:.3f}[/bold]  (1.0 = identical, 0 = unrelated)")

# ── Experiment 4: Cost/speed reality check ───────────────────────────────────
console.rule("[bold green]Experiment 4: Why this matters at scale[/bold green]")
import time
big_batch = sentences * 200  # simulate 1000 chunks
start = time.time()
_ = model.encode(big_batch, show_progress_bar=False)
elapsed = time.time() - start
console.print(f"Embedded {len(big_batch)} chunks in {elapsed:.2f}s locally, at zero API cost.")
console.print("[dim]This is why local embedding models matter for iterating fast during development.[/dim]")