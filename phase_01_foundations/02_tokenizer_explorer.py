"""
Phase 1 — Script 02: Tokenizer Explorer
========================================
Goal: See EXACTLY how text becomes tokens.
This is critical for: chunking strategies, cost estimation,
context window management, and debugging weird model behavior.
"""

import tiktoken
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# We use cl100k_base — the tokenizer used by GPT-4 and approximate to Claude's
# Claude uses a similar BPE tokenizer; tiktoken is the best inspection tool available
enc = tiktoken.get_encoding("cl100k_base")

COLORS = ["bold red", "bold blue", "bold green", "bold magenta",
          "bold yellow", "bold cyan", "bold white", "bold orange1"]


def visualize_tokens(text: str, label: str = ""):
    """Break text into tokens and display them color-coded."""
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]

    console.rule(f"[bold purple]{label}[/bold purple]")
    console.print(f'Input: "{text}"')
    console.print(f"Token count: [bold]{len(tokens)}[/bold]\n")

    # Color-coded visualization
    rich_text = Text()
    for i, piece in enumerate(decoded):
        color = COLORS[i % len(COLORS)]
        rich_text.append(f"[{piece}]", style=color)
    console.print(rich_text)

    # Table breakdown
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Token ID", style="dim", width=12)
    table.add_column("Text piece", width=20)
    table.add_column("Bytes", style="dim")

    for token_id, piece in zip(tokens, decoded):
        table.add_row(
            str(token_id),
            repr(piece),
            str(len(piece.encode("utf-8")))
        )
    console.print(table)


# ── Experiment 1: Common English ─────────────────────────────────────────────
visualize_tokens("The cat sat on the mat.", "Experiment 1: Simple English")

# ── Experiment 2: Technical content ──────────────────────────────────────────
visualize_tokens(
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "Experiment 2: Python code (efficient)"
)

# ── Experiment 3: Non-English (costs more!) ───────────────────────────────────
visualize_tokens("人工知能は素晴らしい", "Experiment 3: Japanese (expensive!)")

# ── Experiment 4: The UUID problem ────────────────────────────────────────────
visualize_tokens(
    "user_id: 7f3b9c21-4a8e-4d1f-b2c7-9e0f1a2b3c4d",
    "Experiment 4: UUID (each char = its own token)"
)

# ── Experiment 5: Cost estimator ─────────────────────────────────────────────
console.rule("[bold green]Cost Estimator[/bold green]")

documents = [
    "Annual report summary for Q4 2024. Revenue grew by 23% year-over-year.",
    "The mitochondria is the powerhouse of the cell.",
    "AI systems are transforming how we build software in 2025.",
]

# Claude Sonnet pricing (approximate as of 2025)
INPUT_COST_PER_MILLION  = 3.00   # USD
OUTPUT_COST_PER_MILLION = 15.00  # USD

table = Table(box=box.SIMPLE, header_style="bold")
table.add_column("Document", width=50)
table.add_column("Tokens", justify="right")
table.add_column("Cost (input)", justify="right")

for doc in documents:
    n = len(enc.encode(doc))
    cost = (n / 1_000_000) * INPUT_COST_PER_MILLION
    table.add_row(doc[:50], str(n), f"${cost:.6f}")

console.print(table)
console.print("[dim]Tip: For a 1000-document RAG corpus, estimate total ingestion tokens carefully.[/dim]")