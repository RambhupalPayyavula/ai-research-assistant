"""
Phase 1 — Script 01: First API Call
====================================
Goal: Understand the raw structure of an LLM API call.
Every future agent, RAG pipeline, and tool call builds on exactly this.
"""

import os
from dotenv import load_dotenv
import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

load_dotenv()
console = Console()

# ── The client ──────────────────────────────────────────────────────────────
# This is your gateway to Claude. One instance, reuse it everywhere.
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_claude(system: str, user: str, model: str = "claude-sonnet-4-5") -> str:
    """
    The fundamental unit of all GenAI engineering.
    
    Parameters
    ----------
    system : str   — The persona, rules, and constraints for the model.
                     Think of this as the job description.
    user   : str   — The actual request from the user (or from your pipeline).
    model  : str   — Which Claude variant to use.
    
    Returns
    -------
    str — The model's response text.
    """
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": user}
        ]
    )
    return message.content[0].text


# ── Experiment 1: Bare minimum call ─────────────────────────────────────────
console.rule("[bold purple]Experiment 1: Bare minimum call[/bold purple]")

response = call_claude(
    system="You are a helpful assistant.",
    user="What is the capital of France?"
)
console.print(Panel(response, title="Response", border_style="purple"))


# ── Experiment 2: System prompt shapes everything ───────────────────────────
console.rule("[bold cyan]Experiment 2: Same question, radically different system prompts[/bold cyan]")

question = "What is machine learning?"

personas = [
    {
        "name": "The Professor",
        "system": "You are a university professor. Explain concepts formally with precise terminology. Always include a formal definition first."
    },
    {
        "name": "The 5-year-old explainer",
        "system": "You explain everything as if the person is 5 years old. Use simple words, fun analogies, and short sentences."
    },
    {
        "name": "The Bullet-point Engineer",
        "system": "You are a senior engineer. Respond ONLY in bullet points. Maximum 5 bullets. Each bullet under 15 words. No preamble."
    }
]

for persona in personas:
    response = call_claude(system=persona["system"], user=question)
    console.print(Panel(
        response,
        title=f"[bold]{persona['name']}[/bold]",
        border_style="cyan"
    ))


# ── Experiment 3: Inspect the full API response object ──────────────────────
console.rule("[bold blue]Experiment 3: The full response object[/bold blue]")

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=256,
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Name 3 primary colors."}]
)

console.print(f"[bold]Model:[/bold]          {message.model}")
console.print(f"[bold]Stop reason:[/bold]    {message.stop_reason}")
console.print(f"[bold]Input tokens:[/bold]   {message.usage.input_tokens}")
console.print(f"[bold]Output tokens:[/bold]  {message.usage.output_tokens}")
console.print(f"[bold]Total tokens:[/bold]   {message.usage.input_tokens + message.usage.output_tokens}")
console.print(f"[bold]Response:[/bold]       {message.content[0].text}")

# INSIGHT: input_tokens is what you pay for context.
# output_tokens is what you pay for generation.
# On Claude Sonnet: input is cheaper than output — always minimize output length for pipelines.