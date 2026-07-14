"""
Phase 1 — Script 03: Prompt Engineering Lab
=============================================
Goal: Internalize the 6 core prompt patterns every GenAI engineer must know.
These patterns are the building blocks of every pipeline we build.
"""

import os
from dotenv import load_dotenv
import anthropic
from rich.console import Console
from rich.panel import Panel

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
console = Console()


def claude(system: str, user: str) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return msg.content[0].text


def show(title: str, response: str, color: str = "purple"):
    console.print(Panel(response, title=title, border_style=color))


# ── Pattern 1: ROLE + TASK + FORMAT ──────────────────────────────────────────
console.rule("[bold]Pattern 1: Role + Task + Format[/bold]")
# The most reliable prompt structure. Never skip any of the three parts.

response = claude(
    system="""You are a senior software architect with 15 years of experience.
Your task is to review code and identify issues.
Always respond in this exact format:
ISSUES: <numbered list of problems>
SEVERITY: <Critical / High / Medium / Low>
FIX: <one-line fix for the most critical issue>""",
    user="""
def get_user(id):
    query = "SELECT * FROM users WHERE id = " + id
    return db.execute(query)
"""
)
show("Pattern 1: Role + Task + Format", response, "red")


# ── Pattern 2: FEW-SHOT (show, don't tell) ───────────────────────────────────
console.rule("[bold]Pattern 2: Few-shot examples[/bold]")
# Give 2-3 examples of input→output. The model learns the pattern, not just the rule.

response = claude(
    system="""You classify customer support tickets into categories.

Examples:
Input: "My payment failed three times but I was still charged"
Output: {"category": "billing", "urgency": "high", "sentiment": "frustrated"}

Input: "How do I export my data to CSV?"
Output: {"category": "how-to", "urgency": "low", "sentiment": "neutral"}

Input: "I love the new dashboard design!"
Output: {"category": "feedback", "urgency": "low", "sentiment": "positive"}

Now classify the following ticket. Respond ONLY with valid JSON, nothing else.""",
    user="I've been waiting 3 weeks for my refund and nobody is responding to my emails!!!"
)
show("Pattern 2: Few-shot — consistent structured JSON", response, "yellow")


# ── Pattern 3: CHAIN OF THOUGHT ──────────────────────────────────────────────
console.rule("[bold]Pattern 3: Chain of Thought[/bold]")
# Forcing the model to reason step-by-step dramatically improves accuracy
# on math, logic, and multi-step problems.

response = claude(
    system="""Solve problems by thinking step by step.
Show your reasoning explicitly before giving the final answer.
Format: THINKING: <your reasoning> | ANSWER: <final answer>""",
    user="""A store sells apples for $0.50 each and oranges for $0.75 each.
Sarah buys some apples and oranges. She spends exactly $5.25 total
and buys 9 pieces of fruit. How many of each did she buy?"""
)
show("Pattern 3: Chain of Thought reasoning", response, "green")


# ── Pattern 4: OUTPUT SCHEMA ENFORCEMENT ─────────────────────────────────────
console.rule("[bold]Pattern 4: Strict output schema[/bold]")
# In production pipelines, you MUST parse the model's output.
# Use XML tags or JSON — they are far more reliable than markdown headers.

response = claude(
    system="""You are a document analyzer. Extract structured information.
ALWAYS respond using EXACTLY this XML structure, nothing before or after:
<analysis>
  <topic>main topic in 3 words max</topic>
  <sentiment>positive | negative | neutral</sentiment>
  <key_facts>
    <fact>first key fact</fact>
    <fact>second key fact</fact>
    <fact>third key fact</fact>
  </key_facts>
  <confidence>0.0 to 1.0</confidence>
</analysis>""",
    user="""Global temperatures have risen by 1.1°C since pre-industrial times.
The last decade was the hottest on record. Arctic sea ice is declining
at a rate of 13% per decade. Scientists warn of tipping points."""
)
show("Pattern 4: XML schema — parse this reliably with ElementTree", response, "blue")


# ── Pattern 5: NEGATIVE CONSTRAINTS ──────────────────────────────────────────
console.rule("[bold]Pattern 5: Negative constraints[/bold]")
# Telling the model what NOT to do is often more effective than what TO do.

response = claude(
    system="""You summarize technical articles for a general audience.
Rules:
- Do NOT use jargon or technical terms
- Do NOT use bullet points — write in flowing prose only
- Do NOT exceed 3 sentences
- Do NOT start with "The article" or "This article"
- Do NOT include any caveats or qualifications""",
    user="""Researchers at MIT have developed a novel attention mechanism that
reduces the quadratic complexity of self-attention to linear time by using
locality-sensitive hashing to approximate the full attention matrix,
enabling transformer models to process sequences of up to 64,000 tokens
without memory overflow on consumer-grade GPUs."""
)
show("Pattern 5: Negative constraints shape the output precisely", response, "magenta")


# ── Pattern 6: SELF-CONSISTENCY CHECK (the RAG guardian) ─────────────────────
console.rule("[bold]Pattern 6: Self-consistency / Grounding[/bold]")
# Critical for RAG: force the model to stay within provided context.
# This pattern is the foundation of your Research Assistant's answer layer.

context = """
DOCUMENT: Q3 2024 Earnings Report — TechCorp Inc.
Revenue: $2.4B (up 18% YoY)
Net income: $340M (down 12% YoY due to R&D investment)
Headcount: 12,400 employees (up 8%)
New products launched: AI Analytics Suite, CloudSync Pro
CEO Quote: "We are doubling down on AI infrastructure for 2025."
"""

response = claude(
    system=f"""You are a financial analyst assistant.
Answer questions using ONLY the information in the provided document.
If the answer is not in the document, say exactly: "This information is not in the provided document."
Do not use any external knowledge. Do not speculate.

DOCUMENT:
{context}""",
    user="What was TechCorp's profit margin and how many new products did they launch?"
)
show("Pattern 6: Grounded response — the RAG answer pattern", response, "cyan")