"""
Phase 6 — Script 01: Golden Evaluation Dataset
==================================================
Goal: a deliberately diverse set of test questions — not just "easy" ones —
covering every failure mode discovered across Phases 1-5.
"""

GOLDEN_SET = [
    # ── Answerable, single-fact ──────────────────────────────────────────
    {"question": "What was TechCorp's Q3 2024 revenue?", "expected_answerable": True},
    {"question": "How many employees does TechCorp have?", "expected_answerable": True},

    # ── Answerable, multi-fact (exercises Phase 5's multi-agent path) ────
    {"question": "What was TechCorp's financial performance and what are they investing in next?", "expected_answerable": True},
    {"question": "What products did TechCorp launch and how did headcount change?", "expected_answerable": True},

    # ── Unanswerable — verifies the Phase 1 refusal rule still holds ─────
    {"question": "What is TechCorp's stock price today?", "expected_answerable": False},
    {"question": "What is TechCorp's mission statement?", "expected_answerable": False},
    {"question": "Who is TechCorp's main competitor?", "expected_answerable": False},

    # ── Edge cases ────────────────────────────────────────────────────────
    {"question": "Tell me about the company", "expected_answerable": True},  # vague but answerable
    {"question": "What did the CEO say about 2025 plans?", "expected_answerable": True},
]

if __name__ == "__main__":
    print(f"Golden set: {len(GOLDEN_SET)} questions")
    answerable = sum(1 for q in GOLDEN_SET if q["expected_answerable"])
    print(f"  Answerable: {answerable}  |  Unanswerable (refusal expected): {len(GOLDEN_SET) - answerable}")