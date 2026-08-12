"""
Phase 4 — Script 01: Tool Use Basics
=======================================
Goal: see the raw tool_use / tool_result round trip with your own eyes,
using a simple calculator tool before wiring in retrieve() from Phase 3.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from core.llm_client import LLMClient
from rich.console import Console
from rich.panel import Panel

console = Console()
llm = LLMClient()

# ── Define a tool schema ─────────────────────────────────────────────────
CALCULATOR_TOOL = {
    "name": "calculate",
    "description": "Perform a basic arithmetic calculation. Use this whenever "
                    "the user asks a question requiring numeric computation. "
                    "Do NOT use for anything other than arithmetic.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A Python-evaluable arithmetic expression, e.g. '12 * 47 + 3'",
            }
        },
        "required": ["expression"],
    },
}


def calculate(expression: str) -> str:
    """The ACTUAL function that runs when Claude requests the tool."""
    try:
        # eval is fine here because this is a controlled, local learning exercise —
        # NEVER eval() untrusted input in a real system; use a safe math parser instead.
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def run_with_tools(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    console.print(f"[dim]--> Sending to Claude with tools available...[/dim]")
    response = llm.client.messages.create(
        model=llm.model,
        max_tokens=1024,
        tools=[CALCULATOR_TOOL],
        messages=messages,
    )

    console.print(f"[dim]<-- stop_reason: {response.stop_reason}[/dim]")

    # If Claude wants to use a tool, stop_reason will be "tool_use"
    if response.stop_reason == "tool_use":
        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        console.print(f"[bold amber]Claude wants to call:[/bold amber] {tool_use_block.name}({tool_use_block.input})")

        # YOUR code actually executes it — Claude never runs anything itself
        tool_result = calculate(**tool_use_block.input)
        console.print(f"[bold teal]Tool result:[/bold teal] {tool_result}")

        # Send the result back so Claude can give a final answer
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result,
            }]
        })

        final_response = llm.client.messages.create(
            model=llm.model, max_tokens=1024, tools=[CALCULATOR_TOOL], messages=messages
        )
        return final_response.content[0].text

    return response.content[0].text


if __name__ == "__main__":
    questions = [
        "What is 4,827 multiplied by 193, minus 1,000?",
        "What is the capital of France?",  # should NOT trigger the tool
    ]
    for q in questions:
        console.rule(f"[bold]{q}[/bold]")
        answer = run_with_tools(q)
        console.print(Panel(answer, title="Final Answer", border_style="cyan"))