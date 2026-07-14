"""
core/llm_client.py
===================
The single reusable gateway to Claude for the entire project.
Every phase — RAG, agents, multi-agent, eval — calls through this.

Design decisions (the "why" that matters in interviews):
- Token/cost tracking on every call: you cannot optimize what you don't measure.
- Retry with exponential backoff: real APIs fail transiently; production code
  must not crash on a single dropped connection or rate limit.
- Structured logging: this becomes the foundation for Phase 6 observability.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path
from datetime import datetime

import anthropic
from dotenv import load_dotenv
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("llm_client")





# Claude Sonnet pricing — update these if pricing changes; centralizing
# them here means every cost calculation in the app stays consistent.
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},   # per million tokens
}


@dataclass
class CallResult:
    """Everything you need to know about a single LLM call — not just the text."""
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    estimated_cost_usd: float = field(init=False)

    def __post_init__(self):
        prices = PRICING.get(self.model, PRICING["claude-sonnet-4-5"])
        self.estimated_cost_usd = (
            (self.input_tokens / 1_000_000) * prices["input"]
            + (self.output_tokens / 1_000_000) * prices["output"]
        )


class LLMClient:
    """
    Wraps the Anthropic SDK with retry logic, cost tracking, and logging.
    This is the ONLY place in the codebase that talks directly to the API.
    """

    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.total_cost_usd = 0.0
        self.total_calls = 0

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        )),
        reraise=True,
    )
    def call(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CallResult:
        """
        The fundamental method every higher-level function calls through.

        Parameters
        ----------
        system      : the persona/rules/constraints
        messages    : full conversation history — [{"role": "user"/"assistant", "content": "..."}]
                      (stateless API — YOU own the history, the model doesn't)
        max_tokens  : output cap — also a cost control lever
        temperature : 0.0 for factual/structured tasks, higher for creative tasks
        """
        start = time.time()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )

        latency = time.time() - start

        result = CallResult(
            text=response.content[0].text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_seconds=latency,
        )

        self.total_cost_usd += result.estimated_cost_usd
        self.total_calls += 1

        logger.info(
            f"call #{self.total_calls} | "
            f"tokens in={result.input_tokens} out={result.output_tokens} | "
            f"cost=${result.estimated_cost_usd:.5f} | "
            f"latency={latency:.2f}s | "
            f"running_total=${self.total_cost_usd:.4f}"
        )

        return result

    def simple(self, system: str, user_message: str, **kwargs) -> str:
        """Convenience method for single-turn calls — returns just the text."""
        result = self.call(system=system, messages=[{"role": "user", "content": user_message}], **kwargs)
        return result.text
    
    def log_usage_to_file(self, log_path: str = "usage_log.jsonl"):
        """Append a usage record — this becomes real data for Phase 6 evaluation."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    



# ── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = LLMClient()
    answer = client.simple(
        system="You are a concise assistant.",
        user_message="In one sentence, what is retrieval-augmented generation?"
    )
    print(f"\nResponse: {answer}")
    print(f"Total cost so far: ${client.total_cost_usd:.6f}")

