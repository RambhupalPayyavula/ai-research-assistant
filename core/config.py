"""
core/config.py
===============
Centralized config — no magic numbers or hardcoded strings scattered
across the codebase. Change one place, affects the whole system.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    MODEL: str = "claude-sonnet-4-5"
    MAX_TOKENS_ANSWER: int = 1024
    TEMPERATURE_FACTUAL: float = 0.0
    TEMPERATURE_CREATIVE: float = 0.5
    CHUNK_SIZE_TOKENS: int = 500       # will matter starting Phase 3
    CHUNK_OVERLAP_TOKENS: int = 50
    TOP_K_RETRIEVAL: int = 5           # will matter starting Phase 2

settings = Config()