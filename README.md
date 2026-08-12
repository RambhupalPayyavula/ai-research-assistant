
## Known technical debt (to resolve during Phase 7 integration)
- retrieve() logic is currently duplicated across Phase 3, 4, and 5 — should be
  consolidated into a single core/retrieval.py used by all three.
