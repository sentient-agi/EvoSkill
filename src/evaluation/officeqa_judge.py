"""Hybrid LLM + code scorer for OfficeQA.

LLM does **decomposition + alignment only** — it parses the gold target and
predicted answer into matched component pairs, normalizing each value (strip
commas / currency symbols / unit words from numbers, normalize category
labels). It does no arithmetic and emits no score.

Code does **scoring** — for each component pair:
  - type=number  → raised-cosine smooth on relative error (matches the
                   shape of the legacy `_score_multi_tolerance`):
                     rel_err ≤ 0.01           → 1.0   (full-credit zone)
                     0.01 < rel_err < 0.10    → cosine decay 1.0 → 0.0
                     rel_err ≥ 0.10           → 0.0
  - type=category → case-insensitive whitespace-stripped exact match
  - pred missing  → 0.0

Final score = mean across components. Multi-part answers therefore get
partial credit (1 of 2 list items right → 0.5).

Implementation uses the Anthropic Python SDK directly (not dspy), with
prompt caching enabled at the 1-hour TTL on the static system message —
each evolution run can have hundreds of judge calls and a long-lived prefix
cache pays off (~10× cheaper input tokens for the cached portion after the
first call).

Failure handling: each call retries up to MAX_RETRIES_PER_CALL with
exponential backoff. After MAX_CONSECUTIVE_FAILURES failed calls in a row,
the next failure raises and aborts the entire evolution — the loop is
fundamentally bound to the Anthropic API, so a sustained outage means
results would be untrustworthy and we should stop rather than silently
score everything 0.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from functools import lru_cache
from typing import Any

from anthropic import Anthropic


# Use the API model id (date-suffixed). `claude-haiku-4-5-20251001` was the
# Haiku variant; for the judge we want Sonnet 4.6 — more reliable per-call.
JUDGE_MODEL = "claude-sonnet-4-6"

MAX_RETRIES_PER_CALL = 3
MAX_CONSECUTIVE_FAILURES = 3

# Cosine-decay zone bounds (rel_err). Match _score_multi_tolerance.
SOFT_REL_ERR = 0.01
HARD_REL_ERR = 0.10

# 1-hour prompt cache. Beta header required.
CACHE_TTL = "1h"
EXTENDED_CACHE_BETA = "extended-cache-ttl-2025-04-11"

_consecutive_failures = 0
_failure_lock = threading.Lock()

_client: Anthropic | None = None
_client_lock = threading.Lock()


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = Anthropic(
                    default_headers={"anthropic-beta": EXTENDED_CACHE_BETA},
                )
    return _client


# The SYSTEM message is constant across calls, so it gets a cache-control
# breakpoint and the API serves it from the prefix cache after the first
# call (within the 1-hour TTL window).
SYSTEM_INSTRUCTIONS = """\
You are grading a single OfficeQA answer (a U.S. Treasury benchmark whose answers are mostly numerical / financial). Decompose the gold target into components, match each one to the corresponding component in the prediction, and return a JSON array. The grading framework's code computes the final score from your decomposition — your job is alignment and value extraction, not arithmetic.

For each component, output an object with three fields:

- `type`: `"number"` if the component is numeric (a quantity, rate, year, ratio, etc.), `"category"` if it's a named/labeled value (e.g., `"surplus"`, `"Highway Trust Fund"`, `"deficit"`).

- `gold`: the cleaned value extracted from the gold target.
    - For numbers: a JSON number (float or int) with NO commas, currency symbols, or unit words. So `"$1,169.41 million"` → `1169.41`, `"7,199"` → `7199`, `"-19.29"` → `-19.29`.
    - For categories: a short normalized string (lowercase, trimmed).

- `pred`: the cleaned value extracted from the prediction, in the SAME units/scale as `gold` so the two are directly comparable. If the prediction expresses the value in different units (e.g., gold is in millions and pred is in billions), convert into the gold's scale before emitting. If the prediction does not contain a corresponding value at all, output `null`. If the prediction contradicts itself, use whatever value it commits to last.

Component-count rules:
- A list / tuple in gold (e.g. `[190.73, -19.29]`, `[0.012, surplus]`) → one component per item.
- Multiple labeled values (e.g. `GDP=2.3, CPI=4.5`) → one component per label.
- A single value with units → one component (do not split number from unit).

Examples:

Question: "What's the GDP and CPI as of 2010?"
Gold: "GDP=$15.08 billion, CPI=2.3%"
Pred: "GDP was $15,080 million and CPI was 2.3 percent"
→ [{"type":"number","gold":15.08,"pred":15.08},{"type":"number","gold":2.3,"pred":2.3}]

Question: "What is the Gini coefficient and is the fund in surplus or deficit?"
Gold: "[0.012, surplus]"
Pred: "Gini = 0.0121, deficit"
→ [{"type":"number","gold":0.012,"pred":0.0121},{"type":"category","gold":"surplus","pred":"deficit"}]

Question: "What was the difference, in millions of dollars?"
Gold: "1169.41"
Pred: "The absolute difference is approximately $1,169.41 million."
→ [{"type":"number","gold":1169.41,"pred":1169.41}]

Question: "What's the year and the program name?"
Gold: "1990, Highway Trust Fund"
Pred: "The year is 1990."
→ [{"type":"number","gold":1990,"pred":1990},{"type":"category","gold":"highway trust fund","pred":null}]

Output rules:
- Return ONLY the JSON array. No surrounding text, no markdown fences, no commentary.
"""


def _cosine_smooth(rel_err: float) -> float:
    """Raised-cosine decay between SOFT_REL_ERR and HARD_REL_ERR. Matches the
    shape used by the legacy `_score_multi_tolerance`."""
    if rel_err <= SOFT_REL_ERR:
        return 1.0
    if rel_err >= HARD_REL_ERR:
        return 0.0
    t = (rel_err - SOFT_REL_ERR) / (HARD_REL_ERR - SOFT_REL_ERR)
    return (1.0 + math.cos(math.pi * t)) / 2.0


def _component_score(comp: dict[str, Any]) -> float:
    ctype = comp.get("type")
    gold = comp.get("gold")
    pred = comp.get("pred")
    if pred is None:
        return 0.0
    if ctype == "number":
        try:
            g = float(gold)
            p = float(pred)
        except (TypeError, ValueError):
            return 0.0
        if g == 0.0:
            if p == 0.0:
                return 1.0
            return _cosine_smooth(abs(p) / max(SOFT_REL_ERR, 1.0))
        rel_err = abs(p - g) / abs(g)
        return _cosine_smooth(rel_err)
    if ctype == "category":
        g = str(gold).strip().lower()
        p = str(pred).strip().lower()
        return 1.0 if g == p else 0.0
    return 0.0


def _parse_components(raw: str) -> list[dict[str, Any]]:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array found in judge output: {raw!r}")
    arr = json.loads(m.group(0))
    if not isinstance(arr, list) or not arr:
        raise ValueError(f"Judge output is not a non-empty list: {arr!r}")
    return arr


@lru_cache(maxsize=4096)
def _judge(question: str, ground_truth: str, predicted: str) -> tuple[float, ...]:
    """One judge call with retries. Returns per-component scores. Uses
    1-hour prompt caching on the constant system instructions."""
    client = _get_client()
    user_message = (
        f"Question: {question}\n"
        f"Gold target: {ground_truth}\n"
        f"Predicted answer: {predicted}\n\n"
        f"Return ONLY the JSON array."
    )

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES_PER_CALL):
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_INSTRUCTIONS,
                        "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
            # Concatenate any text blocks the model returned.
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            comps = _parse_components(text)
            return tuple(_component_score(c) for c in comps)
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES_PER_CALL - 1:
                time.sleep(2 ** attempt)
    assert last_exc is not None
    raise last_exc


def score_officeqa(question: str, ground_truth: str, predicted: str) -> float:
    """Score in [0, 1] = mean of per-component scores. See module docstring."""
    global _consecutive_failures
    try:
        per_component = _judge(question, ground_truth, predicted)
    except Exception as e:
        with _failure_lock:
            _consecutive_failures += 1
            cf = _consecutive_failures
        if cf >= MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"OfficeQA judge failed {cf} calls in a row "
                f"(threshold = {MAX_CONSECUTIVE_FAILURES}); aborting evolution. "
                f"Last error: {e!r}"
            ) from e
        print(
            f"[JUDGE] API failure #{cf}/{MAX_CONSECUTIVE_FAILURES} "
            f"(returning 0.0): {e!r}",
            flush=True,
        )
        return 0.0

    with _failure_lock:
        _consecutive_failures = 0
    return sum(per_component) / len(per_component)
