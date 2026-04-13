# Evaluation Layer

The evaluation folder has 3 layers:
1. **Runners** — HOW to run the agent on many questions (`evaluate.py`, `eval_full.py`)
2. **Reward function** — HOW to decide if an answer is correct (`reward.py`)
3. **Task-specific scorers** — Custom scoring for specific benchmarks

---

## Layer 1: Running the Agent on Questions

### `evaluate.py` — Used by the self-improving loop

```python
evaluate_agent_parallel(agent, items, max_concurrent=2, cache=None) → list[EvalResult]
```

- Takes `(question, ground_truth)` pairs, runs agent in parallel
- **17-minute timeout** per question
- **Cache support**: checks `cache.get(question)` before running, stores via `cache.set()` after
- **Graceful failure**: Timeout/exception → `trace = None` (0 points, not a crash)
- Uses `asyncio.Semaphore` to limit concurrency

Result type:
```python
@dataclass
class EvalResult:
    question: str
    ground_truth: str
    trace: AgentTrace | None    # None = failed/timed out
```

### `eval_full.py` — Used by `evoskill eval` and standalone scripts

```python
evaluate_full(agent, items, output_path, max_concurrent=5, resume=True) → list[IndexedEvalResult]
```

Heavy-duty version with these extras:
- **Items include index**: tracks which CSV row
- **Saves to disk incrementally**: appends to pickle file after every question (crash-safe)
- **Resume mode**: on restart, skips successful indices, re-runs only failures

| Runner | Used by | Cache? | Resume? | Saves to disk? |
|--------|---------|--------|---------|----------------|
| `evaluate_agent_parallel` | The loop (`runner.py`) | Yes (RunCache) | No | No |
| `evaluate_full` | `evoskill eval`, scripts | No | Yes (pickle) | Yes (incremental) |

---

## Layer 2: Reward Function (`reward.py`)

Entry point:
```python
score_answer(ground_truth, predicted, tolerance=0.0) → 1.0 or 0.0
```

### Decision Tree

```
Does the ground truth contain numbers?
│
├── YES: Does the prediction contain numbers?
│   │
│   ├── YES: How many numbers in ground truth?
│   │   │
│   │   ├── MULTIPLE (e.g., "10 and 20"):
│   │   │   ALL ground truth numbers must appear in prediction
│   │   │   Each within tolerance AND text must overlap
│   │   │
│   │   └── SINGLE (e.g., "4.2 million"):
│   │       1. Extract base number + unit (4.2, "million")
│   │       2. Filter year-like numbers (1900-2100) from prediction
│   │          UNLESS ground truth itself is a year
│   │       3. Find closest match within tolerance
│   │       4. Check text overlap if GT has text
│   │          ("March 1977" → month must match too)
│   │
│   └── NO: → 0.0
│
└── NO: Text-only comparison
    1. Lowercase, strip quotes
    2. Remove parenthetical abbreviations like "(OASI)"
    3. Check if GT is a substring of prediction
    4. Check exact match
```

### Helper Functions

**`extract_numbers_with_context(text)`** — Finds numbers with ±20 chars of context:
```
"Revenue was 4.2 billion" → [(4.2, "revenue was 4.2 billion", False, False)]
```

**`detect_unit_in_context(context)`** — Looks for unit words:
```
"4.2 billion" → ("billion", 1e9)
"543 million" → ("million", 1e6)
"42"          → (None, 1.0)
```

**`normalize_number_with_units(number, context)`** — Returns base number + unit. Does NOT multiply: "543 million" → `(543, "million")`, not `543000000`.

**`is_likely_year(num)`** — Is num between 1900-2100 and integer? Filters incidental year references.

**`has_significant_text(text)`** — Does text have words beyond numbers/units?
```
"March 1977"   → True ("march" is significant)
"543 million"  → False (just number + unit)
```

**`check_text_overlap(gt, pred)`** — For hybrid answers like "March 1977", checks text parts match:
```
GT="March 1977", Pred="March 1977" → True
GT="March 1977", Pred="April 1977" → False
GT="March 1977", Pred="1977"       → False
```

---

## Layer 3: Task-Specific Scorers

### `sealqa_scorer.py` — LLM-as-judge

```python
score_sealqa(question, ground_truth, predicted) → 0.0 or 1.0
```

Uses GPT-5-mini (via OpenRouter) to grade answers:
1. Fills grading template with examples of CORRECT/INCORRECT/NOT_ATTEMPTED
2. Sends to LLM, gets back "A", "B", or "C"
3. "A" = 1.0, everything else = 0.0

Why not string matching? SEAL-QA answers are semantically complex — "San Francisco" should match "San Francisco, California".

### `dabstep_scorer.py` — Numeric + string matching

```python
question_scorer(input1, input2) → True/False
```

Decision flow:
1. Numbers with commas → extract numeric, compare with `math.isclose(rel_tol=1e-4)`
2. Lists (contains `;` or `,`) → split, sort, compare each pair
3. Plain numbers → numeric comparison with rounding
4. Strings → exact match, word subset check, or `SequenceMatcher` similarity > 0.95

### `livecodebench_scorer.py` — Code execution in Docker

```python
score_livecodebench(question, ground_truth, predicted) → 0.0 or 1.0
```

Steps:
1. Extract Python code from response (regex: ` ```python ... ``` `)
2. Parse test cases from ground_truth (JSON: `[{input, output}, ...]`)
3. Run code in Docker sandbox (`llm_sandbox.SandboxSession`, 5s timeout per test)
4. Compare stdout to expected output (exact match)
5. Pass@1: ALL tests must pass → 1.0, otherwise 0.0

---

## How the Loop Uses Scoring

### Multi-tolerance scoring (default)

The loop's default scorer calls `score_answer` at 5 tolerance levels and takes a weighted average:

```
Tolerances: [0%, 1%, 2.5%, 5%, 10%]
Weights:    [1.0, 0.83, 0.67, 0.5, 0.33]  (computed as 1/(1+20*tol))
```

Exact match ≈ 1.0, close answer ≈ 0.3-0.5, wrong answer = 0.0.

### Two scoring contexts

1. **Failure detection** (training samples): `score < 0.8` → failure, sent to proposer
2. **Program evaluation** (validation set): average score across all questions, compared to frontier
