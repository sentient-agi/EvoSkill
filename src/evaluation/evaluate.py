import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from tqdm import tqdm

from src.harness import Agent, AgentTrace
from src.harness.sdk_config import get_sdk
from src.harness.utils import eval_score_callback

if TYPE_CHECKING:
    from src.cache import RunCache

T = TypeVar("T")

# Sentinel for the `cache` parameter's default, distinguishing "caller passed
# nothing — auto-construct one" from "caller explicitly disabled caching with
# cache=None". Without this sentinel, every oneshot eval / ad-hoc script that
# forgets to pass an explicit RunCache silently runs without writing any
# cache entries — wasting future re-runs' budget. Auto-attaching a default
# cache makes every caller benefit automatically.
_AUTO_CACHE = object()


def _extract_model(options) -> str:
    """Best-effort model id extraction from either dict or ClaudeAgentOptions."""
    if isinstance(options, dict):
        return str(options.get("model", "") or "")
    return str(getattr(options, "model", "") or "")


def _extract_effort(options) -> str:
    """Extract the thinking-effort knob ("low"|"medium"|"high"|"max") from
    options for cache keying. EvoSkill experiments run with adaptive thinking,
    so `effort` is the only discriminating signal across runs.

    Returns "" when not set, which causes RunCache to leave the path key
    unchanged (preserving legacy callers' cache reachability).
    """
    if isinstance(options, dict):
        return str(options.get("effort", "") or "")
    return str(getattr(options, "effort", "") or "")


def _extract_system_prompt(options) -> str:
    """Best-effort system-prompt extraction for the cache key.

    Claude SDK packs the user-provided system text inside a preset dict:
    `system_prompt = {"type": "preset", "preset": "claude_code", "append": "<user text>"}`.
    Other SDKs (opencode, openhands, codex, goose) typically use a flat
    `system` string. We check both patterns and return the user-authored
    string — that's what the cache key should reflect, since the preset
    base is constant across runs and only the appended user text varies.

    Returns "" when nothing is detectable, which causes RunCache to fall
    back to its on-disk prompt-file hash.
    """
    if isinstance(options, dict):
        sp = options.get("system_prompt")
        if isinstance(sp, dict):
            return str(sp.get("append", "") or "")
        if isinstance(sp, str):
            return sp
        return str(options.get("system", "") or "")
    sp = getattr(options, "system_prompt", None)
    if isinstance(sp, dict):
        return str(sp.get("append", "") or "")
    if isinstance(sp, str):
        return sp
    return str(getattr(options, "system", "") or "")


def _build_default_cache():
    """Lazily construct a default RunCache pointing at `<cwd>/.cache/runs/`.

    Matches the path the run_loop's RunCache uses, so cache entries written
    by oneshot scripts are visible to subsequent evolution runs (and vice
    versa). Returns None if construction fails for any reason (e.g.
    not in a git-tracked directory) — fail-soft so caching is opportunistic
    rather than required.
    """
    try:
        from src.cache import RunCache, CacheConfig
        return RunCache(CacheConfig(
            cache_dir=Path(".cache/runs"),
            enabled=True,
            cwd=Path.cwd(),
        ))
    except Exception:
        return None


@dataclass
class EvalResult(Generic[T]):
    """Result of evaluating a single question."""
    question: str
    ground_truth: str
    trace: AgentTrace[T] | None


async def evaluate_agent_parallel(
    agent: Agent[T],
    items: list[tuple[str, str]],
    max_concurrent: int = 2,
    *,
    cache: "RunCache | None | object" = _AUTO_CACHE,
    tag_prefix: str = "val",
) -> list[EvalResult[T]]:
    """
    Run agent on multiple questions in parallel.

    Args:
        agent: The agent to evaluate
        items: List of (question, ground_truth) tuples
        max_concurrent: Max concurrent agent runs (default 2)
        cache: Caching behavior. Three modes:
                - omit / not pass (default `_AUTO_CACHE` sentinel) — auto-
                  construct a RunCache pointing at `<cwd>/.cache/runs/`.
                  Every oneshot eval / ad-hoc script gets caching for free.
                - `None` — explicitly disable caching for this call.
                - `<RunCache instance>` — use the caller-provided cache
                  (the loop's runner does this so its config flows through).
        tag_prefix: Prefix for per-run human tags shown in stdout (e.g.
                    "val 3/8"). Set to "train" when this helper is reused
                    for training-sample runs.

    Returns:
        List of EvalResult containing question, ground_truth, and trace
    """
    # Resolve auto-cache. Done once up front so per-question cache lookups
    # below see a stable RunCache (or None).
    if cache is _AUTO_CACHE:
        cache = _build_default_cache()

    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(items)

    from src.harness.utils import eval_run_ground_truth as _gt_var

    async def run_one(idx: int, question: str, ground_truth: str) -> EvalResult[T]:
        tag = f"{tag_prefix} {idx + 1}/{total}"
        async with semaphore:
            try:
                # Outer hard limit derived from Agent.TIMEOUT_SECONDS + 120s
                # buffer (one API_TIMEOUT_MS worth). Existing callers using
                # the default 720s budget see 840s here; runners that override
                # Agent.TIMEOUT_SECONDS (e.g. for short-experiment evals) get
                # a proportional outer ceiling without a separate flag.
                from src.harness.agent import Agent as _Agent
                async with asyncio.timeout(_Agent.TIMEOUT_SECONDS + 120):
                    # Check cache first. We pull the system prompt from the
                    # agent's resolved options and pass it to the cache so
                    # the key reflects what the LLM actually receives —
                    # critical for callers (oneshot scripts, run_loop) that
                    # construct prompts dynamically rather than from disk.
                    trace = None
                    sdk = get_sdk()
                    opts = agent._get_options()
                    model = _extract_model(opts)
                    sys_prompt = _extract_system_prompt(opts)
                    effort = _extract_effort(opts)
                    if cache is not None:
                        trace = cache.get(
                            question,
                            agent.response_model,
                            sdk=sdk,
                            model=model,
                            system_prompt=sys_prompt,
                            effort=effort,
                        )

                    # Cache miss - run agent
                    if trace is None:
                        # Surface ground_truth on the agent.run span via contextvar,
                        # mirroring eval_full.py so reviewers can compare GT vs
                        # final_answer side-by-side in Phoenix.
                        gt_token = _gt_var.set(str(ground_truth))
                        try:
                            trace = await agent.run(question, tag=tag)
                        finally:
                            _gt_var.reset(gt_token)
                        # Store in cache
                        if cache is not None:
                            cache.set(
                                question, trace, sdk=sdk, model=model,
                                system_prompt=sys_prompt, effort=effort,
                            )

            except asyncio.TimeoutError:
                print(f"Eval timed out (12min) for: {question[:50]}...")
                trace = None
            except Exception as e:
                print(f"Failed on question: {question[:50]}... Error: {e}")
                trace = None
            return EvalResult(question=question, ground_truth=ground_truth, trace=trace)

    # Manual tqdm + as_completed so we can show a running mean score
    # in the bar postfix as samples finish. The score callback is published
    # by the runner via `eval_score_callback` contextvar; if it's missing
    # (e.g. caller didn't set one), we fall back to plain progress display.
    scorer = eval_score_callback.get()

    async def _run_indexed(idx: int, q: str, gt: str) -> tuple[int, EvalResult[T]]:
        return idx, await run_one(idx, q, gt)

    tasks = [asyncio.create_task(_run_indexed(i, q, gt)) for i, (q, gt) in enumerate(items)]
    results: list[EvalResult[T] | None] = [None] * total
    score_sum = 0.0
    score_n = 0
    pass_n = 0
    pbar = tqdm(total=total, desc="Evaluating", mininterval=10)
    try:
        for fut in asyncio.as_completed(tasks):
            idx, res = await fut
            results[idx] = res
            if scorer is not None:
                if res.trace is not None and getattr(res.trace, "output", None) is not None:
                    try:
                        ans = str(res.trace.output.final_answer)
                        s = float(scorer(res.question, ans, res.ground_truth))
                    except Exception:
                        s = 0.0
                else:
                    s = 0.0
                score_sum += s
                score_n += 1
                if s >= 1.0:
                    pass_n += 1
                pbar.set_postfix({
                    "avg": f"{score_sum / score_n:.3f}",
                    "pass": f"{pass_n}/{score_n}",
                })
            pbar.update(1)
    finally:
        pbar.close()
    return [r for r in results if r is not None]
