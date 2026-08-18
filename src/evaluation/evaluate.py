import asyncio
import os
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


class MemoryBudgetSemaphore:
    """Concurrency gate that tracks memory allocation instead of a flat count.

    Each acquire specifies how many MB this task needs. The gate blocks until
    enough budget is free. A flat max_concurrent cap is also enforced so we
    never exceed N simultaneous containers regardless of memory.

    Grant staggering: shares admission.py's GrantStagger with
    DynamicAdmission (get_task_gate's own gate), gated behind the identical
    EVOSKILL_DYNAMIC_CONCURRENCY=1 opt-in. Before this, every docker-backend
    launch that sets memory_budget_mb+task_memory_fn (which is every current
    one — run_tb2_evo.py, run_swe_rebench_evo.py, run_tb2_base_rollout.py,
    etc.) used THIS class instead of get_task_gate(), so a newly-freed slot
    fired its docker-container-creation call immediately with zero pacing —
    a burst of N simultaneously-eligible acquires (e.g. a full train+val
    pool with nothing previously in flight) could fire N container-creation
    calls in the same instant regardless of the AIMD/stagger fix, since that
    fix lived only in DynamicAdmission, a class this path never touched.
    """

    def __init__(self, budget_mb: int, max_concurrent: int):
        self._budget_mb = budget_mb
        self._allocated = 0
        self._count = 0
        self._max_concurrent = max_concurrent
        self._cond = asyncio.Condition()
        self._stagger = None
        from src.harness.admission import dynamic_enabled, GrantStagger
        if dynamic_enabled():
            self._stagger = GrantStagger()

    async def acquire(self, mem_mb: int):
        async with self._cond:
            while (self._allocated + mem_mb > self._budget_mb
                   or self._count >= self._max_concurrent):
                await self._cond.wait()
            self._allocated += mem_mb
            self._count += 1
        if self._stagger is not None:
            delay = self._stagger.reserve()
            if delay > 0.0:
                try:
                    await asyncio.sleep(delay)
                except BaseException:
                    # Mirrors DynamicAdmission.acquire()'s same guard: the
                    # slot (allocated MB + count) is already reserved above,
                    # so a cancellation/error mid-stagger-sleep must roll it
                    # back here or it leaks forever, silently shrinking
                    # effective concurrency.
                    await self.release(mem_mb)
                    raise

    async def release(self, mem_mb: int):
        async with self._cond:
            self._allocated -= mem_mb
            self._count -= 1
            self._cond.notify_all()

# Sentinel for the `cache` parameter's default, distinguishing "caller passed
# nothing — auto-construct one" from "caller explicitly disabled caching with
# cache=None". Without this sentinel, every oneshot eval / ad-hoc script that
# forgets to pass an explicit RunCache silently runs without writing any
# cache entries — wasting future re-runs' budget. Auto-attaching a default
# cache makes every caller benefit automatically.
_AUTO_CACHE = object()


def _extract_project_root(options) -> str:
    """Extract project_root (workspace path) from agent options.

    ClaudeAgentOptions uses 'cwd' instead of 'project_root'.
    """
    if isinstance(options, dict):
        return str(options.get("project_root", "") or options.get("cwd", "") or "")
    return str(
        getattr(options, "project_root", "") or
        getattr(options, "cwd", "") or ""
    )


def _extract_model(options) -> str:
    """Best-effort model id extraction from either dict or ClaudeAgentOptions."""
    if isinstance(options, dict):
        return str(options.get("model_id", "") or options.get("model", "") or "")
    return str(getattr(options, "model_id", "") or getattr(options, "model", "") or "")


def _extract_effort(options) -> str:
    """Extract a thinking-mode key from options for cache discrimination.

    Combines enable_thinking + thinking_effort (direct harness) or effort
    (CC harness) into a single string so runs with different thinking
    configs produce different cache keys.

    Returns "" when not set, preserving legacy cache reachability.
    """
    if isinstance(options, dict):
        enable = options.get("enable_thinking")
        if enable is False:
            return "thinking_off"
        effort = str(options.get("thinking_effort", "") or options.get("effort", "") or "")
        return effort
    effort = str(getattr(options, "effort", "") or "")
    return effort


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
    score: float | None = None  # Populated by scorer during eval; reuse instead of re-scoring


async def evaluate_agent_parallel(
    agent: Agent[T],
    items: list[tuple[str, str]],
    max_concurrent: int = 2,
    *,
    cache: "RunCache | None | object" = _AUTO_CACHE,
    tag_prefix: str = "val",
    per_item_tags: list[str] | None = None,
    memory_budget_mb: int | None = None,
    task_memory_fn: "callable | None" = None,
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
        memory_budget_mb: Total memory budget for concurrent tasks (MB).
                    When set with task_memory_fn, uses MemoryBudgetSemaphore
                    instead of a flat count semaphore.
        task_memory_fn: Callable(question_str) -> int returning the task's
                    memory_mb. Required when memory_budget_mb is set.

    Returns:
        List of EvalResult containing question, ground_truth, and trace
    """
    # Resolve auto-cache. Build from the agent's workspace so the cache key
    # reflects the agent's actual skill state (not the project root's).
    if cache is _AUTO_CACHE:
        try:
            opts = agent._get_options()
            workspace = _extract_project_root(opts)
            if workspace:
                from src.cache import RunCache, CacheConfig
                ws = Path(workspace)
                cache = RunCache(CacheConfig(
                    cache_dir=Path(".cache/runs"),
                    enabled=True,
                    cwd=ws,
                    live_skills_dir=ws / ".claude" / "skills",
                ))
            else:
                cache = _build_default_cache()
        except Exception:
            cache = _build_default_cache()

    use_mem_budget = memory_budget_mb is not None and task_memory_fn is not None
    if use_mem_budget:
        mem_sem = MemoryBudgetSemaphore(memory_budget_mb, max_concurrent)
    else:
        # Plain Semaphore by default; the process-wide AIMD admission
        # controller when EVOSKILL_DYNAMIC_CONCURRENCY=1 (drop-in interface:
        # async acquire, sync release).
        from src.harness.admission import get_task_gate
        flat_sem = get_task_gate(max_concurrent)

    # FLEET-WIDE ROLLOUT ADMISSION, for EVERY benchmark that goes through this
    # function — not just the one that asked for it. Wired here rather than in a
    # runner because per-runner wiring is how SSB v2 ended up the only harness with
    # a fleet gate: two tau3 legs each ran their OWN AIMD controller with range
    # [4,16], so the machine could carry 32 in flight while neither controller knew
    # the other existed. `fleet.py` caps LLM CALLS proportionally; `seats.py` caps
    # concurrent ROLLOUTS across processes with STRICT priority. This is the latter.
    #
    # COMPOSED, never substituted: an unset EVOSKILL_SEAT_TOTAL reproduces prior
    # behaviour EXACTLY (SeatPool.from_env returns None), so no existing run changes
    # unless it opts in. Setting the seat total below the local cap makes the fleet
    # gate binding.
    #
    # ORDER MATTERS — scarce cross-process resource FIRST, then the local slot.
    # Reversed, a process holds a local slot while blocked on a seat and its own
    # later rollouts queue behind a coroutine that is itself waiting: self-inflicted
    # head-of-line blocking. Same ordering SSB v2 uses.
    _fleet_seats = None
    try:
        from src.harness.seats import SeatPool
        _fleet_seats = SeatPool.from_env(
            member=os.environ.get("EVOSKILL_SEAT_MEMBER") or tag_prefix)
    except Exception as _e:                       # pragma: no cover
        print(f"  [SEATS] pool unavailable ({type(_e).__name__}: {_e}); "
              f"falling back to LOCAL admission only", flush=True)
    if _fleet_seats is not None:
        _inner_sem = mem_sem if use_mem_budget else flat_sem

        class _FleetGate:
            """Composes the fleet seat pool with whatever local gate was chosen."""

            async def acquire(self):
                await _fleet_seats.acquire()
                try:
                    await _inner_sem.acquire()
                except BaseException:
                    _fleet_seats.release()        # never strand a seat on a failed local acquire
                    raise

            def release(self):
                try:
                    _inner_sem.release()
                finally:
                    _fleet_seats.release()

            async def __aenter__(self):
                await self.acquire(); return self

            async def __aexit__(self, *exc):
                self.release()

        _g = _FleetGate()
        if use_mem_budget:
            mem_sem = _g
        else:
            flat_sem = _g
        _s = _fleet_seats.snapshot()
        print(f"  [SEATS] FLEET pool total={_s['total']} used={_s['used']} "
              f"member={_fleet_seats.member} priority={_fleet_seats.priority} "
              f"(local cap {max_concurrent})", flush=True)
    total = len(items)

    from src.harness.utils import eval_run_ground_truth as _gt_var

    # Per-batch cache accounting. Every fresh completion must end in exactly
    # one bucket; `unaccounted` > 0 in the audit line means a completion
    # escaped the write path entirely — the silent-loss signature of the
    # 2026-07-22 nex evo run (71 of 117 rollouts never cached, zero log
    # lines). Counted with plain ints under the event loop (single thread).
    from collections import Counter as _Counter
    _cache_audit: "dict[str, int]" = _Counter()

    async def run_one(idx: int, question: str, ground_truth: str) -> EvalResult[T]:
        if per_item_tags and idx < len(per_item_tags):
            _my_tag = per_item_tags[idx]
            _split_total = sum(1 for t in per_item_tags if t == _my_tag)
            _split_idx = sum(1 for t in per_item_tags[:idx] if t == _my_tag) + 1
            tag = f"{_my_tag} {_split_idx}/{_split_total}"
        else:
            tag = f"{tag_prefix} {idx + 1}/{total}"

        task_mem = task_memory_fn(question) if use_mem_budget else 0

        if use_mem_budget:
            await mem_sem.acquire(task_mem)
        else:
            await flat_sem.acquire()
        try:
            # Resolve cache key components BEFORE the timeout scope so they're
            # always available in the except handlers for caching failures.
            sdk = get_sdk()
            opts = agent._get_options()
            model = _extract_model(opts)
            sys_prompt = _extract_system_prompt(opts)
            effort = _extract_effort(opts)
            trace = None

            try:
                _outer_timeout = getattr(agent, 'timeout_seconds', 720) + 120
                async with asyncio.timeout(_outer_timeout):
                    if cache is not None:
                        trace = cache.get(
                            question,
                            agent.response_model,
                            sdk=sdk,
                            model=model,
                            system_prompt=sys_prompt,
                            effort=effort,
                        )
                        if trace is not None:
                            _cache_audit["hit"] += 1

                    if trace is None:
                        _cache_audit["fresh"] += 1
                        from src.harness.agent import is_provider_failure, make_partial_error_trace
                        _max_task_retries = 3
                        for _task_attempt in range(_max_task_retries):
                            gt_token = _gt_var.set(str(ground_truth))
                            try:
                                trace = await agent.run(question, tag=tag)
                            except Exception as _run_exc:
                                # A raised exception (e.g. a docker/OCI exec
                                # failure) must go through the SAME retry
                                # budget as an is_provider_failure()-flagged
                                # returned trace, not bypass it straight to
                                # the single-shot generic `except Exception`
                                # handler further down this timeout scope
                                # (2026-07-11 audit: this previously gave
                                # exception-raising failures only 1 attempt
                                # instead of the intended 3).
                                _q_short = question[:60].replace('\n', ' ')
                                print(f"    [task retry] agent.run() raised {type(_run_exc).__name__}: {_run_exc} on: {_q_short}", flush=True)
                                trace = make_partial_error_trace(
                                    f"[ERROR] {type(_run_exc).__name__}: {_run_exc}",
                                    f"{type(_run_exc).__name__}: {_run_exc}",
                                )
                            finally:
                                _gt_var.reset(gt_token)
                            if not is_provider_failure(trace):
                                break
                            if _task_attempt < _max_task_retries - 1:
                                _q_short = question[:60].replace('\n', ' ')
                                print(f"    [task retry {_task_attempt+1}/{_max_task_retries}] provider failure on: {_q_short}", flush=True)
                                await asyncio.sleep(10)
                        if cache is not None:
                            if is_provider_failure(trace):
                                _cache_audit["skip_provider_failure"] += 1
                                _q_short = question[:60].replace('\n', ' ')
                                print(
                                    f"    [CACHE-SKIP] provider-failure trace NOT cached "
                                    f"(result={(getattr(trace, 'result', '') or '')[:80]!r}, "
                                    f"parse_error={(getattr(trace, 'parse_error', '') or '')[:80]!r}) "
                                    f"on: {_q_short}", flush=True)
                            else:
                                _w = cache.set(
                                    question, trace, sdk=sdk, model=model,
                                    system_prompt=sys_prompt, effort=effort,
                                )
                                _cache_audit["write_ok" if _w else "write_refused"] += 1
                                if not _w:
                                    # SPLIT OUT THE WALL-LIMITED SKIP. Both land in
                                    # `write_refused` for reconciliation (so `unaccounted`
                                    # still balances), but they mean opposite things while
                                    # babysitting: a wall-limited skip is EXPECTED and
                                    # informative (#182 — the record must not freeze an
                                    # outcome our own wall decided), whereas any other
                                    # refusal is a cache fault worth stopping for. Counting
                                    # them as one number is how an expected 3 and a broken 3
                                    # look identical in the audit line.
                                    from src.cache.run_cache import (
                                        _termination_reason as _tr, _wall_skip_reasons as _wsr)
                                    _rs = _wsr()
                                    _r = _tr(getattr(trace, "output", None)) if _rs else None
                                    _q_short = question[:60].replace('\n', ' ')
                                    if _r and str(_r).strip().lower() in _rs:
                                        _cache_audit["write_refused_wall_limited"] += 1
                                        print(f"    [CACHE-SKIP] wall-limited ({_r}) NOT cached "
                                              f"— will re-execute under the current wall, not "
                                              f"replay this one: {_q_short}", flush=True)
                                    else:
                                        print(f"    [CACHE-SKIP] RunCache.set refused write (see RunCache WARNING above) on: {_q_short}", flush=True)
                        # Admission signal: a FRESH run finished without
                        # provider failure — infra held up, regardless of
                        # task score. Cache hits deliberately don't count
                        # (no sandbox was exercised).
                        if not is_provider_failure(trace):
                            from src.harness.admission import report_clean_completion
                            report_clean_completion()

            except asyncio.TimeoutError:
                timeout_s = getattr(agent, 'timeout_seconds', 720) + 120
                print(f"Eval timed out ({timeout_s}s) for: {question[:50]}...")
                from src.harness.agent import make_partial_error_trace
                trace = make_partial_error_trace(
                    "[TIMEOUT]", f"Eval timeout ({timeout_s}s)", duration_ms=int(timeout_s * 1000),
                )
                # [TIMEOUT] is a _PROVIDER_FAILURE_MARKERS entry — a timeout here
                # can be a transient infra stall (network, outer/inner budget
                # mismatch), not necessarily genuine solver incapability, so it
                # must go through the same is_provider_failure() gate as every
                # other cache.set() call site rather than caching unconditionally.
                from src.harness.agent import is_provider_failure
                _cache_audit["outer_timeout"] += 1
                if cache is not None and not is_provider_failure(trace):
                    try:
                        cache.set(question, trace, sdk=sdk, model=model,
                                  system_prompt=sys_prompt, effort=effort)
                    except Exception as _ce:
                        print(f"    [CACHE-SKIP] timeout-trace cache.set raised {type(_ce).__name__}: {_ce}", flush=True)
            except Exception as e:
                _cache_audit["exception_path"] += 1
                print(f"Failed on question: {question[:50]}... Error: {e}")
                # Partial cost/turns preserved, same as the sibling timeout
                # handler above (bug found in a Jul 1 audit: every one of
                # these generic-exception handlers hardcoded zero right next
                # to a timeout handler that correctly read the same context
                # vars).
                from src.harness.agent import make_partial_error_trace
                trace = make_partial_error_trace(
                    f"[ERROR] {type(e).__name__}: {e}", f"{type(e).__name__}: {e}",
                )
        finally:
            if use_mem_budget:
                await mem_sem.release(task_mem)
            else:
                flat_sem.release()
        return EvalResult(question=question, ground_truth=ground_truth, trace=trace)

    # Manual tqdm + as_completed so we can show a running mean score
    # in the bar postfix as samples finish. The score callback is published
    # by the runner via `eval_score_callback` contextvar; if it's missing
    # (e.g. caller didn't set one), we fall back to plain progress display.
    scorer = eval_score_callback.get()

    async def _run_indexed(idx: int, q: str, gt: str) -> tuple[int, EvalResult[T]]:
        return idx, await run_one(idx, q, gt)

    def _score(res: EvalResult[T]) -> float:
        if res.trace is not None and getattr(res.trace, "inline_score", None) is not None:
            return res.trace.inline_score
        if res.trace is not None and getattr(res.trace, "output", None) is not None:
            try:
                ans = str(res.trace.output.final_answer)
                return float(scorer(res.question, ans, res.ground_truth))
            except Exception:
                return 0.0
        return 0.0

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
                s = _score(res)
                res.score = s
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

    # Post-batch pass: a task can still show the dangling-tool-call /
    # provider-failure signature after run_one's own inline retries (e.g.
    # a task that only fails while concurrency is still high). Rerunning
    # now, after the bulk of the batch has drained and concurrency has
    # dropped, gives contention-caused failures a real chance to succeed
    # instead of silently landing in the batch's scores as a wasted 0.
    from src.harness.agent import is_provider_failure
    # 2 rounds, not 1: a single post-batch rerun attempt gave a task hitting
    # a transient infra failure (e.g. docker/OCI contention) no further
    # backstop -- its score froze at whatever that one rerun produced, even
    # if that rerun itself failed for the same transient reason (2026-07-11
    # audit, after a diagnostic run hit exactly this on 3/25 ARC-AGI-2 val
    # tasks).
    _max_batch_rerun_rounds = 2
    for _rerun_round in range(_max_batch_rerun_rounds):
        buggy_idxs = [
            i for i, r in enumerate(results)
            if r is not None and is_provider_failure(r.trace)
        ]
        if not buggy_idxs:
            break
        print(
            f"  [batch rerun {_rerun_round + 1}/{_max_batch_rerun_rounds}] {len(buggy_idxs)} task(s) still buggy "
            f"after inline retries, rerunning: {[items[i][0][:40] for i in buggy_idxs]}",
            flush=True,
        )
        rerun_tasks = [asyncio.create_task(_run_indexed(i, items[i][0], items[i][1])) for i in buggy_idxs]
        for fut in asyncio.as_completed(rerun_tasks):
            idx, res = await fut
            if scorer is not None:
                res.score = _score(res)
            results[idx] = res

    # Cache-write audit: every fresh completion must be accounted for as a
    # confirmed write, a refused write, a provider-failure skip, or a
    # timeout/exception path. unaccounted > 0 = silent loss (the exact
    # signature of the 2026-07-22 nex evo run) — investigate immediately.
    if cache is not None:
        _fr = _cache_audit.get("fresh", 0)
        _acct = (_cache_audit.get("write_ok", 0) + _cache_audit.get("write_refused", 0)
                 + _cache_audit.get("skip_provider_failure", 0)
                 + _cache_audit.get("outer_timeout", 0) + _cache_audit.get("exception_path", 0))
        _unacct = _fr - _acct
        print(
            f"  [CACHE-AUDIT] fresh={_fr} hits={_cache_audit.get('hit', 0)} "
            f"writes_ok={_cache_audit.get('write_ok', 0) or 0} "
            f"write_refused={_cache_audit.get('write_refused', 0)}"
            f"(wall_limited={_cache_audit.get('write_refused_wall_limited', 0)}) "
            f"skip_provider_failure={_cache_audit.get('skip_provider_failure', 0)} "
            f"outer_timeout={_cache_audit.get('outer_timeout', 0)} "
            f"exception_path={_cache_audit.get('exception_path', 0)} "
            f"unaccounted={_unacct}", flush=True)
        if _unacct > 0:
            print(
                f"  [CACHE-AUDIT] *** WARNING: {_unacct} fresh completion(s) never reached "
                f"a cache-write decision — silent cache loss, do NOT trust resume for this batch ***",
                flush=True)

    return [r for r in results if r is not None]
