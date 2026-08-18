"""SpreadsheetBench v2 solver agent.

Same multi-turn Docker-based approach as SSB v1.
Each task gets an isolated workspace with the input spreadsheet.
The solver modifies it and saves to ./output/.
Inline scoring compares output cells against the golden answer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.harness.agent import Agent, AgentTrace
from src.harness.direct.options import apply_provider_pin, build_direct_options
from src.harness.utils import eval_run_ground_truth
from src.agent_profiles.ssbv2_agent import get_solver_prompt, SSBv2AgentResponse
from src.agent_profiles.ssbv2_agent.task_workspace import setup_task_workspace, get_output_dir
from src.agent_profiles.ssbv2_agent.scorer import score_and_diff
from src.harness.errors import UngradableEpisode


def _preserve_raw_output(output_file) -> None:
    """Copy the solver's workbook BEFORE ``score_and_diff`` mutates it.

    ``score_and_diff`` transforms ``proc_file`` IN PLACE — ``_add_xlfn_prefixes``,
    ``_force_iterative_calc``, ``_recalculate_with_libreoffice``,
    ``_solve_datatables_with_uno``, ``_restore_unrecalculable_cells`` — and
    ``snapshot_output`` runs afterwards. Without this, the only surviving copy of
    a rollout's output is the POST-recalculation one, and "what did the solver
    actually write?" is unanswerable from the archive.

    That gap had teeth. On 2026-08-01 two consecutive evolvers reached opposite
    conclusions about whether re-saving a workbook destroys cached values —
    epoch 2 shipped "never call wb.save()", epoch 3 deleted that rule for
    "re-saving is harmless, freeze everything to constants" — and neither could
    have settled it, because the pipeline they were diagnosing had already
    rewritten the only evidence they had. The orchestrator (me) then "refuted"
    epoch 3 from the same post-recalc file and was wrong too.

    WRITE-ONCE, deliberately. There are TWO ``score_and_diff`` call sites in this
    file: the in-span fast path and the ``inline_score is None`` fallback. If the
    first recalculates and then raises before setting ``inline_score``, the
    second runs against an already-recalculated file — so a last-write-wins copy
    would silently preserve exactly the wrong bytes. Earliest copy wins.

    Never raises: this is forensics, not scoring. Losing the raw copy must not
    fail a rollout.
    """
    import shutil
    try:
        out = Path(output_file)
        raw_dir = out.parent.parent / "output_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dst = raw_dir / out.name
        if not dst.exists():
            shutil.copy2(out, dst)
    except Exception as e:  # noqa: BLE001 — forensics must never break scoring
        import logging
        logging.getLogger(__name__).warning(
            f"raw-output preserve failed for {output_file}: {e}"
        )

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SSBV2_TASKS_CACHE = PROJECT_ROOT / ".cache" / "ssbv2_tasks"

# Single source of truth for the per-task solver wall-clock budget. The
# SSBv2SolverAgent below builds its own Agent(...) with an explicit
# timeout_seconds — that per-instance value overrides Agent.TIMEOUT_SECONDS
# set at the class level by run_ssbv2_evo.py, so bumping that env var alone
# silently had no effect on the actual per-task timeout (found Jul 5: the
# outer eval-loop wait_for and this inner Agent timeout must be changed
# together or the smaller one always wins).
SOLVER_TIMEOUT_SECONDS = int(os.environ.get("EVOSKILL_SOLVER_TIMEOUT_SECONDS", "3600"))


# SSB v2 builds options in TWO places — `_make_opts` (the outer agent) and
# `run()` (the inner Agent that actually executes the rollout). The 2026-07-30
# hard-pin fix was applied only to the first, which never executes a query, so
# `allow_fallbacks` was never sent at all. Both sites now call the SHARED
# helper, which also gates on the DeepSeek family; see its docstring.
_apply_provider_pin = apply_provider_pin


class SSBv2SolverAgent(Agent[SSBv2AgentResponse]):
    """Per-task-isolating solver for SSB v2."""

    def __init__(
        self,
        *,
        model: str,
        budget: int,
        shared_skills_dir: Path,
        provider_order: list[str] | None = None,
        thinking_effort: str = "",
        tasks_cache: Path | None = None,
        **kwargs,
    ):
        def _make_opts():
            opts = build_direct_options(
                system=get_solver_prompt(),
                schema=SSBv2AgentResponse.model_json_schema(),
                tools=["Bash", "Skill"],
                project_root=str(shared_skills_dir.parent.parent),
                model=model,
                budget_type="turns",
                budget_value=budget,
                bash_read_only=False,
                provider_order=provider_order or [],
                thinking_effort=thinking_effort,
            )
            opts["enable_thinking"] = True
            # Hard provider pin for the SOLVER, DeepSeek family only. Without
            # it `provider_order` is a PREFERENCE and OpenRouter falls through
            # to another host when the pinned one is busy. Note this is the
            # NON-executing options dict (see the module comment above) — the
            # pin that matters is the one in run().
            _apply_provider_pin(opts, model)
            # Artifact contract (2026-07-22, parity with SSB v1): score is
            # file-based, so no bonus answer turn at budget exhaustion (a
            # persisted [HARNESS] marker records it), no JSON-salvage turn,
            # in-span scoring even without a final assistant message,
            # BUDGET tag on the span title.
            opts["artifact_contract"] = True
            return opts

        super().__init__(
            options=_make_opts,
            response_model=SSBv2AgentResponse,
            **kwargs,
        )
        self.model = model
        self.budget = budget
        self.shared_skills_dir = shared_skills_dir
        self.provider_order = provider_order or []
        self.thinking_effort = thinking_effort
        self.tasks_cache = tasks_cache or SSBV2_TASKS_CACHE

    async def run(
        self, query: str, tag: str | None = None, sample_idx: int | None = None
    ) -> AgentTrace[SSBv2AgentResponse]:
        """Solve one task.

        ``sample_idx`` MUST be passed whenever N>1 samples of the same task can
        run concurrently. Without it every sample shares one workspace and one
        output file: siblings wipe each other's ``output/`` on entry, a sample
        can be scored against another's file or one caught mid-write, and a
        mid-write file makes the LibreOffice recalc silently produce nothing so
        formula cells read as ``None`` — a guaranteed 0 on possibly-correct
        work. Left as None (the default) for every ``n_samples=1`` call site,
        which keeps their on-disk layout byte-for-byte unchanged.
        """
        try:
            meta = json.loads(query)
        except (json.JSONDecodeError, TypeError):
            return AgentTrace(
                duration_ms=0, total_cost_usd=0.0, num_turns=0,
                usage={}, result="[ERROR] Invalid question JSON",
                is_error=True, output=None,
                parse_error="question JSON parse failed", messages=[],
            )

        task_id = meta["task_id"]
        spreadsheet_path = meta["spreadsheet_path"]
        spreadsheet_filename = meta["spreadsheet_filename"]
        instruction = meta["instruction"]

        from src.agent_profiles.ssbv2_agent.dataset import TASK_INDEX

        # Golden is deliberately NOT staged into the workspace — the solver's
        # container mounts this dir, and a Jul 1 trace audit caught solvers
        # opening the staged golden file mid-rollout (GT leakage). The scorer
        # reads golden from TASK_INDEX host paths; the evolver reads it via
        # its read-only dataset mount.
        task_ws = setup_task_workspace(
            task_id=task_id,
            spreadsheet_path=spreadsheet_path,
            tasks_cache_root=self.tasks_cache,
            shared_skills_dir=self.shared_skills_dir,
            sample_idx=sample_idx,
        )

        def _resolve_output_file() -> Path | None:
            # Same sample_idx as setup_task_workspace above — mismatching them
            # is exactly the bug this parameter exists to prevent.
            output_dir = get_output_dir(task_id, self.tasks_cache, sample_idx)
            expected_file = output_dir / spreadsheet_filename
            if expected_file.exists():
                return expected_file
            # Solver saved under an unexpected name — fall back to glob,
            # but sorted so a leftover LibreOffice temp/lock file doesn't
            # nondeterministically win depending on filesystem order.
            xlsx_files = sorted(output_dir.glob("*.xlsx"))
            return xlsx_files[0] if xlsx_files else None

        # Scoring computes the full cell diff anyway; stash it here so the
        # snapshot step can persist it as DIFF.txt for the evolver (see
        # score_and_diff / snapshot_output).
        _diff_holder: list[str] = []

        def _diff_header(passed: bool, n_match: int, n_total: int, diff: str) -> str:
            return (
                f"{'PASS' if passed else 'FAIL'} — {n_match}/{n_total} graded cells match\n"
                f"(scorer's own diff, computed at scoring time; full, uncapped)\n\n{diff}\n"
            )

        def _score_fn(_query: str, _answer: str, _gt: str) -> float:
            # Runs inside the low-level executor, BEFORE the trace/span is
            # finalized (see direct/executor.py's score_fn hook) — this is
            # what lets Phoenix show the real [OK]/[WRONG] label instead of
            # falling back to scoring the raw, unresolved final_answer text
            # (which for a file-based benchmark is a relative path that
            # can't be checked for existence from the harness's own cwd).
            try:
                output_file = _resolve_output_file()
                if output_file is None:
                    _diff_holder.append(_diff_header(False, 0, 0, "Output file not found"))
                    return 0.0
                info = TASK_INDEX.get(task_id)
                if not info:
                    return 0.0
                _preserve_raw_output(output_file)   # BEFORE the in-place recalc
                passed, n_match, n_total, diff = score_and_diff(
                    info["golden_path"], str(output_file), info["answer_position"],
                    orig_input_file=spreadsheet_path,
                )
                _diff_holder.append(_diff_header(passed, n_match, n_total, diff))
                return 1.0 if passed else 0.0
            except UngradableEpisode:
                raise      # infra fault, not a wrong answer -- src/harness/errors.py
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Inline SSBv2 scoring failed for {task_id}: {e}")
                return 0.0

        opts = build_direct_options(
            system=get_solver_prompt(),
            schema=SSBv2AgentResponse.model_json_schema(),
            tools=["Bash", "Skill"],
            project_root=str(task_ws),
            model=self.model,
            budget_type="turns",
            budget_value=self.budget,
            bash_read_only=False,
            provider_order=self.provider_order,
            thinking_effort=self.thinking_effort,
        )
        opts["enable_thinking"] = True
        opts["score_fn"] = _score_fn
        opts["artifact_contract"] = True  # see _make_opts note (2026-07-22)
        # THIS is the dict that actually executes the rollout (see the inner
        # Agent below), so the hard pin has to be applied HERE, not only in
        # _make_opts. Omitting it left `order` as a preference and OpenRouter
        # served SSB v2 rollouts from whichever host was free.
        _apply_provider_pin(opts, self.model)

        # Agent.TIMEOUT_SECONDS defaults to 720s (tuned for CC-harness/fast
        # models) — every SSB v2 launch script wraps this call in its own
        # asyncio.wait_for at the module-level SOLVER_TIMEOUT_SECONDS, so
        # that (not 720s) is the actual intended per-task budget. Without
        # this override the 720s default fires first and silently caps
        # every run below its documented budget — latent for fast models
        # (DSV4F rarely reasons long enough to hit it), but real: found via
        # GLM 5.2 timing out at 720s with only 13/120 turns used, "not
        # retrying" per Agent's own by-design over-budget handling.
        solver = Agent(lambda: opts, SSBv2AgentResponse, name="solver", timeout_seconds=SOLVER_TIMEOUT_SECONDS)
        solver.sdk_override = "direct"

        solver_instruction = (
            f"Task: {task_id}\n\n"
            f"{instruction}\n\n"
            f"Input spreadsheet: ./spreadsheets/{spreadsheet_filename}\n"
            f"Save your modified file to: ./output/{spreadsheet_filename}\n"
        )

        # The executor's fast-path in-span scorer (gives Phoenix the
        # [OK]/[WRONG] span label + Events entry) only fires when
        # eval_ctx.ground_truth is non-empty — _score_fn above ignores its
        # own _gt arg and does real file-based comparison via closures, so
        # any non-empty value unblocks it. Use task_id: human-readable in
        # the eval.ground_truth span attribute, no golden content exposed.
        # Token-based set/reset keeps this scoped to this task's own async
        # context, not leaked across concurrent sibling tasks.
        _gt_token = eval_run_ground_truth.set(task_id)
        try:
            trace = await solver.run(solver_instruction, tag=tag)
        finally:
            eval_run_ground_truth.reset(_gt_token)

        # Fallback: score_fn only runs when the executor has ground_truth +
        # a non-empty final_answer wired through; if either was missing,
        # inline_score is still None here and we score the same way as a
        # last resort (also skipped entirely on a cache hit). Guarded on
        # trace.inline_score is None, so once the fast path above sets it,
        # this never re-runs score_and_diff a second time for the same task.
        if trace.inline_score is None:
            try:
                output_file = _resolve_output_file()
                if output_file is not None:
                    info = TASK_INDEX.get(task_id)
                    if info:
                        _preserve_raw_output(output_file)   # BEFORE the in-place recalc
                        passed, n_match, n_total, diff = score_and_diff(
                            info["golden_path"], str(output_file), info["answer_position"],
                            orig_input_file=spreadsheet_path,
                        )
                        _diff_holder.append(_diff_header(passed, n_match, n_total, diff))
                        trace.inline_score = 1.0 if passed else 0.0
            except UngradableEpisode:
                raise      # infra fault, not a wrong answer -- src/harness/errors.py
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Inline SSBv2 scoring failed for {task_id}: {e}")

        # Snapshot output for cross-iteration comparison (independent of
        # which scoring path ran, and skipped on a cache hit since there's
        # no fresh output to snapshot). The scorer's diff rides along as
        # DIFF.txt so the evolver never re-computes comparisons.
        if trace.inline_score is not None:
            try:
                import re
                iter_match = re.search(r'iter.?(\d+)', tag or '')
                iter_num = int(iter_match.group(1)) if iter_match else 0
                _phase = "midgate" if "midgate" in (tag or "").lower() else ""
                from src.agent_profiles.ssbv2_agent.task_workspace import snapshot_output
                snapshot_output(
                    task_id, self.tasks_cache, iter_num,
                    diff_text=_diff_holder[-1] if _diff_holder else None,
                    phase=_phase,
                    sample_idx=sample_idx,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Output snapshot failed for {task_id}: {e}")

        return trace
