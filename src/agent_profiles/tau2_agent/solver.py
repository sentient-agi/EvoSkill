"""tau2-bench solver -- drop-in `Agent` replacement.

Drives one tau2-bench episode by spawning a subprocess worker
(`_tau2_worker.py`) inside tau2-bench's own isolated venv
(`third_party/tau2-bench/.venv`), via
`uv run --project <tau2_root> python _tau2_worker.py`.

CONFIRMED DEVIATION from the original design doc's §7/judgment-call #4: the
design recommended merging tau2-bench directly into EvoSkill's own venv
(the ALFWorld precedent: `import alfworld.agents.environment` straight
from `alfworld_agent/episode.py`) and driving `run_single_task` via
`asyncio.to_thread`. This build instead keeps tau2-bench fully isolated
and drives it via a subprocess, because:

  1. tau2-bench's `--all-extras` dependency surface (required even for
     pure text-mode use -- `tau2/__init__.py` unconditionally imports the
     voice stack) pulls in elevenlabs, deepgram-sdk, pyaudio, boto3,
     google-cloud-aiplatform, gymnasium, scikit-learn, jiwer, and more --
     none of it relevant to the retail domain this build targets, and all
     of it a real conflict/bloat risk for the SAME production venv every
     other EvoSkill benchmark (ALFWorld, TB2, SWE-rebench, ...) shares for
     live evolution runs.
  2. A subprocess boundary satisfies every functional requirement the
     design's in-process design was solving for -- skill injection
     (passed as a request field, see `agent_wiring.py`), the cost-tracking
     `generate()` monkeypatch (installed inside the worker process, see
     `cost_tracking.py`), and the `run_single_task` call itself -- without
     needing `ContextVar`/`threading.local` coordination across an
     `asyncio.to_thread` boundary, because each task gets its own OS
     process rather than sharing a thread pool.
  3. Real concurrency comes from multiple OS processes (via
     `asyncio.create_subprocess_exec`, naturally async, no `to_thread`
     needed) rather than thread scheduling inside one interpreter -- at
     least as good a fit for `evaluate_agent_parallel`'s concurrency
     semaphore as the design's own proposal.

Trade-off: per-episode `uv run` startup overhead (~1-2s, empirically) on
top of tau2's own multi-turn LLM round-trip latency -- negligible in
practice, confirmed during this build's smoke tests.

See `third_party/README.md`'s tau2-bench section for the vendoring
location/provenance and `agent_wiring.py`'s module docstring for a SECOND
confirmed deviation (inline skill content, not a progressive-disclosure
tool call -- structurally incompatible with tau2's own DB-check replay
evaluator).

4-domain extension (retail/airline/telecom/banking_knowledge) notes:
  - `domain` already travels PER-TASK inside the `question` JSON payload
    (`_run_impl` below reads `meta.get("domain", self.domain)`), so a single
    `Tau2Solver` instance can drive episodes across all 4 domains in one
    run (e.g. a multi-domain self-evolution loop) -- no per-call solver
    change was needed for the execution path itself.
  - `self.domain` (the CONSTRUCTOR-level default) still matters for the
    state loaded once at `__init__` time as a fallback/cache-key input:
    `self._policy_text` (`_load_policy_text()`) and, formerly, a single
    fixed `self._tier_manifest`. The tier-manifest is now looked up
    PER-TASK-DOMAIN (`_manifest_for_domain`, lazily cached), not read from a
    single fixed attribute -- retail and airline both use bare numeric
    task IDs ("0".."N"), confirmed via direct inspection, so a single
    domain-fixed manifest would silently misclassify user-sim tier for
    whichever domain ISN'T `self.domain` when their IDs collide (e.g. a
    retail-tier verdict for airline task "5"). This is a genuine, verified
    hazard for a multi-domain single-instance run, not a hypothetical.
  - `_load_policy_text()` has domain-specific branches for `telecom`
    (no `policy.md` -- concatenates `main_policy.md` + the "manual"
    tech-support file into the SAME `<main_policy>`/`<tech_support_policy>`
    wrapper tau2's own `telecom/environment.py:get_environment()` builds)
    and `banking_knowledge` (no `policy.md` -- policy is retrieval-variant
    AND task-specific, resolved live inside the worker; this fallback just
    reads the variant's static prompt TEMPLATE verbatim, ignoring the
    task-specific golden-doc substitution, since it is used only as a
    construction-time placeholder / RunCache cache-key input -- the byte-
    exact real prompt is always recovered post-hoc from the worker's own
    `SystemMessage`, see `_tau2_worker._extract_system_prompt`).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import signal
import tempfile
import sys
import time
from pathlib import Path
from typing import Any

from opentelemetry import trace as otel_trace

from src.agent_profiles.tau2_agent.dataset import load_raw_tasks
from src.agent_profiles.tau2_agent.user_tier_gate import get_tier, load_manifest
from src.harness._tracing import _set_oi_messages
from src.harness.agent import AgentTrace
from src.schemas import AgentResponse

_tracer = otel_trace.get_tracer("tau2_solver")

_WORKER_PATH = Path(__file__).parent / "_tau2_worker.py"

# banking_knowledge only: retrieval-variant name -> its static prompt TEMPLATE
# file (`<tau2_root>/data/tau2/domains/banking_knowledge/prompts/`). Used ONLY by
# `_load_policy_text()`'s construction-time / cache-key text (see the module
# docstring) -- NOT a re-implementation of tau2's own variant resolution
# (`resolve_variant`/`build_policy`), which remains the sole source of truth for
# what the real episode runs.
#
# The previous comment here described a hardcoded 2-entry dict that was "not
# exhaustive" and fell back to "a generic non-crashing placeholder ... rather
# than raising". Both halves of that are now false: the table is derived from
# tau2's own and covers every variant, and an unresolvable variant RAISES. A
# comment left describing removed behaviour is the same defect as a legend
# promising absent pins — it is read as documentation and it lies.
def _banking_variant_prompt_files(tau2_root: Path) -> dict[str, str]:
    """variant name -> prompt-template filename, DERIVED FROM TAU2'S OWN TABLE.

    This replaced a hand-maintained 2-entry dict (`bm25`, `bm25_reranker`)
    that silently mis-resolved every other variant. banking_knowledge ships
    21 variants; the lookup fell through to `f"{variant}.md"`, which does not
    exist for any of them, and `_load_policy_text` then returned a PLACEHOLDER
    string. Measured 2026-08-04 on the pinned variant `bm25_grep`: the policy
    EvoSkill believed the solver had was the placeholder, while tau2's own
    table maps it to `classic_rag_bm25.md`.

    That mattered twice over. The placeholder text is (a) an input to the
    RunCache tree_hash, so every variant hashed IDENTICALLY and the cache
    could replay one variant's rollouts for another — the same
    variant-not-in-the-cache-key hole documented as tau3 trap #2, reached by
    a second route; and (b) reachable by the evolver as "the solver's policy".

    Parsed rather than imported: importing tau2 drags its --all-extras surface
    into EvoSkill's venv (the reason the original copy existed). Parsed rather
    than copied: a copy of another project's table is a copy that drifts, and
    this one already had.
    """
    src = (tau2_root / "src" / "tau2" / "domains" / "banking_knowledge"
           / "retrieval.py")
    text = src.read_text(encoding="utf-8")
    block = re.compile(
        r'"(?P<name>[a-z0-9_]+)":\s*RetrievalVariant\((?P<body>.*?)\n    \),',
        re.S)
    tmpl = re.compile(r'prompt_template=PROMPTS_DIR\s*/\s*"(?P<file>[^"]+)"')
    out: dict[str, str] = {}
    for m in block.finditer(text):
        t = tmpl.search(m.group("body"))
        if t:
            out[m.group("name")] = t.group("file")
    if not out:
        # Shape changed upstream. Refuse loudly rather than hand back an empty
        # map that would send every variant down the placeholder path again.
        raise RuntimeError(
            f"could not parse any RetrievalVariant prompt_template out of "
            f"{src} — tau2's table shape changed; fix the parser rather than "
            f"reinstating a hardcoded copy")
    return out

# tau2's own SYSTEM_PROMPT/AGENT_INSTRUCTION template + text
# (`src/tau2/agent/llm_agent.py:23-41` in the pinned v1.0.0 checkout),
# reproduced here so `_get_options()["system"]` reflects the REAL prompt
# tau2 builds (domain policy + agent instruction), keeping RunCache's
# cache key sensitive to skill changes without needing to import tau2 (and
# its heavy --all-extras surface) into EvoSkill's own venv just to read
# two string constants. If tau2 changes this template upstream, this
# reconstruction can drift -- acceptable given the vendored clone is
# pinned to a fixed commit (17e07b1, tag v1.0.0).
_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

_SYSTEM_PROMPT_TEMPLATE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def _normalize_model_id(model: str) -> str:
    """Strip the "openrouter/" routing prefix and ":nitro"/":floor" routing
    suffixes so two model strings that resolve to the same underlying model
    compare equal (mirrors `cost_tracking._strip_routing_suffix`'s
    convention; duplicated rather than imported because `cost_tracking.py`
    is documented to run ONLY inside the tau2 worker subprocess venv, while
    this module runs in EvoSkill's own venv)."""
    base = model[len("openrouter/"):] if model.startswith("openrouter/") else model
    for suffix in (":nitro", ":floor"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def _load_skill_block(workspace: str | Path) -> str:
    """Concatenate all skill .md files under workspace/.claude/skills/ into
    one block. NOT used for the live episode's system prompt anymore (see
    `_load_skills` + agent_wiring.py's real progressive-disclosure tool
    mechanism) -- kept ONLY as `_get_options()`'s RunCache tree-hash input
    (a stable, full-content-sensitive string so the cache key still busts
    on any skill body edit, not just a name/description change)."""
    skills_dir = Path(workspace) / ".claude" / "skills"
    if not skills_dir.is_dir():
        return ""
    parts = []
    for md in sorted(skills_dir.rglob("*.md")):
        content = md.read_text(encoding="utf-8").strip()
        if content:
            parts.append(content)
    if not parts:
        return ""
    return "## Skills\n\n" + "\n\n---\n\n".join(parts)


def _load_skills(workspace: str | Path) -> dict[str, dict[str, str]]:
    """Parse each skill .md file under workspace/.claude/skills/ into
    {name: {"name", "description", "content"}} -- name/description come
    from the file's frontmatter (falling back to the containing directory
    name / a generic description), mirroring
    scripts/run_swe_rebench_evo.py's `SWERebenchSolver._load_skills`
    exactly, so both harnesses parse skill files the same way. Used for
    REAL progressive disclosure (agent_wiring.py): only name+description
    go in the system prompt; `content` is resolved on demand by the
    `skill` tool call, not injected up front."""
    skills: dict[str, dict[str, str]] = {}
    skills_dir = Path(workspace) / ".claude" / "skills"
    if not skills_dir.is_dir():
        return skills
    for md in sorted(skills_dir.rglob("*.md")):
        content = md.read_text(encoding="utf-8").strip()
        if not content:
            continue
        name = md.parent.name
        description = ""
        if content.startswith("---"):
            end = content.find("---", 3)
            if end > 0:
                for line in content[3:end].splitlines():
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line[5:].strip().strip("'\"")
                    elif line.startswith("description:"):
                        description = line[12:].strip().strip("'\"")
        skills[name] = {
            "name": name,
            "description": description or f"Skill from {md.parent.name}",
            "content": content,
        }
    return skills


# HARD PIN BY DEFAULT for this solver. `pin_request_body_provider` only writes
# allow_fallbacks when EVOSKILL_SOLVER_ALLOW_FALLBACKS is set; with it unset a DeepSeek
# model gets `{"order": ["deepseek"]}` and NOTHING else -- a PREFERENCE that OpenRouter
# still reroutes, and which is indistinguishable from a real pin in every log we emit.
# Verified: env unset -> {'order': ['deepseek']}; env=0 -> adds allow_fallbacks=False.
# Set at import so it holds however this solver is driven, not only via the tau3 runner.
os.environ.setdefault("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")


def _provider_pin_for(model: str) -> dict | None:
    """The OpenRouter `provider` block for `model`, or None to leave routing alone.

    Delegates to the ONE shared implementation so the family gate and the
    allow_fallbacks contract cannot drift: DeepSeek gets a HARD pin (the standing
    directive), everything else -- notably GLM 5.2 on :nitro -- keeps its multi-provider
    routing. Returns None rather than an empty dict so the worker can tell "no pin" from
    "pin with no fields".

    EVOSKILL_SOLVER_PROVIDER_ORDER overrides the family rule with an explicit,
    comma-separated provider list and hard-pins it (order + allow_fallbacks). This exists for
    CROSS-FAMILY work: a non-DeepSeek solver otherwise keeps multi-provider routing, and for a
    measurement that is not a neutral default. `inclusionai/ling-3.0-flash` is the case that
    forced it -- its two endpoints differ in CONTEXT LENGTH (novita 262k vs deepinfra 131k) as
    well as price, so unpinned routing gives different episodes different context limits inside
    one leg, and #169 means the served provider is not recorded, so the resulting mix is not
    even detectable afterwards. Unset = previous behaviour exactly.
    """
    try:
        from src.harness.direct.options import pin_request_body_provider
    except Exception:
        return None
    explicit = [x.strip() for x in
                os.environ.get("EVOSKILL_SOLVER_PROVIDER_ORDER", "").split(",") if x.strip()]
    if explicit:
        return pin_request_body_provider({"model": model}, explicit, hard=True).get("provider")
    body = pin_request_body_provider({"model": model}, ["deepseek"]
                                     if "deepseek" in (model or "").lower() else None)
    return body.get("provider")

class Tau2Solver:
    """Drop-in `Agent` replacement that drives a tau2-bench episode.

    Exposes `.run(question, tag=None) -> AgentTrace[AgentResponse]`,
    `._get_options()`, `.name`, `.sdk_override = "direct"`,
    `.response_model = AgentResponse` -- the exact shape
    `evaluate_agent_parallel`'s auto-cache path
    (`src/evaluation/evaluate.py`) expects, matching `ALFWorldSolver`/
    `TB2Solver`. No changes to `src/evaluation/evaluate.py`,
    `src/loop/runner.py`, or `src/harness/agent.py` are needed.
    """

    TIMEOUT_SECONDS = 900

    def __init__(
        self,
        *,
        model: str,
        workspace: str | Path,
        domain: str = "retail",
        # gpt-5-nano ROLE-PLAYS THE USER for tau3. the maintainer, 2026-08-06 — this SUPERSEDES
        # the 0804b §E3 decision to keep gpt-4.1 for the reference tier. Both tiers are now
        # nano, so the tier gate is inert by intent rather than by the accidental collapse
        # that happened earlier today.
        # CONSEQUENCE, and it is not small: the user simulator generates half of every
        # episode, so EVERY cached tau3 episode was produced by a different instrument and
        # is invalid for reuse. #183's sidecar guard is exactly what catches this — it will
        # REFUSE the existing cache dirs rather than silently replay gpt-4.1 conversations
        # as if they were nano's. New runs need fresh cache dirs, and no number measured
        # before this change is comparable with one measured after it.
        reference_user_model: str = "openrouter/openai/gpt-5-nano",
        cheap_user_model: str = "openrouter/openai/gpt-5-nano",
        judge_model: str = "openrouter/openai/gpt-4.1",
        tau2_root: str | Path | None = None,
        max_steps: int = 200,
        max_errors: int = 10,
        episode_timeout: int = 460,
        inject_skills: bool = True,
        retrieval_config: str | None = None,
    ):
        self.model = model
        self.workspace = Path(workspace)
        self.domain = domain
        self.reference_user_model = reference_user_model
        self.cheap_user_model = cheap_user_model
        self.judge_model = judge_model
        self.tau2_root = Path(tau2_root) if tau2_root else (
            Path(__file__).resolve().parents[3] / "third_party" / "tau2-bench"
        )
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.episode_timeout = episode_timeout
        self.inject_skills = inject_skills
        # banking_knowledge ONLY -- which retrieval variant builds the
        # agent's knowledge-base policy/tools for that domain (see
        # `_run_impl`'s `_resolve_retrieval_config`). None (the default)
        # means "let `_run_impl` fall back to tau2's own documented default
        # ('bm25', `DEFAULT_RETRIEVAL_VARIANT` in
        # `banking_knowledge/retrieval.py`) for banking_knowledge tasks, and
        # do nothing at all for every other domain" -- completely inert for
        # retail/airline/telecom regardless of this value.
        self.retrieval_config = retrieval_config

        self.response_model = AgentResponse
        self.name = "tau2_solver"
        self.sdk_override = "direct"
        # Mirror the base `Agent` class's `self.timeout_seconds` instance
        # attribute (lowercase) -- NOT redundant with the `TIMEOUT_SECONDS`
        # class attr above. `src/evaluation/evaluate.py`'s
        # `evaluate_agent_parallel` (the val-gate/mid-gate path every
        # `SelfImprovingLoop` iteration runs through) computes its OWN outer
        # timeout as `getattr(agent, 'timeout_seconds', 720) + 120` --
        # confirmed by reading that module. Without this attribute, that
        # `getattr` silently falls through to the hardcoded 720s default
        # (720+120=840s) instead of whatever a launcher script overrides
        # `Tau2Solver.TIMEOUT_SECONDS` to (e.g. 3000s) -- a real, confirmed
        # bug found during pre-launch review: reasoning-heavy multi-turn
        # tau2 episodes can legitimately run past 840s, and every val
        # evaluation would silently truncate them at that shorter budget
        # regardless of the launcher's intended timeout. Setting this here
        # (read AFTER any class-attr override, since it's evaluated at
        # instance-construction time) fixes this for both the constructor
        # default (900s) and any launcher's `Tau2Solver.TIMEOUT_SECONDS = X`
        # override performed before construction.
        #
        # THE OUTER WALL MUST EXCEED THE INNER ONE, AND IS NOW DERIVED SO IT
        # CANNOT DRIFT. Measured 2026-08-04 on tau3 banking_knowledge: a run
        # launched with EVOSKILL_TAU2_EPISODE_TIMEOUT=1800 still lost 4 of 18
        # episodes at EXACTLY 900.0s, because `episode_timeout` is handed to
        # tau2's own simulation while THIS value bounds the whole coroutine.
        # The tighter one binds, silently, and raising only the inner knob was
        # completely inert.
        #
        # The two walls are not interchangeable, which is why the ordering
        # matters rather than just the size. The INNER wall fires inside tau2
        # and yields a real result: `termination_reason: "timeout"`, a reward,
        # and a cost. The OUTER wall is `asyncio.wait_for` and yields an error
        # trace with NO termination_reason, NO cost and NO score — which is
        # exactly the signature those 4 episodes had, and it is unattributable
        # after the fact. So the outer must always leave the inner room to fire
        # and be reported: inner + margin, never below the 900s default.
        #
        # THE MARGIN HAS TWO JOBS, AND ONLY ONE OF THEM WAS SIZED (#175, fixed
        # 2026-08-06). It must cover:
        #   (1) what happens AFTER the simulation returns but still inside the
        #       coroutine — NL-assertion judging, scoring, trace assembly. Judging
        #       is an LLM call, so it is not negligible. This is the 300s part, and
        #       it was the only part the previous version accounted for.
        #   (2) THE WAIT UNTIL THE INNER WALL CAN EVEN BE OBSERVED. tau2 evaluates
        #       `timeout` in exactly one place — `_check_timeout()`, called only
        #       from `_check_half_duplex_termination` (tau3-bench
        #       `orchestrator/orchestrator.py:750`), which also returns early while
        #       `to_role == Role.ENV`. So the inner wall fires at TURN BOUNDARIES
        #       only. Once it is exceeded mid-turn, nothing happens until that turn
        #       completes, and a hung call chain makes that up to
        #       `worst_case_turn` = (retries+1) x per_call away.
        #
        # With the old numbers that second term was 1800s against a 300s margin, so
        # the ordering this block's comment asserts ("the outer must always leave
        # the inner room to fire") was VACUOUS — 6x too small. Measured cost: 3
        # episodes killed by the outer wall at 2100s with no termination_reason, no
        # cost and no score, each blocking its leg's result JSON, on tasks where
        # tau2's own rule would have scored a plain 0.0.
        #
        # Deriving term (2) from the same function the worker configures litellm
        # with means the two cannot drift; hardcoding it here is what broke.
        from ._tau2_protocol import llm_call_budget as _llm_budget
        _per_call, _retries, _worst_turn = _llm_budget(float(episode_timeout))
        _post_sim_s = int(os.environ.get("EVOSKILL_TAU2_OUTER_MARGIN_S", "300"))
        _outer_margin_s = int(_worst_turn) + _post_sim_s
        self.timeout_seconds = max(self.TIMEOUT_SECONDS,
                                   int(episode_timeout) + _outer_margin_s)
        # Recorded so a leg's banner/result can show the ordering that is actually
        # in force rather than the one a reader assumes from the two knobs.
        self._wall_budget = {
            "inner_episode_timeout_s": int(episode_timeout),
            "llm_per_call_s": _per_call, "llm_retries": _retries,
            "worst_case_turn_s": _worst_turn, "post_sim_margin_s": _post_sim_s,
            "outer_wall_s": self.timeout_seconds,
        }
        # A GUARD, NOT A MEASUREMENT. On the default path this holds by
        # construction, so reporting it as a boolean field would be a check that
        # cannot fail (#154). It is written as an assertion because it CAN be
        # broken from outside — a negative EVOSKILL_TAU2_OUTER_MARGIN_S, a launcher
        # pinning TIMEOUT_SECONDS below the derived value, or a future edit to
        # either budget — and the failure mode it guards against is silent: not a
        # crash, just episodes that come back unscoreable. Fail at construction
        # instead, before any rollout is spent.
        assert int(episode_timeout) + _worst_turn < self.timeout_seconds, (
            f"outer wall {self.timeout_seconds}s cannot observe inner wall "
            f"{episode_timeout}s: a turn may run {_worst_turn:.0f}s "
            f"({_retries}+1 x {_per_call:.0f}s) past it, and tau2 only checks the "
            f"inner wall at turn boundaries (#175)")

        self._raw_tasks = load_raw_tasks(domain, tau2_root=self.tau2_root)
        # Per-task-domain tier-manifest cache (NOT a single fixed
        # `self._tier_manifest`): retail and airline both use bare numeric
        # task IDs ("0".."N", confirmed via direct inspection), so a single
        # domain-fixed manifest would silently misclassify user-sim tier
        # whenever a multi-domain run (one `Tau2Solver` instance driving
        # several domains via the per-task `question` payload, see module
        # docstring) hits an ID that collides across domains. Preload
        # `self.domain`'s own manifest eagerly (same eager-load behavior as
        # before, for any existing single-domain caller); other domains'
        # manifests are loaded lazily, on first use, by `_manifest_for_domain`.
        self._tier_manifest_cache: dict[str, dict] = {domain: load_manifest(domain)}
        self._policy_text = self._load_policy_text()

        # Self-play detection (measurement-validity requirement): the tau2
        # leaderboard's reference config uses DIFFERENT models for agent vs
        # user-sim; some community submissions (e.g. GLM-5.2's own
        # submission) use the SAME model for both ("self-play") and flag it
        # explicitly as non-standard. Previously nothing in this build
        # detected or surfaced this -- a solver model that happened to
        # collide with `reference_user_model`/`cheap_user_model` would
        # silently look like a normal run. Checked once at construction
        # time (cheap, and both user models are fixed per-run) and again
        # surfaced per-episode via a span attribute (see `_run_impl`).
        self.self_play_flags: list[str] = self._check_self_play()
        if self.self_play_flags:
            print(
                f"[tau2_solver] WARNING: self-play detected -- solver model "
                f"'{self.model}' resolves to the same underlying model as "
                f"{', '.join(self.self_play_flags)}. This is NON-STANDARD "
                f"vs. the tau2 leaderboard reference config (agent and "
                f"user-sim are normally different models) -- flag any "
                f"results from this run as self-play, not directly "
                f"comparable to standard leaderboard entries.",
                file=sys.stderr, flush=True,
            )

    def _check_self_play(self) -> list[str]:
        agent_norm = _normalize_model_id(self.model)
        flags = []
        if agent_norm == _normalize_model_id(self.reference_user_model):
            flags.append(f"reference_user_model={self.reference_user_model}")
        if agent_norm == _normalize_model_id(self.cheap_user_model):
            flags.append(f"cheap_user_model={self.cheap_user_model}")
        return flags

    def _manifest_for_domain(self, domain: str) -> dict[str, dict]:
        """Per-task-domain tier-manifest, lazily loaded and cached (see
        `__init__`'s docstring comment on why a single fixed manifest is
        unsafe once one `Tau2Solver` instance can drive multiple domains)."""
        manifest = self._tier_manifest_cache.get(domain)
        if manifest is None:
            manifest = load_manifest(domain)
            self._tier_manifest_cache[domain] = manifest
        return manifest

    def _load_policy_text(self) -> str:
        domains_root = self.tau2_root / "data" / "tau2" / "domains"

        if self.domain == "telecom":
            # No `policy.md` for telecom -- tau2's own
            # `telecom/environment.py:get_environment()` (policy_type=
            # "manual", which is what domain name "telecom" registers to,
            # confirmed via `registry.py`) concatenates `main_policy.md` +
            # `tech_support_manual.md` into this exact wrapper at runtime.
            main_policy = (domains_root / "telecom" / "main_policy.md").read_text(encoding="utf-8")
            tech_support_policy = (domains_root / "telecom" / "tech_support_manual.md").read_text(encoding="utf-8")
            return (
                "<main_policy>\n" + main_policy + "\n</main_policy>\n"
                "<tech_support_policy>\n" + tech_support_policy + "\n</tech_support_policy>"
            )

        if self.domain == "banking_knowledge":
            # No `policy.md` -- the real policy is retrieval-variant AND
            # task-specific (`build_policy(variant, knowledge_base, task)`),
            # resolved live inside the worker subprocess. This is a
            # construction-time placeholder ONLY (cache-key input / defensive
            # fallback -- see module docstring): reads the variant's static
            # prompt TEMPLATE file verbatim, ignoring the task-specific
            # golden-doc substitution `golden_retrieval`-style variants do.
            variant = self.retrieval_config or "bm25"
            table = _banking_variant_prompt_files(self.tau2_root)
            if variant not in table:
                raise RuntimeError(
                    f"banking_knowledge variant {variant!r} has no "
                    f"prompt_template in tau2's own RETRIEVAL_VARIANTS table "
                    f"({len(table)} parsed). Known: {sorted(table)[:8]}...")
            prompt_path = (domains_root / "banking_knowledge" / "prompts"
                           / table[variant])
            if not prompt_path.exists():
                raise RuntimeError(
                    f"variant {variant!r} maps to {table[variant]} but that "
                    f"file is absent at {prompt_path}")
            # REFUSE, never placeholder. The old code returned a placeholder
            # string here, and because this value feeds the RunCache tree_hash
            # a placeholder makes every variant hash the same — a silent
            # cross-variant cache collision. Raising costs a crash at
            # construction; the placeholder cost a wrong measurement nobody
            # could see. Note this is the CONSTRUCTION-TIME reconstruction and
            # is deliberately variant-static: it does NOT include the
            # task-specific golden-doc substitution that `golden_retrieval`
            # applies per task. The byte-exact prompt the solver really ran is
            # captured post-hoc from the worker's own SystemMessage — see
            # `_tau2_worker._extract_system_prompt`, and prefer THAT whenever
            # showing a human or the evolver "the solver's prompt".
            # EXPAND `{{component:NAME}}` the way tau2 does, so this text is the
            # policy the solver really gets rather than a template of it.
            # Mirrors `retrieval.load_prompt_template`: single pass,
            # `components/NAME.md` verbatim. Found by reading render 4's output —
            # the fix that stopped showing a retail prompt left TWO raw macros
            # (`policy_header`, `additional_instructions`) sitting in the policy
            # the evolver was handed.
            #
            # `{{full_knowledge_base}}` is deliberately NOT expanded: tau2 fills
            # it from a live KnowledgeBase object, it is ~698 documents, and
            # inlining a corpus here would both dwarf the prompt and put the
            # whole KB into the cache key. `full_kb` is not a variant we run.
            _text = prompt_path.read_text(encoding="utf-8")
            _cdir = prompt_path.parent / "components"

            def _sub(m: "re.Match[str]") -> str:
                f = _cdir / f"{m.group(1)}.md"
                if not f.exists():
                    # Loud, not silent: a missing component would otherwise leave
                    # a literal `{{component:x}}` in text presented as policy.
                    raise RuntimeError(
                        f"variant {variant!r} references component "
                        f"{m.group(1)!r} but {f} is absent")
                return f.read_text(encoding="utf-8")

            _text = re.sub(r"\{\{component:(\w+)\}\}", _sub, _text)
            return _text

        policy_path = domains_root / self.domain / "policy.md"
        return policy_path.read_text(encoding="utf-8")

    def _get_options(self) -> dict[str, Any]:
        return {"model_id": self.model, "system": self._build_system_prompt()}

    def _build_system_prompt(self) -> str:
        base = _SYSTEM_PROMPT_TEMPLATE.format(
            domain_policy=self._policy_text, agent_instruction=_AGENT_INSTRUCTION,
        )
        if self.inject_skills:
            skill_block = _load_skill_block(self.workspace)
            if skill_block:
                base = base + "\n\n" + skill_block
        return base

    async def run(self, question: str, tag: str | None = None) -> AgentTrace[AgentResponse]:
        t0 = time.monotonic()
        try:
            # `self.timeout_seconds`, NOT the class attr. They diverged: __init__
            # derives the instance value from episode_timeout (so the outer wall
            # always clears the inner one) while this line read the raw 900s class
            # default and ignored it. `evaluate_agent_parallel` already reads the
            # instance attribute, so the two paths disagreed about the budget —
            # the val gate honoured the derived value and the solver's own
            # coroutine did not.
            return await asyncio.wait_for(self._run_impl(question, tag, t0), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return self._error_trace(f"Episode timed out after {self.timeout_seconds}s (OUTER wall; inner episode_timeout={self.episode_timeout}s)", t0)

    async def _run_impl(self, question: str, tag: str | None, t0: float) -> AgentTrace[AgentResponse]:
        try:
            meta = json.loads(question)
            task_id = meta["task_id"]
            domain = meta.get("domain", self.domain)
        except (json.JSONDecodeError, KeyError) as e:
            return self._error_trace(f"Bad question JSON: {e}", t0)

        # Per-TASK-domain manifest lookup -- NOT `self._tier_manifest` (see
        # `__init__`'s comment): retail/airline share bare numeric task IDs,
        # so a domain-fixed manifest would silently cross-classify tier for
        # whichever domain isn't `self.domain` in a multi-domain run.
        tier = get_tier(domain, task_id, self._manifest_for_domain(domain))
        user_model = self.reference_user_model if tier == "reference" else self.cheap_user_model
        skills = _load_skills(self.workspace) if self.inject_skills else {}

        # banking_knowledge ONLY -- resolve which retrieval variant builds
        # the agent's knowledge-base policy/tools this episode. Every other
        # domain leaves this None, which `_tau2_worker.py` passes straight
        # through to `TextRunConfig(retrieval_config=None, ...)` -- confirmed
        # identical to today's pre-extension behavior (no `env_kwargs` at
        # all) for retail/airline/telecom.
        retrieval_config = None
        if domain == "banking_knowledge":
            retrieval_config = self.retrieval_config or "bm25"  # tau2's own DEFAULT_RETRIEVAL_VARIANT

        span_name = f"tau2 | {tag or task_id}"
        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("llm.model_name", self.model)
            span.set_attribute("tau2.domain", domain)
            span.set_attribute("tau2.task_id", str(task_id))
            span.set_attribute("tau2.user_sim_tier", tier)
            span.set_attribute("tau2.user_sim_model", user_model)
            if retrieval_config:
                span.set_attribute("tau2.retrieval_config", retrieval_config)
            if self.self_play_flags:
                span.set_attribute("tau2.self_play", True)
                span.set_attribute("tau2.self_play_detail", ", ".join(self.self_play_flags))

            req = {
                "domain": domain,
                "task_id": task_id,
                "agent_model": self.model,
                "user_model": user_model,
                "judge_model": self.judge_model,
                "skills": skills,
                "seed": _deterministic_seed(task_id),
                "max_steps": self.max_steps,
                "max_errors": self.max_errors,
                "timeout": self.episode_timeout,
                "retrieval_config": retrieval_config,
                # PROVIDER PIN, COMPUTED HERE AND SHIPPED AS DATA.
                # The worker runs inside tau2-bench's OWN venv, which deliberately has
                # none of EvoSkill's dependencies -- this module's docstring is explicit
                # that the bridge files carry "zero EvoSkill-internal imports". So the
                # worker cannot call `pin_request_body_provider`, and a hand-rolled
                # provider block there is exactly what tests/test_ssbv2_provider_pin.py
                # forbids (six sites did it and every one forgot allow_fallbacks).
                # The parent owns the RULE; the worker only applies the dict it is given.
                "provider_pin": _provider_pin_for(self.model),
                # WHICH MODEL THE PIN BELONGS TO. Without this the worker applies the
                # block to EVERY completion call it wraps -- including the user simulator
                # and the judge -- and OpenRouter answers a deepseek `order` on
                # openai/gpt-5-nano with "No endpoints found", 404ing every episode.
                # Measured 2026-08-06 on the first real nano leg: 60/60 episodes failed.
                # The family gate lives in `_provider_pin_for`; this carries its SCOPE.
                "provider_pin_model": self.model,
            }

            # INACTIVITY WATCHDOG (tau3 parity with SSB v2's liveness module).
            # A single total-duration wall cannot separate a slow-but-working episode from
            # a dead one -- SSB v2 measured both populations in the SAME duration band
            # (43 rollouts completing 720-1643s at 98-100% occupancy, versus 11 hung ones
            # with 690-1452s of dead air). tau3 had only the duration wall, so every
            # wall-kill was scored 0.0 and the two causes were indistinguishable. That is
            # why the measured 7.9% wall-kill rate could not be interpreted, and why two
            # "timed out" task_094 episodes both PASSED when re-run on a quiet host.
            _hb_path = Path(tempfile.gettempdir()) / f"evoskill_tau3_hb_{os.getpid()}_{tag}.txt"
            try:
                _hb_path.write_text(f"{time.time():.3f} start\n")
            except OSError:
                pass
            try:
                proc = await asyncio.create_subprocess_exec(
                    "uv", "run", "--project", str(self.tau2_root), "python", str(_WORKER_PATH),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ,
                         # Per-episode heartbeat file. `cost_tracking.wrapped` touches it
                         # around EVERY generate() call, which is the only per-turn hook
                         # available (tau2 runs its loop inside run_single_task).
                         "EVOSKILL_TAU2_HEARTBEAT_FILE": str(_hb_path)},
                    # New session (== `setsid`) so this `uv` process becomes
                    # its own process-group leader. REQUIRED for the
                    # timeout-kill below to actually work: `uv run` spawns
                    # the real `python _tau2_worker.py` as a GRANDCHILD, and
                    # `proc.kill()` on just the direct `uv` child leaves that
                    # grandchild running (confirmed empirically -- a
                    # process-only `proc.kill()` left `_tau2_worker.py`
                    # alive and reachable via `pgrep` well after the parent
                    # was gone). Killing the whole process group is the only
                    # way to guarantee the actual worker (and its real
                    # OpenRouter API calls) stops.
                    start_new_session=True,
                )
            except Exception as e:
                span.set_attribute("tau2.error", str(e))
                return self._error_trace(f"Worker subprocess failed to launch: {e}", t0)

            async def _watch_inactivity() -> str:
                """Return a kill reason when the worker stops making turns, else run forever.

                Polls the heartbeat mtime rather than streaming stderr: stdout is a single
                framed payload read by communicate(), and reading one pipe while the other
                fills is the classic subprocess deadlock. A file mtime crosses the process
                boundary with no such hazard.
                """
                budget = float(os.environ.get("EVOSKILL_TAU2_INACTIVITY_S", "300"))
                poll = 10.0
                while True:
                    await asyncio.sleep(poll)
                    try:
                        idle = time.time() - _hb_path.stat().st_mtime
                    except OSError:
                        continue          # file not yet written; the total wall still bounds us
                    if idle > budget:
                        return (f"no LLM call started or finished for {idle:.0f}s "
                                f"(budget {budget:.0f}s)")

            _watch = asyncio.ensure_future(_watch_inactivity())
            try:
                _comm = asyncio.ensure_future(proc.communicate(json.dumps(req).encode()))
                done, _ = await asyncio.wait({_comm, _watch},
                                             return_when=asyncio.FIRST_COMPLETED)
                if _watch in done:
                    # INACTIVITY KILL. This is an INFRA outcome, not a capability one:
                    # per SSB v2's liveness contract it must NOT be scored, NOT cached and
                    # SHOULD be retried -- unlike a total-wall stop, which is a real
                    # cacheable 0 because the episode was making progress and ran out.
                    reason = _watch.result()
                    _comm.cancel()
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    span.set_attribute("tau2.exit_reason", "INACTIVITY_KILL")
                    span.set_attribute("tau2.inactivity_detail", reason)
                    return self._error_trace(
                        f"infrastructure_error: inactivity watchdog killed the episode — "
                        f"{reason}. No verdict was produced, so this is NOT a wrong answer: "
                        f"exclude it, do not cache it, retry it.", t0)
                _watch.cancel()
                stdout, stderr = _comm.result()
            except asyncio.CancelledError:
                # `run()`'s outer `asyncio.wait_for(..., timeout=TIMEOUT_SECONDS)`
                # cancels this coroutine on timeout. Cancelling
                # `proc.communicate()` does NOT terminate the child OS
                # process (or its own children) -- confirmed by reading this
                # file's prior version, which had no `proc.kill()`/
                # `terminate()` anywhere at all. Without this, a hung worker
                # (e.g. a stalled OpenRouter call not caught by tau2's own
                # internal `episode_timeout`, or a `uv run` bootstrap hang)
                # would be silently orphaned, continuing to run and spend
                # real API cost with nothing left to reap it -- the same bug
                # class already found and fixed for TB2.1 elsewhere in this
                # codebase. Kill the WHOLE process group (see
                # `start_new_session=True` above), then propagate the
                # cancellation so `run()`'s `wait_for` still raises
                # `TimeoutError` exactly as before this fix.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # already exited between the timeout firing and here
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    pass
                raise

            if proc.returncode != 0 and not stdout:
                stderr_tail = stderr.decode(errors="replace")[-4000:]
                span.set_attribute("tau2.error", stderr_tail)
                return self._error_trace(
                    f"Worker subprocess exited {proc.returncode} with no output. stderr tail:\n{stderr_tail}", t0,
                )

            # PARSE THE FRAMED PAYLOAD, not the whole stream. See _tau2_worker's write site:
            # stdout is shared with tau2/litellm/domain code, and a stray print used to
            # destroy a completed episode (20% of a 15-episode leg on 2026-08-06). Take
            # everything after the LAST marker; without a marker, fall back to the whole
            # stream so an older worker still parses.
            _raw = stdout.decode(errors="replace")
            from ._tau2_protocol import RESULT_MARKER as _RM
            _payload = _raw.rsplit(_RM, 1)[-1].strip() if _RM in _raw else _raw.strip()
            try:
                result = json.loads(_payload)
            except json.JSONDecodeError as e:
                stderr_tail = stderr.decode(errors="replace")[-4000:]
                _polluted = (_RM in _raw)
                return self._error_trace(
                    f"Worker produced non-JSON stdout: {e}. "
                    f"marker_present={_polluted} payload_head={_payload[:200]!r}. "
                    f"stderr tail:\n{stderr_tail}", t0,
                )

            if not result.get("ok"):
                span.set_attribute("tau2.error", result.get("error", "unknown"))
                return self._error_trace(
                    f"tau2 episode failed: {result.get('error')}\n{result.get('traceback', '')}", t0,
                )

            return self._build_success_trace(result, tier, user_model, t0, span, stderr)

    def _build_success_trace(
        self, result: dict, tier: str, user_model: str, t0: float, span, stderr: bytes = b"",
    ) -> AgentTrace[AgentResponse]:
        cost_ledger = result.get("cost_ledger", [])
        cost_by_call: dict[str, float] = {}
        # #207: also roll up TOKENS per call, not just cost. The ledger has
        # carried per-entry prompt_tokens/completion_tokens all along, but only
        # `cost_usd` was ever persisted -- so when a model had no pricing row,
        # `price_usd`'s NaN sentinel was the ONLY thing that survived into the
        # cache record and the spend became permanently unrecoverable rather
        # than merely unreported. That is what happened to all 600 Arm B/C
        # tau3 episodes on 2026-08-08: adding the missing price rows fixes
        # future runs but cannot repair those, because the token counts were
        # never written down. Persisting tokens makes a missing price row a
        # REPAIRABLE bookkeeping gap (multiply by price after the fact) instead
        # of a data-loss event.
        tok_by_call: dict[str, dict[str, int]] = {}
        model_by_call: dict[str, str] = {}
        for entry in cost_ledger:
            cn = entry["call_name"]
            cost_by_call[cn] = cost_by_call.get(cn, 0.0) + entry["cost_usd"]
            t = tok_by_call.setdefault(cn, {"prompt": 0, "completion": 0})
            t["prompt"] += int(entry.get("prompt_tokens") or 0)
            t["completion"] += int(entry.get("completion_tokens") or 0)
            if entry.get("model"):
                model_by_call[cn] = entry["model"]
        # NaN-POISONED, not NaN-filtered: a prior version of this code did
        # `sum(v for v in cost_by_call.values() if v == v)`, which silently
        # DROPPED any unpriced-model component (an explicit NaN sentinel
        # from `cost_tracking.price_usd`, see that module's docstring) from
        # the total instead of surfacing it -- exactly the "trustworthy-
        # looking $0.00" failure mode `cost_tracking.py` exists to prevent,
        # just relocated one level up. A plain `sum()` lets NaN propagate
        # naturally (nan + x == nan in IEEE754), so `total_cost_usd` (the
        # one field every downstream cost rollup reads) comes out visibly
        # NaN -- loud, not silent -- the moment any component is unpriced.
        total_cost = sum(cost_by_call.values())

        cost_warnings = [
            line for line in stderr.decode(errors="replace").splitlines()
            if line.startswith("[cost_tracking]")
        ] if stderr else []
        if math.isnan(total_cost):
            span.set_attribute(
                "tau2.cost_warning",
                "; ".join(cost_warnings) or "NaN cost component with no captured [cost_tracking] stderr line",
            )

        final_answer = json.dumps({
            "task_id": result["task_id"],
            "domain": result["domain"],
            "reward": result["reward"],
            "reward_basis": result["reward_basis"],
            "reward_breakdown": result["reward_breakdown"],
            "db_check": result["db_check"],
            "action_checks": result["action_checks"],
            "communicate_checks": result["communicate_checks"],
            "nl_assertions": result["nl_assertions"],
            "termination_reason": result["termination_reason"],
            # #197: `termination_reason` says MAX_STEPS for BOTH the turn cap and the agent
            # clean-time wall, so it cannot tell a loop from a slow episode. The worker
            # computes the discriminator; this dict is an EXPLICIT key list, so a key added
            # there and not here is silently dropped -- which is exactly what happened on the
            # first attempt at this fix, and is the same two-copies defect as #194.
            "budget_wall_hit": result.get("budget_wall_hit"),
            "budget_wall_detail": result.get("budget_wall_detail"),
            "user_sim_tier": tier,
            "user_sim_model": user_model,
            "cost_usd": {
                "agent": cost_by_call.get("agent_response", 0.0),
                "user": cost_by_call.get("user_simulator_response", 0.0),
                "judge": cost_by_call.get("nl_assertions_eval", 0.0),
                "total": total_cost,
                # Populated only when `total` is NaN -- the captured
                # `[cost_tracking] <call_name>: ...` stderr line(s) that
                # explain which model had no pricing entry, so a human
                # doesn't have to reproduce the run to find out (previously
                # this diagnostic only existed in the discarded subprocess
                # stderr on the FAILURE path -- the success path never read
                # stderr at all).
                "warnings": cost_warnings,
            },
            # #207: the raw material `cost_usd` is derived from. Always written,
            # not only on the NaN path -- a number you can only reconstruct when
            # something already went wrong is a number you will not have when
            # you discover the problem later. `model` is recorded per call so
            # the price lookup can be redone against the RIGHT id (the agent,
            # user-sim and judge are three different models).
            "tokens": {
                cn: {**tok_by_call.get(cn, {"prompt": 0, "completion": 0}),
                     "model": model_by_call.get(cn, "")}
                for cn in ("agent_response", "user_simulator_response", "nl_assertions_eval")
            },
        }, default=str)

        reward = result["reward"]
        messages = result["messages"]
        num_turns = sum(1 for m in messages if m.get("role") == "assistant")

        # Per-turn spans -- one per real LLM call (solver "agent" turns AND
        # user-simulator "user" turns), nested under this episode span so
        # both participants' reasoning/output/tool-call token breakdown are
        # visible in the SAME Phoenix trace, not just the episode-level
        # summary. Built from `turn_log` (worker-side, _tau2_worker.py's
        # _messages_to_turn_log) rather than the `messages` dicts above,
        # which are shaped for transcript rendering and drop usage/cost.
        #
        # Explicit start_time/end_time (nanosecond epoch): without this,
        # start_span()+end() back-to-back in this synchronous loop -- which
        # only runs AFTER the whole episode has already finished -- stamps
        # every turn's span at the same instant, collapsing all of them to
        # ~0ms latency in Phoenix regardless of how long the real LLM call
        # took (confirmed via a live trace: turn.6/turn.7 start timestamps
        # were ~2ms apart even though the real episode took 2m38s). Fix:
        # reconstruct a real, sequential, non-overlapping timeline from each
        # turn's own `generation_time_seconds` (tau2's `generate()` always
        # records this per-message), anchored so the LAST turn ends at "now"
        # (this code runs immediately after the episode truly completes).
        # This won't perfectly reproduce each turn's true historical
        # wall-clock position (tool-execution/network time between turns
        # isn't accounted for, so the reconstructed window undershoots the
        # full episode duration and sits near its end) but every turn's
        # OWN duration is now real, and turns remain in correct relative
        # proportion to each other -- which is what matters for spotting
        # "which turn took long."
        # `turn_log` is now an ORDERED event stream covering every message in
        # the episode except the system prompt (`_tau2_worker._messages_to_turn_log`):
        # `kind == "llm"` entries are real solver/user-simulator generate()
        # calls (one Phoenix span each, as before); `kind == "tool"` entries
        # are tool-call results with no generate() call behind them (no span
        # of their own, but their content must still land in the "history so
        # far" fed to later turns' spans, or that history silently omits
        # every tool round-trip -- previously true whenever the episode used
        # tools, since `turn_log` used to only keep the assistant/user
        # subset).
        #
        # The REAL system prompt (byte-exact, read off tau2's own
        # SystemMessage inside the worker -- see `_extract_system_prompt` --
        # not solver.py's own hand-transcribed `_build_system_prompt()`
        # reconstruction, used only as a defensive fallback in case tau2
        # somehow didn't seed one) is prepended as the first input message on
        # EVERY turn's span, matching this project's other Phoenix traces
        # (`_tracing.py`'s `emit_live_turn`/`finalise_root_span`).
        system_prompt = result.get("system_prompt") or self._build_system_prompt()
        # #198: the USER SIMULATOR's own system prompt, captured byte-exact off tau2's
        # `UserSimulator.system_prompt` property in the worker. There is deliberately NO
        # fallback reconstruction here -- see `_install_user_prompt_capture`'s docstring. Absent
        # (an older worker, or a user class whose property we did not wrap) shows as no system
        # prompt on the user spans, which is the pre-#198 state, not a wrong one.
        user_system_prompt = result.get("user_system_prompt")
        span.set_attribute("tau2.user_system_prompt_captured", bool(user_system_prompt))
        if result.get("user_system_prompt_variants"):
            # Non-zero means the simulator's prompt CHANGED mid-episode. Not fatal (the worker
            # deliberately does not raise on a display-only field) but it makes every user span
            # in this episode show only the first variant, so surface it rather than let the
            # trace quietly misrepresent later turns.
            span.set_attribute(
                "tau2.user_system_prompt_variants", result["user_system_prompt_variants"])

        turn_log = result.get("turn_log", [])
        _llm_entries = [t for t in turn_log if t.get("kind") == "llm"]
        _durations_ns = [
            max(int((t.get("generation_time_seconds") or 0.1) * 1e9), 1_000_000)
            for t in _llm_entries
        ]
        _cursor_ns = time.time_ns() - sum(_durations_ns)

        # Running "history so far" -- every message emitted by either
        # participant OR a tool result, in true causal order. Rebuilt
        # per-turn as `[system] + history` for that turn's input_messages,
        # then extended with the turn's own output AFTER its span closes, so
        # turn K's span shows turns 1..K-1 (not including K itself).
        # TWO HISTORIES, because the two participants genuinely see different things.
        # A single shared stream (what this used to be) makes the USER-simulator span
        # display the agent's retrieval documents and tool calls, which the user
        # simulator never receives -- tau2 filters them out at
        # `user/user_simulator_base.py:42-48` (`is_valid_user_history_message`: a
        # UserMessage, an AssistantMessage ONLY IF `not is_tool_call()`, or a ToolMessage
        # ONLY IF `requestor == "user"`), and `flip_roles` RAISES if an agent tool-call
        # message ever reaches it. The trace's own token counts prove the same thing from
        # the other direction: user turns sit flat at ~1.4k prompt tokens while agent turns
        # climb past 53k as documents accumulate. Showing that context on a user span
        # violates the standing rule that Phoenix must display the AUTHENTIC LLM context,
        # and it invites exactly the conclusion a reader drew from it -- that the user
        # simulator is leaking the agent's tool results.
        history: list[dict[str, Any]] = []          # agent's view: everything, in order
        user_history: list[dict[str, Any]] = []     # user sim's view: tau2's filter
        # TWO COUNTERS, because they answer different questions.
        # `turn_num` is the GLOBAL index over llm entries. It must stay dense and in step with
        #   `_durations_ns`, which was built from the same filtered list -- it is the timeline
        #   cursor, not a label.
        # `role_turns` is PER-PARTICIPANT and is what the span NAME shows (the maintainer, 2026-08-07).
        #   The global index made `tau2.agent.turn.N` unreadable against the agent's own budget:
        #   `EVOSKILL_TAU2_AGENT_TURNS` caps AGENT turns (see _tau2_worker._install_agent_budget,
        #   which counts `cost_tracking.call_count("agent_response")`), so under global numbering
        #   a cap of 100 is reached somewhere near span name "turn.200" and varies per episode
        #   with how often the user simulator spoke -- measured at 38-45% agent share across 82
        #   banking transcripts. A reader checking "did this episode hit the turn cap" against
        #   the span names got an answer that was wrong by roughly a factor of two.
        # The global index is NOT lost -- it is preserved as `tau2.turn_index_global` on every
        # span, so interleaving order stays recoverable after the rename.
        turn_num = 0
        role_turns = {"agent": 0, "user": 0}
        for entry in turn_log:
            if entry.get("kind") != "llm":
                # Tool-result entry: no generate() call, no span -- just
                # fold its content into the history stream for later turns.
                history.append({"role": "tool", "content": entry.get("content") or ""})
                # ...and into the USER's history only when the USER requested it.
                if entry.get("requestor") == "user":
                    user_history.append({"role": "tool", "content": entry.get("content") or ""})
                continue

            turn_num += 1
            dur_ns = _durations_ns[turn_num - 1]
            role = entry.get("role")
            label = "user" if role == "user" else "agent"
            role_turns[label] += 1
            start_ns = _cursor_ns
            end_ns = start_ns + dur_ns
            _cursor_ns = end_ns

            # EACH PARTICIPANT GETS ITS OWN VIEW.
            # Agent turn: the agent's system prompt + the full causal stream.
            # User turn:  the FILTERED stream, and the USER SIMULATOR'S OWN system prompt
            #   (#198, the maintainer 2026-08-07). This block previously showed NO system prompt on user
            #   turns, reasoning that `system_prompt` is the AGENT's and attaching it would
            #   assert the user simulator had been given the agent's policy -- false, and the
            #   #146 failure mode exactly. That reasoning was right about the agent's prompt and
            #   wrong in its conclusion that nothing faithful existed to show: the user
            #   simulator's own prompt does exist, it simply never enters the transcript, so it
            #   had to be captured at its source instead of read off the message stream. It now
            #   is -- byte-exact, see `_tau2_worker._install_user_prompt_capture`. Absent (older
            #   worker / unwrapped user class) falls back to the previous no-system-prompt
            #   display rather than to a reconstruction: absent still beats wrong.
            if label == "user":
                input_msgs = (
                    [{"role": "system", "content": user_system_prompt}]
                    if user_system_prompt else []
                )
                input_msgs.extend(user_history)
            else:
                input_msgs = [{"role": "system", "content": system_prompt}]
                input_msgs.extend(history)

            output_msgs: list[dict[str, Any]] = []
            content = entry.get("content") or ""
            if content:
                # #243 (the maintainer, 2026-08-09): ALWAYS "assistant", never the transcript role.
                # `llm.output_messages` is BY DEFINITION what the model in THIS span generated,
                # so its role is `assistant` whichever character the model plays. Using the
                # transcript role put `role: "user"` on the user simulator's own output and made
                # the span contradict ITSELF: the input side (below, ~line 1129) already flips
                # correctly, mirroring tau2's `flip_roles`, so the simulator's PRIOR outputs were
                # shown as `assistant` while its CURRENT output was shown as `user` -- one
                # speaker, one span, two roles. The user simulator is an LLM; "user" names the
                # part it plays in the dialogue, not the role of its completion.
                # Agent turns were already correct (their transcript role is "assistant"); this
                # makes the rule uniform instead of accidentally right on one branch.
                output_msgs.append({"role": "assistant", "content": content})
            tool_calls = entry.get("tool_calls") or []
            for tc in tool_calls:
                tc_args = json.dumps(tc.get("arguments"), default=str, ensure_ascii=False)
                output_msgs.append({
                    "role": "tool_call",
                    "content": f"Tool: {tc.get('name')}\n\n{tc_args}",
                })

            turn_span = _tracer.start_span(
                f"tau2.{label}.turn.{role_turns[label]}",
                context=otel_trace.set_span_in_context(span),
                start_time=start_ns,
            )
            try:
                turn_span.set_attribute("openinference.span.kind", "LLM")
                turn_span.set_attribute("llm.model_name", entry.get("model") or "")
                # The span NAME carries the per-participant index; both indices are recorded so
                # the rename costs no information. `turn_index_global` is the position in the
                # interleaved conversation (what the name used to show) and is what to sort by
                # when comparing an agent turn against the user turn that provoked it.
                turn_span.set_attribute("tau2.turn_index_role", role_turns[label])
                turn_span.set_attribute("tau2.turn_index_global", turn_num)
                turn_span.set_attribute("output.value", content)
                _set_oi_messages(turn_span, "llm.input_messages", input_msgs, max_per_msg=100_000)
                _set_oi_messages(turn_span, "llm.output_messages", output_msgs)
                if entry.get("prompt_tokens") is not None:
                    turn_span.set_attribute("llm.token_count.prompt", entry["prompt_tokens"])
                if entry.get("completion_tokens") is not None:
                    turn_span.set_attribute("llm.token_count.completion", entry["completion_tokens"])
                if entry.get("reasoning_tokens") is not None:
                    turn_span.set_attribute("llm.token_count.completion_details.reasoning", entry["reasoning_tokens"])
                if entry.get("cost_usd") is not None:
                    turn_span.set_attribute("tau2.turn_cost_usd", entry["cost_usd"])
                if tool_calls:
                    turn_span.set_attribute("llm.tool_calls_count", len(tool_calls))
                    turn_span.set_attribute("tau2.tool_calls", json.dumps(tool_calls, default=str))
                if entry.get("generation_time_seconds") is not None:
                    turn_span.set_attribute("tau2.generation_time_seconds", entry["generation_time_seconds"])
            finally:
                turn_span.end(end_time=end_ns)

            # Extend history AFTER the span closes, so this turn's own span
            # (built above) only ever sees turns 1..turn_num-1.
            if content:
                history.append({"role": role, "content": content})
                # The USER's view, mirroring tau2's `flip_roles`
                # (user/user_simulator_base.py:62-90): the user simulator sees its own
                # prior messages as `assistant` and the agent's as `user`, and it sees an
                # agent message ONLY when that message carries no tool calls -- flip_roles
                # raises outright on an agent tool-call message. So an agent turn whose
                # content accompanies tool calls is deliberately withheld here too.
                if role == "user":
                    user_history.append({"role": "assistant", "content": content})
                elif not tool_calls:
                    user_history.append({"role": "user", "content": content})
            for tc in tool_calls:
                tc_args = json.dumps(tc.get("arguments"), default=str, ensure_ascii=False)
                history.append({
                    "role": "tool_call",
                    "content": f"Tool: {tc.get('name')}\n\n{tc_args}",
                })
                # NOT added to user_history: the agent's tool CALLS are invisible to the
                # user simulator, same rule as its tool RESULTS.

        # NL-assertion judge span(s) -- nested under the same episode span as
        # the turn spans, positioned strictly AFTER the last conversation
        # turn ends (judging only starts once `run_simulation` returns the
        # completed trajectory, see the judge-timing audit this fix is
        # based on). One parent "tau2.judge" span carries the actual
        # generate() call's real input (system+user judge prompt, which
        # already embeds the full transcript + the NL assertions being
        # graded -- see `evaluate_nl_assertions` upstream) and output (raw
        # verdict JSON), model, tokens, cost, and a real non-zero duration;
        # one lightweight child span per individual NL assertion verdict
        # (there can be several assertions judged by that SAME single LLM
        # call, so children share the parent's timing/cost rather than
        # inventing a fake per-assertion split).
        nl_assertions = result.get("nl_assertions") or []
        judge_ledger_entries = [
            e for e in (result.get("cost_ledger") or []) if e.get("call_name") == "nl_assertions_eval"
        ]
        if nl_assertions and judge_ledger_entries:
            judge_entry = judge_ledger_entries[0]
            judge_dur_ns = max(
                int((judge_entry.get("generation_time_seconds") or 0.1) * 1e9), 1_000_000,
            )
            judge_start_ns = max(_cursor_ns, time.time_ns())
            judge_end_ns = judge_start_ns + judge_dur_ns

            judge_span = _tracer.start_span(
                "tau2.judge",
                context=otel_trace.set_span_in_context(span),
                start_time=judge_start_ns,
            )
            try:
                judge_span.set_attribute("openinference.span.kind", "LLM")
                judge_span.set_attribute("llm.model_name", judge_entry.get("model") or self.judge_model)
                response_content = judge_entry.get("response_content") or ""
                judge_span.set_attribute("output.value", response_content)
                prompt_messages = judge_entry.get("prompt_messages") or []
                if prompt_messages:
                    _set_oi_messages(judge_span, "llm.input_messages", prompt_messages, max_per_msg=200_000)
                if response_content:
                    _set_oi_messages(
                        judge_span, "llm.output_messages",
                        [{"role": "assistant", "content": response_content}],
                    )
                if judge_entry.get("prompt_tokens") is not None:
                    judge_span.set_attribute("llm.token_count.prompt", judge_entry["prompt_tokens"])
                if judge_entry.get("completion_tokens") is not None:
                    judge_span.set_attribute("llm.token_count.completion", judge_entry["completion_tokens"])
                if judge_entry.get("cost_usd") is not None:
                    judge_span.set_attribute("tau2.judge_cost_usd", judge_entry["cost_usd"])
                if judge_entry.get("generation_time_seconds") is not None:
                    judge_span.set_attribute("tau2.generation_time_seconds", judge_entry["generation_time_seconds"])
                judge_span.set_attribute("tau2.nl_assertions", json.dumps(nl_assertions, default=str))
            finally:
                judge_span.end(end_time=judge_end_ns)

            for k, check in enumerate(nl_assertions):
                assertion_span = _tracer.start_span(
                    f"tau2.judge.assertion.{k}",
                    context=otel_trace.set_span_in_context(judge_span),
                    start_time=judge_start_ns,
                )
                try:
                    assertion_span.set_attribute("openinference.span.kind", "CHAIN")
                    assertion_text = check.get("nl_assertion") or ""
                    met = bool(check.get("met"))
                    justification = check.get("justification") or ""
                    assertion_span.set_attribute("input.value", assertion_text)
                    assertion_span.set_attribute("output.value", justification)
                    assertion_span.set_attribute("tau2.nl_assertion", assertion_text)
                    assertion_span.set_attribute("tau2.nl_assertion_met", met)
                    assertion_span.set_attribute("tau2.nl_assertion_justification", justification)
                finally:
                    assertion_span.end(end_time=judge_end_ns)

            _cursor_ns = judge_end_ns

        span.set_attribute("tau2.reward", reward)
        span.set_attribute("tau2.termination_reason", result["termination_reason"] or "")
        span.set_attribute("tau2.cost_usd", total_cost)
        # PER-ROLE COST ON THE SPAN, not just the total (user-flagged 2026-08-06 from the
        # Phoenix trace of tau3_smoke_e3_v3_train-s3, which showed only `tau2.cost_usd`).
        # The split already existed in the result payload's `cost_usd` dict — three of the
        # four numbers simply never reached the span, so answering "what does the user
        # simulator actually cost us?" meant re-reading cache records instead of looking at
        # the trace. That is the same shape as the verdict-icon gap: the datum existed and
        # the observability surface did not carry it.
        #
        # Costs come from `cost_by_call`, keyed by tau2's own call names:
        #   agent_response -> the customer-service AGENT (the thing under evaluation)
        #   user_simulator_response -> the simulated CUSTOMER (gpt-4.1 reference tier /
        #                              gpt-5-nano cheap tier, per user_tier_gate)
        #   nl_assertions_eval -> the JUDGE, and it is 0.0 unless the task has
        #                         nl_assertions in its reward_basis
        # Reported even when 0.0 so an absent role is visibly zero rather than absent —
        # a missing attribute reads as "not measured", which is a different claim.
        _cost_agent = cost_by_call.get("agent_response", 0.0)
        _cost_user = cost_by_call.get("user_simulator_response", 0.0)
        _cost_judge = cost_by_call.get("nl_assertions_eval", 0.0)
        span.set_attribute("tau2.cost_agent_usd", _cost_agent)
        span.set_attribute("tau2.cost_user_sim_usd", _cost_user)
        span.set_attribute("tau2.cost_judge_usd", _cost_judge)
        span.set_attribute("tau2.cost_total_usd", total_cost)
        # The share is what the model-choice decision actually turns on (0804b K3/K4
        # retracted a "59% of cost" claim that was 18.7%), so compute it here rather than
        # leaving every reader to divide. Guarded because total_cost can be NaN when a
        # model has no pricing entry.
        if isinstance(total_cost, float) and total_cost == total_cost and total_cost > 0:
            span.set_attribute("tau2.cost_user_sim_pct", 100.0 * _cost_user / total_cost)
            span.set_attribute("tau2.cost_agent_pct", 100.0 * _cost_agent / total_cost)
        # Green/red pass-fail icon in Phoenix's span list, same convention as
        # SWE-rebench's episode spans (run_swe_rebench_evo.py's
        # `_episode.set_status(...)`) -- tau2 never set this before (every
        # row showed the same neutral UNSET icon regardless of reward),
        # user-flagged 2026-07-11.
        from opentelemetry.trace import StatusCode as _TauStatusCode
        span.set_status(_TauStatusCode.OK if reward >= 1.0 else _TauStatusCode.ERROR)

        output = AgentResponse(
            reasoning=(
                f"tau2 episode ended ({result['termination_reason']}), reward={reward}. "
                f"reward_basis={result['reward_basis']}, breakdown={result['reward_breakdown']}."
            ),
            final_answer=final_answer,
            confidence=100.0 if reward >= 1.0 else 0.0,
        )

        trace = AgentTrace(
            duration_ms=int((time.monotonic() - t0) * 1000),
            total_cost_usd=total_cost,
            num_turns=num_turns,
            # CLEAN TIME rides in `usage` because AgentTrace has named fields only:
            # the worker's `result["clean_time"]` was being DROPPED here, silently, and
            # the loss would only have surfaced ~10 minutes later as an absent field on
            # the first completed episode. `usage` is free-form and already persisted
            # (it is where `reward`/`user_sim_tier` survive the round-trip), so it is
            # the one channel that reaches the cache record without a schema change.
            # NOTE this is a REPORTING path only -- the budget itself is enforced in the
            # worker against tau2's own termination check, so scores are unaffected by
            # whether this field lands.
            usage={"reward": reward, "user_sim_tier": tier,
                   **({"clean_time": result["clean_time"]} if result.get("clean_time") else {})},
            result=final_answer,
            is_error=False,
            output=output,
            messages=messages,
        )
        trace.inline_score = float(reward)
        return trace

    def _error_trace(self, msg: str, t0: float) -> AgentTrace[AgentResponse]:
        return AgentTrace(
            duration_ms=int((time.monotonic() - t0) * 1000),
            total_cost_usd=0.0,
            num_turns=0,
            usage={},
            result=msg,
            is_error=True,
            output=None,
            parse_error=msg,
            messages=[],
        )


def _deterministic_seed(task_id: str) -> int:
    import hashlib
    return int(hashlib.sha256(str(task_id).encode()).hexdigest()[:8], 16)
