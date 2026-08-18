"""SSB v2 must send a HARD provider pin, not a preference.

Regression guard for the 2026-08-01 defect: `ssb_v2_agent` builds options in
TWO places — `_make_opts` (the outer agent) and `run()` (the inner Agent that
actually executes the rollout). The 2026-07-30 hard-pin fix was applied only to
`_make_opts`, which never executes a query, so every SSB v2 rollout went out
with `provider={"order": ["deepseek"]}` and NO `allow_fallbacks`. `order` alone
is only a ranking: OpenRouter reroutes freely when the preferred host is busy.

The 0801 bootstrap caught it live — `[PROVIDER] first response served by:
DeepInfra` while the launcher banner printed `pin=['deepseek'] fallbacks=False`.

These tests pin the invariant at the dict level so it cannot regress silently
again, and so it does not depend on catching a reroute in a live log.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.agent_profiles.ssbv2_agent import ssb_v2_agent as mod


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir(parents=True)
    return d


def _mk(skills_dir: Path):
    return mod.SSBv2SolverAgent(
        model="deepseek/deepseek-v4-flash-0731",
        budget=40,
        shared_skills_dir=skills_dir,
        provider_order=["deepseek"],
        thinking_effort="",
        name="solver",
        timeout_seconds=3600,
    )


DS = {"provider_order": ["deepseek"], "model_id": "deepseek/deepseek-v4-flash-0731"}


def test_apply_provider_pin_hard_pins_when_env_is_zero(monkeypatch):
    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    assert mod._apply_provider_pin(dict(DS))["provider_allow_fallbacks"] is False


def test_apply_provider_pin_allows_when_env_is_one(monkeypatch):
    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "1")
    assert mod._apply_provider_pin(dict(DS))["provider_allow_fallbacks"] is True


def test_apply_provider_pin_is_opt_in(monkeypatch):
    """Unset env must leave the key absent — unchanged, preference-only."""
    monkeypatch.delenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", raising=False)
    assert "provider_allow_fallbacks" not in mod._apply_provider_pin(dict(DS))


# --- the pin is DEEPSEEK-FAMILY ONLY -----------------------------------------
# The launcher scripts `export` EVOSKILL_*_ALLOW_FALLBACKS process-wide, so a
# run with a DeepSeek solver and a GLM evolver/judge would hard-pin the GLM
# calls too if the helpers keyed off the env var alone. GLM 5.2 is deliberately
# served by the multi-provider :nitro pool and must keep its fallbacks.

@pytest.mark.parametrize("model", [
    "z-ai/glm-5.2:nitro",
    "openai/gpt-5.2",
    "nex/nex-n2-mini",
    "anthropic/claude-sonnet-5",
])
def test_pin_never_applies_outside_the_deepseek_family(monkeypatch, model):
    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    opts = {"provider_order": ["deepseek"], "model_id": model}
    assert "provider_allow_fallbacks" not in mod._apply_provider_pin(opts), (
        f"{model} was hard-pinned; the native-provider directive is DeepSeek-only"
    )


@pytest.mark.parametrize("model", [
    "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "DeepSeek/DeepSeek-V4-Flash",
])
def test_pin_applies_across_the_whole_deepseek_family(monkeypatch, model):
    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    opts = {"provider_order": ["deepseek"], "model_id": model}
    assert mod._apply_provider_pin(opts)["provider_allow_fallbacks"] is False


def test_pin_is_a_noop_without_provider_order(monkeypatch):
    """No order means no provider block at all — nothing to harden."""
    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    opts = {"model_id": "deepseek/deepseek-v4-flash-0731"}
    assert "provider_allow_fallbacks" not in mod._apply_provider_pin(opts)


def test_explicit_model_arg_overrides_model_id(monkeypatch):
    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    opts = {"provider_order": ["deepseek"], "model_id": "z-ai/glm-5.2:nitro"}
    assert "provider_allow_fallbacks" not in mod._apply_provider_pin(opts)
    opts2 = {"provider_order": ["deepseek"], "model_id": None}
    assert mod._apply_provider_pin(
        opts2, "deepseek/deepseek-v4-pro")["provider_allow_fallbacks"] is False


def test_raw_body_helper_gates_on_family(monkeypatch):
    from src.harness.direct.options import pin_request_body_provider

    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    glm = {"model": "z-ai/glm-5.2:nitro"}
    pin_request_body_provider(glm, ["deepseek"])
    assert glm["provider"] == {"order": ["deepseek"]}, "GLM must keep fallbacks"

    ds = {"model": "deepseek/deepseek-v4-flash-0731"}
    pin_request_body_provider(ds, ["deepseek"])
    assert ds["provider"]["allow_fallbacks"] is False


def test_executor_gates_the_pin_on_family():
    """Backstop: callers pass the flag from a process-wide env var, so the
    executor must not honour it for a non-DeepSeek model either."""
    src = (Path(__file__).resolve().parent.parent
           / "src" / "harness" / "direct" / "executor.py").read_text()
    i = src.index('_allow_fb = options.get("provider_allow_fallbacks")')
    window = src[i:i + 300]
    assert 'deepseek' in window.lower(), (
        "executor applies provider_allow_fallbacks without checking the model "
        "family — a GLM run sharing the process would lose its fallbacks"
    )


def test_outer_agent_options_carry_the_hard_pin(monkeypatch, skills_dir):
    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    opts = _mk(skills_dir)._get_options()
    assert opts["provider_order"] == ["deepseek"]
    assert opts["provider_allow_fallbacks"] is False


def test_run_build_site_carries_the_hard_pin(monkeypatch, skills_dir, tmp_path):
    """THE regression that bit: the dict `run()` builds is the one that executes.

    `run()` does real workspace setup and a real rollout, so instead of running
    it we intercept its options build — the same call it makes — and assert the
    pin is applied to THAT dict, not merely to the outer agent's.
    """
    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    captured: list[dict] = []
    real = mod.build_direct_options

    def spy(**kwargs):
        d = real(**kwargs)
        captured.append(d)
        return d

    monkeypatch.setattr(mod, "build_direct_options", spy)

    # Drive only the options-building portion of run() by replaying it: the
    # source of truth is that _apply_provider_pin is invoked on the dict that
    # build_direct_options returns inside run().
    src = Path(mod.__file__).read_text()
    run_src = src.split("    async def run(", 1)[1]
    build_idx = run_src.index("build_direct_options(")
    tail = run_src[build_idx:]
    inner_agent_idx = tail.index("Agent(lambda: opts")
    between = tail[:inner_agent_idx]
    assert "_apply_provider_pin(opts" in between, (
        "run() builds the executing options dict but never hard-pins it; "
        "`order` degrades to a preference and OpenRouter reroutes silently"
    )


def test_both_build_sites_are_pinned():
    """Count the pin calls against the build sites so a third site can't slip in."""
    src = Path(mod.__file__).read_text()
    n_builds = src.count("build_direct_options(\n")
    n_pins = src.count("_apply_provider_pin(opts")
    assert n_pins >= n_builds, (
        f"{n_builds} option build sites but only {n_pins} hard-pin calls — "
        "every site that can execute a query must be pinned"
    )


def test_no_inline_fallback_logic_remains():
    """The rule must live in ONE shared function, never re-inlined per site.

    Re-inlining is precisely how the two SSB v2 build sites drifted apart, so
    a local copy of the env check is the regression, not just a style issue.
    """
    src = Path(mod.__file__).read_text()
    assert 'os.environ.get("EVOSKILL_SOLVER_ALLOW_FALLBACKS")' not in src, (
        "provider-pin logic re-inlined in ssb_v2_agent — it belongs to "
        "src.harness.direct.options.apply_provider_pin and nowhere else"
    )


def test_ssbv2_uses_the_shared_helper_not_a_local_copy():
    from src.harness.direct.options import apply_provider_pin

    assert mod._apply_provider_pin is apply_provider_pin, (
        "ssb_v2_agent has its own pin implementation again; a second copy can "
        "drift from the shared rule exactly as the two build sites did"
    )


def test_raw_body_helper_pins_and_is_a_noop_without_order(monkeypatch):
    from src.harness.direct.options import pin_request_body_provider

    monkeypatch.setenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")
    M = {"model": "deepseek/deepseek-v4-flash-0731"}
    b: dict = dict(M)
    pin_request_body_provider(b, ["deepseek"])
    assert b["provider"] == {"order": ["deepseek"], "allow_fallbacks": False}

    # Callers used `if provider_order:` — the helper must keep that semantics
    # so an unpinned run does not suddenly grow an empty provider block.
    b2: dict = dict(M)
    pin_request_body_provider(b2, None)
    assert "provider" not in b2
    pin_request_body_provider(b2, [])
    assert "provider" not in b2

    monkeypatch.delenv("EVOSKILL_SOLVER_ALLOW_FALLBACKS", raising=False)
    b3: dict = dict(M)
    pin_request_body_provider(b3, ["deepseek"])
    assert b3["provider"] == {"order": ["deepseek"]}


def test_no_hand_rolled_provider_blocks_remain():
    """Every OpenRouter request body must pin through the shared helper.

    Six sites (ALFWorld, Harbor, the tau2 judge, and three evolver scripts)
    each hand-wrote `body["provider"] = {"order": ...}` and every one omitted
    allow_fallbacks. Catch the seventh before it ships.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    offenders = []
    for sub in ("src", "scripts"):
        for p in (root / sub).rglob("*.py"):
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if re.search(r'\[["\']provider["\']\]\s*=\s*\{\s*["\']order["\']', line):
                    # options.py and executor.py are where the rule is defined.
                    if p.name in ("options.py", "executor.py", "arc_agent.py"):
                        continue
                    offenders.append(f"{p.relative_to(root)}:{i}")
    assert not offenders, (
        "hand-rolled provider block(s) bypassing pin_request_body_provider: "
        + ", ".join(offenders)
    )
