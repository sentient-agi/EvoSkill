#!/usr/bin/env python3
"""Claude Code pins for every `claude` child the orchestrator starts (AO intake/brief, clarify,
prompt author, evolver) — and the check that the session ran on the model it was pinned to.

ONE place for the rule, so a shell launcher and the Python callers cannot drift: a launcher evals
`python3 -m src.orchestrator.claude_pins --shell --model M` instead of carrying a copy.

WHY EACH PIN EXISTS (all verified in the strings of the Claude Code 2.1.281 binary, not inferred
from names):

  * THE MODEL ID, never an alias. The binary carries two `opus` alias tables (opus ->
    claude-opus-5-5 under a signed catalog with a validity window, and opus -> claude-opus-5), so
    `opus[1m]` can change meaning inside one CC version. Callers pass `claude-opus-5-5`.
  * --effort <level> (flag registered as `new Go("--effort <level>", ...)`) and
    CLAUDE_CODE_EFFORT_LEVEL (the binary: "CLAUDE_CODE_EFFORT_LEVEL overrides effort for this
    session"). Both are set to the SAME value, so a user settings.json `effortLevel` (e.g. xhigh)
    cannot silently override the flag. EVOSKILL_CLAUDE_EFFORT, default "high" (Opus 5's
    default; Opus 5.5's own default is medium). "" / none / off = no flag (explicit opt-out).
    Applied only to first-party Claude models (quota_scope): a gateway model gets no Anthropic
    effort parameter.
  * CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1 — the binary: `function bD(){return
    !a.CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK&&!pZ()}` gates the refusal->fallback-model switch.
  * CLAUDE_CODE_NO_MODEL_FALLBACK=1 — the binary: "CLAUDE_CODE_NO_MODEL_FALLBACK is set: model
    substitution is disabled" / "forbids model substitution" (pZ() above, so it also disables the
    refusal fallback).
  * DISABLE_AUTOUPDATER=1 — the binary: `if(Oe(process.env.DISABLE_AUTOUPDATER))return{type:"env",
    envVar:"DISABLE_AUTOUPDATER"}`. Stops the CHILD from updating the install mid-run. It cannot
    stop another CC process (the operator's own session) from moving ~/.local/bin/claude, which is
    why claude_code_version is recorded per session and EVOSKILL_EXPECT_CLAUDE_CODE_VERSION can
    make a change a HALT.
  * CLAUDE_CODE_SUBAGENT_MODEL=<model> + CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1, for an explicit
    claude-* id only. Without FORCE an agent definition's own `model:` (e.g. a haiku Explore agent)
    wins (`se(n,e){if(a.CLAUDE_CODE_SUBAGENT_MODEL_FORCE)n=void 0; ...}`), and the model check
    below would then HALT a correct session over a subagent nobody chose.

THE CHECK. `check_session(log_or_text, requested_model)` parses the stream-json events the CLI
wrote and asserts every `system/init` model and every assistant `message.model` (top level and
subagents) is the requested explicit id (optionally with a -YYYYMMDD snapshot suffix or [1m]).
It is exact-family: `claude-opus-5-5` does NOT satisfy a request for `claude-opus-5`, and vice
versa. `result.modelUsage` entries of the opus/sonnet/fable families must match too; haiku entries
there are the CLI's own background calls and are recorded, not asserted. An alias request
(`opus[1m]`) is recorded but not asserted ("checked": False), so completed runs are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections import Counter
from pathlib import Path

OPUS_55 = "claude-opus-5-5"
EFFORT_ENV = "EVOSKILL_CLAUDE_EFFORT"
DEFAULT_EFFORT = "high"
VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_OPT_OUT = ("", "none", "off", "default")
AO_MODEL_ENV = "EVOSKILL_AO_MODEL"
LEGACY_AO_DEFAULT = "opus[1m]"
EXPECT_CC_VERSION_ENV = "EVOSKILL_EXPECT_CLAUDE_CODE_VERSION"

# Set for EVERY claude child, whatever the model. Names verified in the binary (module doc).
PIN_ENV_ALWAYS = {
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
}


class ModelPinError(ValueError):
    """A pin setting is malformed (never a reason to launch unpinned)."""


def ao_model(explicit: str | None = None) -> str:
    """The AO model for intake/brief/clarify: explicit arg > $EVOSKILL_AO_MODEL > legacy alias."""
    return (explicit or "").strip() or os.environ.get(AO_MODEL_ENV, "").strip() or LEGACY_AO_DEFAULT


def on_subscription(model: str | None) -> bool:
    from .quota_scope import consumes_claude_quota
    return consumes_claude_quota(model)


def effort(env: dict | None = None) -> str | None:
    """The effort level, or None for an explicit opt-out. Unset means DEFAULT_EFFORT."""
    e = os.environ if env is None else env
    raw = e.get(EFFORT_ENV)
    if raw is None:
        return DEFAULT_EFFORT
    v = raw.strip().lower()
    if v in _OPT_OUT:
        return None
    if v not in VALID_EFFORTS:
        raise ModelPinError(f"{EFFORT_ENV}={raw!r} is not one of {VALID_EFFORTS} (or empty/none "
                            f"for no --effort flag)")
    return v


def effort_argv(model: str | None, env: dict | None = None) -> list:
    if not on_subscription(model):
        return []
    lv = effort(env)
    return ["--effort", lv] if lv else []


def expected_model(model: str | None) -> str | None:
    """The id a session must report, or None when the request is an alias (not asserted)."""
    m = (model or "").strip()
    if m.endswith("[1m]"):
        m = m[:-4]
    return m if re.fullmatch(r"claude-[a-z0-9.-]+", m or "") else None


def model_matches(resolved: str | None, expected: str) -> bool:
    return bool(re.fullmatch(re.escape(expected) + r"(-\d{8})?(\[1m\])?", (resolved or "").strip()))


def pinned_env(model: str | None, env: dict | None = None) -> dict:
    """`env` (default os.environ) plus every pin for a claude child running `model`."""
    out = dict(os.environ if env is None else env)
    out.update(PIN_ENV_ALWAYS)
    if on_subscription(model):
        lv = effort(out)
        if lv:
            out["CLAUDE_CODE_EFFORT_LEVEL"] = lv
        else:
            out.pop("CLAUDE_CODE_EFFORT_LEVEL", None)
    exp = expected_model(model)
    if exp:
        out["CLAUDE_CODE_SUBAGENT_MODEL"] = exp
        out["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] = "1"
    return out


def pin_record(model: str | None, env: dict | None = None) -> dict:
    """What was pinned, for the AO/evolver records (values are not secrets)."""
    e = pinned_env(model, env)
    keys = [*PIN_ENV_ALWAYS, "CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CODE_SUBAGENT_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL_FORCE"]
    return {"requested_model": model, "expected_model": expected_model(model),
            "effort_argv": effort_argv(model, e), "env": {k: e[k] for k in keys if k in e}}


def _events(source):
    if isinstance(source, (str, Path)) and not str(source).lstrip().startswith("{") \
            and "\n" not in str(source) and Path(str(source)).is_file():
        text = Path(str(source)).read_text(errors="replace")
    else:
        text = str(source or "")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue                              # a partial final line on a live log
        if isinstance(e, dict):
            yield e


def session_models(source) -> dict:
    """Every model the CLI reported in a stream-json log (path or text)."""
    init, versions = [], []
    top, sub = Counter(), Counter()
    usage, n = {}, 0
    for e in _events(source):
        n += 1
        t = e.get("type")
        if t == "system" and e.get("subtype") == "init":
            if e.get("model"):
                init.append(str(e["model"]))
            if e.get("claude_code_version"):
                versions.append(str(e["claude_code_version"]))
        elif t == "assistant":
            m = (e.get("message") or {}).get("model")
            if m:
                (sub if e.get("parent_tool_use_id") else top)[str(m)] += 1
        elif t == "result" and isinstance(e.get("modelUsage"), dict):
            for k, v in e["modelUsage"].items():
                usage[str(k)] = {kk: v.get(kk) for kk in ("inputTokens", "outputTokens",
                                                          "costUSD")} if isinstance(v, dict) else v
    resolved = init[0] if init else (top.most_common(1)[0][0] if top else None)
    return {"init_models": init, "claude_code_versions": sorted(set(versions)),
            "claude_code_version": versions[0] if versions else None,
            "assistant_models": dict(top), "subagent_models": dict(sub),
            "model_usage": usage, "resolved_model": resolved, "n_events": n}


def check_session(source, requested_model: str | None, *,
                  expect_cc_version: str | None = None) -> dict:
    """Assert the session ran on the requested explicit model. See the module doc."""
    s = session_models(source)
    exp = expected_model(requested_model)
    want_v = (expect_cc_version if expect_cc_version is not None
              else os.environ.get(EXPECT_CC_VERSION_ENV, "")).strip()
    viol, notes = [], []
    if exp is not None:
        seen = [*s["init_models"], *s["assistant_models"], *s["subagent_models"]]
        if not seen:
            viol.append("the session reported NO model (no system/init and no assistant event) — "
                        "cannot show it ran on " + exp)
        for m in s["init_models"]:
            if not model_matches(m, exp):
                viol.append(f"system/init model {m!r} != {exp}")
        for m, c in s["assistant_models"].items():
            if not model_matches(m, exp):
                viol.append(f"{c} assistant message(s) from {m!r} != {exp}")
        for m, c in s["subagent_models"].items():
            if not model_matches(m, exp):
                viol.append(f"{c} SUBAGENT message(s) from {m!r} != {exp}")
        for m in s["model_usage"]:
            if "haiku" in m.lower():
                notes.append(f"modelUsage {m}: CLI background call (not asserted)")
            elif not model_matches(m, exp):
                viol.append(f"result.modelUsage reports {m!r} != {exp}")
    if want_v and s["claude_code_versions"] and s["claude_code_versions"] != [want_v]:
        viol.append(f"claude_code_version {s['claude_code_versions']} != expected {want_v} "
                    f"(${EXPECT_CC_VERSION_ENV})")
    return {**s, "requested_model": requested_model, "expected_model": exp,
            "checked": exp is not None, "expect_claude_code_version": want_v or None,
            "violations": viol, "notes": notes, "ok": not viol}


def result_text(stdout: str) -> str:
    """The final `result` text of a `-p --output-format stream-json` run (or legacy `json`)."""
    last = None
    for e in _events(stdout):
        if e.get("type") == "result":
            last = e
    if last is not None and isinstance(last.get("result"), str):
        return last["result"]
    try:                                            # legacy --output-format json: one object
        env = json.loads(stdout)
        if isinstance(env, dict) and isinstance(env.get("result"), str):
            return env["result"]
    except (json.JSONDecodeError, TypeError):
        pass
    return stdout


def shell_lines(model: str, env: dict | None = None) -> str:
    """bash assignments for a shell launcher: exports + EFFORT_ARGV=(...). Values are validated
    tokens (effort level, model id), still shell-quoted."""
    e = pinned_env(model, env)
    base = dict(os.environ if env is None else env)
    lines = []
    for k in [*PIN_ENV_ALWAYS, "CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CODE_SUBAGENT_MODEL",
              "CLAUDE_CODE_SUBAGENT_MODEL_FORCE"]:
        if k in e:
            lines.append(f"export {k}={shlex.quote(e[k])}")
        elif k in base:
            lines.append(f"unset {k}")
    lines.append("EFFORT_ARGV=(" + " ".join(shlex.quote(x) for x in effort_argv(model, e)) + ")")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--shell", action="store_true", help="print bash pin assignments")
    ap.add_argument("--check-log", default="", help="check a stream-json log against --model")
    a = ap.parse_args(argv)
    try:
        if a.shell:
            sys.stdout.write(shell_lines(a.model))
            return 0
        if a.check_log:
            r = check_session(Path(a.check_log), a.model)
            print(json.dumps(r, indent=1))
            return 0 if r["ok"] else 1
    except ModelPinError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 4
    print(json.dumps(pin_record(a.model), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
