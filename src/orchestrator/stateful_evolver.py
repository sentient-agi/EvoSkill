"""Stateful evolver: the loop-facing side -- the session validation, the system prompt / epoch message rendering,
and the verifies the stateful step list binds by name.

This module imports the loop LAZILY (inside each function): the loop imports it the same way, so neither pulls the
other in at import time.

The renderer is two fixed templates (src/orchestrator/prompts/) with four placeholders, and the evolver's material is
<run>/evo_e{K}/ (evo_layout); the loop computes no score table and runs no leakage or path-naming check on the
prompt. What is here: validate_session, render_system_prompt, render_epoch_message,
run_render_session_prompt, v_session_prompt_rendered, run_build_evo_layout, v_evo_layout, v_session_contained,
v_session_read_nothing_forbidden, v_session_continued, v_session_snapshotted,
run_session_finish.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from src.orchestrator import claude_pins
from src.orchestrator import daytona_evolver as de
from src.orchestrator import evolver_template as ET
from src.orchestrator import stateful_layout as SL

EFFORT_REMEDY = ("export EVOSKILL_CLAUDE_EFFORT=default -- an unset variable means 'high' in this repo "
                 "(claude_pins.DEFAULT_EFFORT)")


def _runtime(ctx) -> dict | None:
    """The run's RUNTIME.json as a dict, None when the run dir has none (or ctx names no run dir)."""
    rd = getattr(ctx, "run_dir", None)
    if not isinstance(rd, (str, os.PathLike)):
        return None
    f = Path(rd) / "RUNTIME.json"
    if not f.is_file():
        return None
    try:
        rt = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{f} unreadable ({type(e).__name__})")
    if not isinstance(rt, dict):
        raise ValueError(f"{f} must be an object")
    return rt


def validate_session(ctx) -> None:
    """`ctx.evolver_session` must be one of stateful_layout.SESSIONS, must agree
    with the run's RUNTIME.json when one exists (the driver and the upload check read that file), and
    "stateful" HALTs unless ALL nine rules hold. Each message names the key and the remedy. Raises
    loop.Halt; returns None. "per_epoch" (absent) returns at once: today's path is untouched."""
    import src.orchestrator.loop as L
    key = SL.SESSION_KEY
    v = getattr(ctx, "evolver_session", SL.PER_EPOCH)
    if not isinstance(v, str) or v not in SL.SESSIONS:
        raise L.Halt(f"RUNTIME.json {key}={v!r} is not one of {SL.SESSIONS}. Absent means 'per_epoch'; "
                     f"a typo must not silently mean either.")
    try:
        rt = _runtime(ctx)
    except ValueError as e:
        raise L.Halt(str(e))
    if rt is not None:
        try:
            declared = SL.evolver_session(runtime=rt)
        except ValueError as e:
            raise L.Halt(str(e))
        if declared != v:
            raise L.Halt(f"RUNTIME.json {key}={declared!r} but this Ctx carries {key}={v!r}: the loop, the driver "
                         f"and the upload check read the same file and must agree (main() reads the key from "
                         f"RUNTIME.json; a hand-built Ctx must match it)")
    if v != SL.STATEFUL:
        return
    head = f"RUNTIME.json {key}=stateful"
    # 1. the loop renders the message; no per-epoch AO exists in a stateful session
    if getattr(ctx, "prompt_mode", "ao") != "template":
        raise L.Halt(f"{head} needs prompt_mode=template (got {getattr(ctx, 'prompt_mode', 'ao')!r}): the loop "
                     f"renders the message and the system prompt; no per-epoch AO exists in a stateful session. "
                     f"Set prompt_mode: \"template\" or drop {key}.")
    # 2. the long-lived container lives on the leg pool's machine
    host = getattr(ctx, "evolver_host", "local")
    pl = L._evolver_placement(ctx, configured=True) if host == "daytona" else None
    if host != "daytona" or pl != "pool_container":
        raise L.Halt(f"{head} needs evolver_host=daytona with evolver_daytona.placement=pool_container "
                     f"(got evolver_host={host!r}, placement={pl!r}): the one long-lived container is a pool "
                     f"container on the run's leg machine. Declare both, or drop {key}.")
    # 3. a session cannot resume a container that epoch 1 never had
    cfg = getattr(ctx, "evolver_daytona", {}) or {}
    if cfg.get("placement_from_epoch") is not None:
        raise L.Halt(f"{head} with evolver_daytona.placement_from_epoch={cfg.get('placement_from_epoch')!r}: a "
                     f"session cannot resume a container that epoch 1 never had. Remove placement_from_epoch "
                     f"(every epoch runs in the run's container), or drop {key}.")
    # 4. the score table and the cumulative /evo/traces/epoch_* need every epoch's evidence
    ev = getattr(ctx, "evolver_evidence", "current_epoch")
    if ev != "all_epochs":
        raise L.Halt(f"{head} needs evolver_evidence=all_epochs (got {ev!r}): the score table and the cumulative "
                     f"/evo/traces/epoch_* need every epoch's evidence (the per-epoch upload is a delta of the "
                     f"all-epochs grant). Declare evolver_evidence: \"all_epochs\", or drop {key}.")
    # 5. the container layout is the rollouts layout
    from src.orchestrator import rollout_evidence as _re
    lay = getattr(ctx, "evidence_layout", "digests")
    if lay != "evo":
        raise L.Halt(f"{head} needs {_re.LAYOUT_KEY}=evo (got {lay!r}): /evo/traces/epoch_K/<task id>_s<i>.jsonl, "
                     f"/evo/solver_workspaces/epoch_K/ and /evo/scores/epoch_K.json "
                     f"(src/orchestrator/evo_layout.py) are what the evolver is told to read. Declare {_re.LAYOUT_KEY}: "
                     f"\"evo\", or drop {key}.")
    # 6. one claude session, on an explicit model id
    spec = getattr(ctx, "spec", None)
    harness = getattr(spec, "evolver_harness", "claude_cli")
    model = getattr(spec, "evolver_model", None)
    if harness != "claude_cli":
        raise L.Halt(f"{head} needs evolver_harness=claude_cli (got {harness!r}): the session is `claude -p "
                     f"--session-id` / `--resume`; there is no other harness with a resumable session.")
    if claude_pins.expected_model(model) is None:
        raise L.Halt(f"{head} needs an explicit claude-* evolver_model in SPEC.json (got {model!r}): the session "
                     f"check asserts the id every epoch; an alias is not asserted.")
    # 7. no --effort flag: the evolver runs at Anthropic's default effort
    try:
        lv = claude_pins.effort(os.environ)
    except claude_pins.ModelPinError as e:
        raise L.Halt(f"{head}: {e}; {EFFORT_REMEDY}")
    if lv is not None:
        raise L.Halt(f"stateful evolver runs {model} at Anthropic's default effort : the effort stays "
                     f"unset (use anthropic's default); {EFFORT_REMEDY}; the shell that runs the loop exports "
                     f"{claude_pins.EFFORT_ENV}={os.environ.get(claude_pins.EFFORT_ENV)!r} -> --effort {lv}")
    # 8. the documentation block must agree with the live channel
    if rt is None:
        raise L.Halt(f"{head}: the run has no RUNTIME.json to read claude.effort from; declare claude.effort: "
                     f"\"default\" there (the documentation block must agree with the live channel).")
    cl = rt.get("claude") if isinstance(rt.get("claude"), dict) else {}
    if cl.get("effort") != "default":
        raise L.Halt(f"{head}: RUNTIME.json claude.effort={cl.get('effort')!r} but the live channel is "
                     f"{claude_pins.EFFORT_ENV}=default (no --effort flag); the documentation block must agree: "
                     f"set claude.effort: \"default\".")
    # 9. the leg pool keeps the machine; its TTL sizes the egress proxy's life
    dl = rt.get("daytona_leg") if isinstance(rt.get("daytona_leg"), dict) else {}
    if dl.get("pool") is not True:
        raise L.Halt(f"{head} needs RUNTIME.json daytona_leg.pool: true (got {dl.get('pool')!r}): the container "
                     f"lives on the pool's machine between epochs.")
    ttl = dl.get("pool_ttl_minutes")
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl <= 0:
        raise L.Halt(f"{head} needs RUNTIME.json daytona_leg.pool_ttl_minutes (a positive number; got {ttl!r}): it "
                     f"sizes the container's egress proxy life (max_seconds = pool_ttl_minutes*60).")


# ══ The evolver's texts: two FIXED templates under src/orchestrator/prompts/, rendered by the loop with four
# ══ placeholders ({n_samples}; {utc_now}, {deadline_utc}, {N} and {N-1}). The loop computes nothing else for the
# ══ evolver: no facts, no digests, no score tables, no report format. Work done by the loop on the evolver's behalf
# ══ adds complexity and ties the method to one benchmark; the evolver model can read the raw material itself.
RenderError = ET.RenderError
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_TEMPLATE = PROMPTS_DIR / "EVOLVER_SYSTEM.md"
MESSAGE_TEMPLATE = PROMPTS_DIR / "EPOCH_MESSAGE.md"
UTC_FMT = "%Y-%m-%d %H:%M"
_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z_0-9-]*\}")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fill(template: str, values: dict, what: str) -> str:
    """The template with every `{key}` of `values` replaced (in the given order); an unfilled placeholder or a
    host-side path / dir name in the result is a RenderError."""
    text = template
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    left = _PLACEHOLDER_RE.findall(text)
    if left:
        raise RenderError(f"{what}: unfilled placeholder(s) {sorted(set(left))[:4]}")
    hits = SL.forbidden_mac_strings(text)
    if hits:
        raise RenderError(f"{what} names a host-side path or dir: {hits[:2]}")
    return text


def render_system_prompt(*, n_samples: int) -> dict:
    """EVOLVER_SYSTEM.md (Part 1): the fixed template with {n_samples}. Rendered ONCE at epoch 1 and passed with
    --append-system-prompt at every epoch. Pure: the same input gives the same bytes."""
    n = SL._epoch_index(n_samples, floor=1)
    tpl = SYSTEM_TEMPLATE.read_text(encoding="utf-8")
    text = _fill(tpl, {"n_samples": n}, "system prompt")
    data = text.encode("utf-8")
    if len(data) > de.MAX_PROMPT_BYTES:
        raise RenderError(f"system prompt is {len(data)} B > {de.MAX_PROMPT_BYTES} (one argv string)")
    return {"text": text, "sha256": _sha_bytes(data), "bytes": len(data), "n_samples": n,
            "template": str(SYSTEM_TEMPLATE), "template_sha256": _sha_bytes(tpl.encode("utf-8"))}


def render_epoch_message(epoch: int, *, utc_now: str, deadline_utc: str) -> dict:
    """ITER{N}_PROMPT.md (Part 2): the ONE template with {utc_now}, {deadline_utc}, {N} and {N-1}; the whole user
    message of the evolver's turn. Both times are 'YYYY-MM-DD HH:MM' UTC."""
    N = SL._epoch_index(epoch, floor=1)
    for k, v in (("utc_now", utc_now), ("deadline_utc", deadline_utc)):
        try:
            time.strptime(str(v), UTC_FMT)
        except (TypeError, ValueError):
            raise RenderError(f"{k}={v!r} is not a '{UTC_FMT}' UTC time")
    tpl = MESSAGE_TEMPLATE.read_text(encoding="utf-8")
    text = _fill(tpl, {"N-1": N - 1, "N": N, "utc_now": utc_now, "deadline_utc": deadline_utc}, f"epoch-{N} message")
    data = text.encode("utf-8")
    if len(data) > de.MAX_PROMPT_BYTES:
        raise RenderError(f"epoch-{N} message is {len(data)} B > {de.MAX_PROMPT_BYTES} (one argv string)")
    return {"text": text, "sha256": _sha_bytes(data), "bytes": len(data), "epoch": N, "utc_now": utc_now,
            "deadline_utc": deadline_utc, "template": str(MESSAGE_TEMPLATE),
            "template_sha256": _sha_bytes(tpl.encode("utf-8"))}


def utc_str(t: float | None = None) -> str:
    return time.strftime(UTC_FMT, time.gmtime(time.time() if t is None else t))


# ── the render step ("evolver message + system prompt rendered") ──────────────────────────────────
def _render_inputs(ctx) -> dict:
    """What the two renders read for THIS epoch: SPEC n_samples and RUNTIME evolver_timeout_s (the deadline). Raises
    loop.Halt on a run the stateful render cannot serve."""
    import src.orchestrator.loop as L
    if L._evolver_session(ctx) != SL.STATEFUL:
        raise L.Halt(f"the stateful render step ran on a run declaring {SL.SESSION_KEY}="
                     f"{getattr(ctx, 'evolver_session', SL.PER_EPOCH)!r}")
    S = getattr(ctx.spec, "n_samples", None)
    if not isinstance(S, int) or isinstance(S, bool) or S < 1:
        raise L.Halt(f"SPEC n_samples={S!r}: the system prompt states the samples per task")
    timeout = getattr(ctx, "evolver_timeout_s", None)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise L.Halt(f"RUNTIME evolver_timeout_s={timeout!r}: the message states the epoch's deadline from it")
    return {"run_dir": Path(ctx.run_dir), "epoch": ctx.epoch, "n_samples": S, "timeout_s": int(timeout)}


def _render_record(ctx, inp: dict, sp: dict, msg: dict, sysp: Path) -> dict:
    import src.orchestrator.loop as L
    return {
        "mode": "template", "session": SL.STATEFUL, "epoch": ctx.epoch, "model_call": False,
        "what": ("ITER{N}_PROMPT.md (the epoch message) and EVOLVER_SYSTEM.md (the appended system prompt, rendered "
                 "once at epoch 1) rendered by the loop from the two fixed templates in src/orchestrator/prompts/; no "
                 "model was called; nothing else is computed for the evolver"),
        "renderer": {"module": "src/orchestrator/stateful_evolver.py",
                     "sha256": _sha_bytes(Path(__file__).read_bytes())},
        "prompt": {"path": str(ctx.paths.prompt), "sha256": msg["sha256"], "bytes": msg["bytes"],
                   "utc_now": msg["utc_now"], "deadline_utc": msg["deadline_utc"],
                   "template": msg["template"], "template_sha256": msg["template_sha256"]},
        "system_prompt": {"path": str(sysp), "sha256": sp["sha256"], "bytes": sp["bytes"], "rendered_epoch": 1,
                          "template": sp["template"], "template_sha256": sp["template_sha256"]},
        "n_samples": inp["n_samples"], "evolver_timeout_s": inp["timeout_s"],
        "record": L.RENDER_RECORD.format(epoch=ctx.epoch),
    }


def run_render_session_prompt(ctx) -> None:
    """THE LOOP RENDERS THE EVOLVER'S MESSAGE AND, AT EPOCH 1, THE SYSTEM PROMPT. No model call. Refuses rather than
    overwrites: an existing ITER{N}_PROMPT.md, e{N}_PROMPT_RENDER.json or e{N}_PROMPT_AUTHOR.json is evidence a person
    may need. EVOLVER_SYSTEM.md is written once, at epoch 1 (an existing byte-identical one is kept); a later epoch
    requires it and never rewrites it. The message carries the send time and the deadline (send time + RUNTIME
    evolver_timeout_s), recorded so the verify re-renders the same bytes."""
    import src.orchestrator.loop as L
    rd = Path(ctx.run_dir)
    p = ctx.paths.prompt
    rec_p = rd / L.RENDER_RECORD.format(epoch=ctx.epoch)
    sysp = rd / SL.SYSTEM_PROMPT_FILE
    for f in (p, rec_p, rd / f"e{ctx.epoch}_PROMPT_AUTHOR.json"):
        if f.exists():
            raise L.Halt(f"refusing to render over {f.name}, which exists. Move {p.name}, {rec_p.name} and any "
                         f"e{ctx.epoch}_PROMPT_AUTHOR.json aside (keep them) and re-run --step.")
    inp = _render_inputs(ctx)
    now = time.time()
    try:
        sp = render_system_prompt(n_samples=inp["n_samples"])
        msg = render_epoch_message(ctx.epoch, utc_now=utc_str(now), deadline_utc=utc_str(now + inp["timeout_s"]))
    except RenderError as e:
        raise L.Halt(f"the evolver texts cannot be rendered: {e}")
    if ctx.epoch >= 2:
        if not sysp.is_file():
            raise L.Halt(f"epoch {ctx.epoch} of a stateful session needs {sysp.name}, rendered at epoch 1; it is "
                         f"absent (the session's system prompt is never re-rendered)")
        actual = _sha_bytes(sysp.read_bytes())
        e1 = rd / L.RENDER_RECORD.format(epoch=1)
        try:
            rec1 = json.loads(e1.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise L.Halt(f"{e1.name} (the epoch-1 render record) is unreadable ({type(e).__name__})")
        if ((rec1.get("system_prompt") or {}).get("sha256")) != actual:
            raise L.Halt(f"{sysp.name} hashes to {actual[:12]}, {e1.name} recorded "
                         f"{str((rec1.get('system_prompt') or {}).get('sha256'))[:12]}: edited since epoch 1")
        if sp["sha256"] != actual:
            raise L.Halt(f"the system prompt no longer re-renders to {sysp.name} ({sp['sha256'][:12]} vs "
                         f"{actual[:12]}): the template or SPEC n_samples changed since epoch 1; the session's system "
                         f"prompt is fixed for the run")
    elif sysp.exists():
        if sysp.is_symlink() or not sysp.is_file() or _sha_bytes(sysp.read_bytes()) != sp["sha256"]:
            raise L.Halt(f"refusing to render over {sysp.name}, which exists with other bytes. Move it aside (keep "
                         f"it) and re-run --step.")
    record = _render_record(ctx, inp, sp, msg, sysp)
    record["rendered_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    if ctx.epoch == 1 and not sysp.exists():
        with open(sysp, "x", encoding="utf-8") as fh:
            fh.write(sp["text"])
    with open(p, "x", encoding="utf-8") as fh:
        fh.write(msg["text"])
    tmp = rec_p.with_name(f".{rec_p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str))
    os.replace(str(tmp), str(rec_p))


def _read_record(rd: Path, epoch: int):
    """(record, problem) of e{epoch}_PROMPT_RENDER.json."""
    import src.orchestrator.loop as L
    rec_p = rd / L.RENDER_RECORD.format(epoch=epoch)
    if not rec_p.is_file():
        return None, f"no {rec_p.name}"
    try:
        rec = json.loads(rec_p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{rec_p.name} is unreadable ({type(e).__name__})"
    if not isinstance(rec, dict):
        return None, f"{rec_p.name} is not an object"
    return rec, ""


def v_session_prompt_rendered(ctx):
    """The prompt-rendered check for the stateful session. The message exists with the loop's render record beside it and is byte-for-byte
    what the record says; the system prompt is the epoch-1 render (its sha in this epoch's record == the file ==
    e1's record); and RE-RENDERING BOTH NOW from the templates (the message with the RECORDED send time and deadline)
    gives the same bytes: an edited file, an edited template or a changed SPEC n_samples all fail here."""
    import src.orchestrator.loop as L
    rd = Path(ctx.run_dir)
    p = ctx.paths.prompt
    ao = rd / f"e{ctx.epoch}_PROMPT_AUTHOR.json"
    if ao.is_file():
        return False, (f"{ao.name} exists: this epoch's message was authored by the AO, but the run declares a "
                       f"stateful session (the loop renders it). Move {p.name} and {ao.name} aside and re-run --step.")
    if not p.is_file():
        return False, f"no message at {p.name} — the loop has not rendered it yet"
    rec, prob = _read_record(rd, ctx.epoch)
    if rec is None:
        return False, f"{p.name} exists but {prob}: a message the loop cannot attribute to a render"
    rec_name = L.RENDER_RECORD.format(epoch=ctx.epoch)
    if rec.get("mode") != "template" or rec.get("session") != SL.STATEFUL or rec.get("model_call") is not False:
        return False, (f"{rec_name} is not a stateful template render record (mode {rec.get('mode')!r}, session "
                       f"{rec.get('session')!r})")
    actual = _sha_bytes(p.read_bytes())
    pr = rec.get("prompt") or {}
    if pr.get("sha256") != actual:
        return False, (f"{p.name} hashes to {actual[:12]} but {rec_name} recorded {str(pr.get('sha256'))[:12]}: "
                       f"edited after the render")
    sysp = rd / SL.SYSTEM_PROMPT_FILE
    if sysp.is_symlink() or not sysp.is_file():
        return False, f"no system prompt at {sysp.name}"
    sys_actual = _sha_bytes(sysp.read_bytes())
    want = rec.get("system_prompt") or {}
    if want.get("sha256") != sys_actual or want.get("rendered_epoch") != 1:
        return False, (f"{sysp.name} hashes to {sys_actual[:12]} but {rec_name} recorded "
                       f"{str(want.get('sha256'))[:12]} (rendered_epoch {want.get('rendered_epoch')!r}): the system "
                       f"prompt was edited after the epoch-1 render")
    if ctx.epoch >= 2:
        rec1, prob = _read_record(rd, 1)
        if rec1 is None:
            return False, f"the epoch-1 render record is needed to attribute {sysp.name}: {prob}"
        if ((rec1.get("system_prompt") or {}).get("sha256")) != sys_actual:
            return False, f"{sysp.name} is not the epoch-1 render ({rec_name} and e1's record disagree with the file)"
    try:
        inp = _render_inputs(ctx)
        sp = render_system_prompt(n_samples=inp["n_samples"])
        msg = render_epoch_message(ctx.epoch, utc_now=str(pr.get("utc_now")), deadline_utc=str(pr.get("deadline_utc")))
    except (L.Halt, RenderError) as e:
        return False, f"the texts no longer re-render from what is on disk: {e}"
    if sp["sha256"] != sys_actual:
        return False, (f"re-rendering the system prompt now gives {sp['sha256'][:12]}, but {sysp.name} is "
                       f"{sys_actual[:12]}: the template or SPEC n_samples changed since epoch 1")
    if msg["sha256"] != actual:
        return False, (f"re-rendering the message now (send time {pr.get('utc_now')}, deadline {pr.get('deadline_utc')}) "
                       f"gives {msg['sha256'][:12]}, but {p.name} is {actual[:12]}: the template changed since the render. "
                       f"Move the message and {rec_name} aside and re-run --step.")
    return True, (f"rendered by the loop from the two templates (no model call): message {msg['bytes']} B (sent "
                  f"{pr.get('utc_now')} UTC, deadline {pr.get('deadline_utc')} UTC), system prompt {sp['bytes']} B "
                  f"(epoch-1 render, sha256:{sys_actual[:12]}); both re-render byte-identically now")


# ── the evolver's material: <run>/evo_e{K}/ from the leg stores ("evolver material laid out") ────
def _evo_source_leg(ctx, K: int):
    """The leg whose store holds epoch K's rollouts: K = 0 the declared base train leg; K >= 1 skills_vK's train leg."""
    import src.orchestrator.loop as L
    if K == 0:
        return Path(ctx.base_leg) if getattr(ctx, "base_leg", "") else None
    return L.EpochPaths(Path(ctx.run_dir), K).leg("train")


def run_build_evo_layout(ctx) -> None:
    """Build <run>/evo_e{N-1}/ (this epoch's newest material) from its leg's store when absent (evo_layout.build:
    the raw session logs, the solver workspaces, scores.json). An existing dir is never rebuilt here."""
    import src.orchestrator.loop as L
    from src.orchestrator import evo_layout as EL
    out = ctx.paths.evo
    if out.is_dir():
        return
    leg = _evo_source_leg(ctx, ctx.epoch - 1)
    if leg is None or not Path(leg).is_file():
        raise L.Halt(f"the leg behind {out.name} ({leg}) is not on disk")
    try:
        s = EL.build(out, leg_json=leg, epoch=ctx.epoch - 1)
    except (EL.LayoutError, OSError, ValueError) as e:
        raise L.Halt(f"{out.name} could not be built: {e}")
    print(f"  [evo layout] {out.name}: {s['rollouts']} rollouts (scored {s['scored']}, mean {s['mean_score']}, "
          f"grader_failed {s['grader_failed']}); {s['files']} file(s), "
          f"traces {s['trace_bytes'] / 1e6:.1f} MB")


def v_evo_layout(ctx):
    """Every evo_e0..evo_e{N-1} is exactly what evo_layout.build writes from its leg's store now (evo_layout.check:
    each measured rollout's trace byte-identical to the store's stream, its workspace present, scores.json == the
    store's verdicts as scores in leg order, nothing else, no symlink)."""
    from src.orchestrator import evo_layout as EL
    probs = []
    dirs = ctx.paths.evo_dirs()
    for K, d in enumerate(dirs):
        leg = _evo_source_leg(ctx, K)
        if leg is None or not Path(leg).is_file():
            probs.append(f"{d.name}: its leg ({leg}) is not on disk")
            continue
        if not d.is_dir():
            probs.append(f"{d.name}: absent (the runner builds this epoch's; an earlier epoch's must already exist)")
            continue
        p = EL.check(d, leg_json=leg)
        if p:
            probs.append(f"{d.name}: " + "; ".join(p[:3]))
    if probs:
        return False, ("; ".join(probs[:4]) + ". REMEDY: move the bad dir OUT of the run dir (never delete material an "
                       "evolver read) and re-run --step")
    n = sum(EL.summary(d)["rollouts"] for d in dirs)
    return True, (f"{len(dirs)} epoch dir(s) evo_e0..evo_e{len(dirs) - 1} re-derived from their leg stores: {n} rollouts "
                  f"(raw session logs, solver workspaces, scores.json), nothing computed")


SESSION_DIR = "e{epoch}_evolver_session"          # <run>/e{N}_evolver_session/ (the downloaded projects tree)
MODEL_RECORD = "e{epoch}_EVOLVER_MODEL.json"       # loop._record_evolver_model's record (pins + the log's models)
FINISH_OUT = "EVOLVER_SESSION_FINISH.out"          # the finish subprocess's output, appended per attempt
FINISH_TIMEOUT_S = 1200
FINISH_SCRIPT = "scripts/daytona/evolver_session.py"
_TOKEN_ENV = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY")
_MACHINE_GONE = ("deleted", "released", "failed")   # evolver_session.py finish's `present` rule


def _read_json(p: Path) -> tuple:
    """(dict, "") or (None, problem)."""
    if not p.is_file():
        return None, f"no {p.name}"
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{p.name} is unreadable ({type(e).__name__}: {e})"
    if not isinstance(d, dict):
        return None, f"{p.name} is not an object"
    return d, ""


def _state_of(rd: Path) -> tuple:
    """(state record, "") or (None, problem): the session's identity (stateful_layout.load_state)."""
    try:
        st = SL.load_state(rd)
    except ValueError as e:
        return None, f"the session state record is unreadable: {e}"
    if st is None:
        return None, (f"no session state record at {SL.state_path(rd)}: the container cannot be attributed to a "
                      f"session (the stateful driver writes it at the first dispatch)")
    return st, ""


def _receipt_of(ctx) -> tuple:
    """(receipt, "") or (None, problem) for this epoch's e{N}_EVOLVER_CONTAINMENT.json."""
    return _read_json(Path(ctx.run_dir) / SL.RECEIPT_FILE.format(epoch=ctx.epoch))


def _session_of(receipt: dict) -> tuple:
    """(daytona block, container block, session block) of a receipt."""
    dd = receipt.get("daytona") if isinstance(receipt.get("daytona"), dict) else {}
    c = dd.get("container") if isinstance(dd.get("container"), dict) else {}
    ses = c.get("session") if isinstance(c.get("session"), dict) else {}
    return dd, c, ses


def _probe_sha_for(state: dict, c: dict) -> tuple:
    """(the creation-time probe sha to assert, note) or (None, problem): the state's for the session's current
    container; a container the session later REBUILT from (state.rebuilds[*].old_cid) carries its own."""
    cid = c.get("cid")
    if cid and cid == state.get("cid"):
        return (state.get("container") or {}).get("probe_pre_sha256"), ""
    if cid and cid in {r.get("old_cid") for r in state.get("rebuilds") or []}:
        return c.get("probe_pre_sha256"), f" (container {cid} was later rebuilt; its own creation-time probe)"
    return None, (f"the receipt's container {cid!r} is neither the session's ({state.get('cid')!r}) nor one the "
                  f"session rebuilt from")


def _transcript(root: Path, sid: str) -> Path:
    """The ONE session transcript inside a downloaded / snapshotted projects tree."""
    return Path(root) / Path(de.VM_SESSION_DIR).relative_to(de.VM_PROJECTS) / f"{sid}.jsonl"


def _tree_shas(d: Path) -> dict:
    """{rel: sha256} of every regular file under d (no symlink followed; a symlink is listed as such)."""
    out = {}
    d = Path(d)
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d).as_posix()
        if p.is_symlink():
            out[rel] = "symlink"
        elif p.is_file():
            out[rel] = _sha_bytes(p.read_bytes())
    return out


def _stream_events(log: Path) -> list:
    out = []
    for line in log.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def v_session_contained(ctx):
    """#294 for the stateful session: this epoch's receipt is a pool-container receipt of the
    run's ONE session (container.session.mode stateful, its sid == the state record's == daytona.session_id),
    the state marks the epoch collected, and pool_container_evolver.check_pool_container_receipt(session=
    "stateful") holds with the writable dirs [/evo/staging, /evo/notes] and the SAME creation-time probe (the
    state's probe_pre_sha256; a rebuilt-from container's own). Once the session is torn down (the finish record
    exists), the teardown rules apply to the record too (the final check at DRIVE DONE). Then the evidence half:
    the receipt names exactly the declared all-epochs evidence dirs, unchanged before/after (the host hashes)."""
    import src.orchestrator.loop as L
    from src.orchestrator import pool_container_evolver as pce
    rd = Path(ctx.run_dir)
    rec_p = rd / SL.RECEIPT_FILE.format(epoch=ctx.epoch)
    log = L._evolver_log_path(ctx)
    if not rec_p.is_file():
        if log.is_file():
            return False, (f"{log.name} exists but no {rec_p.name}: the stateful driver (scripts/daytona/"
                           f"evolver_session.py) writes its receipt before the launch, so this evolver did not run "
                           f"through the run's container session")
        return True, (f"{L.UNMEASURED}no {rec_p.name} yet — the evolver has not been dispatched for this epoch. "
                      f"The dispatch step ahead of this one owns that ordering.")
    d, prob = _read_json(rec_p)
    if d is None:
        return False, prob
    if not d.get("sandboxed") or d.get("containment") != "daytona":
        return False, (f"world MISMATCH: RUNTIME evolver_session=stateful (a pool container on Daytona) but "
                       f"{rec_p.name} records containment={d.get('containment')!r} sandboxed={d.get('sandboxed')!r}")
    dd, c, ses = _session_of(d)
    if dd.get("placement") != pce.PLACEMENT:
        return False, (f"placement MISMATCH: RUNTIME evolver_daytona.placement={pce.PLACEMENT!r} but {rec_p.name} "
                       f"records {dd.get('placement') or 'own_sandbox'!r}.")
    if ses.get("mode") != SL.STATEFUL:
        return False, (f"{rec_p.name} records no stateful session (container.session.mode={ses.get('mode')!r}): the "
                       f"per-epoch driver ran, not the run's container session")
    state, prob = _state_of(rd)
    if state is None:
        return False, prob
    sid = state.get("sid")
    if not sid or ses.get("sid") != sid or dd.get("session_id") != sid:
        return False, (f"{rec_p.name} records session {str(ses.get('sid'))[:8]!r} / daytona.session_id "
                       f"{str(dd.get('session_id'))[:8]!r}, not the run's session {str(sid)[:8]!r} (state record)")
    ep = (state.get("epochs") or {}).get(str(ctx.epoch)) or {}
    if ep.get("status") != "collected":
        return False, (f"the state record marks epoch {ctx.epoch} {ep.get('status')!r}, not collected "
                       f"({ep.get('why') or 'the driver did not finish its collect'})")
    probe_sha, note = _probe_sha_for(state, c)
    if probe_sha is None:
        return False, note or f"the state record carries no creation-time probe sha for container {c.get('cid')!r}"
    kw = L._pool_container_check_kw(ctx)    # require_snapshot_ref None = not pinned: recorded, not compared
    teardown, td_note = None, ""
    td_p = rd / SL.TEARDOWN_RECORD
    if state.get("torn_down") or td_p.is_file():
        td, prob = _read_json(td_p)
        if td is None:
            return False, f"the session is torn down but its record cannot be read: {prob}"
        if td.get("cid") == c.get("cid"):
            teardown, td_note = td, "; teardown record applied"
    ok, det = pce.check_pool_container_receipt(d, log=log, staging=None, session=SL.STATEFUL,
                                               rw_paths=[r["path"] for r in SL.STATEFUL_RW], probe_pre_sha256=probe_sha,
                                               teardown=teardown, **kw)
    if not ok:
        return False, f"pool-container containment (stateful): {det}"
    want_ev = sorted(str(Path(p).resolve()) for p in L._evidence_dirs(ctx))
    got_ev = sorted(str(Path(str(p)).resolve()) for p in (d.get("evidence_dirs") or []))
    if got_ev != want_ev:
        return False, (f"{rec_p.name} records evidence dirs {[Path(p).name for p in got_ev]} but "
                       f"evolver_evidence={getattr(ctx, 'evolver_evidence', '')!r} at epoch {ctx.epoch} declares "
                       f"{[Path(p).name for p in want_ev]}. The dispatch did not grant the evidence this run declared.")
    before = d.get("evidence_sha256_before")
    if not isinstance(before, dict) or not before:
        return False, (f"{rec_p.name} names evidence dirs but records no pre-dispatch hash — whether the evolver "
                       f"changed its own evidence cannot be established")
    after, src = d.get("evidence_sha256_after"), "recorded after the evolver exited"
    if not isinstance(after, dict) or not after:
        from src.orchestrator.evidence_surface import hashes as _hashes
        after, src = _hashes(list(before)), "computed now (no post-run hash recorded)"
    changed = sorted(Path(p).name for p in before if after.get(p) != before[p])
    if changed:
        return False, (f"the evidence CHANGED between dispatch and {src}: {changed}. A granted digest dir is "
                       f"read-only evidence for this and every later epoch; find what wrote it before trusting "
                       f"anything built on it.")
    return True, (f"evolver ran in the session's container {c.get('cid')} (epoch {ctx.epoch}, "
                  f"{'resumed' if ses.get('resume') else 'first turn'}, session {sid[:8]}{note}); {det}"
                  f"{td_note}; evidence all_epochs: {len(before)} dir(s), unchanged ({src})")


def v_session_read_nothing_forbidden(ctx):
    """#213 for the stateful session: stateful_layout.check_stateful_upload PASS (the container held exactly this
    run's train-side files + the adapter's reference/, cumulatively; re-derived from the host bytes now). A SKIP is
    a failure (an audit that cannot run must not read as one that passed). No id scan of the log or the transcripts
    (they echo what the evolver read, which is the manifest), nor of what it WROTE."""
    import src.orchestrator.loop as L
    rd, N = Path(ctx.run_dir), ctx.epoch
    log = L._evolver_log_path(ctx)
    if not log.is_file():
        return False, (f"no evolver log at {log.name}, so neither the container-content (#213) nor the held-out (#186) "
                       f"audit can run. Declare evolver_log if it lives elsewhere — an audit that cannot run must "
                       f"not read as one that passed.")
    d, prob = _receipt_of(ctx)
    if d is None:
        return False, f"{prob}: the container's upload cannot be audited"
    listing, prob = _read_json(rd / SL.LISTING_FILE.format(epoch=N))
    if listing is None:
        return False, f"{prob}: the container's cumulative listing cannot be audited"
    state, prob = _state_of(rd)
    if state is None:
        return False, prob
    # the state may already carry later epochs' rows (the board re-reads an earlier epoch): judge epoch N's view
    st_n = dict(state, uploaded=[r for r in state.get("uploaded") or [] if int(r.get("epoch", 0)) <= N])
    split = Path(ctx.spec.split_path)
    c = SL.check_stateful_upload(run_dir=rd, epoch=N, split=split, state=st_n, receipt=d, listing=listing)
    if c.status == "FAIL":
        return False, f"{c.name}: {c.detail}"
    if c.status == "SKIP":
        return False, f"{c.name} could not run: {c.detail}"
    # No id scan of the log / transcripts here: they echo what the evolver READ, which is exactly the
    # manifest (train material + the adapter's reference/, granted unscanned); the upload check above is the proof.
    return True, (f"the container-content audit passed against {log.name}: the container held only this run's train-side "
                  f"files and the adapter's reference/ (cumulative through epoch {N}, {len(st_n['uploaded'])} rows)")


def v_session_continued(ctx):
    """The ONE claude session continued. The stream log's first system/init `session_id` == the
    state's sid == the receipt's daytona.session_id; the launch argv carried `--session-id <sid>` at the
    container's first turn and `--resume <sid>` after (never both), `--append-system-prompt`, no `--effort`;
    e{N}_EVOLVER_MODEL.json's pins carry no effort; the transcript
    <run>/e{N}_evolver_session/-evo-work/<sid>.jsonl exists and, for N >= 2, the previous epoch's copy (or the
    snapshot the session was rebuilt from) is a byte PREFIX of it; the log ends with a `result` event and the
    receipt's claude_rc is 0 -- unless the receipt records a DEADLINE stop (at the deadline the
    staging is taken as it stands; the stopped turn has no `result` event and a non-zero rc)."""
    import src.orchestrator.loop as L
    rd, N = Path(ctx.run_dir), ctx.epoch
    d, prob = _receipt_of(ctx)
    if d is None:
        return False, prob
    dd, c, ses = _session_of(d)
    state, prob = _state_of(rd)
    if state is None:
        return False, prob
    sid = state.get("sid") or ""
    if dd.get("session_id") != sid or ses.get("sid") != sid:
        return False, (f"the receipt's session {str(dd.get('session_id'))[:8]!r} is not the run's {sid[:8]!r} (state "
                       f"record): another claude session ran this epoch")
    log = L._evolver_log_path(ctx)
    if not log.is_file():
        return False, f"no evolver log at {log.name}"
    events = _stream_events(log)
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    if init is None:
        return False, f"{log.name} carries no system/init event: which session ran cannot be shown"
    if init.get("session_id") != sid:
        return False, (f"{log.name}'s init names session {str(init.get('session_id'))[:8]!r}, not the run's "
                       f"{sid[:8]!r}: the evolver ran in another session")
    deadline = dd.get("deadline_stop")
    if not deadline:
        if not any(e.get("type") == "result" for e in events):
            return False, f"{log.name} has no `result` event: the turn did not end"
        if dd.get("claude_rc") != 0:
            return False, (f"the receipt records claude_rc={dd.get('claude_rc')!r}; a turn that did not exit 0 is not "
                           f"continued")
    argv = [str(x) for x in (dd.get("launch_argv") or [])]
    created = ses.get("created_epoch")
    resume = bool(ses.get("resume"))
    have_sid = argv[argv.index("--session-id") + 1] if "--session-id" in argv[:-1] else None
    have_res = argv[argv.index("--resume") + 1] if "--resume" in argv[:-1] else None
    if resume:
        if have_res != sid or have_sid is not None:
            return False, (f"a resumed turn must launch `--resume {sid[:8]}` and no `--session-id`; the argv carries "
                           f"--resume {str(have_res)[:8]!r}, --session-id {str(have_sid)[:8]!r}")
    else:
        if N != created:
            return False, (f"a first turn (`--session-id`) at epoch {N} of a session created at epoch {created}: "
                           f"the session did not continue")
        if have_sid != sid or have_res is not None:
            return False, (f"the first turn must launch `--session-id {sid[:8]}` and no `--resume`; the argv carries "
                           f"--session-id {str(have_sid)[:8]!r}, --resume {str(have_res)[:8]!r}")
    if "--append-system-prompt" not in argv:
        return False, "the launch argv carries no --append-system-prompt (the session's system prompt was not sent)"
    if "--effort" in argv:
        return False, (f"the launch argv carries --effort {argv[argv.index('--effort') + 1:][:1]}: the evolver "
                       f"runs with no --effort flag (EVOSKILL_CLAUDE_EFFORT=default)")
    m, prob = _read_json(rd / MODEL_RECORD.format(epoch=N))
    if m is None:
        return False, f"{prob} (the loop writes it after the dispatch): the pins cannot be shown"
    pins = m.get("pins") if isinstance(m.get("pins"), dict) else {}
    if pins.get("effort_argv") or "CLAUDE_CODE_EFFORT_LEVEL" in (pins.get("env") or {}):
        return False, (f"{MODEL_RECORD.format(epoch=N)} records an effort pin (effort_argv {pins.get('effort_argv')!r}, "
                       f"env CLAUDE_CODE_EFFORT_LEVEL {(pins.get('env') or {}).get('CLAUDE_CODE_EFFORT_LEVEL')!r}): "
                       f"the evolver runs at Anthropic's default effort")
    t = _transcript(rd / SESSION_DIR.format(epoch=N), sid)
    if t.is_symlink() or not t.is_file() or t.stat().st_size == 0:
        return False, f"no transcript at {t.relative_to(rd)}: the session's projects tree was not installed"
    prefix_note = ""
    if N >= 2:
        rb = ses.get("rebuilt") if isinstance(ses.get("rebuilt"), dict) else None
        if rb and rb.get("from_snapshot"):
            prev, what = _transcript(Path(rb["from_snapshot"]) / "projects", sid), f"the rebuild's {Path(rb['from_snapshot']).name} snapshot"
        else:
            prev, what = _transcript(rd / SESSION_DIR.format(epoch=N - 1), sid), f"epoch {N - 1}'s installed copy"
            if not prev.is_file():
                prev = _transcript(SL.snapshot_dir(rd, N - 1) / "projects", sid)
                what = f"epoch {N - 1}'s snapshot"
        if prev.is_symlink() or not prev.is_file():
            return False, f"no previous transcript to compare with ({what} is absent): the session's growth cannot be shown"
        pb, tb = prev.read_bytes(), t.read_bytes()
        if not tb.startswith(pb) or len(tb) <= len(pb):
            return False, (f"the transcript did not GROW from {what}: {len(pb)} B -> {len(tb)} B, prefix "
                           f"{'kept' if tb.startswith(pb) else 'REWRITTEN'}; the session was not resumed")
        prefix_note = f"; the transcript grew {len(pb)} -> {len(tb)} B over {what} (prefix kept)"
    return True, (f"session {sid[:8]} continued: log init session_id == state == receipt; launched "
                  f"{'--resume' if resume else '--session-id'} {sid[:8]} with --append-system-prompt and no --effort "
                  f"(pins: no effort); "
                  + (f"stopped at its deadline (claude_rc {dd.get('claude_rc')!r})" if deadline
                     else "result event present, claude_rc 0") + prefix_note)


def v_session_snapshotted(ctx):
    """The snapshot check: .cache/evolver_state/<run>/e<N>/ exists and verifies (every listed file at its sha, nothing
    unlisted, no symlink), its MANIFEST names the session's sid and this epoch and is not contaminated, its
    transcript hashes equal to <run>/e{N}_evolver_session/-evo-work/<sid>.jsonl, its notes/ equal
    <run>/e{N}_evolver_notes/ sha-for-sha, and the state record marks epoch N collected with this snapshot."""
    rd, N = Path(ctx.run_dir), ctx.epoch
    state, prob = _state_of(rd)
    if state is None:
        return False, prob
    sid = state.get("sid") or ""
    snap = SL.snapshot_dir(rd, N)
    if not snap.is_dir():
        return False, f"no snapshot at {snap}: the resume material for epoch {N} was not written"
    probs = SL.verify_snapshot(snap)
    if probs:
        return False, f"snapshot {snap.name} does not verify: {probs[:3]}"
    man, prob = _read_json(snap / SL.SNAPSHOT_MANIFEST)
    if man is None:
        return False, f"snapshot {snap.name}: {prob}"
    if man.get("sid") != sid:
        return False, f"snapshot {snap.name} is of session {str(man.get('sid'))[:8]!r}, not the run's {sid[:8]!r}"
    if man.get("epoch") != N:
        return False, f"snapshot {snap.name} records epoch {man.get('epoch')!r}, not {N}"
    if man.get("contaminated"):
        return False, f"snapshot {snap.name} is marked contaminated: it must not be resumed from"
    ts, tr = _transcript(snap / "projects", sid), _transcript(rd / SESSION_DIR.format(epoch=N), sid)
    for p in (ts, tr):
        if p.is_symlink() or not p.is_file():
            return False, f"no transcript at {p}"
    if _sha_bytes(ts.read_bytes()) != _sha_bytes(tr.read_bytes()):
        return False, f"the snapshot's transcript differs from {tr.relative_to(rd)}: the resume material is not the session"
    a, b = _tree_shas(snap / SL.SNAPSHOT_WORKSPACE), _tree_shas(rd / SL.WORKSPACE_DIR.format(epoch=N))
    if a != b:
        diff = sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k] != b[k])
        return False, f"the snapshot's notes/ differ from {SL.WORKSPACE_DIR.format(epoch=N)}/: {diff[:4]}"
    ep = (state.get("epochs") or {}).get(str(N)) or {}
    if ep.get("status") != "collected" or ep.get("snapshot") != str(snap):
        return False, (f"the state record marks epoch {N} {ep.get('status')!r} with snapshot {ep.get('snapshot')!r}, "
                       f"not collected at {snap}")
    return True, (f"{snap} verifies ({man.get('n_files')} file(s), {man.get('bytes')} B): transcript {sid[:8]} == "
                  f"{SESSION_DIR.format(epoch=N)}'s, notes ({len(a)} file(s)) == {SL.WORKSPACE_DIR.format(epoch=N)}'s; "
                  f"state marks epoch {N} collected")


# ── DRIVE DONE: the teardown ─────────────────────────────────────────────────────────
def teardown_record_problems(record, ledger_path=None, *, state: dict | None = None) -> list:
    """The self-check of EVOLVER_SESSION_TEARDOWN.json (evolver_session.py's rule, applied here by the loop):
    the schema, the session's cid / sandbox / sid when `state` is given, then either `machine_absent` with a
    machine.json read showing the sandbox gone, or a verified + wiped `down` with a ledgered container_down."""
    from src.orchestrator import pool_container_evolver as pce
    if not isinstance(record, dict) or record.get("schema") != "evoskill.evolver_session_teardown/1":
        return ["not a teardown record (schema)"]
    probs = []
    if state is not None:
        for k in ("cid", "sandbox_id", "sid"):
            if record.get(k) != state.get(k):
                probs.append(f"the record's {k} {str(record.get(k))[:12]!r} != the session's {str(state.get(k))[:12]!r}")
    ma = record.get("machine_absent")
    if ma is not None:
        row = ma.get("machine_json") if isinstance(ma, dict) else {}
        gone = row is None or (isinstance(row, dict) and (row.get("state") in _MACHINE_GONE
                                                          or str(row.get("sandbox_id")) != str(record.get("sandbox_id"))))
        if not isinstance(ma, dict) or "machine_json" not in ma or not gone:
            probs.append("machine_absent claimed, but the record carries no machine.json read showing the sandbox gone")
        return probs
    dn = record.get("down") if isinstance(record.get("down"), dict) else {}
    if dn.get("verified") is not True or dn.get("wiped") is not True:
        probs.append(f"down not verified (verified {dn.get('verified')!r}, wiped {dn.get('wiped')!r})")
    ledger = ledger_path or os.environ.get("EVOSKILL_DAYTONA_LEDGER",
                                           str(de.REPO / ".cache" / "daytona" / "ledger.jsonl"))
    if not pce.ledger_container_down(ledger, sandbox_id=str(record.get("sandbox_id") or ""),
                                     cid=str(record.get("cid") or "")):
        probs.append("no ledgered, verified 'container_down' row for this container")
    return probs


def _machine_absent(rd: Path, state: dict) -> tuple:
    """(absent, row): whether the pool's machine.json shows the session's sandbox gone (absent from the file,
    another sandbox current, or deleted / released / failed). (None, None) when there is no readable
    machine.json: not a reading, so the flag is NOT passed and `finish` decides (it refuses, naming the flag)."""
    from src.orchestrator import pool_container_evolver as pce
    try:
        rec = json.loads(pce.pool_machines_file(rd).read_text())
    except (OSError, ValueError):
        return None, None
    if not isinstance(rec, dict) or rec.get("schema") != pce.POOL_MACHINES_SCHEMA:
        return None, None
    row = (rec.get("machines") or {}).get(rec.get("current") or "") or None
    if row is None:
        return True, None
    gone = str(row.get("sandbox_id")) != str(state.get("sandbox_id")) or row.get("state") in _MACHINE_GONE
    return gone, row


def _final_receipt_check(ctx, state: dict, record: dict) -> str:
    """The last collected epoch's receipt re-checked with the teardown record (the final containment check)."""
    import src.orchestrator.loop as L
    from src.orchestrator import pool_container_evolver as pce
    rd = Path(ctx.run_dir)
    done = [int(k) for k, v in (state.get("epochs") or {}).items() if (v or {}).get("status") == "collected"]
    if not done:
        raise L.Halt(f"the session state record marks no epoch collected ({ {k: (v or {}).get('status') for k, v in (state.get('epochs') or {}).items()} }); "
                     f"nothing shows the container ever finished a turn")
    K = max(done)
    d, prob = _read_json(rd / SL.RECEIPT_FILE.format(epoch=K))
    if d is None:
        raise L.Halt(f"the last collected epoch's receipt cannot be read: {prob}")
    _dd, c, _ses = _session_of(d)
    probe_sha, note = _probe_sha_for(state, c)
    if probe_sha is None:
        raise L.Halt(f"epoch {K}'s receipt: {note}")
    kw = L._pool_container_check_kw(ctx)
    log = rd / f"tau3_evolver_e{K}.log" if K != ctx.epoch else L._evolver_log_path(ctx)
    ok, det = pce.check_pool_container_receipt(d, log=log, staging=None, session=SL.STATEFUL,
                                               rw_paths=[r["path"] for r in SL.STATEFUL_RW], probe_pre_sha256=probe_sha,
                                               teardown=record, **kw)
    if not ok:
        raise L.Halt(f"the final containment check (epoch {K}'s receipt with the teardown record) fails: {det}")
    return f"epoch {K}'s receipt re-checked with the teardown record: {det}"


def run_session_finish(ctx) -> None:
    """DRIVE DONE (sections 2.4 / 3.7): tear the run's one evolver container down through
    `scripts/daytona/evolver_session.py finish --run-dir <run>` (the loop's env minus tokens; `--machine-absent-ok`
    only when the pool's machine.json shows the session's sandbox absent / deleted), wait FINISH_TIMEOUT_S, require
    rc 0, then re-verify the record (down verified + wiped, a ledgered container_down; or machine_absent with the
    machine.json read) and the last collected epoch's receipt against it. Raises loop.Halt; the container survives
    a HALT (a re-run point). A session already torn down with a verifying record is a no-op."""
    import subprocess
    import src.orchestrator.loop as L
    rd = Path(ctx.run_dir).resolve()
    if L._evolver_session(ctx) != SL.STATEFUL:
        return
    state, prob = _state_of(rd)
    if state is None:
        raise L.Halt(f"DRIVE DONE, but {prob}: nothing shows the run's evolver container was ever created, and nothing "
                     f"can tear it down. If a session ran, find its container with `scripts/daytona/evolver_container.py "
                     f"cleanup` by hand.")
    rec_p = rd / SL.TEARDOWN_RECORD
    if state.get("torn_down"):
        record, prob = _read_json(rec_p)
        if record is None:
            raise L.Halt(f"the state record says the session is torn down, but {prob}")
        probs = teardown_record_problems(record, state=state)
        if probs:
            raise L.Halt(f"the session is recorded torn down, but {rec_p.name} does not verify: {probs}")
        print(f"  evolver session {state.get('sid', '')[:8]} already torn down ({record.get('utc')}); "
              f"{_final_receipt_check(ctx, state, record)}")
        return
    absent, row = _machine_absent(rd, state)
    from src.orchestrator.host_tools import python_executable
    argv = [python_executable(), str(de.REPO / FINISH_SCRIPT), "finish",
            "--run-dir", str(rd)]
    if absent:
        argv.append("--machine-absent-ok")
    env = dict(os.environ)
    for k in (*_TOKEN_ENV, "EVOSKILL_TAU3_GATE_ONLY"):
        env.pop(k, None)
    out = rd / FINISH_OUT
    print(f"  evolver session {state.get('sid', '')[:8]}: tearing down container {state.get('cid')} on "
          f"{str(state.get('sandbox_id'))[:8]}" + (" (machine absent from the pool)" if absent else "") + f" -> {out.name}")
    with open(out, "ab") as fh:
        fh.write(f"\n=== finish {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} argv {argv}\n".encode())
        fh.flush()
        p = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             cwd=str(de.REPO))
        try:
            rc = p.wait(timeout=FINISH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise L.Halt(f"evolver_session.py finish still running after {FINISH_TIMEOUT_S}s (pid {p.pid}). It has NOT "
                         f"been killed — inspect {out.name}; the container stays until a finish completes.")
    if rc != 0:
        tail = out.read_text(errors="replace")[-600:]
        raise L.Halt(f"evolver_session.py finish exited {rc}"
                     + (" (a dispatch or another session command is live; wait for it)" if rc == 4 else "")
                     + f"; the container {state.get('cid')} is NOT torn down. Re-run --drive, or "
                     f"`{FINISH_SCRIPT} finish --run-dir {rd}` by hand"
                     + (" with --machine-absent-ok if machine.json shows the machine gone" if absent is None else "")
                     + f":\n  {tail}")
    record, prob = _read_json(rec_p)
    if record is None:
        raise L.Halt(f"finish exited 0 but {prob}: the teardown is not recorded")
    state2, prob = _state_of(rd)
    if state2 is None or not state2.get("torn_down"):
        raise L.Halt(f"finish exited 0 but the state record is not marked torn down ({prob or 'torn_down null'})")
    probs = teardown_record_problems(record, state=state2)
    if probs:
        raise L.Halt(f"{rec_p.name} does not verify: {probs}; the container may still be up")
    print(f"  evolver session torn down: {'machine absent, no down' if record.get('machine_absent') else 'down verified + wiped, ledgered'}; "
          f"{_final_receipt_check(ctx, state2, record)}")
