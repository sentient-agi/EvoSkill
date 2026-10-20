#!/usr/bin/env python3
"""The OPERATOR <-> AO conversation, as one durable artifact.

WHY THIS EXISTS. A conversation scattered across channels that never meet is not a record of a dialogue:

  * the message `prompt_author` composes and hands to the AO         (argv, then a file)
  * the AO's final reply                                             (its stream-json log)
  * questions the AO asks through `loop.ask()` and my answers        (BLOCKED_*/ANSWERS_* json)
  * the loop's own reports of each step                              (stdout)

Four formats, four locations, no ordering between them. This module is that record's home: ONE append-only JSONL
per run, written by every channel, in speaking order, readable top to bottom (`--markdown`).

WHERE IT LIVES, AND WHY THAT MATTERS. `<run>.operator/CONVERSATION.jsonl` -- the operator
directory, a SIBLING of the run dir. Same reasoning as ANSWERS/BLOCKED (#268): no agent job ever
declares the operator dir as a write target, so a future sandbox-grant mistake cannot let an
optimizer edit the record of what it was told. An agent that can rewrite the transcript of its own
instructions is an agent whose instructions cannot be audited.

FORMAT. One JSON object per line, append-only, never rewritten:

    {"ts": "2026-08-12T01:02:03", "ts_ns": 176..., "speaker": "operator"|"ao",
     "channel": "intake"|"clarify"|"authoring"|"ask"|"answer"|"invocation"|"report"|"note",
     "epoch": 1|null, "content": "...verbatim...", "meta": {...}}

Append-only JSONL rather than a rewritten JSON document for the #244 reason: a record written
only on clean exit is not a record. A crashed step leaves its turns behind.

RECORDING NEVER RAISES. A full disk must not kill an epoch. But losing the record silently is
exactly the failure this module was written to fix, so `record()` returns a note on failure, the
note is printed, and `verify_covers()` -- wired in as a loop step -- turns a silent write failure
into a declared HALT at the next step rather than an absence nobody notices.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SPEAKERS = ("operator", "ao")
CHANNELS = ("intake", "clarify", "authoring", "ask", "answer", "invocation", "report", "note")


def operator_dir(run_dir) -> Path:
    """Sibling `<run>.operator/`. Duplicated from loop.py DELIBERATELY, and it is three lines.

    Importing loop.py here would be a cycle: loop imports this module to record turns. The
    alternative -- a third module holding one path join -- buys nothing. `test_conversation.py`
    asserts the two agree, so a drift is a test failure rather than a silent second location.
    """
    rd = Path(run_dir)
    return rd.parent / f"{rd.name}.operator"


def path_for(run_dir) -> Path:
    return operator_dir(run_dir) / "CONVERSATION.jsonl"


def record(run_dir, *, speaker: str, channel: str, content: str,
           epoch: int | None = None, meta: dict | None = None) -> str:
    """Append one turn. Returns "" on success, or a human-readable note on failure.

    The speaker/channel vocabularies are CLOSED and violations raise. That is not the same
    contract as the write failure below: a bad speaker is a coding error at the call site and
    should be loud, while a failed write is an environment problem and must not kill a run.
    """
    if speaker not in SPEAKERS:
        raise ValueError(f"speaker must be one of {SPEAKERS}, got {speaker!r}")
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    if content is None:
        raise ValueError("content is None — a turn with no content is not a turn")
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ts_ns": time.time_ns(),
           "speaker": speaker, "channel": channel, "epoch": epoch,
           "content": str(content), "meta": meta or {}}
    p = path_for(run_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append with an explicit flush: two processes (the loop and a leg script)
        # can hold this file open at once, and a short append under O_APPEND is atomic on the
        # platforms this runs on. A lock would serialise the loop against its own children for no
        # gain — turns are independent and ordering is carried by ts_ns, not by file position.
        with p.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        return ""
    except OSError as e:
        note = (f"\n  [conversation] could NOT record this turn to {p} "
                f"({type(e).__name__}: {e}). The run continues; the transcript is now incomplete "
                f"and `conversation covers this epoch` will HALT until it is fixed.")
        print(note, file=sys.stderr)
        return note


def record_once(run_dir, *, key: str, speaker: str, channel: str, content: str,
                epoch: int | None = None, meta: dict | None = None) -> str:
    """`record`, but only if no turn with this `key` is already in the transcript.

    WHY IT IS NEEDED. A verify is re-run many times per invocation — evaluate() calls every step,
    advance() calls the first failing one twice, and `_resolve_question` calls it once per round.
    A turn appended on every call would give a transcript in which one question appears eleven
    times, which is worse than a missing turn: it makes the dialogue unreadable and it inflates
    every count computed from it.

    DEDUPED AGAINST THE FILE, NOT AGAINST IN-PROCESS STATE. Each `--step` is a separate process,
    so an in-memory set would forget across invocations — which is exactly the boundary the
    duplicates cross. The key lives in `meta.key`, so the check is a scan of what was actually
    written rather than of a parallel bookkeeping file that could disagree with it.
    """
    for t in read(run_dir):
        if (t.get("meta") or {}).get("key") == key:
            return ""
    return record(run_dir, speaker=speaker, channel=channel, content=content, epoch=epoch,
                  meta={**(meta or {}), "key": key})


def read(run_dir) -> list[dict]:
    """Every turn, in speaking order. A malformed line is kept as a turn, not dropped.

    Dropping an unparseable line would make a corrupted transcript look merely SHORT, and short is
    indistinguishable from "nothing was said". A visible `[unparseable]` turn is the honest form.
    """
    p = path_for(run_dir)
    if not p.is_file():
        return []
    out = []
    for i, line in enumerate(p.read_text(errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"line is a {type(rec).__name__}, not an object")
        except (json.JSONDecodeError, ValueError) as e:
            rec = {"ts": "", "ts_ns": 0, "speaker": "operator", "channel": "note", "epoch": None,
                   "content": f"[unparseable line {i + 1}: {e}] {line[:400]}",
                   "meta": {"unparseable": True}}
        out.append(rec)
    out.sort(key=lambda r: (r.get("ts_ns") or 0))
    return out


def verify_covers(run_dir, epoch: int) -> tuple[bool, str]:
    """(ok, detail) — is there a real two-sided exchange recorded for this epoch?

    WHAT THIS DOES NOT CHECK, deliberately: whether the content is *good*. It checks that both
    parties spoke, because the #301 failure was one-sided-ness, and a one-sided transcript passes
    every check that only counts turns. Anything stronger would be judging content, which is the
    operator's job.
    """
    turns = read(run_dir)
    if not turns:
        return False, (f"no conversation recorded at {path_for(run_dir)} — every operator<->AO "
                       f"turn is supposed to append here. An absent transcript means either "
                       f"nothing was said (impossible: this step is running) or recording failed.")
    bad =[t for t in turns if t.get("meta", {}).get("unparseable")]
    mine = [t for t in turns if t.get("epoch") in (epoch, None)]
    ops = [t for t in mine if t["speaker"] == "operator"]
    aos = [t for t in mine if t["speaker"] == "ao"]
    if bad:
        return False, (f"{len(bad)} unparseable line(s) in {path_for(run_dir).name} — the "
                       f"transcript is corrupt, first: {bad[0]['content'][:160]}")
    if not ops:
        return False, (f"{len(turns)} turn(s) recorded but NONE from the operator for epoch "
                       f"{epoch}. A transcript with only the AO's side is the #301 defect exactly.")
    if not aos:
        return False, (f"{len(turns)} turn(s) recorded but NONE from the AO for epoch {epoch}.")
    chans = sorted({t["channel"] for t in mine})
    return True, (f"{len(mine)} turn(s) for epoch {epoch} — {len(ops)} operator, {len(aos)} AO; "
                  f"channels {chans}; {path_for(run_dir).name}")


# ── exchanges ────────────────────────────────────────────────────────────────────────────────

def exchanges(turns: list[dict]) -> list[dict]:
    """Group turns into (operator says X -> AO says Y) exchanges.

    WHY PAIR ON THE OPERATOR TURN. Every AO utterance in this system is a response to something
    the operator did — an invocation, an answer, a composed message. Even the AO ASKING a question
    is its response to being invoked. So an exchange opens on an operator turn and absorbs every
    AO turn until the next one, so the transcript reads as a dialogue rather than as a pile of
    one-sided turns.

    An AO turn with no operator turn before it (the transcript starting mid-run, or a replay of an
    older run) opens exchange 0 with an explicit placeholder rather than being silently attached
    to a later instruction it did not answer.
    """
    out: list[dict] = []
    cur: dict | None = None
    for t in turns:
        if t["speaker"] == "operator":
            if cur is not None and cur["ao"]:
                out.append(cur)
                cur = None
            if cur is None:
                cur = {"ops": [], "ao": [], "t0": t["ts_ns"], "t1": t["ts_ns"],
                       "epoch": t.get("epoch")}
            cur["ops"].append(t)
            cur["t1"] = max(cur["t1"], t["ts_ns"])
            if cur.get("epoch") is None:
                cur["epoch"] = t.get("epoch")
        else:
            if cur is None:
                cur = {"ops": [], "ao": [], "t0": t["ts_ns"], "t1": t["ts_ns"],
                       "epoch": t.get("epoch")}
            cur["ao"].append(t)
            cur["t1"] = max(cur["t1"], t["ts_ns"])
            if cur.get("epoch") is None:
                cur["epoch"] = t.get("epoch")
    if cur is not None:
        out.append(cur)
    return out


def _join(turns: list[dict]) -> str:
    """Verbatim, with a channel header per turn so a reader can see WHICH channel each came from.

    Never truncated here.
    """
    return "\n\n".join(f"[{t['channel']}]\n{t['content']}" for t in turns)


def render_markdown(run_dir) -> str:
    """The transcript as readable markdown — for a handoff note, a PR, or a human skim."""
    turns = read(run_dir)
    lines = [f"# Operator <-> AO conversation — {Path(run_dir).name}", "",
             f"{len(turns)} turn(s), {len(exchanges(turns))} exchange(s).", ""]
    for i, x in enumerate(exchanges(turns)):
        ep = x.get("epoch")
        lines.append(f"## Exchange {i + 1}" + (f" (epoch {ep})" if ep is not None else ""))
        for t in x["ops"] + x["ao"]:
            who = "OPERATOR" if t["speaker"] == "operator" else "AO"
            lines += [f"**{who}** · `{t['channel']}` · {t['ts']}", "", t["content"], ""]
    return "\n".join(lines)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--markdown", action="store_true", help="print the transcript and exit")
    ap.add_argument("--say", nargs=3, metavar=("SPEAKER", "CHANNEL", "CONTENT"),
                    help="append one turn (CONTENT may be @path to read a file)")
    ap.add_argument("--epoch", type=int)
    a = ap.parse_args()
    if a.say:
        speaker, channel, content = a.say
        if content.startswith("@"):
            content = Path(content[1:]).read_text()
        note = record(a.run_dir, speaker=speaker, channel=channel, content=content, epoch=a.epoch)
        print(note or f"recorded {speaker}/{channel} ({len(content)} chars) -> "
                      f"{path_for(a.run_dir)}")
        return 1 if note else 0
    if a.markdown:
        print(render_markdown(a.run_dir))
        return 0
    turns = read(a.run_dir)
    print(f"{len(turns)} turn(s), {len(exchanges(turns))} exchange(s) at "
          f"{path_for(a.run_dir)}")
    for t in turns:
        print(f"  {t['ts']}  {t['speaker']:8s} {t['channel']:11s} "
              f"{t['content'][:90].splitlines()[0] if t['content'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
