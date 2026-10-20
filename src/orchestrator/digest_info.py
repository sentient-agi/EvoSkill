"""DIGEST_INFO.json: what an SSB v2 digest dir shows in full and what it shortens .

WHY THIS EXISTS. The loop-rendered evolver prompt (evolver_template.py) describes what each digest
holds: the grader's output, the cells the solver changed, the transcript. A digest builder may cut a
scorer diff (2500+800 chars), list only the first 60 changed cells and cut every long tool output
to 1200+600, and a prompt that says "in full" over such a dir would be false.

So the builder writes, beside the task files, one machine-readable header per digest dir, and
the renderer states only what that header says, after checking the header against the bytes:

  grader_output   full (default) or clipped; how many scorer-diff sections, how many clipped.
  changed_cells   on or off (default off: its lines carry training input values); cap, sections, capped lists.
  tool_output     the clip limits (kept: 1200+600) and how many tool results / tool calls were
                  shortened. Every cut carries the CLIP marker below, never a silent slice.
  files           per task file: sha256 and the same counts, so an edit after the build, a file
                  from another build, or a header whose counts the bytes contradict is refused.

WHY A MODULE WITH NO PACKAGE IMPORTS. The builder imports nothing from src/ (it loads
skill_library.py by path); it loads this file by path the same way, verifies the header it is about
to write with the SAME reader the renderer uses, and refuses to promote a surface its own header
contradicts. The renderer imports it normally.

The header names train task files and counts only: no cache, snapshot or source path, because the
evolver can read the digest dir.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

INFO_NAME = "DIGEST_INFO.json"
SCHEMA = "evoskill.digest_info/1"
SIDECARS = frozenset({"REGRESSIONS", "README", "INDEX"})   # digest_coverage.SIDECARS

# The markers the builder writes at every cut, each on its own line. Parsed here; a test pins the
# builder's text to these patterns.
CLIP_PREFIX = "[DIGEST CLIP — "
CAP_PREFIX = "[DIGEST CAP — "
CLIP_RE = re.compile(r"(?m)^\[DIGEST CLIP — ([\d,]+) of ([\d,]+) chars not shown")
CAP_RE = re.compile(r"(?m)^\[DIGEST CAP — ([\d,]+) more changed cell")
# The older builder's marker ("    ...[N chars elided]...") and its changed-cells cap text. A dir
# carrying either was built before this header existed and must be rebuilt, not described.
LEGACY_CLIP_RE = re.compile(r"\.\.\.\[\d+ chars elided\]\.\.\.")
LEGACY_CAP_RE = re.compile(r"\*\*\d+ cell\(s\) changed by the solver\. First \d+ shown\.\*\*")

ROLLOUT_RE = re.compile(r"(?m)^## Rollout s\d+ — (?:PASS|FAIL) \(")
SCORER_RE = re.compile(r"(?m)^## Scorer diff — \S+ \(authoritative\)$")
CHANGED_RE = re.compile(r"(?m)^## What the SOLVER changed — ")

FILE_KEYS = ("grader_sections", "grader_clipped", "changed_sections", "changed_capped",
             "tool_results_clipped", "tool_calls_clipped")


class DigestInfoError(ValueError):
    """The dir's header is missing, malformed, or contradicted by the task files' bytes."""


def task_files(d) -> list:
    """Top-level per-task *.md files (sidecars excluded), the set evolver_template names."""
    return sorted(p for p in Path(d).glob("*.md") if p.stem not in SIDECARS)


def scan_text(text: str) -> dict:
    """Counts read from a task file's bytes alone, independent of any header.

    The region before the first rollout header holds the instruction, the scorer diffs and the
    changed-cells sections; clip markers there are grader-output clips. Markers after it are tool
    clips (results and calls together: the bytes cannot tell them apart, the header can)."""
    m = ROLLOUT_RE.search(text)
    pre, post = (text[:m.start()], text[m.start():]) if m else (text, "")
    return {
        "grader_sections": len(SCORER_RE.findall(pre)),
        "grader_clipped": len(CLIP_RE.findall(pre)),
        "changed_sections": len(CHANGED_RE.findall(pre)),
        "changed_capped": len(CAP_RE.findall(pre)),
        "tool_clipped": len(CLIP_RE.findall(post)),
        "legacy_markers": len(LEGACY_CLIP_RE.findall(text)) + len(LEGACY_CAP_RE.findall(text)),
    }


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def problems(d, info: dict | None = None) -> list:
    """Why `info` (default: the dir's own DIGEST_INFO.json) does not truthfully describe dir `d`.
    [] means every count it states is what the bytes show."""
    d = Path(d)
    if info is None:
        p = d / INFO_NAME
        if not p.is_file():
            return [f"no {INFO_NAME}: the dir was built by an older digest builder (or by "
                    f"hand), so what it shows in full cannot be stated. Move it aside and rebuild "
                    f"it with a digest builder that writes {INFO_NAME}"]
        try:
            info = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            return [f"{p} is not readable JSON: {e}"]
    if not isinstance(info, dict) or info.get("schema") != SCHEMA:
        return [f"{d / INFO_NAME}: schema is {info.get('schema') if isinstance(info, dict) else info!r}, "
                f"not {SCHEMA}"]
    probs = []
    go, cc, to = info.get("grader_output"), info.get("changed_cells"), info.get("tool_output")
    files = info.get("files")
    if not all(isinstance(x, dict) for x in (go, cc, to, files)):
        return [f"{d / INFO_NAME}: grader_output, changed_cells, tool_output and files must all be "
                f"objects"]
    if not isinstance(go.get("full"), bool) or not isinstance(cc.get("on"), bool):
        probs.append("grader_output.full and changed_cells.on must be true or false")
    for k in ("clip_head", "clip_tail"):
        if not isinstance(to.get(k), int) or to[k] <= 0:
            probs.append(f"tool_output.{k} must be a positive int, got {to.get(k)!r}")
    if go.get("full") is True and (go.get("clip_head") is not None or go.get("clip_tail") is not None):
        probs.append("grader_output.full is true but clip limits are recorded for it")
    if go.get("full") is False:
        for k in ("clip_head", "clip_tail"):
            if not isinstance(go.get(k), int) or go[k] <= 0:
                probs.append(f"grader_output is clipped but grader_output.{k} is {go.get(k)!r}")
    if cc.get("on") is True and (not isinstance(cc.get("max_cells"), int) or cc["max_cells"] < 0):
        probs.append(f"changed_cells is on but max_cells is {cc.get('max_cells')!r} (0 = no cap)")

    on_disk = {p.name: p for p in task_files(d)}
    if set(on_disk) != set(files):
        extra, gone = sorted(set(on_disk) - set(files)), sorted(set(files) - set(on_disk))
        probs.append(f"task files differ from the header: on disk only {extra[:4]}, header only "
                     f"{gone[:4]}")
    if info.get("task_files") != len(files):
        probs.append(f"task_files is {info.get('task_files')!r} but the header lists {len(files)}")
    tot = {k: 0 for k in FILE_KEYS}
    for name in sorted(set(on_disk) & set(files)):
        rec = files[name]
        if not isinstance(rec, dict) or any(not isinstance(rec.get(k), int) for k in FILE_KEYS):
            probs.append(f"{name}: header entry lacks integer counts {list(FILE_KEYS)}")
            continue
        raw = on_disk[name].read_bytes()
        if sha256_bytes(raw) != rec.get("sha256"):
            probs.append(f"{name}: bytes changed since the header was written (sha256 differs)")
        s = scan_text(raw.decode("utf-8", errors="replace"))
        if s["legacy_markers"]:
            probs.append(f"{name}: carries {s['legacy_markers']} marker(s) of the older builder")
        for k in ("grader_sections", "grader_clipped", "changed_sections", "changed_capped"):
            if s[k] != rec[k]:
                probs.append(f"{name}: header says {k}={rec[k]}, the bytes show {s[k]}")
        if s["tool_clipped"] != rec["tool_results_clipped"] + rec["tool_calls_clipped"]:
            probs.append(f"{name}: header says {rec['tool_results_clipped']} tool result(s) + "
                         f"{rec['tool_calls_clipped']} tool call(s) clipped, the bytes show "
                         f"{s['tool_clipped']} clip marker(s)")
        for k in FILE_KEYS:
            tot[k] += rec[k]
    want = {("grader_output", "sections"): tot["grader_sections"],
            ("grader_output", "clipped"): tot["grader_clipped"],
            ("changed_cells", "sections"): tot["changed_sections"],
            ("changed_cells", "lists_capped"): tot["changed_capped"]}
    for (sec, k), v in want.items():
        if info[sec].get(k) != v:
            probs.append(f"{sec}.{k} is {info[sec].get(k)!r} but the files sum to {v}")
    for sec, k in (("results", "tool_results_clipped"), ("calls", "tool_calls_clipped")):
        got = (to.get(sec) or {}).get("clipped")
        if got != tot[k]:
            probs.append(f"tool_output.{sec}.clipped is {got!r} but the files sum to {tot[k]}")
    if go.get("full") is True and tot["grader_clipped"]:
        probs.append(f"grader_output.full is true but {tot['grader_clipped']} scorer diff(s) are clipped")
    if cc.get("on") is False and tot["changed_sections"]:
        probs.append(f"changed_cells is off but {tot['changed_sections']} changed-cells section(s) "
                     f"are present")
    if cc.get("on") is True and cc.get("max_cells") == 0 and tot["changed_capped"]:
        probs.append("changed_cells has no cap (max_cells 0) but capped lists are present")
    n_clip = (tot["grader_clipped"] + tot["changed_capped"] + tot["tool_results_clipped"]
              + tot["tool_calls_clipped"])
    if info.get("clipped_items_total") != n_clip:
        probs.append(f"clipped_items_total is {info.get('clipped_items_total')!r}, the files sum to "
                     f"{n_clip}")
    return probs


def read(d) -> dict:
    """The dir's DIGEST_INFO.json, after problems() found nothing. Raises DigestInfoError."""
    probs = problems(d)
    if probs:
        raise DigestInfoError(f"{Path(d)}: " + "; ".join(probs[:6])
                              + (f" (+{len(probs) - 6} more)" if len(probs) > 6 else ""))
    return json.loads((Path(d) / INFO_NAME).read_text())
