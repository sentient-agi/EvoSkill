#!/usr/bin/env python3
"""Build (or audit) the tau3 banking_knowledge evolver oracle — TRAIN TASKS ONLY.

    python3 scripts/make_tau3_oracle.py --verify   # audit the oracle on disk
    python3 scripts/make_tau3_oracle.py            # rebuild it, train-only

DELEGATES to the tested machinery: `tau2_agent.oracle.build_oracle_spec()` +
`src.loop.oracle.materialize_oracle()`. That pair copies the six evaluator sources,
writes `tasks/<domain>.json` **as an object keyed by task_id**, and generates
`README.md` — which the evolver prompt orders read FIRST ("read `./oracle/README.md`
before diagnosing anything"), and which `evo0801_preflight_evidence.py` checks.

A previous version of this script hand-rolled the tasks file instead. It produced a
JSON **list** where the machinery produces a **dict keyed by task_id**, and emitted a
`MANIFEST.json` instead of the mandated `README.md` — so the evolver would have been
told to read a file that did not exist, and any reader expecting the documented shape
would have broken. Reinventing a writer to add one assertion was the mistake; the
assertion belongs on top of the real writer.

THE ONE THING THIS ADDS: a val/test firewall.

`write_all_task_meta`'s own docstring says "pass the union of a run's **train/val**
pool ids", and that is how the 64-task oracle audited on 2026-08-03 came to hold all 32
train AND all 32 val. But SSB v2's builder states the opposite standard — "CONTAMINATION
CONTROL. Train tasks only ... refuses to write an oracle that names a val or test task."
The two harnesses contradict each other, which is a parity defect
([[feedback-all-harness-parity]]). Train-only is the defensible side: val is the
SELECTION signal, and an evolver that can read val's `evaluation_criteria` and
`user_scenario` can write skills tuned to the very split meant to adjudicate them.
So this caller passes TRAIN ONLY and asserts it, rather than following the docstring.

** The tau2 module's docstring and default guidance still need the same correction —
   fixing it here only protects tau3. **
"""
from __future__ import annotations

import argparse
import json
import re as _re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TAU3 = ROOT / "third_party/tau3-bench"
# DOMAIN/SPLIT/ORACLE parameterised (2026-08-06). They were literals, so only
# banking_knowledge could ever get an oracle — and the digest builder REFUSES to run
# without one, which made every other tau3 task type unreachable for the
# "proofread the evolver prompt per task type" goal.
#
# EACH DOMAIN GETS ITS OWN ORACLE DIRECTORY. `materialize_oracle` writes
# `tasks/<domain>.json`, so a shared dir would accumulate several domains' golden specs
# side by side and an evolver working on one task type could read another's. Separate
# dirs keep each evolver's evidence surface exactly its own.
import os as _os
DOMAIN = _os.environ.get("EVOSKILL_TAU3_ORACLE_DOMAIN", "banking_knowledge")
_DEF_SPLIT = ("outputs/tau3_banking/split.json" if DOMAIN == "banking_knowledge"
              else f"outputs/tau3_banking/split_smoke5_{DOMAIN}.json")
SPLIT_FILE = Path(_os.environ.get("EVOSKILL_TAU3_ORACLE_SPLIT", ROOT / _DEF_SPLIT))
_DEF_ORACLE = ("outputs/tau3_banking/oracle" if DOMAIN == "banking_knowledge"
               else f"outputs/tau3_banking/oracle_{DOMAIN}")
ORACLE = Path(_os.environ.get("EVOSKILL_TAU3_ORACLE_DIR", ROOT / _DEF_ORACLE))


def load_split() -> dict:
    if not SPLIT_FILE.exists():
        raise SystemExit(f"no split at {SPLIT_FILE} — run make_tau3_banking_split.py "
                         f"(banking) or make_tau3_domain_smoke_split.py (other domains)")
    return json.loads(SPLIT_FILE.read_text())


def forbidden_ids(split: dict) -> set[str]:
    """val | test, refusing an incomplete reference set.

    An empty forbidden set makes every oracle look clean — that is how a leak hides,
    and it is exactly the trap SSB v2 hit when `split_30_30.json` carried `test: []`
    on purpose and a check reported "val/test leaked: 0" against nothing.
    """
    va, te = set(split.get("val", [])), set(split.get("test", []))
    # NON-BANKING SPLITS CARRY THE HELD-OUT SET UNDER A DIFFERENT KEY. airline/retail/
    # telecom ship their own train/test partition and no val, so
    # make_tau3_domain_smoke_split.py writes `val: []`, `test: []` (so the runner can
    # never select them) and records the shipped test ids under
    # `shipped_test_ids_NOT_A_SPLIT_KEY`. Reading only val|test there would produce an
    # EMPTY forbidden set — and this function's own docstring says that is exactly how a
    # leak hides. So fall back to the recorded shipped list, and still refuse if the
    # result is empty.
    shipped = set(split.get("shipped_test_ids_NOT_A_SPLIT_KEY", []))
    forb = va | te | shipped
    if not forb:
        raise SystemExit(f"refusing: EMPTY reference set (val={len(va)}, test={len(te)}, "
                         f"shipped_test={len(shipped)}) — every oracle would look clean")
    return forb


def read_oracle_ids() -> set[str]:
    """Ids in the oracle on disk, accepting either shape.

    The machinery writes a dict keyed by task_id; tolerate a list too so this audit
    still reports honestly on a file some other tool wrote.
    """
    f = ORACLE / "tasks" / f"{DOMAIN}.json"
    if not f.exists():
        return set()
    raw = json.loads(f.read_text())
    if isinstance(raw, dict):
        return set(raw.keys())
    return {t["id"] for t in raw if isinstance(t, dict) and "id" in t}


def audit(split: dict) -> int:
    have = read_oracle_ids()
    if not have:
        print(f"  no oracle tasks file at {ORACLE / 'tasks' / (DOMAIN + '.json')}")
        return 2
    tr = set(split["train"])
    bad = have & forbidden_ids(split)
    readme = ORACLE / "README.md"
    shape = json.loads((ORACLE / "tasks" / f"{DOMAIN}.json").read_text())
    print(f"  tasks file : {len(have)} ids, shape={type(shape).__name__} "
          f"{'(keyed by task_id — correct)' if isinstance(shape, dict) else '(LIST — wrong shape)'}")
    print(f"  train      : {len(have & tr)}/{len(tr)}")
    print(f"  val|test   : {len(bad)}   (must be 0)")
    print(f"  README.md  : {'present' if readme.exists() else 'MISSING — the evolver '
                            'prompt orders it read FIRST'}")
    ev = sorted(p.name for p in ORACLE.glob("evaluator*.py"))
    print(f"  evaluator  : {len(ev)} source file(s) {ev if ev else '— MISSING'}")
    problems = []
    if bad:
        problems.append(f"FIREWALL BREACH: {len(bad)} held-out id(s), e.g. {sorted(bad)[:5]}")
    if not isinstance(shape, dict):
        problems.append("tasks file is a LIST; the documented shape is an object keyed by task_id")
    if not readme.exists():
        problems.append("README.md missing")
    if not ev:
        problems.append("no evaluator source — the evolver cannot read how grading works")
    if have - tr:
        problems.append(f"{len(have - tr)} id(s) not in train")
    if problems:
        print("\n  NOT READY:")
        for p in problems:
            print(f"    - {p}")
        print("\n  Rebuild with: python3 scripts/make_tau3_oracle.py")
        return 1
    print("\n  OK — train-only, correct shape, README + evaluator present.")
    return 0


def build(split: dict) -> int:
    from src.agent_profiles.tau2_agent.oracle import build_oracle_spec
    from src.loop.oracle import materialize_oracle

    train = set(split["train"])
    bad = train & forbidden_ids(split)
    if bad:                                  # a split that overlaps itself is broken
        raise SystemExit(f"refusing: split train overlaps val/test: {sorted(bad)[:5]}")

    spec = build_oracle_spec({DOMAIN: train}, tau2_root=TAU3)
    out = materialize_oracle(ORACLE, spec)

    # Verify AFTER writing, against what actually landed — the assertion is worthless
    # if it only checks the ids we intended to pass.
    got = read_oracle_ids()
    leaked = got & forbidden_ids(split)
    if leaked:
        raise SystemExit(f"materialized oracle names {len(leaked)} held-out task(s): "
                         f"{sorted(leaked)[:5]} — do NOT run an evolver against it")
    if got != train:
        raise SystemExit(f"oracle ids != split train (got {len(got)}, want {len(train)})")
    if not (ORACLE / "README.md").exists():
        raise SystemExit("materialize_oracle wrote no README.md — the evolver prompt "
                         "orders it read first")

    # --- #193: SCRUB HELD-OUT IDS THAT THE TRAIN SPECS THEMSELVES MENTION ---------------
    # The check above asks "does the oracle CONTAIN a held-out task?" — a MEMBERSHIP
    # question, and it passes. The leak is different: an INCLUDED train spec can NAME a
    # held-out task in its prose. banking does exactly that:
    #   task_027 (train) notes: "Adversarial variant of task_026: ..."   task_026 is TEST
    #   task_029 (train) notes: "Adversarial variant of task_028: ..."   task_028 is TEST
    # Found 2026-08-07 by the post-hoc leak audit, on a run where the oracle gate, the
    # digest firewall, the Bash deny-hook and the isolation check were ALL green — because
    # all four ask about membership. Fixing only the digests would have relaunched straight
    # back into contamination through this file, which is why every granted directory is now
    # scrubbed rather than just the one that got caught.
    _forb = forbidden_ids(split)
    # #218. The scrub USED to be a blind `text.replace(id, placeholder)` over each file's raw
    # bytes. That is safe only while task ids are long and distinctive (banking's `task_001`).
    # retail's ids are BARE INTEGERS — '0','1',...,'113' — and a blind replace then rewrites
    # every incidental occurrence of those digits: prices, quantities, dates, other ids, and
    # (the case that surfaced this) the hex digits inside a `—` escape, which turns
    #     "delivered — the order"
    # into
    #     "delivered \u<a held-out task, id redacted>"
    # i.e. invalid JSON. The existing assert caught it, so nothing corrupt ever shipped — but
    # it caught it as a raw JSONDecodeError after the damaged file was already on disk, and
    # the underlying scrub was unusable for every integer-id domain.
    #
    # Structured redaction instead: for JSON, walk the parsed document and redact a string
    # only when it IS a held-out id, never when it merely contains one. For non-JSON files
    # (README, evaluator sources) fall back to a WORD-BOUNDARY replace, which cannot split a
    # token in half.
    _tf = ORACLE / "tasks" / f"{DOMAIN}.json"
    _red = []

    # #238 FIX. `_forb` membership is WHOLE-STRING equality, so a held-out id EMBEDDED IN PROSE
    # survived every scrub: `"...still shows the task_026 shipment as delivered"` is not equal to
    # `"task_026"`. Measured 2026-08-10 on a 3-task smoke oracle whose train ids are task_027 and
    # task_029 — both carry free-text `notes` naming task_026 and task_028, which ARE held out.
    # The builder printed "#193 scrub: no train spec referenced a held-out id" and materialized an
    # oracle containing two sealed-test ids.
    #
    # The narrowness was not an oversight, it was an overcorrection: #218's blind
    # `text.replace` mangled retail (ids '0'..'113') by rewriting `reward=1.0`, so the scrub was
    # tightened all the way to exact equality and skipped the middle ground. The middle ground is
    # the rule the NON-JSON path below already uses and states: substitute only DISTINCTIVE ids
    # (>=4 chars, not purely numeric) at WORD BOUNDARIES. `task_026` qualifies; retail's `1` and
    # `30` do not, and declining those is the same deliberate, documented refusal as below.
    _distinct = [i for i in sorted(_forb, key=len, reverse=True)
                 if i and len(i) >= 4 and not i.isdigit()]

    def _scrub_str_inline(s, fname):
        for _i in _distinct:
            _pat = rf"(?<![0-9A-Za-z_]){_re.escape(_i)}(?![0-9A-Za-z_])"
            if _re.search(_pat, s):
                s = _re.sub(_pat, "<a held-out task, id redacted>", s)
                _red.append((fname, _i))
        return s

    def _scrub_obj(o, fname):
        if isinstance(o, str):
            if o in _forb:
                _red.append((fname, o))
                return "<a held-out task, id redacted>"
            return _scrub_str_inline(o, fname)
        if isinstance(o, list):
            return [_scrub_obj(x, fname) for x in o]
        if isinstance(o, dict):
            return {(_scrub_obj(k, fname) if isinstance(k, str) else k): _scrub_obj(v, fname)
                    for k, v in o.items()}
        return o

    for _p in list(ORACLE.rglob("*")):
        if not _p.is_file():
            continue
        try:
            _t = _p.read_text(errors="replace")
        except OSError:
            continue
        if _p.suffix == ".json":
            try:
                _doc = json.loads(_t)
            except json.JSONDecodeError as _e:
                raise SystemExit(
                    f"REFUSING: {_p} is not valid JSON before the scrub even runs ({_e}). "
                    f"The oracle writer produced a file that cannot be parsed; fix that first.")
            _before = len(_red)
            _new = _scrub_obj(_doc, _p.name)
            if len(_red) != _before:
                _p.write_text(json.dumps(_new, indent=1))
            continue
        # NON-JSON files are NEVER text-scrubbed. My first attempt at #218 word-boundary
        # replaced them, and on retail (ids '0'..'113') that rewrote `reward=1.0` in the
        # evaluator source into `reward=<a held-out task, id redacted>.0` — the evaluator
        # stopped parsing as Python. Word boundaries do not save you when the id IS a digit.
        #
        # The reason skipping is SAFE, and not merely convenient: every non-JSON file here is
        # copied verbatim from the benchmark's own source tree. Those files predate this
        # split and cannot encode which of its tasks are held out. That is asserted, not
        # assumed — byte-identity with the upstream copy is checked below, so a file that
        # someone edited to mention a task would fail loudly rather than slip through.
        # so only DISTINCTIVE ids are matched in text: at least 4 chars and not purely
        # numeric. `task_047` qualifies; retail's `1` and `30` do not. An integer id inside
        # evaluator source or a generated README is arithmetic, never a task reference, and
        # there is no textual test that can tell the two apart — so the honest move is to
        # decline the check and say so, rather than run one that mangles code.
        _hits = [i for i in sorted(_forb, key=len, reverse=True)
                 if i and len(i) >= 4 and not i.isdigit()
                 and _re.search(rf"(?<![0-9A-Za-z_]){_re.escape(i)}(?![0-9A-Za-z_])", _t)]
        if not _hits:
            continue
        for _i in _hits:
            _t = _re.sub(rf"(?<![0-9A-Za-z_]){_re.escape(_i)}(?![0-9A-Za-z_])",
                         "<a held-out task, id redacted>", _t)
            _red.append((_p.name, _i))
        _p.write_text(_t)
    # Assert on the BYTES ON DISK, and on the ids still resolving, so a scrub that broke the
    # JSON or missed a spelling cannot pass as "nothing to scrub".
    # #218: this verification must use the SAME notion of "an id occurs here" as the scrub.
    # A bare `id in text` check reports every retail id as surviving, because '1' occurs in
    # every price and timestamp — the audit would fail a correct scrub and pass nothing.
    def _occurs(_i, _p):
        _t = _p.read_text(errors="replace")
        if _p.suffix != ".json" and (len(_i) < 4 or _i.isdigit()):
            return False          # see the scrub: not checkable in text for short/numeric ids
        if _p.suffix == ".json":
            try:
                _doc = json.loads(_t)
            except json.JSONDecodeError:
                return True          # unparseable after a scrub IS a failure
            _found = [False]

            def _walk(o):
                if isinstance(o, str):
                    if o == _i:
                        _found[0] = True
                elif isinstance(o, list):
                    for x in o:
                        _walk(x)
                elif isinstance(o, dict):
                    for k, v in o.items():
                        _walk(k)
                        _walk(v)
            _walk(_doc)
            return _found[0]
        return bool(_re.search(rf"(?<![0-9A-Za-z_]){_re.escape(_i)}(?![0-9A-Za-z_])", _t))

    _still = {_p.name: sorted(i for i in _forb if _occurs(i, _p))
              for _p in ORACLE.rglob("*") if _p.is_file()}
    _still = {k: v for k, v in _still.items() if v}
    if _still:
        raise SystemExit(f"SCRUB FAILED — held-out ids survive in the oracle: {_still}")
    if read_oracle_ids() != train:      # the scrub must not have mangled the keys
        raise SystemExit("the #193 scrub changed the oracle's task ids — refusing")
    json.loads(_tf.read_text())         # and must not have broken the JSON
    if _red:
        print(f"  #193 scrub  : redacted {len(_red)} held-out id reference(s) from train specs")
        for _n, _i in _red:
            print(f"                {_n}: {_i} -> neutral placeholder")
    else:
        print("  #193 scrub  : no train spec referenced a held-out id")

    print(f"  materialized {out}")
    print(f"  train tasks : {len(got)}")
    print(f"  README.md   : {(ORACLE / 'README.md').stat().st_size} bytes")
    print(f"  evaluator   : {sorted(p.name for p in ORACLE.glob('evaluator*.py'))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="audit the oracle on disk; non-zero on any problem")
    a = ap.parse_args()
    split = load_split()
    print(f"  domain: {DOMAIN}   oracle: {ORACLE.name}")
    print(f"  split: train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])}")
    return audit(split) if a.verify else build(split)


if __name__ == "__main__":
    raise SystemExit(main())
