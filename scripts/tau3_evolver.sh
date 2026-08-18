#!/usr/bin/env bash
# Launch a TRUE isolated evolver for one iteration of the tau3 banking_knowledge run.
#
#   scripts/tau3_evolver.sh <iteration> <prompt_file> <digest_dir> <staging_dir>
#   scripts/tau3_evolver.sh 1 /path/to/ITER1_PROMPT.md \
#       outputs/tau3_banking/digests_e0_train \
#       outputs/tau3_banking/skills_v1_staging
#
# WHY A SUBPROCESS AND NOT A SUBAGENT. An Agent-tool subagent lacks `Workflow` and
# `TaskCreate`/`TaskUpdate` (probed 2026-08-01: 54 tools, no Workflow). A headless
# `claude -p` is a genuine top-level session with the full toolset, so the evolver
# can orchestrate however it judges best rather than however the orchestrator
# decided for it. the maintainer, 2026-08-04: "use opus 5 claude code session as the
# evolver — same as the ssb v2 setup."
#
# WHY AN ISOLATED WORKING DIRECTORY. A CC session loads CLAUDE.md and the user's
# auto-memory from its project path. This project's MEMORY.md carries the
# orchestrator's accumulated findings, including prior skill hypotheses and their
# measured nulls. An evolver that loads those is not independent; it is the
# orchestrator's priors laundered through a fresh process.
#
# ACCESS IS GRANTED NARROWLY. --add-dir names the digest, staging, oracle and
# policy-corpus directories THEMSELVES, never their parent: `outputs/tau3_banking/`
# also holds split.json (with its SELECTION_WARNING), the val and test task ids,
# and prior legs' result JSONs — every prior the isolation exists to remove. Note
# the corpus lives under third_party/, so naming it does NOT open the repo: see
# the corpus gate below for why that one grant is both safe and necessary.
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW THIS DIFFERS FROM scripts/evo0801_evolver.sh, AND WHY.
#
# evo0801 RENDERS the prompt from a contract template plus values derived from the
# evidence. Here the ORCHESTRATOR COMPOSES THE PROMPT BY HAND (the maintainer's explicit
# instruction), so there is no template to render — and that removes the guard
# evo0801 got its "derive, never hardcode" rule from. The rule therefore inverts:
# this launcher does not produce the numbers, it CHECKS THEM. Every factual claim
# the prompt makes about its own evidence is re-derived from the digests and
# compared, and a disagreement is a refusal.
#
# That inversion is strictly stronger where it matters. evo0801's failure was a
# hardcoded "33 of 180" carried from the bootstrap into epoch 2, whose real figure
# differed — a template can only carry one such error per site, whereas a
# hand-written prompt can invent any number anywhere. The class of error is the
# same and the exposure is larger, so the check has to be tighter, not looser.
set -euo pipefail

# EVOLVER MODEL, parameterised 2026-08-08 for the evolver ablation (the maintainer).
# Was hardcoded "opus[1m]" at three sites; a fork would have drifted. Default is
# UNCHANGED so every existing runbook reproduces exactly.
EVOLVER_MODEL="${EVOSKILL_EVOLVER_MODEL:-opus[1m]}"
# An empty model reaches CC as `--model ""` and dies with `API Error: 400 No models
# provided` after ~200ms — a failure that reads like a dead run, not a config error.
# The default above makes this unreachable today; the guard is here so that stays true
# if anyone ever makes the default conditional.
if [ -z "$EVOLVER_MODEL" ]; then
  echo "REFUSING: EVOLVER_MODEL is empty. CC would fail with 400 No models provided." >&2
  exit 16
fi

# ── #319: A DEEPSEEK-FAMILY EVOLVER MUST ROUTE THROUGH THE PINNING PROXY ─────
# ssbv2_selfevo e1/e2 dispatched `claude` straight at OpenRouter; unpinned routing
# rerolls the serving host per request, and gen-id sampling later showed those two
# sessions served by Baidu/Bedrock/StreamLake/Io Net with ZERO DeepSeek-native
# generations. The guard refuses that dispatch shape; the pin itself lives in
# scripts/cc_proxy.py (#314), which must be running and declaring it at /pin.
. "$(dirname "$0")/lib/deepseek_pin_guard.sh"
require_deepseek_native_route "$EVOLVER_MODEL" || exit 4


ROOT="${EVOSKILL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ITER=${1:?usage: tau3_evolver.sh <iteration> <prompt_file> <digest_dir> <staging_dir>}
PROMPT_FILE=${2:?}
DIGESTS=${3:?}
STAGING=${4:?}

[[ "$PROMPT_FILE" = /* ]] || PROMPT_FILE="$ROOT/$PROMPT_FILE"
[[ "$DIGESTS" = /* ]] || DIGESTS="$ROOT/$DIGESTS"
[[ "$STAGING" = /* ]] || STAGING="$ROOT/$STAGING"
[ -f "$PROMPT_FILE" ] || { echo "no prompt file: $PROMPT_FILE" >&2; exit 2; }
[ -d "$DIGESTS" ]     || { echo "no digest dir: $DIGESTS" >&2; exit 2; }
mkdir -p "$STAGING"

# ── the oracle is MANDATORY ──────────────────────────────────────────────────
# Epochs 1-3 of the 0801 run went out blind purely because nothing checked, and
# two of them reached OPPOSITE wrong conclusions about a scoring fact the oracle
# README states outright. A blind evolver must fail loudly at launch.
# DOMAIN is read here too (before the corpus block needs it) so the oracle path and the
# corpus decision cannot disagree about which task type this launch is for.
EVO_DOMAIN=${EVOSKILL_TAU3_DOMAIN:-banking_knowledge}
_ORC_DEF=$([ "$EVO_DOMAIN" = banking_knowledge ] && echo oracle || echo "oracle_$EVO_DOMAIN")
ORACLE=${EVOSKILL_TAU3_ORACLE:-$ROOT/outputs/tau3_banking/$_ORC_DEF}
# THE RUN DIRECTORY, parameterised. ORACLE already had an env override but the previous-skills
# lookup, the stream log and the trace all hardcoded outputs/tau3_banking. Consequence, hit on
# 2026-08-06: a SMOKE run living in outputs/tau3_smoke_0806 could not find its own iter-1
# skills (exit 6), and its evolver log/trace were written into the REAL run's directory, where
# a later reader would take them for the real run's artifacts. Same parameterisation task #160
# applied to evo0801_evolver.sh; this is the tau3 half of that harness-parity fix.
RUN_DIR=${EVOSKILL_TAU3_RUN_DIR:-$ROOT/outputs/tau3_banking}
[ -d "$RUN_DIR" ] || { echo "no run dir: $RUN_DIR" >&2; exit 2; }
[ -d "$ORACLE" ] || { echo "no oracle dir: $ORACLE — run scripts/make_tau3_oracle.py" >&2; exit 7; }
[ -f "$ORACLE/README.md" ] || { echo "oracle has no README.md: $ORACLE" >&2; exit 7; }

# ── A BENCHMARK WITH NO ORACLE AT ALL ───────────────────────────────────────
# The checks below exist to catch a BROKEN oracle. A benchmark that legitimately exposes NO golden
# specifications is a different case, and conflating the two is how a tau3 run would silently skip
# its own firewall. So it must be DECLARED, never inferred: SSB v2's ground truth is a reference
# WORKBOOK, and granting those would hand the evolver the train answers outright, so there is
# nothing to expose and nothing to validate.
#
# The directory and its README are still required. A granted path that does not exist is
# indistinguishable at the tool layer from one the evolver failed to read, and an empty directory
# that does not say why it is empty is indistinguishable from an oracle build that failed.
# EVO_SPLIT IS NOT PART OF THE ORACLE BLOCK. It feeds the SEALED-TEST LEAK HOOK further down
# (EVOSKILL_EVOLVER_SPLIT, ~lines 780 and 852), which is the guard that stops a held-out task
# id reaching the evolver. Leaving it inside the oracle firewall meant that declaring the
# oracle absent also unset the split the leak hook checks against — under `set -u` that
# crashed loudly, which is lucky: the quiet version of this bug disarms a sealed-test guard.
EVO_SPLIT="${EVOSKILL_TAU3_EVOLVER_SPLIT:-$ROOT/outputs/tau3_banking/split_2020.json}"

# THE SPLIT IS NOT OPTIONAL AND ITS ABSENCE IS NOT A WARNING.
# Until 2026-08-18 an unreadable split only printed "oracle gate : SKIPPED", and on the
# SSB v2 path (EVOSKILL_ORACLE_ABSENT=1) that whole block is skipped so it printed NOTHING
# — while evolver_leak_hook.py silently dropped its held-out-id detector and the banner
# below still said ARMED. The default above exists on no clone, so a hand-run straight from
# README.md reached that state BY DEFAULT. This gate is deliberately outside the oracle
# if/else: the split feeds the sealed-test guard, not the oracle firewall.
python3 - "$EVO_SPLIT" <<'SPLITPY' || exit 14
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
def refuse(msg):
    print(f"REFUSING: {msg}", file=sys.stderr)
    print("  The leak hook needs the split to tell train from held-out; without it every", file=sys.stderr)
    print("  task id is denied and the evolver cannot work. Point", file=sys.stderr)
    print("  EVOSKILL_TAU3_EVOLVER_SPLIT at this run's split json.", file=sys.stderr)
    raise SystemExit(1)
if not p.is_file():
    refuse(f"no split file at {p}")
try:
    d = json.loads(p.read_text())
except Exception as e:
    refuse(f"split at {p} is unparseable: {e}")
held = set(d.get("val", [])) | set(d.get("test", []))
if not held:
    refuse(f"split at {p} names no val/test arm - nothing to protect")
if not set(d.get("train", [])):
    refuse(f"split at {p} names no train arm")
print(f"  split      : {p.name} - {len(set(d['train']))} train, {len(held)} held-out [OK]")
SPLITPY

if [ "${EVOSKILL_ORACLE_ABSENT:-0}" = "1" ]; then
  echo "ORACLE DECLARED ABSENT — this benchmark exposes no golden specifications."
  echo "  granted + self-describing: $ORACLE/README.md"
  echo "  skipping evaluator/task-meta/split-firewall checks; the digests are the whole surface."
else
# tau2 grades with an evaluator, not a scorer.py — that is the tau3 analogue of
# evo0801's scorer.py gate, and naming the wrong file would let a broken oracle
# through.
[ -f "$ORACLE/evaluator.py" ] || { echo "oracle has no evaluator.py: $ORACLE" >&2; exit 7; }
# Was hardcoded to banking_knowledge.json, so every other task type died at exit 7
# ("oracle has no task meta") even with its oracle correctly built.
ORACLE_TASKS="$ORACLE/tasks/$EVO_DOMAIN.json"
[ -f "$ORACLE_TASKS" ] || { echo "oracle has no task meta: $ORACLE_TASKS" >&2; exit 7; }
N_ORACLE=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$ORACLE_TASKS")
[ "${N_ORACLE:-0}" -gt 0 ] || { echo "oracle task meta is empty: $ORACLE_TASKS" >&2; exit 7; }

# ── ORACLE vs SPLIT FIREWALL ────────────────────────────────────────────────
# The oracle is TRAIN-ONLY BY CONSTRUCTION, but only relative to the split it was
# BUILT from. Nothing here previously compared it to the split this run uses, and
# `ORACLE` has its own default (line ~68) INDEPENDENT of whatever the digests were
# built against -- so fixing the digest builder's oracle does not fix this one.
#
# FIRED FOR REAL 2026-08-06. The epoch-1 evolver launched against the old 32-task
# oracle while the experiment used the new 20/20/57 split: 7 of the new VAL tasks
# and 17 of the new TEST tasks were readable, i.e. the evolver could see
# `evaluation_criteria` and `user_scenario` for the very tasks meant to adjudicate
# it. Killed ~30s in, 0 skills written. The banner had ALREADY PRINTED
# "oracle: 32 task specs" against a 20-task split and every [OK] gate still
# passed, because each gate checked internal consistency and none compared the
# oracle to the split.
#
# A held-out spec reaching the optimizer is not a degraded result, it is no result.
# So this refuses rather than warns.
if [ -f "$EVO_SPLIT" ]; then
  python3 - "$EVO_SPLIT" "$ORACLE_TASKS" <<'PY' || exit 8
import json, sys
split = json.loads(open(sys.argv[1]).read())
oracle = json.loads(open(sys.argv[2]).read())
oids = set(oracle) if isinstance(oracle, dict) else {t["id"] for t in oracle}
tr = set(split["train"])
held = set(split.get("val", [])) | set(split.get("test", []))
leak = oids & held
if leak:
    raise SystemExit(
        f"REFUSING TO LAUNCH: the oracle exposes {len(leak)} HELD-OUT task spec(s) "
        f"{sorted(leak)[:6]}{'...' if len(leak) > 6 else ''} "
        f"(val={len(oids & set(split.get('val', [])))}, test={len(oids & set(split.get('test', [])))}). "
        f"An evolver that can read a held-out task's evaluation_criteria has been shown the "
        f"answer to its own gate. Rebuild the oracle against THIS split:\n"
        f"  EVOSKILL_TAU3_ORACLE_SPLIT={sys.argv[1]} EVOSKILL_TAU3_ORACLE_DIR=<dir> "
        f"python3 scripts/make_tau3_oracle.py\n"
        f"then pass EVOSKILL_TAU3_ORACLE=<dir>.")
missing = tr - oids
if missing:
    raise SystemExit(
        f"REFUSING TO LAUNCH: the oracle is MISSING {len(missing)} train task spec(s) "
        f"{sorted(missing)[:6]}. It was built from a smaller/older train arm, so the "
        f"evolver lacks specs for tasks it is being asked to learn from.")
print(f"  oracle gate : {len(oids)} specs == split train ({len(tr)}), "
      f"{len(held)} held-out ids absent [OK]")
PY
else
  echo "  oracle gate : SKIPPED — no split at $EVO_SPLIT" >&2
fi

fi

# ── the POLICY CORPUS is granted, and that is a deliberate decision ──────────
# The digests collapse each retrieval result to its ranked list and tell the
# reader the full document bodies are in the corpus. Without this grant that
# sentence would be a promise the environment does not keep — the same defect as
# #146, where the evolver was shown a solver prompt the solver never used. A
# pointer to a directory the reader cannot open is worse than no pointer: it
# invites reasoning about text that was never available.
#
# WHY IT IS NOT LEAKAGE. Checked 2026-08-04: 698 files, 2.8 MB, and NOT ONE
# contains a task id, `evaluation_criteria` or `reward_basis`. It is pure policy
# prose — the identical corpus the SOLVER retrieved from, shared across all three
# splits, so it says nothing about which tasks are val or test. The val firewall
# protects TASK SPECS, which live in the oracle (train-only by construction), not
# the documents.
#
# WHY IT IS NECESSARY. "Wrong because it never retrieved the policy" and "wrong
# although it read the policy" are different failures with different remedies,
# and the second one cannot be diagnosed without the policy text. That
# distinction is the single most useful thing this benchmark offers.
#
# THE CORPUS IS A banking_knowledge CONCEPT, AND THE GRANT IS NOW CONDITIONAL
# (2026-08-06). banking_knowledge is a knowledge-RETRIEVAL task with 698 policy documents
# behind it; airline, retail and telecom have no document corpus at all. This block
# required one unconditionally, so any non-banking domain died here at exit 7 before the
# gate could even read its digests — the same "banking-only by accident" shape as the
# hardcoded DOMAIN in the runner and the digest builder.
CORPUS=${EVOSKILL_TAU3_CORPUS:-$ROOT/third_party/tau3-bench/data/tau2/domains/$EVO_DOMAIN/documents}
if [ "$EVO_DOMAIN" = "banking_knowledge" ]; then
  [ -d "$CORPUS" ] || { echo "no document corpus: $CORPUS" >&2; exit 7; }
  N_CORPUS=$(find "$CORPUS" -name '*.json' | wc -l | xargs)
  [ "${N_CORPUS:-0}" -gt 0 ] || { echo "document corpus is empty: $CORPUS" >&2; exit 7; }
else
  # Absent is EXPECTED here, not tolerated-and-hoped-for. If a corpus does turn up for a
  # domain that should not have one, say so rather than silently granting it: an
  # unexplained read-only grant is an evidence-surface change nobody decided on.
  if [ -d "$CORPUS" ]; then
    N_CORPUS=$(find "$CORPUS" -name '*.json' | wc -l | xargs)
    echo "  corpus     : UNEXPECTED — $EVO_DOMAIN has $N_CORPUS document(s) at $CORPUS;"
    echo "               granting it, but this domain was not expected to have one."
  else
    CORPUS=""
    N_CORPUS=0
    echo "  corpus     : none (correct — only banking_knowledge has a policy corpus)"
  fi
fi

# ── derive every fact from the evidence ──────────────────────────────────────
N_DIGESTS=$(find "$DIGESTS" -name '*.md' | wc -l | xargs)
[ "${N_DIGESTS:-0}" -gt 0 ] || { echo "no *.md digests in $DIGESTS" >&2; exit 9; }

# `|| true` on every extraction is load-bearing, not defensive noise. grep exits 1
# when nothing matches, and under `set -e` that kills the launcher with an EMPTY
# stderr — and for several of these it matches nothing exactly when the run is
# HEALTHIEST (no wall-kills, no truncation). A derived value whose extraction
# fails on the good case is worse than no derived value. (evo0801 shipped that
# bug once; it is not repeated here.)
DERIVED=$(python3 - "$DIGESTS" <<'PY' || true
import ast, pathlib, re, shlex, sys, collections
d = pathlib.Path(sys.argv[1])
pas = tot = 0
turns, durs = [], []
terms = collections.Counter()
kinds = collections.Counter()
variants, tasks = set(), 0
for f in sorted(d.glob("*.md")):
    head = f.read_text(errors="replace")[:4000]
    m = re.search(r"\*\*Result: (\d+)/(\d+) samples passed\.\*\*", head)
    if not m:
        continue
    tasks += 1
    pas += int(m.group(1)); tot += int(m.group(2))
    v = re.search(r"^# \S+\s+\(\S+, variant (\S+)\)", head, re.M)
    if v:
        variants.add(v.group(1))
    for key, sink in (("turns", turns), ("duration_s", durs)):
        mm = re.search(rf"{key}=(\[[^\]]*\])", head)
        if mm:
            try:
                sink.extend(x for x in ast.literal_eval(mm.group(1)) if x is not None)
            except (ValueError, SyntaxError):
                pass
    mm = re.search(r"termination=(\[[^\]]*\])", head)
    if mm:
        try:
            for t in ast.literal_eval(mm.group(1)):
                terms[str(t)] += 1
        except (ValueError, SyntaxError):
            pass
    # OUTCOME KINDS, a DIFFERENT axis from termination: `termination` says how the
    # episode ended, `kind` says whether the reward it carries means anything. An
    # `ungraded` sample's 0.0 is a product over an empty reward_basis, so counting
    # it as a wrong answer inflates the graded-failure rate. Derived here rather
    # than retyped into the prompt, because retyping is exactly how the epoch-0
    # prompt came to describe an ungraded 2033s timeout as "finished normally".
    mm = re.search(r"kind=(\[[^\]]*\])", head)
    if mm:
        try:
            for k in ast.literal_eval(mm.group(1)):
                kinds[str(k)] += 1
        except (ValueError, SyntaxError):
            pass
print(f"N_TASKS={tasks}")
print(f"PASS={pas}")
print(f"TOTAL={tot}")
print(f"VARIANT={sorted(variants)[0] if len(variants) == 1 else ''}")
print(f"N_VARIANTS={len(variants)}")
print(f"TURN_MIN={min(turns) if turns else ''}")
print(f"TURN_MAX={max(turns) if turns else ''}")
print(f"DUR_MAX={max(durs) if durs else ''}")
# SHELL-QUOTE the composed values. `eval "$DERIVED"` runs these as shell
# assignments, and a kind label is `ungraded(timeout)` — bare parentheses are a
# shell syntax error that kills the launcher before any gate runs. Caught by the
# dry run; a value assembled from data must never be pasted into eval unquoted.
print("TERMS=" + shlex.quote(",".join(f"{k}:{v}" for k, v in sorted(terms.items()))))
print("KINDS=" + shlex.quote(",".join(f"{k}:{v}" for k, v in sorted(kinds.items()))))
print("UNGRADED=" + str(sum(v for k, v in kinds.items() if k not in ("pass", "fail"))))
PY
)
eval "$DERIVED"
[ -n "${TOTAL:-}" ] && [ "${TOTAL:-0}" -gt 0 ] || {
  echo "could not derive the batch rate from $DIGESTS — no '**Result: k/N samples passed'" >&2
  echo "  markers found. tau3_build_digests.py writes that line and this launcher" >&2
  echo "  greps it; if you changed its wording, change it in both places." >&2
  exit 9; }

# ONE VARIANT, OR REFUSE. The retrieval variant decides which TOOLS the agent has,
# i.e. what the task IS, and it is NOT in the cache key. Digests from two variants
# would describe two different experiments while the prompt names one.
#
# BANKING ONLY (2026-08-06). `retrieval_variant` exists on banking_knowledge's env
# constructor and nowhere else, so airline/retail/telecom digests legitimately span ZERO
# variants and this check refused them at exit 9. The exactly-one requirement still holds
# for banking; the else-branch keeps the check meaningful elsewhere by requiring exactly
# ZERO, so banking digests handed to a non-banking launch are still caught.
if [ "$EVO_DOMAIN" = banking_knowledge ]; then
[ "${N_VARIANTS:-0}" = "1" ] || {
  echo "the digests span ${N_VARIANTS} retrieval variants — refusing. The variant" >&2
  echo "  decides which tools the solver had, so a mixed set is two experiments." >&2
  exit 9; }
else
  # A non-retrieval domain must span ZERO variants. More than zero means these digests
  # came from banking — i.e. the wrong evidence for this launch — so it still refuses.
  [ "${N_VARIANTS:-0}" = "0" ] || {
    echo "domain $EVO_DOMAIN has no retrieval variants, but its digests report" >&2
    echo "  ${N_VARIANTS} — these digests are not this domain's. Refusing." >&2
    exit 9; }
fi

echo "TAU3 EVOLVER iter=$ITER"
echo "  digests    : $N_DIGESTS files, $N_TASKS tasks, variant ${VARIANT:-n/a}"
echo "  batch      : $PASS/$TOTAL episodes passed  (DERIVED, never hardcoded)"
echo "  turns       : ${TURN_MIN:-?}-${TURN_MAX:-?}   max duration ${DUR_MAX:-?}s"
echo "  termination : ${TERMS:-(none recorded)}"
echo "  outcome kind: ${KINDS:-(no kind= field — rebuild the digests)}"
[ "${UNGRADED:-0}" -gt 0 ] && {
  echo "                ^^ ${UNGRADED} episode(s) carry a reward NO CHECK PRODUCED"
  echo "                   (empty reward_basis). Those 0.0s are the ABSENCE of a"
  echo "                   verdict, not wrong answers. Do not describe them as"
  echo "                   graded failures, and do not call them normal finishes."
} || true
# N_ORACLE is set only by the tau3 branch above. Under EVOSKILL_ORACLE_ABSENT that branch does
# not run, and `set -u` turned this display line into a hard exit 1 AFTER the evolver had already
# printed its whole banner — a crash dressed as a refusal. Say "none exposed" rather than "0 task
# specs": zero specs and no-oracle-by-design read identically in a log and mean opposite things.
if [ "${EVOSKILL_ORACLE_ABSENT:-0}" = "1" ]; then
  echo "  oracle     : none exposed (this benchmark has no golden specs to grant)"
else
  echo "  oracle     : $N_ORACLE task specs (TRAIN only — val/test absent by construction)"
fi
[ -n "$CORPUS" ] && \
  echo "  corpus     : $N_CORPUS policy documents (read-only; no task specs in it)" || true

# ── THE CONSISTENCY GATE: does my hand-written prompt agree with the evidence? ─
# This is the piece evo0801 does not have and tau3 needs, because the prompt is
# authored rather than rendered. Three checks, each targeting a mistake that has
# actually been made:
#
#  1. The prompt must STATE the batch rate, and it must be the derived one. Any
#     `k/N episodes`-shaped claim that disagrees is a refusal. (evo0801 carried a
#     stale "33 of 180" into the wrong epoch.)
#  2. The prompt must name the retrieval VARIANT the evidence came from. This is
#     the tau3 analogue of evo0801's solver-image probe and it guards the exact
#     #146 failure: an evolver told about a solver that does not exist. #146 was
#     a retail policy shown on a banking run; naming the wrong variant is the
#     same error with a smaller blast radius and no louder a signal.
#  3. The prompt must not claim a task count other than the derived one.
python3 - "$PROMPT_FILE" "$PASS" "$TOTAL" "$VARIANT" "$N_TASKS" <<'PY' || exit 10
import re, sys, pathlib
prompt = pathlib.Path(sys.argv[1]).read_text(errors="replace")
# #255. STRIP MARKDOWN EMPHASIS BEFORE MATCHING. Every check below matches a MEANING
# ("25 train tasks", "19/75", the variant name), never a formatting choice — but emphasis
# markers land INSIDE those phrases and break literal adjacency. The first AO-authored
# prompt wrote "**25** train tasks", so the task-count gate saw no checkable claim and
# refused to launch a perfectly correct prompt. Same root cause as #254, third instance:
# a gate that matches literal adjacency is broken by an author who bolds a number.
# Normalising emphasis away loosens NOTHING semantically — the digits and words are
# unchanged — it only stops formatting from being read as content.
prompt = re.sub(r"\*|__", "", prompt)   # ALL asterisks: these gates match PHRASES, not list
#   structure, so turning "* item" into " item" costs nothing while `*25*` and `**25**` both
#   become checkable. Lossless for this purpose.
# DELIBERATE LIMIT: single underscores are NOT stripped. `_25_` would need it, but a single
# underscore is not reliably emphasis here — these prompts legitimately contain identifiers
# like doc_credit_cards_001 and termination reasons like user_stop, and mangling those would
# corrupt real content to catch a rare formatting choice. `**`, `*` and `__` cover it.
pas, tot, variant, n_tasks = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
bad = []

# 1. every episode-rate claim must equal the derived pair
#
# #208 FIXED (2026-08-09). The matcher used to treat EVERY `k/N episodes|rollouts` phrase as a
# PASS-RATE claim, so an UPTAKE disclosure -- "54 of 90 rollouts invoked the skill tool" --
# was compared against the pass rate and refused. That is not a near-miss: disclosing library
# size and measured uptake is the one prompt intervention that ever changed this optimizer's
# behaviour (banking epoch 4 cut the library 14% the round it was told), so a gate that
# forbids saying it forbids the intervention.
#
# Classify by the VERB near the number, not by the noun after it. A pass-rate claim says
# passed/scored/solved/succeeded; an uptake or population claim says invoked/loaded/used/ran.
# Anything that is NOT a pass claim is ignored for the arithmetic below -- and every ignored
# claim is PRINTED, because a gate that silently drops what it cannot classify is the vacuous
# shape this project keeps rediscovering (#154). If the classifier is ever wrong, the log says
# so on the line above the launch.
#
# The window must be TIGHT. A first attempt looked +/-90 chars around the match and
# misclassified the uptake claim as a pass claim, because the batch sentence's own "passed"
# sat inside that window ("...68 of 90 episodes passed. 54 of 90 rollouts invoked..."). The
# predicate that classifies a claim is the verb IMMEDIATELY after it, so decide on the next
# 25 characters first and only widen when they are inconclusive.
#
# FAILS CLOSED BY CONSTRUCTION: if this classifier ever wrongly IGNORES the batch claim,
# `claims` ends up empty and the "states NO episode pass rate" branch below refuses the
# launch. A misclassification is therefore loud, never a silent pass.
pat = re.compile(r"(\d+)\s*(?:/|of|out of)\s*(\d+)\s+(?:episodes?|rollouts?)", re.I)
_PASSY = re.compile(r"pass|scor|solv|succeed|correct|fail", re.I)
_UPTAKE = re.compile(r"invok|load|consult|use[ds]?\b|using|call|read|fire", re.I)
claims, ignored = set(), []
for m in pat.finditer(prompt):
    tail, head = prompt[m.end():m.end() + 25], prompt[max(0, m.start() - 40):m.start()]
    if _PASSY.search(tail):
        verdict = "pass"
    elif _UPTAKE.search(tail):
        verdict = "uptake"
    else:                       # inconclusive -> widen, and default to NOT a pass claim
        verdict = "pass" if _PASSY.search(head + prompt[m.end():m.end() + 60]) else "uptake"
    if verdict == "pass":
        claims.add((int(m.group(1)), int(m.group(2))))
    else:
        ignored.append((m.group(0).strip(),
                        (m.group(0) + tail).replace("\n", " ").strip()[:70]))
if ignored:
    print(f"  prompt gate : {len(ignored)} non-pass-rate k/N claim(s) NOT checked against the "
          f"batch (uptake/population, classified by context):", file=sys.stderr)
    for txt, ctx in ignored[:6]:
        print(f"                {txt!r}  <- ...{ctx}...", file=sys.stderr)
if not claims:
    bad.append(
        f"the prompt states NO episode pass rate. The evidence's own headline is "
        f"{pas}/{tot}; a prompt that never states it leaves the evolver to compute "
        f"the batch rate from {n_tasks} separate files, which is how a wrong one gets "
        f"quoted back at you.")
# #220 (2026-08-09): a STRATUM is not a stale batch rate. The rule used to be "every
# k/N episodes claim must equal the derived pair", which was right while a prompt only ever
# stated one rate. Retail's iter-1 prompt legitimately decomposes the batch by scoring
# channel -- 19/27 episodes on the assertion-bearing tasks, 40/63 on the database-only ones
# -- and 19+40=59, 27+63=90 reconciles exactly with the batch. The old rule refused it, and a
# guard that refuses true prose is a guard that gets switched off.
#
# The replacement keeps the thing worth guarding (evo0801 carried a stale "33 of 180" into
# the wrong epoch) and adds the thing a decomposition must prove: SUB-CLAIMS MUST PARTITION.
# Any non-batch claim must be a strict sub-population, and the non-batch claims must contain
# a subset summing EXACTLY to the batch. So a stale rate still refuses (it does not reconcile),
# a typo'd stratum refuses (it does not sum), and a CHERRY-PICKED stratum refuses too -- you
# cannot show the evolver the flattering half of a split and omit its complement.
subs = sorted(claims - {(pas, tot)})
if (pas, tot) not in claims and claims:
    for a, b in subs:
        bad.append(
            f"the prompt claims {a}/{b} episodes but never states the batch rate {pas}/{tot}. "
            f"Fix the prompt, or digest the leg you meant.")
elif subs:
    for a, b in subs:
        if b >= tot or a > pas:
            bad.append(
                f"the prompt claims {a}/{b} episodes, which is not a sub-population of the "
                f"batch {pas}/{tot}. Fix the prompt, or digest the leg you meant.")
    if not bad:
        reach = {(0, 0)}
        for a, b in subs:
            reach |= {(x + a, y + b) for x, y in reach}
        if (pas, tot) not in reach:
            bad.append(
                f"the prompt states sub-rates {subs} alongside the batch {pas}/{tot}, but no "
                f"combination of them adds up to it. Either they are wrong, or a stratum is "
                f"missing — state the complementary stratum so the decomposition reconciles "
                f"rather than showing the evolver one half of a split.")

# 2. the variant must be named
if variant and variant not in prompt:
    bad.append(
        f"the prompt never names the retrieval variant {variant!r}. The variant "
        f"decides which TOOLS the solver had — i.e. what the task IS — so a prompt "
        f"that omits it lets the evolver assume a different instrument. This is "
        f"defect #146 in general form.")

# 3. task-count claims ABOUT THIS BATCH.
#    A bare `(\d+) tasks` regex was wrong: it flagged "32 of the benchmark's 97 tasks",
#    where 97 is a TRUE statement about tau3-bench and says nothing about this batch. A
#    guard that forces true prose to be contorted gets relaxed, and then it guards
#    nothing — so it matches only phrasings that MEAN this batch, and separately requires
#    at least one such phrasing to exist.
batch_pat = re.compile(
    r"(\d+)\s+(?:train(?:ing)?\s+tasks\b"          # "32 train tasks"
    r"|tasks\s*(?:x|\u00d7)"                          # "32 tasks x 3 samples"
    r"|tasks\s+in\s+this\s+batch\b)", re.I)
found = {int(x) for x in batch_pat.findall(prompt)}
if not found:
    bad.append(
        f"the prompt never states how many tasks are in this batch in a form that can be "
        f"checked. Write it as '{n_tasks} train tasks', '{n_tasks} tasks x <N> samples' or "
        f"'{n_tasks} tasks in this batch' — a bare count cannot be distinguished from a "
        f"true remark about the benchmark's total size.")
for n in found:
    if n != n_tasks:
        bad.append(f"the prompt says {n} tasks in this batch; the digests contain {n_tasks}.")

if bad:
    print("PROMPT/EVIDENCE MISMATCH — refusing to launch:", file=sys.stderr)
    for b in bad:
        print("  * " + b, file=sys.stderr)
    sys.exit(1)
print("  prompt gate : batch rate, variant and task count all agree with the digests [OK]")
PY

# ── seed staging from the previous iteration ─────────────────────────────────
# From iteration 2 on the evolver is EDITING a live skill set, not authoring from
# scratch: the traces it reads were produced by a solver carrying skills_v{N-1},
# and the actions available to it are create/edit/delete against what exists.
# Launching iteration 2 with an empty staging dir turns a multi-iteration
# trajectory into N independent one-shot attempts.
PREV=$((ITER - 1))
PREV_SKILLS="$RUN_DIR/skills_v${PREV}"
if [ "$ITER" -gt 1 ]; then
  [ -d "$PREV_SKILLS" ] || { echo "missing previous skills: $PREV_SKILLS" >&2; exit 6; }
  cp -R "$PREV_SKILLS"/. "$STAGING"/
fi
N_SEED=$(find "$STAGING" -name SKILL.md | wc -l | xargs)
echo "  staging    : $STAGING ($N_SEED skill(s) seeded from iter $PREV)"

# ── isolated scratch cwd + session identity ──────────────────────────────────
# WORK IS KEYED BY ARM, NOT ONLY BY ITERATION (2026-08-08, the 3-arm ablation).
# It was `tau3_evolver_e${ITER}` flat, so every arm's epoch-N evolver shared ONE
# cwd. The launcher writes PROMPT.md into that cwd, and from epoch 2 on a prompt
# states its own arm's measured train rate — so arm B's epoch-2 evolver would
# have opened its cwd and found arm A's numbers sitting there. That is a
# cross-arm leak in the one experiment whose entire claim is that the arms differ
# only in the factor under test. Unset tag reproduces the old path exactly.
WORK_TAG="${EVOSKILL_TAU3_WORK_TAG:-}"
WORK="${EVOSKILL_EVOLVER_WORK_BASE:-${TMPDIR:-/tmp}}/evoskill_tau3_evolver_${WORK_TAG:+${WORK_TAG}_}e${ITER}"
LOG=$RUN_DIR/tau3_evolver_e${ITER}.log
# `scratch/` is NAMED IN THE PROMPT's contract as available scratch space, so it
# has to exist before the evolver's first exploratory command. evo0801 creates it;
# this launcher did not, which made the contract promise a directory that was not
# there — the #146 failure shape (an evolver told about something unreal) and an
# arm-parity break against the SSB v2 launcher. Found by cognitive proofread,
# 2026-08-04, before dispatch.
mkdir -p "$WORK/scratch"
# A CC session loads CLAUDE.md from its cwd, so a stray one here would void the
# isolation this whole launcher exists to provide. evo0801 ASSERTS this; tau3 only
# PRINTED "isolated — no CLAUDE.md, no MEMORY.md" and checked nothing. A printed
# claim with no check is the same family as a vacuous guard: it cannot fail.
#
# #179, 2026-08-06 — AND THE CHECK WAS POINTED AT THE WRONG DIRECTORY, because THIS
# LAUNCHER NEVER `cd`-ED. It built $WORK, wrote PROMPT.md into it, derived PROJDIR
# from it and recorded `"cwd": "$WORK"` in the trace — then ran `claude` from
# whatever cwd the caller happened to have, which for every tau3 launch so far was
# the REPO ROOT. Consequences: the evolver's session landed in
# projects/<the repo's CC project slug> (verified), the whole repo was readable
# regardless of the deliberately narrow --add-dir grants, and this stray-file guard
# was inspecting a directory the evolver never entered — so it could not fire and
# protected nothing.
#
# Audited before fixing rather than assumed: the epoch-1, -2 and -3 session
# transcripts were grepped for reads of split.json, SELECTION_WARNING, other epochs'
# digests, prior result JSONs, .claude/skills and the memo. Every epoch touched ONLY
# its own granted dirs, so no contamination actually occurred and no epoch needs
# re-running — but the isolation was nominal, not real, for all three.
#
# The `cd` below makes $WORK the true cwd, which is what finally gives this guard
# something to guard. CC also searches PARENTS for CLAUDE.md, so the chain is checked
# too — guarding only the leaf would be the same partial check in a new place.
for _stray in CLAUDE.md MEMORY.md .claude; do
  [ -e "$WORK/$_stray" ] && {
    echo "isolation VOID: $WORK/$_stray exists — the evolver would load it." >&2
    echo "  Remove it, or the run inherits the orchestrator's priors." >&2
    exit 5; }
done
_chain=$(dirname "$WORK")
while [ "$_chain" != "/" ]; do
  [ -e "$_chain/CLAUDE.md" ] && {
    echo "isolation VOID: $_chain/CLAUDE.md is on the cwd's PARENT chain and CC" >&2
    echo "  searches upward, so the evolver would load it." >&2
    exit 5; }
  _chain=$(dirname "$_chain")
done
SID=$(uuidgen)
# Claude Code derives its project dir from the cwd by replacing '/', '.' AND '_'
# with '-'. Replacing only '/' yields a path that does not exist, so the sidecar
# would point at nothing.
PROJKEY=$(python3 -c "import re,sys; print(re.sub(r'[/._]','-',sys.argv[1]))" "$WORK")
PROJDIR="$HOME/.claude/projects/$PROJKEY"

PROMPT=$(cat "$PROMPT_FILE")
[ -n "$PROMPT" ] || { echo "prompt file is empty: $PROMPT_FILE" >&2; exit 2; }
# The post-compaction handover artifact: the EXACT bytes dispatched, in the
# evolver's own cwd, so a compacted instance re-reads what it was actually sent
# rather than a template.

# #188: A DRY RUN WRITES ONLY INTO ITS OWN THROWAWAY SUBDIR.
# The guard below used to send a dry run's trace to "$WORK/..." on the reasoning
# that a dry run must not clobber a real one — but $WORK is the LIVE evolver's
# cwd, so it relocated the clobber rather than preventing it. And this PROMPT.md
# write ran unconditionally, ABOVE the dry-run check, so a dry run rewrote the
# live run's record of what it was actually dispatched. Observed 2026-08-06:
# dry-running iteration 1 while iteration 1 was running overwrote both files in
# the live $WORK at 21:47, twelve minutes into the real run. Content happened to
# be identical, so nothing was lost — the mechanism is the finding.
if [ -n "${EVOSKILL_TAU3_DRY_RUN:-}" ]; then
  WORK_OUT="$WORK/dryrun"
  mkdir -p "$WORK_OUT"
  TRACE_OUT="$WORK_OUT/tau3_evolver_e${ITER}.trace.json"
else
  WORK_OUT="$WORK"
  TRACE_OUT="$RUN_DIR/tau3_evolver_e${ITER}.trace.json"
fi
# The dispatched prompt is written where the evolver's cwd can reach it, so a
# compacted instance re-reads what it was actually sent rather than a template.
printf '%s' "$PROMPT" > "$WORK_OUT/PROMPT.md"
cat > "$TRACE_OUT" <<EOF
{
  "iteration": ${ITER},
  "session_id": "$SID",
  "cwd": "$WORK",
  "project_dir": "$PROJDIR",
  "session_transcript": "$PROJDIR/$SID.jsonl",
  "subagent_transcripts": "$PROJDIR/$SID/subagents/",
  "stream_json_log": "$LOG",
  "prompt_file": "$PROMPT_FILE",
  "prompt_dispatched": "$WORK_OUT/PROMPT.md",
  "prompt_chars": $(printf '%s' "$PROMPT" | wc -c | xargs),
  "digests": "$DIGESTS",
  "staging": "$STAGING",
  "oracle": "$ORACLE",
  "corpus": "$CORPUS",
  "n_corpus_documents": $N_CORPUS,
  "n_digests": $N_DIGESTS,
  "retrieval_variant": "$VARIANT",
  "batch_pass": $PASS,
  "batch_total": $TOTAL
}
EOF
echo "  cwd        : $WORK   (isolated — no CLAUDE.md, no MEMORY.md)"
echo "  session    : $PROJDIR/$SID.jsonl"

# THE GRANT LIST, BUILT ONCE. Three sites named the four dirs separately (the dry-run
# echo, the Phoenix pipeline, the plain dispatch), so a conditional corpus would have had
# to be got right three times — and `--add-dir ""` passes an EMPTY argument rather than
# omitting the flag. One array, expanded everywhere.
GRANTS=(--add-dir "$DIGESTS" --add-dir "$STAGING" --add-dir "$ORACLE")
[ -n "$CORPUS" ] && GRANTS+=(--add-dir "$CORPUS")

# ── #294: KERNEL CONTAINMENT FOR THE `claude` CHILD, opt-in ──────────────────
# EVOSKILL_EVOLVER_SANDBOX=1 wraps the dispatch below in sandbox-exec under the SAME
# profile the `direct` harness has had since #250.
#
# WHY IT IS WORTH DOING. Everything protecting this dispatch today is a hook or a flag, and
# both have been found vacuous: --add-dir does not confine Bash at all (#186, measured on the
# live epoch-1 evolver: 49 Bash calls, many inside third_party/, despite four narrow grants);
# the peer-skill hook false-positived on the bare word "skills" (#215) and its fix was partial
# (#217); and the #213 peer-artifact audit examined ZERO records on every real log while
# reporting "passed" (#248). A kernel deny cannot be vacuous in any of those ways.
#
# WHY IT IS OPT-IN AND NOT ON. This profile had never met a real `claude` until 2026-08-11,
# when the AO became the first thing to run under it and died twice over (#291: it could not
# authenticate, and a DENIED ancestor of cwd is a fatal EPERM). The evolver does strictly more
# than the AO — six concurrent subagents, Write, Bash, a Phoenix pipe — so "the AO works under
# it" is not evidence that this does. Default OFF until a full evolver has completed under it.
#
# WHY ONLY THE CHILD. This launcher reads $EVO_SPLIT for its own oracle-vs-split firewall, and
# the split lists every held-out id. Sandboxing the whole script would mean granting the
# launcher exactly the artifact the sandbox exists to hide.
#
# THE PROFILE IS NOT BUILT HERE. `python3 -m src.orchestrator.evolver_dispatch --emit-profile`
# owns those rules; a shell copy of them is #154's shape (two implementations, silent drift,
# and the untested one is the one that runs).
SB=()
if [ "${EVOSKILL_EVOLVER_SANDBOX:-1}" = "1" ] && ! command -v sandbox-exec >/dev/null 2>&1; then
  echo "REFUSING: kernel-level evolver containment needs sandbox-exec (macOS-only), which this host lacks." >&2
  echo "  To run WITHOUT it, set EVOSKILL_EVOLVER_SANDBOX=0 — the PreToolUse leak hooks and --add-dir" >&2
  echo "  scoping still apply, but the evolver process is no longer kernel-confined: a hook gap would" >&2
  echo "  let it read held-out task specs. Acceptable for exploratory runs; record it for reported ones." >&2
  exit 9
fi
if [ "${EVOSKILL_EVOLVER_SANDBOX:-1}" = "1" ]; then
  _SBPROF="$WORK/evolver_e${ITER}.sb"
  _sb_args=(--epoch "$ITER" --work-dir "$WORK" --log "$LOG" --prompt-path "$PROMPT_FILE"
            --grant "$DIGESTS" --grant "$STAGING" --grant "$ORACLE"
            --write "$WORK" --write "$STAGING"
            --hide "$RUN_DIR"
            # #292, evolver half. The CLI writes AND fstats its own session transcript at
            # ~/.claude/projects/<slug-of-cwd>/<sid>.jsonl, and #291 denies read on
            # ~/.claude/projects to keep OTHER runs' conversations out of reach. Write was
            # allowed and the fstat was not, so the process died at startup with
            # `EPERM: operation not permitted, fstat` inside a Bun WriteStream -- a stack trace,
            # not a path, which is why only a real launch could find it. Grant back exactly THIS
            # run's project dir; every other one stays denied. Measured 2026-08-11 on the first
            # real contained evolver dispatch.
            --grant "$PROJDIR")
mkdir -p "$PROJDIR"          # #299: exists before the profile is built, so preflight can probe it
  [ -n "$CORPUS" ] && _sb_args+=(--grant "$CORPUS")
  # PREFLIGHT BEFORE SPENDING A DISPATCH. It probes exec, import, every declared read and the
  # log dir's is_dir() under the REAL profile and reports ALL gaps at once — four consecutive
  # attempts once each died on a different single missing grant.
  ( cd "$ROOT" && python3 -m src.orchestrator.evolver_dispatch --preflight "${_sb_args[@]}" ) \
    || { echo "REFUSING: evolver containment preflight failed (see above). Nothing spent." >&2; exit 17; }
  ( cd "$ROOT" && python3 -m src.orchestrator.evolver_dispatch \
      --emit-profile "$_SBPROF" "${_sb_args[@]}" >/dev/null ) \
    || { echo "REFUSING: could not emit the evolver sandbox profile." >&2; exit 17; }
  [ -s "$_SBPROF" ] || { echo "REFUSING: emitted profile is empty: $_SBPROF" >&2; exit 17; }
  SB=(sandbox-exec -f "$_SBPROF")
  echo "  containment : sandbox-exec -f $_SBPROF  [#294]"
else
  echo "  containment : NONE for the claude child — EXPLICITLY DISABLED via" \
       "EVOSKILL_EVOLVER_SANDBOX=0. The child can reach the split file, held-out specs and" \
       "peer arms' skills; only the #186/#215/#248 hooks stand in the way. Recorded in the" \
       "receipt so this epoch is never mistaken for a contained one." >&2
fi
# THE RECEIPT. #288's lesson: a feature whose absence is invisible is claimed, not shipped.
# Written before dispatch so a crashed launch still records which world it was in.
python3 - "$RUN_DIR/e${ITER}_EVOLVER_CONTAINMENT.json" "${EVOSKILL_EVOLVER_SANDBOX:-1}" \
         "${_SBPROF:-}" "$WORK" <<'PY' || true
import json, sys
out, on, prof, work = sys.argv[1:5]
json.dump({"sandboxed": on == "1", "profile": prof, "work_dir": work,
           "containment": "sandbox-exec" if on == "1" else "NONE (hooks + --add-dir only)"},
          open(out, "w"), indent=1)
PY

# ── #186: A PreToolUse DENY HOOK, BECAUSE --add-dir DOES NOT CONFINE BASH ────
# --add-dir confines Read/Glob/Grep. Bash it does not confine at all: it can `cd`
# anywhere this user can. Measured 2026-08-06 on the LIVE epoch-1 evolver — it ran
# 49 Bash calls, many of them inside third_party/tau3-bench/, despite four narrow
# grants. Two directories from what it legitimately read sit:
#
#     data/tau2/domains/<dom>/tasks.json           all 97 specs, train+val+test
#     tests/test_domains/test_<dom>/tasks/test_task_NNN.py
#         a prose walkthrough naming the held-out task's user, entities and exact
#         expected end state (test_task_028.py lists all six corrected values).
#
# The #179 note above says the `cd` into $WORK closed repo-wide readability. IT DID
# NOT — it fixed session placement and CLAUDE.md loading only, and the claim should
# not be trusted for Bash. This hook is the part that enforces.
#
# Scope is deliberately narrow: TASK SPECS for held-out ids. The environment itself
# (tools.py, db.json, documents/) stays readable — it is identical for train and
# held-out tasks, the agent touches it at runtime, and blocking it would forbid
# env-aware evolution rather than prevent leakage.
# ── #212: A SECOND PreToolUse HOOK, FORCING SYNCHRONOUS SUBAGENT DISPATCH ────
# The Agent tool runs subagents in BACKGROUND by default. The parent's tool_result is
# only "Async agent launched successfully"; the real result arrives later as a
# task_notification that starts a NEW turn. Claude models ride that protocol. DSV4F
# does not — measured 2026-08-08 on direct OpenRouter routing with a forced 2-subagent
# probe: async gave THREE separate result events and a final "RESULT: MISSING" (having
# already received one token and dropped it), while run_in_background=false gave one
# turn-chain and both correct tokens. This is the defect that made the 2026-08-08 DSV4F
# evolver probe look like a model failure while all six of its subagents ran clean.
#
# IT IS A HOOK, NOT A PROMPT LINE, ON PURPOSE. The evolver prompt is the experimental
# object; the Opus and Sonnet arms ran on a prompt that says nothing about subagent
# scheduling. Adding a line for one arm is exactly the extra scaffolding that was
# stripped from the Arm A epoch-2 prompt for parity. Changing the HARNESS keeps the
# prompt byte-comparable across arms.
#
# ON BY DEFAULT, because this is a latent defect that merely happened not to bite
# Claude models, and the standing rule is that a fix applies to ALL harnesses unless
# technically incompatible. Set EVOSKILL_TAU3_ASYNC_AGENTS=1 to restore the old
# behaviour when reproducing a pre-fix run.
SYNC_AGENT_HOOK_JSON=""
if [ "${EVOSKILL_TAU3_ASYNC_AGENTS:-0}" != "1" ]; then
  # DOUBLE quotes, so $ROOT EXPANDS HERE. The first version of this used single quotes,
  # which left the literal string `$ROOT` in the JSON — and a heredoc does NOT re-expand a
  # variable's VALUE, so nothing downstream ever fixed it. The other two hooks are written
  # directly inside the heredoc, where $ROOT does expand, which is why only this one broke.
  # Live consequence, 2026-08-08 epoch 2:
  #   PreToolUse:Agent hook error: python3: can't open file
  #   '.../digests_e2_train/$ROOT/scripts/evolver_sync_agent_hook.py'
  # The hook therefore NEVER fired in a real evolver run; the epoch-1 run happened to
  # dispatch zero subagents, so nothing surfaced it until an epoch that used one.
  SYNC_AGENT_HOOK_JSON=",
      {
        \"matcher\": \"Agent\",
        \"hooks\": [
          {\"type\": \"command\",
           \"command\": \"python3 '$ROOT/scripts/evolver_sync_agent_hook.py'\"}
        ]
      }"
fi
HOOK_SETTINGS="$WORK_OUT/leak_hook_settings.json"
cat > "$HOOK_SETTINGS" <<EOF
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Read|Grep|Glob",
        "hooks": [
          {"type": "command",
           "command": "EVOSKILL_EVOLVER_SPLIT='$EVO_SPLIT' EVOSKILL_LEAK_DENY_PATHS='${EVOSKILL_LEAK_DENY_PATHS:-}' python3 '$ROOT/scripts/evolver_leak_hook.py'"}
        ]
      },
      {
        "matcher": "Bash|Read|Grep|Glob|Write|Edit",
        "hooks": [
          {"type": "command",
           "command": "EVOSKILL_EVOLVER_STAGING='$STAGING' python3 '$ROOT/scripts/evolver_peer_skill_hook.py'"}
        ]
      }$SYNC_AGENT_HOOK_JSON
    ]
  }
}
EOF
# The settings file is JSON that a shell heredoc assembled from a conditional string.
# Parse it before CC does: a malformed --settings is accepted silently by some CC
# versions and would disarm the LEAK hook too, which is the one that guards the
# experiment's validity.
if ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$HOOK_SETTINGS" 2>/dev/null; then
  echo "REFUSING: $HOOK_SETTINGS is not valid JSON. Both PreToolUse hooks —" >&2
  echo "  including the held-out-spec leak guard — would be silently disarmed." >&2
  exit 14
fi
# Prove the sync hook REWRITES before trusting it to (#154 vacuous-guard shape again:
# a hook that runs but changes nothing looks identical to one that works).
if [ "${EVOSKILL_TAU3_ASYNC_AGENTS:-0}" != "1" ]; then
  # PROBE THE COMMAND STRING THE SETTINGS FILE ACTUALLY CONTAINS, not the script by a path
  # this shell reconstructs. The first version ran `python3 "$ROOT/scripts/..."` directly —
  # which passes whenever the SCRIPT is fine, and says nothing about whether the WIRING is.
  # It printed "ARMED and probe-verified" for a settings file whose command was
  # `python3 '$ROOT/scripts/...'` (unexpanded) and could never execute. A guard that cannot
  # observe the failure it guards against is the #154 shape — and I wrote this one TODAY,
  # in the commit that was meant to prevent exactly that.
  _sprobe='{"tool_name":"Agent","tool_input":{"description":"d","prompt":"p","subagent_type":"general-purpose"}}'
  _synccmd=$(python3 - "$HOOK_SETTINGS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for h in d["hooks"]["PreToolUse"]:
    if h["matcher"] == "Agent":
        print(h["hooks"][0]["command"])
        break
PY
)
  if [ -z "$_synccmd" ]; then
    echo "REFUSING: no Agent matcher found in $HOOK_SETTINGS." >&2
    exit 15
  fi
  case "$_synccmd" in
    *'$ROOT'*|*'$'*)
      echo "REFUSING: the sync-agent hook command contains an UNEXPANDED variable:" >&2
      echo "    $_synccmd" >&2
      echo "  It cannot execute, and CC reports only a per-call 'hook error' the model" >&2
      echo "  never surfaces as a failure (#212 regression, 2026-08-08)." >&2
      exit 15;;
  esac
  if ! printf '%s' "$_sprobe" | eval "$_synccmd" 2>/dev/null \
       | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d['hookSpecificOutput']['updatedInput']['run_in_background'] is False else 1)"; then
    echo "REFUSING: the sync-agent hook did NOT rewrite run_in_background to false." >&2
    echo "  Command as wired: $_synccmd" >&2
    echo "  Subagent results would be delivered asynchronously on a later turn, which" >&2
    echo "  a non-Claude evolver does not consume (#212). Fix the hook before dispatch." >&2
    exit 15
  fi
  echo "  sync agents: ARMED, WIRING-verified (ran the settings file's own command, #212)"
else
  echo "  sync agents: DISABLED by EVOSKILL_TAU3_ASYNC_AGENTS=1 — subagent results will"
  echo "               arrive on later turns; only a Claude-family evolver handles that."
fi
# A DENY PROBE MUST REQUIRE EXIT 2, NOT MERELY NON-ZERO.
#
# Under Claude Code a PreToolUse hook blocks ONLY on exit code 2. Every other non-zero
# code is a NON-BLOCKING error: the message goes to the transcript and the tool call
# PROCEEDS. So `if <hook>; then REFUSE; fi` — refusing only when the hook exits 0 —
# accepts a CRASHING hook as proof of protection.
#
# That is not hypothetical. Until 2026-08-18 this tree's evolver_leak_hook.py imported
# `evolver_leak_audit`, a module that had never been ported here. Every invocation died
# ModuleNotFoundError -> exit 1, the probe read "not 0, therefore blocked", and the
# launcher printed "leak hook : ARMED and probe-verified" on BOTH release smoke runs
# while providing zero enforcement. The receipt was the only thing that worked.
#
# _hook_rc runs a probe and yields the hook's exit code, so the caller can compare it to
# the ONE code that means "blocked" instead of to the set of codes that means "something
# happened". Stderr is captured and surfaced on refusal — a crash's traceback is the
# fastest possible diagnosis, and discarding it is how the original failure stayed hidden.
#
# It reports through globals rather than an exit status on purpose: this script runs under
# `set -e`, so a helper that RETURNED the hook's non-zero code would abort the launcher at
# the call site — and the whole point is to inspect that code.
_HOOK_RC=0
_HOOK_ERR=""
_hook_rc() {  # _hook_rc <payload_json> <hook_path>  -> sets _HOOK_RC, _HOOK_ERR
  local _payload="$1" _hook="$2" _errf
  _errf=$(mktemp)
  _HOOK_RC=0
  printf '%s' "$_payload" | python3 "$_hook" >/dev/null 2>"$_errf" || _HOOK_RC=$?
  _HOOK_ERR=$(cat "$_errf")
  rm -f "$_errf"
}
# Prove the hook DENIES before trusting it to. A hook that silently fails open is
# worse than none: the launcher would report protection it is not providing, which
# is the vacuous-guard shape that has bitten this codebase repeatedly (#154).
# THE PROBE MUST NAME MATERIAL THIS RUN'S BENCHMARK ACTUALLY HOLDS. Hardcoded to tau3's pool, it
# was meaningless on an SSB v2 run — and because the hook ALSO had no SSB v2 patterns, the probe
# correctly came back "did not block" and refused dispatch. Both halves are now benchmark-aware:
# SSB v2's ground truth is the reference workbook under $SSBV2_ROOT, so that root is both the
# probe target and the declared deny path handed to the hook.
if [ -n "${EVOSKILL_LEAK_DENY_PATHS:-}" ]; then
  _probe_target="${EVOSKILL_LEAK_DENY_PATHS%%:*}/held_out_answer.xlsx"
else
  _probe_target="$ROOT/third_party/tau3-bench/data/tau2/domains/$EVO_DOMAIN/tasks.json"
fi
_probe='{"tool_name":"Bash","tool_input":{"command":"cat '"$_probe_target"'"}}'
export EVOSKILL_EVOLVER_SPLIT="$EVO_SPLIT"
export EVOSKILL_LEAK_DENY_PATHS="${EVOSKILL_LEAK_DENY_PATHS:-}"
_hook_rc "$_probe" "$ROOT/scripts/evolver_leak_hook.py"
if [ "$_HOOK_RC" -ne 2 ]; then
  echo "REFUSING: the leak hook did NOT block a read of the full task pool." >&2
  echo "  Expected exit 2 from scripts/evolver_leak_hook.py; got $_HOOK_RC." >&2
  if [ "$_HOOK_RC" -eq 0 ]; then
    echo "  Exit 0 = the hook examined the payload and ALLOWED it. Held-out specs" >&2
    echo "  would be reachable by Bash. The matcher is wrong for this benchmark." >&2
  else
    echo "  A non-zero code that is NOT 2 is a NON-BLOCKING error under Claude Code:" >&2
    echo "  the message is shown and the tool call proceeds anyway. The hook is" >&2
    echo "  crashing, so it enforces NOTHING. Its stderr was:" >&2
    printf '%s\n' "$_HOOK_ERR" | sed 's/^/    /' >&2
  fi
  exit 13
fi
# THE PROBE THAT CAN ACTUALLY OBSERVE A DEAD ID-DETECTOR.
# The path probe above is satisfied by LEAK_PATH_RE alone, which still matches when the
# held-out-id half of the hook is dead — that is precisely how the 2026-08-18 fail-open
# survived a green "ARMED" banner through two release smokes. So take a REAL held-out id
# from THIS run's split and require a deny, and run THE COMMAND STRING THE SETTINGS FILE
# ACTUALLY CONTAINS, the way the sync-agent probe above does (#212) — not a path this
# shell rebuilds with env it supplies itself, which cannot detect a mis-wired command.
_leakcmd=$(python3 - "$HOOK_SETTINGS" <<'LEAKPY'
import json, sys
for m in json.load(open(sys.argv[1]))["hooks"]["PreToolUse"]:
    c = m["hooks"][0]["command"]
    if "evolver_leak_hook.py" in c:
        print(c); break
LEAKPY
)
if [ -z "$_leakcmd" ]; then
  echo "REFUSING: no evolver_leak_hook.py entry in $HOOK_SETTINGS — the hook is not wired." >&2
  exit 13
fi
case "$_leakcmd" in
  *'$ROOT'*|*'$EVO_SPLIT'*|*'$STAGING'*)
    echo "REFUSING: the leak-hook command carries an UNEXPANDED variable and cannot run:" >&2
    echo "    $_leakcmd" >&2
    echo "  CC reports only a per-call 'hook error' the model never surfaces (#212)." >&2
    exit 13;;
esac
_heldid=$(python3 -c "
import json
d = json.load(open('$EVO_SPLIT'))
h = sorted(set(d.get('val', [])) | set(d.get('test', [])))
print(h[0].replace('/', '_'))")
_iprobe='{"tool_name":"Read","tool_input":{"file_path":"/tmp/'"$_heldid"'.json"}}'
_IRC=0
_ierrf=$(mktemp)
# `bash -c`, NOT `eval`: measured 2026-08-18, a bare `eval` as the last stage of a
# pipeline under this script's `set -e` reports 1 even when the hook exits 2 and prints
# BLOCKED — i.e. it turns a WORKING guard into a refusal, and would have turned a
# fail-open into "1" too, which this probe treats as not-blocked. A fresh `bash -c` does
# not inherit errexit, returns the hook's real code, and still runs the settings file's
# own command string, which is the property that matters (#212).
printf '%s' "$_iprobe" | bash -c "$_leakcmd" >/dev/null 2>"$_ierrf" || _IRC=$?
_IERR=$(cat "$_ierrf"); rm -f "$_ierrf"
if [ "$_IRC" -ne 2 ]; then
  echo "REFUSING: the leak hook did NOT block held-out task id '$_heldid'." >&2
  echo "  Expected exit 2; got $_IRC, running the settings file's own command:" >&2
  echo "    $_leakcmd" >&2
  echo "  Held-out ids are nameable to the evolver. If the split is unusable the hook" >&2
  echo "  drops to denying EVERY id, so a 0 here means the wiring itself is wrong." >&2
  if [ "$_IRC" -ne 0 ]; then
    echo "  A non-zero code that is NOT 2 does not block under Claude Code — the hook is" >&2
    echo "  crashing and enforces NOTHING. Its stderr was:" >&2
    printf '%s\n' "$_IERR" | sed 's/^/    /' >&2
  fi
  exit 13
fi
# ... and prove it ALLOWS ordinary environment reads. Without a must-ALLOW case the
# matcher can be widened until it denies everything, which LOOKS like maximal safety and
# is in fact a dead evolver: a blocked read is invisible in the result, the model simply
# works around it. Same reason the peer-skill hook carries #215's must-allow probes below.
_lprobe='{"tool_name":"Bash","tool_input":{"command":"ls -la '"$DIGESTS"'"}}'
_hook_rc "$_lprobe" "$ROOT/scripts/evolver_leak_hook.py"
if [ "$_HOOK_RC" -ne 0 ]; then
  echo "REFUSING: the leak hook denies a read of this run's OWN evidence dir." >&2
  echo "  Expected exit 0; got $_HOOK_RC. The evolver could not read the material it" >&2
  echo "  is supposed to learn from. Its stderr was:" >&2
  printf '%s\n' "$_HOOK_ERR" | sed 's/^/    /' >&2
  exit 21
fi
# #213: prove the PEER-SKILL hook denies too. The probe is the EXACT command the DSV4F
# epoch-1 evolver used to read the Opus reference arm's accepted skill bodies, so this
# check fails if that specific hole ever reopens — not merely if the hook is importable.
_pprobe='{"tool_name":"Bash","tool_input":{"command":"cd '"$ROOT"'/outputs/tau3_banking_2525 && for f in skills_v1/*/SKILL.md; do cat \"$f\"; done"}}'
export EVOSKILL_EVOLVER_STAGING="$STAGING"
_hook_rc "$_pprobe" "$ROOT/scripts/evolver_peer_skill_hook.py"
if [ "$_HOOK_RC" -ne 2 ]; then
  echo "REFUSING: the peer-skill hook did NOT block a read of another run's skills_v1." >&2
  echo "  Expected exit 2; got $_HOOK_RC (only 2 blocks a tool call; see the note above)." >&2
  echo "  Another optimizer's answers to THESE SAME train tasks would be readable, which" >&2
  echo "  measures copying rather than evolution (#213). Fix the hook before dispatching." >&2
  if [ "$_HOOK_RC" -ne 0 ]; then printf '%s\n' "$_HOOK_ERR" | sed 's/^/    /' >&2; fi
  exit 17
fi
# ... and prove it still ALLOWS this run's own staging, or the evolver cannot revise.
_aprobe='{"tool_name":"Bash","tool_input":{"command":"ls -la '"$STAGING"'"}}'
if ! printf '%s' "$_aprobe" | EVOSKILL_EVOLVER_STAGING="$STAGING" \
     python3 "$ROOT/scripts/evolver_peer_skill_hook.py" >/dev/null 2>&1; then
  echo "REFUSING: the peer-skill hook blocks this run's OWN staging dir ($STAGING)." >&2
  echo "  The evolver could not read or revise its own skills. Fix before dispatching." >&2
  exit 18
fi
# #213 case 2: the bare `skills_v<N>` form carries NO slash and follows a `cd` this hook
# cannot see. It must stay denied — the #215 fix narrowed the matcher and this is the case
# a careless narrowing would drop.
_pprobe2='{"tool_name":"Bash","tool_input":{"command":"cd '"$ROOT"'/outputs/tau3_banking_2525 && for d in skills_v1 skills_v2; do ls -la $d; done"}}'
_hook_rc "$_pprobe2" "$ROOT/scripts/evolver_peer_skill_hook.py"
if [ "$_HOOK_RC" -ne 2 ]; then
  echo "REFUSING: the peer-skill hook did NOT block the bare 'skills_v1 skills_v2' form." >&2
  echo "  Expected exit 2; got $_HOOK_RC (only 2 blocks a tool call; see the note above)." >&2
  echo "  This is #213's second command; the #215 narrowing must not have dropped it." >&2
  if [ "$_HOOK_RC" -ne 0 ]; then printf '%s\n' "$_HOOK_ERR" | sed 's/^/    /' >&2; fi
  exit 19
fi
# #215: the two commands this hook WRONGLY denied on 2026-08-08. Both read the run's own
# evidence and merely contain the word "skills". Without a must-ALLOW case the matcher can
# be tightened back into over-blocking and nothing would notice — over-blocking is silent,
# because a denied evolver just works around it.
for _fp in \
  'D=./digests; echo "########## 097 skills ##########"; grep -n "tool.skill" $D/task_097.md' \
  'python3 - <<EOF
skills_loaded = 0
print(skills_loaded)
EOF'
do
  _fpjson=$(python3 -c 'import json,sys; print(json.dumps({"tool_name":"Bash","tool_input":{"command":sys.argv[1]}}))' "$_fp")
  if ! printf '%s' "$_fpjson" | EVOSKILL_EVOLVER_STAGING="$STAGING" \
       python3 "$ROOT/scripts/evolver_peer_skill_hook.py" >/dev/null 2>&1; then
    echo "REFUSING: the peer-skill hook denies a legitimate read of this run's OWN evidence." >&2
    echo "  Offending probe: $_fp" >&2
    echo "  This is #215 — the word 'skills' in prose or an identifier is not a path." >&2
    exit 20
  fi
done
echo "  peer skills: ARMED and probe-verified (other runs' skills_v* denied, #213)"
HOOKFLAG=(--settings "$HOOK_SETTINGS")
echo "  leak hook  : ARMED — path-probe exit 2, held-out-ID probe exit 2 for '$_heldid'"
echo "               via the settings file's own command,"
echo "               allow-probe returned exit 0 (own evidence dir still readable)"
# Kept OUT of GRANTS on purpose: the Phoenix branch below asserts "exactly 4
# grants" and GRANTS is what the dry run prints as the add-dir set. Folding a
# --settings pair in would make both of those quietly wrong.

# ── DRY RUN — validate everything, launch nothing ────────────────────────────
# Exists so nothing ever has to shadow `claude` on PATH to inspect a launch. On
# 2026-08-02 a test that meant to shadow it got the stub path wrong, the PATH
# prepend was a silent no-op, `claude` resolved to the real binary, and a genuine
# evolver ran unmonitored for 19 minutes with --dangerously-skip-permissions,
# exit 0 throughout. With a dry run there is nothing to get wrong.
if [ -n "${EVOSKILL_TAU3_DRY_RUN:-}" ]; then
  echo "  DRY RUN — no process launched, and the quota gate was NOT consulted."
  echo "  would have run: claude -p <${#PROMPT} chars> --model $EVOLVER_MODEL --session-id $SID \\"
  echo "    ${GRANTS[*]} ${HOOKFLAG[*]} \\"
  echo "    --thinking-display summarized --dangerously-skip-permissions --output-format stream-json --verbose"
  # This echo referenced ${GRANTS[*]} while GRANTS was defined 47 lines BELOW it,
  # so under `set -u` every dry run this launcher ever ran died right here with
  # "GRANTS[*]: unbound variable" — the one mechanism built so a launch can be
  # inspected without launching had never once completed. Found 2026-08-06 while
  # arming the #186 hook. The definitions now sit above this block.
  exit 0
fi

# ── FIVE-HOUR QUOTA GATE, from a reading this launcher takes ITSELF ──────────
# the maintainer's rule: never launch a NEW evolver above 0.96 five-hour utilization; an
# evolver already running when it crosses is left alone.
#
# scripts/quota_gate.py cannot serve tau3: it reads the last rate_limit_event out
# of the SSB run's evolver logs, and for a tau3 launch every such reading is both
# stale and about a different run. Staleness is the trap — utilization only RISES
# within a window, so a stale LOW reading proves nothing about now, and treating
# it as clear is fail-open on the one gate whose failure kills a run mid-flight
# with nothing written (2026-08-01).
#
# So take a fresh reading here. One word on haiku costs a rounding error against
# the window it is measuring, and a reading this process just took cannot be
# stale by construction.
#
# ABSENT utilization means the window RESET (verified 2026-08-04 05:22: a fresh
# window reports {"status":"allowed"} with no utilization field at all). Absent +
# allowed, from a FRESH probe, is therefore CLEAR — the branch that made this
# unsafe before was accepting absent+allowed from a STALE log.
# #304. THE GATE FOLLOWS THE MODEL, NOT THE RUN. This gate measures the CLAUDE five-hour
# subscription window. An evolver served through a provider gateway (DSV4F, GLM, …) consumes
# none of it, so gating that launch on this reading blocks a run on a reading about a resource
# it never touches — which is exactly what happened to ao6's epoch 1 on 2026-08-13.
#
# THE ANSWER IS COMPUTED IN ONE PLACE, src/orchestrator/quota_scope.py, and this script does not
# carry a second copy of the model-family rule (#147's shape; the shell copy would be the one no
# test can reach). The loop passes the verdict in; a hand launch that does not set it falls back
# to ASKING that module directly, and only if the module itself cannot be reached do we gate
# unconditionally — the conservative direction in every branch.
_GATE_APPLIES=1
if [ -n "${EVOSKILL_EVOLVER_ON_CLAUDE_QUOTA:-}" ]; then
  _GATE_APPLIES="$EVOSKILL_EVOLVER_ON_CLAUDE_QUOTA"
elif python3 -c "import src.orchestrator.quota_scope" 2>/dev/null; then
  if python3 -m src.orchestrator.quota_scope --model "$EVOLVER_MODEL" --quiet; then
    _GATE_APPLIES=1
  else
    _GATE_APPLIES=0
  fi
fi
if [ "$_GATE_APPLIES" = "0" ]; then
  # SAID OUT LOUD, because a gate that silently did not run looks in a log exactly like a gate
  # that ran and passed.
  echo "  quota gate : NOT APPLICABLE — $EVOLVER_MODEL is served through a provider gateway and"
  echo "               consumes credit, not the Claude five-hour window (#304). Spend is bounded"
  echo "               by the run's cost ceiling instead."
fi
if [ -z "${EVOSKILL_TAU3_SKIP_QUOTA_GATE:-}" ] && [ "$_GATE_APPLIES" != "0" ]; then
  # ONE implementation, in scripts/quota_fresh_reading.py, so the launcher does
  # not carry an untestable copy of the same rule. That duplication is defect
  # #147's exact shape and the launcher's copy would be the one that rots,
  # because it is the one no test can reach.
  set +e
  GATE=$(python3 "$ROOT/scripts/quota_fresh_reading.py")
  GATE_RC=$?
  set -e
  echo "  quota gate : $GATE"
  if [ "$GATE_RC" -ne 0 ]; then
    echo "REFUSING to launch: the five-hour quota gate is not clear (rc=$GATE_RC)." >&2
    echo "  Rule: never start a NEW evolver at >= 0.96 utilization — it dies" >&2
    echo "  mid-run and writes nothing (2026-08-01). UNKNOWN blocks like BLOCK:" >&2
    echo "  fail-open on this gate is what that incident was." >&2
    exit 11
  fi
fi


# ── dispatch ────────────────────────────────────────────────────────────────
# --add-dir on the LEAF dirs only. The parent outputs/tau3_banking/ holds
# split.json (val and test ids, plus its SELECTION_WARNING) and every prior leg's
# result JSON — exactly the priors the isolation exists to remove.
#
# LIVE PHOENIX TRACING, opt-in via EVOSKILL_TAU3_PHOENIX_PROJECT. The evolver is a
# headless top-level CC process, so nothing in our harness sits inside it and
# there is no place to hang a tracer. Piping stream-json through the streamer
# gives Phoenix a live span tree while writing the log VERBATIM, so every existing
# consumer of $LOG is unaffected; without the var the launch is byte-unchanged.
#
# The PID recorded is then the PIPELINE's, not claude's — which is what we want:
# caffeinate -w and the liveness triggers should track the whole pipeline, since
# the trace is only complete once the streamer has flushed.
if [ -n "${EVOSKILL_TAU3_PHOENIX_PROJECT:-}" ]; then
  # #221 (2026-08-09): this branch USED to hand the grants to `bash -c` as fixed positional
  # args $3..$6 and refuse any domain whose grant list was not exactly 4 long. That refusal
  # fired on retail's first epoch-1 dispatch: retail ships no policy corpus, so it has 3
  # grants, and live tracing was unavailable for every non-banking domain — i.e. the arms
  # that most needed watching were the ones that could not be watched.
  #
  # Now it passes "${GRANTS[@]}" as trailing args and `shift`s the fixed ones off, so the
  # list is whatever GRANTS says for this domain. That also closes the last copy of the
  # defect noted at the plain dispatch below: the grant set was written out THREE times
  # (dry-run echo, this branch, plain branch) and a grant added to GRANTS reached only one
  # of them. All three now read the same array, so they cannot drift apart again.
  echo "  live Phoenix -> ${EVOSKILL_TAU3_PHOENIX_PROJECT} (${#GRANTS[@]} grant args)"
  # ${12} is the #186 leak-hook settings file — same enforcement as the plain
  # dispatch. Omitting it here would make protection depend on whether Phoenix
  # tracing happened to be on, which is not a property isolation may have.
  # #179 AGAIN, IN THIS BRANCH ONLY (found 2026-08-08). The plain dispatch below cds
  # into $WORK; this one never did. It therefore inherited the CALLER's cwd, so the
  # isolation of a Phoenix-traced evolver depended on which directory the operator
  # happened to launch from — undocumented and invisible. Launching from the repo root
  # put the evolver's cwd at the repo root, where `.claude/skills/` sits and would be
  # loaded into its session: skills the Opus and Sonnet arms never had.
  #
  # The epoch-1 Opus reference run is the ONLY completed run that took this branch, and
  # it is CLEAN — its own session init records a cwd inside the disposable per-job tmp
  # directory (.../tmp/tau3_evolver_e1)
  # and a skills list containing only the built-in user skills. It was launched from a
  # cwd that made the inherited value correct by luck. Nothing already collected is
  # invalidated; the fragility is what is being removed.
  #
  # ${13} is $WORK. The grants and every path below were resolved absolute earlier, so
  # the cd is safe here for the same reason it is safe in the plain branch.
  # ${14} is $EVOLVER_MODEL, PASSED, not inherited. It is assigned at line 50 WITHOUT
  # export, so inside this single-quoted `bash -c` — a fresh process — it expanded to the
  # empty string and CC died instantly with `API Error: 400 No models provided`. The plain
  # branch below is unaffected because the parent shell expands "$EVOLVER_MODEL" there.
  # Latent since this branch was written; it only ever surfaces when the launching shell
  # does not happen to have EVOLVER_MODEL exported. Found 2026-08-08 on the first DSV4F
  # dispatch that combined live Phoenix with an explicit model.
  # Fixed args first, then `shift` them off so whatever remains in "$@" is the grant list.
  # The ${10}-style indices this replaced were the reason the list had to be fixed-length,
  # and they are also how the ${14}/EVOLVER_MODEL bug above stayed invisible for a week.
  # #294: the sandbox prefix travels as its own positional and is applied INSIDE the
  # subshell, after the cd. Passing it as a string to be re-split would reintroduce the
  # word-splitting class of bug the ${10}-indexing comment above is about.
  nohup bash -c '
    _work=$1; _prompt=$2; _sid=$3; _model=$4; _settings=$5
    _root=$6; _log=$7; _proj=$8; _epoch=$9; shift 9
    _promptmd=$1; _sbprof=$2; shift 2
    cd "$_work" || exit 9
    _sb=(); [ -n "$_sbprof" ] && _sb=(sandbox-exec -f "$_sbprof")
    "${_sb[@]}" claude -p "$_prompt" --model "$_model" --session-id "$_sid" \
      "$@" \
      --settings "$_settings" \
      --thinking-display summarized \
      --dangerously-skip-permissions --output-format stream-json --verbose \
    | python3 "$_root/scripts/phoenix_stream_evolver.py" \
        --log "$_log" --project "$_proj" --epoch "$_epoch" --prompt "$_promptmd"
  ' _ "$WORK" "$PROMPT" "$SID" "$EVOLVER_MODEL" "$HOOK_SETTINGS" \
      "$ROOT" "$LOG" "$EVOSKILL_TAU3_PHOENIX_PROJECT" "$ITER" "$WORK_OUT/PROMPT.md" \
      "${_SBPROF:-}" \
      "${GRANTS[@]}" \
      > "$LOG.streamer.stderr" 2>&1 &
else
  # #179: `cd` INTO THE ISOLATED DIR. Without it `claude` inherited the caller's cwd —
  # the repo root — which made the whole repo readable regardless of how narrowly the
  # --add-dir grants were drawn, and put the session under this project's directory.
  # Every path used after this point ($LOG, $TRACE_OUT, $ROOT/scripts/..., the four
  # grants) was already resolved to an absolute path above, so the cd is safe.
  # "${GRANTS[@]}", not a hand-written --add-dir list. The list was spelled out
  # here AND in the Phoenix branch AND echoed from GRANTS in the dry run, so the
  # array's own comment ("One array, expanded everywhere") was false: nothing
  # added to GRANTS reached either real dispatch. Caught while arming the #186
  # hook, whose whole value depends on being on the ACTUAL command line.
  ( cd "$WORK" && exec nohup "${SB[@]}" claude -p "$PROMPT" \
      --model "$EVOLVER_MODEL" \
      --session-id "$SID" \
      "${GRANTS[@]}" "${HOOKFLAG[@]}" \
      --thinking-display summarized \
      --dangerously-skip-permissions \
      --output-format stream-json --verbose \
      > "$LOG" 2>&1 ) &
fi
PID=$!
# Non-negotiable on any background process: the machine must not sleep through it.
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -w $PID &
echo "  PID=$PID  log=$LOG"

# Record the PID so a stray launch is DISCOVERABLE. This launcher backgrounds and
# exits, which means a launch nobody is tracking looks exactly like no launch.
echo "$PID" > "$RUN_DIR/tau3_evolver_e${ITER}.pid"   # RUN_DIR, not a hardcoded path:
# the LOG (343) and TRACE_OUT (403) were parameterised and this one site was missed, so a
# SMOKE run wrote its pid files into the REAL run's directory, where a later reader takes
# them for the real run's evidence that an evolver launched.

# VERIFY THE ISOLATION AGAINST THE RUNNING PROCESS, not against the code that meant to
# create it (#179). The old launcher PRINTED "isolated — no CLAUDE.md, no MEMORY.md" and
# recorded `"cwd": "$WORK"` in the trace while the process actually ran in the repo root;
# both the claim and the record were false for three consecutive epochs and nothing
# noticed, because nothing asked the OS. `lsof -d cwd` asks the OS.
#
# A warning rather than an abort: by the time this runs the evolver is already working,
# and killing a healthy run over a reporting discrepancy costs more than flagging it. It
# is loud, and it names the two values that disagree.
sleep 3
_realcwd=$(lsof -a -p "$PID" -d cwd -Fn 2>/dev/null | grep '^n' | sed 's/^n//' | head -1)
if [ -z "$_realcwd" ]; then
  # The recorded PID may be a wrapper (pipeline/subshell); walk to the claude child.
  # PICK THE `claude` CHILD SPECIFICALLY, not `head -1`. Under live Phoenix the wrapper
  # forks a two-process pipeline (claude | phoenix_stream_evolver.py), so `head -1` was a
  # coin flip on which one got measured — and measuring the streamer while claude ran
  # somewhere else would report isolation VERIFIED that nothing had verified. That is
  # the vacuous-guard shape this check exists to prevent, reproduced inside the check.
  _kid=""
  for _c in $(pgrep -P "$PID" 2>/dev/null); do
    case "$(ps -o comm= -p "$_c" 2>/dev/null)" in
      *claude*) _kid="$_c"; break ;;
    esac
    # Grandchild: `nohup bash -c` puts its own shell between $PID and claude.
    for _g in $(pgrep -P "$_c" 2>/dev/null); do
      case "$(ps -o comm= -p "$_g" 2>/dev/null)" in
        *claude*) _kid="$_g"; break 2 ;;
      esac
    done
  done
  [ -z "$_kid" ] && _kid=$(pgrep -P "$PID" 2>/dev/null | head -1)
  [ -n "$_kid" ] && _realcwd=$(lsof -a -p "$_kid" -d cwd -Fn 2>/dev/null \
    | grep '^n' | sed 's/^n//' | head -1)
fi
if [ -n "$_realcwd" ] && [ "$_realcwd" != "$WORK" ]; then
  echo "  *** ISOLATION WARNING: the evolver's REAL cwd is $_realcwd," >&2
  echo "      not the \$WORK dir this launcher reports ($WORK). Everything under" >&2
  echo "      that cwd is readable regardless of the --add-dir grants. Audit the" >&2
  echo "      session transcript for reads outside the four granted dirs." >&2
elif [ -n "$_realcwd" ]; then
  echo "  isolation : VERIFIED against the running process — cwd is $_realcwd"
else
  echo "  isolation : could not read the process cwd (lsof); NOT verified" >&2
fi
