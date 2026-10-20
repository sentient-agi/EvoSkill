#!/usr/bin/env python3
"""Operator-owned infrastructure config: ONE file per run, `<run_dir>/INFRA.json`.

  * Models, providers, pins, sandboxes and timeouts are operator-owned infrastructure: one config
    file, never written or chosen by the benchmark adaptor.
  * The grader's recalculation timeouts (EVOSKILL_RECALC_TIMEOUT_SECONDS,
    EVOSKILL_DATATABLE_UNO_TIMEOUT_SECONDS; scorer defaults 180 / 300) are declared here. They only
    matter when a recalculation would otherwise time out.

WHAT THIS MODULE DOES
  * load(path) / load_run(run_dir): parse + validate against a CLOSED schema. Every block and key
    is required and nothing else is allowed (keys starting with "_" are comments). A missing key
    never falls back to a default, because a silent default is exactly how a pin stops reaching a
    query.
  * agree(run_dir, env=None) -> [Disagreement]: INFRA.json against everything else that states the
    same fact today -- the frozen SPEC.json, SPEC_EXTRA.json, RUNTIME.json (solver_env, claude,
    evolver_*, daytona_leg with the leg driver's own defaults filled in), the leg driver constants
    (LEG_SETTINGS / RESOURCES / ALLOW_DOMAINS / KNOBS in scripts/daytona/ssb_leg_daytona.py, EXPECTED /
    DEFAULTS in ssb_leg_daytona_loop.py), the Claude sandbox constants (daytona_evolver.py), the
    Claude pins (claude_pins.py), the frozen runner's solver reasoning effort, the Dockerfiles on
    disk, the secret FILES (existence and permissions by stat only -- never opened), and the
    environment. A value already in the shell that differs from INFRA is a disagreement: it is
    never silently overwritten.
  * solver_env / grader_env / claude_env / data_env / derived_env: the environment variables INFRA
    implies, in the exact string form their consumers parse.
  * THE BENCHMARK DATA: block "data" = the canonical dataset root and the
    repaired-file overlay (src/orchestrator/data_overlay.py) as OPERATOR-OWNED config instead of a
    bare EVOSKILL_SSBV2_DATA_OVERLAY. agree() checks the root against the leg driver's DATA and the
    variables daytona_evolver.data_root() reads, re-hashes PATCHES.json against overlay_patches_sha256 and
    re-verifies every entry (data_overlay.verify); the variable joins the derived env, and one set in
    the shell while INFRA declares no overlay (null) is a disagreement too.

The leg driver (scripts/daytona/ssb_leg_daytona_loop.py) reads the grader timeouts from here, puts
them into every leg sandbox's leg.env and records them in runtime_env.knobs; it and the grade driver
(scripts/daytona/ssb_grade_daytona.py) take the data overlay from here
(ssb_leg_daytona_loop.data_overlay_from_infra).

THE CLAUDE CODE SOLVER VARIANT: `solver.harness == "claude_code"` with
`solver.gateway == "anthropic_oauth"` declares a solver that is Claude Code (`claude -p`) on the run's
one credential (claude_code.token_file, fd 3 inside the container; null = ANTHROPIC_API_KEY), NOT a
model behind OpenRouter. For
that variant the OpenRouter pins are empty by rule (provider_order / provider_only / quantizations
[], key_file == claude_code.token_file, expect_served_model == model, reasoning_effort "none" = no
--effort flag: Claude Code's own default thinking effort), RUNTIME.json must declare NO `solver_env`
(the CC leg launcher scripts/ssbv2_leg_cc.sh refuses every EVOSKILL_SOLVER_PROVIDER_* / _EXPECT_* pin
in its environment) and its `leg_script` is that launcher; agree() then skips the comparisons that
state the DeepSeek instrument (LEG_SETTINGS, ENV_FILE / the key variable, the frozen runner's
THINKING_EFFORT) and compares scripts/cc_solver/cc_solver_lib.py's MODEL and MAX_TURNS (with SPEC
agent_turns) instead. An openrouter INFRA validates and agrees exactly as before. `evolver.effort` /
`ao.effort` also accept "default" = no --effort flag (claude_pins' opt-out: EVOSKILL_CLAUDE_EFFORT=
default; the stateful-evolver build requires it).

A TAU3 RUN (SPEC benchmark "tau3", the Claude Code solver through the bridge on the
pool machine built from scripts/daytona/Dockerfile.tau3_leg). agree() reads SPEC.json's benchmark once:
the statements that describe the SSB instrument only -- cc_solver_lib MAX_TURNS vs SPEC agent_turns (the
tau3 bridge passes no --max-turns), ssb_leg_daytona_loop EXPECTED (the SSB leg image) and
the SSB data root (ssb_leg_daytona DATA, daytona_evolver.data_root() and the variables it reads) -- are
compared only for "ssbv2" (or when SPEC.json cannot be read, which is itself named). Everything the machine,
the Claude sandbox and the pins run with, the Dockerfiles on disk, data.root on disk and the overlay are
compared for every benchmark. A leg Dockerfile in LAYERED_LEG_DOCKERFILES (Dockerfile.tau3_leg = the bytes
of Dockerfile.ssb_leg + the /opt/tau3 layer) satisfies the Claude-image base rule through its base, which
agree() re-checks on disk (sha and byte prefix).

STDLIB ONLY (imported by the loop's interpreter, the Daytona SDK venv and the test venv). Other
modules' constants are read with `ast` from their source, never by importing them, so a check has
no import-time side effects and needs no third-party package.

CLI:  python -m src.orchestrator.infra check --run-dir outputs/<run> [--json] [--no-env]
      exit 0 = INFRA.json validates and agrees with everything; 1 = disagreements (listed);
      4 = INFRA.json missing or invalid.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "evoskill-infra/1"
INFRA_FILE = "INFRA.json"
EXIT_OK, EXIT_DISAGREE, EXIT_INVALID = 0, 1, 4

# The env names the two grader timeouts travel under (read by src/agent_profiles/ssbv2_agent/scorer.py at
# import: `int(os.environ.get(<name>, "180"|"300"))`).
RECALC_TIMEOUT_ENV = "EVOSKILL_RECALC_TIMEOUT_SECONDS"
DATATABLE_UNO_TIMEOUT_ENV = "EVOSKILL_DATATABLE_UNO_TIMEOUT_SECONDS"
GRADER_TIMEOUT_KNOBS = (RECALC_TIMEOUT_ENV, DATATABLE_UNO_TIMEOUT_ENV)
# The data overlay's variable (data_overlay.ENV; a test pins the two equal) and the two variables
# daytona_evolver.data_root() reads (EVOSKILL_SSBV2_DATA wins, else SSBV2_ROOT; it has no default).
DATA_OVERLAY_ENV = "EVOSKILL_SSBV2_DATA_OVERLAY"
DATA_ROOT_ENV = "EVOSKILL_SSBV2_DATA"
SSBV2_ROOT_ENV = "SSBV2_ROOT"
SSBV2_DATA_SUFFIX = "data/spreadsheetbench-v2"          # daytona_evolver.data_root(): $SSBV2_ROOT/<this>
# The run's one Claude Code credential (src/orchestrator/cc_account.py): the token file's path, or none and
# ANTHROPIC_API_KEY.
CC_TOKEN_FILE_ENV = "EVOSKILL_CC_TOKEN_FILE"
CC_API_KEY_ENV = "ANTHROPIC_API_KEY"
# The routing variables RUNTIME.json solver_env carries (loop.SOLVER_ENV_KEYS; the model comes from
# the spec there, so it is not one of them).
SOLVER_ROUTING_KEYS = ("EVOSKILL_SOLVER_PROVIDER_ORDER", "EVOSKILL_SOLVER_PROVIDER_ONLY",
                       "EVOSKILL_SOLVER_ALLOW_FALLBACKS", "EVOSKILL_SOLVER_REQUIRE_PARAMETERS",
                       "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS", "EVOSKILL_SOLVER_EXPECT_PROVIDER",
                       "EVOSKILL_SOLVER_EXPECT_MODEL")
# Defaults the loop applies when RUNTIME.json is silent (src/orchestrator/loop.py Ctx); agree()
# compares the EFFECTIVE value, so an absent key that falls back to something else is caught.
LOOP_DEFAULTS = {"prompt_author_model": "opus[1m]", "evolver_host": "local",
                 "evolver_sandbox": True, "evolver_timeout_s": 5400, "leg_timeout_s": 3600}
# Files whose constants agree() reads (repo-relative).
LEG_DRIVER_BASE = "scripts/daytona/ssb_leg_daytona.py"
LEG_DRIVER_LOOP = "scripts/daytona/ssb_leg_daytona_loop.py"
LEG_LIB_LOOP = "scripts/daytona/daytona_lib_loop.py"
LEG_SCRIPT_DAYTONA = "ssbv2_leg_daytona.sh"
# The Claude Code solver's launcher (solver.harness "claude_code") and the module whose constants the
# CC leg runs with (MODEL, MAX_TURNS).
LEG_SCRIPT_CC = "ssbv2_leg_cc.sh"
CC_SOLVER_LIB = "scripts/cc_solver/cc_solver_lib.py"
HARNESS_CLAUDE_CODE, GATEWAY_ANTHROPIC_OAUTH, GATEWAY_OPENROUTER = "claude_code", "anthropic_oauth", "openrouter"
# claude_pins' opt-out spelling accepted for evolver.effort / ao.effort: EVOSKILL_CLAUDE_EFFORT=default
# means NO --effort flag (claude_pins.effort -> None), the one spelling that is not empty and not a
# word ("none", "off") a reader could take for a level.
EFFORT_DEFAULT = "default"
EVOLVER_MODULE = "src/orchestrator/daytona_evolver.py"
PINS_MODULE = "src/orchestrator/claude_pins.py"
RUNNER = "scripts/run_ssbv2_base30_dsv4f_n3.py"
DATA_OVERLAY_MODULE = "src/orchestrator/data_overlay.py"
# The benchmark whose instrument the SSB-only statements above describe (SPEC.json spec.benchmark).
BENCHMARK_SSB = "ssbv2"
# Leg Dockerfile -> the Dockerfile whose bytes it starts with (Dockerfile.tau3_leg = the SSB
# leg Dockerfile + the /opt/tau3 layer; tau3_cc_smoke.dockerfile_meta refuses to build it otherwise). The Claude
# image (claude_sandbox) is built FROM that base, so evolver and solver still share its environment.
LAYERED_LEG_DOCKERFILES = {"scripts/daytona/Dockerfile.tau3_leg": "scripts/daytona/Dockerfile.ssb_leg"}


class InfraError(ValueError):
    """INFRA.json is missing, unreadable or does not validate. `.problems` lists every reason."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class Disagreement:
    source: str          # where the other statement lives: "SPEC.json", "RUNTIME.json", "env", ...
    field: str           # the key there
    infra: object        # what INFRA.json says
    found: object        # what the other source says (or implies, after its own defaults)
    note: str = ""

    def __str__(self) -> str:
        s = f"{self.source}: {self.field}: INFRA={self.infra!r} but found {self.found!r}"
        return s + (f" ({self.note})" if self.note else "")


# ───────────────────────────────────────────────────────────────────────────── schema
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CLAUDE_ID = re.compile(r"claude-[a-z0-9.-]+")      # an explicit id: no alias, no [1m] suffix
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
_NO = object()


def _valid_efforts() -> tuple:
    try:
        from .claude_pins import VALID_EFFORTS      # ONE list of efforts, shared with the pins
        return tuple(VALID_EFFORTS)
    except ImportError:                             # run outside the package: read it, same list
        return tuple(_constants(REPO / PINS_MODULE, ("VALID_EFFORTS",)).get("VALID_EFFORTS", ()))


def _t_str(v):
    return None if isinstance(v, str) and v.strip() else "a non-empty string"


def _t_bool(v):
    return None if isinstance(v, bool) else "true or false"


def _t_int(lo, hi):
    def f(v):
        ok = isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi
        return None if ok else f"an integer in [{lo}, {hi}]"
    return f


def _t_num(lo, hi):
    def f(v):
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi
        return None if ok else f"a number in [{lo}, {hi}]"
    return f


def _t_enum(*vals):
    return lambda v: None if v in vals else f"one of {list(vals)}"


def _t_list():
    def f(v):
        ok = (isinstance(v, list) and v and all(isinstance(x, str) and x.strip() for x in v)
              and len(set(v)) == len(v))
        return None if ok else "a non-empty list of distinct non-empty strings"
    return f


def _t_str_list():
    """A list of distinct non-empty strings, EMPTY allowed (the OpenRouter pins of a Claude Code solver
    are empty by rule; validate() requires them non-empty for gateway openrouter)."""
    def f(v):
        ok = (isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)
              and len(set(v)) == len(v))
        return None if ok else "a list of distinct non-empty strings"
    return f


def _t_sha(v):
    return None if isinstance(v, str) and _SHA256.fullmatch(v) else "a lowercase sha256 hex digest"


def _t_claude(v):
    return None if isinstance(v, str) and _CLAUDE_ID.fullmatch(v) else \
        "an explicit claude-* model id (no alias such as opus[1m], no [1m] suffix)"


def _t_abs(v):
    return None if isinstance(v, str) and v.startswith("/") else "an absolute path"


def _t_opt(t):
    """t, or JSON null (= none declared)."""
    def f(v):
        why = None if v is None else t(v)
        return f"{why} or null" if why else None
    return f


def _t_rel(v):
    return None if isinstance(v, str) and v and not v.startswith("/") and ".." not in v.split("/") \
        else "a repo-relative path"


def _t_env_name(v):
    return None if isinstance(v, str) and _ENV_NAME.fullmatch(v) else "an environment variable NAME"


def _t_effort(v):
    ok = v in _valid_efforts() or v == EFFORT_DEFAULT
    return None if ok else f"one of {list(_valid_efforts()) + [EFFORT_DEFAULT]}"


def _t_resources(v):
    if not isinstance(v, dict) or set(v) != {"cpu", "memory", "disk"}:
        return "an object with exactly cpu, memory, disk"
    bad = [k for k, x in v.items() if not (isinstance(x, int) and not isinstance(x, bool) and x > 0)]
    return f"positive integers ({bad})" if bad else None


SCHEMA_SPEC = {
    "solver": {
        "model": _t_str, "gateway": _t_enum(GATEWAY_OPENROUTER, GATEWAY_ANTHROPIC_OAUTH),
        # the OpenRouter key file (gateway openrouter); a Claude Code solver: claude_code.token_file (null too
        # when the run authenticates with ANTHROPIC_API_KEY)
        "key_file": _t_opt(_t_abs), "key_variable": _t_env_name,
        # the three pin lists: non-empty for gateway openrouter, empty for a Claude Code solver
        # (validate()'s solver cross-rules)
        "provider_order": _t_str_list(), "provider_only": _t_str_list(),
        "allow_fallbacks": _t_bool, "require_parameters": _t_bool, "quantizations": _t_str_list(),
        "expect_provider": _t_str, "expect_served_model": _t_str,
        "reasoning_effort": _t_enum("none", "low", "medium", "high"),
        "harness": _t_enum("direct", "claude_cli", HARNESS_CLAUDE_CODE),
    },
    "evolver": {
        "model": _t_claude, "effort": _t_effort, "harness": _t_enum("claude_cli"),
        "host": _t_enum("daytona", "local"), "sandbox": _t_bool, "timeout_s": _t_int(60, 86400),
    },
    "ao": {"model": _t_claude, "effort": _t_effort, "host": _t_enum("daytona", "local")},
    # token_file: the run's ONE Claude Code credential (src/orchestrator/cc_account.py), a subscription token file;
    # null = no token file, every claude session authenticates with ANTHROPIC_API_KEY from the environment.
    "claude_code": {"version": _t_str, "linux_x64_sha256": _t_sha, "token_file": _t_opt(_t_abs),
                    "no_model_fallback": _t_bool},
    # snapshot_ref (both sandboxes): the ref Daytona derives when it builds the Dockerfile, so it cannot be known
    # before the first build; null = not pinned (every create accepts and records the ref Daytona reports; the
    # Dockerfile's sha256 stays the image's identity), a string = that exact build is required.
    "claude_sandbox": {
        "dockerfile": _t_rel, "dockerfile_sha256": _t_sha, "base_dockerfile_sha256": _t_sha,
        "snapshot_ref": _t_opt(_t_str), "libreoffice": _t_str, "resources": _t_resources,
        "egress": _t_list(), "ttl_minutes": _t_int(10, 1440),
    },
    "leg_sandbox": {
        "host": _t_enum("daytona", "local"), "dockerfile": _t_rel, "dockerfile_sha256": _t_sha,
        "image_id": _t_str, "snapshot_ref": _t_opt(_t_str), "libreoffice": _t_str,
        "grading_openpyxl": _t_str, "resources": _t_resources, "egress": _t_list(),
        "max_sandboxes": _t_int(1, 64), "concurrency": _t_int(1, 30),
        "max_heavy_per_shard": _t_int(1, 30), "ttl_minutes": _t_int(10, 1440),
        "shard_wall_min": _t_int(10, 1440), "leg_timeout_s": _t_int(60, 172800),
    },
    "grader": {"recalc_timeout_s": _t_int(1, 7200), "datatable_uno_timeout_s": _t_int(1, 7200)},
    "data": {"root": _t_abs, "overlay_patches_json": _t_opt(_t_abs),
             "overlay_patches_sha256": _t_opt(_t_sha)},
    "guards": {"leg_spend_cap_usd": _t_num(0.01, 100000), "run_spend_cap_usd": _t_num(0.01, 100000),
               "min_account_balance_usd": _t_num(0, 100000)},
    "other_models": {"user_sim": _t_str, "judge": _t_str},
}
TOP_SCALARS = {"schema": _t_enum(SCHEMA), "run": _t_str}


def validate(obj) -> list:
    """Every problem with a parsed INFRA object ([] = valid). Closed and complete."""
    if not isinstance(obj, dict):
        return ["INFRA.json must be a JSON object"]
    probs = []
    for k in obj:
        if not k.startswith("_") and k not in SCHEMA_SPEC and k not in TOP_SCALARS:
            probs.append(f"unknown top-level key {k!r}")
    for k, t in TOP_SCALARS.items():
        if k not in obj:
            probs.append(f"missing key {k!r}")
        elif t(obj[k]):
            probs.append(f"{k} must be {t(obj[k])}, got {obj[k]!r}")
    for blk, keys in SCHEMA_SPEC.items():
        b = obj.get(blk, _NO)
        if b is _NO:
            probs.append(f"missing block {blk!r}")
            continue
        if not isinstance(b, dict):
            probs.append(f"{blk} must be an object")
            continue
        for k in b:
            if not k.startswith("_") and k not in keys:
                probs.append(f"unknown key {blk}.{k}")
        for k, t in keys.items():
            if k not in b:
                probs.append(f"missing key {blk}.{k}")
                continue
            why = t(b[k])
            if why:
                probs.append(f"{blk}.{k} must be {why}, got {b[k]!r}")
    if probs:
        return probs
    # cross-field rules (only on an otherwise valid object)
    probs.extend(_solver_rules(obj))
    if obj["ao"]["effort"] != obj["evolver"]["effort"]:
        probs.append(f"ao.effort {obj['ao']['effort']!r} != evolver.effort "
                     f"{obj['evolver']['effort']!r}: one variable (EVOSKILL_CLAUDE_EFFORT, "
                     f"claude_pins.EFFORT_ENV) sets the effort of every claude child")
    g = obj["guards"]
    if g["leg_spend_cap_usd"] > g["run_spend_cap_usd"]:
        probs.append("guards.leg_spend_cap_usd exceeds guards.run_spend_cap_usd")
    ls, cs = obj["leg_sandbox"], obj["claude_sandbox"]
    want_id = "x86daytona-" + ls["dockerfile_sha256"][:12]
    if ls["image_id"] != want_id:
        probs.append(f"leg_sandbox.image_id {ls['image_id']!r} != {want_id!r} (the leg driver "
                     f"derives it from the Dockerfile sha)")
    if cs["base_dockerfile_sha256"] != ls["dockerfile_sha256"] and ls["dockerfile"] not in LAYERED_LEG_DOCKERFILES:
        probs.append("claude_sandbox.base_dockerfile_sha256 != leg_sandbox.dockerfile_sha256 (the "
                     "Claude image is built FROM the leg image: evolver and solver share one "
                     "environment)")
    if "daytona" in (obj["evolver"]["host"], obj["ao"]["host"]) and not obj["evolver"]["sandbox"]:
        probs.append("evolver.sandbox false with a daytona host: the loop refuses that contradiction")
    d = obj["data"]
    if (d["overlay_patches_json"] is None) != (d["overlay_patches_sha256"] is None):
        probs.append("data.overlay_patches_json and data.overlay_patches_sha256 must both be null (no "
                     "overlay) or both be set (the sha pins the PATCHES.json the path names)")
    return probs


def solver_is_claude_code(inf: dict) -> bool:
    """True when the run's solver is Claude Code on the OAuth token (solver.harness "claude_code"),
    not a model behind OpenRouter. Decides which statements agree() compares (module doc)."""
    return (inf.get("solver") or {}).get("harness") == HARNESS_CLAUDE_CODE


def _solver_rules(obj: dict) -> list:
    """The solver block's cross-field rules: the two gateways are two shapes and nothing in between.
      openrouter      a model behind OpenRouter: harness direct|claude_cli, the three pin lists non-empty.
      anthropic_oauth harness claude_code only; no OpenRouter pin (the three lists empty), the key
                      file IS the Claude Code token file, the served model is the model itself,
                      reasoning_effort "none" (no --effort flag: Claude Code's default thinking
                      effort), leg host daytona (the CC leg runs in containers
                      on the run's Daytona machine; scripts/ssbv2_leg_cc.sh has no local path)."""
    s, probs = obj["solver"], []
    cc = s["harness"] == HARNESS_CLAUDE_CODE
    if cc and s["gateway"] != GATEWAY_ANTHROPIC_OAUTH:
        probs.append(f"solver.harness {HARNESS_CLAUDE_CODE!r} requires solver.gateway "
                     f"{GATEWAY_ANTHROPIC_OAUTH!r} (Claude Code runs on the OAuth token, not behind "
                     f"OpenRouter), got {s['gateway']!r}")
    if not cc and s["gateway"] == GATEWAY_ANTHROPIC_OAUTH:
        probs.append(f"solver.gateway {GATEWAY_ANTHROPIC_OAUTH!r} is the Claude Code solver's gateway: "
                     f"it requires solver.harness {HARNESS_CLAUDE_CODE!r}, got {s['harness']!r}")
    if s["gateway"] == GATEWAY_OPENROUTER and not s["key_file"]:
        probs.append(f"solver.key_file must be an absolute path for gateway {GATEWAY_OPENROUTER!r} (the file the "
                     f"leg driver reads the OpenRouter key from)")
    if s["gateway"] == GATEWAY_OPENROUTER:
        for k in ("provider_order", "provider_only", "quantizations"):
            if not s[k]:
                probs.append(f"solver.{k} must be a non-empty list of distinct non-empty strings for "
                             f"gateway {GATEWAY_OPENROUTER!r} (the pin every leg runs with)")
    if cc:
        for k in ("provider_order", "provider_only", "quantizations"):
            if s[k]:
                probs.append(f"solver.{k} must be [] for a Claude Code solver (no OpenRouter pin; the "
                             f"leg launcher refuses one), got {s[k]!r}")
        if s["key_file"] != obj["claude_code"]["token_file"]:
            probs.append(f"solver.key_file {s['key_file']!r} != claude_code.token_file "
                         f"{obj['claude_code']['token_file']!r}: a Claude Code solver runs on the "
                         f"run's one credential (the token file, read on fd 3 by the leg driver; both null = "
                         f"ANTHROPIC_API_KEY)")
        if s["expect_served_model"] != s["model"]:
            probs.append(f"solver.expect_served_model {s['expect_served_model']!r} != solver.model "
                         f"{s['model']!r}: Claude Code serves the model it is given (no served-model "
                         f"alias behind a gateway)")
        if s["reasoning_effort"] != "none":
            probs.append(f"solver.reasoning_effort must be 'none' for a Claude Code solver (no --effort "
                         f"flag: Claude Code's default thinking effort; scripts/cc_solver/cc_solver_lib.py "
                         f"builds no such flag), got {s['reasoning_effort']!r}")
        if obj["leg_sandbox"]["host"] != "daytona":
            probs.append(f"leg_sandbox.host must be 'daytona' for a Claude Code solver (the CC leg runs "
                         f"in containers on the run's one Daytona machine), got "
                         f"{obj['leg_sandbox']['host']!r}")
    return probs


def load(path) -> dict:
    """Parse and validate one INFRA.json. Raises InfraError listing every problem."""
    p = Path(path)
    if not p.is_file():
        raise InfraError([f"no {p}"])
    try:
        obj = json.loads(p.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InfraError([f"{p} is not readable JSON: {e}"])
    probs = validate(obj)
    if probs:
        raise InfraError(probs)
    return obj


def load_run(run_dir) -> dict | None:
    """The run's INFRA.json, validated; None when the run has none."""
    f = Path(run_dir) / INFRA_FILE
    return load(f) if f.exists() else None


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ───────────────────────────────────────────────────────────────────────────── derived env
def _b01(v: bool) -> str:
    return "1" if v else "0"


def solver_env(inf: dict) -> dict:
    """The EVOSKILL_SOLVER_* variables, in the form the runner / harness parse (LEG_SETTINGS').
    A Claude Code solver reads ONE of them (EVOSKILL_SOLVER_MODEL, exported by the loop from the spec):
    the routing pins are OpenRouter's and must not exist in its environment (env_conflicts)."""
    s = inf["solver"]
    if solver_is_claude_code(inf):
        return {"EVOSKILL_SOLVER_MODEL": s["model"]}
    return {"EVOSKILL_SOLVER_MODEL": s["model"],
            "EVOSKILL_SOLVER_PROVIDER_ORDER": ",".join(s["provider_order"]),
            "EVOSKILL_SOLVER_PROVIDER_ONLY": ",".join(s["provider_only"]),
            "EVOSKILL_SOLVER_ALLOW_FALLBACKS": _b01(s["allow_fallbacks"]),
            "EVOSKILL_SOLVER_REQUIRE_PARAMETERS": _b01(s["require_parameters"]),
            "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS": ",".join(s["quantizations"]),
            "EVOSKILL_SOLVER_EXPECT_PROVIDER": s["expect_provider"],
            "EVOSKILL_SOLVER_EXPECT_MODEL": s["expect_served_model"]}


def grader_env(inf: dict) -> dict:
    """The two grader timeouts, as the scorer reads them (integer seconds as strings)."""
    g = inf["grader"]
    return {RECALC_TIMEOUT_ENV: str(int(g["recalc_timeout_s"])),
            DATATABLE_UNO_TIMEOUT_ENV: str(int(g["datatable_uno_timeout_s"]))}


def claude_env(inf: dict) -> dict:
    """The variables claude_pins / cc_account / the Daytona evolver read for every claude child. The token file
    travels as its PATH only (EVOSKILL_CC_TOKEN_FILE); none declared = ANTHROPIC_API_KEY from the environment."""
    out = {"EVOSKILL_AO_MODEL": inf["ao"]["model"],
           "EVOSKILL_EVOLVER_MODEL": inf["evolver"]["model"],
           "EVOSKILL_CLAUDE_EFFORT": inf["evolver"]["effort"],
           "EVOSKILL_EXPECT_CLAUDE_CODE_VERSION": inf["claude_code"]["version"]}
    if inf["claude_code"]["token_file"]:
        out[CC_TOKEN_FILE_ENV] = inf["claude_code"]["token_file"]
    return out


def data_env(inf: dict) -> dict:
    """EVOSKILL_SSBV2_DATA_OVERLAY when INFRA declares an overlay; {} when it declares none (null)."""
    pj = inf["data"]["overlay_patches_json"]
    return {DATA_OVERLAY_ENV: pj} if pj else {}


def derived_env(inf: dict) -> dict:
    return {**solver_env(inf), **grader_env(inf), **claude_env(inf), **data_env(inf)}


# How each derived variable is compared with a value found elsewhere (the consumer's own parsing).
def _kind(name: str) -> str:
    if name in ("EVOSKILL_SOLVER_PROVIDER_ORDER", "EVOSKILL_SOLVER_PROVIDER_ONLY",
                "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS"):
        return "list"
    if name == "EVOSKILL_SOLVER_ALLOW_FALLBACKS":
        return "bool"            # loop._export_solver_env: anything but 0/""/false/False is true
    if name == "EVOSKILL_SOLVER_REQUIRE_PARAMETERS":
        return "flag1"           # options.provider_extra_from_env: only "1" is true
    if name in GRADER_TIMEOUT_KNOBS:
        return "int"
    if name == "EVOSKILL_CLAUDE_EFFORT":
        return "effort"          # claude_pins.effort: strip + lower
    if name in (DATA_OVERLAY_ENV, CC_TOKEN_FILE_ENV):
        return "path"            # data_overlay._key / cc_account: a path, compared normalised
    return "str"


def _norm(kind: str, v):
    s = "" if v is None else str(v)
    if kind == "list":
        return tuple(x.strip() for x in s.split(",") if x.strip())
    if kind == "bool":
        return s.strip() not in ("0", "", "false", "False")
    if kind == "flag1":
        return s.strip() == "1"
    if kind == "int":
        try:
            return int(s.strip())
        except ValueError:
            return s
    if kind == "effort":
        return s.strip().lower()
    if kind == "path":
        return os.path.normpath(os.path.abspath(s)) if s else ""
    return s.strip()


def same_env_value(name: str, a, b) -> bool:
    k = _kind(name)
    return _norm(k, a) == _norm(k, b)


def env_conflicts(inf: dict, env=None) -> list:
    """A variable INFRA implies that is ALREADY in `env` (default os.environ) with another value.
    Unset is not a conflict: the loop / leg driver set these from INFRA. The data overlay's variable
    set (non-empty) while INFRA declares no overlay IS one: data_overlay would substitute bytes the
    operator did not declare. Empty counts as unset for it (data_overlay reads "" as no overlay)."""
    env = os.environ if env is None else env
    out = []
    for name, want in derived_env(inf).items():
        if name == DATA_OVERLAY_ENV and not env.get(name):
            continue
        if name in env and not same_env_value(name, want, env[name]):
            out.append(Disagreement("env", name, want, env[name],
                                    "a conflicting value in the shell is not overwritten; unset "
                                    "it or change INFRA.json"))
    if not inf["claude_code"]["token_file"] and env.get(CC_TOKEN_FILE_ENV):
        out.append(Disagreement("env", CC_TOKEN_FILE_ENV, None, env[CC_TOKEN_FILE_ENV],
                                "INFRA.json declares no token file (claude_code.token_file null: the run "
                                "authenticates with ANTHROPIC_API_KEY); unset it or declare the token file"))
    if not data_env(inf) and env.get(DATA_OVERLAY_ENV):
        out.append(Disagreement("env", DATA_OVERLAY_ENV, None, env[DATA_OVERLAY_ENV],
                                "INFRA.json declares no data overlay (data.overlay_patches_json "
                                "null); unset it or declare the overlay in INFRA.json"))
    if solver_is_claude_code(inf):
        # a DeepSeek leftover in the shell: the CC leg launcher refuses every one of these pins, so a
        # drive started from such a shell would HALT at its first leg.
        for name in SOLVER_ROUTING_KEYS:
            if env.get(name):
                out.append(Disagreement("env", name, None, env[name],
                                        "a Claude Code solver reads no OpenRouter pin (the leg "
                                        "launcher scripts/ssbv2_leg_cc.sh refuses it); unset it"))
    return out


# ───────────────────────────────────────────────────────────────────────────── other sources
_UNKNOWN = object()


def _eval_node(node, known: dict):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        pass
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
            and node.value.id in known and isinstance(node.slice, ast.Constant):
        try:
            return known[node.value.id][node.slice.value]
        except (KeyError, TypeError, IndexError):
            return _UNKNOWN
    if isinstance(node, ast.Dict):                  # partially literal: keep what is
        out = {}
        for k, v in zip(node.keys, node.values):
            kk = _eval_node(k, known) if k is not None else _UNKNOWN
            vv = _eval_node(v, known)
            if kk is not _UNKNOWN and vv is not _UNKNOWN:
                out[kk] = vv
        return out
    return _UNKNOWN


def _constants(path: Path, names: tuple) -> dict:
    """Top-level NAME = <literal> assignments of a Python file, read with ast (never imported).
    A dict is kept partially when some values are not literals (e.g. str(REPO / ...))."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = [t.id for t in node.targets if isinstance(t, ast.Name)], node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            targets, value = [node.target.id], node.value
        else:
            continue
        for t in targets:
            if t in names:
                v = _eval_node(value, out)
                if v is not _UNKNOWN:
                    out[t] = v
    return out


def _read_json(p: Path):
    try:
        return json.loads(p.read_text()), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return None, f"unreadable: {e}"


class _Collector:
    def __init__(self):
        self.items: list = []

    def eq(self, source, field, infra, found, note=""):
        if infra != found:
            self.items.append(Disagreement(source, field, infra, found, note))

    def add(self, source, field, infra, found, note=""):
        self.items.append(Disagreement(source, field, infra, found, note))


def _secret_file(c: _Collector, what: str, path: str, *, private: bool) -> None:
    """Existence (and, with `private`, owner-only permissions) by stat ONLY. Never opened."""
    p = Path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        c.add("file", what, path, "missing")
        return
    except OSError as e:
        c.add("file", what, path, f"stat failed: {e}")
        return
    if not p.is_file():
        c.add("file", what, path, "not a regular file")
    elif private and st.st_mode & 0o077:
        c.add("file", what, path, oct(st.st_mode & 0o777),
              "readable or writable by group/other; the loop refuses such a token file")


def _benchmark(run_dir: Path):
    """SPEC.json's spec.benchmark, read once by agree(); None when SPEC.json is missing or unreadable (_check_spec
    names the file then, and every SSB statement is compared as before)."""
    raw, err = _read_json(run_dir / "SPEC.json")
    sp = raw.get("spec", raw) if isinstance(raw, dict) else None
    return sp.get("benchmark") if isinstance(sp, dict) and not err else None


def _ssb(benchmark) -> bool:
    """The SSB-only statements (MAX_TURNS, EXPECTED, the SSB data root) bind this run: ssbv2, or SPEC unreadable."""
    return benchmark in (BENCHMARK_SSB, None)


def _check_spec(c: _Collector, inf: dict, run_dir: Path, repo: Path = REPO, benchmark=BENCHMARK_SSB) -> None:
    raw, err = _read_json(run_dir / "SPEC.json")
    if err:
        c.add("SPEC.json", "(file)", "present", err)
        return
    sp = raw.get("spec", raw) if isinstance(raw, dict) else {}
    s, e = inf["solver"], inf["evolver"]
    if solver_is_claude_code(inf) and _ssb(benchmark):
        # the turn budget the CC leg runs with (--max-turns) is the spec's agent_turns (SSB only: the tau3 bridge
        # passes no --max-turns)
        got = _cc_constants(repo).get("MAX_TURNS")
        c.eq(f"{CC_SOLVER_LIB} MAX_TURNS", "agent_turns", sp.get("agent_turns"), got,
             "SPEC agent_turns vs the --max-turns the Claude Code leg passes")
    c.eq("SPEC.json", "solver_model", s["model"], sp.get("solver_model"))
    c.eq("SPEC.json", "solver_provider_order", s["provider_order"],
         list(sp.get("solver_provider_order") or []))
    c.eq("SPEC.json", "solver_allow_fallbacks", s["allow_fallbacks"], sp.get("solver_allow_fallbacks"))
    c.eq("SPEC.json", "solver_harness", s["harness"], sp.get("solver_harness"))
    c.eq("SPEC.json", "evolver_model", e["model"], sp.get("evolver_model"))
    c.eq("SPEC.json", "evolver_harness", e["harness"], sp.get("evolver_harness"))
    c.eq("SPEC.json", "user_sim_model", inf["other_models"]["user_sim"], sp.get("user_sim_model"))
    c.eq("SPEC.json", "judge_model", inf["other_models"]["judge"], sp.get("judge_model"))
    ceil = sp.get("cost_ceiling_usd")
    run_cap = inf["guards"]["run_spend_cap_usd"]
    if not isinstance(ceil, (int, float)) or run_cap > ceil:
        c.add("SPEC.json", "cost_ceiling_usd", f"guards.run_spend_cap_usd={run_cap}", ceil,
              "the run spend cap must stay at or below the plan's cost ceiling")


def _check_spec_extra(c: _Collector, inf: dict, run_dir: Path) -> None:
    raw, err = _read_json(run_dir / "SPEC_EXTRA.json")
    if err == "missing":
        return
    if err:
        c.add("SPEC_EXTRA.json", "(file)", "readable", err)
        return
    s, e = inf["solver"], inf["evolver"]
    for k, want in (("solver_provider_order", s["provider_order"]),
                    ("solver_allow_fallbacks", s["allow_fallbacks"]),
                    ("solver_harness", s["harness"]), ("evolver_harness", e["harness"])):
        if k in raw:
            got = list(raw[k]) if k == "solver_provider_order" and isinstance(raw[k], list) else raw[k]
            c.eq("SPEC_EXTRA.json", k, want, got)


def _check_runtime(c: _Collector, inf: dict, run_dir: Path, loop_defaults: dict) -> None:
    rt, err = _read_json(run_dir / "RUNTIME.json")
    if err:
        c.add("RUNTIME.json", "(file)", "present", err)
        return
    S = "RUNTIME.json"
    cc = solver_is_claude_code(inf)
    if cc:
        # a Claude Code solver: NO solver_env block at all (the loop would export OpenRouter pins the
        # CC leg launcher refuses)
        if "solver_env" in rt:
            c.add(S, "solver_env", None, rt["solver_env"],
                  "solver_env declared for a Claude Code solver: the leg reads no OpenRouter pin")
        decl, want = {}, {}
    else:
        # solver_env: exactly the INFRA-derived routing variables
        decl = rt.get("solver_env") or {}
        want = {k: v for k, v in solver_env(inf).items() if k in SOLVER_ROUTING_KEYS}
    for k, v in want.items():
        if k not in decl:
            c.add(S, f"solver_env.{k}", v, None, "not declared: the leg would not be told")
        elif not same_env_value(k, v, decl[k]):
            c.add(S, f"solver_env.{k}", v, decl[k])
    for k in decl:
        if k not in want:
            c.add(S, f"solver_env.{k}", None, decl[k], "not a variable INFRA.json derives")

    def eff(key):
        return rt[key] if key in rt else LOOP_DEFAULTS[key]

    c.eq(S, "prompt_author_model", inf["ao"]["model"], eff("prompt_author_model"),
         "" if "prompt_author_model" in rt else "absent: the loop's default applies")
    c.eq(S, "evolver_host", inf["evolver"]["host"], eff("evolver_host"))
    c.eq(S, "evolver_sandbox", inf["evolver"]["sandbox"], bool(eff("evolver_sandbox")))
    # loop.main: RUNTIME evolver_timeout_s is read only for a daytona host; otherwise 5400.
    ev_t = int(rt["evolver_timeout_s"]) if (rt.get("evolver_timeout_s") and
                                            rt.get("evolver_host") == "daytona") \
        else LOOP_DEFAULTS["evolver_timeout_s"]
    c.eq(S, "evolver_timeout_s", inf["evolver"]["timeout_s"], ev_t)
    c.eq(S, "leg_timeout_s", inf["leg_sandbox"]["leg_timeout_s"],
         int(rt["leg_timeout_s"]) if rt.get("leg_timeout_s") else LOOP_DEFAULTS["leg_timeout_s"])
    ls = rt.get("leg_script") or ""
    if cc:
        # the loop picks the leg driver by this path alone (Ctx.leg_script): a Claude Code solver runs
        # through scripts/ssbv2_leg_cc.sh, never the DeepSeek launcher
        if Path(ls).name != LEG_SCRIPT_CC:
            c.add(S, "leg_script", f"(solver harness claude_code) .../{LEG_SCRIPT_CC}", ls or None)
    elif inf["leg_sandbox"]["host"] == "daytona" and Path(ls).name != LEG_SCRIPT_DAYTONA:
        c.add(S, "leg_script", f"(leg host daytona) .../{LEG_SCRIPT_DAYTONA}", ls or None)
    if inf["leg_sandbox"]["host"] == "local" and Path(ls).name == LEG_SCRIPT_DAYTONA:
        c.add(S, "leg_script", "(leg host local)", ls)
    # the Claude sandbox (the AO and the evolver run on one, from the same config block)
    if "daytona" in (inf["evolver"]["host"], inf["ao"]["host"]):
        ed = rt.get("evolver_daytona")
        if not isinstance(ed, dict):
            c.add(S, "evolver_daytona", "an object", ed)
        else:
            cs = inf["claude_sandbox"]
            # oauth_token_file may be left out: the driver then takes the run's one credential from the
            # environment (cc_account.resolve: EVOSKILL_CC_TOKEN_FILE, which the loop sets from INFRA).
            if "oauth_token_file" in ed:
                c.eq(S, "evolver_daytona.oauth_token_file", inf["claude_code"]["token_file"],
                     ed.get("oauth_token_file"))
            for k, v in (("ttl_minutes", cs["ttl_minutes"]),
                         ("require_libreoffice", cs["libreoffice"]),
                         ("require_snapshot_ref", cs["snapshot_ref"])):
                c.eq(S, f"evolver_daytona.{k}", v, ed.get(k))
    # the documentation block of the Claude pins (if the run has one, it must say the same)
    cl = rt.get("claude")
    if isinstance(cl, dict):
        for k, v in (("ao_model", inf["ao"]["model"]),
                     ("evolver_model_from_spec", inf["evolver"]["model"]),
                     ("effort", inf["evolver"]["effort"]),
                     ("expect_claude_code_version", inf["claude_code"]["version"])):
            if k in cl:
                c.eq(S, f"claude.{k}", v, cl[k])
    # the leg driver's block, with ITS defaults filled in (ssb_leg_daytona_loop.load_runtime_block)
    if inf["leg_sandbox"]["host"] == "daytona":
        blk = rt.get("daytona_leg") or {}
        if not isinstance(blk, dict):
            c.add(S, "daytona_leg", "an object", blk)
            blk = {}
        l, g = inf["leg_sandbox"], inf["guards"]
        for k, v in (("max_sandboxes", l["max_sandboxes"]), ("concurrency", l["concurrency"]),
                     ("max_heavy_per_shard", l["max_heavy_per_shard"]),
                     ("ttl_minutes", l["ttl_minutes"]), ("shard_wall_min", l["shard_wall_min"]),
                     ("require_snapshot_ref", l["snapshot_ref"]),
                     ("require_libreoffice", l["libreoffice"]),
                     ("leg_spend_cap_usd", g["leg_spend_cap_usd"]),
                     ("run_spend_cap_usd", g["run_spend_cap_usd"]),
                     ("min_account_balance_usd", g["min_account_balance_usd"])):
            got = blk[k] if k in blk else loop_defaults.get(k, "(no default)")
            note = "" if k in blk else "absent: the leg driver's DEFAULTS apply"
            if isinstance(v, (int, float)) and isinstance(got, (int, float)) \
                    and not isinstance(got, bool):
                if float(v) != float(got):
                    c.add(S, f"daytona_leg.{k}", v, got, note)
            else:
                c.eq(S, f"daytona_leg.{k}", v, got, note)


def _cc_constants(repo: Path) -> dict:
    """MODEL / MAX_TURNS of scripts/cc_solver/cc_solver_lib.py (ast, never imported); {} when the file
    is absent (then every comparison against it is a named disagreement with found None)."""
    p = repo / CC_SOLVER_LIB
    return _constants(p, ("MODEL", "MAX_TURNS")) if p.is_file() else {}


def _check_code(c: _Collector, inf: dict, repo: Path, benchmark=BENCHMARK_SSB) -> None:
    """The constants the leg driver, the Claude sandbox driver, the pins and the runner RUN WITH.
    For a Claude Code solver the statements of the DeepSeek instrument (LEG_SETTINGS, ENV_FILE and the
    key variable, the frozen runner's THINKING_EFFORT) are not compared -- that solver never runs
    them -- and cc_solver_lib's MODEL is compared instead; everything the machine, the grader and the
    Claude sandbox run with (RESOURCES, ALLOW_DOMAINS, KNOBS, EXPECTED, the CC pins, the Dockerfiles)
    is compared for both. EXPECTED (ssb_leg_daytona_loop: the SSB leg image) only for an SSB run; a
    layered leg Dockerfile (LAYERED_LEG_DOCKERFILES) has its base re-checked on disk for every benchmark."""
    s, ls, cs, cc = inf["solver"], inf["leg_sandbox"], inf["claude_sandbox"], inf["claude_code"]
    is_cc = solver_is_claude_code(inf)
    lb = _constants(repo / LEG_DRIVER_BASE, ("LEG_SETTINGS", "RESOURCES", "ALLOW_DOMAINS", "KNOBS"))
    src = LEG_DRIVER_BASE
    if is_cc:
        # The Claude Code leg passes the run's SPEC solver_model as --model (cc_leg_driver LegBox.for_leg);
        # cc_solver_lib.MODEL is only its default, and SPEC.solver_model == INFRA solver.model is checked above.
        pass
    else:
        legset = lb.get("LEG_SETTINGS") or {}
        for k, v in solver_env(inf).items():
            if not same_env_value(k, v, legset.get(k)):
                c.add(f"{src} LEG_SETTINGS", k, v, legset.get(k))
    c.eq(f"{src} RESOURCES", "RESOURCES", ls["resources"], lb.get("RESOURCES"))
    c.eq(f"{src} ALLOW_DOMAINS", "ALLOW_DOMAINS", ",".join(ls["egress"]), lb.get("ALLOW_DOMAINS"))
    knobs = lb.get("KNOBS") or {}
    for k, v in grader_env(inf).items():         # the baseline's KNOBS must not pin another value
        if k in knobs and not same_env_value(k, v, knobs[k]):
            c.add(f"{src} KNOBS", k, v, knobs[k])
    if _ssb(benchmark):                          # the SSB leg image the SSB leg driver pins
        ll = _constants(repo / LEG_DRIVER_LOOP, ("EXPECTED",))
        exp = ll.get("EXPECTED") or {}
        src = f"{LEG_DRIVER_LOOP} EXPECTED"
        c.eq(src, "dockerfile_sha256", ls["dockerfile_sha256"], exp.get("dockerfile_sha256"))
        c.eq(src, "snapshot_ref", ls["snapshot_ref"], exp.get("snapshot_ref"))
        c.eq(src, "image_id", ls["image_id"], exp.get("image_id"))
        c.eq(src, "libreoffice", ls["libreoffice"], exp.get("libreoffice"))
        c.eq(src, "grading_openpyxl_prefix", ls["grading_openpyxl"], exp.get("grading_openpyxl_prefix"))
    if not is_cc:
        lib = _constants(repo / LEG_LIB_LOOP, ("ENV_FILE",))
        c.eq(f"{LEG_LIB_LOOP} ENV_FILE", "ENV_FILE", s["key_file"], lib.get("ENV_FILE"),
             "the file the leg driver reads the OpenRouter key from")
        if f'"{s["key_variable"]}"' not in (repo / LEG_LIB_LOOP).read_text():
            c.add(LEG_LIB_LOOP, "key variable", s["key_variable"], "not read there")
    ev = _constants(repo / EVOLVER_MODULE,
                    ("ALLOW_DOMAINS", "RESOURCES", "BASE_DOCKERFILE_SHA", "CC_VERSION",
                     "CC_LINUX_X64_SHA"))
    src = EVOLVER_MODULE
    _ccv = os.environ.get("EVOSKILL_CC_VERSION")
    if _ccv:      # a run pinned to another verified release compares INFRA with that release's entry
        _rel = (_constants(repo / "scripts/daytona/evolver_container_ctl.py", ("CC_RELEASES",)).get("CC_RELEASES") or {}).get(_ccv)
        src = f"scripts/daytona/evolver_container_ctl.py CC_RELEASES[{_ccv}] (EVOSKILL_CC_VERSION)"
        ev = {**ev, "CC_VERSION": _ccv if _rel else None, "CC_LINUX_X64_SHA": _rel[0] if _rel else None}
    c.eq(src, "CC_VERSION", cc["version"], ev.get("CC_VERSION"))
    c.eq(src, "CC_LINUX_X64_SHA", cc["linux_x64_sha256"], ev.get("CC_LINUX_X64_SHA"))
    c.eq(src, "ALLOW_DOMAINS", ",".join(cs["egress"]), ev.get("ALLOW_DOMAINS"))
    c.eq(src, "RESOURCES", cs["resources"], ev.get("RESOURCES"))
    c.eq(src, "BASE_DOCKERFILE_SHA", cs["base_dockerfile_sha256"], ev.get("BASE_DOCKERFILE_SHA"))
    pins = _constants(repo / PINS_MODULE, ("PIN_ENV_ALWAYS",)).get("PIN_ENV_ALWAYS") or {}
    c.eq(PINS_MODULE, "PIN_ENV_ALWAYS.CLAUDE_CODE_NO_MODEL_FALLBACK",
         "1" if cc["no_model_fallback"] else None, pins.get("CLAUDE_CODE_NO_MODEL_FALLBACK"))
    if not is_cc:                                # the frozen DeepSeek runner is not the CC solver
        rn = _constants(repo / RUNNER, ("THINKING_EFFORT",))
        te = rn.get("THINKING_EFFORT", _UNKNOWN)
        want_te = "" if s["reasoning_effort"] == "none" else s["reasoning_effort"]
        if te is _UNKNOWN or te != want_te:
            c.add(RUNNER, "THINKING_EFFORT", want_te, None if te is _UNKNOWN else te,
                  "the frozen runner sends this reasoning effort (\"\" = none)")
    for what, rel, want in (("leg_sandbox.dockerfile", ls["dockerfile"], ls["dockerfile_sha256"]),
                            ("claude_sandbox.dockerfile", cs["dockerfile"], cs["dockerfile_sha256"])):
        p = repo / rel
        got = sha256_file(p) if p.is_file() else "missing"
        c.eq("disk", f"sha256({rel})", want, got, f"{what} as built from this checkout")
    base_rel = LAYERED_LEG_DOCKERFILES.get(ls["dockerfile"])
    if base_rel:
        # the layered leg image's base is the image the Claude image is built FROM (validate() defers that rule here)
        bp, lp = repo / base_rel, repo / ls["dockerfile"]
        got = sha256_file(bp) if bp.is_file() else "missing"
        c.eq("disk", f"sha256({base_rel})", cs["base_dockerfile_sha256"], got,
             f"the base of {ls['dockerfile']} (claude_sandbox.base_dockerfile_sha256)")
        if bp.is_file() and lp.is_file() and not lp.read_bytes().startswith(bp.read_bytes()):
            c.add("disk", f"{ls['dockerfile']} prefix", f"the bytes of {base_rel}", "differs",
                  "a layered leg Dockerfile starts with its base's bytes")


def _data_overlay():
    try:
        from . import data_overlay                  # stdlib only, no import-time side effects
        return data_overlay
    except ImportError:                             # run outside the package: the same file, by path
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_overlay", REPO / DATA_OVERLAY_MODULE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _npath(v):
    return os.path.normpath(v) if isinstance(v, str) and v else v


def _path_constant(path: Path, name: str):
    """The last top-level `NAME = Path("<literal>")` (or `NAME = "<literal>"`) of a Python file, read
    with ast; None when NAME is assigned anything else (or not at all)."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    out = None
    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)):
            continue
        v, out = node.value, None
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "Path" \
                and len(v.args) == 1 and not v.keywords:
            v = v.args[0]
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out = v.value
    return out


def _check_data(c: _Collector, inf: dict, repo: Path, env, benchmark=BENCHMARK_SSB) -> None:
    """Block "data": the root every other statement uses, and the overlay's PATCHES.json (pinned by
    sha, every entry re-verified now). `env` None = the environment is not compared. The SSB data
    root's other statements (the leg driver's DATA when it is a literal, and the variables
    daytona_evolver.data_root() reads -- it has no default; the loop exports data.root as
    EVOSKILL_SSBV2_DATA) only for an SSB run; the root on disk and the overlay for every benchmark."""
    d = inf["data"]
    root = d["root"]
    if _ssb(benchmark):
        leg = _path_constant(repo / LEG_DRIVER_BASE, "DATA") if (repo / LEG_DRIVER_BASE).is_file() else None
        if leg is not None and _npath(leg) != _npath(root):
            c.add(f"{LEG_DRIVER_BASE} DATA", "DATA", root, leg,
                  "the tree every leg and grading sandbox mirrors (ssb_leg_daytona_loop / ssb_grade_daytona "
                  "reuse it)")
    if not Path(root).is_dir():
        c.add("disk", "data.root", root, "missing")
    if env is not None and _ssb(benchmark):
        if env.get(DATA_ROOT_ENV):
            if _npath(env[DATA_ROOT_ENV]) != _npath(root):
                c.add("env", DATA_ROOT_ENV, root, env[DATA_ROOT_ENV],
                      "daytona_evolver.data_root() reads this first")
        elif env.get(SSBV2_ROOT_ENV):
            implied = os.path.join(env[SSBV2_ROOT_ENV], SSBV2_DATA_SUFFIX)
            if _npath(implied) != _npath(root):
                c.add("env", SSBV2_ROOT_ENV, root, implied,
                      "daytona_evolver.data_root() reads this root when EVOSKILL_SSBV2_DATA is unset")
    pj = d["overlay_patches_json"]
    if pj is None:
        return
    p = Path(pj)
    got = sha256_file(p) if p.is_file() else "missing"
    c.eq("file", "sha256(data.overlay_patches_json)", d["overlay_patches_sha256"], got,
         "the PATCHES.json on disk now")
    if got == "missing":
        return
    do = _data_overlay()
    try:
        do.verify(p, root=root)
    except do.OverlayError as e:
        c.add("file", "data.overlay_patches_json entries", "every entry verified", str(e),
              "data_overlay.verify: canonical == sha256_before, patched == sha256_after")
    rec, err = _read_json(p)
    src = rec.get("src", root) if isinstance(rec, dict) else err
    # src null = the run's data root (data_overlay.trees), so only an explicit src can name another tree
    if src is not None and _npath(src) != _npath(root):
        c.add("PATCHES.json", "src", root, src, "the overlay must patch the tree data.root declares; "
              "otherwise no canonical path matches and nothing is substituted")


def agree(run_dir, env=None, *, repo=None, check_env: bool = True) -> list:
    """Every disagreement between <run_dir>/INFRA.json and the other statements of the same facts.
    [] = they all agree. Raises InfraError when INFRA.json is missing or invalid."""
    run_dir = Path(run_dir)
    repo = Path(repo) if repo else REPO
    inf = load(run_dir / INFRA_FILE)
    c = _Collector()
    c.eq("run dir", "name", inf["run"], run_dir.resolve().name,
         "an INFRA.json copied from another run must be re-declared, not reused")
    defaults = _constants(repo / LEG_DRIVER_LOOP, ("EXPECTED", "DEFAULTS")).get("DEFAULTS") or {}
    bm = _benchmark(run_dir)                     # once: which statements describe this run's instrument
    _check_spec(c, inf, run_dir, repo, bm)
    _check_spec_extra(c, inf, run_dir)
    _check_runtime(c, inf, run_dir, defaults)
    _check_code(c, inf, repo, bm)
    _check_data(c, inf, repo, (os.environ if env is None else env) if check_env else None, bm)
    # The token file must be owner-only (loop._evolver_host refuses otherwise); the key file only
    # has to exist (the leg driver reads it in-process; its mode is not a rule anywhere today).
    if inf["claude_code"]["token_file"]:
        _secret_file(c, "claude_code.token_file", inf["claude_code"]["token_file"], private=True)
    elif check_env and not ((os.environ if env is None else env).get(CC_API_KEY_ENV) or "").strip():
        c.add("env", CC_API_KEY_ENV, "set (claude_code.token_file is null)", None,
              "no Claude Code credential: declare a token file or export ANTHROPIC_API_KEY")
    if inf["solver"]["key_file"]:
        _secret_file(c, "solver.key_file", inf["solver"]["key_file"], private=False)
    if check_env:
        c.items.extend(env_conflicts(inf, env))
    return c.items


def check(run_dir, env=None, *, repo=None, check_env: bool = True) -> dict:
    """agree() as a record (for the CLI and the loop): never raises."""
    f = Path(run_dir) / INFRA_FILE
    rec = {"infra_file": str(f), "infra_sha256": sha256_file(f) if f.is_file() else None,
           "valid": False, "problems": [], "disagreements": [], "ok": False}
    try:
        items = agree(run_dir, env, repo=repo, check_env=check_env)
    except InfraError as e:
        rec["problems"] = e.problems
        return rec
    rec.update(valid=True, disagreements=[asdict(d) | {"text": str(d)} for d in items],
               ok=not items)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.orchestrator.infra",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="validate INFRA.json and list every disagreement")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--json", action="store_true", help="print the record as JSON")
    p.add_argument("--no-env", action="store_true", help="skip the environment comparison")
    a = ap.parse_args(argv)
    rec = check(a.run_dir, check_env=not a.no_env)
    if a.json:
        print(json.dumps(rec, indent=1, default=str))
    elif not rec["valid"]:
        print(f"INVALID {rec['infra_file']}:")
        for pr in rec["problems"]:
            print(f"  - {pr}")
    elif rec["ok"]:
        print(f"AGREE {rec['infra_file']} (sha256 {rec['infra_sha256'][:12]}): SPEC, SPEC_EXTRA, "
              f"RUNTIME, leg driver, Claude sandbox, pins, runner, Dockerfiles, data root and "
              f"overlay, secret files"
              f"{'' if a.no_env else ', environment'} all say the same")
    else:
        print(f"DISAGREE {rec['infra_file']} (sha256 {rec['infra_sha256'][:12]}): "
              f"{len(rec['disagreements'])} difference(s)")
        for d in rec["disagreements"]:
            print(f"  - {d['text']}")
    if not rec["valid"]:
        return EXIT_INVALID
    return EXIT_OK if rec["ok"] else EXIT_DISAGREE


if __name__ == "__main__":
    sys.exit(main())
