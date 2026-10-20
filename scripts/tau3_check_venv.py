#!/usr/bin/env python3
"""Compare a tau3-bench venv's package list with the pinned list (tau3_venv_freeze.txt).

Strict replay needs the rollouts and the grading to run the same tau2 code, so every venv that runs or grades a tau3
conversation (a local third_party/tau3-bench/.venv and every image built for tau3) must hold the same packages. The
pinned list is scripts/tau3_env/tau3_venv_freeze.txt; its header names the command that builds such a venv.

  tau3_check_venv.py --python PY    [--pin PIN] [--print]   the venv of interpreter PY lists its own packages
  tau3_check_venv.py --freeze FILE  [--pin PIN]             a list printed elsewhere (FILE '-' = stdin)

--python runs PY once, isolated (-I), with PYTHON_DOTENV_DISABLED=1 and LITELLM_LOCAL_MODEL_COST_MAP=True in its
environment. The child reads the dist-info of PY's own site-packages with importlib.metadata and prints name and version
of each; it never imports tau2 and needs neither uv nor pip (an image may have removed both). An editable install is
listed by its name and version, the way `uv pip list --format freeze` lists it, so the checkout's location never counts.
--freeze takes lines in the same form; `uv pip freeze` prints an editable as `-e file:///...`, which has no name and is
reported as a difference (use `uv pip list --format freeze`, or --python).

Names compare in PEP 503 form (lower case, runs of -_. as one -). Lines starting with # are comments, except that the
pin's `# python: CPython 3.12.13` line is compared with PY's interpreter when --python is given. --print writes PY's list
in the pin's form to stdout and compares nothing.

rc 0: the same packages at the same versions (and the same interpreter). rc 1: differences, one line each on stdout.
rc 2: the pin or the list cannot be read (a missing file, a missing interpreter, a malformed or duplicate pin line).
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "scripts" / "tau3_env" / "tau3_venv_freeze.txt"
LINE_RE = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(===?|@)\s*(\S.*)")
PYTHON_RE = re.compile(r"#\s*python:\s*(\S.*\S)\s*")
LIST_TIMEOUT_S = 120

# Runs inside the venv's interpreter: its own site-packages only, nothing imported from the packages.
LISTER = r"""
import importlib.metadata as md, json, platform, sysconfig
paths = []
for key in ("purelib", "platlib"):
    p = sysconfig.get_paths()[key]
    if p not in paths:
        paths.append(p)
dists = [[d.metadata["Name"], d.version] for d in md.distributions(path=paths)]
print(json.dumps({"python": platform.python_implementation() + " " + platform.python_version(), "dists": dists}))
"""


class PinError(ValueError):
    pass


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(text: str, where: str, strict: bool) -> tuple[dict[str, str], list[str], str | None]:
    """(name -> spec, lines without a name, the '# python:' value). spec is '==1.0.1' (or '@ url', '===x').

    strict (the pin): a line without a name or a name listed twice raises PinError. Otherwise such lines are returned,
    and a second spec for the same name is kept as a line without a name so that it shows as a difference."""
    specs: dict[str, str] = {}
    unnamed: list[str] = []
    python = None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = PYTHON_RE.fullmatch(line)
            if m:
                python = m.group(1)
            continue
        m = LINE_RE.fullmatch(line)
        if not m:
            if strict:
                raise PinError(f"{where}:{n}: a line without a package name: {line!r}")
            unnamed.append(line)
            continue
        name, spec = canon(m.group(1)), f"{m.group(2)}{m.group(3).strip()}"
        if name in specs:
            if strict:
                raise PinError(f"{where}:{n}: {name} is listed twice")
            if specs[name] != spec:
                unnamed.append(f"{name}{spec} (a second entry for {name})")
            continue
        specs[name] = spec
    return specs, unnamed, python


def list_venv(python: str) -> tuple[list[str], str]:
    """The venv's packages as 'name==version' lines (sorted by name) and its interpreter, e.g. 'CPython 3.12.13'.

    The path is made absolute but its symlinks are kept: a venv's bin/python is a link to the base interpreter, and
    only the link's own location makes it start in the venv."""
    python = os.path.abspath(python)
    if not (Path(python).is_file() and os.access(python, os.X_OK)):
        raise PinError(f"no interpreter at {python}")
    env = {**os.environ, "PYTHON_DOTENV_DISABLED": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    try:
        cp = subprocess.run([python, "-I", "-c", LISTER], capture_output=True, text=True, env=env,
                            timeout=LIST_TIMEOUT_S, cwd="/")
    except (OSError, subprocess.TimeoutExpired) as e:
        raise PinError(f"{python} could not list its packages: {e}") from e
    if cp.returncode != 0:
        raise PinError(f"{python} could not list its packages (rc {cp.returncode}): {cp.stderr.strip()[-2000:]}")
    try:
        out = json.loads(cp.stdout.strip().splitlines()[-1])
        rows = sorted({(canon(str(n)), str(v)) for n, v in out["dists"]})
        return [f"{n}=={v}" for n, v in rows], str(out["python"])
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise PinError(f"{python} printed an unreadable package list: {cp.stdout[-2000:]!r}") from e


def differences(pin: dict[str, str], got: dict[str, str], unnamed: list[str]) -> list[str]:
    out = []
    for name in sorted(pin.keys() | got.keys()):
        if name not in got:
            out.append(f"missing  {name}{pin[name]}  (in the pin, not in the venv)")
        elif name not in pin:
            out.append(f"extra    {name}{got[name]}  (in the venv, not in the pin)")
        elif pin[name] != got[name]:
            out.append(f"changed  {name}: pin {pin[name]}, venv {got[name]}")
    for line in unnamed:
        hint = "  (an editable from `uv pip freeze`: list with `uv pip list --format freeze`, or use --python)" \
            if line.startswith("-e") else ""
        out.append(f"unnamed  {line}{hint}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--python", help="the venv's interpreter, e.g. /opt/tau3/venv/bin/python")
    src.add_argument("--freeze", help="a file of name==version lines ('-' = stdin)")
    ap.add_argument("--pin", default=str(PIN), help=f"the pinned list (default {PIN})")
    ap.add_argument("--print", dest="print_list", action="store_true",
                    help="with --python: print the venv's list in the pin's form and compare nothing")
    a = ap.parse_args(argv)
    if a.print_list and not a.python:
        ap.error("--print needs --python")
    try:
        if a.python:
            lines, py = list_venv(a.python)
            if a.print_list:
                print(f"# python: {py}")
                print("\n".join(lines))
                return 0
            source, text = a.python, "\n".join(lines)
        else:
            py = None
            source = "stdin" if a.freeze == "-" else a.freeze
            text = sys.stdin.read() if a.freeze == "-" else Path(a.freeze).read_text(encoding="utf-8")
        pin_path = Path(a.pin)
        pin, _, pin_py = parse(pin_path.read_text(encoding="utf-8"), str(pin_path), strict=True)
        if not pin:
            raise PinError(f"{pin_path}: no package lines")
        got, unnamed, _ = parse(text, source, strict=False)
    except (OSError, PinError) as e:
        print(f"tau3_check_venv: {e}", file=sys.stderr)
        return 2
    diffs = differences(pin, got, unnamed)
    if py is not None and pin_py is not None and py != pin_py:
        diffs.append(f"python   pin {pin_py}, venv {py}")
    if diffs:
        print(f"tau3_check_venv: {len(diffs)} difference(s) between {source} and {pin_path}")
        for d in diffs:
            print(f"  {d}")
        return 1
    tail = f", {py}" if py is not None else ""
    print(f"tau3_check_venv: equal: {len(pin)} packages{tail} ({source} vs {pin_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
