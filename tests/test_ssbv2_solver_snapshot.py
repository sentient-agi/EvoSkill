"""The freeze and the leg must refuse the SAME solver models.

The constraint lived only in scripts/ssbv2_leg.sh — the LAST step — so an operator could
brief the AO with any solver, freeze a spec, build the oracle, and only then find that no
leg would ever run. Moving it to ExperimentSpec.validate() fixed that only if the two rules
agree; the first version accepted any `-NNNN` suffix and so still let `…-0901` through the
freeze into an exit-4 leg. This reads the shell's own case statement rather than restating
it, so extending one list and forgetting the other fails here.
"""
import dataclasses
import re
from pathlib import Path

import pytest

from src.orchestrator.spec import SSBV2_SOLVER_SNAPSHOTS, ExperimentSpec

LEG = Path(__file__).resolve().parents[1] / "scripts" / "ssbv2_leg.sh"


def _leg_allow_list() -> tuple[str, ...]:
    m = re.search(r"case \"\$MODEL\" in\s*\n\s*([^)]+)\)", LEG.read_text())
    assert m, "could not find the leg launcher's `case \"$MODEL\" in` allow-list"
    return tuple(p.strip().lstrip("*") for p in m.group(1).split("|"))


def _spec(**over):
    f = {x.name: x for x in dataclasses.fields(ExperimentSpec)}
    kw = {n: (x.default if x.default is not dataclasses.MISSING
              else x.default_factory() if x.default_factory is not dataclasses.MISSING
              else ("" if x.type in ("str", str) else 0)) for n, x in f.items()}
    kw.update(over)
    return ExperimentSpec(**kw)


def _freeze_refuses(model: str) -> bool:
    return any("dated solver snapshot" in e
               for e in _spec(benchmark="ssbv2", solver_model=model, name="x").validate())


def test_the_two_allow_lists_are_identical():
    assert set(SSBV2_SOLVER_SNAPSHOTS) == set(_leg_allow_list()), (
        f"spec allows {sorted(SSBV2_SOLVER_SNAPSHOTS)} but scripts/ssbv2_leg.sh allows "
        f"{sorted(_leg_allow_list())} — a spec can freeze a model no leg will run")


@pytest.mark.parametrize("model,refused", [
    ("deepseek/deepseek-v4-flash-0731", False),
    ("deepseek/deepseek-v4-pro-0813", False),
    ("deepseek/deepseek-v4-flash", True),      # the rolling tag: a different model entirely
    ("some/model-0901", True),                 # dated, but not allow-listed — the drift case
    ("some/model-2026", True),
])
def test_freeze_matches_the_leg_model_for_model(model, refused):
    assert _freeze_refuses(model) is refused, model
    in_leg = model.endswith(_leg_allow_list())
    assert _freeze_refuses(model) is not in_leg, (
        f"{model}: freeze refuses={_freeze_refuses(model)} but the leg accepts={in_leg}")


def test_other_benchmarks_are_unaffected():
    assert not _freeze_refuses.__wrapped__ if False else True
    assert not any("dated solver snapshot" in e for e in
                   _spec(benchmark="tau3", solver_model="deepseek/deepseek-v4-flash",
                         name="x").validate())
