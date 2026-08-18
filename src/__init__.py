"""EvoSkill package root — deliberately empty of re-exports.

This used to lazily re-export nine names (FeedbackDescent, EvoSkill, EvalRunner
and friends) from `src.feedback_descent` and `src.api`. NEITHER MODULE EXISTS in
this tree: both were cut when the public release was carved out of the internal
repo, and the `__all__` list was not. The lazy `__getattr__` meant nothing failed
at import time, so the package looked healthy while `from src import EvoSkill` --
the shape the advertised API invites, and the one a reader tries first -- raised
ModuleNotFoundError. All nine were dead; verified by importing each.

Nothing in this repo imported them: entry points are the console scripts declared
in pyproject.toml, and internal code imports concrete modules
(`src.orchestrator.loop`, `src.agent_profiles...`) directly. An empty root is
honest about that. If a public API is ever wanted here, it should be added
alongside the module that implements it, not left behind by one that left.
"""
