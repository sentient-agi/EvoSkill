"""tau2-bench agent wiring: EvoSkill skill injection into tau2's LLMAgent.

Runs ONLY inside the `_tau2_worker.py` subprocess, under tau2-bench's own
isolated venv -- this module imports tau2 internals directly and is never
imported from EvoSkill's own venv/process.

Mechanism (confirmed clean, non-hacky integration point, no monkeypatching
of tau2 orchestration needed): `build_agent()` (`src/tau2/runner/build.py`)
resolves `TextRunConfig.agent` through `registry.get_agent_factory(name)`
and calls `factory(tools=..., domain_policy=..., llm=..., llm_args=...,
task=..., ...)`. Registering our own factory under a fresh name via
`registry.register_agent_factory(factory, name)` (factory-then-name
argument order, confirmed at `src/tau2/registry.py:107`) lets us subclass
`LLMAgent` and override its `system_prompt` property
(`src/tau2/agent/llm_agent.py:78-82`) -- with zero changes to tau2's own
orchestration code.

REAL PROGRESSIVE DISCLOSURE (2026-07-12, supersedes the prior "inline full
content" design -- see git history for that version's own docstring,
which documented the deviation this fix now closes): skill NAMES +
DESCRIPTIONS go in the system prompt; full CONTENT is loaded on demand via
a genuine `skill` tool call, matching every other harness (executor.py's
`_exec_skill` / SWERebenchSolver's own skill tool).

The original attempt at this (per the prior version's docstring) broke
tau2's own DB-check evaluator the moment the agent actually called the
tool:

    ValueError: Unknown tool 'skill' encountered during replay. The tool
    does not exist in the current environment.

Root cause (confirmed by reading `Environment.set_state()`,
`environment.py:355-360`): the DB/ENV reward component replays EVERY tool
call in the message history against a FRESH `Environment` instance
(`registry.get_env_constructor(domain)`, same entry point
`build_environment`/`evaluator_env.py`'s predicted/gold environments both
use -- confirmed no separate/bypassing construction path exists), and
hard-fails via `_has_tool()` if a tool call's name isn't registered on the
domain's own `ToolKitBase` instance. An ad-hoc tool appended only to the
agent's tool list (the naive `as_tool()`/`add_stop_tool()`-style approach)
is invisible to this replay check.

THE FIX: `skill` is registered as a REAL, non-mutating (`ToolType.THINK`,
`mutates_state=False`) tool on the domain's toolkit, so:
  - `_has_tool('skill')` -> True (it's genuinely on the toolkit now).
  - `_is_mutating_tool('skill')` -> False -> replay SKIPS re-executing it
    (environment.py:366-368: "Non-mutating tools ... don't change state --
    skip them ... to avoid ... non-deterministic output comparison
    issues.") -- exactly the intended behavior for a pure lookup.

Since `ToolKitType`'s metaclass computes `_func_tools` ONCE at
class-DEFINITION time from that class's own body (confirmed:
`toolkit.py:22-40`; monkeypatching a method onto an already-defined class
afterward does NOT retroactively register it), a fresh method can't be
bolted onto e.g. `RetailTools` after the fact -- it must be defined in a
class body at class-creation time. So `_skill_subclass_for()` below
dynamically builds `type("EvoSkillRetailTools", (RetailTools,), {"skill":
...})` via `type()` (whose `attrs` dict genuinely contains `skill`,
triggering the metaclass's own registration path), and the domain's
registered environment constructor is wrapped to reassign
`env.tools.__class__` to this subclass right after construction --
`ToolKitBase.tools`/`use_tool()` read from `self._func_tools`, a class-
level property, so this correctly makes `skill` appear as a real,
replay-safe tool with zero new instance state (no `__init__` override
needed; the subclass adds only a method).

`registry.register_domain()` raises on double-registration
(`registry.py:178-190`) -- can't cleanly re-register "retail" through the
public API a second time, so `_wrap_domain_environments()` writes directly
to `registry._domains[name]` (the same internal dict `register_domain`
itself populates) -- a deliberate, minimal, one-line monkeypatch rather
than fighting the guard, done exactly once per domain per process
(`_wrapped_domains` set) since every domain module registers itself once
at import time and this only needs to wrap that existing entry, not
replace the whole registration mechanism.

Skill content resolution: `_current_skills` is a plain module-level dict
(name -> content), not a ContextVar -- each task gets its own OS
subprocess (`_tau2_worker.py`), so there is no concurrent-access hazard
within a process to guard against (same reasoning as
`cost_tracking.py`'s ledger design).
"""
from __future__ import annotations

import os
from typing import Any

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgent
from tau2.environment.toolkit import ToolType, is_tool
from tau2.registry import registry

# name -> full skill content, set once per worker-process invocation by
# register_evoskill_agent() before the episode runs.
_current_skills: dict[str, str] = {}

_skill_subclass_cache: dict[type, type] = {}
_wrapped_domains: set[str] = set()


def _resolve_skill(name: str) -> str:
    if name not in _current_skills:
        available = ", ".join(sorted(_current_skills.keys())) or "(none)"
        return f"Error: unknown skill '{name}'. Available skills: {available}"
    return _current_skills[name]


def _skill_subclass_for(base_cls: type) -> type:
    """Dynamically build (and cache) a ToolKitBase subclass of `base_cls`
    with a real `skill` tool registered via the metaclass -- see module
    docstring for why this can't just be monkeypatched onto `base_cls`
    directly."""
    if base_cls not in _skill_subclass_cache:
        def skill(self, name: str) -> str:
            """Load a skill's full instructions by name.

            Args:
                name: The skill name to load (from the Available Skills
                    section of your instructions).
            """
            return _resolve_skill(name)

        tool_fn = is_tool(ToolType.THINK, mutates_state=False)(skill)
        _skill_subclass_cache[base_cls] = type(
            f"EvoSkill{base_cls.__name__}", (base_cls,), {"skill": tool_fn}
        )
    return _skill_subclass_cache[base_cls]


def _wrap_domain_environments() -> None:
    """Wrap every currently-registered domain's environment constructor
    (`registry._domains[name]`) so the returned Environment's `tools`
    instance is reassigned to the skill-bearing subclass. Idempotent --
    each domain is wrapped at most once per process, and wrapping is a
    no-op if a domain has no `.tools` (shouldn't happen for the 4 domains
    this build covers, but fail safe rather than crash the episode over a
    skill-loading nicety)."""
    for name in list(registry._domains.keys()):
        if name in _wrapped_domains:
            continue
        _wrapped_domains.add(name)
        original = registry._domains[name]

        def make_wrapped(original_ctor):
            def wrapped(*args: Any, **kwargs: Any):
                env = original_ctor(*args, **kwargs)
                if getattr(env, "tools", None) is not None:
                    env.tools.__class__ = _skill_subclass_for(type(env.tools))
                return env
            return wrapped

        registry._domains[name] = make_wrapped(original)


class EvoSkillLLMAgent(LLMAgent):
    """LLMAgent subclass that appends a progressive-disclosure skill INDEX
    (names + descriptions only) to the system prompt. Full content is
    resolved on demand via the real `skill` tool (see module docstring)."""

    def __init__(
        self,
        tools,
        domain_policy: str,
        llm: str,
        llm_args: dict | None = None,
        *,
        skill_index: str = "",
        **_ignored: Any,
    ):
        self._skill_index = skill_index or ""
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION)
        if self._skill_index:
            return base + "\n\n" + self._skill_index
        return base


def _build_skill_index(skills: dict[str, dict[str, str]]) -> str:
    lines = ["## Skills", ""]
    lines.append(
        "Skills are detailed instruction sets. Only their names and trigger "
        "conditions are shown here to save context space. When a trigger "
        'condition matches your task, call the `skill` tool with `{"name": '
        '"<skill-name>"}` to load the full instructions. See available '
        "skills below."
    )
    lines.append("")
    if skills:
        for info in skills.values():
            lines.append(f"- {info['name']}: {info['description']}")
    else:
        lines.append("(No skills available yet.)")
    return "\n".join(lines)


def register_evoskill_agent(skills: dict[str, dict[str, str]]) -> str:
    """Register a fresh 'evoskill_llm_agent_<pid>' factory closing over this
    task's skill index, wrap domain environments for real skill-tool
    resolution, and return the registered name to pass as
    `TextRunConfig.agent`.

    tau2's registry raises on double-registration under the same name
    (`register_agent_factory`, `src/tau2/registry.py`), so this registers
    under a name scoped to the current process (one task per worker
    process invocation -- see module docstring) rather than a fixed
    literal, so a repeated call within the same process (e.g. a retry) is
    idempotent rather than erroring.
    """
    global _current_skills
    _current_skills = {info["name"]: info["content"] for info in skills.values()}
    _wrap_domain_environments()

    skill_index = _build_skill_index(skills)
    name = f"evoskill_llm_agent_{os.getpid()}"

    def factory(tools, domain_policy, **kwargs):
        return EvoSkillLLMAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
            skill_index=skill_index,
        )

    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(factory, name)
    return name
