"""A scripted tau2 agent and customer that play a tau3 task's golden actions through tau2's own Orchestrator, with no
LLM anywhere.

The rollout harness puts Claude Code in the agent's seat of tau2's Orchestrator and gpt-5-nano in the
customer's. This pair takes both seats and plays the task's expected actions instead, so build_text_orchestrator +
run() produce, offline, the conversation tests/tau3_golden.py builds by hand: the same messages, the same tool-call ids
and the same tool results, now with tau2 assigning the order, the steps, the timestamps and the termination. It is the
oracle for the bundle writer (scripts/tau3_cc/tau2_bundle.py, tests/test_tau2_bundle.py) and, later, for the harness's
fake-claude protocol tests.

  - register() adds both to tau2's registry, once per process: the agent through registry.register_agent_factory
    (AGENT; tau2/registry.py:106-134), the customer through registry.register_user (USER; registry.py:84-104).
    config() is the TextRunConfig that selects them for banking_knowledge with the bm25_grep variant; everything else
    is tau2's default (max_steps 200, max_errors 10, no communication-protocol enforcement).
  - The agent factory receives the task (tau2/runner/build.py:117-125) and builds the Script: the task's expected
    actions and a cursor. tau2 hands the customer only str(task.user_scenario) (build.py:168-177), so the factory
    leaves its Script in a one-slot hand-over that the customer's constructor takes; build_text_orchestrator builds the
    agent and then the customer (build.py:395-412), and the customer checks that the Script is for its scenario.
  - Each side, on its turn, plays the next expected action when it is its own: ONE ToolCall per message (an
    AssistantMessage for the agent's, a UserMessage for the customer's), with tau3_golden.call_id's ids (toolu_01... for
    the agent, as Claude Code's; call_... for the customer, as the simulator's). Otherwise it hands the turn over with
    tau3_golden's lines: the customer opens with OPENING, the agent says TO_USER before a customer action, the customer
    says TO_AGENT when the next action is the agent's or none is left, the agent says CLOSING when none is left, and the
    customer answers the closing line with ###STOP### (termination user_stop). That is tau3_golden.golden_messages,
    message for message.
  - Every scripted message costs 0.0 and carries a raw_data dict standing in for a provider's raw reply, so a test can
    see the bundle writer remove it.
  - ScriptedAgent.end_on_call(n, exc) makes the agent raise exc on its n-th generate_next_message call instead of
    answering: how a test ends a conversation from inside the agent (tau2_bundle.HarnessEnd) or checks that another
    exception propagates.

IMPORT: tau2 at import time, so only under the tau3-bench venv, in a process whose environment already holds
PYTHON_DOTENV_DISABLED=1, LITELLM_LOCAL_MODEL_COST_MAP=True and TAU2_DATA_DIR (tests/tau3_golden.child_env), with tests/
on sys.path (for tau3_golden). Nothing here calls a model: no user simulator, agent LLM or judge is built.
"""
from __future__ import annotations

from typing import Optional

import tau3_golden as G
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import TextRunConfig
from tau2.registry import registry
from tau2.user.user_simulator_base import HalfDuplexUser, UserState

AGENT = "scripted_golden_agent"
USER = "scripted_golden_user"
NO_MODEL = "none (scripted)"
_HANDOVER: list = []            # the agent factory's Script, taken by the customer built next


class Script:
    """One task's expected actions, shared by the scripted agent and customer; `at` is the next one to play."""

    def __init__(self, task):
        self.task = task
        crit = task.evaluation_criteria
        self.actions = list((crit.actions if crit is not None else None) or [])
        self.at = 0

    def next_by(self) -> Optional[str]:
        """Who plays the next expected action ('assistant' or 'user'); None when all are played."""
        return self.actions[self.at].requestor if self.at < len(self.actions) else None

    def call(self, requestor: str) -> ToolCall:
        a = self.actions[self.at]
        if a.requestor != requestor:
            raise RuntimeError(f"{self.task.id}: action {self.at} is the {a.requestor}'s, not the {requestor}'s")
        tc = ToolCall(id=G.call_id(self.task.id, self.at, requestor), name=a.name, arguments=dict(a.arguments or {}),
                      requestor=requestor)
        self.at += 1
        return tc


def _raw(side: str, n: int) -> dict:
    return {"fixture": "fixtures_tau2_scripted", "side": side, "turn": n}


class ScriptedAgent(HalfDuplexAgent):
    """Plays the task's agent actions; its state is the list of messages it has seen and sent."""

    def __init__(self, tools, domain_policy: str, script: Script):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.script = script
        self.calls = 0
        self.seed = None
        self.end_at = None
        self.end_with = None

    def end_on_call(self, n: int, exc: BaseException) -> None:
        self.end_at, self.end_with = n, exc

    def get_init_state(self, message_history=None) -> list:
        return list(message_history or [])

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def generate_next_message(self, message, state):
        self.calls += 1
        state.append(message)
        if self.calls == self.end_at:
            raise self.end_with
        nxt = self.script.next_by()
        if nxt == "assistant":
            out = AssistantMessage(role="assistant", content=None, tool_calls=[self.script.call("assistant")],
                                   cost=0.0, raw_data=_raw("agent", self.calls))
        else:
            out = AssistantMessage(role="assistant", content=G.TO_USER if nxt == "user" else G.CLOSING, cost=0.0,
                                   raw_data=_raw("agent", self.calls))
        state.append(out)
        return out, state


def create_scripted_agent(tools, domain_policy, task=None, **_ignored):
    """tau2's agent factory signature (build_agent passes tools, domain_policy, llm, llm_args, task, ...)."""
    if task is None:
        raise ValueError("the scripted agent plays a task's expected actions: build it with the task")
    script = Script(task)
    _HANDOVER[:] = [script]
    return ScriptedAgent(tools, domain_policy, script)


class ScriptedUser(HalfDuplexUser):
    """Plays the task's customer actions and ends with ###STOP###; takes the Script its agent left."""

    def __init__(self, tools=None, instructions: Optional[str] = None, llm=None, llm_args=None):
        super().__init__(instructions=instructions, tools=tools)
        if not _HANDOVER:
            raise RuntimeError("the scripted customer is built after its scripted agent (build_text_orchestrator)")
        self.script = _HANDOVER.pop()
        if instructions != str(self.script.task.user_scenario):
            raise ValueError(f"the scripted agent's Script is for {self.script.task.id}, not this customer's scenario")
        self.calls = 0
        self.seed = None

    def get_init_state(self, message_history=None) -> UserState:
        return UserState(system_messages=[], messages=list(message_history or []))

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def generate_next_message(self, message, state):
        self.calls += 1
        first = not state.messages
        state.messages.append(message)
        nxt = self.script.next_by()
        if first:
            out = UserMessage(role="user", content=G.OPENING, cost=0.0, raw_data=_raw("user", self.calls))
        elif nxt == "user":
            out = UserMessage(role="user", content=None, tool_calls=[self.script.call("user")], cost=0.0,
                              raw_data=_raw("user", self.calls))
        elif nxt is None and isinstance(message, AssistantMessage) and not message.is_tool_call():
            out = UserMessage(role="user", content=G.STOP, cost=0.0, raw_data=_raw("user", self.calls))
        else:
            out = UserMessage(role="user", content=G.TO_AGENT, cost=0.0, raw_data=_raw("user", self.calls))
        state.messages.append(out)
        return out, state


def register() -> None:
    """Add the pair to tau2's registry (a no-op the second time in one process)."""
    if registry.get_agent_factory(AGENT) is None:
        registry.register_agent_factory(create_scripted_agent, AGENT)
    if USER not in registry.get_users():
        registry.register_user(ScriptedUser, USER)


def config(**kw) -> TextRunConfig:
    """The TextRunConfig that runs the scripted pair on banking_knowledge with the tests' retrieval variant."""
    base = dict(domain=G.DOMAIN, agent=AGENT, user=USER, llm_agent=NO_MODEL, llm_args_agent={}, llm_user=NO_MODEL,
                llm_args_user={}, retrieval_config=G.VARIANT)
    base.update(kw)
    return TextRunConfig(**base)
