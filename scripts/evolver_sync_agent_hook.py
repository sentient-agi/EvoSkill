#!/usr/bin/env python3
"""PreToolUse hook: force every Agent (subagent) dispatch to run SYNCHRONOUSLY.

WHY THIS EXISTS (#212). CC 2.1.220's Agent tool dispatches subagents in BACKGROUND
mode by default. The parent's tool_result is only:

    "Async agent launched successfully. (This tool result is internal metadata ...)"

The subagent's ACTUAL result arrives later as a `task_notification` system event,
which in `claude -p` starts a NEW turn. Claude models are trained to ride that
protocol — wait, receive the notification, resume. Non-Claude parents are not.

Measured 2026-08-08 with DIRECT OpenRouter routing (no proxy), a forced two-subagent
smoke test on deepseek-v4-flash-0731:

    default (async)          -> THREE separate `result` events. The parent said "I'll
                                wait for their completion notifications", its turn
                                ENDED, each notification re-invoked it fresh, and by
                                the third invocation it had lost the thread. Final
                                answer: "RESULT: MISSING". It had already received one
                                token in an earlier invocation and dropped it. It also
                                broke the "do not read the file yourself" instruction
                                and Read the file directly, because delegation appeared
                                to have produced nothing.
    run_in_background=false  -> ONE turn-chain (turns=3, 13.1s). Both subagent results
                                came back as ordinary tool_results. Final answer:
                                "RESULT: HALYARD-7731 PLIMSOLL-4402". Correct.

This is why the 2026-08-08 DSV4F evolver probe looked like a model failure. It was
not: all six of its subagents ran clean (32/43/36/39/45/33 turns, err=0 on all six)
while the parent collected "tool use was rejected" and no-result turns. Children
healthy, parent starved — a result-return-path defect, not a capability ceiling.

WHY A HOOK AND NOT A PROMPT LINE. The evolver prompt is the experimental object. The
Opus and Sonnet arms ran on a prompt that says nothing about subagent scheduling;
adding a line for the DSV4F arm would be exactly the kind of extra scaffolding that
was stripped from the Arm A epoch-2 prompt for parity (handoff H4). A hook changes
the HARNESS, which is allowed to differ where a model cannot otherwise run at all —
the same accommodation Arm C needed with `reasoning.enabled=false` (H5). State it as
a limitation; do not hide it.

PARITY FOOTNOTE. Opus dispatched 0 subagents across 8 runs, so this hook is a no-op
for it. Sonnet dispatched 25/9/8/5 async subagents and handled them correctly, so its
completed run is unaffected and is NOT invalidated by this hook's existence.

FAILS OPEN, DELIBERATELY. A malformed payload or an unexpected schema returns 0 with
no modification: the worst case is the pre-existing async behaviour, which is what we
already have. This hook can only improve the situation, never wedge the evolver.
Contrast evolver_leak_hook.py, which fails CLOSED because it guards correctness of
the experiment rather than convenience.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # never wedge the evolver on a malformed payload

    if payload.get("tool_name") != "Agent":
        return 0

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0

    # Already synchronous — emit nothing so the transcript is not cluttered with
    # no-op rewrites, and so `updatedInput` only ever appears where it changed something.
    if tool_input.get("run_in_background") is False:
        return 0

    updated = dict(tool_input)
    updated["run_in_background"] = False

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": (
                    "Forced synchronous subagent dispatch (#212): the async path returns "
                    "the result via a task_notification on a later turn, which this "
                    "parent model does not consume."
                ),
                "updatedInput": updated,
            },
            "suppressOutput": True,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
