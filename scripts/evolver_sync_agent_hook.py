#!/usr/bin/env python3
"""PreToolUse hook: force every Agent (subagent) dispatch to run SYNCHRONOUSLY.

WHY THIS EXISTS. Claude Code's Agent tool dispatches subagents in BACKGROUND mode by default. The parent's
tool_result is only:

    "Async agent launched successfully. (This tool result is internal metadata ...)"

The subagent's ACTUAL result arrives later as a `task_notification` system event, which in `claude -p` starts a NEW
turn. A parent model that does not consume that protocol ends its turn waiting, is re-invoked fresh by each
notification, and loses the results. With run_in_background=false the subagent results come back as ordinary
tool_results inside one turn chain.

WHY A HOOK AND NOT A PROMPT LINE. The evolver prompt is the experimental object and stays the same for every model;
a hook changes the HARNESS instead. A parent that never dispatches subagents is unaffected, and one that already
handles async subagents gets the same results synchronously.

FAILS OPEN, DELIBERATELY. A malformed payload or an unexpected schema returns 0 with no modification: the worst case
is the default async behaviour. This hook can only improve the situation, never wedge the evolver.
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
                    "Forced synchronous subagent dispatch: the async path returns "
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
