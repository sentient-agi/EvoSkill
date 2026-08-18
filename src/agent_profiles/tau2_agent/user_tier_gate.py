"""User-simulator model-tier gate: decide which tau2 tasks require a
reference-tier (not cheap) user-simulator model.

Hard requirement (empirically motivated, see the investigation this build
followed): a fixed agent run against 3 airline tasks x {reference gpt-4.1
vs cheap gpt-5-nano} as user-sim produced IDENTICAL reward (1.0/1.0) in all
6 pairs, yet reading the transcripts found the cheap user-sim (a) got
stuck in a verbatim-repeat loop then fully inverted its role mid-
conversation, and (b) caved on a fabricated claim after just one round of
pushback where it was supposed to sustain it for 5. Reward-based screening
completely missed both defects -- so the user-sim tier must be gated on
task content, never left to a reward-based check, and must fail SAFE to
reference-tier when in doubt.

No structural field on `Task`/`UserScenario`/`EvaluationCriteria` signals
"requires sustained adversarial persona" (checked `description`,
`user_scenario.persona` -- null on all 114 retail tasks --
`user_scenario.instructions`, `evaluation_criteria.reward_basis` fully).
`reward_basis` is actively misleading as a proxy: the empirical defect was
found on **airline**, the one domain whose `reward_basis` NEVER includes
NL_ASSERTION (0/50) -- so gating on reward_basis would have shipped this
exact gap. This module instead runs a one-time, offline, cached LLM
classifier over each task's user_scenario text (persona + instructions +
description notes), using a reference-tier judge model (misclassifying
here is exactly the failure this mechanism exists to prevent, so the
classifier itself must not be a cheap model).

Manifest format (`data/tau2_user_tier_gate/<domain>.json`):
    {"<task_id>": {"requires_reference_tier": bool, "reasoning": str}, ...}
Committed to git so every verdict is human-auditable/overridable by hand,
and so the gate costs nothing per evolution run (see
`scripts/build_tau2_user_tier_gate.py` for the offline builder).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from src.harness.direct.options import pin_request_body_provider

_RUBRIC = """
You are helping build a customer-service simulation testbed. Given the
scenario a simulated CUSTOMER will act out below, decide whether playing
this customer realistically requires SUSTAINED adversarial or deceptive
persona behavior across multiple conversation turns -- specifically:

(a) sustaining a fabricated or false claim under repeated pushback from
    the agent (not just stating it once),
(b) insisting or refusing repeatedly across several turns rather than
    conceding quickly once the agent responds reasonably, or
(c) adversarially switching topics or testing the agent's robustness
    under pressure.

A cheap/weak model playing this customer is at real risk of caving on (a)
after one round, or drifting/inverting roles during a long insistence loop
in (b)/(c). If the scenario is just a straightforward information request
or a single clearly-stated need (even if emotionally phrased), answer
false.

Scenario:
<persona>{persona}</persona>
<task_instructions>{task_instructions}</task_instructions>
<reason_for_call>{reason_for_call}</reason_for_call>
<known_info>{known_info}</known_info>
<unknown_info>{unknown_info}</unknown_info>
<description_notes>{description_notes}</description_notes>

Respond with ONLY a JSON object, no other text:
{{"requires_reference_tier": <true|false>, "reasoning": "<one or two sentences>"}}
""".strip()


def _manifest_path(domain: str, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "data" / "tau2_user_tier_gate" / f"{domain}.json"


def load_manifest(domain: str, repo_root: Path | None = None) -> dict[str, dict]:
    path = _manifest_path(domain, repo_root)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_manifest(domain: str, manifest: dict[str, dict], repo_root: Path | None = None) -> None:
    path = _manifest_path(domain, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def get_tier(domain: str, task_id: str, manifest: dict[str, dict]) -> str:
    """Return "reference" or "cheap" for this task.

    Fail-safe default: a task_id missing from the manifest (classifier
    never ran, or errored) defaults to "reference", never a silent
    downgrade -- this is a hard requirement, not a suggestion.
    """
    entry = manifest.get(task_id)
    if entry is None or entry.get("requires_reference_tier") is not False:
        return "reference"
    return "cheap"


async def classify_user_tier(
    task: dict[str, Any],
    *,
    judge_model: str,
    api_key: str | None = None,
    provider_order: list[str] | None = None,
) -> dict[str, Any]:
    """Classify a single raw task dict (tau2's own Task JSON schema).

    Returns {"requires_reference_tier": bool, "reasoning": str}. On any
    parse/API failure, fails SAFE: returns requires_reference_tier=True
    with the error recorded in `reasoning` -- never silently defaults to
    "cheap".
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    # This hits OpenRouter's own REST API directly (not through litellm), so
    # the model id must be OpenRouter's own bare form ("openai/gpt-4.1"),
    # NOT litellm's "openrouter/<...>" routing-prefix convention used
    # elsewhere in this package (solver.py/worker, which go through tau2's
    # own litellm-based `generate()`). Confirmed empirically: OpenRouter's
    # API 400s with "openrouter/openai/gpt-4.1 is not a valid model ID" if
    # the prefix is left on -- strip it defensively so callers can pass
    # either form.
    if judge_model.startswith("openrouter/"):
        judge_model = judge_model[len("openrouter/"):]
    instructions = (task.get("user_scenario") or {}).get("instructions") or {}
    prompt = _RUBRIC.format(
        persona=(task.get("user_scenario") or {}).get("persona") or "(none)",
        task_instructions=instructions.get("task_instructions") or "(none)",
        reason_for_call=instructions.get("reason_for_call") or "(none)",
        known_info=instructions.get("known_info") or "(none)",
        unknown_info=instructions.get("unknown_info") or "(none)",
        description_notes=(task.get("description") or {}).get("notes") or "(none)",
    )
    body: dict[str, Any] = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    pin_request_body_provider(body, provider_order)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or "{}"
            parsed = json.loads(content)
        return {
            "requires_reference_tier": bool(parsed.get("requires_reference_tier", True)),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception as e:  # noqa: BLE001 -- classifier failure must fail
                            # SAFE (reference tier), not silently swallow;
                            # the exception is surfaced in `reasoning`.
        return {
            "requires_reference_tier": True,
            "reasoning": f"Classifier error, defaulted to reference-tier: {type(e).__name__}: {e}",
        }
