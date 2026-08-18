import json

from pydantic import BaseModel, Field, field_validator


class SkillEvolverResponse(BaseModel):
    """Response from the unified skill evolver agent."""

    action: str
    """What you did, e.g. 'created counting-rules', 'edited critical-reasoning and deleted old-verification'."""

    skill_name: str
    """Name of the primary skill created or edited."""

    summary: str
    """One-line summary of what changed and why."""

    justification: str
    """Which failure(s) motivated this change and what root cause you identified."""

    notebook: str
    """Structured handoff note for FUTURE evolver iterations.

    NOT length-capped (0725). The old "≤1000 tokens" was advisory only — bare
    attribute docstrings do not reach the model (`model_json_schema()` renders
    this field as `{"title": "Notebook", "type": "string"}`), so the sole live
    instance was in prompt.md and it was never binding: the 20x3 episode's note
    came in at ~479 tokens, 47% of the cap. Do not reinstate it here expecting
    an effect — the prompt is the only lever.

    Later iterations see every batch's note verbatim, so this is the loop's
    only persistent memory across iterations — write it for a successor who has
    NOT seen your batch. Use exactly these five sections:
    VERIFIED FINDINGS: beliefs proven with trace/diff evidence (cite the task ids).
    HYPOTHESES: plausible but unverified — label them so successors don't
    mistake them for facts.
    EDITS MADE: what you changed in the skills and why.
    PREDICTIONS: concrete, checkable claims about the NEXT batch (e.g.
    "sign-convention failures should disappear on Debugging tasks").
    OPEN QUESTIONS: what you couldn't resolve with this batch's evidence."""

    @field_validator("action", "skill_name", "summary", "justification", "notebook", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        # Models occasionally emit a structured value where a string is
        # required — observed 2026-07-29 (fact-a iter-1): DSV4F returned the
        # notebook as a dict keyed by its five section names. That episode's
        # 25 minutes of analysis and its skill edits were then discarded over
        # formatting. Serialize instead of failing: a JSON-rendered notebook
        # is a perfectly readable handoff note.
        if isinstance(v, str):
            return v
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, indent=1)
        return str(v)
