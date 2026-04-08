from src.agent_profiles.skill_generator import get_project_root

OPENCODE_OUTPUT_INSTRUCTIONS = """
Always end your response with a JSON object on its own line in this exact format:
{"final_answer": "<your answer here>", "reasoning": "<your step-by-step reasoning here>"}
"""

_MODEL_MAP = {
    "sonnet": ("anthropic", "claude-sonnet-4-6"),
    "opus": ("anthropic", "claude-opus-4-6"),
    "haiku": ("anthropic", "claude-haiku-4-5"),
}
_DEFAULT_PROVIDER = "opencode"
_DEFAULT_MODEL = "big-pickle"


def _resolve_model(model: str | None) -> tuple[str, str]:
    if model and model in _MODEL_MAP:
        return _MODEL_MAP[model]
    if model and "/" in model:
        return model.split("/", 1)
    return _DEFAULT_PROVIDER, model or _DEFAULT_MODEL


def make_opencode_agent_options_from_task(
    task_description: str,
    model: str | None = None,
    data_dirs: list[str] | None = None,
):
    """Factory for OpenCode base agent options.

    Skills in .opencode/skills/ are discovered natively by the opencode agent
    via its Skill tool — no manual injection needed.
    """
    provider_id, model_id = _resolve_model(model)

    def factory() -> dict:
        system = task_description.strip() + "\n\n" + OPENCODE_OUTPUT_INSTRUCTIONS.strip()
        return {
            "system": system,
            "provider_id": provider_id,
            "model_id": model_id,
            "mode": "build",
        }
    return factory
