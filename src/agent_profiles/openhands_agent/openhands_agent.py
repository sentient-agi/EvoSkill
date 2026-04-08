"""OpenHands agent factory.

Skills are injected into the system prompt by base.py. This factory only provides
the connection config (model, api_key, system prompt, cwd).
"""

import os
from src.agent_profiles.skill_generator import get_project_root
from src.schemas import AgentResponse


OUTPUT_INSTRUCTIONS = """

## Output Format

You MUST respond with a JSON object as the final content of your response. The JSON must have exactly these fields:

```json
{
  "final_answer": "<your answer here>",
  "reasoning": "<brief explanation of how you arrived at the answer>"
}
```

Do not include any text after the JSON object.
"""


def make_openhands_agent_options(
    task_description: str,
    model: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    data_dirs: list[str] | None = None,
):
    """Create a factory for OpenHands agent options.

    Skills are loaded into the system prompt by base.py from .agents/skills/.
    This factory just provides connection config.

    Args:
        task_description: The task description to use as system prompt.
        model: Model identifier (e.g. "anthropic/claude-sonnet-4-5-20250929").
        model_id: Backward-compatible alias for model.
        api_key: API key for the model provider.
        data_dirs: Extra data dirs, currently unused for OpenHands.

    Returns:
        A callable that returns an options dict for the OpenHands harness.
    """
    project_root = get_project_root()
    from src.harness import load_prompt_text, resolve_openhands_llm_config

    llm_config = resolve_openhands_llm_config(
        model=model or model_id,
        api_key=api_key,
    )

    def factory() -> dict:
        prompt_text = load_prompt_text(
            harness="openhands",
            project_root=project_root,
            fallback_text=task_description,
        )
        return {
            "model_id": llm_config["model"],
            "api_key": llm_config["api_key"],
            "base_url": llm_config["base_url"],
            "system": prompt_text + OUTPUT_INSTRUCTIONS,
            "cwd": project_root,
        }

    return factory
