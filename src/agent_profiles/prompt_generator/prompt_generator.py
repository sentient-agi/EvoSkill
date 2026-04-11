from __future__ import annotations

from typing import Any

from src.harness import build_claudecode_options, build_opencode_options, is_claude_sdk
from src.schemas import PromptGeneratorResponse
from src.agent_profiles.prompt_generator.prompt import PROMPT_GENERATOR_SYSTEM_PROMPT


PROMPT_GENERATOR_TOOLS = ["Read", "Bash", "Glob", "Grep", "WebFetch", "WebSearch", "TodoWrite", "BashOutput"]


def get_prompt_generator_options(
    model: str | None = None,
    project_root: str | None = None,
) -> Any:
    if is_claude_sdk():
        return build_claudecode_options(
            system=PROMPT_GENERATOR_SYSTEM_PROMPT.strip(),
            schema=PromptGeneratorResponse.model_json_schema(),
            tools=PROMPT_GENERATOR_TOOLS,
            project_root=project_root,
            model=model,
        )
    return build_opencode_options(
        system=PROMPT_GENERATOR_SYSTEM_PROMPT.strip(),
        schema=PromptGeneratorResponse.model_json_schema(),
        tools=PROMPT_GENERATOR_TOOLS,
        project_root=project_root,
        model=model,
    )


def make_prompt_generator_options(
    *,
    project_root: str | None = None,
    model: str | None = None,
):
    return get_prompt_generator_options(model=model, project_root=project_root)


prompt_generator_options = get_prompt_generator_options()
