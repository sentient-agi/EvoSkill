from typing import Union
from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import PromptGeneratorResponse
from src.agent_profiles.prompt_generator.prompt import PROMPT_GENERATOR_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.sdk_config import is_opencode_sdk


PROMPT_GENERATOR_TOOLS = ["Read", "Bash", "Glob", "Grep", "WebFetch", "WebSearch", "TodoWrite", "BashOutput"]


def get_prompt_generator_options(
    model: str | None = None,
    provider: str | None = None,
) -> Union[ClaudeAgentOptions, dict]:
    prompt_text = PROMPT_GENERATOR_SYSTEM_PROMPT.strip()

    if is_opencode_sdk():
        return {
            "system": prompt_text,
            "model_id": model or "gpt-oss-120b",
            "provider_id": provider or "arc",
            "tools": {tool: True for tool in PROMPT_GENERATOR_TOOLS},
            "format": {
                "type": "json_schema",
                "schema": PromptGeneratorResponse.model_json_schema(),
            },
        }

    system_prompt = {
        "type": "preset",
        "preset": "claude_code",
        "append": prompt_text,
    }
    output_format = {
        "type": "json_schema",
        "schema": PromptGeneratorResponse.model_json_schema(),
    }
    options = ClaudeAgentOptions(
        output_format=output_format,
        system_prompt=system_prompt,
        allowed_tools=PROMPT_GENERATOR_TOOLS,
        cwd=get_project_root(),
    )
    if model:
        options.model = model
    return options


def make_prompt_generator_options(model: str | None = None, provider: str | None = None):
    def factory() -> Union[ClaudeAgentOptions, dict]:
        return get_prompt_generator_options(model=model, provider=provider)
    return factory


# Backward compat
prompt_generator_options = ClaudeAgentOptions(
    output_format={"type": "json_schema", "schema": PromptGeneratorResponse.model_json_schema()},
    system_prompt={"type": "preset", "preset": "claude_code", "append": PROMPT_GENERATOR_SYSTEM_PROMPT.strip()},
    allowed_tools=PROMPT_GENERATOR_TOOLS,
    cwd=get_project_root(),
)
