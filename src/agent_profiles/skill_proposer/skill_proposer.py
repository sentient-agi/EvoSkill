from typing import Union
from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import SkillProposerResponse
from src.agent_profiles.skill_proposer.prompt import SKILL_PROPOSER_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.sdk_config import is_opencode_sdk


SKILL_PROPOSER_TOOLS = [
    "Read",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
]


def get_skill_proposer_options(
    model: str | None = None,
    provider: str | None = None,
) -> Union[ClaudeAgentOptions, dict]:
    prompt_text = SKILL_PROPOSER_SYSTEM_PROMPT.strip()

    if is_opencode_sdk():
        return {
            "system": prompt_text,
            "model_id": model or "gpt-oss-120b",
            "provider_id": provider or "arc",
            "tools": {tool: True for tool in SKILL_PROPOSER_TOOLS},
            "format": {
                "type": "json_schema",
                "schema": SkillProposerResponse.model_json_schema(),
            },
        }

    system_prompt = {
        "type": "preset",
        "preset": "claude_code",
        "append": prompt_text,
    }
    output_format = {
        "type": "json_schema",
        "schema": SkillProposerResponse.model_json_schema(),
    }
    options = ClaudeAgentOptions(
        output_format=output_format,
        system_prompt=system_prompt,
        allowed_tools=SKILL_PROPOSER_TOOLS,
        cwd=get_project_root(),
    )
    if model:
        options.model = model
    return options


def make_skill_proposer_options(model: str | None = None, provider: str | None = None):
    def factory() -> Union[ClaudeAgentOptions, dict]:
        return get_skill_proposer_options(model=model, provider=provider)
    return factory


# Backward compat
skill_proposer_options = ClaudeAgentOptions(
    output_format={"type": "json_schema", "schema": SkillProposerResponse.model_json_schema()},
    system_prompt={"type": "preset", "preset": "claude_code", "append": SKILL_PROPOSER_SYSTEM_PROMPT.strip()},
    allowed_tools=SKILL_PROPOSER_TOOLS,
    cwd=get_project_root(),
)
