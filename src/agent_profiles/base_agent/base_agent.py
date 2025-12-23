from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import AgentResponse
from src.agent_profiles.base_agent.prompt import BASE_AGENT_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root
import os


BASE_AGENT_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]


base_agent_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": BASE_AGENT_SYSTEM_PROMPT.strip()
}
base_agent_output_format = {
            "type": "json_schema",
            "schema": AgentResponse.model_json_schema()
        }

file_path = os.path.join(get_project_root(), "transformed/")

base_agent_options = ClaudeAgentOptions(
    system_prompt=base_agent_system_prompt,
    output_format=base_agent_output_format,
    allowed_tools=BASE_AGENT_TOOLS,
    setting_sources=["user", "project"],  # Load Skills from filesystem
    permission_mode='acceptEdits',
    add_dirs = [file_path],
    cwd=get_project_root(),
)