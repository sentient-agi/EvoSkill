"""Tests for src/registry/models.py and src/registry/sdk_utils.py.

We mock claude_agent_sdk.ClaudeAgentOptions so tests run without the real SDK.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_mock_claude_options(**kwargs):
    """Return a mock that behaves like ClaudeAgentOptions."""
    opts = MagicMock()
    opts.system_prompt = kwargs.get("system_prompt", {"type": "preset", "preset": "claude_code"})
    opts.allowed_tools = kwargs.get("allowed_tools", ["Read", "Bash"])
    opts.output_format = kwargs.get("output_format", None)
    return opts


# ===========================================================================
# ProgramConfig
# ===========================================================================

class TestProgramConfig:
    def _make_base(self, **overrides):
        from src.registry.models import ProgramConfig

        defaults = {
            "name": "base",
            "system_prompt": {"type": "preset", "preset": "claude_code"},
            "allowed_tools": ["Read", "Bash"],
        }
        defaults.update(overrides)
        return ProgramConfig(**defaults)

    # --- Construction ---

    def test_minimal_construction(self):
        config = self._make_base()
        assert config.name == "base"
        assert config.generation == 0
        assert config.parent is None

    def test_default_allowed_tools_is_empty_list(self):
        from src.registry.models import ProgramConfig

        config = ProgramConfig(
            name="x",
            system_prompt={"type": "preset", "preset": "claude_code"},
        )
        assert config.allowed_tools == []

    def test_default_metadata_is_empty_dict(self):
        config = self._make_base()
        assert config.metadata == {}

    def test_default_output_format_is_none(self):
        config = self._make_base()
        assert config.output_format is None

    # --- with_metadata ---

    def test_with_metadata_returns_new_instance(self):
        config = self._make_base()
        updated = config.with_metadata(score=0.85)
        assert updated is not config
        assert updated.metadata["score"] == 0.85

    def test_with_metadata_does_not_mutate_original(self):
        config = self._make_base()
        config.with_metadata(score=0.9)
        assert "score" not in config.metadata

    def test_with_metadata_merges_existing(self):
        config = self._make_base(metadata={"key": "value"})
        updated = config.with_metadata(score=0.5)
        assert updated.metadata["key"] == "value"
        assert updated.metadata["score"] == 0.5

    # --- with_score ---

    def test_with_score(self):
        config = self._make_base()
        scored = config.with_score(0.75)
        assert scored.get_score() == pytest.approx(0.75)

    def test_get_score_returns_none_when_not_set(self):
        config = self._make_base()
        assert config.get_score() is None

    # --- with_timestamp ---

    def test_with_timestamp_adds_created_at(self):
        config = self._make_base()
        stamped = config.with_timestamp()
        assert "created_at" in stamped.metadata
        # Should be a parseable ISO datetime
        datetime.fromisoformat(stamped.metadata["created_at"])

    def test_with_timestamp_does_not_mutate_original(self):
        config = self._make_base()
        config.with_timestamp()
        assert "created_at" not in config.metadata

    # --- mutate ---

    def test_mutate_creates_child_with_parent_reference(self):
        parent = self._make_base()
        child = parent.mutate("child")
        assert child.parent == "program/base"

    def test_mutate_increments_generation(self):
        parent = self._make_base()
        child = parent.mutate("child")
        assert child.generation == 1

    def test_mutate_inherits_system_prompt_when_not_overridden(self):
        parent = self._make_base()
        child = parent.mutate("child")
        assert child.system_prompt == parent.system_prompt

    def test_mutate_overrides_system_prompt(self):
        parent = self._make_base()
        new_prompt = {"type": "text", "content": "New prompt"}
        child = parent.mutate("child", system_prompt=new_prompt)
        assert child.system_prompt == new_prompt

    def test_mutate_inherits_tools_when_not_overridden(self):
        parent = self._make_base(allowed_tools=["Read", "Bash"])
        child = parent.mutate("child")
        assert child.allowed_tools == ["Read", "Bash"]

    def test_mutate_overrides_tools(self):
        parent = self._make_base(allowed_tools=["Read"])
        child = parent.mutate("child", allowed_tools=["Write", "Bash"])
        assert set(child.allowed_tools) == {"Write", "Bash"}

    def test_mutate_adds_timestamp_to_child(self):
        parent = self._make_base()
        child = parent.mutate("child")
        assert "created_at" in child.metadata

    def test_deep_mutation_chain_increments_generation(self):
        base = self._make_base()
        gen1 = base.mutate("gen1")
        gen2 = gen1.mutate("gen2")
        gen3 = gen2.mutate("gen3")
        assert gen3.generation == 3

    def test_mutate_metadata_overrides_defaults(self):
        parent = self._make_base()
        child = parent.mutate("child", metadata={"custom": "value"})
        assert child.metadata.get("custom") == "value"


# ===========================================================================
# sdk_utils — merge_system_prompt, add_tools, remove_tools
# (config_to_options and options_to_config need ClaudeAgentOptions mock)
# ===========================================================================

class TestMergeSystemPrompt:
    def test_append_to_empty_base(self):
        from src.registry.sdk_utils import merge_system_prompt

        base = {"type": "preset", "preset": "claude_code"}
        result = merge_system_prompt(base, append="extra instructions")
        assert result["append"] == "extra instructions"

    def test_append_to_existing_append(self):
        from src.registry.sdk_utils import merge_system_prompt

        base = {"type": "preset", "preset": "claude_code", "append": "first"}
        result = merge_system_prompt(base, append="second")
        assert "first" in result["append"]
        assert "second" in result["append"]

    def test_prepend_to_empty_base(self):
        from src.registry.sdk_utils import merge_system_prompt

        base = {"type": "preset"}
        result = merge_system_prompt(base, prepend="intro")
        assert result["append"].startswith("intro")

    def test_prepend_to_existing_append(self):
        from src.registry.sdk_utils import merge_system_prompt

        base = {"append": "tail"}
        result = merge_system_prompt(base, prepend="head")
        assert result["append"].startswith("head")
        assert "tail" in result["append"]

    def test_no_modification_when_no_args(self):
        from src.registry.sdk_utils import merge_system_prompt

        base = {"type": "preset", "preset": "claude_code"}
        result = merge_system_prompt(base)
        assert result == base

    def test_does_not_mutate_input(self):
        from src.registry.sdk_utils import merge_system_prompt

        base = {"type": "preset"}
        original = dict(base)
        merge_system_prompt(base, append="extra")
        assert base == original


class TestAddRemoveTools:
    def _base_config(self, tools):
        from src.registry.models import ProgramConfig

        return ProgramConfig(
            name="base",
            system_prompt={"type": "preset"},
            allowed_tools=tools,
        )

    def test_add_tools_returns_new_config(self):
        from src.registry.sdk_utils import add_tools

        config = self._base_config(["Read"])
        updated = add_tools(config, ["Bash"])
        assert config is not updated

    def test_add_tools_includes_new_tools(self):
        from src.registry.sdk_utils import add_tools

        config = self._base_config(["Read"])
        updated = add_tools(config, ["Bash", "Write"])
        assert "Bash" in updated.allowed_tools
        assert "Write" in updated.allowed_tools
        assert "Read" in updated.allowed_tools

    def test_add_tools_deduplicates(self):
        from src.registry.sdk_utils import add_tools

        config = self._base_config(["Read"])
        updated = add_tools(config, ["Read", "Bash"])
        assert updated.allowed_tools.count("Read") == 1

    def test_add_tools_does_not_mutate_original(self):
        from src.registry.sdk_utils import add_tools

        config = self._base_config(["Read"])
        add_tools(config, ["Bash"])
        assert config.allowed_tools == ["Read"]

    def test_remove_tools_returns_new_config(self):
        from src.registry.sdk_utils import remove_tools

        config = self._base_config(["Read", "Bash"])
        updated = remove_tools(config, ["Bash"])
        assert config is not updated

    def test_remove_tools_excludes_specified(self):
        from src.registry.sdk_utils import remove_tools

        config = self._base_config(["Read", "Bash", "Write"])
        updated = remove_tools(config, ["Bash"])
        assert "Bash" not in updated.allowed_tools
        assert "Read" in updated.allowed_tools
        assert "Write" in updated.allowed_tools

    def test_remove_nonexistent_tool_is_noop(self):
        from src.registry.sdk_utils import remove_tools

        config = self._base_config(["Read"])
        updated = remove_tools(config, ["NonExistentTool"])
        assert updated.allowed_tools == ["Read"]

    def test_remove_all_tools(self):
        from src.registry.sdk_utils import remove_tools

        config = self._base_config(["Read", "Bash"])
        updated = remove_tools(config, ["Read", "Bash"])
        assert updated.allowed_tools == []


class TestOptionsToConfig:
    """Test options_to_config with a dict (OpenCode-style) to avoid needing the real SDK."""

    def test_opencode_dict_options(self):
        from src.registry.sdk_utils import options_to_config

        options = {
            "system": "You are helpful.",
            "tools": {"read": True, "bash": True},
            "format": {"type": "json_schema"},
            "mode": "build",
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-4-6",
            "cwd": "/some/path",
        }
        config = options_to_config(options, name="my-program")
        assert config.name == "my-program"
        assert "read" in config.allowed_tools
        assert "bash" in config.allowed_tools
        assert config.metadata["sdk"] == "opencode"

    def test_created_at_added_to_metadata(self):
        from src.registry.sdk_utils import options_to_config

        options = {"system": "x", "tools": {}}
        config = options_to_config(options, name="prog")
        assert "created_at" in config.metadata

    def test_custom_metadata_merged(self):
        from src.registry.sdk_utils import options_to_config

        options = {"system": "x", "tools": {}}
        config = options_to_config(options, name="prog", metadata={"custom": "val"})
        assert config.metadata["custom"] == "val"

    def test_parent_and_generation_set(self):
        from src.registry.sdk_utils import options_to_config

        options = {"system": "x", "tools": {}}
        config = options_to_config(
            options, name="child", parent="program/base", generation=2
        )
        assert config.parent == "program/base"
        assert config.generation == 2

    def test_tools_list_converted_to_allowed_tools(self):
        from src.registry.sdk_utils import options_to_config

        # tools can also be a list
        options = {"system": "x", "tools": ["read", "bash"]}
        config = options_to_config(options, name="prog")
        assert "read" in config.allowed_tools


class TestConfigToOptions:
    """Test config_to_options for the opencode path (dict return)."""

    def _make_opencode_config(self, tools=None):
        from src.registry.models import ProgramConfig

        return ProgramConfig(
            name="base",
            system_prompt={"type": "text", "content": "You are helpful."},
            allowed_tools=tools or ["Read"],
            metadata={"sdk": "opencode", "mode": "build", "provider_id": "anthropic", "model_id": "claude-sonnet-4-6"},
        )

    def test_opencode_config_returns_dict(self):
        from src.registry.sdk_utils import config_to_options

        config = self._make_opencode_config()
        result = config_to_options(config, cwd="/tmp")
        assert isinstance(result, dict)

    def test_opencode_dict_has_correct_system(self):
        from src.registry.sdk_utils import config_to_options

        config = self._make_opencode_config()
        result = config_to_options(config, cwd="/tmp")
        assert "You are helpful." in result["system"]

    def test_opencode_dict_tools_mapped(self):
        from src.registry.sdk_utils import config_to_options

        config = self._make_opencode_config(tools=["Read", "Bash"])
        result = config_to_options(config, cwd="/tmp")
        assert result["tools"] == {"Read": True, "Bash": True}

    def test_opencode_dict_cwd_set(self):
        from src.registry.sdk_utils import config_to_options

        config = self._make_opencode_config()
        result = config_to_options(config, cwd="/custom/path")
        assert result["cwd"] == "/custom/path"
