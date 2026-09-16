"""Test that update_frontier handles empty scored list gracefully.

Regression test for: https://github.com/sentient-agi/EvoSkill/issues/60
"""

import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch

import yaml

from src.registry.manager import ProgramManager
from src.registry.models import ProgramConfig


def _init_repo(tmp):
    subprocess.run(["git", "init"], cwd=tmp, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp, capture_output=True)


def test_update_frontier_no_scored_programs():
    """update_frontier should not crash when frontier has no scored programs.
    
    This directly tests the fix for the IndexError that occurred when
    get_frontier_with_scores() returned an empty list and scored[-1] was accessed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _init_repo(tmp)
        manager = ProgramManager(tmp)
        
        # Create base program (no score) and mark it frontier
        base_config = ProgramConfig(name="base", system_prompt={"type": "default"}, allowed_tools=[])
        manager.create_program("base", base_config)
        manager.mark_frontier("base")
        
        # Create the iter-1 program branch (update_frontier will commit to it)
        iter_config = ProgramConfig(name="iter-1", system_prompt={"type": "default"}, allowed_tools=[], parent="program/base")
        manager.create_program("iter-1", iter_config, parent="base")
        
        # Mock get_frontier_with_scores to return empty (simulating all-unscored frontier)
        # This is the exact scenario that caused the IndexError before the fix
        with patch.object(manager, 'get_frontier_with_scores', return_value=[]):
            with patch.object(manager, 'mark_frontier') as mock_mark:
                result = manager.update_frontier("iter-1", 0.85)
                assert result is True
                mock_mark.assert_called_once_with("iter-1")


def test_update_frontier_mixed_scored_and_unscored():
    """update_frontier should skip unscored programs and compare against scored ones."""
    with tempfile.TemporaryDirectory() as tmp:
        _init_repo(tmp)
        manager = ProgramManager(tmp)
        
        # Create base (unscored) and mark frontier
        base_config = ProgramConfig(name="base", system_prompt={"type": "default"}, allowed_tools=[])
        manager.create_program("base", base_config)
        manager.mark_frontier("base")
        
        # Create iter-1 program branch
        iter_config = ProgramConfig(name="iter-1", system_prompt={"type": "default"}, allowed_tools=[], parent="program/base")
        manager.create_program("iter-1", iter_config, parent="base")
        
        # Mock: get_frontier_with_scores returns one scored program (not base)
        with patch.object(manager, 'get_frontier_with_scores', return_value=[("other", 0.5)]):
            with patch.object(manager, 'mark_frontier'):
                # 0.85 > 0.5, so should succeed
                result = manager.update_frontier("iter-1", 0.85)
                assert result is True
