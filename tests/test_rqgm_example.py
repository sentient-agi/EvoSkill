import importlib.util
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_offline_rqgm_example_runs_end_to_end():
    path = Path(__file__).resolve().parents[1] / "examples" / "rqgm_toy" / "run.py"
    spec = importlib.util.spec_from_file_location("rqgm_toy_example", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = await module.run_demo()
    assert result["replacement"] is True
    assert result["epoch_vector"] == {"judge": 2}
