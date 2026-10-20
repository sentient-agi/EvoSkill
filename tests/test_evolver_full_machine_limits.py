"""The evolver's container may take the whole pool machine (8 CPUs, 32 GB) with a much higher
cpu.weight than the rollouts beside it. RUNTIME evolver_daytona.container takes cpu_weight and mem_high_ratio."""
import pytest

from src.orchestrator import pool_container_evolver as pce


def test_full_machine_limits_map_to_the_controller_names():
    out = pce.container_limits_cfg({"container": {"cpus": 8, "mem_max_gib": 32, "cpu_weight": 10000,
                                                  "mem_high_ratio": 0.85, "pids_max": 2048}})
    assert out == {"cpus": 8.0, "mem_max_bytes": 32 << 30, "cpu_weight": 10000, "mem_high_ratio": 0.85,
                   "pids_max": 2048}


@pytest.mark.parametrize("bad", [{"cpu_weight": 0}, {"cpu_weight": 10001}, {"cpu_weight": 1.5}, {"cpu_weight": True},
                                 {"mem_high_ratio": 0.2}, {"mem_high_ratio": 1.2}, {"mem_high_ratio": "0.9"},
                                 {"cpu_shares": 5}])
def test_bad_values_refuse(bad):
    with pytest.raises(ValueError):
        pce.container_limits_cfg({"container": bad})


def test_absent_keys_leave_the_controller_defaults():
    assert pce.container_limits_cfg({"container": {"cpus": 2, "mem_max_gib": 6}}) == {"cpus": 2.0, "mem_max_bytes": 6 << 30}
