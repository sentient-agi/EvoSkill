"""Offline deterministic smoke test for RQGM evaluator co-evolution."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import (  # noqa: E402
    AnchorExample,
    CallableEvaluatorExecutor,
    CallableEvaluatorProposer,
    CheckpointSchedule,
    EvaluatorCandidate,
    EvaluatorSlot,
    InMemoryAnchorProvider,
    RQGMController,
    RQGMRuntime,
    RoleTask,
)


def build_runtime(_config=None) -> RQGMRuntime:
    """The same synchronous runtime-factory shape used by the CLI."""
    seed = EvaluatorCandidate.create(
        slot_id="judge",
        role_id="solver",
        artifact={"accept_through": 1},
        source="seed",
        candidate_id="seed-judge",
    )
    anchor = InMemoryAnchorProvider(
        "toy-private-anchor",
        "v1",
        [AnchorExample(str(value), value, True) for value in range(1, 6)],
    )

    async def judge(candidate, artifact, expected, _context):
        predicted = artifact <= candidate.artifact["accept_through"]
        return predicted == expected

    proposer = CallableEvaluatorProposer(
        lambda incumbent, slot_id, checkpoint, public_context: [
            EvaluatorCandidate.create(
                slot_id=slot_id,
                role_id=incumbent.role_id,
                artifact={"accept_through": 5},
                parent_id=incumbent.id,
                source="toy-workspace",
                candidate_id=f"judge-checkpoint-{checkpoint}",
            )
        ]
    )
    return RQGMRuntime(
        slots=[
            EvaluatorSlot(
                "judge",
                "solver",
                seed,
                anchor,
                CallableEvaluatorExecutor(judge),
                proposer,
            )
        ],
        role_tasks=[
            RoleTask("solver", "learned-check", "learned", "judge"),
            RoleTask("solver", "fixed-check", "fixed"),
        ],
    )


async def run_demo() -> dict:
    runtime = build_runtime()
    controller = RQGMController(
        slots=runtime.slots,
        role_tasks=runtime.role_tasks,
        minimum_node_outcomes=1,
        checkpoint_schedule=CheckpointSchedule(explicit=(2,)),
    )
    await controller.evaluate_learned(
        slot_id="judge",
        node_id="toy-node",
        role_id="solver",
        task_id="learned-check",
        artifact=1,
        expected=True,
    )
    controller.record_fixed_outcome(
        node_id="toy-node",
        role_id="solver",
        task_id="fixed-check",
        raw_score=1.0,
    )
    event = await controller.checkpoint()
    await controller.evaluate_learned(
        slot_id="judge",
        node_id="toy-node",
        role_id="solver",
        task_id="learned-check",
        artifact=5,
        expected=True,
    )
    result = {
        "replacement": event["slots"]["judge"]["replacement"],
        "epoch_vector": controller.epoch_vector,
        "invalidated_records": sum(
            not record.valid for record in controller.ledger.records
        ),
        "fixed_records_preserved": sum(
            record.valid and record.criterion_kind == "fixed"
            for record in controller.ledger.records
        ),
        "endpoint_score": controller.node_selection("toy-node").score,
    }
    assert result["replacement"] is True
    assert result["epoch_vector"] == {"judge": 2}
    assert result["invalidated_records"] == 1
    assert result["fixed_records_preserved"] == 1
    return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), indent=2))
