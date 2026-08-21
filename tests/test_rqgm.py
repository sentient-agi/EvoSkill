import json
from pathlib import Path

import pytest

from src.evaluation.rqgm import (
    AnchorExample,
    CallableEvaluatorExecutor,
    CallableEvaluatorProposer,
    CheckpointSchedule,
    EvaluatorCandidate,
    EvaluatorSlot,
    InMemoryAnchorProvider,
    RQGMController,
    RoleTask,
    beta_lower_belief,
)


def candidate(slot: str, role: str, quality: int, name: str) -> EvaluatorCandidate:
    return EvaluatorCandidate.create(
        slot_id=slot, role_id=role, artifact={"quality": quality}, candidate_id=name
    )


def executor():
    # A quality-q evaluator accepts anchor artifacts <= q.
    return CallableEvaluatorExecutor(
        lambda evaluator, artifact, expected, context: float(
            (artifact <= evaluator.artifact["quality"]) == expected
        )
    )


def provider(name: str, count: int = 5) -> InMemoryAnchorProvider:
    return InMemoryAnchorProvider(
        name,
        "v1",
        [AnchorExample(str(i), i, True) for i in range(1, count + 1)],
    )


def slot(slot_id: str, role: str, quality: int, proposer=None, count: int = 5):
    return EvaluatorSlot(
        slot_id,
        role,
        candidate(slot_id, role, quality, f"{slot_id}-seed"),
        provider(f"anchor-{slot_id}", count),
        executor(),
        proposer,
    )


def test_beta_lower_belief_uses_exact_posterior_quantile():
    assert beta_lower_belief(3, 0) == pytest.approx(0.05**0.25)
    assert beta_lower_belief(0, 3) == pytest.approx(1 - 0.95**0.25)


def test_anchor_provider_is_defensively_immutable_and_candidates_are_serializable():
    anchors = InMemoryAnchorProvider(
        "immutable", "v1", [AnchorExample("x", {"secret": [1]}, True)]
    )
    digest = anchors.digest
    anchors.examples()[0].artifact["secret"].append(2)
    assert anchors.digest == digest
    assert anchors.examples()[0].artifact == {"secret": [1]}

    with pytest.raises(TypeError, match="JSON-serializable"):
        EvaluatorCandidate.create(
            slot_id="s", role_id="r", artifact={"callable": lambda: None}
        )


def test_checkpoint_schedule_caps_batches_exactly():
    controller = RQGMController(
        slots=[slot("review", "writer", 5)],
        role_tasks=[RoleTask("writer", "paper", "learned", "review")],
        checkpoint_schedule=CheckpointSchedule(explicit=(2, 4, 8)),
    )
    assert controller.batch_capacity(10) == 2
    controller.validation_outcomes = 2
    assert controller.checkpoint_due()
    assert controller.batch_capacity(10) == 2


@pytest.mark.asyncio
async def test_two_slots_replace_atomically_and_selectively_erase():
    def make_proposer(quality):
        return CallableEvaluatorProposer(
            lambda incumbent, slot_id, checkpoint, context: [
                candidate(slot_id, incumbent.role_id, quality, f"{slot_id}-better")
            ]
        )

    controller = RQGMController(
        slots=[
            slot("review", "writer", 1, make_proposer(5)),
            slot("grade", "prover", 1, make_proposer(5)),
        ],
        role_tasks=[
            RoleTask("writer", "paper", "learned", "review"),
            RoleTask("prover", "proof", "learned", "grade"),
            RoleTask("coder", "tests", "fixed"),
        ],
        minimum_node_outcomes=1,
        checkpoint_schedule=CheckpointSchedule(explicit=(3,)),
    )
    await controller.evaluate_learned(
        slot_id="review",
        node_id="n1",
        role_id="writer",
        task_id="paper",
        artifact=1,
        expected=True,
    )
    await controller.evaluate_learned(
        slot_id="grade",
        node_id="n1",
        role_id="prover",
        task_id="proof",
        artifact=1,
        expected=True,
    )
    fixed = controller.record_fixed_outcome(
        node_id="n1", role_id="coder", task_id="tests", raw_score=1.0
    )
    event = await controller.checkpoint()

    assert event["epoch_vector"] == {"review": 2, "grade": 2}
    assert event["slots"]["review"]["replacement"] is True
    assert event["slots"]["grade"]["replacement"] is True
    assert fixed.valid is True
    assert sum(not record.valid for record in controller.ledger.records) == 2
    assert controller.node_selection("n1") is None
    assert controller.pending_revaluation == {"n1"}


@pytest.mark.asyncio
async def test_tie_keeps_incumbent_and_insufficient_evidence_is_invalid():
    tie = CallableEvaluatorProposer(
        lambda incumbent, slot_id, checkpoint, context: [
            candidate(slot_id, incumbent.role_id, 5, "tie")
        ]
    )
    controller = RQGMController(
        slots=[slot("review", "writer", 5, tie, count=3)],
        role_tasks=[RoleTask("writer", "paper", "learned", "review")],
        minimum_anchor_outcomes=5,
    )
    event = await controller.checkpoint()
    assert event["slots"]["review"]["replacement"] is False
    assert controller.slots["review"].archive[-1].status == "invalid"


@pytest.mark.asyncio
async def test_anchor_execution_errors_count_as_failures_not_missing_evidence():
    async def judge(evaluator, artifact, expected, context):
        if evaluator.artifact.get("throws") and artifact > 5:
            raise RuntimeError("selective crash")
        return True

    challenger = EvaluatorCandidate.create(
        slot_id="review",
        role_id="writer",
        artifact={"throws": True},
        candidate_id="crasher",
    )
    evaluator_slot = EvaluatorSlot(
        "review",
        "writer",
        EvaluatorCandidate.create(
            slot_id="review", role_id="writer", artifact={}, candidate_id="safe"
        ),
        provider("ten", count=10),
        CallableEvaluatorExecutor(judge),
        CallableEvaluatorProposer(lambda *_args: [challenger]),
    )
    controller = RQGMController(
        slots=[evaluator_slot],
        role_tasks=[RoleTask("writer", "paper", "learned", "review")],
    )
    event = await controller.checkpoint()
    assert event["slots"]["review"]["replacement"] is False
    assert challenger.anchor_successes == 5
    assert challenger.anchor_failures == 5
    assert challenger.anchor_invalid == 5


@pytest.mark.asyncio
async def test_proposer_failure_keeps_incumbent():
    def broken(*args):
        raise RuntimeError("provider unavailable")

    controller = RQGMController(
        slots=[slot("review", "writer", 5, CallableEvaluatorProposer(broken))],
        role_tasks=[RoleTask("writer", "paper", "learned", "review")],
    )
    event = await controller.checkpoint()
    assert event["slots"]["review"]["replacement"] is False
    assert "provider unavailable" in event["slots"]["review"]["proposal_error"]


@pytest.mark.asyncio
async def test_proposer_context_cannot_see_anchor_evidence_or_examples():
    seen = {}

    def inspect_context(incumbent, slot_id, checkpoint, context):
        seen.update(context)
        return []

    controller = RQGMController(
        slots=[slot("review", "writer", 5, CallableEvaluatorProposer(inspect_context))],
        role_tasks=[RoleTask("writer", "paper", "learned", "review")],
    )
    await controller.checkpoint()
    serialized = json.dumps(seen)
    assert "anchor_successes" not in serialized
    assert "anchor_failures" not in serialized
    assert "PRIVATE" not in serialized


@pytest.mark.asyncio
async def test_workspace_registered_challenger_is_considered_once():
    controller = RQGMController(
        slots=[slot("review", "writer", 1)],
        role_tasks=[RoleTask("writer", "paper", "learned", "review")],
    )
    challenger = candidate("review", "writer", 5, "workspace-node-evaluator")
    controller.register_challenger(challenger)
    event = await controller.checkpoint()
    assert event["slots"]["review"]["promoted"] == challenger.id
    assert challenger.created_checkpoint == 1
    with pytest.raises(ValueError, match="duplicate evaluator candidate ID"):
        controller.register_challenger(challenger)


@pytest.mark.asyncio
async def test_selective_erasure_preserves_unchanged_cells():
    better = CallableEvaluatorProposer(
        lambda incumbent, slot_id, checkpoint, context: [
            candidate(slot_id, incumbent.role_id, 5, "better")
        ]
    )
    controller = RQGMController(
        slots=[slot("review", "writer", 1, better)],
        role_tasks=[
            RoleTask("writer", "learned", "learned", "review"),
            RoleTask("writer", "fixed", "fixed"),
        ],
        minimum_node_outcomes=1,
    )
    await controller.evaluate_learned(
        slot_id="review",
        node_id="n",
        role_id="writer",
        task_id="learned",
        artifact=1,
        expected=True,
    )
    controller.record_fixed_outcome(
        node_id="n",
        role_id="writer",
        task_id="fixed",
        raw_score=1.0,
    )
    await controller.checkpoint()
    assert not controller.cell_is_current("n", controller.role_tasks[0])
    assert controller.cell_is_current("n", controller.role_tasks[1])
    assert controller.pending_revaluation == {"n"}


def test_uniform_task_role_endpoints_and_pooled_working_posterior():
    controller = RQGMController(
        slots=[slot("review", "writer", 5)],
        role_tasks=[
            RoleTask("writer", "a", "fixed"),
            RoleTask("writer", "b", "fixed"),
            RoleTask("coder", "c", "fixed"),
        ],
        minimum_node_outcomes=1,
    )
    for node, scores in {
        "general": (1, 0, 1),
        "writer-specialist": (1, 1, 0),
    }.items():
        for task, score in zip(("a", "b", "c"), scores):
            role = "writer" if task in ("a", "b") else "coder"
            controller.record_fixed_outcome(
                node_id=node, role_id=role, task_id=task, raw_score=score
            )
    generalist, specialists = controller.select_endpoints(
        ["general", "writer-specialist"]
    )
    assert generalist is not None
    selection = controller.node_selection("general")
    assert selection.pooled_score == pytest.approx(beta_lower_belief(2, 1))
    expected_writer = (beta_lower_belief(1, 0) + beta_lower_belief(0, 1)) / 2
    expected_coder = beta_lower_belief(1, 0)
    assert selection.score == pytest.approx((expected_writer + expected_coder) / 2)
    assert specialists["writer"].node_id == "writer-specialist"
    assert specialists["coder"].node_id == "general"


def test_least_measured_scheduler_balances_role_then_task():
    controller = RQGMController(
        slots=[slot("review", "writer", 5)],
        role_tasks=[
            RoleTask("writer", "a", "fixed"),
            RoleTask("writer", "b", "fixed"),
            RoleTask("coder", "c", "fixed"),
        ],
        minimum_node_outcomes=1,
    )
    first = controller.least_measured_cell("n", {"a", "b", "c"})
    assert first == RoleTask("coder", "c", "fixed")
    controller.record_fixed_outcome(
        node_id="n", role_id="coder", task_id="c", raw_score=1
    )
    second = controller.least_measured_cell("n", {"a", "b", "c"})
    assert second == RoleTask("writer", "a", "fixed")


@pytest.mark.asyncio
async def test_state_restore_checks_anchor_hash(tmp_path: Path):
    state = tmp_path / "rqgm.json"
    original = slot("review", "writer", 5)
    controller = RQGMController(
        slots=[original],
        role_tasks=[RoleTask("writer", "paper", "learned", "review")],
        state_path=state,
    )
    await controller.checkpoint()
    restored = RQGMController.restore(
        state,
        runtime_slots=[slot("review", "writer", 5)],
    )
    assert restored.to_dict() == controller.to_dict()

    changed = slot("review", "writer", 5, count=6)
    with pytest.raises(ValueError, match="anchor metadata mismatch"):
        RQGMController.restore(state, runtime_slots=[changed])


def test_serialized_state_does_not_contain_anchor_examples(tmp_path: Path):
    state = tmp_path / "rqgm.json"
    secret = "PRIVATE-ANCHOR-CONTENT"
    private_provider = InMemoryAnchorProvider(
        "secret", "v1", [AnchorExample(str(i), secret, True) for i in range(5)]
    )
    private_slot = EvaluatorSlot(
        "review",
        "writer",
        candidate("review", "writer", 5, "seed"),
        private_provider,
        executor(),
    )
    RQGMController(
        slots=[private_slot],
        role_tasks=[RoleTask("writer", "paper", "learned", "review")],
        state_path=state,
    )
    assert secret not in state.read_text(encoding="utf-8")
    assert json.loads(state.read_text(encoding="utf-8"))["slots"]["review"]["anchor"][
        "digest"
    ]
