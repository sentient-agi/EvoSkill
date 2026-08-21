from types import SimpleNamespace
import importlib

import pytest

from src.evaluation.evaluate import EvalResult
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
)
from src.loop.config import LoopConfig
from src.loop.runner import SelfImprovingLoop
from src.schemas import AgentResponse


class FakeManager:
    def __init__(self, cwd):
        self.cwd = cwd
        self.invalidated = []

    def get_current_name(self):
        return "node"

    def invalidate_scores(self, names):
        self.invalidated.extend(names)


def make_controller() -> RQGMController:
    seed = EvaluatorCandidate.create(
        slot_id="judge", role_id="solver", artifact={"quality": 1}, candidate_id="seed"
    )
    anchor = InMemoryAnchorProvider(
        "private", "v1", [AnchorExample(str(i), i, True) for i in range(1, 6)]
    )

    async def judge(evaluator, artifact, expected, context):
        if context.get("anchor"):
            return artifact <= evaluator.artifact["quality"]
        return artifact == expected

    proposer = CallableEvaluatorProposer(
        lambda incumbent, slot_id, checkpoint, context: [
            EvaluatorCandidate.create(
                slot_id=slot_id,
                role_id=incumbent.role_id,
                artifact={"quality": 5},
                parent_id=incumbent.id,
                candidate_id="challenger",
            )
        ]
    )
    evaluator_slot = EvaluatorSlot(
        "judge", "solver", seed, anchor, CallableEvaluatorExecutor(judge), proposer
    )
    return RQGMController(
        slots=[evaluator_slot],
        role_tasks=[
            RoleTask("solver", "learned", "learned", "judge"),
            RoleTask("solver", "fixed", "fixed"),
        ],
        minimum_node_outcomes=1,
        checkpoint_schedule=CheckpointSchedule(explicit=(2,)),
    )


def make_loop(tmp_path, controller):
    return SelfImprovingLoop(
        LoopConfig(cache_enabled=False, concurrency=1),
        SimpleNamespace(base=object()),
        FakeManager(tmp_path),
        train_pools={"learned": [("q", "ok")]},
        val_data=[
            ("learned question", "ok", "learned"),
            ("fixed question", "ok", "fixed"),
        ],
        scorer=lambda _question, predicted, expected: float(predicted == expected),
        evaluator_controller=controller,
    )


@pytest.mark.asyncio
async def test_runner_rejudges_only_replaced_slot(monkeypatch, tmp_path):
    controller = make_controller()
    loop = make_loop(tmp_path, controller)

    async def fake_evaluate(_agent, items, **_kwargs):
        return [
            EvalResult(
                question=question,
                ground_truth=expected,
                trace=SimpleNamespace(
                    output=AgentResponse(final_answer="ok", reasoning=""),
                    total_cost_usd=0.0,
                ),
            )
            for question, expected in items
        ]

    runner_module = importlib.import_module("src.loop.runner")
    monkeypatch.setattr(runner_module, "evaluate_agent_parallel", fake_evaluate)
    score = await loop._evaluate(loop.val_data)

    assert score > 0
    assert controller.validation_outcomes == 3
    assert controller.epoch_vector == {"judge": 2}
    fixed = [r for r in controller.ledger.records if r.criterion_kind == "fixed"]
    learned = [r for r in controller.ledger.records if r.criterion_kind == "learned"]
    assert len(fixed) == 1 and fixed[0].valid
    assert len(learned) == 2
    assert sum(r.valid for r in learned) == 1
    assert controller.pending_revaluation == set()


@pytest.mark.asyncio
async def test_training_feedback_does_not_enter_validation_ledger(tmp_path):
    controller = make_controller()
    loop = make_loop(tmp_path, controller)
    score = await loop._score_feedback_answer("q", "ok", "ok", "learned")
    assert score == 1.0
    assert controller.validation_outcomes == 0
    assert controller.ledger.records == []


@pytest.mark.asyncio
async def test_training_fixed_scorer_keeps_existing_normalization(tmp_path):
    plain_loop = make_loop(tmp_path, None)
    assert await plain_loop._score_feedback_answer("q", " OK ", "ok", "learned") == 1.0

    controller = make_controller()
    rqgm_loop = make_loop(tmp_path, controller)
    assert await rqgm_loop._score_feedback_answer("q", " OK ", "ok", "fixed") == 1.0
