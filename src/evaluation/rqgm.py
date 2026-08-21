"""RQGM controlled evaluator evolution for EvoSkill.

The outer search remains EvoSkill's frontier search.  This module implements
epoch-frozen evaluators, anchor-only checkpoint replacement, slot-local
selective erasure, and current-epoch best-belief selection.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal, Protocol, Sequence

CriterionKind = Literal["fixed", "learned"]
CandidateStatus = Literal["pending", "valid", "invalid", "incumbent"]


def beta_lower_belief(successes: int, failures: int, epsilon: float = 0.05) -> float:
    """Return the epsilon quantile of Beta(1+successes, 1+failures)."""
    if successes < 0 or failures < 0 or not 0 < epsilon < 1:
        raise ValueError("invalid Beta posterior arguments")
    try:
        from scipy.stats import beta
    except ImportError as exc:
        raise RuntimeError(
            "RQGM requires SciPy; empirical accuracy is not best-belief"
        ) from exc
    value = float(beta.ppf(epsilon, 1 + successes, 1 + failures))
    if not math.isfinite(value):
        raise RuntimeError("non-finite Beta posterior quantile")
    return value


@dataclass(frozen=True)
class OutcomeAdapter:
    threshold: float = 0.8

    def __call__(self, raw_score: float | bool) -> bool:
        value = float(raw_score)
        if not math.isfinite(value):
            raise ValueError("outcome score must be finite")
        return value >= self.threshold


@dataclass(frozen=True)
class AnchorExample:
    item_id: str
    artifact: Any
    expected: Any


class AnchorProvider(Protocol):
    provider_id: str
    version: str

    @property
    def digest(self) -> str: ...
    def examples(self) -> Sequence[AnchorExample]: ...


class InMemoryAnchorProvider:
    """Private, hash-pinned anchor provider for adapters and tests."""

    def __init__(
        self, provider_id: str, version: str, examples: Sequence[AnchorExample]
    ) -> None:
        if not examples:
            raise ValueError("anchor provider cannot be empty")
        self.provider_id, self.version = provider_id, version
        self._examples = copy.deepcopy(tuple(examples))
        canonical = json.dumps(
            [asdict(item) for item in self._examples],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        self._digest = hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def digest(self) -> str:
        return self._digest

    def examples(self) -> Sequence[AnchorExample]:
        return copy.deepcopy(self._examples)


@dataclass
class EvaluatorCandidate:
    id: str
    slot_id: str
    role_id: str
    artifact: dict[str, Any]
    parent_id: str | None = None
    workspace_node: str | None = None
    source: str = "seed"
    created_checkpoint: int = 0
    runtime: dict[str, Any] = field(default_factory=dict)
    status: CandidateStatus = "pending"
    anchor_successes: int = 0
    anchor_failures: int = 0
    anchor_invalid: int = 0
    best_belief: float | None = None

    @classmethod
    def create(
        cls,
        *,
        slot_id: str,
        role_id: str,
        artifact: dict[str, Any],
        parent_id: str | None = None,
        workspace_node: str | None = None,
        source: str = "proposer",
        created_checkpoint: int = 0,
        runtime: dict[str, Any] | None = None,
        candidate_id: str | None = None,
    ) -> "EvaluatorCandidate":
        artifact_copy, runtime_copy = (
            copy.deepcopy(dict(artifact)),
            copy.deepcopy(dict(runtime or {})),
        )
        try:
            json.dumps(
                {"artifact": artifact_copy, "runtime": runtime_copy}, ensure_ascii=False
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "evaluator candidate artifact/runtime must be JSON-serializable"
            ) from exc
        return cls(
            candidate_id or str(uuid.uuid4()),
            slot_id,
            role_id,
            artifact_copy,
            parent_id,
            workspace_node,
            source,
            created_checkpoint,
            runtime_copy,
        )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def proposer_dict(self) -> dict[str, Any]:
        """Candidate context without held-out anchor evidence."""
        return {
            "id": self.id,
            "slot_id": self.slot_id,
            "role_id": self.role_id,
            "artifact": self.artifact,
            "parent_id": self.parent_id,
            "workspace_node": self.workspace_node,
            "source": self.source,
            "runtime": self.runtime,
        }


RQGMEvaluator = EvaluatorCandidate


class EvaluatorExecutor(Protocol):
    async def judge(
        self,
        candidate: EvaluatorCandidate,
        *,
        artifact: Any,
        expected: Any,
        context: dict[str, Any],
    ) -> float | bool: ...


class CallableEvaluatorExecutor:
    def __init__(
        self, fn: Callable[..., float | bool | Awaitable[float | bool]]
    ) -> None:
        self.fn = fn

    async def judge(
        self,
        candidate: EvaluatorCandidate,
        *,
        artifact: Any,
        expected: Any,
        context: dict[str, Any],
    ) -> float | bool:
        result = self.fn(candidate, artifact, expected, context)
        return await result if inspect.isawaitable(result) else result


class EvaluatorProposer(Protocol):
    async def propose(
        self,
        *,
        incumbent: EvaluatorCandidate,
        slot_id: str,
        checkpoint_index: int,
        public_context: dict[str, Any],
    ) -> Iterable[EvaluatorCandidate]: ...


class CallableEvaluatorProposer:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    async def propose(
        self,
        *,
        incumbent: EvaluatorCandidate,
        slot_id: str,
        checkpoint_index: int,
        public_context: dict[str, Any],
    ) -> Iterable[EvaluatorCandidate]:
        result = self.fn(incumbent, slot_id, checkpoint_index, public_context)
        return tuple(await result if inspect.isawaitable(result) else result)


class AgentEvaluatorProposer:
    """Use an EvoSkill provider-backed Agent to propose evaluator artifacts.

    ``parser`` converts the agent's final text into candidate artifacts.  The
    prompt contains aggregate anchor evidence only; private examples never
    enter the proposer context.
    """

    def __init__(
        self,
        agent: Any,
        parser: Callable[[str, EvaluatorCandidate, int], Iterable[dict[str, Any]]],
    ) -> None:
        self.agent, self.parser = agent, parser

    async def propose(
        self,
        *,
        incumbent: EvaluatorCandidate,
        slot_id: str,
        checkpoint_index: int,
        public_context: dict[str, Any],
    ) -> Iterable[EvaluatorCandidate]:
        prompt = (
            "Propose improved evaluator artifacts for the frozen slot. "
            "Do not request or infer held-out anchor examples. Return only the "
            "format required by the evaluator parser.\n"
            + json.dumps(public_context, ensure_ascii=False, default=str)
        )
        trace = await self.agent.run(prompt)
        if trace.output is None:
            raise RuntimeError("evaluator proposer returned no parsed output")
        text = getattr(trace.output, "final_answer", None) or str(trace.output)
        return tuple(
            EvaluatorCandidate.create(
                slot_id=slot_id,
                role_id=incumbent.role_id,
                artifact=artifact,
                parent_id=incumbent.id,
                workspace_node=incumbent.workspace_node,
                source="evoskill-agent",
                created_checkpoint=checkpoint_index,
            )
            for artifact in self.parser(text, incumbent, checkpoint_index)
        )


class SandboxedEvaluatorExecutor:
    """Delegate evaluator execution to a capability-restricted backend."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    async def judge(
        self,
        candidate: EvaluatorCandidate,
        *,
        artifact: Any,
        expected: Any,
        context: dict[str, Any],
    ) -> float | bool:
        policy = {
            "network": False,
            "filesystem": "ephemeral",
            "persist_outputs": False,
            "anchor": bool(context.get("anchor")),
        }
        result = self.backend.run_evaluator(
            evaluator_artifact=candidate.artifact,
            task_artifact=artifact,
            expected=expected,
            policy=policy,
        )
        return await result if inspect.isawaitable(result) else result


@dataclass
class UtilityRecord:
    record_id: str
    node_id: str
    role_id: str
    task_id: str
    criterion_kind: CriterionKind
    raw_score: float
    outcome: bool
    artifact_id: str | None = None
    slot_id: str | None = None
    epoch_id: int | None = None
    valid: bool = True
    invalidated_at_checkpoint: int | None = None
    invalidation_reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        node_id: str,
        role_id: str,
        task_id: str,
        criterion_kind: CriterionKind,
        raw_score: float,
        outcome: bool,
        artifact_id: str | None = None,
        slot_id: str | None = None,
        epoch_id: int | None = None,
    ) -> "UtilityRecord":
        if criterion_kind == "learned" and (slot_id is None or epoch_id is None):
            raise ValueError("learned utility requires slot and epoch")
        if criterion_kind == "fixed" and slot_id is not None:
            raise ValueError("fixed utility cannot depend on a slot")
        return cls(
            str(uuid.uuid4()),
            node_id,
            role_id,
            task_id,
            criterion_kind,
            float(raw_score),
            bool(outcome),
            artifact_id,
            slot_id,
            epoch_id,
        )


class UtilityLedger:
    """Append-only evidence; erasure is an auditable tombstone."""

    def __init__(self, records: Iterable[UtilityRecord] = ()) -> None:
        self.records = list(records)

    def append(self, record: UtilityRecord) -> UtilityRecord:
        self.records.append(record)
        return record

    def invalidate_slot(self, slot_id: str, checkpoint: int) -> int:
        count = 0
        for record in self.records:
            if (
                record.valid
                and record.criterion_kind == "learned"
                and record.slot_id == slot_id
            ):
                record.valid = False
                record.invalidated_at_checkpoint = checkpoint
                record.invalidation_reason = f"slot {slot_id} transitioned"
                count += 1
        return count

    def valid_records(
        self,
        *,
        node_id: str | None = None,
        role_id: str | None = None,
        task_id: str | None = None,
    ) -> list[UtilityRecord]:
        return [
            r
            for r in self.records
            if r.valid
            and (node_id is None or r.node_id == node_id)
            and (role_id is None or r.role_id == role_id)
            and (task_id is None or r.task_id == task_id)
        ]


@dataclass(frozen=True)
class RoleTask:
    role_id: str
    task_id: str
    criterion_kind: CriterionKind
    slot_id: str | None = None

    def __post_init__(self) -> None:
        if self.criterion_kind == "learned" and self.slot_id is None:
            raise ValueError("learned cell requires slot_id")
        if self.criterion_kind == "fixed" and self.slot_id is not None:
            raise ValueError("fixed cell cannot have slot_id")


@dataclass
class EvaluatorSlot:
    slot_id: str
    role_id: str
    incumbent: EvaluatorCandidate
    anchor_provider: AnchorProvider
    executor: EvaluatorExecutor
    proposer: EvaluatorProposer | None = None
    outcome_adapter: OutcomeAdapter = field(default_factory=OutcomeAdapter)
    epoch: int = 1
    archive: list[EvaluatorCandidate] = field(default_factory=list)
    last_checkpoint: int = 0
    _anchor_pin: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (self.incumbent.slot_id, self.incumbent.role_id) != (
            self.slot_id,
            self.role_id,
        ):
            raise ValueError("incumbent does not match slot")
        if not self.archive:
            self.archive.append(self.incumbent)
        self.incumbent.status = "incumbent"
        self._anchor_pin = self._current_anchor_metadata()

    def _current_anchor_metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.anchor_provider.provider_id,
            "version": self.anchor_provider.version,
            "digest": self.anchor_provider.digest,
            "count": len(self.anchor_provider.examples()),
        }

    def anchor_metadata(self) -> dict[str, Any]:
        return dict(self._anchor_pin)

    def verified_anchor_examples(self) -> Sequence[AnchorExample]:
        examples = self.anchor_provider.examples()
        current = {
            "provider_id": self.anchor_provider.provider_id,
            "version": self.anchor_provider.version,
            "digest": self.anchor_provider.digest,
            "count": len(examples),
        }
        if current != self._anchor_pin:
            raise RuntimeError(
                f"anchor provider changed during run for slot {self.slot_id}"
            )
        return examples


@dataclass(frozen=True)
class CheckpointSchedule:
    explicit: tuple[int, ...] = ()
    start: int = 32
    ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.explicit and tuple(sorted(set(self.explicit))) != self.explicit:
            raise ValueError(
                "explicit checkpoints must be positive, sorted, and unique"
            )
        if any(v <= 0 for v in self.explicit):
            raise ValueError("checkpoints must be positive")
        if not self.explicit and (self.start <= 0 or self.ratio <= 1):
            raise ValueError("invalid exponential checkpoint schedule")

    def boundaries_through(self, consumed: int) -> list[int]:
        if self.explicit:
            return [v for v in self.explicit if v <= consumed]
        values, value = [], self.start
        while value <= consumed:
            values.append(value)
            value = max(value + 1, int(math.ceil(value * self.ratio)))
        return values

    def next_after(self, consumed: int) -> int | None:
        if self.explicit:
            return next((v for v in self.explicit if v > consumed), None)
        value = self.start
        while value <= consumed:
            value = max(value + 1, int(math.ceil(value * self.ratio)))
        return value


@dataclass(frozen=True)
class NodeSelection:
    node_id: str
    score: float
    pooled_score: float
    role_scores: dict[str, float]


class RQGMController:
    STATE_VERSION = 3

    def __init__(
        self,
        *,
        slots: Sequence[EvaluatorSlot],
        role_tasks: Sequence[RoleTask],
        epsilon: float = 0.05,
        minimum_anchor_outcomes: int = 5,
        minimum_node_outcomes: int = 5,
        checkpoint_schedule: CheckpointSchedule | None = None,
        anchor_repeats: int = 1,
        state_path: str | Path | None = None,
        ledger: UtilityLedger | None = None,
    ) -> None:
        if not slots or not role_tasks:
            raise ValueError("RQGM requires slots and role/task cells")
        if (
            not 0 < epsilon < 1
            or min(minimum_anchor_outcomes, minimum_node_outcomes, anchor_repeats) <= 0
        ):
            raise ValueError("invalid RQGM evidence configuration")
        self.slots = {s.slot_id: s for s in slots}
        if len(self.slots) != len(slots):
            raise ValueError("duplicate slot ID")
        candidate_ids = [candidate.id for slot in slots for candidate in slot.archive]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("evaluator candidate IDs must be globally unique")
        self.role_tasks = tuple(role_tasks)
        for cell in role_tasks:
            if cell.slot_id and cell.slot_id not in self.slots:
                raise ValueError(f"unknown slot {cell.slot_id}")
        self.epsilon, self.minimum_anchor_outcomes = epsilon, minimum_anchor_outcomes
        self.minimum_node_outcomes, self.anchor_repeats = (
            minimum_node_outcomes,
            anchor_repeats,
        )
        self.checkpoint_schedule = checkpoint_schedule or CheckpointSchedule()
        self.state_path = Path(state_path) if state_path else None
        self.ledger = ledger or UtilityLedger()
        self.validation_outcomes = 0
        self.checkpoint_index = 0
        self.events: list[dict[str, Any]] = []
        self.pending_revaluation: set[str] = set()
        self.persist()

    @property
    def epoch_vector(self) -> dict[str, int]:
        return {k: s.epoch for k, s in self.slots.items()}

    @property
    def next_checkpoint(self) -> int | None:
        return self.checkpoint_schedule.next_after(self.validation_outcomes)

    def batch_capacity(self, requested: int) -> int:
        if requested < 0:
            raise ValueError("negative batch size")
        nxt = self.next_checkpoint
        return (
            requested if nxt is None else min(requested, nxt - self.validation_outcomes)
        )

    def checkpoint_due(self) -> bool:
        done = {
            e["outcome_budget"] for e in self.events if e.get("kind") == "checkpoint"
        }
        return any(
            v == self.validation_outcomes and v not in done
            for v in self.checkpoint_schedule.boundaries_through(
                self.validation_outcomes
            )
        )

    def record_fixed_outcome(
        self,
        *,
        node_id: str,
        role_id: str,
        task_id: str,
        raw_score: float,
        outcome_adapter: OutcomeAdapter | None = None,
        artifact_id: str | None = None,
    ) -> UtilityRecord:
        adapter = outcome_adapter or OutcomeAdapter()
        record = UtilityRecord.create(
            node_id=node_id,
            role_id=role_id,
            task_id=task_id,
            criterion_kind="fixed",
            raw_score=raw_score,
            outcome=adapter(raw_score),
            artifact_id=artifact_id,
        )
        self.ledger.append(record)
        self.validation_outcomes += 1
        if self.node_selection(node_id) is not None:
            self.pending_revaluation.discard(node_id)
        self.persist()
        return record

    def role_task_for(self, task_id: str) -> RoleTask:
        """Resolve an EvoSkill category/task to its configured RQGM cell."""
        matches = [cell for cell in self.role_tasks if cell.task_id == task_id]
        if len(matches) == 1:
            return matches[0]
        if not matches and len(self.role_tasks) == 1:
            return self.role_tasks[0]
        if not matches:
            raise KeyError(f"no RQGM role/task cell configured for {task_id!r}")
        raise ValueError(f"task ID {task_id!r} is ambiguous across roles")

    def register_challenger(self, candidate: EvaluatorCandidate) -> None:
        """Register a workspace-produced evaluator for the next checkpoint."""
        slot = self.slots.get(candidate.slot_id)
        if slot is None or candidate.role_id != slot.role_id:
            raise ValueError("challenger does not belong to a configured slot")
        if any(
            candidate.id == existing.id
            for existing_slot in self.slots.values()
            for existing in existing_slot.archive
        ):
            raise ValueError(f"duplicate evaluator candidate ID {candidate.id!r}")
        # A workspace adapter need not know the controller checkpoint index.
        # Registration makes the candidate eligible at exactly the next
        # checkpoint and never retroactively at an earlier one.
        candidate.created_checkpoint = self.checkpoint_index + 1
        candidate.status = "pending"
        slot.archive.append(candidate)
        self.persist()

    async def judge_learned(
        self,
        *,
        slot_id: str,
        artifact: Any,
        expected: Any = None,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Judge an artifact without adding search evidence (e.g. train feedback)."""
        slot = self.slots[slot_id]
        return float(
            await slot.executor.judge(
                slot.incumbent,
                artifact=artifact,
                expected=expected,
                context=dict(context or {}),
            )
        )

    async def evaluate_learned(
        self,
        *,
        slot_id: str,
        node_id: str,
        role_id: str,
        task_id: str,
        artifact: Any,
        expected: Any = None,
        artifact_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> UtilityRecord:
        slot = self.slots[slot_id]
        if slot.role_id != role_id:
            raise ValueError("role does not belong to slot")
        raw = await self.judge_learned(
            slot_id=slot_id, artifact=artifact, expected=expected, context=context
        )
        record = UtilityRecord.create(
            node_id=node_id,
            role_id=role_id,
            task_id=task_id,
            criterion_kind="learned",
            raw_score=raw,
            outcome=slot.outcome_adapter(raw),
            artifact_id=artifact_id,
            slot_id=slot_id,
            epoch_id=slot.epoch,
        )
        self.ledger.append(record)
        self.validation_outcomes += 1
        if self.node_selection(node_id) is not None:
            self.pending_revaluation.discard(node_id)
        self.persist()
        return record

    def _task_records(self, node_id: str, cell: RoleTask) -> list[UtilityRecord]:
        records = self.ledger.valid_records(
            node_id=node_id, role_id=cell.role_id, task_id=cell.task_id
        )
        if cell.criterion_kind == "fixed":
            return [r for r in records if r.criterion_kind == "fixed"]
        epoch = self.slots[cell.slot_id or ""].epoch
        return [r for r in records if r.slot_id == cell.slot_id and r.epoch_id == epoch]

    def cell_is_current(self, node_id: str, cell: RoleTask) -> bool:
        """Return whether a role/task cell has any non-stale evidence.

        This is the selective-erasure boundary: fixed criteria and learned
        criteria owned by unchanged slots remain current after a transition.
        """
        return bool(self._task_records(node_id, cell))

    def least_measured_cell(
        self, node_id: str, available_task_ids: Iterable[str]
    ) -> RoleTask:
        """Paper-style role-then-task balancing for one validation outcome."""
        available = set(available_task_ids)
        cells = [cell for cell in self.role_tasks if cell.task_id in available]
        if not cells:
            raise ValueError("no configured RQGM cells are present in validation data")
        by_role: dict[str, list[RoleTask]] = {}
        for cell in cells:
            by_role.setdefault(cell.role_id, []).append(cell)
        counts = {cell: len(self._task_records(node_id, cell)) for cell in cells}
        role = min(
            by_role,
            key=lambda role_id: (
                sum(counts[cell] for cell in by_role[role_id]),
                role_id,
            ),
        )
        return min(by_role[role], key=lambda cell: (counts[cell], cell.task_id))

    def node_selection(self, node_id: str) -> NodeSelection | None:
        role_values: dict[str, list[float]] = {}
        pooled: list[UtilityRecord] = []

        def belief(records: Sequence[UtilityRecord]) -> float:
            successes = sum(record.outcome for record in records)
            return beta_lower_belief(successes, len(records) - successes, self.epsilon)

        for cell in self.role_tasks:
            records = self._task_records(node_id, cell)
            # A node cannot be a current-epoch endpoint while any active cell
            # has no evidence under its current criterion.
            if not records:
                return None
            pooled.extend(records)
            role_values.setdefault(cell.role_id, []).append(belief(records))
        if len(pooled) < self.minimum_node_outcomes:
            return None
        # The pooled score is the HGM-compatible working posterior used for
        # balanced search diagnostics.  The reported generalist follows the
        # paper's endpoint definition: uniform tasks within each role, then
        # uniform roles.  Specialists select on the corresponding role score.
        role_scores = {
            role: sum(values) / len(values) for role, values in role_values.items()
        }
        generalist = sum(role_scores.values()) / len(role_scores)
        return NodeSelection(node_id, generalist, belief(pooled), role_scores)

    def select_endpoints(
        self, node_ids: Iterable[str]
    ) -> tuple[NodeSelection | None, dict[str, NodeSelection]]:
        eligible = [
            s for node in node_ids if (s := self.node_selection(node)) is not None
        ]
        if not eligible:
            return None, {}
        generalist = max(eligible, key=lambda s: (s.score, s.node_id))
        specialists = {
            role: max(eligible, key=lambda s: (s.role_scores.get(role, -1), s.node_id))
            for role in {c.role_id for c in self.role_tasks}
        }
        return generalist, specialists

    async def _score_candidate(
        self, slot: EvaluatorSlot, candidate: EvaluatorCandidate
    ) -> None:
        successes = failures = invalid = 0
        for item in slot.verified_anchor_examples():
            for repeat in range(self.anchor_repeats):
                try:
                    raw = await slot.executor.judge(
                        candidate,
                        artifact=item.artifact,
                        expected=item.expected,
                        context={
                            "anchor": True,
                            "anchor_item_id": item.item_id,
                            "repeat": repeat,
                            "network_allowed": False,
                            "persistent_workspace": False,
                        },
                    )
                    if slot.outcome_adapter(raw):
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    invalid += 1
                    # Execution/parse failures are failed anchor judgments,
                    # never missing-at-random evidence a candidate can exploit.
                    failures += 1
        candidate.anchor_successes, candidate.anchor_failures = successes, failures
        candidate.anchor_invalid = invalid
        if successes + failures < self.minimum_anchor_outcomes:
            candidate.status, candidate.best_belief = "invalid", None
        else:
            candidate.status = "valid"
            candidate.best_belief = beta_lower_belief(successes, failures, self.epsilon)

    async def _plan_slot(self, slot: EvaluatorSlot) -> dict[str, Any]:
        incumbent, error = slot.incumbent, None
        challengers = [
            candidate
            for candidate in slot.archive
            if candidate.id != incumbent.id
            and candidate.created_checkpoint > slot.last_checkpoint
        ]
        if slot.proposer:
            try:
                proposed = list(
                    await slot.proposer.propose(
                        incumbent=incumbent,
                        slot_id=slot.slot_id,
                        checkpoint_index=self.checkpoint_index + 1,
                        public_context={
                            "checkpoint_index": self.checkpoint_index + 1,
                            "outcome_budget": self.validation_outcomes,
                            "epoch": slot.epoch,
                            "incumbent": incumbent.proposer_dict(),
                        },
                    )
                )
                for candidate in proposed:
                    if all(existing.id != candidate.id for existing in challengers):
                        challengers.append(candidate)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        scorable = []
        for candidate in challengers:
            identity_collision = any(
                existing.id == candidate.id and existing is not candidate
                for existing_slot in self.slots.values()
                for existing in existing_slot.archive
            )
            if identity_collision or (candidate.slot_id, candidate.role_id) != (
                slot.slot_id,
                slot.role_id,
            ):
                candidate.status = "invalid"
            else:
                scorable.append(candidate)
                if all(old.id != candidate.id for old in slot.archive):
                    slot.archive.append(candidate)
        await asyncio.gather(
            self._score_candidate(slot, incumbent),
            *(self._score_candidate(slot, c) for c in scorable),
        )
        winner = incumbent
        for candidate in scorable:
            if candidate.status != "valid" or candidate.best_belief is None:
                continue
            if winner.status != "valid" or winner.best_belief is None:
                winner = candidate
            elif candidate.best_belief > winner.best_belief:
                # Strict comparison is intentional: ties preserve the frozen
                # incumbent and avoid evaluator churn.
                winner = candidate
        return {
            "slot": slot,
            "incumbent": incumbent,
            "challengers": challengers,
            "winner": winner,
            "proposal_error": error,
        }

    async def checkpoint(self) -> dict[str, Any]:
        plans = await asyncio.gather(*(self._plan_slot(s) for s in self.slots.values()))
        index, summaries = self.checkpoint_index + 1, {}
        for plan in plans:
            slot, old, winner = plan["slot"], plan["incumbent"], plan["winner"]
            replaced, invalidated = winner.id != old.id, 0
            if replaced:
                invalidated = self.ledger.invalidate_slot(slot.slot_id, index)
                old.status, winner.status, slot.incumbent = "valid", "incumbent", winner
                slot.epoch += 1
                self.pending_revaluation.update(
                    r.node_id
                    for r in self.ledger.records
                    if r.slot_id == slot.slot_id and not r.valid
                )
            else:
                # Anchor scoring temporarily marks every scorable candidate
                # valid; the frozen incumbent remains the incumbent.
                old.status = "incumbent"
            slot.last_checkpoint = index
            summaries[slot.slot_id] = {
                "incumbent_before": old.id,
                "challengers": [c.id for c in plan["challengers"]],
                "promoted": winner.id,
                "replacement": replaced,
                "selective_erasure": invalidated,
                "proposal_error": plan["proposal_error"],
                "anchor_summary": {
                    "successes": winner.anchor_successes,
                    "failures": winner.anchor_failures,
                    "invalid": winner.anchor_invalid,
                    "best_belief": winner.best_belief,
                },
            }
        self.checkpoint_index = index
        event = {
            "kind": "checkpoint",
            "checkpoint_index": index,
            "outcome_budget": self.validation_outcomes,
            "epoch_vector": self.epoch_vector,
            "slots": summaries,
        }
        self.events.append(event)
        self.persist()
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "epsilon": self.epsilon,
            "minimum_anchor_outcomes": self.minimum_anchor_outcomes,
            "minimum_node_outcomes": self.minimum_node_outcomes,
            "anchor_repeats": self.anchor_repeats,
            "checkpoint_schedule": asdict(self.checkpoint_schedule),
            "validation_outcomes": self.validation_outcomes,
            "checkpoint_index": self.checkpoint_index,
            "epoch_vector": self.epoch_vector,
            "role_tasks": [asdict(c) for c in self.role_tasks],
            "slots": {
                sid: {
                    "role_id": s.role_id,
                    "epoch": s.epoch,
                    "last_checkpoint": s.last_checkpoint,
                    "incumbent_id": s.incumbent.id,
                    "archive": [c.public_dict() for c in s.archive],
                    "anchor": s.anchor_metadata(),
                }
                for sid, s in self.slots.items()
            },
            "ledger": [asdict(r) for r in self.ledger.records],
            "events": self.events,
            "pending_revaluation": sorted(self.pending_revaluation),
        }

    def persist(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.state_path)

    @classmethod
    def restore(
        cls, state_path: str | Path, *, runtime_slots: Sequence[EvaluatorSlot]
    ) -> "RQGMController":
        path, runtime = Path(state_path), {s.slot_id: s for s in runtime_slots}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("state_version") != cls.STATE_VERSION or set(runtime) != set(
            payload["slots"]
        ):
            raise ValueError("saved RQGM state is incompatible with runtime")
        slots = []
        for sid, saved in payload["slots"].items():
            base = runtime[sid]
            if base.anchor_metadata() != saved["anchor"]:
                raise ValueError(f"anchor metadata mismatch for slot {sid}")
            archive = [EvaluatorCandidate(**c) for c in saved["archive"]]
            by_id = {c.id: c for c in archive}
            slots.append(
                EvaluatorSlot(
                    sid,
                    saved["role_id"],
                    by_id[saved["incumbent_id"]],
                    base.anchor_provider,
                    base.executor,
                    base.proposer,
                    base.outcome_adapter,
                    int(saved["epoch"]),
                    archive,
                    int(saved["last_checkpoint"]),
                )
            )
        schedule = payload["checkpoint_schedule"]
        controller = cls(
            slots=slots,
            role_tasks=[RoleTask(**c) for c in payload["role_tasks"]],
            epsilon=payload["epsilon"],
            minimum_anchor_outcomes=payload["minimum_anchor_outcomes"],
            minimum_node_outcomes=payload["minimum_node_outcomes"],
            checkpoint_schedule=CheckpointSchedule(
                tuple(schedule["explicit"]), schedule["start"], schedule["ratio"]
            ),
            anchor_repeats=payload["anchor_repeats"],
            ledger=UtilityLedger(UtilityRecord(**r) for r in payload["ledger"]),
        )
        controller.validation_outcomes, controller.checkpoint_index = (
            payload["validation_outcomes"],
            payload["checkpoint_index"],
        )
        controller.events, controller.pending_revaluation = (
            payload["events"],
            set(payload["pending_revaluation"]),
        )
        controller.state_path = path
        controller.persist()
        return controller
