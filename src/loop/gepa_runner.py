"""dspy.GEPA skill optimizer loop — naive reflective baseline for EvoSkill comparison."""

import asyncio
from pathlib import Path
from typing import Callable

import dspy
import os

from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramManager

from .config import LoopConfig
from .runner import LoopAgents, LoopResult, _log, _score_multi_tolerance


# Static base prompt path (same as SelfImprovingLoop's _prompt_path)
_PROMPT_PATH = (
    Path(get_project_root())
    / "src" / "agent_profiles" / "base_agent" / "prompt.txt"
)
_INITIAL_SKILLS_PLACEHOLDER = (
    "\n\n## Skills\n"
    "(No skills yet. Skills will be discovered through prompt optimization.)"
)

#_SKILL_USAGE = ("You should use the skills available to you, seen under '## Skills', to solve the problems")

class GEPAFeedbackMetric:
    """Wraps the deep_agents scorer for dspy.GEPA.

    Returns {"score": float, "feedback": str} as required by GEPA's metric API.
    """

    def __init__(self, scorer: Callable[[str, str, str], float]):
        self._scorer = scorer

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        question = gold.question
        ground_truth = gold.answer
        predicted = getattr(pred, "answer", str(pred))

        try:
            score = self._scorer(
                question,
                predicted.strip().lower(),
                ground_truth.strip().lower(),
            )
        except Exception:
            score = 0.0

        # dspy.Evaluate (used internally by GEPA for valset scoring) calls this
        # metric with no extra args and sums the results — it needs a float.
        # Only return the dict form when GEPA is requesting feedback for reflection.
        if pred_trace is not None or pred_name is not None:
            if score >= 0.8:
                feedback = f"Correct. Predicted: {predicted}"
            else:
                feedback = (
                    f"Incorrect. Expected: {ground_truth}. "
                    f"Got: {predicted}. Score: {score:.2f}."
                )
            return {"score": score, "feedback": feedback}

        return score


class QASignature(dspy.Signature):
    """Answer the given question following the instructions exactly."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc=(
            "Your final answer only — no reasoning here. "
            "Follow any formatting instructions given in the question exactly "
            "(e.g. code enclosed in backticks, specific output format, etc.)."
        )
    )


class QAModule(dspy.Module):
    """dspy module whose instructions GEPA will evolve into skill content.

    Initialized with the static base system prompt + an empty skills placeholder.
    GEPA's reflection LM will iteratively improve the instructions section,
    discovering and adding skills to address observed failures.
    """

    def __init__(self, initial_instructions: str):
        super().__init__()
        sig = QASignature.with_instructions(initial_instructions)
        self.predict = dspy.ChainOfThought(sig)

    def forward(self, question: str):
        try:
            return self.predict(question=question)
        except Exception:
            # Parse failures (AdapterParseError, etc.) score as 0 rather than
            # crashing the whole evaluation batch.
            return dspy.Prediction(answer="", reasoning="")


class GEPALoop:
    """dspy.GEPA-based naive skill optimizer (baseline for EvoSkill comparison).

    - Base system prompt is STATIC (loaded once from prompt.txt, never modified).
    - GEPA evolves the instructions field, which serves as the skills container.
    - After compile, optimized instructions are extracted and exported as SKILL.md.
    - Returns LoopResult with same interface as SelfImprovingLoop.
    """

    def __init__(
        self,
        config: LoopConfig,
        agents: LoopAgents,        # accepted for interface compatibility; not used
        manager: ProgramManager,   # accepted for interface compatibility; not used
        train_pools: dict[str, list[tuple[str, str]]],
        val_data: list[tuple[str, str, str]],
        scorer: Callable[[str, str, str], float] | None = None,
        student_model: str = "claude-opus-4-5-20251101",
        reflection_model: str | None = None,
        provider: str = "anthropic",
        prompt_path: Path = _PROMPT_PATH,
    ):
        self.config = config
        self.train_pools = train_pools
        self.val_data = val_data
        self.scorer = scorer or _score_multi_tolerance
        self.student_model = student_model
        self.reflection_model = reflection_model or student_model
        self.provider = provider
        self.prompt_path = prompt_path

        # Set after compile() for use by export_best_skills
        self._optimized_module: QAModule | None = None
        self._initial_instructions: str = ""

    def _build_initial_instructions(self) -> str:
        """Load static base prompt + empty skills placeholder."""
        base_prompt = self.prompt_path.read_text().strip()
        return base_prompt + _INITIAL_SKILLS_PLACEHOLDER

    def _to_dspy_examples(self) -> list[dspy.Example]:
        """Flatten train_pools into dspy.Example list."""
        examples = []
        for category, pool in self.train_pools.items():
            for question, answer in pool:
                ex = dspy.Example(
                    question=question, answer=answer, category=category
                ).with_inputs("question")
                examples.append(ex)
        return examples

    def _to_dspy_val(self) -> list[dspy.Example]:
        """Convert val_data tuples to dspy.Example list."""
        return [
            dspy.Example(question=q, answer=a, category=c).with_inputs("question")
            for q, a, c in self.val_data
        ]

    def _evaluate_module(self, module: QAModule) -> float:
        """Evaluate optimized dspy module on val_data using the scorer."""
        correct = 0.0
        total = 0
        for question, answer, _ in self.val_data:
            try:
                pred = module(question=question)
                predicted = getattr(pred, "answer", str(pred))
                correct += self.scorer(
                    question,
                    predicted.strip().lower(),
                    answer.strip().lower(),
                )
            except Exception as e:
                _log("", f"  [EVAL ERROR] {question[:40]}... ({type(e).__name__}: {e})")
            total += 1
        return correct / total if total > 0 else 0.0

    def _extract_skill_content(self, optimized: QAModule) -> str:
        """Extract the skill additions from the optimized module's instructions.

        Returns the portion of instructions after the static base prompt —
        the skill content GEPA discovered. Falls back to full instructions
        if the diff cannot be computed.
        """
        optimized_instructions = optimized.predict.signature.instructions
        base_prefix = self._initial_instructions.split("\n\n## Skills")[0]
        if optimized_instructions.startswith(base_prefix):
            skill_start = optimized_instructions.find("\n\n## Skills")
            if skill_start != -1:
                return optimized_instructions[skill_start + 2:]  # strip leading \n\n
        # Fallback: return full optimized instructions as skill content
        return optimized_instructions

    async def run(self) -> LoopResult:
        """Run dspy.GEPA skill optimization and return LoopResult."""
        _log(
            "GEPA INIT",
            f"student={self.student_model}, reflection={self.reflection_model}, provider={self.provider}",
        )

        # 1. Static base prompt + skills placeholder → initial instructions
        self._initial_instructions = self._build_initial_instructions()

        # 2. Configure dspy LMs using same model/provider settings as base agent
        if "arc" in self.provider:
            arc_key = os.environ.get("ARC_LLM_API_KEY")
            student_lm = dspy.LM(
                f"openai/{self.student_model}", 
                temperature=0.7, 
                api_key=arc_key, 
                api_base="https://llm-api.arc.vt.edu/api/v1",
            )
            reflection_lm = dspy.LM(
                f"openai/{self.reflection_model}",
                temperature=1.0,
                max_tokens=32000,
                api_key=arc_key, 
                api_base="https://llm-api.arc.vt.edu/api/v1",
            )
        else:
            student_lm = dspy.LM(
                f"{self.provider}/{self.student_model}", temperature=0.7
            )
            reflection_lm = dspy.LM(
                f"{self.provider}/{self.reflection_model}",
                temperature=1.0,
                max_tokens=32000,
            )
            
        dspy.configure(lm=student_lm)

        # 3. Prepare data
        trainset = self._to_dspy_examples()
        valset = self._to_dspy_val()
        _log("GEPA DATA", f"trainset={len(trainset)}, valset={len(valset)}")

        # 4. Create module (static prompt + empty skills) and metric
        module = QAModule(self._initial_instructions)
        metric = GEPAFeedbackMetric(self.scorer)

        # 5. Instantiate dspy.GEPA — budget maps to max_iterations full eval passes
        optimizer = dspy.GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            max_full_evals=self.config.max_iterations,
            num_threads=self.config.concurrency,
        )

        # 6. Compile (dspy.GEPA is synchronous — run in executor to avoid blocking)
        _log("GEPA RUN", f"max_full_evals={self.config.max_iterations}")
        event_loop = asyncio.get_event_loop()
        optimized = await event_loop.run_in_executor(
            None,
            lambda: optimizer.compile(
                student=module,
                trainset=trainset,
                valset=valset,
            ),
        )

        # 7. Store optimized module for export_best_skills()
        self._optimized_module = optimized

        # 8. Evaluate optimized module on val_data
        _log("GEPA EVAL", f"Evaluating on {len(self.val_data)} val samples...")
        best_score = self._evaluate_module(optimized)
        _log("GEPA DONE", f"Best score: {best_score:.4f}")

        return LoopResult(
            frontier=[("gepa-optimized", best_score)],
            best_program="gepa-optimized",
            best_score=best_score,
            iterations_completed=self.config.max_iterations,
        )

    def export_best_skills(
        self,
        target_branch: str | None = None,
        run_dir: str | Path | None = None,
    ) -> list[str]:
        """Export GEPA-optimized skill instructions to run_dir as SKILL.md.

        Mirrors export_best_skills() in runner.py — writes to
        run_dir/.claude/skills/gepa-learned-skill/SKILL.md.

        Args:
            target_branch: Unused (kept for interface compatibility with SelfImprovingLoop).
            run_dir: Path to .evoskill-runs/<session>/ dir.

        Returns:
            List of skill names exported.
        """
        if self._optimized_module is None:
            _log("EXPORT", "No optimized module — run() must be called first")
            return []

        skill_content = self._extract_skill_content(self._optimized_module)
        skill_name = "gepa-learned-skill"

        if run_dir:
            dest = Path(run_dir) / ".claude" / "skills" / skill_name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "SKILL.md").write_text(skill_content)
            _log("EXPORT", f"Exported GEPA skill to {run_dir}: [{skill_name}]")
        else:
            _log("EXPORT", "No run_dir provided — skill not exported")
            return []

        return [skill_name]