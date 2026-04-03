"""dspy.GEPA skill optimizer loop — naive reflective baseline for EvoSkill comparison."""

import asyncio
import http.client
from pathlib import Path
from typing import Callable

import dspy
import logging as _logging
import json
import os
import re

# Suppress dspy's verbose prompt/instruction printing during GEPA compilation
_logging.getLogger("dspy").setLevel(_logging.WARNING)

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


def _resolve_env_value(value: str) -> str:
    """Resolve '{env:VAR_NAME}' patterns to actual environment variable values."""
    match = re.fullmatch(r"\{env:(\w+)\}", value)
    if match:
        var_name = match.group(1)
        result = os.environ.get(var_name)
        if result is None:
            raise EnvironmentError(
                f"Environment variable '{var_name}' is not set "
                f"(required by opencode.json)"
            )
        return result
    return value


def _load_provider_config(provider: str, model: str) -> dict:
    """Load provider config from opencode.json.

    Returns a dict with keys: api_key (str | None), api_base (str | None),
    model_prefix (str), max_tokens (int | None).
    """
    config_path = Path(get_project_root()) / "opencode.json"
    if not config_path.exists():
        raise FileNotFoundError(f"opencode.json not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    providers = config.get("provider", {})

    # If VLLM_BASE_URL is set and provider is 'vllm', synthesize config on the fly
    if provider == "vllm" and provider not in providers:
        vllm_base_url = os.environ.get("VLLM_BASE_URL")
        if vllm_base_url:
            vllm_ctx = int(os.environ.get("VLLM_MAX_MODEL_LEN", "65536"))
            providers["vllm"] = {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": vllm_base_url, "apiKey": "EMPTY"},
                "models": {model: {"limit": {"context": vllm_ctx, "output": vllm_ctx // 2}}},
            }

    if provider not in providers:
        raise ValueError(
            f"Provider '{provider}' not found in opencode.json. "
            f"Available providers: {list(providers.keys())}"
        )

    prov_cfg = providers[provider]
    options = prov_cfg.get("options", {})
    npm = prov_cfg.get("npm", "")
    models = prov_cfg.get("models", {})

    raw_key = options.get("apiKey", "")
    api_key = _resolve_env_value(raw_key) if raw_key else None

    api_base = options.get("baseURL")

    if api_base:
        model_prefix = "openai"
    elif npm == "@ai-sdk/google":
        model_prefix = "gemini"
    else:
        model_prefix = provider

    max_tokens = None
    if model in models:
        max_tokens = models[model].get("limit", {}).get("output")

    return {
        "api_key": api_key,
        "api_base": api_base,
        "model_prefix": model_prefix,
        "max_tokens": max_tokens,
    }


def _web_search(query: str) -> str:
    """Search Google via Serper and return top organic results."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set."
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    conn.close()
    results = []
    for item in data.get("organic", [])[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        results.append(f"{title}\n{snippet}\n{link}")
    return "\n\n".join(results) if results else "No results found."


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
                predicted,
                ground_truth,
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


class CodeQASignature(dspy.Signature):
    """Solve the given coding problem following the instructions exactly."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc=(
            "Your complete Python solution enclosed in a ```python ... ``` code block. "
            "No explanation outside the code block."
        )
    )


class QAModule(dspy.Module):
    """dspy module whose instructions GEPA will evolve into skill content.

    Initialized with the static base system prompt + an empty skills placeholder.
    GEPA's reflection LM will iteratively improve the instructions section,
    discovering and adding skills to address observed failures.
    """

    def __init__(
        self,
        initial_instructions: str,
        code_task: bool = False,
        tools: list[Callable] | None = None,
    ):
        super().__init__()
        sig_class = CodeQASignature if code_task else QASignature
        sig = sig_class.with_instructions(initial_instructions)
        if tools:
            self.predict = dspy.ReAct(sig, tools=tools)
        elif code_task:
            # Code tasks: use Predict (single answer field) — ChainOfThought's separate
            # reasoning field causes models to embed the solution in reasoning and omit
            # the answer field, causing AdapterParseError on every example.
            self.predict = dspy.Predict(sig)
        else:
            self.predict = dspy.ChainOfThought(sig)

    def forward(self, question: str):
        return self.predict(question=question)


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
        session_dir: Path | None = None,
        code_task: bool = False,
        question_preprocessor: Callable[[str], str] | None = None,
        web_search: bool = False,
    ):
        self.config = config
        self.train_pools = train_pools
        self.val_data = val_data
        self.scorer = scorer or _score_multi_tolerance
        self.student_model = student_model
        self.reflection_model = reflection_model or student_model
        self.provider = provider
        self.prompt_path = prompt_path
        self.session_dir = session_dir
        self.code_task = code_task
        self.question_preprocessor = question_preprocessor
        self.web_search = web_search

        # Set after compile() for use by export_best_skills
        self._optimized_module: QAModule | None = None
        self._initial_instructions: str = ""

    def _build_initial_instructions(self) -> str:
        """Load static base prompt + empty skills placeholder."""
        base_prompt = self.prompt_path.read_text().strip()
        return base_prompt + _INITIAL_SKILLS_PLACEHOLDER

    def _preprocess(self, question: str) -> str:
        """Apply question_preprocessor if set, otherwise return as-is."""
        return self.question_preprocessor(question) if self.question_preprocessor else question

    def _to_dspy_examples(self) -> list[dspy.Example]:
        """Flatten train_pools into dspy.Example list."""
        examples = []
        for category, pool in self.train_pools.items():
            for question, answer in pool:
                ex = dspy.Example(
                    question=self._preprocess(question), answer=answer, category=category
                ).with_inputs("question")
                examples.append(ex)
        return examples

    def _to_dspy_val(self) -> list[dspy.Example]:
        """Convert val_data tuples to dspy.Example list."""
        return [
            dspy.Example(question=self._preprocess(q), answer=a, category=c).with_inputs("question")
            for q, a, c in self.val_data
        ]

    def _evaluate_module(self, module: QAModule) -> float:
        """Evaluate optimized dspy module on val_data using the scorer."""
        correct = 0.0
        total = 0
        for question, answer, _ in self.val_data:
            try:
                pred = module(question=self._preprocess(question))
                predicted = getattr(pred, "answer", str(pred))
                correct += self.scorer(question, predicted, answer)
            except Exception as e:
                _log("", f"  [EVAL ERROR] {question[:40]}... ({type(e).__name__}: {e})")
            total += 1
        return correct / total if total > 0 else 0.0


    async def run(self) -> LoopResult:
        """Run dspy.GEPA skill optimization and return LoopResult."""
        _log(
            "GEPA INIT",
            f"student={self.student_model}, reflection={self.reflection_model}, provider={self.provider}",
        )

        # 1. Static base prompt + skills placeholder → initial instructions
        self._initial_instructions = self._build_initial_instructions()

        # 2. Configure dspy LMs using same model/provider settings as base agent
        # cache=False: dspy's disk cache uses diskcache (SQLite), which corrupts
        # under concurrent thread access from GEPA's parallel evaluation workers.
        student_cfg = _load_provider_config(self.provider, self.student_model)
        reflection_cfg = _load_provider_config(self.provider, self.reflection_model)

        lm_kwargs = {"cache": False}
        if student_cfg["api_key"]:
            lm_kwargs["api_key"] = student_cfg["api_key"]
        if student_cfg["api_base"]:
            lm_kwargs["api_base"] = student_cfg["api_base"]

        prefix = student_cfg["model_prefix"]
        student_lm = dspy.LM(
            f"{prefix}/{self.student_model}",
            temperature=0.7,
            max_tokens=student_cfg["max_tokens"],
            **lm_kwargs,
        )
        reflection_lm = dspy.LM(
            f"{prefix}/{self.reflection_model}",
            temperature=1.0,
            max_tokens=reflection_cfg["max_tokens"],
            **lm_kwargs,
        )
            
        dspy.configure(lm=student_lm)

        # 3. Prepare data
        trainset = self._to_dspy_examples()
        valset = self._to_dspy_val()
        _log("GEPA DATA", f"trainset={len(trainset)}, valset={len(valset)}")

        # 4. Create module (static prompt + empty skills) and metric
        tools = [_web_search] if self.web_search else None
        module = QAModule(self._initial_instructions, code_task=self.code_task, tools=tools)
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

        # 7. Store optimized module and immediately save prompt to disk
        self._optimized_module = optimized
        inner_predict = object.__getattribute__(optimized, '__dict__')['predict']
        prompt_text = inner_predict.signature.instructions
        out_dir = self.session_dir if self.session_dir is not None else self.prompt_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = out_dir / "gepa_prompt.txt"
        prompt_path.write_text(prompt_text)
        _log("GEPA SAVE", f"Optimized prompt saved to {prompt_path}")

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
        """Save the full GEPA-optimized prompt to gepa_prompt.txt next to prompt.txt.

        Args:
            target_branch: Unused (kept for interface compatibility with SelfImprovingLoop).
            run_dir: Unused (kept for interface compatibility with SelfImprovingLoop).

        Returns:
            List containing "gepa_prompt" if successful, else [].
        """
        if self._optimized_module is None:
            _log("EXPORT", "No optimized module — run() must be called first")
            return []

        predict_obj = self._optimized_module.predict
        # ReAct: GEPA optimizes .react.signature (the inner Predict driving the tool loop)
        # ChainOfThought: wraps an inner Predict at .predict
        # Predict: exposes .signature directly
        if isinstance(predict_obj, dspy.ReAct):
            prompt_text = predict_obj.react.signature.instructions
        else:
            inner = getattr(predict_obj, "predict", predict_obj)
            prompt_text = inner.signature.instructions
        out_dir = self.session_dir if self.session_dir is not None else self.prompt_path.parent
        out_path = out_dir / "gepa_prompt.txt"
        out_path.write_text(prompt_text)
        _log("EXPORT", f"Saved best prompt to {out_path}")
        return ["gepa_prompt"]