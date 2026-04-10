"""Shared CLI helpers used by run/eval without importing the full run command."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.cli.config import ProjectConfig


def load_and_split(cfg: ProjectConfig):
    from src.api.data_utils import stratified_split

    data = pd.read_csv(cfg.dataset_path)

    renames: dict[str, str] = {}
    if cfg.dataset.question_column != "question":
        renames[cfg.dataset.question_column] = "question"
    if cfg.dataset.ground_truth_column != "ground_truth":
        renames[cfg.dataset.ground_truth_column] = "ground_truth"
    if renames:
        data.rename(columns=renames, inplace=True)

    if cfg.dataset.category_column and cfg.dataset.category_column in data.columns:
        if cfg.dataset.category_column != "category":
            data.rename(columns={cfg.dataset.category_column: "category"}, inplace=True)
    elif "category" not in data.columns:
        data["category"] = "default"

    return stratified_split(
        data,
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
    )


def infer_provider(model: str) -> str:
    """Infer the LLM provider from the configured model name."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    return "anthropic"


async def call_llm(provider: str, model: str, prompt: str) -> str:
    """Call the requested LLM provider and return the raw text response."""
    if provider == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    if provider == "openai":
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("openai package not installed. Run: uv add openai") from exc

        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    if provider == "google":
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai package not installed. Run: uv add google-genai") from exc

        client = genai.Client()
        response = await client.aio.models.generate_content(model=model, contents=prompt)
        return response.text

    raise ValueError(f"Unknown provider: {provider}")


def make_scorer(cfg: ProjectConfig):
    from src.loop.runner import _score_multi_tolerance

    if cfg.scorer.type == "exact":
        def exact(question: str, predicted: str, ground_truth: str) -> float:
            return 1.0 if str(predicted).strip().lower() == str(ground_truth).strip().lower() else 0.0

        return exact

    if cfg.scorer.type == "multi_tolerance":
        return _score_multi_tolerance

    if cfg.scorer.type == "llm":
        rubric = cfg.scorer.rubric or "Award 1.0 if correct, 0.0 if wrong."
        model = cfg.scorer.model or "claude-sonnet-4-6"
        provider = cfg.scorer.provider or infer_provider(model)

        async def llm_score(question: str, predicted: str, ground_truth: str) -> float:
            prompt = (
                f"Question: {question}\n"
                f"Expected: {ground_truth}\n"
                f"Got: {predicted}\n\n"
                f"Rubric: {rubric}\n\n"
                "Reply with only a number between 0.0 and 1.0."
            )
            try:
                text = await call_llm(provider, model, prompt)
                return float(text.strip())
            except (ValueError, Exception):
                return 0.0

        def llm_scorer(question: str, predicted: str, ground_truth: str) -> float:
            return asyncio.get_event_loop().run_until_complete(
                llm_score(question, predicted, ground_truth)
            )

        return llm_scorer

    if cfg.scorer.type == "script":
        import shlex
        import subprocess

        def script_scorer(question: str, predicted: str, ground_truth: str) -> float:
            cmd = cfg.scorer.command.format(predicted=predicted, expected=ground_truth)
            result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
            try:
                return float(result.stdout.strip())
            except ValueError:
                return 0.0

        return script_scorer

    return _score_multi_tolerance
