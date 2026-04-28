![Sentient Logo](assets/sentient-logo-new-M.png)

# EvoSkill: Automated Skill discovery For Multi-Agent Systems

**Automatically discover high-performance agent skills for any task!**

[![Homepage](https://img.shields.io/badge/Sentient-Homepage-%23EAEAEA?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHNyZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNDEuMzMzIiBoZWlnaHQ9IjM0MS4zMzMiIHZpZXdCb3g9IjAgMCAyNTYgMjU2Ij48cGF0aCBkPSJNMTMyLjUgMjguNC0yOC40YzAtLjMuMi0uNS41LS41aDI3LjJhLjUuNSAwIDAgMSAuNS41djI3LjJjMCAuMy0uMi41LS41LjVIMTMyLjVhLjUuNSAwIDAgMS0uNS0uNXoiLz48L3N2Zz4%3D&link=https%3A%2F%2Fsentient.xyz%2F)](https://sentient.xyz/)
[![GitHub](https://img.shields.io/badge/Github-sentient_agi-181717?logo=github)](https://github.com/sentient-agi)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SentientAGI-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/Sentientagi)
[![Discord](https://img.shields.io/badge/Discord-SentientAGI-7289da?logo=discord&logoColor=white&color=7289da)](https://discord.gg/sentientfoundation)
[![Twitter Follow](https://img.shields.io/badge/-SentientAGI-grey?logo=x&link=https%3A%2F%2Fx.com%2FSentientAGI%2F)](https://x.com/SentientAGI)

[Paper (Preprint)](https://www.sentient.xyz/blog/recursive-open-meta-agent) •
[Build Agents for $$$](https://www.sentient.xyz/)

## 📑 Table of Contents

- [🧠 What is EvoSkill?](#-what-is-evoskill)
- [🏗️ How It Works](#️-how-it-works)
- [📦 Installation & Setup](#-installation--setup)
- [🆓 Running on a Budget (Free & Open Models)](#-running-on-a-budget-free--open-models)
- [🐍 Python API](#-python-api)
- [⚡ Quickstart: Running the Self-Improvement Loop](#-quickstart-running-the-self-improvement-loop)
- [📊 Running Evaluations](#-running-evaluations)
- [🔑 Key Concepts](#-key-concepts)
- [🧩 Extending EvoSkill: Adding a New Task](#-extending-evoskill-adding-a-new-task)
- [📚 Citation](#-citation)
- [📄 License](#-license)

---

## 🧠 What is EvoSkill?

EvoSkill is a self-improving agent framework that **automatically discovers high-performance skills** for AI agents. Rather than relying on manual prompt engineering, EvoSkill runs an evolutionary loop that tests an agent on benchmark questions, identifies failure patterns, proposes improvements (new skills or prompt mutations), evaluates the changes, and keeps the best-performing variants.

The core insight is simple: treat agent configurations as programs that can be iterated on automatically. Each "program" is a versioned combination of a system prompt and a set of skills. EvoSkill maintains a **frontier** of the top-N performing programs, uses failures to drive targeted improvements, and tracks everything through git branches for full reproducibility.

EvoSkill has been validated on multiple benchmarks including DABStep (data analysis), SEAL-QA (search-augmented QA), and OfficeQA, demonstrating that automated skill discovery can match or exceed hand-tuned agent configurations.

---

## 🏗️ How It Works

![EvoSkill Architecture](assets/evoskill.jpg)

The self-improvement loop follows five stages:

1. **Base Agent** — Attempts benchmark questions using the current best program (system prompt + skills).
2. **Proposer** — Analyzes failure cases and proposes targeted skill or prompt changes to address them.
3. **Generator** — Creates the proposed changes: writes new skill files or rewrites the system prompt.
4. **Evaluator** — Scores the new program variant on a held-out validation set to measure improvement.
5. **Frontier** — Tracks the top-N performing programs as git branches; the best survive to the next iteration.

This cycle repeats for a configurable number of iterations, automatically converging on stronger agent configurations.

---

## 📦 Installation & Setup

**Requirements:**

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- Docker (for LiveCodeBench evaluation with secure code execution sandbox)

**Install dependencies:**

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

**Environment variables:**
Either login to **Claude Code** or use your API key

```bash
# Required — used by the Claude agent SDK
export ANTHROPIC_API_KEY=your-key-here
```

**SDK and Model Selection:**

Use `--sdk` and `--model` to configure which SDK and model to use:

```bash
# Claude SDK (default)
uv run python scripts/run_eval.py --sdk claude --model claude-sonnet-4-5-20250514

# OpenCode SDK with different models
uv run python scripts/run_eval.py --sdk opencode --model deepseek-ai/DeepSeek-V3
uv run python scripts/run_eval.py --sdk opencode --model google/gemini-2.0-flash-exp
```

**Dataset preparation:**

Place your benchmark datasets in the `.dataset/` directory:

- DABStep: `.dataset/dabstep_data.csv`
- SEAL-QA: `.dataset/seal-0.csv`
- OfficeQA: see `scripts/run_eval.py` for expected path

---

## 🆓 Running on a Budget (Free & Open Models)

You do **not** need a paid Claude Code subscription to use EvoSkill.
The framework natively supports multiple SDKs and model providers through the `--sdk` flag.

### ✅ Validate your environment first

Before running the loop, use the included environment validator to catch missing API keys, wrong Python versions, or misconfigured dependencies in one command:

```bash
python scripts/validate_env.py
```

Check a specific SDK and model combination:

```bash
python scripts/validate_env.py --sdk opencode --model google/gemini-2.0-flash-exp
```

List all free and low-cost model options:

```bash
python scripts/validate_env.py --list-free-options
```

---

### Option 1 — Gemini Flash via OpenRouter (Free, no billing required)

[OpenRouter](https://openrouter.ai) offers `google/gemini-2.0-flash-exp` at zero cost. Sign up, get your key, and run:

```bash
export OPENROUTER_API_KEY=your-key-here
uv run python scripts/run_eval.py --sdk opencode --model google/gemini-2.0-flash-exp
```

### Option 2 — DeepSeek-V3 (Very low cost)

DeepSeek-V3 via OpenRouter is one of the most capable open models at a fraction of the cost of proprietary APIs:

```bash
export OPENROUTER_API_KEY=your-key-here
uv run python scripts/run_eval.py --sdk opencode --model deepseek-ai/DeepSeek-V3
```

### Option 3 — Anthropic Free Tier (Claude Haiku)

Sign up for a free API key at [console.anthropic.com](https://console.anthropic.com). Claude Haiku is the lowest-cost Anthropic model and works well for exploration runs:

```bash
export ANTHROPIC_API_KEY=your-key-here
uv run python scripts/run_eval.py --sdk claude --model claude-haiku-4-5-20251001
```

### 💡 Tip: Limit samples to explore for free

Use `--num-samples` to cap how many benchmark questions are evaluated. This keeps API costs near zero while you validate your setup:

```bash
uv run python scripts/run_eval.py --sdk opencode --model google/gemini-2.0-flash-exp --num-samples 20
```

| SDK | Model | Cost | Notes |
|---|---|---|---|
| `opencode` | `google/gemini-2.0-flash-exp` | **Free** | OpenRouter free tier |
| `opencode` | `deepseek-ai/DeepSeek-V3` | ~$0.001/1k tokens | Extremely low cost |
| `claude` | `claude-haiku-4-5-20251001` | Low | Cheapest Anthropic model |

---

## 🐍 Python API

EvoSkill provides a high-level Python API that reduces the boilerplate needed to run the self-improvement loop or standalone evaluations to just a few lines.

### `EvoSkill` — Run the self-improvement loop

```python
from src.api import EvoSkill

# Minimal — uses task defaults
result = await EvoSkill(dataset=".dataset/seal-0.csv", task="sealqa").run()

# Full configuration
evo = EvoSkill(
    task="sealqa",
    model="sonnet",
    mode="skill_only",
    max_iterations=20,
    frontier_size=3,
    concurrency=4,
    train_ratio=0.18,
    val_ratio=0.12,
    continue_mode=False,
)
result = await evo.run()

# Synchronous usage (wraps asyncio.run)
result = EvoSkill(task="base").run_sync()

# Preview dataset splits without running
print(evo.dataset_info)
```

### `EvalRunner` — Run standalone evaluation

```python
from src.api import EvalRunner

summary = await EvalRunner(
    task="sealqa",
    model="sonnet",
    max_concurrent=8,
).run()

print(f"Accuracy: {summary.accuracy:.1%} ({summary.correct}/{summary.successful})")
```

### Built-in tasks

Three tasks are registered out of the box:

| Task | Agent | Default Dataset | Scorer |
|---|---|---|---|
| `"base"` | Base agent | `.dataset/new_runs_base/solved_dataset.csv` | Multi-tolerance (default) |
| `"sealqa"` | SEAL-QA agent | `.dataset/seal-0.csv` | LLM-graded (GPT) |

```python
from src.api import list_tasks
print(list_tasks())  # ['base', 'dabstep', 'sealqa']
```

### Registering a custom task

```python
from src.api import TaskConfig, register_task

register_task(TaskConfig(
    name="my_task",
    make_agent_options=make_my_agent_options,
    scorer=my_scorer_fn,  # (question, predicted, ground_truth) -> float
    column_renames={"label": "ground_truth", "topic": "category"},
    default_dataset=".dataset/my_data.csv",
))

result = await EvoSkill(task="my_task").run()
```

---

## ⚡ Quickstart: Running the Self-Improvement Loop

The CLI scripts remain available for users who prefer the command line.

Run the evolutionary skill discovery loop on a benchmark:

**OfficeQA:**

```bash
python scripts/run_loop.py --mode skill_only --max-iterations 20
```

**SEAL-QA:**

```bash
python scripts/run_loop_sealqa.py --mode skill_only --max-iterations 20
```

**Key CLI flags:**

| Flag | Description | Default |
|---|---|---|
| `--mode` | Evolution mode: `skill_only` or `prompt_only` | `skill_only` |
| `--max-iterations` | Number of improvement iterations | `20` |
| `--frontier-size` | Number of top programs to keep | `3` |
| `--concurrency` | Concurrent evaluations | `4` |
| `--continue` | Resume from existing frontier | off |
| `--no-cache` | Disable run caching | off |
| `--model` | Base agent model (`opus`, `sonnet`, `haiku`) | `opus` |

---

## 📊 Running Evaluations

Evaluate an agent configuration on a full benchmark dataset:

**OfficeQA:**

```bash
python scripts/run_eval.py --model opus --max-concurrent 8
```

**SEAL-QA:**

```bash
python scripts/run_eval_sealqa.py --model opus --max-concurrent 8
```

Common eval flags: `--output <path>`, `--max-concurrent <n>`, `--num-samples <n>`, `--no-resume`.

---

## 🔑 Key Concepts

- **Program** — A versioned agent configuration (system prompt + skills), stored as a git branch.
- **Frontier** — The top-N highest-scoring programs, tracked via git tags and branches.
- **Evolution Mode** — `skill_only` discovers new reusable skills; `prompt_only` optimizes the system prompt directly.
- **Skill** — A reusable capability file written to `.claude/skills/` that the agent can invoke during execution.
- **Proposer** — Analyzes agent failures and suggests what skill or prompt change would help.
- **Generator** — Takes a proposal and produces the actual skill file or prompt rewrite.

---

## 🧩 Extending EvoSkill: Adding a New Task

EvoSkill is designed to be extended to new benchmarks. There are two approaches: using the **Python API** (recommended) or creating **standalone scripts**.

### Option A: Using `register_task` (recommended)

#### 1. Create an Agent Profile

Add a new directory under `src/agent_profiles/` for your task:

```
src/agent_profiles/my_task_agent/
├── __init__.py
├── my_task_agent.py    # Options factory
└── prompt.txt          # (optional) task-specific system prompt
```

Your agent module should expose a `make_*_agent_options` factory that returns `ClaudeAgentOptions`. See `src/agent_profiles/dabstep_agent/dabstep_agent.py` or `src/agent_profiles/sealqa_agent/sealqa_agent.py` for reference.

Then register the exports in `src/agent_profiles/__init__.py`.

#### 2. Create a Scorer (optional)

Add a scorer under `src/evaluation/` that compares the agent's output to ground truth:

```python
# src/evaluation/my_task_scorer.py

def score_my_task(question: str, predicted: str, ground_truth: str) -> float:
    """Return 1.0 if correct, 0.0 otherwise."""
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
```

For more complex grading (e.g. partial credit or LLM-based judging), see `src/evaluation/sealqa_scorer.py`. If no scorer is provided, the default multi-tolerance scorer is used.

#### 3. Register and run

```python
from src.api import TaskConfig, register_task, EvoSkill, EvalRunner

register_task(TaskConfig(
    name="my_task",
    make_agent_options=make_my_task_agent_options,
    scorer=score_my_task,
    column_renames={"label": "ground_truth", "topic": "category"},
    default_dataset=".dataset/my_data.csv",
))

# Run the self-improvement loop
result = await EvoSkill(task="my_task").run()

# Or run a standalone evaluation
summary = await EvalRunner(task="my_task", model="sonnet").run()
```

### Option B: Standalone scripts

You can also create scripts directly under `scripts/` following the existing patterns.

**Evaluation script** — loads your dataset and runs `evaluate_full()`:

```python
from src.agent_profiles import Agent, make_my_task_agent_options
from src.evaluation.eval_full import evaluate_full
from src.schemas import AgentResponse

agent = Agent(make_my_task_agent_options(model="opus"), AgentResponse)
results = await evaluate_full(agent=agent, items=items, output_path=output, ...)
```

See `scripts/run_eval_dabstep.py` for a complete example.

**Loop script** — follow the pattern in `scripts/run_loop.py`. The key ingredients are:

- A **dataset split** function (train set for failure analysis, validation set for scoring)
- Your **agent options factory** and **scorer** wired into `SelfImprovingLoop`
- A `LoopConfig` with your chosen mode (`skill_only` or `prompt_only`)

---

## 📚 Citation

If you use EvoSkill in your research, please cite:

```bibtex
@misc{alzubi2026evoskillautomatedskilldiscovery,
  title={EvoSkill: Automated Skill Discovery for Multi-Agent Systems},
  author={Salaheddin Alzubi and Noah Provenzano and Jaydon Bingham and Weiyuan Chen and Tu Vu},
  year={2026},
  eprint={2603.02766},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2603.02766},
}
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.