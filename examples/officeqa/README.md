# OfficeQA Example

A minimal, self-contained example to try EvoSkill on your laptop. Uses a 10-question sample from the OfficeQA benchmark (U.S. Treasury bulletin Q&A) with just 9 source documents.

## What's included

```
examples/officeqa/
├── .evoskill/
│   ├── config.toml          # Pre-configured for this sample
│   └── task.md              # Task description and constraints
├── data/
│   ├── officeqa_sample.csv  # 10 questions (6 easy, 4 hard)
│   └── treasury_bulletins/  # 9 source documents referenced by the questions
└── README.md
```

## Quick start

### 1. Install EvoSkill

```bash
# From the repo root
pip install -e .
```

### 2. Set up an isolated git repo

EvoSkill stores program versions as git branches. Run the setup script so those branches stay inside this example directory and don't touch the parent repo:

```bash
cd examples/officeqa
bash setup.sh
```

### 3. Choose a config and run

#### Option A: Claude (default)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

evoskill eval --verbose
evoskill run
```

Uses `.evoskill/config.toml` (`claude` harness, `anthropic/claude-sonnet-4-6`).

#### Option B: OpenRouter via OpenCode

```bash
brew install opencode
export OPENROUTER_API_KEY="sk-or-..."

evoskill eval --config .evoskill/config.openrouter.toml --verbose
evoskill run --config .evoskill/config.openrouter.toml
```

Uses `.evoskill/config.openrouter.toml` (`opencode` harness, `openrouter/openai/gpt-5-mini`). Swap the model ID for any other OpenRouter model.

### 4. Inspect results

```bash
evoskill skills    # List discovered skills
evoskill logs      # View logs
evoskill diff      # Compare skill versions
```

## Configuration notes

| Setting | Default | Why |
|---------|---------|-----|
| `harness.name` | `claude` | Uses Claude Code as the agent runtime |
| `concurrency` | `2` | Conservative for laptop use; increase if you have higher rate limits |
| `evolution.iterations` | `3` | Enough to see improvement without a long wait |
| `train_ratio` / `val_ratio` | `0.4` / `0.2` | Higher ratios since we only have 10 questions |

## Dataset

The sample covers questions about U.S. federal expenditures, debt, interest rates, and international claims spanning 1939–2003. Each question references one treasury bulletin document in `data/treasury_bulletins/`.

| Difficulty | Count | Example |
|-----------|-------|---------|
| Easy | 6 | "What were the total claims made by the U.S on Zaire in 1997?" |
| Hard | 4 | "What were the total expenditures for U.S national defense in 1940?" |
