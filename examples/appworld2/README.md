# EvoSkill × AppWorld via HALO

Run EvoSkill's self-improving evolution loop on the [AppWorld benchmark](https://appworld.dev) (9 simulated apps, 457 APIs, 728 tasks) using [HALO's](https://github.com/inference-labs-inc/halo) proven agent pipeline.

## How It Works

```
EvoSkill (prompt evolution)          HALO (task execution)
─────────────────────────           ──────────────────────

SelfImprovingLoop                   OpenAI Agents SDK + Claude (via LiteLLM)
  │                                   │
  ├── Sample train tasks              ├── Read instructions.txt (evolved prompt)
  │                                   ├── API predictor (457 → ~20 tools)
  ├── HALOAgent.run(task) ──────────> ├── Agent loop: Claude ↔ AppWorld MCP
  │                                   ├── Official AppWorld evaluation
  ├── Collect failures <──────────── └── Results on disk
  │
  ├── Prompt proposer (Claude)
  │   └── Analyzes failure traces
  │
  ├── Prompt generator (Claude)
  │   └── Rewrites instructions.txt
  │
  ├── HALOAgent.run(val tasks) ─────> HALO runs with NEW prompt
  │
  └── Keep or discard based on score
```

EvoSkill evolves the system prompt. HALO executes the benchmark. They communicate through one file: `instructions.txt`.

---

## Prerequisites

- Python 3.12+
- `ANTHROPIC_API_KEY` in environment
- ~500MB disk for AppWorld data

## Setup

### Step 1: Clone EvoSkill and switch to the branch

```bash
git clone https://github.com/sentient-agi/EvoSkill.git
cd EvoSkill
git checkout evoskill-appworld
```

### Step 2: Install EvoSkill

```bash
pip install -e .
```

### Step 3: Install AppWorld + HALO agent harness

```bash
pip install appworld
pip install openai-agents litellm
```

### Step 4: Install HALO's AppWorld fork (for the agent runner)

```bash
# Clone HALO repo (we only need the AppWorld demo)
git clone https://github.com/inference-labs-inc/halo.git /path/to/halo

# Install the AppWorld agent harness
cd /path/to/halo/demo/appworld
pip install -e .
pip install -e './experiments[openai_agents]'

# Pull LFS files and run setup
git lfs install && git lfs pull
task setup  # or manually: appworld install --repo && appworld download data
```

### Step 5: Download AppWorld data into EvoSkill

```bash
cd /path/to/EvoSkill/examples/appworld2
APPWORLD_ROOT=$(pwd) appworld download data
```

This creates `data/` with 733 tasks (~190MB).

### Step 6: Update config.json

Edit `examples/appworld2/config.json`:

```json
{
    "appworld_root": "/absolute/path/to/EvoSkill/examples/appworld2",
    "model": "claude-sonnet-4-20250514",
    "experiment_name": "evoskill",
    "max_steps": 50
}
```

Set `appworld_root` to the **absolute path** of `examples/appworld2/`.

### Step 7: Set API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

All commands run from the EvoSkill root directory.

### Test the pipeline (single task)

```bash
python -m examples.appworld2.scripts.test_halo_agent --task-id 530b157_1
```

Expected: `PASS (score=1.0)` — sends money via Venmo, official AppWorld evaluation.

### Test on multiple tasks

```bash
# First 10 dev tasks
python -m examples.appworld2.scripts.test_halo_agent --n-tasks 10

# Specific tasks
python -m examples.appworld2.scripts.test_halo_agent \
  --task-id 50e1ac9_1 \
  --task-id fac291d_1 \
  --task-id 530b157_1

# All 57 dev tasks (full baseline)
python -m examples.appworld2.scripts.test_halo_agent
```

### Run the evolution loop

```bash
# Small test (2 train, 2 val, 1 iteration)
python -m examples.appworld2.scripts.run_evolution \
  --n-train 2 --n-val 2 --max-iterations 1

# Full run (20 train, 17 val, 5 iterations)
python -m examples.appworld2.scripts.run_evolution \
  --n-train 20 --n-val 17 --max-iterations 5
```

The evolution loop will:
1. Evaluate baseline prompt on validation tasks
2. Run training tasks, collect failures
3. Propose and generate an improved prompt
4. Re-evaluate with the new prompt
5. Keep if improved, discard if not
6. Repeat

---

## File Structure

```
examples/appworld2/
├── .claude/
│   ├── prompts/
│   │   └── instructions.txt      ← The prompt being evolved (git-versioned)
│   └── feedback_history.md       ← Past proposals + outcomes
├── experiments/
│   └── prompts/                  ← HALO reads prompts from here
│       ├── api_predictor.txt
│       └── function_calling_agent/
│           ├── instructions.txt  ← Synced copy of .claude/prompts/instructions.txt
│           └── demos.json        ← 3 few-shot examples (not evolved)
├── config.json                   ← Single source of truth for paths + model
├── scripts/
│   ├── halo_agent.py             ← HALOAgent — bridges HALO ↔ EvoSkill
│   ├── build_config.py           ← Builds HALO's runner config
│   ├── run_evolution.py          ← Wires into SelfImprovingLoop
│   └── test_halo_agent.py        ← Test/baseline runner
├── data/                         ← AppWorld benchmark data (gitignored, ~190MB)
└── .gitignore
```

---

## How HALOAgent Works

Each task goes through this exact pipeline:

```
HALOAgent.run("task_id:::instruction")
  │
  ├── _sync_prompt_to_halo()
  │   └── Copies .claude/prompts/instructions.txt → experiments/prompts/.../instructions.txt
  │
  ├── _run_halo_and_evaluate() [in a thread, with lock]
  │   │
  │   ├── AppWorld.initializer() → starts 9 app servers + MCP
  │   │
  │   ├── run_agent_on_tasks() → HALO's OpenAI Agents SDK
  │   │   ├── Reads instructions.txt (our evolved prompt)
  │   │   ├── API predictor (457 → ~20 tools)
  │   │   ├── Claude via LiteLLM calls AppWorld MCP tools
  │   │   └── Up to 50 turns
  │   │
  │   ├── evaluate_dataset() → Official AppWorld evaluation
  │   │   ├── Answer correctness
  │   │   └── DB state assertions (Venmo transactions, etc.)
  │   │
  │   └── AppWorld.initializer().__exit__() → stops servers
  │
  ├── Read results from disk:
  │   ├── supervisor.jsonl → agent's answer
  │   ├── lm_calls.jsonl → trace summary (for proposer)
  │   └── evaluations/on_only_{task_id}.json → official score
  │
  └── Return AgentTrace[AgentResponse]
```

---

## Costs and Timing

| Operation | Cost | Time |
|-----------|------|------|
| 1 AppWorld task | ~$1 | ~2 min |
| Proposer + Generator | ~$0.20 | ~30s |
| 1 evolution iteration (2 train + 2 val) | ~$5 | ~10 min |
| 1 evolution iteration (20 train + 17 val) | ~$40 | ~40 min |
| 5 iterations (full run) | ~$200 | ~3.5 hrs |

---

## Benchmark Comparison

| | HALO | EvoSkill |
|---|---|---|
| Approach | Analyze traces → report → human edits prompt | Automated: proposer → generator → evaluate → keep/discard |
| Sonnet on dev (reported) | 73.7% baseline → 89.5% after optimization | TBD |
| Automation | Semi-manual | Fully automated |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'examples'`**
→ Run from the EvoSkill root directory, not from `examples/appworld2/`.

**`OPENAI_API_KEY not set`**
→ HALO creates an OpenAI client even for Claude models. The agent sets a dummy key automatically, but if it fails, set `export OPENAI_API_KEY=unused`.

**`Couldn't set the db_home_path on the server`**
→ Previous AppWorld servers didn't shut down cleanly. Kill orphan processes: `pkill -f "appworld.cli serve"` and retry.

**`FileNotFoundError: .evoskill/feedback_history.md`**
→ The git branch is missing the file. Switch to master: `cd examples/appworld2 && git checkout master`.

**All tasks score 0.0**
→ Check that `ANTHROPIC_API_KEY` is set and valid. Check that AppWorld data is downloaded: `ls examples/appworld2/data/datasets/dev.txt`.
