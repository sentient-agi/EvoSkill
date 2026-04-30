#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# EvoSkill OfficeQA — Interactive Demo
#
# One script to set up, configure, and run the full evolution loop.
#
# Usage:
#   cd examples/officeqa
#   bash demo.sh              # run locally
#   bash demo.sh --remote     # run on Daytona sandbox
# ────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── helpers ───────────────────────────────────────────────────
bold()  { printf "\033[1m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
red()   { printf "\033[31m%s\033[0m" "$1"; }
dim()   { printf "\033[2m%s\033[0m" "$1"; }

step() { printf "\n  [%s] %s\n" "$1" "$2"; }

# ── [1/5] Setup ──────────────────────────────────────────────
step "1/5" "$(bold 'Setting up...')"

if ! command -v evoskill &>/dev/null; then
    printf "    $(red '✗') evoskill not found. Install with: pip install -e .\n"
    exit 1
fi

if ! command -v git &>/dev/null; then
    printf "    $(red '✗') git not found.\n"
    exit 1
fi

# Clean any leftover state from previous runs
rm -rf .git .cache .evoskill/state.json .evoskill/feedback_history.md .evoskill/loop_checkpoint.json .evoskill/reports
rm -f .claude/program.yaml
rm -rf .claude/skills
# Backup seed skills on first run, restore on subsequent runs
if [ ! -d ".seed-skills" ]; then
    cp -R .claude/skills .seed-skills
else
    rm -rf .claude/skills
    cp -R .seed-skills .claude/skills
fi

# Initialize fresh isolated git repo
cat > .evoskill/state.json <<'JSON'
{"original_branch": "main"}
JSON
git init -q
git add .
git commit -q -m "officeqa example — initial commit"
printf "    $(green '✓') Fresh git repo created\n"

# ── [2/6] Choose config ─────────────────────────────────────
step "2/5" "$(bold 'Choose a config:')"

# Build config list from .evoskill/config*.toml
declare -a CONFIG_FILES=()
declare -a CONFIG_NAMES=()
declare -a CONFIG_MODELS=()
declare -a CONFIG_KEYS=()

idx=0
for f in .evoskill/config*.toml; do
    [ -f "$f" ] || continue
    idx=$((idx + 1))
    CONFIG_FILES+=("$f")

    harness=$(grep '^name' "$f" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
    model=$(grep '^model' "$f" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
    CONFIG_NAMES+=("$harness")
    CONFIG_MODELS+=("$model")

    # Determine required API key from model prefix
    case "$model" in
        anthropic/*)    CONFIG_KEYS+=("ANTHROPIC_API_KEY") ;;
        openrouter/*)   CONFIG_KEYS+=("OPENROUTER_API_KEY") ;;
        openai/*)       CONFIG_KEYS+=("OPENAI_API_KEY") ;;
        google/*)       CONFIG_KEYS+=("GOOGLE_API_KEY") ;;
        deepseek/*)     CONFIG_KEYS+=("DEEPSEEK_API_KEY") ;;
        mistral/*)      CONFIG_KEYS+=("MISTRAL_API_KEY") ;;
        groq/*)         CONFIG_KEYS+=("GROQ_API_KEY") ;;
        together/*)     CONFIG_KEYS+=("TOGETHER_API_KEY") ;;
        xai/*)          CONFIG_KEYS+=("XAI_API_KEY") ;;
        codex-*)        CONFIG_KEYS+=("OPENAI_API_KEY") ;;
        *)              CONFIG_KEYS+=("") ;;
    esac

    printf "    %d) %-12s — %s\n" "$idx" "$harness" "$model"
done

if [ "$idx" -eq 0 ]; then
    printf "    $(red '✗') No config files found in .evoskill/\n"
    exit 1
fi

printf "    "
read -rp "> " choice

# Validate choice
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "$idx" ]; then
    printf "    $(red '✗') Invalid choice\n"
    exit 1
fi

i=$((choice - 1))
CHOSEN_FILE="${CONFIG_FILES[$i]}"
CHOSEN_NAME="${CONFIG_NAMES[$i]}"
CHOSEN_MODEL="${CONFIG_MODELS[$i]}"
REQUIRED_KEY="${CONFIG_KEYS[$i]}"

# Check API key
if [ -n "$REQUIRED_KEY" ]; then
    if [ -z "${!REQUIRED_KEY:-}" ]; then
        printf "    $(red '✗') %s is not set\n" "$REQUIRED_KEY"
        read -rp "    Paste your key (or Ctrl-C to cancel): " key_value
        export "$REQUIRED_KEY=$key_value"
    fi
    printf "    $(green '✓') %s is set\n" "$REQUIRED_KEY"
fi

# ── [3/5] Run the loop ───────────────────────────────────────
step "3/5" "$(bold 'Running evolution loop...')"

iterations=$(grep 'iterations' "$CHOSEN_FILE" 2>/dev/null | head -1 | sed 's/[^0-9]//g')
iterations=${iterations:-3}

printf "    Harness: %s | Model: %s | Iterations: %s\n" "$CHOSEN_NAME" "$CHOSEN_MODEL" "$iterations"
printf "    %s\n\n" "$(dim '─────────────────────────────────────────')"

# Build command
CMD="evoskill run --verbose"
if [ "$(basename "$CHOSEN_FILE")" != "config.toml" ]; then
    CMD="$CMD --config $CHOSEN_FILE"
fi

$CMD

printf "\n    %s\n" "$(dim '─────────────────────────────────────────')"

# ── [4/5] Results ────────────────────────────────────────────
step "4/5" "$(bold 'Discovered skills')"
echo ""
evoskill skills 2>/dev/null || printf "    $(dim 'No skills discovered yet.')\n"

# ── [5/5] Cleanup ────────────────────────────────────────────
step "5/5" "$(bold 'Cleaning up')"
rm -rf .git .cache .evoskill/state.json .evoskill/feedback_history.md .evoskill/loop_checkpoint.json .evoskill/reports
rm -f .claude/program.yaml
rm -rf .claude/skills
cp -R .seed-skills .claude/skills
printf "    $(green '✓') Restored to clean state\n"
printf "    Ready for another run: bash demo.sh\n"
echo ""
