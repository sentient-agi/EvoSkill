#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# EvoSkill OfficeQA example — one-time setup
#
# Initializes an isolated git repo so that program branches,
# frontier tags, and loop state stay inside this directory and
# never pollute the parent EvoSkill repository.
#
# Usage:
#   cd examples/officeqa
#   bash setup.sh
# ────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── guard: already initialised ────────────────────────────────
if [ -d ".git" ]; then
    echo "✓ Git repo already exists in $(pwd). Nothing to do."
    exit 0
fi

# ── write state.json (so evoskill reset lands correctly) ──────
cat > .evoskill/state.json <<'JSON'
{"original_branch": "main"}
JSON

# ── init isolated repo ────────────────────────────────────────
git init
git add .
git commit -m "officeqa example — initial commit"

echo ""
echo "✓ Isolated git repo created in $(pwd)"
echo ""
echo "Next steps:"
echo ""
echo "  Option A — Claude:"
echo "    export ANTHROPIC_API_KEY=\"sk-ant-...\""
echo "    evoskill eval --verbose"
echo "    evoskill run"
echo ""
echo "  Option B — OpenRouter via OpenCode:"
echo "    export OPENROUTER_API_KEY=\"sk-or-...\""
echo "    evoskill eval --config .evoskill/config.openrouter.toml --verbose"
echo "    evoskill run --config .evoskill/config.openrouter.toml"
