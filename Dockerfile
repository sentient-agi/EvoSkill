# EvoSkill container image.
#
# Supports all harnesses: Claude, OpenCode, Codex, Goose, OpenHands.
#
# Local (BYOC):
#   docker build -t evoskill .
#   evoskill run --docker
#
# Daytona:
#   docker build -t evoskill .
#   docker tag evoskill <your-registry>/evoskill:latest
#   docker push <your-registry>/evoskill:latest
#   # Set image = "<your-registry>/evoskill:latest" in .evoskill/config.toml

FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# CLI tools for harnesses that need external binaries
#   claude   — npm (claude-agent-sdk spawns this)
#   opencode — npm (opencode harness spawns `opencode serve`)
#   codex    — npm (openai-codex-sdk spawns `codex`)
#   goose    — tarball from GitHub releases (goose harness spawns this)
RUN npm install -g @anthropic-ai/claude-code opencode-ai @openai/codex && npm cache clean --force

RUN ARCH=$(uname -m) && \
    GOOSE_VER=$(curl -sL "https://api.github.com/repos/block/goose/releases/latest" \
      | grep -m1 '"tag_name"' | cut -d'"' -f4) && \
    if [ -z "$GOOSE_VER" ]; then \
      GOOSE_VER=$(curl -sL "https://api.github.com/repos/block/goose/releases" \
        | grep -m1 '"tag_name"' | cut -d'"' -f4); \
    fi && \
    echo "Installing goose ${GOOSE_VER} for ${ARCH}" && \
    curl -fsSL "https://github.com/block/goose/releases/download/${GOOSE_VER}/goose-${ARCH}-unknown-linux-gnu.tar.gz" \
    | tar xz -C /usr/local/bin/

# Python deps — core + all harness SDKs.
# Excludes packages not imported at runtime (torch, datasets, notebook, plotly, etc.)
RUN pip install --no-cache-dir \
    "claude-agent-sdk>=0.1.16" \
    "opencode-ai>=0.1.0a36" \
    "openai-codex-sdk>=0.1.11" \
    "openhands-sdk>=1.16.1" \
    "openhands-tools>=1.16.1" \
    "click>=8.1.8" \
    "rich>=14.2.0" \
    "pandas>=2.3.3" \
    "pydantic>=2.12.5" \
    "pydantic-settings>=2.12.0" \
    "pyyaml>=6.0" \
    "questionary>=2.1.1" \
    "tqdm>=4.60.0" \
    "hatchling" \
    "daytona>=0.1.0"

# Git config for bundle operations
RUN git config --global user.email "evoskill@sandbox" \
    && git config --global user.name "EvoSkill Sandbox" \
    && git config --global init.defaultBranch main

WORKDIR /workspace
