# EvoSkill container image.
#
# Supports all harnesses: Claude, OpenCode, Codex, Goose, OpenHands.
#
# Local (BYOC):
#   docker build -t evoskill .
#   evoskill run --docker
#
# Daytona (cross-compile for x86 if on Apple Silicon):
#   docker buildx build --platform linux/amd64 -t <your-registry>/evoskill:latest --push .
#   # Set image = "<your-registry>/evoskill:latest" in .evoskill/config.toml

FROM python:3.12-slim

# System deps
ARG NODE_MAJOR=20
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gnupg \
        libgomp1 \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# CLI tools for harnesses that need external binaries
#   claude   — npm (claude-agent-sdk spawns this)
#   opencode — npm (opencode harness spawns `opencode serve`)
#   codex    — npm (openai-codex-sdk spawns `codex`)
#   goose    — tarball from GitHub releases (goose harness spawns this)
RUN npm install -g @anthropic-ai/claude-code opencode-ai @openai/codex && npm cache clean --force

ARG GOOSE_VERSION=v1.33.1
RUN ARCH=$(uname -m) && \
    echo "Installing goose ${GOOSE_VERSION} for ${ARCH}" && \
    curl --http1.1 -fSL --retry 5 --retry-delay 10 \
      -o /tmp/goose.tar.gz \
      "https://github.com/block/goose/releases/download/${GOOSE_VERSION}/goose-${ARCH}-unknown-linux-gnu.tar.gz" && \
    tar xz -C /usr/local/bin/ -f /tmp/goose.tar.gz && \
    rm /tmp/goose.tar.gz

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
    "httpx>=0.23.0" \
    "hatchling" \
    "daytona>=0.1.0"

# Non-root user
RUN useradd -m -s /bin/bash evoskill \
    && mkdir -p /workspace && chown evoskill:evoskill /workspace

# Git config for bundle operations
RUN git config --global user.email "evoskill@sandbox" \
    && git config --global user.name "EvoSkill Sandbox" \
    && git config --global init.defaultBranch main

WORKDIR /workspace
USER evoskill
ENV PATH="/home/evoskill/.local/bin:${PATH}"
