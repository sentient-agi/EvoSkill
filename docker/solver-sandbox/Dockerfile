# Solver sandbox: minimal container for BrowseComp solver bash commands.
# Only workspace is mounted. curl/grep/python available. No host filesystem access.
#
# Build: docker build -t evoskill-solver-sandbox docker/solver-sandbox/
# Run:   docker run --rm --network host -v /path/to/workspace:/workspace evoskill-solver-sandbox bash -c "curl -s http://localhost:8099/health"

# BASE PINNED BY DIGEST, not by tag. `python:3.12-slim` is a moving pointer: two builds
# a month apart inherit different OS packages, so an image built from a tag is not the
# image the numbers were produced on, and nothing in the pipeline would say so. Refresh
# deliberately with:
#     docker manifest inspect -v python:3.12-slim | python3 -c "import json,sys; print(json.load(sys.stdin)['Descriptor']['digest'])"
# Resolved 2026-08-18.
FROM python:3.12-slim@sha256:876416ecde9aca2bcc90e1fb0c7a9500bbf749f5788b70f82d4c5a5c2357f8b4

# Install common tools the solver needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    grep \
    findutils \
    coreutils \
    sed \
    gawk \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Working directory = mounted workspace
WORKDIR /workspace

# No default command — each bash call is a separate `docker exec`
CMD ["bash"]
