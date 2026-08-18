# Solver sandbox: minimal container for BrowseComp solver bash commands.
# Only workspace is mounted. curl/grep/python available. No host filesystem access.
#
# Build: docker build -t evoskill-solver-sandbox docker/solver-sandbox/
# Run:   docker run --rm --network host -v /path/to/workspace:/workspace evoskill-solver-sandbox bash -c "curl -s http://localhost:8099/health"

FROM python:3.12-slim

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
