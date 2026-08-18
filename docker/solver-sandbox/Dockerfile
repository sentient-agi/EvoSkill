# Solver sandbox: minimal container for BrowseComp solver bash commands.
# Only workspace is mounted. curl/grep/python available. No host filesystem access.
#
# Build: docker build -t evoskill-solver-sandbox docker/solver-sandbox/
# Run:   docker run --rm --network host -v /path/to/workspace:/workspace evoskill-solver-sandbox bash -c "curl -s http://localhost:8099/health"

# BASE PINNED BY DIGEST, not by tag. `python:3.12-slim` is a moving pointer: two builds
# a month apart inherit different OS packages, so an image built from a tag is not the
# image the numbers were produced on, and nothing in the pipeline would say so. Refresh
# deliberately with:
# REFRESH IT WITH THIS, AND NOTHING ELSE:
#     docker buildx imagetools inspect python:3.12-slim --format '{{.Manifest.Digest}}'
#
# NOT `docker manifest inspect -v ... .Descriptor.digest`, which is what produced the
# previous pin here and was WRONG: on a multi-arch tag that returns one PLATFORM manifest
# (it picked linux/amd64), so the pin silently switched every build on this arm64 host to
# an emulated amd64 image -- an architecture change shipped under the name "reproducibility".
# The value below is the multi-arch INDEX (mediaType application/vnd.oci.image.index.v1+json,
# 16 platforms incl. arm64), so each host still resolves to its own native image while the
# CONTENT stays pinned. Verify a candidate before committing it:
#     docker buildx imagetools inspect python:3.12-slim@<digest> --format '{{.Manifest.MediaType}}'
#   must print application/vnd.oci.image.index.v1+json
# Resolved 2026-08-18.
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

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
