# SpreadsheetBench v2 SOLVER sandbox — BASELINE (no formula engine).
#
# Build: docker build -t ssb-solver:latest docker/ssb-solver/
#
# WHY THIS FILE EXISTS. Until 2026-08-02 this recipe lived ONLY as a heredoc in
# docs/ssb_infra_note.md, recovered here from `docker history ssb-solver:latest`
# on image 974b00f17fbe (built 2026-07-15). That is not a safe place for it: the
# SIBLING image's recipe lived in the same doc, the doc's heredoc omitted
# `python3-uno`, and the 2026-07-15 rebuild of ssb-verifier therefore dropped it
# — every Data-Table UNO solve then failed exit-127, fell back silently to plain
# recalc, and corrupted rescore verdicts (38/157 -> 21/157 on the GLM-5.2
# best-skill leg). A versioned Dockerfile makes that class of silent loss a diff.
#
# THIS IMAGE IS THE CONTROL, NOT THE DEFAULT GOING FORWARD. It has no formula
# engine at all: openpyxl returns None for any formula without a cached value, so
# a solver here CANNOT evaluate the formulas it writes, while the scorer
# recalculates them with LibreOffice and compares values against golden. See
# Dockerfile.libreoffice for the arm that closes that gap, and keep this one
# buildable so the pre-LibreOffice condition stays reproducible.
#
# The live 974b00f17fbe is also tagged `ssb-solver:no-libreoffice-0715`. Prefer
# that tag over `:latest` anywhere the PRE-LibreOffice environment is what you
# mean — `:latest` is a moving target, the archival tag is not.
#
# NOTE ON `docker build` AND THE CORPUS FIREWALL: apt/pip need egress. The
# DOCKER-USER default-DROP chain installed by
# executor.py::_ensure_corpus_only_firewall (only when EVOSKILL_CORPUS_ONLY=1)
# persists in the Colima VM until a restart and WILL make this build fail on
# apt/pip. Check with `colima ssh -- sudo iptables -L DOCKER-USER -n` first.

# BASE PINNED BY DIGEST, not by tag. `python:3.11-slim` is a moving pointer: two builds
# a month apart inherit different OS packages, so an image built from a tag is not the
# image the numbers were produced on, and nothing in the pipeline would say so. Refresh
# deliberately with:
# REFRESH IT WITH THIS, AND NOTHING ELSE:
#     docker buildx imagetools inspect python:3.11-slim --format '{{.Manifest.Digest}}'
#
# NOT `docker manifest inspect -v ... .Descriptor.digest`, which is what produced the
# previous pin here and was WRONG: on a multi-arch tag that returns one PLATFORM manifest
# (it picked linux/amd64), so the pin silently switched every build on this arm64 host to
# an emulated amd64 image -- an architecture change shipped under the name "reproducibility".
# The value below is the multi-arch INDEX (mediaType application/vnd.oci.image.index.v1+json,
# 16 platforms incl. arm64), so each host still resolves to its own native image while the
# CONTENT stays pinned. Verify a candidate before committing it:
#     docker buildx imagetools inspect python:3.11-slim@<digest> --format '{{.Manifest.MediaType}}'
#   must print application/vnd.oci.image.index.v1+json
# Resolved 2026-08-18.
FROM python:3.11-slim@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

# Pinned exactly as the 2026-07-15 image: openpyxl is the only spreadsheet
# library, and it is a READER/WRITER, not an evaluator.
RUN pip install --no-cache-dir openpyxl==3.1.3 pandas==2.2.0 numpy==1.26.4

# /workspace, NOT /app. The image used to ship empty /app/spreadsheets and
# /app/output with WORKDIR /app (an SSB v1 leftover). Solvers occasionally
# `cd /app` on their own prior; because the decoy dirs existed the cd SUCCEEDED,
# the solver did all its work in container-local /app, its output never reached
# the host bind-mount at /workspace, and the task scored 0 while the solver's own
# verification passed (both sides of its compare lived inside the container).
# Cost two val tasks on 2026-07-01 (Financial_Model/02_05, Template/02_04). With
# no /app, a stray `cd /app` fails immediately and the model recovers by itself.
WORKDIR /workspace
