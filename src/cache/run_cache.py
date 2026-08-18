"""
Program-aware run caching for agent evaluations.

Cache invalidates automatically when behavior-affecting files change:
- .claude/skills/** (skill definitions)
- src/agent_profiles/base_agent/prompt.txt (legacy fallback) and
  src/agent_profiles/officeqa_agent/prompt.md (prompt text)

Excludes metadata files like .claude/program.yaml to avoid unnecessary
cache invalidation when only scores or timestamps change.
"""

import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from src.harness import AgentTrace

logger = logging.getLogger(__name__)
# Cache-write anomalies MUST reach the run log (nohup stderr) even when no
# global logging config exists. Python's lastResort handler covers the
# no-handler case, but an otel/phoenix root handler with a higher level can
# swallow WARNINGs silently — attach a dedicated stderr handler so every
# skipped/failed cache write is visible no matter what the app configured.
# (2026-07-22 nex evo run post-mortem: zero cache log lines in a run that
# lost 71 of 117 writes.)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _cache_log_handler = logging.StreamHandler()
    _cache_log_handler.setLevel(logging.WARNING)
    _cache_log_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s RunCache: %(message)s", "%H:%M:%S"))
    logger.addHandler(_cache_log_handler)
    if logger.level in (logging.NOTSET,) or logger.level > logging.WARNING:
        logger.setLevel(logging.WARNING)

T = TypeVar("T", bound=BaseModel)


class CacheConfig(BaseModel):
    """Configuration for RunCache behavior."""

    cache_dir: Path = Path(".cache/runs")
    enabled: bool = True
    store_messages: bool = False  # Whether to cache the full messages list
    hash_length: int = 12  # Length of hash prefix for filenames
    cwd: Path = Path(".")  # Working directory for git commands
    # Path to the LIVE `.claude/skills/` the agent reads from at run time.
    # When workspace ≠ project source root (typical with `--workspace`),
    # the workspace's `.claude/skills/` only contains stub snapshot files
    # — the real skills evolve in the project's `.claude/skills/`. Hashing
    # the workspace dir alone produces a stable cache key even as the
    # actual loaded skills change between iterations, returning stale
    # responses generated under different skill conditions. Pointing here
    # at the live dir fixes that. Defaults to `<cwd>/.claude/skills` for
    # backward compatibility (single-root setups where workspace == project).
    live_skills_dir: Path | None = None
    # Path to the project source root (parent of `src/agent_profiles/...`)
    # so the cache hashes the agent's actual prompt files. Without this,
    # `_get_tree_hash` looks for prompts at `<cwd>/src/agent_profiles/...`
    # which DOES NOT EXIST in a workspace setup → prompt edits go undetected.
    # Defaults to `cwd` for single-root backward compatibility.
    project_source_root: Path | None = None
    # Optional artifact capture: callable(question) -> list[(arcname, dir)]
    # or None. When set, cache writes ALSO tar the listed directories next
    # to the entry, and cache reads restore them before returning the trace
    # — making replay faithful for file-producing benchmarks (a replayed
    # rollout otherwise leaves the task workspace EMPTY, starving any
    # downstream consumer of those files, e.g. the evolver's
    # output-vs-golden diffing; found Jul 1 on SSB v2). A hit whose entry
    # predates artifact capture (no tar on disk) is treated as a MISS so
    # stale artifact-less entries self-invalidate instead of silently
    # replaying without files.
    artifact_resolver: Any = None
    # --- cached_transcript rendering budget ------------------------------
    # Defaults are the values that were hardcoded at the two render sites
    # until 2026-08-04, so leaving these unset is byte-for-byte the old
    # behaviour for every harness.
    #
    # WHY THEY ARE CONFIGURABLE NOW (#149). `cached_transcript` is the ONLY
    # structurally-usable record of a rollout after a cache write, because
    # `messages` does not round-trip through Pydantic and `store_messages`
    # is off by default. On tau3 banking_knowledge the retrieval tool
    # returns ten full policy documents per call, so 26 of 55 stored
    # rollouts blew the 200K total cap — median 60% of turns survived and
    # the worst kept 7 of 86 (a rollout that SCORED 1.0). Because the cap
    # was head-only, what was lost was always the END: the commit, the
    # termination, the outcome. A benchmark whose tool results are large
    # therefore needs a bigger budget, and a benchmark whose transcripts
    # are small should not pay for it.
    transcript_tool_result_max_chars: int = 20_000
    transcript_max_chars: int = 200_000
    # Optional callable(question) -> None, invoked after a successful cache
    # hit (post artifact restore). Use it to reconstruct workspace state
    # that live execution creates OUTSIDE the captured artifact dirs — e.g.
    # staging static input files that the agent's per-task setup would have
    # copied (setup never runs on a hit; found Jul 1 when replayed SSB v2
    # tasks had output/ restored but no spreadsheets/, breaking the
    # evolver's manifest contract). Static inputs are staged from their
    # source rather than captured per-entry: no cache space, no invalidation.
    on_replay: Any = None

    class Config:
        arbitrary_types_allowed = True


class CacheEntry(BaseModel):
    """Schema for a cached run entry."""

    version: str = "1.0"
    created_at: str

    cache_key: dict[str, str]  # tree_hash, question_hash, question
    trace: dict[str, Any]  # Serialized AgentTrace


def _wall_skip_reasons() -> set[str]:
    """Termination reasons that must never be cached, or `set()` when off.

    Opt-in via EVOSKILL_CACHE_SKIP_WALL_LIMITED. Off by default because most
    harnesses have no wall-limited termination concept and would pay a full
    re-execution for entries this would refuse.
    """
    if os.environ.get("EVOSKILL_CACHE_SKIP_WALL_LIMITED", "") in ("", "0"):
        return set()
    return {r.strip().lower() for r in os.environ.get(
        "EVOSKILL_CACHE_WALL_REASONS", "timeout").split(",") if r.strip()}


def _termination_reason(output: Any) -> str | None:
    """Pull `termination_reason` out of an agent output, however it is shaped.

    ONE extractor for both cache sides on purpose. A store-side skip without a
    matching get-side refusal is not half a fix, it is a WORSE state than no fix
    at all: the store side stops adding wall-limited records while the get side
    keeps serving every one already on disk, so the defect looks repaired and
    the stale zeros replay forever. Any future change to the shape of an agent
    output must therefore break or fix both sides together.

    Handles the three shapes this field arrives in: a dict output, a pydantic
    output with the attribute, and `final_answer` holding a JSON *string*
    (the tau2/tau3 worker's framing). Reading the dict's own key first and
    only then parsing the string matters — `final_answer` is the producer's
    serialisation, and guessing at its shape is how a scan of these very
    records once reported a FALSE ZERO ("0 cached timeouts" when 13 existed).
    """
    if output is None:
        return None
    fa = output.get("final_answer") if isinstance(output, dict) else getattr(
        output, "final_answer", None)
    for cand in (output if isinstance(output, dict) else None, fa):
        if isinstance(cand, dict) and cand.get("termination_reason"):
            return str(cand["termination_reason"])
    if fa is not None and not isinstance(fa, dict):
        try:
            parsed = json.loads(str(fa))
        except Exception:
            return None
        if isinstance(parsed, dict) and parsed.get("termination_reason"):
            return str(parsed["termination_reason"])
    return None


class RunCache:
    """
    Program-aware cache for agent run results.

    Keys on hash(behavior-affecting files) + hash(question) to automatically
    invalidate when skills or prompts change. Ignores metadata changes
    (scores, timestamps) to avoid unnecessary cache invalidation.

    Usage:
        cache = RunCache()

        # Check cache (sdk + model are required keyword args — they're part of the key)
        trace = cache.get(question, sdk="claude", model="claude-opus-4-6")
        if trace is None:
            trace = await agent.run(question)
            cache.set(question, trace, sdk="claude", model="claude-opus-4-6")
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize RunCache.

        Args:
            config: Cache configuration. Uses defaults if not provided.
        """
        self.config = config or CacheConfig()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        if self.config.enabled:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_tree_hash(self, system_prompt: str | None = None) -> str:
        """
        Get combined hash of state that affects agent behavior.

        Two content-addressed ingredients (workspace-independent):
          (a) live `.claude/skills/` file content — captures the exact
              skill guidance the solver will see
          (b) the *effective* system prompt the agent will run with

        The hash is purely content-based: two different workspaces with
        identical skills and prompt produce the same tree_hash, enabling
        cross-workspace cache sharing. Git state is intentionally excluded
        — the live skills hash already captures on-disk content, and the
        git SHA added a workspace-specific component that prevented cache
        hits between eval/evo/smoke workspaces with identical solver state.

        Args:
            system_prompt: The exact system prompt the agent will receive.

        Returns:
            Combined hash of behavior-affecting state.
        """
        # Hash the LIVE skills directory (where the agent actually reads
        # skill files from). When the loop runs with --workspace, the
        # live dir is on the project side (e.g. <EvoSkill>/.claude/skills),
        # not the workspace side (which carries only stub snapshots). If
        # we hash the workspace dir, the cache key stays constant across
        # iterations even as skills change, masking the evolver's effect.
        skills_dir = self.config.live_skills_dir or (
            self.config.cwd / ".claude" / "skills"
        )
        # Always compute via _hash_files. Path.glob() returns empty on a
        # nonexistent path, so the result is sha256(no_files) — the same
        # value an existing-but-empty directory produces. Without this
        # normalization, a workspace where `.claude/skills/` was created
        # (e.g. by build_pdf_only_workspace) and one where it hasn't been
        # created yet would produce different tree_hashes for the same
        # logical "no live skill content" state, causing spurious cache
        # misses between consecutive runs with identical agent config.
        skills_h = self._hash_files(skills_dir, "**/*")

        # Prompt hash: prefer the actual system_prompt the agent will use.
        # Fall back to file content when the caller didn't pass one.
        if system_prompt is not None:
            prompt_h = hashlib.sha256(
                ("inline:" + system_prompt).encode("utf-8")
            ).hexdigest()
        else:
            # Fallback: hash whichever known agent prompt files exist on disk.
            source_root = self.config.project_source_root or self.config.cwd
            prompt_paths = [
                source_root / "src" / "agent_profiles" / "solver" / "prompt.txt",
                source_root / "src" / "agent_profiles" / "officeqa_agent" / "prompt.md",
            ]
            hasher = hashlib.sha256()
            for prompt_path in prompt_paths:
                try:
                    if prompt_path.is_file():
                        with open(prompt_path, "rb") as f:
                            hasher.update(prompt_path.name.encode("utf-8"))
                            hasher.update(f.read())
                except (IOError, OSError):
                    pass
            prompt_h = hasher.hexdigest()

        combined = f"skills:{skills_h}|prompt:{prompt_h}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _git_behavior_token(self) -> str | None:
        """Return a git-derived token for the workspace's current skill state.

        Uses ONLY the skills tree SHA (`git rev-parse HEAD:.claude/skills`).
        Previously also incorporated the root tree SHA, but that captured
        unrelated mutations (e.g., ProgramManager committing "Update score"
        to `.claude/program.yaml`) which shifted the tree_hash without any
        change to actual solver behavior. The result was cache MISSES
        between consecutive runs with identical agent config — exactly
        the bug we hit on the PDF-only evo iter-0 baseline (paid $7 to
        re-execute predictions already in cache under a different hash).

        Tree SHAs are content-addressed: identical content produces the
        same SHA regardless of when the commit was made.

        Returns None when git isn't available, the cwd isn't a repo, or
        `.claude/skills` isn't tracked.
        """
        try:
            import subprocess
            try:
                skills_tree = subprocess.check_output(
                    ["git", "rev-parse", "HEAD:.claude/skills"],
                    cwd=self.config.cwd, text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except subprocess.CalledProcessError:
                # Skills dir not tracked on this commit → use empty-tree SHA
                skills_tree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
            return skills_tree
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    def _hash_files(self, dir_path: Path, pattern: str = "**/*") -> str:
        """
        Compute content hash of files matching a pattern in a directory.

        Args:
            dir_path: Path to directory to search in
            pattern: Glob pattern for files to include (default: all files)

        Returns:
            SHA256 hash of matching file contents
        """
        hasher = hashlib.sha256()

        # Get matching files, sorted for consistency
        files = sorted(dir_path.glob(pattern))
        for file_path in files:
            if file_path.is_file():
                # Include relative path and content in hash
                rel_path = file_path.relative_to(dir_path)
                hasher.update(str(rel_path).encode("utf-8"))
                try:
                    with open(file_path, "rb") as f:
                        hasher.update(f.read())
                except (IOError, OSError):
                    # Skip files that can't be read
                    pass

        return hasher.hexdigest()

    @staticmethod
    def _compute_question_hash(question: str) -> str:
        """Compute hash for a question string."""
        return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _artifact_path(cache_path: Path) -> Path:
        return cache_path.with_name(cache_path.stem + ".artifacts.tar.gz")

    @staticmethod
    def _accepts_sample_idx(fn: Any) -> bool:
        """Does this hook take a ``sample_idx`` keyword?

        Preferred over calling with the kwarg and catching TypeError: a
        TypeError raised INSIDE a sample-aware hook is indistinguishable from a
        signature mismatch, so the fallback would silently re-invoke the 1-arg
        form and quietly operate on the wrong (shared) directory — the very
        failure mode task #108 was about.
        """
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        if "sample_idx" in params:
            return True
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def _artifact_spec(
        self, question: str, sample_idx: int | None = None
    ) -> list[tuple[str, Path]] | None:
        """Artifact dirs for one rollout.

        Passes sample_idx to resolvers that accept it (falling back to the
        1-arg form for older callables, same duck-typing pattern trace_db uses
        for task_pointers). With n_samples>1 each sample has its OWN task
        workspace, so a sample-aware resolver returns DISJOINT dirs — without
        that, all N samples captured and restored the same paths and the last
        restore won (task #108, seen live 2026-07-24).
        """
        resolver = self.config.artifact_resolver
        if resolver is None:
            return None
        try:
            if self._accepts_sample_idx(resolver):
                spec = resolver(question, sample_idx=sample_idx)
            else:
                spec = resolver(question)
        except Exception:
            return None
        if not spec:
            return None
        return [(str(arc), Path(d)) for arc, d in spec]

    def _run_on_replay(self, question: str, sample_idx: int | None = None) -> None:
        """Invoke the config's on_replay hook (see CacheConfig.on_replay).

        Sample-aware for the same reason as ``_artifact_spec``: with n_samples>1
        each sample owns its workspace, so a hook that stages inputs must stage
        them into THAT sample's dir. Before this, restore put each sample's
        output under its own workspace while on_replay staged the inputs into the
        bare ``{task_id}/`` dir — so every per-sample workspace came back from
        cache with an output but no inputs, breaking the evolver's
        ``$WS/spreadsheets/`` contract (the SSB v2 Jul 1 lesson, re-armed by the
        2026-07-24 per-sample-workspace change).
        """
        hook = self.config.on_replay
        if hook is None:
            return
        try:
            if self._accepts_sample_idx(hook):
                hook(question, sample_idx=sample_idx)
            else:
                hook(question)
        except Exception as e:
            logger.warning(f"on_replay hook failed for question hash "
                           f"{self._compute_question_hash(question)[:8]}: {e}")

    def _capture_artifacts(self, cache_path: Path, question: str,
                           sample_idx: int | None = None) -> None:
        """Tar the resolver's directories next to the cache entry."""
        spec = self._artifact_spec(question, sample_idx)
        if spec is None:
            return
        import tarfile
        art_path = self._artifact_path(cache_path)
        # #116: the temp name MUST be unique per writer. It used to be a fixed
        # `<name>.tmp`, so two processes capturing the SAME entry concurrently
        # both opened that one path, interleaved their gzip streams, and one
        # renamed a TORN tarball into place. Nothing raised — tarfile.open
        # succeeds, the rename succeeds, and the corruption only surfaces later
        # in _restore_artifacts, on a replay, as wrong files scored as if real.
        # The JSON writers already use this {pid}.{uuid} pattern (see
        # _write_entry); the artifact writer was the one that did not, which is
        # exactly why #115 (sharing one base cache across arms) was blocked.
        tmp = art_path.with_name(f"{art_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with tarfile.open(tmp, "w:gz") as tar:
                for arcname, dir_path in spec:
                    if dir_path.exists():
                        tar.add(dir_path, arcname=arcname)
            # Atomic within a filesystem: a concurrent writer either sees the
            # old entry or this complete one, never a partial tar.
            tmp.rename(art_path)
        except (OSError, tarfile.TarError) as e:
            # TarError is NOT an OSError — a tar-level failure used to escape
            # this handler and leave the temp file behind forever.
            logger.warning(f"Artifact capture failed for {cache_path.name}: {e}")
            tmp.unlink(missing_ok=True)

    def _restore_artifacts(self, cache_path: Path, question: str,
                           sample_idx: int | None = None) -> bool:
        """Restore captured artifacts for a hit. Returns False when the
        entry is artifact-aware but has no tar (predates capture) — the
        caller must treat that hit as a MISS."""
        spec = self._artifact_spec(question, sample_idx)
        if spec is None:
            return True
        art_path = self._artifact_path(cache_path)
        if not art_path.exists():
            return False
        import tarfile
        by_arc = dict(spec)
        try:
            with tarfile.open(art_path, "r:gz") as tar:
                _members = tar.getmembers()
                # Replace only the entries THIS tar carries, instead of
                # rmtree-ing each destination root. A destination can be shared
                # with sibling rollouts (e.g. `_snapshots/{task_id}` holds one
                # dir per sample), and wiping the root deleted their artifacts:
                # restoring sample 1 destroyed sample 0's snapshot+DIFF, which
                # is how task #108's "last restored sample wins" happened.
                # Verified live 2026-07-24 on an N=3 run.
                _to_clear: set[Path] = set()
                for _m in _members:
                    if _m.name.startswith("/") or ".." in _m.name.split("/"):
                        continue
                    _top, _, _rel = _m.name.partition("/")
                    _root = by_arc.get(_top)
                    if _root is None:
                        continue
                    # first path component under the destination root
                    _first = _rel.split("/")[0] if _rel else ""
                    _to_clear.add(_root / _first if _first else _root)
                for _d in _to_clear:
                    if _d.is_dir():
                        shutil.rmtree(_d)
                    elif _d.exists():
                        _d.unlink()
                for member in _members:
                    if member.name.startswith("/") or ".." in member.name.split("/"):
                        continue
                    top, _, rel = member.name.partition("/")
                    target_root = by_arc.get(top)
                    if target_root is None:
                        continue
                    dest = target_root / rel if rel else target_root
                    if member.isdir():
                        dest.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        src = tar.extractfile(member)
                        if src is not None:
                            with src, open(dest, "wb") as out:
                                shutil.copyfileobj(src, out)
        except (OSError, tarfile.TarError) as e:
            logger.warning(f"Artifact restore failed for {cache_path.name}: {e}")
            return False
        return True

    def _get_cache_dir_for_tree(self, tree_hash: str) -> Path:
        """Get the cache directory for a specific tree hash."""
        return self.config.cache_dir / tree_hash[: self.config.hash_length]

    def _get_cache_path(self, tree_hash: str, question: str, sdk: str, model: str, effort: str | None = None, sample_idx: int | None = None) -> Path:
        """Get the cache file path for a tree hash + question + sdk + model + effort + sample.

        `effort` is appended to the keyed string only when non-empty so legacy
        callers that don't pass it keep the same cache paths as before.
        `sample_idx` is appended for best-of-N caching (None = legacy single-trace).
        """
        tree_dir = self._get_cache_dir_for_tree(tree_hash)
        keyed = f"{question.strip()}|sdk={sdk}|model={model}"
        if effort:
            keyed += f"|effort={effort}"
        if sample_idx is not None:
            keyed += f"|sample={sample_idx}"
        question_hash = hashlib.sha256(keyed.encode("utf-8")).hexdigest()[: self.config.hash_length]
        return tree_dir / f"{question_hash}.json"

    def _render_transcript_for_cache(self, trace: AgentTrace[Any]) -> str | None:
        """Pre-render the turn-by-turn transcript before `messages` is stripped.

        SDK message objects (AssistantMessage, ToolUseBlock, …) don't
        round-trip through Pydantic JSON serialization, so the raw `messages`
        list is structurally useless after a cache read regardless of
        `store_messages`. We render here while the SDK objects are still in
        memory; `AgentTrace.summarize()` uses the pre-rendered string on the
        read side, which is what lets the loop's failure-trace files contain
        the solver's actual tool-use history instead of metadata and an empty
        Full Result.

        THIS IS ONE METHOD ON PURPOSE. The same render-then-cap logic used to
        be copy-pasted into `set` and `set_n` with the caps written as literals
        in both places. That is precisely the shape of defect #147 — two
        independent layers carrying the same magic number, so fixing the one
        you found leaves the symptom exactly where it was. Any future change to
        the transcript budget now has one site to change.

        TRUNCATION KEEPS THE TAIL (#149). It used to keep only the head, so an
        over-budget transcript lost its ENDING — the commit, the termination,
        the outcome — which is the half a failure analysis actually needs. On
        tau3 that silently cost 26 of 55 stored rollouts a median 40% of their
        turns, worst case 79 of 86. The elision marker names the missing TURN
        RANGE and not just a character count, because a reader who sees
        "### Turn 7" followed by "### Turn 74" and no explanation will read it
        as an episode that had 8 turns.
        """
        try:
            from src.harness.agent import _render_turn_transcript
            rendered = _render_turn_transcript(
                trace.messages, self.config.transcript_tool_result_max_chars)
        except Exception:  # noqa: BLE001 — a render failure must never lose a cache write
            return None
        if not rendered:
            return None

        cap = self.config.transcript_max_chars
        if cap <= 0 or len(rendered) <= cap:
            return rendered

        # 60/40 split: the head carries the task setup and the opening
        # trajectory, the tail carries the outcome. Both are load-bearing.
        head_n = int(cap * 0.6)
        tail_n = cap - head_n
        head, tail = rendered[:head_n], rendered[-tail_n:]
        dropped = len(rendered) - cap

        _turns = re.findall(r"(?m)^### Turn (\d+)", head)
        last_head = int(_turns[-1]) if _turns else None
        _turns = re.findall(r"(?m)^### Turn (\d+)", tail)
        first_tail = int(_turns[0]) if _turns else None
        if last_head is not None and first_tail is not None and first_tail > last_head + 1:
            span = (f"turns {last_head + 1}-{first_tail - 1} are NOT SHOWN "
                    f"(the episode did run them)")
        else:
            span = "the middle of the episode is NOT SHOWN"
        marker = (f"\n\n...[TRANSCRIPT ELIDED IN THE MIDDLE: {dropped:,} of "
                  f"{len(rendered):,} chars removed; {span}. The head and the "
                  f"TAIL below are contiguous with the real episode; the gap is "
                  f"a storage cap, not a jump in the conversation.]...\n\n")
        return head + marker + tail

    def get(
        self,
        question: str,
        response_model: type[T] | None = None,
        *,
        sdk: str,
        model: str,
        system_prompt: str | None = None,
        effort: str | None = None,
    ) -> AgentTrace[T] | None:
        """
        Retrieve cached trace for current state + question + sdk + model + effort.

        Args:
            question: The question/query string
            response_model: Optional Pydantic model for output validation
            sdk: Active harness SDK (e.g. "claude", "opencode") — part of cache key
            model: Model identifier — part of cache key
            system_prompt: The exact system prompt the agent will receive.
                When provided, the cache key reflects this string; when
                omitted, the key falls back to hashing on-disk prompt
                files (weaker — see _get_tree_hash docstring).
            effort: Thinking-effort knob ("low"|"medium"|"high"|"max"). When
                non-empty, appended to the cache path key so runs at
                different effort levels don't collide. EvoSkill experiments
                run with `thinking={"type":"adaptive"}` so the discriminating
                signal is `effort` alone.

        Returns:
            AgentTrace if cache hit, None if miss or disabled
        """
        if not self.config.enabled:
            return None

        try:
            tree_hash = self._get_tree_hash(system_prompt=system_prompt)
        except RuntimeError as e:
            logger.debug(f"Cache get failed to compute tree hash: {e}")
            return None

        cache_path = self._get_cache_path(tree_hash, question, sdk, model, effort=effort)
        _hit_sample: int | None = None  # which sample's artifacts to restore

        if not cache_path.exists():
            # Fall back to sample_idx=0 from N>1 runs — the first sample
            # is functionally identical to an N=1 run.
            cache_path = self._get_cache_path(tree_hash, question, sdk, model, effort=effort, sample_idx=0)
            _hit_sample = 0
            if not cache_path.exists():
                return None

        try:
            with open(cache_path, "r") as f:
                entry_data = json.load(f)

            entry = CacheEntry.model_validate(entry_data)

            # Reconstruct AgentTrace from cached data
            trace_data = entry.trace

            # WALL-LIMITED ENTRIES ARE REFUSED (opt-in), mirroring the skip in
            # `set`. The wall is OUR configuration, not a property of the task,
            # and it is NOT in the cache key (the key is tree_hash | question |
            # sdk | model | effort) — so a record produced under one wall
            # replays silently into a leg running a different one. That is
            # exactly what happened to the tau3 smoke on 2026-08-06: the #175
            # wall fix landed at 10:20 between legs, and the v2 leg's own retry
            # at 10:23 replayed two 09:29/09:33 timeouts created by the OLD,
            # tighter wall — so half a val gate was measured on one instrument
            # and half on another. Refusing the hit re-executes under the
            # current wall, which is the only way a corrected wall can heal a
            # record the old one killed.
            _wall_reasons = _wall_skip_reasons()
            if _wall_reasons:
                _r = _termination_reason(trace_data.get("output"))
                if _r and _r.strip().lower() in _wall_reasons:
                    logger.warning(
                        f"RunCache.get REFUSE (wall-limited: termination_reason="
                        f"{_r!r}) — re-executing under the CURRENT wall rather "
                        f"than replaying an outcome the old wall decided: "
                        f"{cache_path.name}")
                    return None

            # NOTE: we deliberately RETURN cached entries whose output is
            # None (i.e. the original run timed out / crashed). The cache
            # key already encodes (Q, sdk, model, effort, system_prompt,
            # skill_state) — so an exact match means the same agent in
            # the same configuration will produce the same outcome
            # (modulo LLM stochasticity). Re-executing wastes API cost
            # for an outcome that won't differ. When the skill set
            # changes (e.g., the evolver writes a new skill), the
            # tree_hash changes, the cache misses, and the agent pays
            # for a fresh attempt — which is the right time to retry.
            # An earlier guard here treated null-output entries as miss,
            # but that conflated "stale entries from a different
            # tree_hash" (the actual bug, now fixed by the skills-tree-
            # only `_git_behavior_token`) with "legitimate failures
            # under THIS exact config."

            # Reconstruct output if response_model provided
            if response_model and trace_data.get("output"):
                trace_data["output"] = response_model.model_validate(trace_data["output"])

            # Restore file artifacts alongside the trace; an artifact-aware
            # entry with no tar predates capture → treat as a miss so the
            # rollout re-executes and records artifacts this time.
            if not self._restore_artifacts(cache_path, question, _hit_sample):
                logger.warning(
                    f"RunCache.get: entry {cache_path.name} is artifact-aware "
                    f"but artifact restore failed/missing — treating hit as "
                    f"MISS (rollout will re-execute)")
                return None
            self._run_on_replay(question, _hit_sample)

            # Mark replayed traces so the runner's cost accountant can
            # split paid (fresh inference) from replayed (cache) spend.
            trace_data["from_cache"] = True
            return AgentTrace.model_validate(trace_data)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Corrupted/unvalidatable entry — QUARANTINE, never delete.
            # ValueError covers pydantic ValidationError: a schema change in
            # a NEWER reader must not destroy a paid trace written by an
            # older run (the 2026-07-22 nex evo cache lost 14 entries with
            # zero log lines; this unlink was the only in-code deleter of
            # cache JSONs). The .corrupt file preserves the evidence.
            logger.warning(
                f"RunCache.get: unvalidatable entry {cache_path.name} "
                f"({type(e).__name__}: {str(e)[:200]}) — quarantining as "
                f".json.corrupt, treating as MISS")
            try:
                cache_path.rename(cache_path.with_suffix(".json.corrupt"))
            except OSError:
                pass
            return None

    def set(
        self,
        question: str,
        trace: AgentTrace[Any],
        *,
        sdk: str,
        model: str,
        system_prompt: str | None = None,
        effort: str | None = None,
        sample_idx: int | None = None,
    ) -> bool:
        """
        Cache a trace for current state + question + sdk + model + effort.

        Args:
            question: The question/query string
            trace: The AgentTrace to cache
            sdk: Active harness SDK (e.g. "claude", "opencode") — part of cache key
            model: Model identifier — part of cache key
            system_prompt: Same as get(); MUST match what get() will pass
                so a write here is reachable by a later read. When omitted,
                falls back to on-disk prompt files (weaker key).
            effort: Same as get(); appended to the cache path key when non-empty.

        Returns:
            True when the entry landed on disk, False on ANY skip or failure.
            Every False path logs a WARNING naming the reason — a silent
            no-write cost the 2026-07-22 nex evo run 71 of 117 rollouts
            (~60% of the base eval re-executed on resume).
        """
        _qid = f"{question.strip()[:40]!r}…" if len(question.strip()) > 40 else repr(question.strip())
        if not self.config.enabled:
            logger.warning(f"RunCache.set SKIP (cache disabled): {_qid}")
            return False

        # ── WALL-LIMITED OUTCOMES: opt-in refusal (#152, 2026-08-06) ─────────────
        # The blanket claim just below — "genuine failures (timeout, parse error) ARE
        # cached, they'll recur with the same skill+question" — is true for a parse error
        # and FALSE for a wall-clock timeout. Whether an episode exceeds the wall depends
        # on host load and provider latency, not on skill+question, so the outcome is
        # stochastic and environment-dependent. Freezing it has two costs:
        #   1. a transiently slow episode becomes permanently 0.0 for that cache key, and
        #      a relaunch REPLAYS the 0.0 instead of retrying it;
        #   2. worse for comparability — base and skill legs run at different times under
        #      different load, so whichever leg met heavier load carries more frozen
        #      zeroes, biasing the comparison in a direction nobody chose.
        # Measured 2026-08-06 across the tau3 banking cache dirs: 163 records, of which
        # **13 carry termination_reason "timeout"** and are cached. Note also that a tau2
        # inner-wall timeout has `output != None`, so it never reaches the transient-failure
        # branch below at all.
        #
        # The tau2/tau3 `episode_timeout` is OUR config, not the benchmark's, which is what
        # makes this ours to refuse. The SCORE stays 0.0 — that is tau2's own rule
        # (evaluator.py:119-121 returns reward 0.0 for any termination outside
        # {AGENT_STOP, USER_STOP}, and agent_metrics.py excludes only INFRASTRUCTURE_ERROR
        # from the denominator). Changing the score would diverge from the benchmark and
        # INFLATE our numbers; only the CACHEABILITY is wrong.
        #
        # OPT-IN, and deliberately so: this cache layer is shared by six harnesses and the
        # current behaviour was a considered cost decision. Flipping the default would
        # silently change replay cost for every one of them. Enable per run.
        _reasons = _wall_skip_reasons()
        if _reasons:
            _reason = _termination_reason(getattr(trace, "output", None))
            if _reason and str(_reason).strip().lower() in _reasons:
                logger.warning(
                    f"RunCache.set SKIP (wall-limited: termination_reason="
                    f"{_reason!r}) — the wall is OUR config and the outcome is "
                    f"load-dependent, so caching it would freeze a stochastic 0.0 "
                    f"and bias legs that met heavier load: {_qid}")
                return False

        # Skip caching traces from transient failures only. Genuine
        # failures (parse error) ARE cached — they'll recur
        # with the same skill+question, so retrying wastes money.
        # (Wall-clock timeouts are handled above and are NOT in that category.)
        if trace.output is None:
            result_str = getattr(trace, "result", "") or ""
            is_timeout = "[TIMEOUT]" in result_str or "[WALL_TIMEOUT]" in result_str
            if is_timeout:
                pass  # cache timeouts — they'll recur with the same skill+question
            else:
                turns = getattr(trace, "num_turns", 0) or 0
                if turns <= 1:
                    logger.warning(
                        f"RunCache.set SKIP (output=None, num_turns={turns} <= 1 — "
                        f"likely transient startup failure): {_qid}")
                    return False
            # Rate limit: repeated "You've hit your limit · resets <time>"
            import re as _re
            transcript = getattr(trace, "cached_transcript", "") or ""
            if not transcript:
                msgs = getattr(trace, "messages", []) or []
                transcript = " ".join(str(m) for m in msgs[-10:])
            rate_limit_hits = len(_re.findall(
                r"You've hit your limit \xb7 resets \d+:\d+\w*\s*\([^)]+\)",
                transcript,
            ))
            if rate_limit_hits >= 3:
                logger.warning(
                    f"RunCache.set SKIP (rate-limit spam x{rate_limit_hits} in transcript): {_qid}")
                return False

        try:
            tree_hash = self._get_tree_hash(system_prompt=system_prompt)
        except RuntimeError as e:
            logger.warning(f"RunCache.set SKIP (tree hash failed: {e}): {_qid}")
            return False

        tree_dir = self._get_cache_dir_for_tree(tree_hash)
        tree_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self._get_cache_path(tree_hash, question, sdk, model, effort=effort, sample_idx=sample_idx)

        # Serialize trace
        trace_dict = trace.model_dump()

        # Stored traces are always written from fresh inference. Force the
        # flag false in the JSON so the next get() can authoritatively flip
        # it true at read time.
        trace_dict["from_cache"] = False

        # Pre-render the turn-by-turn transcript before stripping messages.
        # See `_render_transcript_for_cache` for why, and for the two defects
        # (#147-shaped duplication, #149 tail loss) that shaped it.
        if trace_dict.get("cached_transcript") is None:
            rendered = self._render_transcript_for_cache(trace)
            if rendered:
                trace_dict["cached_transcript"] = rendered

        # Optionally strip messages to save space
        if not self.config.store_messages:
            trace_dict["messages"] = []

        # Serialize output if it's a Pydantic model
        if trace.output is not None:
            trace_dict["output"] = trace.output.model_dump()

        keyed_for_record = f"{question.strip()}|sdk={sdk}|model={model}"
        if effort:
            keyed_for_record += f"|effort={effort}"
        entry = CacheEntry(
            version="1.0",
            created_at=datetime.now().isoformat(),
            cache_key={
                "tree_hash": tree_hash[: self.config.hash_length],
                "question_hash": hashlib.sha256(
                    keyed_for_record.encode("utf-8")
                ).hexdigest()[: self.config.hash_length],
                "question": question,
                "sdk": sdk,
                "model": model,
                "effort": effort or "",
            },
            trace=trace_dict,
        )

        # Write atomically (write to temp, then rename). The temp filename
        # includes pid+a random token, not just cache_path's own name --
        # confirmed as an active bug: two concurrent writers targeting the
        # SAME cache key (duplicate scheduling, a retry racing its
        # original, or genuinely concurrent processes sharing this cache
        # dir, all explicitly-supported usage per this module's own
        # docstrings) previously shared one temp file, so their writes
        # could interleave in the same inode before either renamed,
        # producing a torn "atomic" entry.
        # tree_hash is content-addressed on skill-tree + system-prompt state
        # only (see _get_tree_hash docstring) -- it does NOT encode model
        # identity. Two unrelated experiments that happen to share the same
        # skill-tree state (e.g. both v0/no-skill) land in the SAME tree_dir.
        # This is safe for get()/set() themselves (question_hash below already
        # incorporates model/sdk/effort, so lookups never cross-match), but a
        # tree_dir with mixed models is a trap for any external tool that
        # scans/matches cache files directly instead of going through this
        # class -- confirmed in production (2026-07-07): a v0 tree_hash bucket
        # accumulated 140 DSV4F + 140 GLM-5.2 entries, and an ad-hoc external
        # resolution script that matched on (num_turns, total_cost_usd) alone
        # only avoided a false match because the two models' pricing happened
        # to differ by 6-13x -- not a structural guarantee. Surface the mix at
        # write time so it's visible in logs instead of requiring a full audit
        # to discover.
        try:
            existing_models = {
                json.loads(p.read_text(encoding="utf-8")).get("cache_key", {}).get("model")
                for p in tree_dir.glob("*.json")
            }
            existing_models.discard(model)
            existing_models.discard(None)
            if existing_models:
                logger.warning(
                    f"RunCache: tree_dir {tree_dir.name} already has entries from "
                    f"other model(s) {sorted(existing_models)} -- writing {model!r} "
                    f"into the same bucket. Use list_entries(tree_hash, model=...) to "
                    f"enumerate entries for a specific model rather than scanning "
                    f"tree_dir files directly."
                )
        except (OSError, json.JSONDecodeError):
            pass  # best-effort visibility check, never blocks a real cache write

        temp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(entry.model_dump(), f, indent=2, default=str)
            temp_path.rename(cache_path)
        except OSError as e:
            # Previously unguarded -- confirmed as an active bug: an
            # uncaught exception here (disk-full, permission error) propagates
            # to the caller's outer handler, which on at least one call site
            # (evaluate.py's primary success path) discards the just-completed,
            # real agent trace and replaces it with a synthetic [ERROR] trace
            # over what should have been a skippable cache-write failure.
            logger.warning(f"Cache write failed for {cache_path.name}: {e}")
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

        # Post-write verification: the rename must have landed. A missing
        # file here means an external deleter or a filesystem anomaly —
        # either way the caller must know the entry is NOT persisted.
        if not cache_path.exists():
            logger.error(
                f"RunCache.set WROTE-BUT-VANISHED: {cache_path.name} does not "
                f"exist immediately after rename ({_qid})")
            return False

        # Capture file artifacts produced by this rollout (no-op unless
        # config.artifact_resolver is set and matches this question).
        self._capture_artifacts(cache_path, question, sample_idx)
        return True

    def get_n(
        self,
        question: str,
        response_model: type[T] | None = None,
        *,
        n: int,
        sdk: str,
        model: str,
        system_prompt: str | None = None,
        effort: str | None = None,
    ) -> list[AgentTrace[T] | None] | None:
        """Retrieve up to N cached traces for a question.

        Returns a list of length ``n``, positionally aligned with sample_idx —
        index i is either the cached trace for sample_idx=i, or None if that
        sample isn't cached. Returning a dense (gap-removed) list here used to
        let callers re-derive sample_idx via enumerate(), which silently wrote
        a freshly-computed trace's score into a DIFFERENT sample_idx's cache
        slot whenever a cache hit wasn't a contiguous prefix (bug H19).
        Returns None only if caching is disabled or no samples are cached at
        all. The caller should run a fresh sample for every None slot.
        """
        if not self.config.enabled or n <= 0:
            return None
        try:
            tree_hash = self._get_tree_hash(system_prompt=system_prompt)
        except RuntimeError:
            return None
        traces: list[AgentTrace[T] | None] = [None] * n
        any_hit = False
        for i in range(n):
            cache_path = self._get_cache_path(tree_hash, question, sdk, model, effort=effort, sample_idx=i)
            if not cache_path.exists():
                continue
            try:
                with open(cache_path, "r") as f:
                    entry_data = json.load(f)
                entry = CacheEntry.model_validate(entry_data)
                trace_data = entry.trace
                if response_model and trace_data.get("output"):
                    trace_data["output"] = response_model.model_validate(trace_data["output"])
                # Same artifact-or-miss semantics as get(). Each sample's
                # artifacts are captured/restored under its OWN per-sample
                # workspace and only the dirs present in its tar are replaced,
                # so samples no longer clobber each other (was task #108).
                if not self._restore_artifacts(cache_path, question, i):
                    continue
                self._run_on_replay(question, i)
                trace_data["from_cache"] = True
                traces[i] = AgentTrace.model_validate(trace_data)
                any_hit = True
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Quarantine, never delete — same rationale as get().
                logger.warning(
                    f"RunCache.get_n: unvalidatable entry {cache_path.name} "
                    f"({type(e).__name__}: {str(e)[:200]}) — quarantining as "
                    f".json.corrupt, treating as MISS")
                try:
                    cache_path.rename(cache_path.with_suffix(".json.corrupt"))
                except OSError:
                    pass
                continue
        return traces if any_hit else None

    def set_n(
        self,
        question: str,
        traces: list[AgentTrace[Any]],
        *,
        sdk: str,
        model: str,
        system_prompt: str | None = None,
        effort: str | None = None,
    ) -> None:
        """Cache N traces for a question (one per sample index)."""
        if not self.config.enabled:
            return
        try:
            tree_hash = self._get_tree_hash(system_prompt=system_prompt)
        except RuntimeError:
            return
        for i, trace in enumerate(traces):
            cache_path = self._get_cache_path(tree_hash, question, sdk, model, effort=effort, sample_idx=i)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            trace_dict = trace.model_dump()
            trace_dict["from_cache"] = False
            if trace_dict.get("cached_transcript") is None:
                rendered = self._render_transcript_for_cache(trace)
                if rendered:
                    trace_dict["cached_transcript"] = rendered
            if not self.config.store_messages:
                trace_dict["messages"] = []
            if trace.output is not None:
                trace_dict["output"] = trace.output.model_dump()
            keyed = f"{question.strip()}|sdk={sdk}|model={model}"
            if effort:
                keyed += f"|effort={effort}"
            keyed += f"|sample={i}"
            entry = CacheEntry(
                version="1.0",
                created_at=datetime.now().isoformat(),
                cache_key={
                    "tree_hash": tree_hash[: self.config.hash_length],
                    "question_hash": hashlib.sha256(keyed.encode("utf-8")).hexdigest()[: self.config.hash_length],
                    "question": question,
                    "sdk": sdk,
                    "model": model,
                    "effort": effort or "",
                    "sample_idx": str(i),
                },
                trace=trace_dict,
            )
            temp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                with open(temp_path, "w") as f:
                    json.dump(entry.model_dump(), f, indent=2, default=str)
                temp_path.rename(cache_path)
            except OSError as e:
                logger.warning(f"Cache write failed for {cache_path.name}: {e}")
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            self._capture_artifacts(cache_path, question, i)

    def list_entries(
        self,
        tree_hash: str,
        *,
        model: str | None = None,
        sdk: str | None = None,
        effort: str | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate cache entries in a tree_hash's directory, optionally
        filtered by model/sdk/effort.

        A tree_hash directory can legitimately contain entries from multiple
        unrelated experiments (different model/sdk/effort) that happen to
        share the same skill-tree + system-prompt state -- e.g. two different
        solver models both evaluated against an empty ("v0") skill tree.
        get()/set() already handle this correctly (question_hash incorporates
        model/sdk/effort, so lookups never cross-match), but any external code
        that needs to enumerate "all entries for tree X" must filter by these
        fields explicitly -- scanning tree_dir files and matching on other
        signals (e.g. num_turns/cost) is fragile and has produced a real,
        confirmed false-match risk in production. Use this method instead.

        Returns a list of dicts with keys: path (Path), cache_key (dict),
        created_at (str), trace (the raw trace dict, NOT reconstructed into
        an AgentTrace object -- caller's responsibility if needed).
        """
        tree_dir = self._get_cache_dir_for_tree(tree_hash)
        if not tree_dir.exists():
            return []
        out: list[dict[str, Any]] = []
        for p in sorted(tree_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            ck = data.get("cache_key", {})
            if model is not None and ck.get("model") != model:
                continue
            if sdk is not None and ck.get("sdk") != sdk:
                continue
            if effort is not None and ck.get("effort", "") != effort:
                continue
            out.append({
                "path": p,
                "cache_key": ck,
                "created_at": data.get("created_at"),
                "trace": data.get("trace"),
            })
        return out

    def clear(self, tree_hash: str | None = None) -> int:
        """
        Clear cached entries.

        Args:
            tree_hash: If provided, only clear entries for this tree hash.
                      If None, clear all cached entries.

        Returns:
            Number of entries deleted
        """
        if not self.config.enabled:
            return 0

        count = 0

        if tree_hash is not None:
            # Clear specific tree's cache
            tree_dir = self._get_cache_dir_for_tree(tree_hash)
            if tree_dir.exists():
                count = len(list(tree_dir.glob("*.json")))
                shutil.rmtree(tree_dir)
        else:
            # Clear all caches
            if self.config.cache_dir.exists():
                for tree_dir in self.config.cache_dir.iterdir():
                    if tree_dir.is_dir():
                        count += len(list(tree_dir.glob("*.json")))
                shutil.rmtree(self.config.cache_dir)
                self._ensure_cache_dir()

        return count

    def update_inline_score(
        self,
        question: str,
        score: float,
        *,
        sdk: str,
        model: str,
        system_prompt: str | None = None,
        effort: str | None = None,
        sample_idx: int | None = None,
    ) -> bool:
        """Patch inline_score into an existing cache entry without rewriting the full trace."""
        if not self.config.enabled:
            return False
        try:
            tree_hash = self._get_tree_hash(system_prompt=system_prompt)
        except Exception:
            return False
        cache_path = self._get_cache_path(tree_hash, question, sdk, model, effort=effort, sample_idx=sample_idx)
        if not cache_path.exists():
            return False
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("trace", {}).get("inline_score") is not None:
                return False
            data["trace"]["inline_score"] = score
            cache_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            return True
        except Exception:
            return False

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (total entries, size, programs)
        """
        if not self.config.cache_dir.exists():
            return {"total_entries": 0, "total_size_bytes": 0, "programs": 0}

        total_entries = 0
        total_size = 0
        programs = 0

        for tree_dir in self.config.cache_dir.iterdir():
            if tree_dir.is_dir():
                programs += 1
                for cache_file in tree_dir.glob("*.json"):
                    total_entries += 1
                    total_size += cache_file.stat().st_size

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "programs": programs,
        }
