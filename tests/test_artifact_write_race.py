"""Bug #116: the RunCache artifact writer must use a unique temp path.

The two JSON writers already wrote to `<stem>.<pid>.<uuid8>.tmp` and renamed
atomically. `_capture_artifacts` did not — it used a fixed `<name>.tmp`. Two
processes capturing the SAME cache entry therefore opened the identical path
with `tarfile.open(..., "w:gz")`, interleaved their gzip streams, and one
renamed a TORN tarball into place.

Nothing raises when that happens: tarfile.open succeeds, the rename succeeds,
and the corruption only surfaces later in `_restore_artifacts` — on a replay,
as wrong files scored as if they were real. That is why this blocked #115
(pre-roll the train base once and share the cache across arms): sharing one
cache dir across concurrent arms is precisely the triggering condition.
"""

from __future__ import annotations

import inspect
import re
import tarfile

from src.cache import run_cache as rc


SRC = inspect.getsource(rc.RunCache._capture_artifacts)


def test_temp_path_is_unique_per_writer():
    assert "os.getpid()" in SRC and "uuid.uuid4()" in SRC, (
        "the artifact temp name must be unique per writer (#116)"
    )
    assert not re.search(r'with_name\(\s*art_path\.name\s*\+\s*"\.tmp"\s*\)', SRC), (
        "the fixed `<name>.tmp` temp path is the race (#116)"
    )


def test_matches_the_pattern_the_json_writers_already_used():
    """Parity: one convention for every atomic write in this file."""
    whole = inspect.getsource(rc)
    uniq = re.findall(r'\{os\.getpid\(\)\}\.\{uuid\.uuid4\(\)\.hex\[:8\]\}\.tmp', whole)
    assert len(uniq) >= 3, (
        f"expected the unique-tmp pattern at all 3 atomic write sites "
        f"(2 JSON + 1 artifact), found {len(uniq)}"
    )


def test_rename_is_still_used_so_the_swap_stays_atomic():
    assert ".rename(art_path)" in SRC, (
        "a partial tar must never be visible under the real name"
    )


def test_tarerror_is_caught_not_just_oserror():
    """TarError is not an OSError — it used to escape and orphan the temp file."""
    assert "tarfile.TarError" in SRC, "tar-level failures must be handled (#116)"
    assert not re.search(r"except OSError as e:", SRC), (
        "the OSError-only handler let TarError propagate and leak the temp file"
    )
    assert issubclass(tarfile.TarError, Exception)
    assert not issubclass(tarfile.TarError, OSError), (
        "premise check: if TarError ever becomes an OSError this test is moot"
    )


def test_temp_file_is_removed_on_failure():
    assert "tmp.unlink(missing_ok=True)" in SRC, (
        "a failed capture must not leave a temp file behind"
    )


def test_two_writers_generate_different_temp_names():
    """The property the fix rests on, exercised rather than asserted."""
    import os
    import uuid as _uuid
    from pathlib import Path

    art = Path("/tmp/entry.tar.gz")
    names = {
        art.with_name(f"{art.name}.{os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp")
        for _ in range(200)
    }
    assert len(names) == 200, "temp names must not collide between writers"
