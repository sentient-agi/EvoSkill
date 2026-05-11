"""OfficeQA workspace setup: build a fresh PDF-only workspace for solver
and evolver agents.

The workspace gives both agents a uniform view of the corpus — PDFs at the
relative path `treasury_bulletin_pdfs/` — while keeping the actual files
unwritable from the agent's process. The symlinked target is a read-only
APFS clone (`chmod 555`); the user retains full edit access on the source
PDFs at their canonical location.

Call `build_pdf_only_workspace()` at the start of every run script. It is
idempotent: it refreshes the read-only clone from the source on each call,
so the agent always sees the latest user edits.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _refresh_pdf_ro_clone(pdf_source: Path, pdf_ro_clone: Path) -> None:
    """Mirror the source PDF dir into a read-only clone.

    First call uses APFS copy-on-write (`cp -c`) for an instant + zero-cost
    initial copy. Later calls lift the chmod, rsync --delete, then re-apply
    chmod 555. End state: clone matches source byte-for-byte and is
    unwritable.
    """
    pdf_source = pdf_source.resolve()

    if not pdf_source.is_dir():
        raise FileNotFoundError(f"PDF source dir not found: {pdf_source}")

    pdf_ro_clone.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_ro_clone.exists():
        # Initial clone via APFS copy-on-write — instant, zero extra disk
        # until either side mutates. We never mutate the clone (chmod 555
        # below), and the source is mutated only by the user, so the
        # clone-vs-source delta stays small.
        subprocess.run(
            ["cp", "-c", "-R", str(pdf_source) + "/", str(pdf_ro_clone) + "/"],
            check=True,
        )
    else:
        # Refresh path: lift read-only, rsync, re-apply read-only.
        subprocess.run(["chmod", "-R", "u+w", str(pdf_ro_clone)], check=True)
        subprocess.run(
            [
                "rsync", "-a", "--delete",
                str(pdf_source) + "/", str(pdf_ro_clone) + "/",
            ],
            check=True,
        )

    subprocess.run(["chmod", "-R", "555", str(pdf_ro_clone)], check=True)


def build_pdf_only_workspace(
    workspace_dir: Path,
    pdf_source: Path,
    pdf_ro_clone: Path,
) -> Path:
    """Build (or refresh) a PDF-only workspace at `workspace_dir`.

    Idempotent — safe to call before every run. On each call:
      - Refreshes `pdf_ro_clone` from `pdf_source` (APFS clone first time,
        rsync --delete subsequently).
      - Re-applies chmod 555 on the clone.
      - Re-creates the `<workspace>/treasury_bulletin_pdfs` symlink to point
        at the clone.
      - Ensures `.claude/skills/` and `.cache/scratch/` exist.

    Layout after the call:
        <workspace_dir>/
        ├── treasury_bulletin_pdfs/   → pdf_ro_clone  (read-only, 555)
        ├── .claude/skills/           (writable; evolver writes here)
        └── .cache/scratch/           (writable; intermediate work)

    Both solver and evolver should use this workspace as their cwd. Both
    reference PDFs via `treasury_bulletin_pdfs/...` — symmetric paths so
    an evolver replaying a solver trace doesn't have to translate.
    """
    _refresh_pdf_ro_clone(pdf_source, pdf_ro_clone)

    workspace_dir = workspace_dir.resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    pdf_link = workspace_dir / "treasury_bulletin_pdfs"
    if pdf_link.is_symlink() or pdf_link.exists():
        pdf_link.unlink()
    pdf_link.symlink_to(pdf_ro_clone.resolve())

    (workspace_dir / ".claude" / "skills").mkdir(parents=True, exist_ok=True)
    (workspace_dir / ".cache" / "scratch").mkdir(parents=True, exist_ok=True)

    return workspace_dir
