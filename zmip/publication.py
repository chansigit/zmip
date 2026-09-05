"""Recoverable publication of a complete global data/report set.

Files retain their public paths. A journal guards the short multi-file replace
window; the completion receipt is replaced last. Recovery restores the old set
from immutable backups before another writer or report reader proceeds.
"""

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

from . import cache

log = logging.getLogger(__name__)

RECEIPT = ".zmip-global.json"
JOURNAL = ".zmip-publish.json"
DATA_FILES = ("annotated_zmip.h5ad", "zmip_removed.csv", "zmip_reassigned.csv")


def generation(outdir):
    root = Path(outdir)
    dependencies = [root / "zmip_plan.json", root / "lineage_markers.csv"]
    plan_path = root / "zmip_plan.json"
    if plan_path.exists():
        from .lineage import lineage_dir

        plan = json.loads(plan_path.read_text())
        dependencies += [Path(lineage_dir(root, ln["name"])) / ".zmip-complete.json"
                         for ln in plan["lineages"] if ln["zoom"]]
    return {"run_id": cache.run_id(root), "dependencies": {
        str(p.relative_to(root)): cache.file_digest(p) for p in dependencies if p.exists()}}


def complete(outdir, *, check_report=True):
    root = Path(outdir)
    if (root / JOURNAL).exists():
        return False
    try:
        receipt = json.loads((root / RECEIPT).read_text())
        files = receipt["files"]
        return (receipt["run_id"] == generation(root) and set(DATA_FILES).issubset(files)
                and all(cache.file_digest(root / name) == digest for name, digest in files.items()
                        if check_report or name != "report.html"))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def require_complete(outdir):
    if not complete(outdir, check_report=False):
        raise RuntimeError("global output is incomplete, changed or belongs to an earlier run; "
                           "rerun zmip to recover/merge before rebuilding its report")


def _copy_snapshot(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def recover(outdir):
    """Idempotent rollback, including a process killed during an earlier rollback."""
    root = Path(outdir)
    journal_path = root / JOURNAL
    if not journal_path.exists():
        return
    journal = json.loads(journal_path.read_text())
    transaction = root / journal["transaction"]
    # Journal paths are host generated. Still reject malformed traversal on recovery.
    if transaction.parent != root / ".zmip-publish":
        raise ValueError("invalid publication transaction path")
    for name, had_previous in journal["previous"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid publication output path")
        target = root / relative
        if had_previous:
            backup = transaction / "backup" / relative
            temporary = target.with_name(target.name + ".recover")
            temporary.unlink(missing_ok=True)
            _copy_snapshot(backup, temporary)
            os.replace(temporary, target)
        else:
            target.unlink(missing_ok=True)
    journal_path.unlink()
    shutil.rmtree(transaction)
    log.warning("== restored previous global output after interrupted publication")


@contextmanager
def staging(outdir):
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    recover(root)
    parent = root / ".zmip-publish"
    parent.mkdir(exist_ok=True)
    transaction = Path(tempfile.mkdtemp(prefix="merge-", dir=parent))
    stage = transaction / "new"
    stage.mkdir()
    try:
        yield stage
    finally:
        # A surviving journal owns its backups until successful recovery.
        if not (root / JOURNAL).exists() and transaction.exists():
            shutil.rmtree(transaction)


def publish(outdir, stage):
    root, stage = Path(outdir), Path(stage)
    for name in DATA_FILES:
        if not (stage / name).is_file():
            raise ValueError(f"global output missing {name}")
    files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    cache.seal(stage, "global", generation(root), files)
    # Include obsolete global figures so a previous run cannot leak into the new set.
    old = {p.relative_to(root).as_posix() for p in (root / "figures").glob("zmip_*.png")}
    old |= {name for name in (*DATA_FILES, "zmip_fine_legend.csv", "report.html", RECEIPT)
            if (root / name).exists()}
    names = sorted(set(files) | old | {RECEIPT})
    transaction = stage.parent
    previous = {name: (root / name).exists() for name in names}
    for name in names:
        if previous[name]:
            _copy_snapshot(root / name, transaction / "backup" / name)
    cache.write_json(root / JOURNAL, {"transaction": transaction.relative_to(root).as_posix(),
                                     "previous": previous})
    try:
        for name in [n for n in names if n != RECEIPT] + [RECEIPT]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if (stage / name).exists():
                os.replace(stage / name, target)
            else:
                target.unlink(missing_ok=True)
        (root / JOURNAL).unlink()
    except BaseException:
        recover(root)
        raise


def refresh_report_receipt(outdir, out_html):
    root = Path(outdir)
    if Path(out_html).resolve() != (root / "report.html").resolve():
        return
    receipt = json.loads((root / RECEIPT).read_text())
    receipt["files"]["report.html"] = cache.file_digest(out_html)
    cache.write_json(root / RECEIPT, receipt)
