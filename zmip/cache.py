"""Private resume receipts; public CSV, JSON and H5AD contracts stay unchanged."""

import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from .runtime import runtime_identity


@contextmanager
def lock_run(outdir):
    """Allow one parent writer per output directory; children inherit no lock."""
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".zmip.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another zmip run is writing to {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def file_digest(path):
    """Hash incrementally so large H5ADs do not need a second in-memory copy."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def prepare_run(outdir, input_path, options, force=False, agent=None):
    """Reject unverifiable reuse; force starts a new generation before any work.

    `agent` (model, effort, turn budget, harness, endpoint hashes) is recorded for
    the audit trail and refreshed on every run, but never compared: it steers how
    decisions are produced without changing what a finished stage means."""
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"input_sha256": file_digest(input_path), "options": options, "runtime": runtime_identity()}
    # Normalize tuples to JSON lists before comparing with a previous receipt.
    identity = json.loads(json.dumps(identity, sort_keys=True))
    path = root / ".zmip-run.json"
    if not force:
        if path.exists():
            try:
                previous = json.loads(path.read_text())
            except (ValueError, OSError) as exc:
                raise ValueError("invalid resume record; use a new output directory or --force") from exc
            if not isinstance(previous, dict) or previous.get("identity") != identity or not previous.get("run_id"):
                raise ValueError(
                    "input or options changed, or runtime/code changed; use a new output directory or --force"
                )
            if agent is not None and previous.get("agent") != agent:
                write_json(path, {**previous, "agent": agent})
            return previous["run_id"]
        if any(root.glob("*.h5ad")) or (root / "zmip_plan.json").exists() or any(root.glob("*/annotated.h5ad")):
            raise ValueError("legacy outputs have no resume record; use a new output directory or --force")
    run_id = uuid.uuid4().hex
    write_json(path, {"identity": identity, "run_id": run_id, "agent": agent})
    return run_id


def run_id(outdir):
    path = Path(outdir) / ".zmip-run.json"
    return json.loads(path.read_text())["run_id"] if path.exists() else None


def invalidate(outdir, stage):
    (Path(outdir) / f".zmip-{stage}.json").unlink(missing_ok=True)


def seal(outdir, stage, generation, files):
    """Publish completion only after all files have been successfully written."""
    write_json(
        Path(outdir) / f".zmip-{stage}.json",
        {
            "run_id": generation,
            "files": {name: file_digest(Path(outdir) / name) for name in files},
        },
    )


def valid(outdir, stage, generation, files):
    try:
        receipt = json.loads((Path(outdir) / f".zmip-{stage}.json").read_text())
        return (
            receipt["run_id"] == generation
            and set(receipt["files"]) == set(files)
            and all(file_digest(Path(outdir) / name) == receipt["files"][name] for name in files)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
