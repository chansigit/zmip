"""zmip.lineage — one zoom-in lineage end to end (subset → msp.integrate_adata
re-embedding → foreign-lineage scores → annotation agent → <lineage>/ files),
callable in-process or as its own subprocess so several lineages can run side
by side.

Lineages are independent by construction (each works on its own subset and
writes only its own directory; merge_back reads the results from disk), so
running them concurrently changes nothing but wall-clock: the biggest lineage
bounds the round instead of the sum. Concurrency is decided from what this
process may actually use — msp.resources (affinity mask + cgroup memory
limit, never the node's totals; degrades to os.cpu_count()/RAM outside
Slurm/containers) — plus a per-lineage memory estimate, so nothing needs to
be passed in from the job script:

  ZMIP_PARALLEL          hard cap on concurrent lineages (1 = sequential);
                          default min(#lineages, cpus // 2)
  ZMIP_MEM_PER_CELL_MB   RAM a lineage of N cells is assumed to need, per
                          cell (default 0.4 — a 59k-cell msp integrate peaked
                          near 22 GB); plus 0.5 GiB fixed per process

Each child gets OMP/BLAS/numba/MSP_MAX_THREADS = cpus // n_parallel so the
pool doesn't oversubscribe the allocation. Child stdout is streamed into the
parent's log line by line with a "[lineage]" prefix.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time

import pandas as pd
import scanpy as sc
from harness_bridge import configure_logging
from msp.integrate import integrate_adata
from msp.plots import save_single_umap
from msp.resources import available_cpus, available_memory_bytes, current_rss_bytes

from . import cache
from .annotate import BASE_KEY, PARENT_KEY, PREV_SUFFIX, PREVIOUS_COLS, annotate_lineage
from .cli import add_integration_options, parse_harmony
from .foreign import score_foreign
from .plan import _lineage_slugs

log = logging.getLogger(__name__)

SUBSET_FILE = "subset.h5ad"  # the parent's hand-off to a lineage subprocess; removed once loaded
FIXED_BYTES_PER_PROCESS = 512 << 20


def lineage_dir(outdir, name):
    directory = os.path.join(outdir, _lineage_slugs([name])[name])
    root = os.path.realpath(outdir)
    resolved = os.path.realpath(directory)
    if resolved == root or os.path.commonpath([root, resolved]) != root:
        raise ValueError(f"lineage directory escapes output directory: {directory!r}")
    return directory


CONTRACT_FILES = (
    "annotation_proposal.json",
    "annotated.h5ad",
    "report.html",
    "annotation_removed.csv",
    "annotation_reassigned.csv",
)


def _generation(outdir):
    # A lineage also depends on the exact plan and foreign-marker lists.
    return {
        "run_id": cache.run_id(outdir),
        "dependencies": {
            name: cache.file_digest(os.path.join(outdir, name))
            for name in ("zmip_plan.json", "lineage_markers.csv")
            if os.path.exists(os.path.join(outdir, name))
        },
    }


def contract_done(d):
    return cache.valid(d, "complete", _generation(os.path.dirname(d)), CONTRACT_FILES)


def load_result(d):
    # Converters preserve leading zeros and literal NA-like identifiers;
    # pandas still infers the removal flags as booleans.
    text_columns = dict.fromkeys(("cell", "lineage", "cluster", "reassign_to", "fine_label"), str)
    return {
        "dir": d,
        "removed": pd.read_csv(os.path.join(d, "annotation_removed.csv"), converters=text_columns),
        "reassigned": pd.read_csv(os.path.join(d, "annotation_reassigned.csv"), converters=text_columns),
    }


def subset_for(ad, labels, coarse_col, fine_col):
    """The lineage's cells with last round's labels carried as *_prev columns."""
    sub = ad[ad.obs[coarse_col].astype(str).isin(labels).values].copy()
    for c in PREVIOUS_COLS:
        src = {"msp_ann_coarse": coarse_col, "msp_ann_fine": fine_col}[c]
        sub.obs[c + PREV_SUFFIX] = sub.obs[src].astype(str).astype("category")
    return sub


def validate_resolutions(resolutions):
    """Require freshly computed base and parent clusterings before any work."""
    values = tuple(float(r) for r in resolutions)
    if not values or any(not math.isfinite(r) or r <= 0 for r in values):
        raise ValueError("--resolutions must contain finite, positive values")
    if len(set(values)) != len(values):
        raise ValueError("--resolutions must contain unique values")
    required = {float(key.rsplit("_r", 1)[1]) for key in (BASE_KEY, PARENT_KEY)}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(
            f"--resolutions must include {sorted(required)} for zoom-in annotation; "
            f"missing {missing}. Inherited clusterings cannot substitute for this round's results."
        )
    return values


def run_lineage(
    sub,
    name,
    labels,
    all_labels,
    markers,
    outdir,
    *,
    batch_col,
    species,
    h5ad_path,
    resolutions,
    n_top_genes,
    n_pcs,
    n_neighbors,
    harmony_kwargs,
    keys_for_foreign,
    language,
    model,
    effort,
    max_turns,
):
    """sub: the lineage subset from subset_for(). Writes <outdir>/<slug>/ and
    returns its result record."""
    resolutions = validate_resolutions(resolutions)
    d = lineage_dir(outdir, name)
    os.makedirs(d, exist_ok=True)
    cache.invalidate(d, "complete")
    generation = _generation(outdir)
    expected = sub.obs_names.copy()
    log.info(f"== [{name}] re-embedding {sub.n_obs} cells")
    integrate_adata(
        sub,
        batch_col,
        d,
        species=species,
        resolutions=tuple(resolutions),
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        harmony_kwargs=harmony_kwargs,
        inputs=[h5ad_path],
        meta_extra={"zmip_lineage": name, "zmip_coarse_labels": list(labels)},
    )
    figdir = os.path.join(d, "figures")
    log.info(f"== [{name}] foreign-lineage scores")
    foreign_cols = score_foreign(sub, markers, name, keys_for_foreign, d, figdir)
    for c in PREVIOUS_COLS:
        col = c + PREV_SUFFIX
        n = sub.obs[col].nunique()
        save_single_umap(
            sub,
            col,
            os.path.join(figdir, f"umap_{col}.png"),
            repel=True,
            repel_fontsize=8 if n > 15 else 11,
            figsize=(9, 9) if n > 15 else None,
        )
    annotate_lineage(
        sub,
        d,
        name,
        labels,
        sorted(set(all_labels) - set(labels)),
        foreign_cols,
        species=species,
        language=language,
        model=model,
        effort=effort,
        max_turns=max_turns,
    )
    # Validate the files that resume will read, not only the in-memory tables.
    from .merge import _validate_annotation, _validate_partition

    result = load_result(d)
    kept = sc.read_h5ad(os.path.join(d, "annotated.h5ad"), backed="r")
    try:
        _validate_partition(name, expected, kept.obs, result["removed"], result["reassigned"])
        _validate_annotation(name, kept.obs, result["reassigned"], labels, all_labels)
    finally:
        kept.file.close()
    cache.seal(d, "complete", generation, CONTRACT_FILES)
    return result


# ---------------------------------------------------------------- the pool


def _estimate_bytes(n_cells):
    per_cell = float(os.environ.get("ZMIP_MEM_PER_CELL_MB", "0.4")) * (1 << 20)
    return int(n_cells * per_cell) + FIXED_BYTES_PER_PROCESS


def plan_concurrency(todo):
    """(max_parallel, budget_bytes, threads_per_child) for the lineages in
    `todo` (dicts with n_cells), from the resources this process really has."""
    cpus = available_cpus()
    cap = os.environ.get("ZMIP_PARALLEL")
    if cap and cap.strip().isdigit() and int(cap) > 0:
        max_parallel = min(int(cap), len(todo))
    else:
        max_parallel = max(1, min(len(todo), cpus // 2))
    budget = int(available_memory_bytes() * 0.85) - current_rss_bytes()
    threads = max(1, cpus // max(1, max_parallel))
    return max_parallel, budget, threads


def _pump(proc, tag):
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if line.startswith("== [") and f"{tag}]" in line[: len(tag) + 12]:
            log.info(line)  # already tagged by the agent trace ("== [zmip <tag>] ..." / "== [<tag>] ...")
        else:
            log.info(f"[{tag}] {line}")


def _signal_group(proc, sig):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, sig)


def _terminate_pool(signum, frame):
    # Convert scheduler termination into an exception so the pool's finally runs.
    raise SystemExit(128 + signum)


def _finish_child(proc, log_thread):
    proc.wait()
    # A finished lineage must not leave harness descendants holding its pipe.
    _signal_group(proc, signal.SIGTERM)
    if log_thread.ident is not None:
        log_thread.join(timeout=5)
        if log_thread.is_alive():
            _signal_group(proc, signal.SIGKILL)
            log_thread.join(timeout=5)
    if not log_thread.is_alive():
        proc.stdout.close()


def _stop_children(running):
    children = list(running.values())
    for proc, _, _, _ in children:
        _signal_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + 5
    for proc, _, _, _ in children:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=max(0, deadline - time.monotonic()))
    # Kill the whole group even if its leader exited before a descendant.
    for proc, _, _, _ in children:
        _signal_group(proc, signal.SIGKILL)
    for proc, _, _, log_thread in children:
        _finish_child(proc, log_thread)


def run_lineages_parallel(ad, todo, all_labels, outdir, child_args, *, coarse_col, fine_col):
    """todo: plan entries (name, coarse_labels, n_cells) still to run. Writes
    each lineage's subset to <lineage dir>/subset.h5ad and runs
    `python -m zmip.lineage` per lineage under the concurrency plan; returns
    {name: result} for the ones that finished and raises if any failed."""
    todo = sorted(todo, key=lambda ln: -ln["n_cells"])  # biggest first: it bounds the wall-clock
    max_parallel, budget, threads = plan_concurrency(todo)
    log.info(
        f"== zmip lineages in parallel: {len(todo)} to run, up to {max_parallel} at once, "
        f"{threads} thread(s) each, memory budget {budget / 2**30:.1f} GiB "
        f"({available_cpus()} cpu(s), {available_memory_bytes() / 2**30:.1f} GiB available)"
    )
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "MSP_MAX_THREADS"):
        env[k] = str(threads)

    pending = list(todo)
    running = {}  # name -> (proc, est_bytes, t0, log_thread)
    failed, finished = {}, []
    main_thread = threading.current_thread() is threading.main_thread()
    if main_thread:
        previous_term = signal.signal(signal.SIGTERM, _terminate_pool)
    try:
        while pending or running:
            # reap
            for name in list(running):
                proc, est, t0, log_thread = running[name]
                rc = proc.poll()
                if rc is None:
                    continue
                _finish_child(proc, log_thread)
                del running[name]
                took = (time.time() - t0) / 60
                if rc == 0 and contract_done(lineage_dir(outdir, name)):
                    finished.append(name)
                    log.info(f"== [{name}] lineage done in {took:.1f} min")
                else:
                    failed[name] = rc
                    log.warning(f"== [{name}] lineage FAILED (exit {rc}) after {took:.1f} min")
            # launch
            used = sum(est for _, est, _, _ in running.values())
            while pending and len(running) < max_parallel:
                ln = pending[0]
                est = _estimate_bytes(ln["n_cells"])
                if running and used + est > budget:
                    break  # wait for memory; an idle pool always admits the next one
                pending.pop(0)
                name = ln["name"]
                d = lineage_dir(outdir, name)
                os.makedirs(d, exist_ok=True)
                subset_path = os.path.join(d, SUBSET_FILE)
                subset_for(ad, ln["coarse_labels"], coarse_col, fine_col).write_h5ad(subset_path)
                cmd = [sys.executable, "-m", "zmip.lineage", outdir, name, "--subset", subset_path, *child_args]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    bufsize=1,
                    start_new_session=True,
                )
                log_thread = threading.Thread(target=_pump, args=(proc, name), daemon=True)
                running[name] = (proc, est, time.time(), log_thread)
                log_thread.start()
                used += est
                log.info(
                    f"== [{name}] lineage started: {ln['n_cells']} cells, est {est / 2**30:.1f} GiB, "
                    f"{len(running)} running, {len(pending)} waiting"
                )
            if running:
                time.sleep(5)
    finally:
        # Covers subset writes, spawn failures, validation errors and Ctrl-C.
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            _stop_children(running)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, previous_term)
    if failed:
        raise RuntimeError(f"zmip lineage(s) failed: {failed} — re-run to resume (finished lineages are skipped)")
    return {name: load_result(lineage_dir(outdir, name)) for name in finished}


# ---------------------------------------------------------------- subprocess entry


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zmip.lineage", description="run ONE zoom-in lineage (used by zmip's pool)")
    parser.add_argument("outdir")
    parser.add_argument("name")
    parser.add_argument("--subset", required=True, help="the lineage subset written by the parent")
    parser.add_argument("--h5ad", required=True, help="the round's input h5ad (recorded in the lineage's uns)")
    parser.add_argument("--batch-col", required=True)
    parser.add_argument("--species", default=None)
    add_integration_options(parser)
    parser.add_argument("--language", default="English")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--max-turns", type=int, default=200)
    args = parser.parse_args(argv)
    configure_logging("zmip", "msp")
    try:
        args.resolutions = validate_resolutions(args.resolutions)
        harmony_kwargs = parse_harmony(args.harmony)
    except ValueError as exc:
        parser.error(str(exc))

    with open(os.path.join(args.outdir, "zmip_plan.json")) as stream:
        plan = json.load(stream)
    entry = next(ln for ln in plan["lineages"] if ln["name"] == args.name)
    all_labels = {lab for ln in plan["lineages"] for lab in ln["coarse_labels"]}
    mk = pd.read_csv(
        os.path.join(args.outdir, "lineage_markers.csv"), keep_default_na=False, dtype={"lineage": str, "gene": str}
    )
    markers = {g: mk.loc[mk["lineage"] == g, "gene"].tolist() for g in mk["lineage"].unique()}
    sub = sc.read_h5ad(args.subset)
    os.remove(args.subset)
    keys_for_foreign = [f"msp_leiden_r{r}" for r in args.resolutions if r in (1.0, 2.0)]
    run_lineage(
        sub,
        args.name,
        entry["coarse_labels"],
        all_labels,
        markers,
        args.outdir,
        batch_col=args.batch_col,
        species=args.species,
        h5ad_path=args.h5ad,
        resolutions=args.resolutions,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        harmony_kwargs=harmony_kwargs,
        keys_for_foreign=keys_for_foreign,
        language=args.language,
        model=args.model,
        effort=args.effort,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
