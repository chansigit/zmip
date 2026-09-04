"""Validate a small end-to-end run against its original input, including counts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from zmip import cache, publication


def same_matrix(left, right):
    if sparse.issparse(left) or sparse.issparse(right):
        return left.shape == right.shape and (sparse.csr_matrix(left) != sparse.csr_matrix(right)).nnz == 0
    return np.array_equal(left, right, equal_nan=True)


def validate(input_path, outdir):
    root = Path(outdir)
    if not publication.complete(root):
        raise ValueError("global completion receipt is missing, stale or invalid")
    source = sc.read_h5ad(input_path)
    result = sc.read_h5ad(root / "annotated_zmip.h5ad")
    removed = pd.read_csv(root / "zmip_removed.csv", keep_default_na=False, dtype={"cell": str})
    reassigned = pd.read_csv(root / "zmip_reassigned.csv", keep_default_na=False, dtype={"cell": str})
    expected = source.obs_names[~source.obs_names.isin(removed.cell)]
    if not result.obs_names.equals(expected) or result.obs_names.isin(removed.cell).any():
        raise ValueError("survivor/removal coverage differs from input")
    if removed.cell.duplicated().any() or not removed.cell.isin(source.obs_names).all():
        raise ValueError("invalid removal identifiers")
    if reassigned.cell.duplicated().any() or not reassigned.cell.isin(result.obs_names).all():
        raise ValueError("invalid reassignment identifiers")
    original = source[result.obs_names].copy()
    if not result.var_names.equals(original.var_names) or not same_matrix(result.X, original.X):
        raise ValueError("global expression matrix changed")
    for name in source.layers:
        if name not in result.layers or not same_matrix(result.layers[name], original.layers[name]):
            raise ValueError(f"input layer {name!r} changed")
    if original.raw is not None and (result.raw is None or not same_matrix(result.raw.X, original.raw.X)):
        raise ValueError("raw expression matrix changed")
    for column in (c for c in original.obs if c.startswith("msp_ann_")):
        if not result.obs[column].astype(str).equals(original.obs[column].astype(str)):
            raise ValueError(f"original annotation {column!r} changed")
    return {"validated": True, "input_cells": source.n_obs, "output_cells": result.n_obs,
            "genes": result.n_vars, "removed": len(removed), "reassigned": len(reassigned),
            "input_sha256": cache.file_digest(input_path),
            "output_sha256": cache.file_digest(root / "annotated_zmip.h5ad"),
            "preserved_layers": list(source.layers), "runtime": json.loads((root / ".zmip-run.json").read_text())["identity"]["runtime"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("outdir")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()
    with cache.lock_run(args.outdir):
        summary = validate(args.input, args.outdir)
    text = json.dumps(summary, indent=2)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n")
    print(text)
