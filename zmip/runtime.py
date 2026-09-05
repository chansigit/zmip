"""Compatibility checks and reproducible identity for the computation runtime."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

# Schema 1 deliberately retains torch as a legacy identity field. Removing it
# requires an explicit schema migration; it is not a runtime dependency.
PACKAGES = (
    "zmip",
    "msp-sc",
    "agent-harness-bridge",
    "standissect-lite",
    "scanpy",
    "anndata",
    "numpy",
    "scipy",
    "pandas",
    "h5py",
    "numba",
    "scikit-learn",
    "igraph",
    "leidenalg",
    "harmonypy",
    "torch",
    "matplotlib",
    "openai",
    "openai-agents",
    "mcp",
    "claude-agent-sdk",
)
SOURCE_MODULES = ("zmip", "msp", "harness_bridge", "standissect_lite")


def source_digest(root):
    """Detect editable-source changes even when a package version was not bumped."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def runtime_identity():
    versions, sources = {}, {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    for name in SOURCE_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.submodule_search_locations:
            sources[name] = source_digest(next(iter(spec.submodule_search_locations)))
    return {"schema": 1, "python": platform.python_version(), "versions": versions, "source_sha256": sources}


def check_runtime():
    """Test the shared API behavior that package version ranges cannot guarantee."""
    try:
        from harness_bridge import ToolSpec, resolve_agent_config, run_agent
        from msp.evidence import DegCache, DegTables, parse_reference
        from msp.integrate import integrate_adata

        assert all(
            callable(f) for f in (ToolSpec, resolve_agent_config, run_agent, DegCache, DegTables, integrate_adata)
        )
        known = ["5", "0", "1", "5,0", "5,1"]
        if parse_reference("5,1", known) != ("5,1",):
            raise ValueError("MSP splits exact subcluster IDs")
        if set(parse_reference('"5,0","5,1"', known)) != {"5,0", "5,1"}:
            raise ValueError("MSP cannot parse quoted pooled subcluster IDs")
        try:
            parse_reference("5,0,5,1", known)
        except ValueError:
            pass
        else:
            raise ValueError("MSP accepts ambiguous pooled subcluster IDs")
    except (ImportError, AttributeError, TypeError, ValueError, AssertionError) as exc:
        raise RuntimeError(
            "incompatible MSP/harness runtime; install the tested source wheels with "
            "scripts/validate_install.sh (see README). Details: " + str(exc)
        ) from exc


if __name__ == "__main__":
    check_runtime()
    print(json.dumps(runtime_identity(), indent=2))
