import importlib
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from harness_bridge.control import PauseRequested

lineage = importlib.import_module("zmip.lineage")


def test_sigterm_during_subset_write_drains_started_lineage(tmp_path, monkeypatch):
    real_spawn, real_sleep = subprocess.Popen, time.sleep
    processes, writes = [], []
    completed = tmp_path / "completed"

    def spawn(cmd, **kwargs):
        process = real_spawn(
            [
                sys.executable,
                "-c",
                "import pathlib,sys,time; time.sleep(.15); pathlib.Path(sys.argv[1]).touch()",
                str(completed),
            ],
            **kwargs,
        )
        processes.append(process)
        return process

    def write(path):
        writes.append(path)
        if len(writes) == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(lineage.subprocess, "Popen", spawn)
    monkeypatch.setattr(lineage.time, "sleep", lambda _: real_sleep(0.01))
    monkeypatch.setattr(lineage, "subset_for", lambda *a: SimpleNamespace(write_h5ad=write))
    monkeypatch.setattr(lineage, "plan_concurrency", lambda _: (2, 10**12, 1))
    monkeypatch.setattr(lineage, "contract_done", lambda _: completed.exists())
    todo = [dict(name=name, coarse_labels=[name], n_cells=10) for name in ("A", "B", "C")]
    with pytest.raises(PauseRequested) as exc:
        lineage.run_lineages_parallel(
            None, todo, {"A", "B", "C"}, str(tmp_path), [], coarse_col="coarse", fine_col="fine"
        )
    assert exc.value.code == 3 and len(processes) == 1
    assert completed.exists() and processes[0].returncode == 0
