#!/usr/bin/env bash
# Build explicit source wheels, install without editable paths, and test outside the checkout.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 VALIDATION_DIR MSP_SOURCE HARNESS_SOURCE STANDISSECT_SOURCE" >&2
  exit 2
fi
repo=$(cd "$(dirname "$0")/.." && pwd)
validation=$(realpath -m "$1")
mkdir -p "$validation/wheels"
export UV_CACHE_DIR="$validation/uv-cache"
for source in "$repo" "$2" "$3" "$4"; do
  uv build --wheel --out-dir "$validation/wheels" "$source"
done
if [[ ! -x "$validation/env/bin/python" ]]; then
  uv venv "$validation/env" --python "${ZMIP_TEST_PYTHON:-3.12}"
fi
python_bin="$validation/env/bin/python"
uv pip install --python "$python_bin" --index https://download.pytorch.org/whl/cpu \
  --reinstall-package zmip --reinstall-package msp-sc \
  --reinstall-package agent-harness-bridge --reinstall-package standissect-lite \
  --index-strategy unsafe-best-match -c "$repo/constraints-runtime.txt" \
  "$validation"/wheels/*.whl pytest
uv pip check --python "$python_bin"
uv pip freeze --python "$python_bin" > "$validation/requirements-resolved.txt"
sha256sum "$validation"/wheels/*.whl > "$validation/wheel-sha256.txt"
mkdir -p "$validation/tests"
cp -R "$repo/tests/." "$validation/tests/"
(
  cd "$validation"
  export PYTHONPATH= LC_ALL=C LANG=C
  "$python_bin" -m zmip.runtime > runtime.json
  "$python_bin" -c 'import h5py, json, platform; print(json.dumps({"libc": platform.libc_ver(), "hdf5": h5py.version.hdf5_version}, indent=2))' > native-runtime.json
  "$python_bin" -m zmip --help > cli-help.txt
  "$python_bin" -m pytest -q tests | tee pytest.log
)
