# Changelog

All notable changes to zmip. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## 0.3.3 - 2026-09-05

### Added
- A bounded `annotation_status` tool for recovering accepted labels, actions,
  merge/reassignment targets and pending work after an agent session reset.
  Status follows the current clustering after subclustering, identifies removed
  parent merge targets, and pages complete saved entries without flooding context.
- Recovery prompts reconcile host submissions with task progress and process
  small batches of pending clusters before requesting more context.

### Fixed
- Preserve explicit batch/sample artifact decisions as keep for review even when
  submitted with remove_reason=other. Retain the original reason in the host
  audit and reject unsafe saved proposals before applying cell deletions.
- Keep specific independently supported QC removal categories unchanged. Require
  MSP >=0.3.3,<0.4 for the matching annotation safeguard.

## 0.3.2 - 2026-09-05

### Fixed
- Require MSP >=0.3.2,<0.4 so author-label discovery supports pandas string
  dtypes, including pandas 3. Retain the compatibility helper layer.

## 0.3.1 - 2026-09-05

- Require agent-harness-bridge >=0.2.1,<0.3 for bounded host Read support.

### Safety and agent recovery
- Bound agent expression tables to 16 KiB with explicit cluster selection; retain the MSP 0.3 three-argument gene-table API.
- Preserve batch-only annotation removal requests as keep with host audit and review reporting; validate effective lineage labels and reject unsafe legacy proposal application.

### Added
- GitHub Actions for Ruff lint/format, pytest with coverage reporting on Python
  3.10 and 3.12, and importing a built wheel outside the source checkout.
- A tracked follow-up list in `TODO.md`.

### Changed
- Prefer MSP's public annotation/evidence/report helpers, with the seven legacy
  fallbacks isolated in `msp_compat.py` so published MSP 0.3.0 still installs.
- Apply a project-local Ruff policy matching MSP and normalize existing code;
  bind loop-local closure values explicitly without changing computation methods.
- Document PyPI installation and the harmonypy 2 runtime constraint (documentation
  changes following the 0.3.0 release).
- Retain the legacy torch identity field in schema 1 pending an explicit migration;
  this field does not add a torch dependency. As with any source update, the
  source digest changes, so pre-upgrade runs require a fresh output directory.

## 0.3.0 - 2026-09-04

### Changed
- Requires `msp-sc>=0.3.0,<0.4` and imports the evidence layer through its
  public `msp.evidence` names (`DegTables`, `DegCache`, `parse_reference`,
  `cluster_order`, `gene_table`, `stability_table`, `file_inventory`,
  `load_removal_mask`, `load_paga_neighbors`, `DEG_TOOL_DOC`, `DEG_SQL_DOC`)
  instead of the underscore aliases that msp 0.3 removes. `check_runtime`
  probes the same public names.
- Requires `agent-harness-bridge>=0.2.0,<0.3`. Progress lines go through
  `logging` (`zmip` logger family) instead of `print`; the CLI entry points
  (`python -m zmip`, `zmip.lineage`, `zmip.report`) call
  `harness_bridge.configure_logging("zmip", "msp")`, so zmip, msp and the
  bridge share one flushed stdout stream. Failure and budget notices are
  logged at `WARNING`.

## 0.2.0 - 2026-09-04

- Shared agent harness bridge, recoverable publication of global outputs,
  runtime compatibility validation, resume hardening; `msp-sc>=0.2.0,<0.3`.
