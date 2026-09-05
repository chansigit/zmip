# Changelog

All notable changes to zmip. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

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
