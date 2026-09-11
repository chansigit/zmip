# Follow-up work

- Full-size 19Liu engineering acceptance completed (2026-09-05): 81,079 source
  cells went through MSP inspect/annotation and six ZMIP lineages. ZMIP retained
  75,394 cells, removed 1,985 and reassigned 2,656; expression/counts, cell and
  gene order, ledgers and report contents were checked. Reassignment updates
  global labels; transferred cells were not re-embedded in the target lineage.
  The final report, parallel-DE and bounded-status patches have separate
  validation; the full chain was not rerun with all three final files.
  Remaining work: independent biological review of labels and uncertain
  removal decisions, including Myeloid cluster 23, Mural cluster 8 and Stromal
  cluster 45. Sample-condition confounding and mixed marker expression do not
  independently establish technical invalidity; the audit neither validates
  every removal nor concludes these uncertain cells were necessarily removed
  incorrectly.
- ~~Remove the seven private fallbacks in `zmip/msp_compat.py`~~ — done in
  0.3.7 (2026-09-11): minimum MSP dependency moved to `>=0.4.0,<0.5` (msp
  removed the deprecated `msp.harness` shim in 0.4), and `msp_compat.py` is
  now a plain re-export of `msp.evidence`/`msp.report`. Full suite passes;
  a built wheel imports `zmip.msp_compat` cleanly outside the checkout.
- At the next explicit runtime-identity schema migration, remove the legacy
  `torch` version field. Schema 1 retains it to avoid silently changing resume
  identity comparison. Torch is not a computation dependency. Acceptance:
  the new schema omits torch, with documented migration or explicit rejection
  of old-schema runs; an old run must not be silently accepted under a new
  identity contract.
- Establish a measured coverage gate after expanding agent-session tests. CI
  currently reports coverage without claiming an unmeasured percentage target.
