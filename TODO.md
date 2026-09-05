# Follow-up work

- Run full-size 19Liu inspect/annotation followed by ZMIP, checking report sections,
  cell conservation and removal reasons. A successful small run does not validate
  full-dataset performance or biological accuracy; record the actual run evidence.
- The minimum MSP dependency is already `>=0.3.2,<0.4`. Application modules
  use public names; the seven private fallbacks remain confined to
  `zmip/msp_compat.py`. Remove those fallbacks in the next maintenance change,
  without changing the already validated release package for this cleanup.
  Acceptance: all seven helpers resolve through MSP's public API with the
  minimum supported MSP version, compatibility tests and the full suite pass,
  and a built wheel imports outside the checkout. Keep `<0.4` until that
  release is tested.
- At the next explicit runtime-identity schema migration, remove the legacy
  `torch` version field. Schema 1 retains it to avoid silently changing resume
  identity comparison. Torch is not a computation dependency. Acceptance:
  the new schema omits torch, with documented migration or explicit rejection
  of old-schema runs; an old run must not be silently accepted under a new
  identity contract.
- Establish a measured coverage gate after expanding agent-session tests. CI
  currently reports coverage without claiming an unmeasured percentage target.
