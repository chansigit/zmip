# Follow-up work

- Run full-size 19Liu inspect/annotation followed by ZMIP, checking report sections,
  cell conservation and removal reasons. A successful small run does not validate
  full-dataset performance or biological accuracy; record the actual run evidence.
- Once MSP 0.3.1 is published and verified, raise the minimum MSP dependency and
  remove the seven private fallbacks confined to `zmip/msp_compat.py`. Application
  modules already use public names; keep `<0.4` until that release is tested.
- At the next explicit runtime-identity schema migration, remove the legacy
  `torch` version field. Schema 1 retains it to avoid silently changing resume
  identity comparison. Torch is not a computation dependency.
- Establish a measured coverage gate after expanding agent-session tests. CI
  currently reports coverage without claiming an unmeasured percentage target.
