# Public evaluation tools

These directories preserve the final scientific evaluation path without the
experiment workspace that produced it:

- `fixed_independent_r15/` materializes and evaluates the deterministic
  82/18 fixed-independent validation stream.
- `validation_matrix_r16/` builds the 44-cell route/checkpoint matrix, runs a
  selected local cell, and collects local results.
- `validation_ledger_r22/` performs final fail-closed validation and emits the
  compact paper ledger.
- `validation_paper_inputs_r23/` renders figures/tables and seals a complete
  public ledger.
- `downstream_ledger_r25/` validates local canonical downstream results and
  emits the compact nine-route ledger.
- `terminal_eval_addon_r26/` is the envelope-only adapter for the three
  terminal add-on cells. Stale checkpoint specifications are not included.

All runtime inputs are explicit local paths. The tools serialize scientific
settings, public route labels, and content hashes; they do not serialize
credentials, storage locations, scheduler identities, or checkpoint object
inventories. Result directories are create-only and checksum sealed.

The exact corrected r23 scaled-RoPE lm-eval wrapper is kept separately at
`tools/r23_scaledrope/run_canonical_lm_eval.py` with SHA-256
`48827b0f2bb1cb263e6ff5b1d851ce3cd45bd472d87554a86771076b74409466`.
