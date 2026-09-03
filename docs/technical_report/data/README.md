# Public report data

This directory contains the numerical inputs used to regenerate the report.
`SHA256SUMS` is the authoritative inventory. The files contain scientific
measurements only; operational storage paths, scheduler/service identities,
timestamps, private source-run identities, checkpoint inventories, and
checkpoint metadata were removed for release.

## Training and performance inputs

- `llama8b_160b_loss_snapshot_20260827.csv` is the immutable base snapshot
  used by the main loss figures. The figure builder replaces route rows for
  which a later complete ledger is listed below; in particular, its stale
  27/5 tail is not the publication authority.
- `llama8b_160b_training_summary_20260827.csv` is a historical predecessor.
  It preserves then-live statuses and rounded endpoints and is superseded by
  the complete route ledgers and final integration tables below.
- `llama8b_mxfp4_rht_history_20260831.csv`,
  `llama8b_localcta_rht_canonical_metrics_20260830.csv`, and
  `llama8b_operand_h16_history_20260902.csv` are complete 3,815-row route
  trajectories containing only numerical and semantic-route fields.
- `llama8b_localcta_mxfp4_27_5_history_20260903.csv` is the complete
  checkpoint-aware 27/5 depth-hybrid trajectory. It replaces a stale snapshot
  that had been exported while the final continuation was still running.
  Its 3,815 rows are literal W&B observations; the splice excludes overrun
  updates after each restored checkpoint and does not interpolate values.
- `llama8b_terminal_training_history_r1_results_20260903/` contains complete
  3,815-row TE F0L4 and operand-wise fixed-H32 trajectories plus a compact
  table of their exact step-38,140 raw endpoints. Private run and checkpoint
  provenance remains only in the sealed evidence repository.
- `llama8b_pure_v5_recovered_log_points_20260830.csv` contains 74 sparse,
  literal observations. The plot shows these as anchors with a display-only
  LOWESS guide; reported endpoint values remain observed values.
- `llama8b_seed42_overlap_20260828.csv` and
  `llama8b_seed42_loss_adjustment_20260828.csv` record the exact overlap and
  robust seed-alignment calculation.
- The B300 and GB200 performance CSVs are reduced scientific tables, not raw
  service exports.

## Validation and downstream inputs

- `llama8b_final_eval_integration_r1_results_20260903/validation/` contains
  the compact final 49-cell validation ledger. Every complete row scores 768
  sequences and 6,291,456 target tokens.
- `llama8b_final_eval_integration_r1_results_20260903/downstream/` contains
  the corrected scaled-RoPE downstream ledger for eleven exact step-38,000
  route checkpoints.
- The earlier r22 validation and r25 downstream directories remain as public
  predecessors. The final integration is the publication authority.

The validation claim is deliberately limited to **fixed and independent, not
proven held out** because exact exclusion from every training example cannot
be established from the available public evidence.

## Regeneration helpers

- `build_seed42_loss_adjustment.py` derives the seed offsets from public
  `step,loss` control histories.
- `fetch_mxfp4_rht_wandb.py` can export a caller-selected source history. It
  accepts the source identity at runtime, reads authentication only from the
  process environment, and never writes source account or run identifiers.

No checkpoint file or checkpoint metadata file is distributed here. The
reusable conversion, parity, validation, and downstream tools live under
`scripts/evaluation/`, `low_bits_training/analysis/`, and `tools/evaluation/`.
