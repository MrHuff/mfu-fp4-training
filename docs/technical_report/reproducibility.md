# Reproducibility guide

The release separates source-controlled scientific contracts from external
model and dataset bytes. It contains all project code, kernel integrations,
evaluation logic, report inputs, and dependency pins needed to continue the
work. It does not contain credentials, private storage locations, checkpoint
bytes, or checkpoint metadata.

## Source identity

The public export vendors every kernel/runtime component as ordinary files at
the recorded commits. After cloning the one-commit release bundle, verify the
component and file ledgers without updating submodules:

```bash
python tools/release_capsule.py doctor --phase source
scripts/release/bootstrap.sh --verify-only
```

`release/components.json` records the exact upstream commits and content
digests. `SHA256SUMS` binds the flattened tree. Python environments should be
created from the checked-in project dependency files; GPU results additionally
depend on the CUDA and driver versions described in the report.

## Paper

The report is independently rebuildable from its public inputs:

```bash
cd docs/technical_report
sha256sum -c data/SHA256SUMS
make
make arxiv-source
```

The figure builders fail closed on row counts, route/step grids, numerical
geometry, and hashes. The committed PDF is a convenience copy; generated
figures and tables remain derivable from the CSV inputs. `make arxiv-source`
also emits a matching Overleaf ZIP with `main.tex` at its root and verifies a
clean PDFLaTeX build with shell escape disabled.

## Fixed-independent validation

The r15 materializer is
`tools/evaluation/fixed_independent_r15/materialize_fixed_independent.py`.
Provide local Mosaic streams and the exact tokenizer assets through its CLI.
It deterministically selects the 82/18 source mix, hash-shuffles documents,
packs 768 sequences of width 8,193, writes 32 safetensor shards, and seals all
outputs with content hashes. Storage locations are not serialized.

The r16 matrix builder accepts a local, public-schema checkpoint inventory.
The inventory is intentionally external because checkpoints and their metadata
are not publication artifacts. Each evaluation task is selected by a public
checkpoint key and verifies the local `.metadata` hash and exact 32- or
64-shard DCP geometry before conversion.

The numerical path is:

1. `scripts/evaluation/convert_llama8b_dcp_to_hf.py` converts an exact DCP
   checkpoint to a local Hugging Face directory.
2. `scripts/evaluation/validate_llama8b_conversion_parity.py` and
   `validate_llama8b_canonical_parity.py` verify conversion/state and canonical
   TorchTitan numerical parity.
3. `tools/evaluation/validation_matrix_r16/evaluate_matrix_task.py` evaluates
   the fixed token panel with Llama-3.1 scaled RoPE and math SDPA.
4. The r22 collector emits the eight-column scientific ledger. The final
   public integration extends it to 49 exact route/checkpoint cells without
   distributing operational receipts or checkpoint metadata. Synthetic CPU
   tests cover seals, geometry, partial outputs, and checksum behavior.

## Downstream evaluation

`tools/r23_scaledrope/run_canonical_lm_eval.py` is the exact corrected
scaled-RoPE wrapper used for the published panel. Its SHA-256 is
`48827b0f2bb1cb263e6ff5b1d851ce3cd45bd472d87554a86771076b74409466`.
The wrapper pins TorchTitan semantics, transformers, lm-eval, RMSNorm, RoPE,
and the math attention backend.

The r25 collector consumes local route directories containing `metrics.json`,
`canonical-parity.json`, and `SHA256SUMS`. It validates the four shot/metric
contracts and emits only the eight scientific columns included in the report.
The final public integration adds TE F0L4 and the operand-wise fixed-H32
route, producing an eleven-route step-38,000 panel.

## Scientific limits

- Validation is fixed-independent, not proven held out.
- Historical BF16 and TE-native seeds were not recovered; their plot offsets
  are reported as explicit seed-alignment estimates.
- TE F0L4 and the operand-wise fixed-H32 route each have complete public
  3,815-point numerical histories through logged step 38,140. Checkpoint
  bytes and operational checkpoint metadata remain external.
- Checkpoint artifacts are user-supplied external inputs and are never implied
  to be part of this source release.
- Hardware timing results require comparable accelerators, clocks, software,
  and topology; code reproducibility alone does not guarantee identical timing.
