# Fixed-independent validation ledger

`VALIDATION_LEDGER.csv` is the compact 44-cell numerical authority used by the
technical report. Every row is complete and uses the same 768-sequence,
6,291,456-token panel with training-lineage Llama-3.1 scaled-RoPE semantics.

The stream is fixed and independent, but not proven held out. The public
artifact deliberately omits checkpoint locations, metadata inventories,
scheduler identities, and operational receipts. `SHA256SUMS` binds the files
in this directory.
