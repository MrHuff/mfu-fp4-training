# Final public evaluation ledgers

The validation ledger contains 49 completed exact-checkpoint cells. The
downstream ledger contains eleven exact step-38,000 routes evaluated on MMLU,
HellaSwag, WinoGrande, and ARC-Challenge with the paper's common protocol.

The 27/5 depth hybrid has one recoverable exact validation checkpoint: step
38,000. Its NLL is 2.538116353452965 with sequence-level standard error
0.01187564212740326 over 768 sequences (6,291,456 target tokens). No exact
intermediate checkpoint is available for that route, so the report plots this
measurement as a terminal point and does not interpolate a validation curve.

Only scientific route labels, training-token coordinates, metrics, sample counts, and status
are published here. Checkpoint metadata, storage identities, task receipts,
and conversion manifests remain in the separately sealed evidence repository.
