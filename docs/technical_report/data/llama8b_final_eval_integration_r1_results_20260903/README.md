# Final public evaluation ledgers

The validation ledger contains 58 completed exact-checkpoint cells. The
downstream ledger contains eleven exact step-38,000 routes evaluated on MMLU,
HellaSwag, WinoGrande, and ARC-Challenge with the paper's common protocol.

The 27/5 depth hybrid is evaluated at the five common cuts: steps 2,000,
10,000, 18,000, 29,000, and 38,000. Its step-38,000 NLL is
2.538116353452965 with sequence-level standard error 0.01187564212740326 over
768 sequences (6,291,456 target tokens). The report connects evaluated
checkpoints only and does not interpolate or extrapolate validation values.

Only scientific route labels, training-token coordinates, metrics, sample counts, and status
are published here. Checkpoint metadata, storage identities, task receipts,
and conversion manifests remain in the separately sealed evidence repository.
