# Format-Aware Fusion for Fast FP4 Pretraining

*Native MXFP4 and NVFP4 Training on FP4-Compatible Hardware*

This directory is the self-contained source release for the technical report.
It includes the LaTeX source, bibliography, public numerical inputs, figure and
table builders, rendered figures, and an arXiv source-bundle builder.

The paper covers native MXFP4 and CTA-local NVFP4 execution, complete-boundary
fusion, 160B-token Llama-8B training curves, same-accelerator performance,
fixed-independent validation, and a corrected eleven-route downstream panel.
The low-precision output head is retained as a negative result.

Operational scheduler records, checkpoint locations and inventories, source
lineage, run/pod/workload identities, private storage names, service-account
identities, and credentials are intentionally absent. Their removal does not
change the paper's numerical rows. Content hashes in `data/SHA256SUMS` bind
every published data input. Complete F0L4 and operand-wise fixed-H32
trajectories and the final 58-cell validation and eleven-route downstream
ledgers are included as public, identity-free scientific tables.

## Build

From this directory:

```bash
make
```

The target regenerates every figure and table before building
`fp4_training_systems_report.pdf`. It requires Python with pandas, matplotlib,
NumPy, and statsmodels, plus a TeX installation containing `latexmk` and the
packages imported by `main.tex`.

To prepare and verify the deterministic arXiv source archive and the matching
Overleaf-importable ZIP:

```bash
make arxiv-source
```

Both archives and their SHA-256 files are written under `build/`. They contain
the identical dependency-minimal source set, with `main.tex` at archive root.
The packaging check extracts the ZIP and builds it with PDFLaTeX through
`latexmk`, with shell escape disabled. See `reproducibility.md` for the
evaluator and external-input contracts.
