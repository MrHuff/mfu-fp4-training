# Clean public export

`tools/public_clean_export.py` creates the publication boundary. It reads only
tracked Git objects at explicit commits and copies only paths selected by
`release/public_export_manifest.json`. It does not copy a working directory or
reuse the repository's historical object graph.

The output contains:

- `source-tree/`: the complete flattened source tree with no `.git` files;
- `mfu-fp4-training.tar.gz`: a deterministic archive of that tree;
- `mfu-fp4-training.bundle`: a cloneable repository with exactly one new root
  commit and no gitlinks; and
- `BUILD_REPORT.json`: hashes and release-gate status without local paths or
  matched sensitive content.

Inside the source tree, `SHA256SUMS` binds every file and
`release/components.json` binds each vendored component to its upstream URL,
commit, Git tree, content-ledger digest, and license-file candidates. License
names are not inferred from file text. `release/public_release_audit.json`
records only rule names and public-relative paths.

Every rendered paper figure that bypasses the text scanner is bound to an
expected SHA-256 in the export manifest; a same-path binary mutation aborts
the build. Identity-bearing defaults, comments, and examples retained in
essential source are normalized as whole tokens. The inventory records the
source hash, output hash, public path, and rule class for each such file.
Credential-shaped content is never normalized and always blocks the export.

Build from a reviewed source commit into a new path:

```bash
python tools/public_clean_export.py build \
  --source-revision REVIEWED_COMMIT \
  --output ../mfu-fp4-public-export
```

The default resolver uses anonymous HTTPS and refuses non-GitHub hosts,
credential-bearing URLs, mutable dependency refs, or missing exact commits.
`--repo-map URL=LOCAL_GIT_REPOSITORY` and `--offline` are intended for an
audited local mirror or deterministic tests. A local mirror proves object
availability, not public anonymous availability.

Verify a materialized tree without Git metadata:

```bash
python tools/public_clean_export.py verify \
  --tree ../mfu-fp4-public-export/source-tree
```

For a quarantined candidate created with `--allow-release-blockers`, append
`--allow-blocked` to verify content integrity without claiming publication
readiness. Inside that candidate, `scripts/release/run_gates.sh --cpu-only`
uses the corresponding explicit review mode; unsafe content and ledger drift
still fail.

The exporter fails closed on credential-shaped values, private host paths,
storage locations, tracking/job identities, checkpoint payloads or metadata,
compiled and profiling artifacts, internal agent documents, unsafe symlinks,
and unlisted files. It reports paths and rule names, never matched values.

## Publication gates (2026-09-03 UTC)

- The copyright holder selected MIT for project-authored software, including
  authorized custom FP4-runtime portions. The export overlays that scoped
  license into the curated runtime while preserving every included third-party
  and NVIDIA notice. Manuscript prose and original figures remain CC BY 4.0.
- The runtime export is allowlisted. It includes the supported quantizers,
  fused operations, ThunderKittens GEMMs, older MXFP4/NVFP4/localCTA versions,
  and TK low-precision-head experiments. The latter two categories are labeled
  historical and unsupported. Unrelated SageAttention, runtime CUTLASS,
  FlashAttention, QUTLASS, and FA4 baggage is not exported.
- The immutable NVIDIA base-image digest and its image-baked Python, PyTorch,
  CUDA, Triton, and required package versions are recorded in
  `release/container_dependency_lock.json`. This is a container-anchored
  contract, not a claim that similarly named PyPI wheels reproduce the image.
- A fresh, scrubbed single-root candidate passed all six production kernel
  builds, exact-name ABI imports, deterministic MXFP4-v4 and NVFP4-v5
  BF16-reference checks, and the localCTA 2D-weight contract on a GB200 host
  matching the recorded Python, PyTorch, and CUDA software contract. The
  public, path-free result is `release/gpu_gate_receipt.json` and its
  executable entry point is `scripts/release/run_gpu_gates.sh`.
- No container engine was available for an independent pull of the recorded
  image digest. The receipt therefore does not label the matching-host run as
  a cold-container run. It also does not claim a repeated distributed
  64-GPU/160B-token trajectory.
- Before publication, the clean candidate still requires an independent
  secret scan and disposable-clone bundle verification.
- The development history is not publishable. Only the new one-commit bundle
  or deterministic source archive may cross the publication boundary.

`--allow-release-blockers` may be used only to create a quarantined candidate
while a named technical gate is pending. Such an artifact must not be
published.

## Refreshing the paper from its separate evidence history

The canonical evaluation ledger intentionally has a different Git history
from this release-source branch.  Do not merge that history or copy its working
directory.  Freeze an exact evaluation commit, then audit it through Git
objects:

```bash
python tools/prepare_public_release_refresh.py audit \
  --base-revision REVIEWED_RELEASE_SOURCE_COMMIT \
  --paper-repo /path/to/canonical-evaluation-ledger \
  --paper-revision EXACT_40_CHARACTER_COMMIT \
  --json > /tmp/paper-refresh-audit.json
```

The report contains commit/tree identities, changed public-relative paths,
hashes for candidate paper figures, and rule names. It never records matched
secret or private values. A credential finding is a hard stop. Private storage,
tracking, host, and job identities must not be promoted to public provenance.
Operational receipts and checkpoint/conversion manifests stay in the immutable
evidence history; compact scientific CSV ledgers are the publication boundary.

For review, materialize a new path (never the release repository itself):

```bash
python tools/prepare_public_release_refresh.py stage \
  --base-revision REVIEWED_RELEASE_SOURCE_COMMIT \
  --paper-repo /path/to/canonical-evaluation-ledger \
  --paper-revision EXACT_40_CHARACTER_COMMIT \
  --output /tmp/mfu-paper-public-review
```

This creates a scrubbed review tree, removes generated files, proprietary font
and branding assets, and excludes operational evidence. It retains the public
LaTeX style and public reproducibility prose from the reviewed release source.
It is deliberately **not** a publishable artifact: the maintainer must port the
reviewed scientific delta to a new descendant of the release-source branch,
remove identity-bearing CSV columns or replace them with a reviewed public
schema, regenerate every affected table and figure, and update the exact binary
allowlist. Then build `public_clean_export.py` into a fresh directory and run
verification, cold-clone bootstrap, CPU gates, paper build, and arXiv-source
build from disposable clones. Running tests in the sealed source tree is
forbidden because Python and pytest caches make that tree fail closed.

## Disposable bundle verification

After producing a fresh public export, run the complete source-only check from
a throwaway clone rather than inside the sealed source tree:

```bash
scripts/release/verify_public_bundle.sh \
  ../mfu-fp4-public-export/mfu-fp4-training.bundle
```

The verifier checks that the bundle is valid, has one commit and no gitlinks,
then runs export integrity, vendored-source bootstrap, cache-free CPU/source
gates, the report build, and source packaging. The paper packager emits a
deterministic arXiv tarball and an Overleaf ZIP from the same minimal staged
tree; the ZIP has `main.tex` at its root and is extracted and rebuilt with
PDFLaTeX and shell escape disabled.
