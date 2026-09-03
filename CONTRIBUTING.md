# Contributing to MFU FP4 training

Thank you for helping improve the FP4 training system. This repository is a
flattened, reproducible source release: pinned dependencies are ordinary source
directories, not live Git submodules.

## Development environment

Use the recorded Python, PyTorch, CUDA, and GPU architecture contract in
`release/container_dependency_lock.json`. The supported bootstrap sequence is:

```bash
python tools/release_capsule.py doctor --phase source
scripts/release/bootstrap.sh --verify-only
scripts/release/bootstrap.sh
```

The release contract uses Python 3.12.3 and the digest-pinned NVIDIA PyTorch
container. Do not silently replace a pinned dependency with the head of a
similarly named branch.

## Tests

Run the inexpensive source checks before opening a pull request:

```bash
scripts/release/run_gates.sh --cpu-only
```

On a supported Blackwell host, build and test the production extensions:

```bash
scripts/release/build_kernels.sh
scripts/release/run_gpu_gates.sh
```

Changes to training or checkpoint semantics should also exercise a miniature
fresh run and a full-state resume. A model-only load is not a checkpoint-resume
test.

## Adding or changing a numerical route

Keep route identity explicit and versioned:

1. implement the converter or dispatch under `low_bits_training/`;
2. add the corresponding kernel under `fp4_runtime/`;
3. add a new route contract under `configs/paper/` instead of changing an
existing recipe in place;
4. add CPU/source, native ABI, numerical, and performance checks appropriate to
   the route; and
5. record seed, batch geometry, format choices, and resume state explicitly.

Production routes and historical reference implementations are distinguished
in `release/source_support_matrix.json`. Historical MXFP4, NVFP4, localCTA, and
low-precision output-head code is useful for comparison, but it is not covered
by the production build or support promise unless that matrix says otherwise.

## Licensing and provenance

Contributions to separable project-original software are accepted under the
repository's MIT License when the contributor is authorized to grant those
rights. Source inherited from the Graphcore Research training lineage and
other Apache-identified source remains under the Apache-2.0 terms retained in
`LICENSES/APACHE-2.0.txt`. Manuscript prose and original figures are covered
by the scoped CC BY 4.0 grant in `LICENSES/CONTENT.md`. Vendored and adapted
components retain their own terms.

Do not remove or replace an existing copyright, attribution, modification, or
license notice. If you modify Apache-licensed source, add the notice required by
Apache-2.0 section 4(b). See `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`
before changing copied or vendored code.

## Public-source hygiene

Never commit credentials, access tokens, private keys, dataset or tokenizer
objects, model weights, checkpoint contents or metadata, experiment-tracking
identities, cluster objects, private storage locations, compiled extensions,
profiles, or logs. Bind authorized local data and checkpoints through
`release/external_inputs.schema.json`, keeping the completed binding outside
the checkout.

After regenerating and reviewing `SHA256SUMS`, the component inventory, and the
public audit in a clean commit, build the subsequent public snapshot with:

```bash
python scripts/release/build_public_snapshot.py \
  --output ../mfu-fp4-public-snapshot
```

This command verifies the tracked public tree and runs the resulting one-root
bundle through the disposable-clone verifier. Repeat the independent secret
scan against the same final commit before publication.

Pull requests should describe the numerical or systems change, the exact gates
run, and any compatibility or reproduction limitation.
