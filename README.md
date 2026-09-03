# MFU FP4 training

This repository contains the training integration, CUDA kernels, numerical
contracts, paper recipes, and analysis code for the MFU FP4 Llama experiments.
Its dependencies are vendored at recorded commits so a clone does not depend on
mutable branches or nested Git submodules.

The repository does not contain credentials, cluster identities, dataset or
tokenizer objects, model weights, checkpoint contents or metadata, experiment
tracking identities, or private storage locations. Those inputs remain local
to the user and are bound through `release/external_inputs.schema.json`.

Canonical public repository: <https://github.com/MrHuff/mfu-fp4-training>.

## Licensing

Project-authored software for which the copyright holder is authorized to
grant those rights, including authorized custom portions of `fp4_runtime/`, is
provided under the MIT License. Source inherited from the Graphcore Research
training lineage remains under Apache-2.0. Robert Hu's original manuscript,
original figure images, and prose documentation under
`docs/technical_report/` are provided under Creative Commons Attribution 4.0
International. Vendored, copied, and adapted material retains its own terms;
the MIT license does not relicense it. See `LICENSES/CONTENT.md` for the exact
content boundary, `LICENSES/APACHE-2.0.txt` for the retained Apache terms, and
`THIRD_PARTY_NOTICES.md` for the file and component map.

Built with Meta Llama 3. The repository retains the applicable Meta Llama 3
Community License and attribution under `LICENSES/`.

## Start here

1. Verify the source and every vendored component:

   ```bash
   python tools/release_capsule.py doctor --phase source
   scripts/release/bootstrap.sh --verify-only
   ```

2. Build the training image. Both container files use the exact NGC digest
   recorded in `release/container_dependency_lock.json`; `FP4Dockerfile` is a
   compatibility filename with the same base and setup path.

   ```bash
   docker build --pull --file Dockerfile --tag mfu-fp4:25.10 .
   ```

   Pulling from NGC can require registry authentication, which must be supplied
   through the container client and is never copied into this repository. The
   build performs no `apt`, `curl`, or PyPI dependency resolution. It verifies
   the image-baked Python/CUDA contract, builds the custom Transformer Engine
   wheel from the vendored source in a temporary directory with package indexes
   disabled, installs that wheel without dependencies, and verifies its version
   and custom-format source bytes. The training integration itself runs from
   `/opt/mfu` through the fixed `PYTHONPATH`; the public tree has no root Python
   packaging manifest, so the image does not pretend to install one.

3. Start the image with an SM100 GPU, then build the six production runtime
   extensions and run the source and numerical contracts:

   ```bash
   docker run --rm --gpus all --interactive --tty mfu-fp4:25.10
   scripts/release/bootstrap.sh
   scripts/release/build_kernels.sh
   scripts/release/run_gates.sh --cpu-only
   scripts/release/run_gpu_gates.sh
   ```

   The container defaults to `/bin/bash` in `/opt/mfu`. Runtime kernels are
   deliberately built after start because their ABI gate requires an attached
   SM100 GPU; the Docker build does not claim GPU access. The source contract is
   container-anchored: similarly named PyPI wheels do not reproduce NVIDIA's
   image-baked PyTorch/CUDA ABI, and bootstrap fails closed on any mismatch.

   The GPU command imports all six production extensions, compares the
   MXFP4-v4 and NVFP4-v5 GEMMs with BF16 at fixed inputs and thresholds, and
   runs the localCTA 2D-weight reconstruction/GEMM contract. The recorded
   result is `release/gpu_gate_receipt.json`. That receipt was produced on a
   GB200 host matching the NGC-derived software contract; no container engine
   was available to independently pull the image digest, and no distributed
   64-GPU replay is claimed.

4. Bind authorized local tokenizer and dataset inputs using
   `release/external_inputs.example.json`. Keep the completed binding outside
   the checkout. Then plan a route without executing it:

   ```bash
   scripts/release/run_recipe.sh \
     --route ROUTE \
     --inputs LOCAL_INPUT_BINDING.json \
     --nnodes N --nproc-per-node N --node-rank R --master-addr HOST
   ```

   Add `--execute` only after reviewing the generated plan and providing the
   intended distributed environment. Resume uses
   `scripts/release/resume_recipe.sh` and a separately supplied compatible
   local checkpoint. No real checkpoint description belongs in this source
   repository.

## Architecture and developer map

`train.py` is the TorchTitan-based entry point. The project-specific extension
lives in `low_bits_training/`; the exact upstream TorchTitan source is vendored
in `torchtitan_submodule/`. Transformer Engine is in `TransformerEngine/`, and
all custom FP4 CUDA and ThunderKittens code is rooted at `fp4_runtime/`.
`release/source_support_matrix.json` distinguishes supported paper routes from
older MXFP4/NVFP4/localCTA kernels and TK low-precision-head experiments that
are retained as historical, unsupported reference code. The latter are not
silently presented as production routes. The output-head archive intentionally
keeps the minimal TK implementation, training integration, and public-safe
experiment drivers needed to understand the negative result; one-off private
diagnostics and operational logs are not redistributed.

The numerical route registry is split across:

- `configs/paper/llama8b_160b_recipe_contracts.json`, which records the paper
  recipes and evidence status;
- `configs/paper/route_execution.json`, which maps executable route names to
  converters, batch geometry, and environment presets; and
- `low_bits_training/reproduction/`, which enforces the Python-side route
  contracts.

To add or change a route:

1. implement the converter or dispatch in `low_bits_training/quantization/`;
2. add or change the corresponding kernel under `fp4_runtime/`;
3. give the route a new identity and explicit environment/config entry under
   `configs/paper/` rather than silently changing an existing recipe;
4. add CPU/source contracts under `tests/cpu/`, runtime source/ABI tests next
   to the kernel, and a production-shape numerical/performance gate; and
5. test both fresh initialization and full-state resume when resume is claimed.

Fresh and resumed training are deliberately separate commands. A resume must
restore model, optimizer, scheduler, data cursor, distributed geometry, and
quantization state consistently; model weights alone are not a full training
continuation.

The public source inventory is `release/components.json`. `SHA256SUMS` covers
the complete clean tree, while the component inventory records the upstream
URL and commit used for each vendored dependency plus the license files found
at that commit. These records can be checked without inherited Git metadata.

## Reproducing the paper

The manuscript source, figures, source-data tables, and figure builders live
under `docs/technical_report/`. Evaluation conversion and parity code lives in
`scripts/evaluation/` and `tools/evaluation/` when present in the source
revision. Paper configs use compiled regular cross entropy and keep route,
seed, batch geometry, and numerical format choices explicit.

Canonical downstream evaluation has a separate dependency overlay so it cannot
silently replace the training image's PyTorch or CUDA packages:

```bash
scripts/release/install_evaluation_extra.sh /absolute/path/to/eval-venv
/absolute/path/to/eval-venv/bin/python \
  scripts/evaluation/run_canonical_lm_eval.py --torchtitan-root torchtitan_submodule -- \
  run --help
```

The wrapper enforces the Llama-3.1 training-time scaled-RoPE parameters
$(8,1,4,8192)$, TorchTitan RMSNorm semantics, BF16 inference, and the math
attention backend. A standalone TorchTitan checkout must be clean at the pinned
commit. The flattened vendored source is bound by paired commit markers plus the
sealed component and whole-tree file ledgers, rather than inherited nested Git
metadata.

`release/evaluation_environment.json` records the task, shot, metric, seed,
dtype, sequence-length, and TorchTitan semantic contract. The overlay's source
archives and wheels are SHA-256 locked in
`release/evaluation_requirements.lock`; it is installed with `--no-deps` into
a fresh virtual environment using the digest-pinned training container as its
base dependency set.

Inputs that cannot be redistributed are represented only by a local binding
schema. Reproduction therefore has three distinct levels: source and numerical
contract tests, miniature functional replay with user-supplied local inputs,
and full-scale training/evaluation with the paper geometry. Do not interpret a
passed CPU gate as a full GPU or distributed reproduction. The release supports
fresh controlled execution of routes marked `current_release_route`, but no
historical long-run route is labeled bit-exact replay-ready. Recreating a
historical trajectory would additionally require its exact external data order,
compatible checkpoint state where applicable, and any historical pin that the
recipe ledger records as unrecovered.

## Development and release integrity

The vendored directories are normal source directories, not submodules. Do not
run `git submodule update`; the bootstrap and doctor verify their content
against the export ledger instead. If a dependency changes, update its explicit
pin, rebuild the clean export, and review the new license inventory and audit.

The original publication boundary was built from an explicit allowlist into a
new one-commit history. Generated scheduler objects, profiler output, compiled
extensions, logs, private operational notes, and local experiment state must
remain outside the repository.

For a subsequent release from this flattened public history, first regenerate
and review the ledgers and audit in a clean commit, then build a self-contained
snapshot:

```bash
python scripts/release/build_public_snapshot.py \
  --output ../mfu-fp4-public-snapshot
```

The command extracts tracked `HEAD`, requires strict clean-export verification,
creates a deterministic source archive and one-root/no-gitlink bundle, and then
invokes `verify_public_bundle.sh` against a disposable clone. That cold check
covers vendored-source bootstrap, CPU/source gates, the report build, and both
the arXiv and Overleaf source packages without modifying the sealed export.
