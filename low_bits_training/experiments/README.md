# Experiments space

Similarly to TorchTitan [`experiments`](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments) (but with a wider scope), this space is dedicated to experimental code/features/research projects, taking advantage of `low_bits_training` infrastructure, but not mature or stable enough to be added to the core repository.

## What makes a feature experimental?

- Are submodules/subpackages of this folder.
- Cannot be imported in the main code paths of the repository: imports need to be behind JobConfig `--job.experimental_modules=module_xyz` argument.
- The feature may be weakly tested and not compatible with other features.

If you experimental modules requires additional or different dependencies, please add a new requirement variant in the `pyproject.toml`.

## Experimental sub-modules

Every sub-module in the space should have a clear owner, responsible of the maintainance & test coverage. Please update this table and [`CODEOWNERS`](../../CODEOWNERS) at every sub-module added.

| Sub-module  | Owner | Description |
| ----------- | ------- | --------  |
| `mxfp8`    | Paul Balanca (@balancap)  | MXFP8 training experiments (and addons on top of TorchAO). |
| `mx_norm`  | Luke Prince & Callum McLean | MXNorm research project. |
| `umup`     | Luke Prince  | u-MuP research project. |
| `mxfp4`     | Robert Hu  | FP4 resaerch project. |
| `liger`     | Emily Schmidt | Liger kernels integration. |
