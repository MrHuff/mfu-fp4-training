# Third-party notices

This inventory covers the curated public export. It is not a legal
interpretation and does not replace the retained terms. Exact component URLs,
commits, trees, and file-ledger digests are recorded in
`release/components.json` after export.

## Vendored components

| Component | Pin | Retained terms and attribution |
|---|---|---|
| TorchTitan | `20b3de7585696c327bd5aa9f9627f0300abdbf9d` | `torchtitan_submodule/LICENSE`, `torchtitan_submodule/assets/license_header.txt` |
| Transformer Engine | `06b44b8eff1f81f33c2a378515cf05fe2fade3cb` | `TransformerEngine/LICENSE`, `TransformerEngine/Acknowledgements.txt` |
| cuDNN Frontend headers | `97f6cb3b88cacff507cca1280db5650a457d92b3` | `TransformerEngine/3rdparty/cudnn-frontend/LICENSE.txt` and nested retained license files |
| CUTLASS headers used by Transformer Engine | `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` | `TransformerEngine/3rdparty/cutlass/LICENSE.txt`, `CONTRIBUTORS.md` |
| ThunderKittens | `2dd00a943309984ab7c1434a5614f2a63efd933d` | `fp4_runtime/ThunderKittens/LICENSE` |

The export does not vendor the runtime's unrelated SageAttention, top-level
CUTLASS, FlashAttention, QUTLASS, FA4, or Mamba CUDA trees.

## Inherited and adapted source

- The training source descends from the Graphcore Research `gc-training`
  repository. Files carrying a Graphcore Ltd. copyright notice remain under
  the Apache License, Version 2.0 that governed that source lineage. The exact
  retained terms are at `LICENSES/APACHE-2.0.txt`. The root MIT License applies
  only to separable project-original work for which the copyright holder is
  authorized to grant MIT rights.
- TorchTitan-derived files under `low_bits_training/` retain their copyright
  notices and are governed by `torchtitan_submodule/LICENSE` where applicable.
- `low_bits_training/analysis/stream_checkpoints.py` identifies copied PyTorch
  code. The exact PyTorch terms are retained at `LICENSES/PYTORCH.txt`.
- `low_bits_training/components/tiktoken_tokenizer.py` refers to the Meta
  Llama 3 Community License. The license and required attribution are retained
  at `LICENSES/META_LLAMA_3.txt` and `LICENSES/META_LLAMA_3_NOTICE.txt`. The
  public README and repository NOTICE display the required “Built with Meta
  Llama 3” statement.
- The NVIDIA-copyrighted files under
  `fp4_runtime/TK_quantisation/nvfp4*` derive from the Apache-2.0 Transformer
  Engine NVFP4 implementation and were modified for the standalone FP4
  quantizers and successive experimental generations. Their NVIDIA notices
  are retained, their modification is recorded here and in
  `fp4_runtime/LICENSE`, and their applicable terms are retained at
  `LICENSES/APACHE-2.0.txt`.
- `fp4_runtime/fused_ops/csrc/elementwise_mul.cu` and
  `fp4_runtime/fused_ops/csrc/fused_silu_rmsnorm_backward.cu` retain Google
  DeepMind copyright and Apache-2.0 identifiers. The retained terms are at
  `LICENSES/APACHE-2.0.txt`.
- `fp4_runtime/fused_ops/csrc/utils.cuh` and
  `fp4_runtime/fused_ops/csrc/vec.cuh` retain IST Austria and Erik Schultheis
  copyright and Apache-2.0 identifiers. The retained terms are at
  `LICENSES/APACHE-2.0.txt`.
- `low_bits_training/models/nemotron_h_hf/configuration_nemotron_h.py` and
  `low_bits_training/models/nemotron_h_hf/modeling_nemotron_h.py` retain their
  AI21 Labs, Hugging Face, and NVIDIA notices and their Apache-2.0 headers. The
  retained terms are at `LICENSES/APACHE-2.0.txt`.
- `low_bits_training/quantization/cuda_ops/gemm_evaluator.cu` retains its
  inline NVIDIA copyright and BSD-style terms.

## Corporate mark

`docs/technical_report/assets/graphcore-symbol.png` is the Graphcore corporate
mark used to identify the research organization on the report. It is not
covered by the repository's MIT or CC BY 4.0 licenses. Graphcore and the
Graphcore mark remain the property of Graphcore Ltd.

The OpenAI Simple Evals derivative and experimental StableSPAM derivative are
excluded from the public export.

No statement in this file grants additional rights or relicenses third-party
material.
