# Operand-hybrid Wgrad transform replay

This directory preserves the compact, public-safe reference result from the
layerwise replay and operand-microscope tools. It contains no tensor payload,
checkpoint metadata, storage location, job identity, or credential. Recreate
the result from separately authorized local inputs with
`tools/capture_llama_layerwise_replay.py` and
`tools/microscope_llama_quantized_operands.py`.

The SHA-256 of `reference_results.json` is
`5a2df36f92b2338f100377059253d447f0d46db67e63c5c8d1054d161e72fdac`,
identical to the sealed compact result before relocation.

## Question

The active operand hybrid changed two things relative to the completed
MXFP4+RHT run:

1. Dgrad moved from MXFP4 to localCTA row-SR.
2. Wgrad moved from fixed-sign H32 to plain H16.

This local replay isolates the second change on identical saved tensors. It
does not alter or interrupt the active long run.

## Bound inputs

- Model state: completed MXFP4+RHT trajectory at step 28,000. The model and
  capture remain external inputs and are not distributed with this repository.
- Capture geometry: 2,048 consecutive, globally aligned rows.
- Sites: FFN W2 Wgrad at layers 12, 16, and 31.
- X: ffn.w2_input_ref; dY: ffn.output_ref_grad.
- Exact production quantizer SHA-256:
  d4d4b116b703ef7ffc1986934362e7be1d6861bed1520878b9a5a1b79a420d02
- Quantization: MXFP4 E2M1 with E8M0 scale per 32 values; no data or
  scale stochastic rounding on the Wgrad column carriers.
- Fixed-sign H32 uses the production 0x2817 low-16 motif in both H32
  halves. Payload invariance across several seeds/subsequences and a decoded
  geometry oracle independently verified that contract.

For each layer and arm, the native column carriers were decoded and four
deterministic 256-by-256 Wgrad panels were formed from early, middle, and late
columns. The reference was the FP32 contraction of the original BF16 operands.
The paired unquantized transforms preserved the contraction to at most
4.32e-7 relative L2.

## Result

| Arm | Layer 12 | Layer 16 | Layer 31 | Mean |
|---|---:|---:|---:|---:|
| plain H16 (active hybrid) | 0.162313 | 0.150215 | 0.170410 | 0.160979 |
| plain H32 | 0.160509 | 0.148028 | 0.173532 | 0.160689 |
| fixed-sign H32 (completed pure run) | **0.157615** | **0.143425** | **0.162314** | **0.154452** |

Fixed-sign H32 lowers the mean sampled Wgrad relative error by 4.06% versus
plain H16. Plain H32 lowers it by only 0.18%, so block width alone does not
explain the difference. The fixed sign diagonal is the material part of the
gain, especially in layer 31: its X-carrier relative L2 falls from 0.127046
to 0.114261, whereas plain H32 rises to 0.130232.

## Interpretation

The corresponding matched 1,000-step window recorded:
steps 25,710--26,710 averaged loss 2.480890 and grad norm 0.173344, versus
2.441938 and 0.056134 for the completed pure MXFP4+RHT run. The hybrid is
1.60% higher in loss and 3.09 times higher in grad norm, while also running
about 1.9% slower.

The controlled follow-up keeps localCTA row-SR Dgrad and changes only the
Wgrad carrier from plain H16 to fixed-sign H32. A causal comparison must use
the same full model, optimizer, scheduler, data cursor, ranked SR state, batch
geometry, BF16 head, and compiled regular cross entropy.

## Limitations

- The replay used the pure MXFP4+RHT step-28k state rather than the active
  hybrid step-26k state. This limits the result to directional evidence.
- It covers FFN W2 at three layers and four fixed panels per layer, not every
  contraction.
- Quantization error is directional evidence, not proof of a training-loss
  cause. Only the matched full-state continuation pair can establish that.
