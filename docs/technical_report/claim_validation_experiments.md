# FP4 Report Claim-Validation Experiments

Date: 2026-08-27

> **Frozen historical backlog.** This document is not a live queue or current
> result ledger. As of 2026-08-31, use
> `data/experiment_recipe_ledger_20260831.csv`,
> `data/shared_base_training_recipe_20260831.csv`, and `data/SHA256SUMS` for
> every experiment cited by the Llama-8B paper. Canonical downstream scoring
> and both completed RHT trajectories postdate this backlog. Nemotron E06 is
> historical and outside the current Llama-only report.

This ledger lists the experiments still owed before the technical report's
preliminary claims can be stated as publication conclusions. It is organized
around claims, not kernels. A fast microbenchmark is evidence for a mechanism;
it is not evidence for model throughput, convergence, or time-to-target.

## Evidence rules

Every reported run must record:

- low-bits-training and kernel repository SHAs, dirty state, build flags, CUDA
  version, driver version, and Python environment;
- GPU UUID, device index, SM clock samples, temperature, power, throttling
  flags, and other active CUDA processes;
- exact model, sequence length, micro/global batch, gradient accumulation,
  data shard, seed, optimizer state, and graph-capture setting;
- route census for every custom dispatcher, including fallback counts and
  production shapes;
- warmup interval, scored interval, raw step times, tokens/s, MFU convention,
  peak-memory allocation, loss, gradient norm, and non-finite counts; and
- all candidate and control logs, including failed and negative runs.

For model speed decisions, use at least five fresh processes per arm, rotate
execution order, score the predeclared steady window, and report mean, median,
standard deviation, and a paired 95% confidence interval. Do not select the
fastest GPU after seeing candidate results. Select a healthy device by a
candidate-independent clock/thermal screen and run all arms there.

For production-shape numerical tests, report finite counts, zero fraction,
max absolute error, relative L2, relative RMS, cosine similarity, scale error,
and clipping/saturation fraction. Cosine alone is insufficient.

## Priority summary

| ID | Priority | Claim being tested | Current state |
|---|---|---|---|
| E01 | P0 | Characterize MXFP4, CTA-local NVFP4, and global NVFP4 on Llama 8B | W32 GB200 route panel complete; strict all-route pairing still owed |
| E02 | P0 | Tensor-wide amax is a material, removable scaling cost | 1.47-8.07 MFU route proxy; pure-amax estimate |
| E03 | P0 | CTA-local is numerically valid and removes global coordination | One repaired 160B seed complete; repeats/downstream/factorial owed |
| E04 | P0 | Native FP4 reduces Llama 8B time-to-target | Selected 160B curves complete; FP8, downstream evaluation, and active-time accounting pending |
| E05 | Closed | A low-precision output head improves end-to-end training | Rejected: selective lowp candidates lose to matched native BF16 and fail numerics; compiled BF16 promoted |
| E06 | P0 | Nemotron-H fusions deliver the claimed format ordering | Preliminary |
| E07 | P1 | RHT, SR, and 2D scaling costs and quality benefits are understood | Step-38k W32 RHT pairs in resume-gate repair; full factorial owed |
| E08 | P1 | MXFP4 scale rounding direction improves the range/precision tradeoff | Mechanism implemented |
| E09 | P1 | Retained CODA and boundary fusions stack in full models | Individual short gates |
| E10 | P1 | Grouped quantization/GEMM and backward fusions own complete work | Partial |
| E11 | P1 | Numerical behavior remains stable across difficult distributions | Partial |
| E12 | P1 | Mixed MXFP4/CTA-local layer policies beat pure routes | Llama CTA/MX complete but win unproven; v5/MX diverged and is rejected |
| E13 | P2 | Single-GPU gains survive distributed execution | Partial: W32 GB200 panel complete; 72-GPU gate owed |
| E14 | P2 | FP4 reduces accelerator-hours and energy-to-target | Not measured |

The first distributed claim screen was submitted on 2026-07-28 as twelve
matched 32-GPU, single-rack jobs. It covers all six body formats for Llama 8B
and Nemotron-H while holding the native NVFP4 probability-cache output layer
constant. Four `r0` arms completed valid 25-step screens. Eight corrected
retries are queued after infrastructure, runtime-version, allocator, and
faulty-device failures invalidated their first attempts. Job IDs, immutable
inputs, results, failure classifications, and the decision gate are recorded
in `docs/fp4_32gpu_prod_claim_matrix_2026_07_28.md`. These screens are a
prerequisite for, not a replacement for, the repeated and long-run experiments
below.

The August campaign supersedes that operational snapshot for Llama 8B. Exact
W32 GB200 aggregate receipts now cover BF16, TE-native, full TE, F0L4, MXFP4,
CTA-local v4, CTA-local/MXFP4, and global v5. Multiple 160B-token trajectories
also reached step 38,147. Those curves close stability questions for the
observed seed but do not replace seed-controlled downstream evaluation.

## E01: Matched Llama 8B format ranking

**Claim.** On the production Llama 8B shape, fused MXFP4 is fastest,
CTA-local NVFP4 is second, and globally scaled NVFP4 v5 is third.

**Historical arms.**

1. Native BF16 reference at its largest validated resident batch.
2. Fused MXFP4 production policy.
3. CTA-local NVFP4 v4 production policy.
4. Global NVFP4 v5 production policy.
5. Original TE NVFP4 recipe as an external-system reference.

**Historical procedure.**

- Use one preselected healthy GB200 and the same model/data checkpoint.
- Run five fresh A/B/C rotations and reverse the order across rotations.
- Use 25 training steps with steps 6-25 scored, then repeat the full comparison
  with 100 steps to check warmup sensitivity.
- Prove all intended forward, Dgrad, Wgrad, RMS, FFN, QKV, and output-layer
  routes fire. Fail the run if any required route falls back.
- Repeat once on a second preselected GPU to estimate device dependence.

**Required result.** Report tokens/s and MFU with paired intervals. The strict
ordering may be claimed only if each adjacent difference remains positive in
both device blocks. Otherwise report overlapping performance bands.

## E02: Isolate the global tensor-scaling tax

**Claim.** The global amax reduction and completion barrier are a material
part of NVFP4 quantization cost, and CTA-local scaling removes that cost.

**Arms for each production tensor.**

1. Current global NVFP4 quantizer, including amax.
2. The same quantizer given the exact precomputed global scale from arm 1;
   only the timed amax/reduction is removed.
3. The same payload/layout kernel with a fixed neutral scale, used only to
   separate scale arithmetic from memory/layout cost.
4. CTA-local v4 quantizer and consumer GEMM.

**Shapes.**

- Llama 8B QKV forward and backward at M=32,768.
- QKV backward stress shape M=65,536.
- FFN W1/W3 and W2 forward, Dgrad, and Wgrad at
  (M,K,H)=(32,768,4,096,14,336).
- Nemotron-H Mamba input projections at their exact 24 production shapes.

**Measurements.**

- CUDA-event latency for reduction, payload production, scale swizzle, and
  consuming GEMM, with 100 warmups and 1,000 interleaved samples;
- Nsight Systems kernel timeline and dependency edges;
- Nsight Compute DRAM bytes, barrier stalls, achieved occupancy, registers,
  and Tensor Core utilization;
- numerical identity between arms 1 and 2, because they consume the same
  captured scale; and
- a full-model A/B test with the measured scale-path change enabled.

**Required result.** Replace the current 0.28 ms estimate with a measured
delta. The report may attribute the delta to global amax only if arms 1 and 2
produce identical payloads/scales and differ only by the timed reduction.

## E03: CTA-local scale-contract proof

**Claim.** CTA-local v4 is a valid hierarchical NVFP4 scaling strategy, not a
numerical shortcut, and its K-invariant two-sided outer scales are compatible
with the tiled GEMM epilogue.

**Tests.**

- Quantize A and B with deliberately non-neutral, different outer scales.
- Include Gaussian, real activation, heavy-tail, sparse, all-zero, tiny
  backward-gradient, and one-hot-outlier tensors.
- Compare the decoded operands and GEMM result against BF16 and global NVFP4.
- Verify alpha_i and beta_j are constant across the full K reduction.
- Verify the epilogue applies alpha_i * beta_j exactly once after accumulation.
- Explicitly test the historical failure case where folding the outer factor
  into E4M3 underflows tiny blocks; this is a regression test, not a candidate.
- Cover K tails and all production M/N tile boundaries, including the final
  partial tile.

**Training gate.**

- Three 500-step 1.2B seeds against global NVFP4 with matched numerical extras.
- At least one 10B-token Llama 8B continuation from common optimizer state.
- Compare loss versus tokens, gradient distributions, clipping, downstream
  checkpoint metrics, and scale histograms.

**Required result.** No non-finite values, no unexplained zero inflation, and
no systematic shrinkage. Long-run quality must be reported as a confidence
interval, not inferred from one 25-step speed run.

## E04: Llama 8B time-to-target

**Claim.** The selected native FP4 training stack reaches a fixed quality
target in less wall time than BF16, FP8, and the original TE NVFP4 recipe.

**Arms.**

1. Pure BF16 reference.
2. Production FP8 reference.
3. Original TE NVFP4 recipe.
4. Fused MXFP4 body plus regular BF16 output layer.
5. Pure CTA-local NVFP4 v4 plus regular BF16 output layer.
6. Global NVFP4 v5 plus regular BF16 output layer.

**Protocol.**

- Predeclare validation perplexity targets before reading final curves.
- Use the same tokenizer, Dolma data order, optimizer, schedule in tokens,
  global batch, initialization, and evaluation head.
- The completed campaign trains to 160B tokens; evaluate every eligible route
  at the exact step-38,000 common cut before extrapolating beyond that horizon.
  Historical BF16 and TE-native are exact-step matched at that cut, but label
  their unknown seeds and 32,768-token stored-counter offset from the newer
  routes.
- Evaluate frequent enough checkpoints to bound target-crossing time.
- Use at least three 1.2B seeds to estimate recipe variance. For the expensive
  8B study, run one full seed per arm plus independent shorter continuations
  from at least two common mid-training checkpoints.

**Required report.**

- tokens-to-target, active wall time, end-to-end tokens/s, steady tokens/s,
  optimizer steps, restarts, checkpoint/evaluation time, and GPU-hours;
- the convergence tax tau and throughput gain g;
- time ratio (1+tau)/(1+g); and
- the NVIDIA-aligned MMLU, HellaSwag, WinoGrande, and ARC-Challenge panel once
  conversion parity is proven, followed by OLMES core-9 where valid
  checkpoints exist; report exact common-cut and wall-time comparisons
  separately and retain the historical-seed/counter caveats.

## E05: Low-precision output-layer gate — closed negative

**Claim tested.** A low-precision forward head with selective higher-precision
backward preserves useful training quality and produces a net end-to-end win.

**Outcome.** Rejected for the tested designs. The matched native BF16/BF16
reference reached 34,001.64 tokens/s/GPU. Low-precision forward with BF16
`dHidden`/`dWeight` reached 33,946.39 (-0.1625%) and failed the loss gate;
using MXFP8 `dWeight` reached 33,909.50 (-0.2710%) and failed the loss/gradient
gate. In a separate matched gate, compiled regular CE reached 33,712.83,
within 0.2266% of its own 33,789.41 native-BF16 reference and 37.47% faster
than the slow wrapper. Absolute throughput must not be compared across those
two gates. All promoted long runs use the compiled BF16 head.

The historical protocol below records what was considered before this
whole-trainer adjudication. It must not be presented as pending production
work or as evidence that a native FP4 head was promoted.

**Historical arms.**

1. Native BF16 forward and backward output layer.
2. Fully MXFP4 v4 forward and backward with direct G-cache.
3. Fully NVFP4 v4 forward and backward with direct G-cache.

**Historical procedure.**

- Run the three arms for 100B Dolma tokens with a matched BF16 Llama 1B body,
  four GPUs, global batch 512, sequence length 4096, and seed 42.
- On every 50th step, evaluate the exact same hidden states, head weights, and
  labels through the native BF16 head and log `eval_bf16/loss` alongside the
  training-head loss. This paired score is no-grad and does not alter the FP4
  backward path.
- Repeat production-shape isolated numerics for real activations spanning
  early, middle, and late checkpoints.
- Promotion was conditional on the common BF16 evaluation curve; the later
  whole-trainer speed result failed first and stopped promotion.
- Measure loss, hidden-gradient error, weight-gradient error, saturation,
  zero fraction, throughput, and time-to-target.

**Decision.** No break-even token tax is assigned because the tested selective
low-precision arms have negative throughput gain. The 1,000-step P-cache
trajectory remains historical supporting evidence only.

## E06: Nemotron-H 8B fusion and format matrix

**Claim.** Hybrid-specific fusions make Nemotron-H well fused, and the expected
format ordering holds after Mamba paths are included.

**Ablations.**

- RMSNorm fused into all 24 Mamba input projections on/off.
- W1 overlap on/off at identical memory limits.
- Selective-state-space backward grouped/native baseline.
- Prefix launch removal and input-staging copy removal.
- Convolution-gradient/output-projection overlap on/off.
- MXFP4, CTA-local NVFP4, and global NVFP4 for every ablation.

**Procedure.** Use the E01 same-GPU order-rotated protocol, plus 500-step runs
for loss/gradient stability. Record peak memory because retaining W1 overlap is
part of the claimed gain. Include the largest-resident-batch BF16 model control
in every reported Nemotron comparison table.

**Required result.** Attribute the reported approximately 7.5 MFU recovery to
a reproducible on/off comparison. Report tokens/s as primary; do not compare
Nemotron MFU with Llama until the Nemotron FLOP model is corrected.

## E07: Full recipe factorial for RHT, SR, and 2D scaling

**Claim.** Numerical extras have measurable costs, and their quality benefit
can be priced individually.

**Design.**

- Run the 2 x 2 x 2 factorial for RHT, SR, and 2D weight scaling in TE NVFP4.
- Reproduce the same logical factorial in global v5 where an exact native
  implementation exists.
- Separate forward RTNE, backward SR, and update SR rather than using one
  global rounding toggle.
- Add the NVIDIA selective-higher-precision policy as a separate factor, not
  as part of "FP4."

**Measurements.**

- isolated transform, RNG, scale, and layout latency;
- full 500-step 1.2B MFU/tokens/s for all cells;
- three-seed loss versus tokens through a meaningful continuation horizon;
- clipping, quantization bias, gradient-noise scale, and Wgrad error; and
- interaction terms, because RHT can change the value of 2D scaling and SR.

**Required result.** Replace the aggregate -4.09 MFU result with main effects
and interactions. Do not claim that SR or RHT is "free" from fake-quantization
quality studies.

## E08: MXFP4 encode/decode scale-direction ablation

**Claim.** Choosing the E8M0 exponent direction by tensor role is better than
blind maximum-based scaling or universal Half-S.

**Policies.**

1. ceil(log2(s*)) for clipping-safe encoding;
2. round-to-nearest exponent;
3. floor(log2(s*)) for a denser decode grid;
4. universal one-step Half-S;
5. adaptive direction selected from a predeclared clipping/bias criterion.

**Stages.**

- Quantization-only tests on real weights, activations, activation gradients,
  and weight gradients from ten checkpoints.
- Production GEMM output/gradient tests for every operand role.
- Three 500-step 1.2B seeds for all viable policies.
- At least two common-checkpoint Llama 8B continuations for the finalists.

**Measurements.** Scale exponent histogram, saturation fraction, underflow and
zero fraction, MSE, signed bias, relative RMS, GEMM output error, loss versus
tokens, and throughput. Report results separately for Fprop, Dgrad, and Wgrad.

**Required result.** The report may prefer downward scale rounding only for
tensor roles where it improves quality or time-to-target without unacceptable
clipping. A mechanism-level similarity to Half-S is not outcome evidence.

## E09: CODA and boundary-fusion interaction matrix

**Claim.** Retained CODA-inspired kernels improve the complete model and stack
without hidden negative interactions.

**Factors.**

- exact W2-to-next-QKV residual/RMS carrier;
- same-layer attention-output-to-W13 carrier;
- packed Q/K RoPE and strided-Q handoff;
- fused backward sum/RMS carrier;
- v5 fused SwiGLU backward producer;
- CTA-local fused SiLU row/column overlap; and
- regular compiled BF16 output head (the rejected probability cache is kept as
  a negative control only).

**Procedure.**

- Run off, each factor alone, the retained stack, and leave-one-out stack arms.
- Perform this separately for MXFP4, CTA-local v4, and global v5.
- Use production-shape microbenchmarks, a route census, and five paired
  Llama-8B processes per arm.
- Repeat the applicable factors on Nemotron-H.

**Required result.** Report main effects and stack interaction terms. Preserve
negative results such as direct W1/W3/W2 FP4 epilogue payloads and 2D tile RMS
as negative controls; do not silently substitute a PyTorch or Triton path.

## E10: Grouped quantization/GEMM and backward ownership

**Claim.** Grouped native owners reduce traffic and launch cost only when they
finish all layouts and preserve useful overlap.

**Tests.**

- grouped versus independent QKV quantizers;
- grouped QKV Wgrad versus independent Wgrads;
- grouped Dgrad versus independent Dgrads for balanced and unbalanced splits;
- fused sum3 versus two additions;
- fused derivative plus row/column FP4 production versus split producers; and
- graph-captured and eager variants with stable preallocated workspaces.

**Measurements.** Kernel count, launch latency, overlap timeline, allocation
time, TMA-map creation, DRAM bytes, occupancy, full boundary time, and model
time. Verify Philox state advances across graph replays.

**Required result.** A grouped path is retained only if it wins the complete
boundary and model. Reduced Python call count is not sufficient evidence.

## E11: Numerical stress and regression suite

**Claim.** All promoted CUDA/TK routes are robust across production values and
do not rely on neutral scales or accidental buffer lifetime.

**Coverage.**

- MXFP4, global v5, and CTA-local v4;
- forward, Dgrad, Wgrad, residual/RMS, FFN derivative, QKV/RoPE, Mamba
  projection, and output layer;
- eager execution, CUDA Graph capture, recomputation, and at least 100 graph
  replays;
- pointer reuse, workspace rotation, stream-order reversal, K/M/N tails, and
  all-zero/tiny/heavy-tail inputs; and
- non-neutral A and B outer scales.

**Required result.** No fallback, stale-pointer dependence, repeated stochastic
rounding sequence, non-finite output, or unexplained scale-group mismatch.
Numerical thresholds must be operator-specific and recorded in one manifest.

## E12: Mixed MXFP4 and CTA-local layer placement

**Claim.** Range-sensitive layers can use MXFP4 while the rest use faster
CTA-local NVFP4, improving time-to-target over either pure route.

**Policies.**

- pure MXFP4;
- pure CTA-local NVFP4;
- final 4, 8, and 16 layers in MXFP4;
- output layer held constant as the promoted regular BF16 linear plus compiled
  cross entropy;
- sensitivity-selected layers based on predeclared scale/outlier statistics;
  and
- a random layer assignment with the same MXFP4 count as a control.

**Measurements.** Throughput, peak memory, per-layer clipping/scale histograms,
loss versus tokens, tokens-to-target, and downstream metrics.

**Required result.** Selection must be defined before final evaluation and beat
both the pure-route control and count-matched random placement.

## E13: Rack-scale translation

**Claim.** Single-GPU kernel wins survive distributed 72-GPU training.

**Current evidence.** A matched four-GPU Llama bracket retained no-forward-
reshard plus depth-one prefetch and completed without OOMs. The two
automatic-clock controls averaged 107.819 MFU. Depth-two prefetch was
0.276 MFU slower, and a privileged 2,062 MHz request was 0.195 MFU slower.
The locked arm still fell to 1,455--1,620 MHz under approximately
1.16--1.18 kW load, so the provider power policy overrode the request. A
different allocation reached 113.63 MFU with the same software policy. The
refreshed 32-GPU matrix must therefore record all-rank clocks and must not
attribute unpaired allocation differences to code.

**Procedure.**

- Run identical 72-GPU canaries for every finalist.
- Measure compute, collective, optimizer, checkpoint, and input-pipeline time
  separately.
- Compare CUDA/TK route counts per rank and detect stragglers.
- Run at least 500 steady steps after initialization and one checkpoint cycle.

**Required result.** Report aggregate and per-GPU tokens/s, scaling efficiency,
step-time percentiles, communication overlap, straggler spread, and failures.
A single successful four-GPU canary proves functionality, not rack efficiency.

## E14: Accelerator-hours and energy-to-target

**Claim.** Native FP4 is economically faster, not only higher-MFU.

**Procedure.**

- Record active wall time, allocation size, average board power, restarts,
  failed-step work, checkpoints, and evaluations for E04.
- Integrate energy over the active training interval.
- Keep queue delay separate but report it for operational planning.

**Required result.** Publish GPU-hours and kWh to each quality target alongside
MFU and tokens/s. A recipe should be called "faster training" only when
time-to-target improves; it should be called "more efficient" only when the
declared resource metric also improves.

## Publication claim checklist

Before freezing the paper:

- [ ] Every whole-model comparison table reports its BF16 tokens/s and MFU.
- [ ] E01 format ranking completed on two preselected GPUs.
- [ ] E02 global-amax cost measured, replacing the current estimate.
- [ ] E03 long-run CTA-local numerical gate completed.
- [ ] E04 at least one full Llama 8B time-to-target comparison completed.
- [x] E05 current output-layer designs closed as a negative whole-trainer
  result; compiled BF16 promoted.
- [ ] E06 Nemotron-H per-fusion attribution completed.
- [ ] E07 recipe main effects separated.
- [ ] E08 MXFP4 scale-direction result completed or moved to future work.
- [ ] E09 retained CODA stack interaction matrix completed.
- [ ] E13 rack-scale throughput reported for the promoted recipe.
- [ ] All plots regenerated from checked-in raw summaries.
- [ ] Every number in the abstract maps to a log and immutable commit.
