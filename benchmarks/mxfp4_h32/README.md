# MXFP4-v4 fixed-sign H32 performance benchmark

This cluster-neutral harness reproduces the scientific contract of the
world-32 speed probe without carrying its Kubernetes objects, storage paths,
job identities, credentials, dataset payload, or checkpoint metadata.

The route uses MXFP4-v4 for all 32 transformer blocks, custom 2D weight
quantization, and row-oriented stochastic rounding for gradient data. A
paired, fixed-sign H32 transform is applied only to the activation and dY
column copies consumed by Wgrad. The row copies consumed by Fprop and Dgrad
remain untransformed. Weights are not Hadamard transformed. The output linear
and loss gradient remain BF16, and regular cross entropy is compiled with
Inductor.

`MXFP4_RHT_RANDOM_SIGN_MASK=1` is the historical selector name. In the pinned
runtime it selects the deterministic, correlated `0x2817` sign diagonal; it
does not draw a new random mask. `orientation_gate.py` proves that the row
copies remain byte-identical while the paired Wgrad columns change.

## Files

- `benchmark.json`: portable scientific and geometry contract.
- `route_contract.py`: fail-closed environment, batch, producer-state, and
  converted-model checks.
- `low_bits_training/experiments/mxfp4_h32_benchmark_contract.py`: installs
  those checks at conversion/state-construction time and proves the ordinary
  BF16 output head before and after FP32-master conversion.
- `orientation_gate.py`: native GPU check for row/column operand ownership.
- `render.py`: binds local external inputs and emits a hash-sealed effective
  plan outside the checkout. It never emits a cluster Job.
- `runner.py` and `launcher.py`: execute the plan with standard `torchrun`;
  no private launcher is patched.
- `memory_guard.py`: optional local-GPU memory monitor.
- `analyze.py`: strict per-node and cross-node metric reduction.

## Reproduce the probe

Initialize and build the pinned submodules first (see `START_HERE.md`). Keep
model assets and the packed-token dataset outside the repository. Then render
one effective plan per shared run:

```bash
python -m benchmarks.mxfp4_h32.render \
  --model-assets /local/authorized/model-assets \
  --dataset /local/authorized/packed-token-dataset \
  --output-dir /local/results/mxfp4-h32 \
  --plan /local/results/mxfp4-h32/effective-plan.json
```

On each node, provide its zero-based rank and the same rendezvous address:

```bash
python -m benchmarks.mxfp4_h32.runner \
  --plan /local/results/mxfp4-h32/effective-plan.json \
  --node-rank 0 --master-addr coordinator.example --master-port 29500
```

Use `--dry-run` first to inspect the exact command. Authentication, remote
staging, scheduling, and log transport are intentionally outside this tool.
The output directory must not be inside the Git checkout, which prevents
machine-specific paths and result payloads from being committed accidentally.

For the reported geometry, collect one node log from each of eight nodes and
create strict receipts:

```bash
python -m benchmarks.mxfp4_h32.analyze node \
  --log node-0.log --run-id example --route mxfp4-h32 --node-rank 0 \
  --node-count 8 --local-processes 4 --final-step 100 \
  --steady-start 40 --steady-end 100 --world-size 32 \
  --local-batch 4 --gradient-accumulation 4 --global-batch 512 \
  --source-sha256 SOURCE_SHA --route-contract-sha256 CONTRACT_SHA \
  --output node-0.json

python -m benchmarks.mxfp4_h32.analyze aggregate \
  --node-summary node-0.json --node-summary node-1.json \
  --node-summary node-2.json --node-summary node-3.json \
  --node-summary node-4.json --node-summary node-5.json \
  --node-summary node-6.json --node-summary node-7.json \
  --output aggregate.json
```

The original 33 MiB `tokens.bin`, its environment-specific metadata, rendered
Job JSON, private storage prefixes, UIDs, and historical run identities are
deliberately excluded. Supply or generate a compatible local packed-token
dataset. Performance comparability additionally requires equivalent GPU,
software, clocks, topology, and communication configuration.
