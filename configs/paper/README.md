# Paper recipe contracts

`llama8b_160b_recipe_contracts.json` is the machine-readable specification for
the principal long-training routes in the FP4 training-systems paper. It was
transcribed from the canonical experiment recipe ledger at report commit
`b1c0570f0dcb8c03e8cceeb5e8d227e12651bc67`.

The numerical ledger remains an evidence inventory; the public execution
contract is now split into three reviewable pieces:

- `executable/llama3_1_8b_160b.toml` fixes the common model, optimizer,
  scheduler, batch, ordinary BF16 head, compiled regular loss, and checkpoint
  policy without embedding an input or output location;
- `env/*.env` fixes the route-specific kernel, rounding, 2D-weight, fusion,
  and transform controls;
- `route_execution.json` binds a route ID to its converter order, environment,
  world size, local batch, gradient accumulation, and launch/resume policy.

The release source includes the route-complete lineage through
`175d2af248e8ee575805aca57875cda0ee7ae51f` and pins the paired runtime at
`301ab63d354a4f8c24b7c0da499736e3f14b7400`. Several recorded executions used
other source/runtime commits or sealed overlays. The contract preserves those
facts instead of silently mapping every route to the current checkout.

The public release entry point is the repository [`README.md`](../../README.md). Data,
tokenizers, and checkpoints are local external inputs bound through
`release/external_inputs.schema.json`; the repository must never contain their
private locations, credential values, checkpoint metadata, or cluster jobs.

## Reading the contract

The top-level `common_recipe` contains settings shared by the full-horizon
campaign. Each route then records only route-specific format assignment,
rounding/transform behavior, batch geometry, loss path, lineage, and evidence
status.

`reproduction_level` means:

- `recorded_lineage_recipe`: material source/runtime lineage is recorded, but
  exact end-to-end replay is not yet packaged.
- `recipe_only`: the scientific recipe is known, but a critical exact
  historical code pin is missing.
- `partial_attempt`: the full training horizon or evaluation panel is absent.
- `withheld_invalid`: the route failed scientifically and is listed only to
  prevent accidental reuse.

Null values are deliberate. Do not replace them with the current branch or a
nearby run without creating a new experiment identity.

## External inputs and execution

The repository deliberately contains no dataset objects, credentials,
checkpoint bytes or metadata, private storage locations, or cluster jobs.
Supply local authorized inputs using `release/external_inputs.schema.json`.
The release launcher verifies their SHA-256 bindings and combines them with a
route without printing their contents:

```bash
scripts/release/run_recipe.sh \
  --route mxfp4-v4-row-sr-h32-rht-long \
  --inputs /local/private/input-binding.json \
  --nnodes 8 --nproc-per-node 8 --node-rank 0 \
  --master-addr training-rendezvous
```

The default is a non-mutating plan. Add `--execute` on every scheduler-launched
node only after reviewing the plan. Use `resume_recipe.sh` with an external
full-state checkpoint binding for continuation.

An executable current-release route binds and verifies:

- a public source commit, runtime commit, and every kernel/submodule commit;
- an immutable tokenizer revision and dataset manifest with deterministic
  shuffling/sharding and cursor semantics;
- the exact converter ordering, route state, stochastic-rounding seed and
  subsequence, transform mask, and weight orientation;
- world size, local batch, gradient accumulation, activation checkpointing,
  and FSDP topology;
- ordinary BF16 output head and the recorded cross-entropy implementation;
- checkpoint interval, full-state resume behavior, and output identity;
- canonical validation and downstream-evaluation semantics.

The existing files under `train_configs/nvblog_llama3_8b` are 500-step research
proxies. They are not aliases for these 38,147-step contracts. A controlled
rerun of a historically incomplete lineage is executable, but is explicitly
not labeled a bit-exact replay of that old trajectory.

## Validation

Syntax and basic invariants can be checked without a GPU:

```bash
python -m json.tool configs/paper/llama8b_160b_recipe_contracts.json >/dev/null
python - <<'PY'
import json
from pathlib import Path

p = Path("configs/paper/llama8b_160b_recipe_contracts.json")
d = json.loads(p.read_text())
assert d["common_recipe"]["optimizer_updates"] == 38147
assert d["common_recipe"]["tokens_per_update"] == 4194304
assert d["common_recipe"]["scheduled_tokens"] == 160000114688
assert len({r["id"] for r in d["routes"]}) == len(d["routes"])
assert not any(r["reproduction_level"] == "exact_replay_ready" for r in d["routes"])
PY

python -m json.tool configs/paper/route_execution.json >/dev/null
python -m pytest -q tests/release
```
