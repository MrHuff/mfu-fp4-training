#!/usr/bin/env bash
set -euo pipefail

# Whole-training-step MFU tax benchmark for the final CCE backend.
# Holds the NVIDIA-paper 1.2B MXFP4 high-water route fixed and only swaps the
# final internal-loss CCE backend.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
cd "${repo_root}"

timestamp=$(date +%Y%m%d_%H%M%S)
gpu="${GPU:-3}"
steps="${STEPS:-50}"
run_root="${OUT_ROOT:-/tmp/lbt_finalcce_mfu_tax_${timestamp}}"
variants_csv="${VARIANTS:-nvfp4-v4,torch-compile-bf16,mxfp4-v4}"
dataset_kind="${DATASET_KIND:-packed-bin}"
dataset_path="${DATASET_PATH:-/local_nvme/lbt_packed/slimpajama_20b_tokens}"
dataset_kwargs="${DATASET_KWARGS:-{\"num_workers\":8,\"prefetch_factor\":4,\"pin_memory\":false,\"repeat\":false,\"require_full_run\":true}}"

mkdir -p "${run_root}"

IFS=',' read -r -a variants <<< "${variants_csv}"
for raw_label in "${variants[@]}"; do
  label=$(printf '%s' "${raw_label}" | xargs)
  case "${label}" in
    nvfp4-v4)
      backend=nvfp4
      implementation=v4
      quant_mode=enc
      ;;
    mxfp4-v4)
      backend=mxfp4
      implementation=v4
      quant_mode=enc
      ;;
    torch-compile-bf16)
      backend=torch_compile_bf16
      implementation=v2
      quant_mode=enc
      ;;
    *)
      echo "Unknown VARIANTS entry: ${label}" >&2
      echo "Supported: nvfp4-v4, mxfp4-v4, torch-compile-bf16" >&2
      exit 1
      ;;
  esac

  out_dir="${run_root}/${label}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  STEPS="${steps}" \
  OUT_DIR="${out_dir}" \
  FP4_CCE_BACKEND="${backend}" \
  FP4_CCE_IMPLEMENTATION="${implementation}" \
  FP4_CCE_QUANT_MODE="${quant_mode}" \
  tools/run_mxfp4_highwater_repro.sh \
    --training.dataset "${dataset_kind}" \
    --training.dataset-path "${dataset_path}" \
    --training.load-dataset-kwargs "${dataset_kwargs}" \
    --debug.seed 1234
done

python - "${run_root}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
ansi_re = re.compile(r"\x1b\[[0-9;]*m")
step_re = re.compile(
    r"step:\s*(?P<step>\d+).*?"
    r"loss:\s*(?P<loss>[-+0-9.]+).*?"
    r"tps:\s*(?P<tps>[0-9,]+).*?"
    r"tflops:\s*(?P<tflops>[0-9,.]+).*?"
    r"mfu:\s*(?P<mfu>[-+0-9.]+)%"
)

rows = []
for log_path in sorted(run_root.glob("*/train.log")):
    metrics = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = step_re.search(ansi_re.sub("", line))
        if not match:
            continue
        metrics.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "tps": float(match.group("tps").replace(",", "")),
                "tflops": float(match.group("tflops").replace(",", "")),
                "mfu": float(match.group("mfu")),
            }
        )
    if not metrics:
        rows.append({"variant": log_path.parent.name, "status": "NO_METRICS", "log": str(log_path)})
        continue
    steady = [row for row in metrics if row["step"] >= 10] or metrics[1:] or metrics
    rows.append(
        {
            "variant": log_path.parent.name,
            "status": "OK",
            "peak_mfu": max(row["mfu"] for row in metrics),
            "steady_mfu": sum(row["mfu"] for row in steady) / len(steady),
            "steady_tps": sum(row["tps"] for row in steady) / len(steady),
            "last_step": metrics[-1]["step"],
            "last_loss": metrics[-1]["loss"],
            "last_mfu": metrics[-1]["mfu"],
            "last_tps": metrics[-1]["tps"],
            "log": str(log_path),
        }
    )

json_path = run_root / "summary.json"
json_path.write_text(json.dumps(rows, indent=2) + "\n")

by_variant = {row["variant"]: row for row in rows if row.get("status") == "OK"}
baseline = by_variant.get("torch-compile-bf16")
md_lines = [
    "# FP4 CCE MFU Tax Ablation",
    "",
    f"- run root: `{run_root}`",
    "",
    "| Variant | Peak MFU | Steady MFU | Tax vs BF16 Compile | Steady TPS | Last Loss | Log |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    if row.get("status") != "OK":
        md_lines.append(f"| {row['variant']} | - | - | - | - | - | `{row['status']}` |")
        continue
    tax = "-"
    if baseline is not None and row["variant"] != "torch-compile-bf16":
        tax = f"+{row['steady_mfu'] - baseline['steady_mfu']:.2f}"
    elif row["variant"] == "torch-compile-bf16":
        tax = "0.00"
    md_lines.append(
        f"| {row['variant']} | {row['peak_mfu']:.2f} | {row['steady_mfu']:.2f} | "
        f"{tax} | {row['steady_tps']:.0f} | {row['last_loss']:.4f} | `{row['log']}` |"
    )

md_path = run_root / "summary.md"
md_path.write_text("\n".join(md_lines) + "\n")
print(f"Wrote JSON: {json_path}")
print(f"Wrote Markdown: {md_path}")
for row in rows:
    if row.get("status") == "OK":
        print(
            f"{row['variant']}: peak_mfu={row['peak_mfu']:.2f} "
            f"steady_mfu={row['steady_mfu']:.2f} steady_tps={row['steady_tps']:.0f}"
        )
    else:
        print(f"{row['variant']}: {row['status']}")
PY
