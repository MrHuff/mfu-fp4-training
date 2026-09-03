#!/usr/bin/env python3
"""Build the checkpoint-aware Llama-8B 160B-token loss figure.

The primary input is the checksum-pinned observed-point snapshot published
with this report. That snapshot
stops pure-v5 at step 19,000 apart from a terminal marker. A separate,
hash-pinned recovery ledger adds 74 exact later log samples from the same r5
training attempt. Because those fragments are too sparse for the EMA used by
the continuous histories, the figure displays a robust LOWESS trend evaluated
on the common BF16 display grid. The 74 source points are also shown as faint,
unconnected markers. Endpoint values and tabulated endpoint differences use
observations, never the fit.

A checksum-sealed public ledger adds the complete seed-42 TE F0L4 and
operand-wise fixed-H32 trajectories. Both contain the exact logged grid from
step 1 through 38,140; no values are interpolated.

The 27/5 depth-hybrid history is a checkpoint-aware stitch of three W&B
segments.  Its checksum-sealed public ledger contains the exact logged grid
from step 1 through 38,140.  Resume-overrun rows are excluded at the two
checkpoint boundaries, and no value is interpolated.

The operand-wise plain-H16 hybrid is a checksum-sealed 3,815-point numerical
trajectory with no interpolated or missing observations. Operational restore
lineage is deliberately absent from the public CSV.

The historical BF16 and TE-native runs have unknown seeds.  Their plotted
curves receive the separately derived constant seed-42 offsets from the two
2,000-step controls.  The observed snapshot itself is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess


EXPECTED_SOURCE_SHA256 = (
    "5f71dcb1d26ef8085f98135e2ccdcda04e976b0862568b979c2471f6c13361cf"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "0e05fc4bd3fcf5784becf09cf6392bd6a1757a8896a4082544c8b17cc0a1a2ab"
)
EXPECTED_SEED_ADJUSTMENT_SHA256 = (
    "5964b53007c344e1ef30771f925b7b4127a66116f4cf82e11aa1e39dd72b62e9"
)
EXPECTED_RECOVERED_PURE_V5_SHA256 = (
    "e07764e13f754880c35799ce67681b83100c302a062fd05ec98debb0632a5d78"
)
EXPECTED_TE_F0L4_SHA256 = (
    "bc64d6e28ac46ca5370dbde1af171f184bf0cc1ab9da345781b6608d040358d7"
)
EXPECTED_OPERAND_H32_SHA256 = (
    "fdbe8b1fa97079e84049628311a80516e6218b5c81c314a73bb8cc0df897b6ef"
)
EXPECTED_MXFP4_RHT_SHA256 = (
    "1fcbf6226ba2eee92321c4923b53a7bc286539d8dfa6693646af6add23032760"
)
EXPECTED_LOCALCTA_RHT_CANONICAL_SHA256 = (
    "dc7fee140750f618801c09b9e976e868dbae1e7499e73278b710ec4e556919fd"
)
EXPECTED_OPERAND_H16_HISTORY_SHA256 = (
    "b6102b426d85f98e9fcadfac3459d7201b370ae12c48e4417b9170d3cf1be8bb"
)
EXPECTED_DEPTH_HYBRID_HISTORY_SHA256 = (
    "24f9e6a423fa839146530b427c00821f0bad629eac38b6a2983d2103abd8eb42"
)
TARGET_STEP = 38_147

DISPLAY = {
    "bf16": "BF16",
    "te_native": "TE-native",
    "mxfp4_rowsr": "MXFP4 + row-SR",
    "mxfp4_rht": "MXFP4 + row-SR + fixed H32",
    "localcta_repaired": "CTA-local NVFP4",
    "localcta_rht": "CTA-local NVFP4 + fixed H16",
    "pure_v5": "Global NVFP4 v5",
    "te_f0l4": "TE NVFP4 + 4 BF16 blocks",
    "hybrid_localcta_mxfp4": "27/5 depth hybrid",
    "operand_hybrid_plain_h16": "Operand hybrid + plain H16",
    "operand_hybrid_fixed_h32": "Operand hybrid + fixed H32",
}

COLORS = {
    "bf16": "#111827",
    "te_native": "#0284c7",
    "mxfp4_rowsr": "#f59e0b",
    "mxfp4_rht": "#0f766e",
    "localcta_repaired": "#dc2626",
    "localcta_rht": "#be185d",
    "pure_v5": "#16a34a",
    "te_f0l4": "#60a5fa",
    "hybrid_localcta_mxfp4": "#7c3aed",
    "operand_hybrid_plain_h16": "#a21caf",
    "operand_hybrid_fixed_h32": "#ea580c",
}

ORDER = list(DISPLAY)
SEED_ADJUSTED_LINEAGES = ("bf16", "te_native")

TOKENS_PER_STEP = 4_194_304
PUBLIC_SNAPSHOT_COLUMNS = [
    "lineage",
    "display_name",
    "step",
    "loss",
    "global_max_loss",
    "grad_norm",
    "tps_per_gpu",
    "mfu",
    "n_tokens_seen",
    "lr",
    "value_precision",
    "is_interpolated",
    "ema31",
    "segment",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot(source: Path) -> pd.DataFrame:
    actual_sha = sha256(source)
    if actual_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"unexpected loss snapshot SHA-256: {actual_sha}; "
            f"expected {EXPECTED_SOURCE_SHA256}"
        )

    frame = pd.read_csv(source)
    frame = frame[frame["lineage"].isin(ORDER)].copy()
    frame = frame.sort_values(["lineage", "step"])
    frame["ema31"] = frame.groupby("lineage", sort=False)["loss"].transform(
        lambda values: values.ewm(span=31, adjust=False).mean()
    )
    frame["segment"] = "observed_series"

    # Retain one point every 50 updates and every exact endpoint. This is a
    # display-only reduction over observed data, not an interpolation.
    reduced = []
    for _, part in frame.groupby("lineage", sort=False):
        keep = part.iloc[::5]
        keep = pd.concat([keep, part.tail(1)]).drop_duplicates("step", keep="last")
        reduced.append(keep)
    snapshot = pd.concat(reduced, ignore_index=True)

    recovered_endpoints = [
        {
            "lineage": "hybrid_localcta_mxfp4",
            "display_name": DISPLAY["hybrid_localcta_mxfp4"],
            "step": 38_140,
            "loss": 2.4768,
            "grad_norm": 0.0721,
            "ema31": float("nan"),
            "value_precision": "printed_4_decimal",
            "is_interpolated": False,
            "segment": "endpoint_only_no_interpolation",
        },
    ]
    snapshot = pd.concat(
        [snapshot, pd.DataFrame(recovered_endpoints)], ignore_index=True
    )
    snapshot = snapshot.sort_values(["lineage", "step", "segment"])
    return snapshot.reindex(columns=PUBLIC_SNAPSHOT_COLUMNS)


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load the sealed display snapshot without permitting silent drift."""

    actual_sha = sha256(path)
    if actual_sha != EXPECTED_SNAPSHOT_SHA256:
        raise RuntimeError(
            f"unexpected frozen loss snapshot SHA-256: {actual_sha}; "
            f"expected {EXPECTED_SNAPSHOT_SHA256}"
        )
    frame = pd.read_csv(path)
    if list(frame.columns) != PUBLIC_SNAPSHOT_COLUMNS:
        raise RuntimeError("frozen loss snapshot contains non-public columns")
    pure_v5 = frame[frame["lineage"] == "pure_v5"]
    if len(pure_v5) != 382 or set(pure_v5["value_precision"]) != {
        "printed_4_decimal"
    }:
        raise RuntimeError("frozen pure-v5 precision metadata changed")
    return frame


def load_recovered_pure_v5(path: Path) -> pd.DataFrame:
    """Load and validate the sparse post-snapshot pure-v5 evidence ledger."""

    actual_sha = sha256(path)
    if actual_sha != EXPECTED_RECOVERED_PURE_V5_SHA256:
        raise RuntimeError(
            f"unexpected recovered pure-v5 SHA-256: {actual_sha}; "
            f"expected {EXPECTED_RECOVERED_PURE_V5_SHA256}"
        )
    frame = pd.read_csv(path)
    required = {
        "lineage",
        "step",
        "loss",
        "grad_norm",
        "value_precision",
    }
    if set(frame.columns) != required:
        raise RuntimeError("recovered pure-v5 ledger schema is not exact")
    if len(frame) != 74 or frame["step"].duplicated().any():
        raise RuntimeError("recovered pure-v5 ledger must contain 74 unique steps")
    frame = frame.sort_values("step").reset_index(drop=True)
    if int(frame.iloc[0]["step"]) != 20_650 or int(frame.iloc[-1]["step"]) != 38_140:
        raise RuntimeError("recovered pure-v5 step range is invalid")
    exact_constants = {
        "lineage": "pure_v5",
        "value_precision": "printed_4_decimal",
    }
    for column, expected in exact_constants.items():
        if set(frame[column]) != {expected}:
            raise RuntimeError(f"recovered pure-v5 {column} contract is invalid")
    if not frame[["loss", "grad_norm"]].map(math.isfinite).all().all():
        raise RuntimeError("recovered pure-v5 ledger contains nonfinite metrics")

    segment_number = frame["step"].diff().ne(10).cumsum()
    frame["segment"] = segment_number.map(lambda index: f"recovered_{index:02d}")
    for _, part in frame.groupby("segment", sort=False):
        deltas = part.sort_values("step")["step"].diff().dropna()
        if not deltas.eq(10).all():
            raise RuntimeError("a recovered pure-v5 segment bridges an evidence gap")

    frame["display_name"] = DISPLAY["pure_v5"]
    frame["ema31"] = frame.groupby("segment", sort=False)["loss"].transform(
        lambda values: values.ewm(span=31, adjust=False).mean()
    )
    frame["is_interpolated"] = False
    return frame


def load_terminal_history(
    path: Path,
    *,
    source_lineage: str,
    plot_lineage: str,
    expected_sha256: str,
    expected_terminal_loss: float,
    expected_terminal_grad_norm: float,
) -> pd.DataFrame:
    """Load one complete public trajectory without private run provenance."""

    actual_history_sha = sha256(path)
    if actual_history_sha != expected_sha256:
        raise RuntimeError(
            f"unexpected {plot_lineage} history SHA-256: {actual_history_sha}; "
            f"expected {expected_sha256}"
        )
    frame = pd.read_csv(path)
    required = {
        "lineage",
        "display_name",
        "step",
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "total_tps",
        "mfu",
        "n_tokens_seen",
        "lr",
        "max_active_gib",
        "max_reserved_gib",
        "num_alloc_retries",
        "num_ooms",
        "value_precision",
        "is_interpolated",
    }
    if set(frame.columns) != required:
        raise RuntimeError(f"{plot_lineage} history schema is not exact")
    expected_steps = [1, *range(10, 38_141, 10)]
    if len(frame) != 3_815 or frame["step"].astype(int).tolist() != expected_steps:
        raise RuntimeError(f"{plot_lineage} cadence or endpoint changed")
    if frame["step"].duplicated().any() or not frame["step"].is_monotonic_increasing:
        raise RuntimeError(
            f"{plot_lineage} steps are not strictly increasing and unique"
        )
    if set(frame["lineage"]) != {source_lineage}:
        raise RuntimeError(f"{plot_lineage} history contains another lineage")
    if frame["is_interpolated"].astype(str).str.lower().ne("false").any():
        raise RuntimeError(f"{plot_lineage} history contains interpolated evidence")
    numeric = [
        "step",
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "total_tps",
        "mfu",
        "n_tokens_seen",
        "lr",
    ]
    if not frame[numeric].map(math.isfinite).all().all():
        raise RuntimeError(f"{plot_lineage} history contains nonfinite evidence")
    if (frame["n_tokens_seen"].diff().dropna() < 0).any():
        raise RuntimeError(f"{plot_lineage} token counter is not monotone")
    if set(frame["value_precision"]) != {"exported_float_payload"}:
        raise RuntimeError(f"{plot_lineage} precision contract changed")
    terminal = frame.iloc[-1]
    if not (
        math.isclose(
            float(terminal["loss"]),
            expected_terminal_loss,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            float(terminal["grad_norm"]),
            expected_terminal_grad_norm,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise RuntimeError(f"{plot_lineage} terminal evidence changed")

    frame = frame.copy()
    frame["lineage"] = plot_lineage
    frame["display_name"] = DISPLAY[plot_lineage]
    frame["ema31"] = frame["loss"].ewm(span=31, adjust=False).mean()
    frame["segment"] = "observed_series"
    reduced = pd.concat([frame.iloc[::5], frame.tail(1)])
    return reduced.drop_duplicates("step", keep="last")


def load_depth_hybrid_history(path: Path) -> pd.DataFrame:
    """Load the complete checkpoint-aware 27/5 W&B history."""

    actual_sha = sha256(path)
    if actual_sha != EXPECTED_DEPTH_HYBRID_HISTORY_SHA256:
        raise RuntimeError(
            f"unexpected 27/5 history SHA-256: {actual_sha}; "
            f"expected {EXPECTED_DEPTH_HYBRID_HISTORY_SHA256}"
        )
    frame = pd.read_csv(path)
    required = {
        "source_segment",
        "step",
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu",
        "n_tokens_seen",
        "lr",
    }
    if set(frame.columns) != required:
        raise RuntimeError("27/5 history schema is not exact")

    expected_steps = [1, *range(10, 38_141, 10)]
    if len(frame) != 3_815 or frame["step"].astype(int).tolist() != expected_steps:
        raise RuntimeError("27/5 history cadence or endpoint changed")
    if frame["step"].duplicated().any() or not frame["step"].is_monotonic_increasing:
        raise RuntimeError("27/5 history steps are not strictly increasing and unique")

    segment_contract = (
        ("fresh_through_step16000", [1, *range(10, 16_001, 10)]),
        ("resume_step16000_to35000", list(range(16_010, 35_001, 10))),
        ("resume_step35000_to38140", list(range(35_010, 38_141, 10))),
    )
    for segment, steps in segment_contract:
        observed_steps = frame.loc[frame["source_segment"] == segment, "step"].astype(
            int
        )
        if observed_steps.tolist() != steps:
            raise RuntimeError(f"27/5 {segment} splice contract changed")
    if set(frame["source_segment"]) != {item[0] for item in segment_contract}:
        raise RuntimeError("27/5 history contains an unexpected source segment")

    numeric = [
        "step",
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu",
        "n_tokens_seen",
        "lr",
    ]
    if not frame[numeric].map(math.isfinite).all().all():
        raise RuntimeError("27/5 history contains nonfinite evidence")
    expected_tokens = frame["step"].astype(int) * TOKENS_PER_STEP
    if not frame["n_tokens_seen"].astype(int).equals(expected_tokens):
        raise RuntimeError("27/5 token counters do not match the step contract")

    terminal = frame.iloc[-1]
    terminal_contract = {
        "loss": 2.4768424034118652,
        "global_max_loss": 2.9709818363189697,
        "grad_norm": 0.0720905289053917,
    }
    for metric, expected in terminal_contract.items():
        if not math.isclose(
            float(terminal[metric]), expected, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise RuntimeError(f"27/5 terminal {metric} changed")

    frame = frame.copy()
    frame["lineage"] = "hybrid_localcta_mxfp4"
    frame["display_name"] = DISPLAY["hybrid_localcta_mxfp4"]
    frame["value_precision"] = "exported_float_payload"
    frame["is_interpolated"] = False
    frame["ema31"] = frame["loss"].ewm(span=31, adjust=False).mean()
    frame["segment"] = "observed_series"
    return frame


def attach_depth_hybrid_history(
    snapshot: pd.DataFrame, complete_history: pd.DataFrame
) -> pd.DataFrame:
    """Supersede the stale snapshot tail after verifying its shared prefix."""

    lineage = "hybrid_localcta_mxfp4"
    legacy_observed = snapshot[
        (snapshot["lineage"] == lineage) & (snapshot["segment"] == "observed_series")
    ]
    overlap = legacy_observed.merge(
        complete_history,
        on="step",
        how="left",
        suffixes=("_legacy", "_complete"),
        validate="one_to_one",
    )
    if len(legacy_observed) != 712 or overlap["loss_complete"].isna().any():
        raise RuntimeError("27/5 legacy-prefix inventory changed")
    for metric in (
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu",
        "n_tokens_seen",
        "lr",
    ):
        difference = (
            overlap[f"{metric}_legacy"].astype(float)
            - overlap[f"{metric}_complete"].astype(float)
        ).abs()
        if not difference.le(1.0e-12).all():
            raise RuntimeError(f"27/5 complete history disagrees on shared {metric}")

    display_history = pd.concat([complete_history.iloc[::5], complete_history.tail(1)])
    display_history = display_history.drop_duplicates("step", keep="last")
    without_legacy = snapshot[snapshot["lineage"] != lineage].copy()
    return pd.concat([without_legacy, display_history], ignore_index=True, sort=False)


def attach_recovered_pure_v5(
    snapshot: pd.DataFrame, recovered: pd.DataFrame
) -> pd.DataFrame:
    """Replace the legacy terminal marker with its auditable sparse ledger."""

    combined = snapshot.copy()
    legacy_endpoint = (
        (combined["lineage"] == "pure_v5")
        & (combined["step"] == 38_140)
        & (combined["segment"] == "endpoint_only_no_interpolation")
    )
    combined = combined[~legacy_endpoint].copy()
    pure_observed_steps = set(
        combined.loc[combined["lineage"] == "pure_v5", "step"].astype(int)
    )
    recovered_steps = set(recovered["step"].astype(int))
    if pure_observed_steps & recovered_steps:
        raise RuntimeError("recovered pure-v5 ledger overlaps the base snapshot")
    return pd.concat([combined, recovered], ignore_index=True, sort=False)


def load_seed_adjustments(path: Path) -> dict[str, float]:
    actual_sha = sha256(path)
    if actual_sha != EXPECTED_SEED_ADJUSTMENT_SHA256:
        raise RuntimeError(
            f"unexpected seed-adjustment SHA-256: {actual_sha}; "
            f"expected {EXPECTED_SEED_ADJUSTMENT_SHA256}"
        )
    frame = pd.read_csv(path)
    if set(frame["lineage"]) != set(SEED_ADJUSTED_LINEAGES):
        raise RuntimeError("seed-adjustment lineage inventory is not exact")
    if frame["lineage"].duplicated().any():
        raise RuntimeError("seed-adjustment table contains duplicate lineages")
    if not (
        (frame["matched_points"] == 41).all()
        and (frame["first_step"] == 1).all()
        and (frame["last_step"] == 2_000).all()
        and (
            frame["estimator"] == "Huber location of exact-step loss differences"
        ).all()
        and (frame["huber_k"] == 1.345).all()
    ):
        raise RuntimeError("seed-adjustment estimator contract is invalid")
    offsets = frame.set_index("lineage")["loss_offset_seed42_minus_historical"].astype(
        float
    )
    if not offsets.map(math.isfinite).all():
        raise RuntimeError("seed-adjustment table contains a nonfinite offset")
    return offsets.to_dict()


def load_mxfp4_rht_history(path: Path) -> pd.DataFrame:
    """Load the frozen public metric history and reduce it for display."""

    actual_sha = sha256(path)
    if actual_sha != EXPECTED_MXFP4_RHT_SHA256:
        raise RuntimeError(
            f"unexpected MXFP4+RHT history SHA-256: {actual_sha}; "
            f"expected {EXPECTED_MXFP4_RHT_SHA256}"
        )
    frame = pd.read_csv(path)
    required = {
        "lineage",
        "display_name",
        "step",
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu",
        "n_tokens_seen",
        "lr",
        "max_active_gib",
        "max_reserved_gib",
        "num_alloc_retries",
        "num_ooms",
        "value_precision",
        "is_interpolated",
    }
    if set(frame.columns) != required:
        raise RuntimeError("MXFP4+RHT history schema is not exact")
    if len(frame) != 3_815:
        raise RuntimeError(f"unexpected MXFP4+RHT history length: {len(frame)}")
    if set(frame["lineage"]) != {"mxfp4_rht"}:
        raise RuntimeError("MXFP4+RHT history contains another lineage")
    if frame["step"].duplicated().any() or not frame["step"].is_monotonic_increasing:
        raise RuntimeError("MXFP4+RHT steps are not strictly increasing and unique")
    if int(frame.iloc[0]["step"]) != 1 or int(frame.iloc[-1]["step"]) != 38_140:
        raise RuntimeError("MXFP4+RHT history endpoints are not step 1 and 38,140")
    numeric = [
        "step",
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu",
        "n_tokens_seen",
        "lr",
    ]
    if not frame[numeric].map(math.isfinite).all().all():
        raise RuntimeError("MXFP4+RHT history contains nonfinite evidence")

    frame = frame.copy()
    frame["ema31"] = frame["loss"].ewm(span=31, adjust=False).mean()
    frame["segment"] = "observed_series"
    # The source logs every ten updates. Retain every fifth observation (50 updates)
    # plus the exact terminal point, matching the existing display snapshot.
    reduced = pd.concat([frame.iloc[::5], frame.tail(1)])
    return reduced.drop_duplicates("step", keep="last")


def load_localcta_rht_history(path: Path) -> pd.DataFrame:
    """Load the hash-pinned canonical localCTA+RHT numerical trajectory."""

    actual_history_sha = sha256(path)
    if actual_history_sha != EXPECTED_LOCALCTA_RHT_CANONICAL_SHA256:
        raise RuntimeError(
            f"unexpected localCTA+RHT history SHA-256: {actual_history_sha}; "
            f"expected {EXPECTED_LOCALCTA_RHT_CANONICAL_SHA256}"
        )
    frame = pd.read_csv(path)
    required = {
        "canonical",
        "step",
        "tokens",
        "tokens_billions",
        "loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu_percent",
    }
    if set(frame.columns) != required:
        raise RuntimeError("localCTA+RHT history schema is not exact")
    if len(frame) != 3_815:
        raise RuntimeError(f"unexpected localCTA+RHT history length: {len(frame)}")

    expected_steps = [1, *range(10, 38_141, 10)]
    if frame["step"].astype(int).tolist() != expected_steps:
        raise RuntimeError("localCTA+RHT metric cadence or endpoint changed")
    if frame["step"].duplicated().any() or not frame["step"].is_monotonic_increasing:
        raise RuntimeError("localCTA+RHT steps are not strictly increasing and unique")
    if not frame["canonical"].astype(bool).all():
        raise RuntimeError("localCTA+RHT history contains a non-canonical row")

    numeric = [
        "step",
        "tokens",
        "tokens_billions",
        "loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu_percent",
    ]
    if not frame[numeric].map(math.isfinite).all().all():
        raise RuntimeError("localCTA+RHT history contains nonfinite evidence")
    expected_tokens = frame["step"].astype(int) * TOKENS_PER_STEP
    if not frame["tokens"].astype(int).equals(expected_tokens):
        raise RuntimeError("localCTA+RHT token counts do not match the step contract")
    if not ((frame["tokens_billions"] - frame["tokens"] / 1.0e9).abs() < 1.0e-12).all():
        raise RuntimeError("localCTA+RHT billion-token coordinates are inconsistent")

    frame = frame.copy()
    frame["lineage"] = "localcta_rht"
    frame["display_name"] = DISPLAY["localcta_rht"]
    frame["ema31"] = frame["loss"].ewm(span=31, adjust=False).mean()
    frame["segment"] = "observed_series"
    frame["is_interpolated"] = False
    # The export logs every ten updates. Retain every fifth observation (50
    # updates) plus the exact terminal scalar, as for the other long curves.
    reduced = pd.concat([frame.iloc[::5], frame.tail(1)])
    return reduced.drop_duplicates("step", keep="last")


def load_operand_h16_history(path: Path) -> pd.DataFrame:
    """Load the hash-pinned public trajectory for the plain-H16 hybrid."""

    if sha256(path) != EXPECTED_OPERAND_H16_HISTORY_SHA256:
        raise RuntimeError("operand-H16 history hash changed")

    frame = pd.read_csv(path)
    required = {
        "lineage",
        "display_name",
        "step",
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu",
        "n_tokens_seen",
        "lr",
        "max_active_gib",
        "max_reserved_gib",
        "num_alloc_retries",
        "num_ooms",
        "value_precision",
        "is_interpolated",
    }
    if set(frame.columns) != required:
        raise RuntimeError("operand-H16 history schema changed")
    expected_steps = [1, *range(10, 38_141, 10)]
    if len(frame) != 3_815 or frame["step"].astype(int).tolist() != expected_steps:
        raise RuntimeError("operand-H16 history cadence changed")
    if set(frame["lineage"]) != {"operand_hybrid_plain_h16"}:
        raise RuntimeError("operand-H16 history contains another lineage")
    numeric = [
        "step",
        "loss",
        "global_max_loss",
        "grad_norm",
        "tps_per_gpu",
        "mfu",
        "n_tokens_seen",
        "lr",
        "max_active_gib",
        "max_reserved_gib",
        "num_alloc_retries",
        "num_ooms",
    ]
    if not frame[numeric].map(math.isfinite).all().all():
        raise RuntimeError("operand-H16 history contains nonfinite evidence")
    expected_tokens = frame["step"].astype(int) * TOKENS_PER_STEP
    if not frame["n_tokens_seen"].astype(int).equals(expected_tokens):
        raise RuntimeError("operand-H16 token counters changed")
    if frame["is_interpolated"].astype(str).str.lower().ne("false").any():
        raise RuntimeError("operand-H16 history contains interpolated evidence")

    frame = frame.copy()
    frame["ema31"] = frame["loss"].ewm(span=31, adjust=False).mean()
    frame["segment"] = "observed_series"
    reduced = pd.concat([frame.iloc[::5], frame.tail(1)])
    return reduced.drop_duplicates("step", keep="last")


def apply_seed_adjustments(
    snapshot: pd.DataFrame, adjustments: dict[str, float]
) -> pd.DataFrame:
    adjusted = snapshot.copy()
    adjusted["measured_loss"] = adjusted["loss"]
    adjusted["measured_ema31"] = adjusted["ema31"]
    adjusted["seed42_loss_offset"] = 0.0
    adjusted["is_seed42_estimate"] = False
    for lineage in SEED_ADJUSTED_LINEAGES:
        if lineage not in adjustments:
            raise RuntimeError(f"missing seed adjustment for {lineage}")
        selected = adjusted["lineage"] == lineage
        if not selected.any():
            raise RuntimeError(f"loss snapshot is missing {lineage}")
        offset = adjustments[lineage]
        adjusted.loc[selected, "loss"] += offset
        adjusted.loc[selected, "ema31"] += offset
        adjusted.loc[selected, "seed42_loss_offset"] = offset
        adjusted.loc[selected, "is_seed42_estimate"] = True
    return adjusted


def add_exact_step_bf16_gaps(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Match EMA curves to seed-42-adjusted BF16 without interpolation."""

    observed = snapshot[snapshot["segment"] == "observed_series"].copy()
    bf16 = observed[observed["lineage"] == "bf16"][["step", "loss", "ema31"]].rename(
        columns={"loss": "bf16_loss", "ema31": "bf16_ema31"}
    )
    if bf16["step"].duplicated().any():
        raise RuntimeError("BF16 snapshot contains duplicate observed steps")
    observed = observed.merge(bf16, on="step", how="left", validate="many_to_one")
    # Match the sign convention used by NVIDIA et al.: zero is the BF16
    # baseline and negative values mean the low-precision loss is higher.
    observed["relative_difference_percent"] = (
        100.0 * (observed["bf16_ema31"] - observed["ema31"]) / observed["bf16_ema31"]
    )
    observed["tokens_billions"] = (
        observed["step"].astype(float) * TOKENS_PER_STEP / 1.0e9
    )
    return observed


def terminal_points(snapshot: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Return each route's latest raw point and its exact-step BF16 gap."""

    observed_tails = (
        observed.sort_values(["lineage", "step"]).groupby("lineage", sort=False).tail(1)
    )
    recovered = snapshot[snapshot["segment"] != "observed_series"].copy()
    terminals = (
        pd.concat([observed_tails, recovered], ignore_index=True, sort=False)
        .sort_values(["lineage", "step"])
        .drop_duplicates("lineage", keep="last")
    )
    bf16_raw = observed[observed["lineage"] == "bf16"][["step", "loss"]].rename(
        columns={"loss": "terminal_bf16_loss"}
    )
    terminals = terminals.merge(bf16_raw, on="step", how="left", validate="many_to_one")
    terminals["endpoint_relative_difference_percent"] = (
        100.0
        * (terminals["terminal_bf16_loss"] - terminals["loss"])
        / terminals["terminal_bf16_loss"]
    )
    terminals["tokens_billions"] = (
        terminals["step"].astype(float) * TOKENS_PER_STEP / 1.0e9
    )
    return terminals


def recovered_pure_v5_points(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Return sparse r5 samples with segment-local EMA and token coordinates."""

    recovered = snapshot[
        (snapshot["lineage"] == "pure_v5")
        & snapshot["segment"].astype(str).str.startswith("recovered_")
    ].copy()
    recovered["tokens_billions"] = (
        recovered["step"].astype(float) * TOKENS_PER_STEP / 1.0e9
    )
    return recovered.sort_values(["segment", "step"])


def recovered_pure_v5_gap_points(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Return exact-step raw v5/BF16 differences after the dense v5 stream."""

    recovered = recovered_pure_v5_points(snapshot)
    bf16 = snapshot[
        (snapshot["lineage"] == "bf16") & (snapshot["segment"] == "observed_series")
    ][["step", "loss"]].rename(columns={"loss": "bf16_loss"})
    matched = recovered.merge(bf16, on="step", how="inner", validate="one_to_one")
    # The sealed ledgers share 17 intermediate steps and the step-38,140
    # endpoint. Anything else means one of the source inventories drifted.
    if len(matched) != 18 or 38_140 not in set(matched["step"].astype(int)):
        raise RuntimeError("recovered pure-v5/BF16 exact-step inventory changed")
    matched["relative_difference_percent"] = (
        100.0 * (matched["bf16_loss"] - matched["loss"]) / matched["bf16_loss"]
    )
    if not matched["relative_difference_percent"].map(math.isfinite).all():
        raise RuntimeError("recovered pure-v5/BF16 raw differences are nonfinite")
    return matched.sort_values("step")


def estimated_pure_v5_trend(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Fit sparse post-19k r5 evidence and evaluate it on a dense step grid."""

    dense_tail = snapshot[
        (snapshot["lineage"] == "pure_v5")
        & (snapshot["segment"] == "observed_series")
        & (snapshot["step"] >= 17_500)
    ][["step", "loss"]]
    recovered = recovered_pure_v5_points(snapshot)[["step", "loss"]]
    anchors = (
        pd.concat([dense_tail, recovered], ignore_index=True)
        .sort_values("step")
        .drop_duplicates("step", keep="last")
    )
    if len(dense_tail) != 31 or len(recovered) != 74 or len(anchors) != 105:
        raise RuntimeError("pure-v5 late-trend anchor inventory changed")

    display_grid = snapshot[
        (snapshot["lineage"] == "bf16")
        & (snapshot["segment"] == "observed_series")
        & (snapshot["step"] >= 19_000)
    ]["step"].astype(float)
    if (
        len(display_grid) != 384
        or int(display_grid.iloc[0]) != 19_000
        or int(display_grid.iloc[-1]) != 38_140
        or display_grid.duplicated().any()
    ):
        raise RuntimeError("pure-v5 dense display grid changed")

    fitted_loss = lowess(
        endog=anchors["loss"].to_numpy(),
        exog=anchors["step"].to_numpy(),
        frac=0.20,
        it=3,
        xvals=display_grid.to_numpy(),
        is_sorted=True,
        return_sorted=False,
    )
    trend = pd.DataFrame(
        {"step": display_grid.to_numpy(), "estimated_loss": fitted_loss}
    )
    trend["tokens_billions"] = trend["step"] * TOKENS_PER_STEP / 1.0e9
    if len(trend) != 384 or not trend["estimated_loss"].map(math.isfinite).all():
        raise RuntimeError("pure-v5 late-trend fit contract changed")
    return trend


def estimated_pure_v5_gap_trend(
    snapshot: pd.DataFrame,
    recovered_matched: pd.DataFrame,
    loss_trend: pd.DataFrame,
) -> pd.DataFrame:
    """Express the dense-grid v5 visual trend relative to BF16 EMA-31."""

    v5_dense = snapshot[
        (snapshot["lineage"] == "pure_v5")
        & (snapshot["segment"] == "observed_series")
        & (snapshot["step"] >= 17_500)
    ][["step", "loss"]].rename(columns={"loss": "v5_loss"})
    bf16 = snapshot[
        (snapshot["lineage"] == "bf16") & (snapshot["segment"] == "observed_series")
    ][["step", "loss"]].rename(columns={"loss": "bf16_loss"})
    dense_matched = v5_dense.merge(bf16, on="step", how="inner", validate="one_to_one")
    dense_matched["relative_difference_percent"] = (
        100.0
        * (dense_matched["bf16_loss"] - dense_matched["v5_loss"])
        / dense_matched["bf16_loss"]
    )
    if len(dense_matched) != 31 or len(recovered_matched) != 18:
        raise RuntimeError("pure-v5 late relative-trend anchor inventory changed")

    bf16_display = snapshot[
        (snapshot["lineage"] == "bf16")
        & (snapshot["segment"] == "observed_series")
        & (snapshot["step"] >= 19_000)
    ][["step", "ema31"]].rename(columns={"ema31": "bf16_ema31"})
    trend = bf16_display.merge(
        loss_trend[["step", "estimated_loss"]],
        on="step",
        how="inner",
        validate="one_to_one",
    )
    trend["estimated_relative_difference_percent"] = (
        100.0
        * (trend["bf16_ema31"] - trend["estimated_loss"])
        / trend["bf16_ema31"]
    )
    trend["tokens_billions"] = trend["step"] * TOKENS_PER_STEP / 1.0e9
    if (
        len(trend) != 384
        or not trend["estimated_relative_difference_percent"].map(math.isfinite).all()
    ):
        raise RuntimeError("pure-v5 late relative-trend fit contract changed")
    return trend


def plot_recovered_pure_v5(
    axis: plt.Axes,
    recovered: pd.DataFrame,
    trend: pd.DataFrame,
    terminal_step: int,
) -> None:
    """Draw the late display fit and the exact sparse r5 observations."""

    points = recovered[recovered["step"] < terminal_step]
    axis.scatter(
        points["tokens_billions"],
        points["loss"],
        color=COLORS["pure_v5"],
        marker="o",
        s=7,
        linewidths=0,
        alpha=0.25,
        zorder=3,
    )
    axis.plot(
        trend["tokens_billions"],
        trend["estimated_loss"],
        color=COLORS["pure_v5"],
        linestyle="--",
        linewidth=1.9,
        alpha=0.88,
        zorder=4,
    )


def plot_recovered_pure_v5_gaps(
    axis: plt.Axes,
    matched: pd.DataFrame,
    trend: pd.DataFrame,
    terminal_step: int,
) -> None:
    """Draw the late relative display fit and exact shared-step differences."""

    points = matched[matched["step"] < terminal_step]
    axis.scatter(
        points["tokens_billions"],
        points["relative_difference_percent"],
        color=COLORS["pure_v5"],
        marker="o",
        s=7,
        linewidths=0,
        alpha=0.25,
        zorder=3,
    )
    axis.plot(
        trend["tokens_billions"],
        trend["estimated_relative_difference_percent"],
        color=COLORS["pure_v5"],
        linestyle="--",
        linewidth=1.9,
        alpha=0.88,
        zorder=4,
    )


def plot(snapshot: pd.DataFrame, output: Path) -> None:
    snapshot = snapshot[snapshot["lineage"].isin(ORDER)].copy()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
        }
    )
    fig, (loss_axis, gap_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.4),
        gridspec_kw={"height_ratios": (1.08, 1.0)},
        sharex=True,
    )

    observed = add_exact_step_bf16_gaps(snapshot)
    terminals = terminal_points(snapshot, observed)
    recovered_v5 = recovered_pure_v5_points(snapshot)
    recovered_v5_gaps = recovered_pure_v5_gap_points(snapshot)
    recovered_v5_trend = estimated_pure_v5_trend(snapshot)
    recovered_v5_gap_trend = estimated_pure_v5_gap_trend(
        snapshot, recovered_v5_gaps, recovered_v5_trend
    )

    for lineage in ORDER:
        part = observed[observed["lineage"] == lineage].sort_values("step")
        if part.empty:
            continue
        width = 1.9
        loss_axis.plot(
            part["tokens_billions"],
            part["ema31"],
            color=COLORS[lineage],
            linestyle="-",
            linewidth=width,
            label=DISPLAY[lineage],
        )
        gap_axis.plot(
            part["tokens_billions"],
            part["relative_difference_percent"],
            color=COLORS[lineage],
            linestyle="-",
            linewidth=width,
        )

    plot_recovered_pure_v5(
        loss_axis, recovered_v5, recovered_v5_trend, terminal_step=38_140
    )
    plot_recovered_pure_v5_gaps(
        gap_axis,
        recovered_v5_gaps,
        recovered_v5_gap_trend,
        terminal_step=38_140,
    )

    for terminal in terminals.itertuples(index=False):
        isolated = terminal.segment != "observed_series"
        loss_axis.scatter(
            [terminal.tokens_billions],
            [terminal.loss],
            marker="D",
            s=27,
            facecolor="white" if isolated else COLORS[terminal.lineage],
            edgecolor=COLORS[terminal.lineage] if isolated else "white",
            linewidth=1.4 if isolated else 0.6,
            zorder=5,
        )
        if pd.isna(terminal.endpoint_relative_difference_percent):
            continue
        gap_axis.scatter(
            [terminal.tokens_billions],
            [terminal.endpoint_relative_difference_percent],
            marker="D",
            s=29,
            facecolor="white" if isolated else COLORS[terminal.lineage],
            edgecolor=COLORS[terminal.lineage] if isolated else "white",
            linewidth=1.4 if isolated else 0.6,
            zorder=6,
        )

    endpoint = terminals[terminals["lineage"] == "mxfp4_rht"].iloc[0]

    target_tokens_billions = TARGET_STEP * TOKENS_PER_STEP / 1.0e9
    loss_axis.set_title("Training loss")
    loss_axis.set_xlim(0.0, target_tokens_billions)
    loss_axis.set_ylim(2.2, 8.1)
    loss_axis.set_ylabel("Training loss (EMA-31)")
    loss_axis.grid(True, alpha=0.22)

    gap_axis.set_title("Relative difference from BF16 (higher is better)")
    gap_axis.set_ylim(-7.5, 1.0)
    gap_axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.7)
    gap_axis.set_xlabel("Training tokens (billions)")
    gap_axis.set_ylabel("Relative loss difference (%)")
    gap_axis.grid(True, alpha=0.22)

    loss_axis.annotate(
        f"MXFP4 + row-SR + fixed H32\n160B: {endpoint.loss:.4f}",
        xy=(endpoint.tokens_billions, endpoint.loss),
        xytext=(118.0, 2.72),
        color=COLORS["mxfp4_rht"],
        arrowprops={"arrowstyle": "->", "color": COLORS["mxfp4_rht"], "lw": 0.8},
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[lineage],
            linestyle="--" if lineage == "pure_v5" else "-",
            linewidth=1.9,
        )
        for lineage in ORDER
    ]
    labels = [
        f"{DISPLAY[lineage]} (late fit)" if lineage == "pure_v5" else DISPLAY[lineage]
        for lineage in ORDER
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Llama-8B: training loss and relative difference from BF16",
        y=0.99,
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.085,
        top=0.84,
        hspace=0.28,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_zoom(snapshot: pd.DataFrame, output: Path) -> None:
    """Render the same trajectories over the final 40B-token display window."""

    snapshot = snapshot[snapshot["lineage"].isin(ORDER)].copy()
    observed = add_exact_step_bf16_gaps(snapshot)
    terminals = terminal_points(snapshot, observed)
    start_tokens = 120.0
    target_tokens = TARGET_STEP * TOKENS_PER_STEP / 1.0e9
    recovered_v5 = recovered_pure_v5_points(snapshot)
    recovered_v5_gaps = recovered_pure_v5_gap_points(snapshot)
    recovered_v5_trend = estimated_pure_v5_trend(snapshot)
    recovered_v5_gap_trend = estimated_pure_v5_gap_trend(
        snapshot, recovered_v5_gaps, recovered_v5_trend
    )
    recovered_v5 = recovered_v5[recovered_v5["tokens_billions"] >= start_tokens].copy()
    recovered_v5_gaps = recovered_v5_gaps[
        recovered_v5_gaps["tokens_billions"] >= start_tokens
    ].copy()

    fig, (loss_axis, gap_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.8),
        gridspec_kw={"height_ratios": (1.08, 1.0)},
        sharex=True,
    )
    for lineage in ORDER:
        part = observed[
            (observed["lineage"] == lineage)
            & (observed["tokens_billions"] >= start_tokens)
        ].sort_values("step")
        if part.empty:
            continue
        loss_axis.plot(
            part["tokens_billions"],
            part["ema31"],
            color=COLORS[lineage],
            linestyle="-",
            linewidth=1.9,
            label=DISPLAY[lineage],
        )
        gap_axis.plot(
            part["tokens_billions"],
            part["relative_difference_percent"],
            color=COLORS[lineage],
            linestyle="-",
            linewidth=1.9,
        )

    plot_recovered_pure_v5(
        loss_axis, recovered_v5, recovered_v5_trend, terminal_step=38_140
    )
    plot_recovered_pure_v5_gaps(
        gap_axis,
        recovered_v5_gaps,
        recovered_v5_gap_trend,
        terminal_step=38_140,
    )

    for terminal in terminals.itertuples(index=False):
        if terminal.tokens_billions < start_tokens:
            continue
        isolated = terminal.segment != "observed_series"
        loss_axis.scatter(
            [terminal.tokens_billions],
            [terminal.loss],
            marker="D",
            s=27,
            facecolor="white" if isolated else COLORS[terminal.lineage],
            edgecolor=COLORS[terminal.lineage] if isolated else "white",
            linewidth=1.4 if isolated else 0.6,
            zorder=5,
        )
        if pd.notna(terminal.endpoint_relative_difference_percent):
            gap_axis.scatter(
                [terminal.tokens_billions],
                [terminal.endpoint_relative_difference_percent],
                marker="D",
                s=29,
                facecolor="white" if isolated else COLORS[terminal.lineage],
                edgecolor=COLORS[terminal.lineage] if isolated else "white",
                linewidth=1.4 if isolated else 0.6,
                zorder=6,
            )

    loss_axis.set_title("End-window training loss")
    loss_axis.set_ylabel("Training loss (EMA-31)")
    loss_axis.set_ylim(2.30, 2.62)
    loss_axis.grid(True, alpha=0.22)
    gap_axis.set_title("End-window relative difference from BF16")
    gap_axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.7)
    gap_axis.set_xlabel("Training tokens (billions)")
    gap_axis.set_ylabel("Relative loss difference (%)")
    gap_axis.set_ylim(-7.5, 1.0)
    gap_axis.grid(True, alpha=0.22)
    gap_axis.set_xlim(start_tokens, target_tokens)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[lineage],
            linestyle="--" if lineage == "pure_v5" else "-",
            linewidth=1.9,
        )
        for lineage in ORDER
    ]
    labels = [
        f"{DISPLAY[lineage]} (late fit)" if lineage == "pure_v5" else DISPLAY[lineage]
        for lineage in ORDER
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Llama-8B: final 40B-token training window",
        y=0.99,
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.09,
        top=0.81,
        hspace=0.28,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--snapshot-in", type=Path)
    parser.add_argument("--snapshot-out", type=Path)
    parser.add_argument("--seed-adjustment", type=Path, required=True)
    parser.add_argument("--pure-v5-recovered", type=Path, required=True)
    parser.add_argument("--te-f0l4-history", type=Path, required=True)
    parser.add_argument("--operand-h32-history", type=Path, required=True)
    parser.add_argument("--mxfp4-rht-history", type=Path, required=True)
    parser.add_argument("--localcta-rht-history", type=Path, required=True)
    parser.add_argument("--operand-h16-history", type=Path, required=True)
    parser.add_argument("--depth-hybrid-history", type=Path, required=True)
    parser.add_argument("--figure-out", type=Path, required=True)
    parser.add_argument("--zoom-figure-out", type=Path, required=True)
    args = parser.parse_args()

    if args.source is not None:
        if args.snapshot_out is None:
            parser.error("--snapshot-out is required with --source")
        snapshot = build_snapshot(args.source)
        args.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(args.snapshot_out, index=False)
    else:
        snapshot = load_snapshot(args.snapshot_in)
    snapshot = attach_recovered_pure_v5(
        snapshot, load_recovered_pure_v5(args.pure_v5_recovered)
    )
    snapshot = attach_depth_hybrid_history(
        snapshot, load_depth_hybrid_history(args.depth_hybrid_history)
    )
    snapshot = pd.concat(
        [
            snapshot,
            load_terminal_history(
                args.te_f0l4_history,
                source_lineage="te_f0l4_seed42",
                plot_lineage="te_f0l4",
                expected_sha256=EXPECTED_TE_F0L4_SHA256,
                expected_terminal_loss=2.385180711746216,
                expected_terminal_grad_norm=0.060039907693862915,
            ),
            load_terminal_history(
                args.operand_h32_history,
                source_lineage="operand_hybrid_fixed_h32",
                plot_lineage="operand_hybrid_fixed_h32",
                expected_sha256=EXPECTED_OPERAND_H32_SHA256,
                expected_terminal_loss=2.4593048095703125,
                expected_terminal_grad_norm=0.12212450802326202,
            ),
            load_mxfp4_rht_history(args.mxfp4_rht_history),
            load_localcta_rht_history(args.localcta_rht_history),
            load_operand_h16_history(args.operand_h16_history),
        ],
        ignore_index=True,
        sort=False,
    )
    adjustments = load_seed_adjustments(args.seed_adjustment)
    adjusted = apply_seed_adjustments(snapshot, adjustments)
    plot(adjusted, args.figure_out)
    plot_zoom(adjusted, args.zoom_figure_out)


if __name__ == "__main__":
    main()
