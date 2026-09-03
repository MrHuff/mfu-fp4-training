from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


os.environ.setdefault("LBT_LIGHT_IMPORT", "1")

from scripts.evaluation import run_canonical_lm_eval


ROOT = Path(__file__).resolve().parents[2]


def test_release_contract_binds_canonical_wrapper() -> None:
    contract = json.loads((ROOT / "release/evaluation_environment.json").read_text())
    protocol = contract["canonical_protocol"]
    wrapper = ROOT / protocol["wrapper"]

    assert wrapper == Path(run_canonical_lm_eval.__file__).resolve()
    assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == protocol["wrapper_sha256"]
    assert protocol["source_identity"] == {
        "authority": "release/torchtitan_gitlink.txt",
        "vendored_marker": "torchtitan_submodule/.lbt_torchtitan_commit",
        "torchtitan_commit": "20b3de7585696c327bd5aa9f9627f0300abdbf9d",
    }


def test_canonical_wrapper_uses_training_scaled_rope() -> None:
    frequencies = run_canonical_lm_eval._pinned_frequencies(
        ROOT / "torchtitan_submodule"
    )

    from torchtitan.models.llama3.model.model import precompute_freqs_cis

    no_scaling = precompute_freqs_cis(
        dim=128,
        end=8192,
        theta=500000.0,
        scaling_args=None,
    )
    assert frequencies.shape == (8192, 64)
    assert frequencies.dtype is torch.complex64
    assert not torch.equal(frequencies, no_scaling)
