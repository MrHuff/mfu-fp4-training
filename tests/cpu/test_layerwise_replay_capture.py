# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import math
import json
import os
from pathlib import Path
import sys

import pyarrow as pa
import pytest
import torch


os.environ.setdefault("LBT_LIGHT_IMPORT", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from low_bits_training.analysis.layerwise_replay_capture import (  # noqa: E402
    LayerCaptureWriter,
    _tt_equivalent_hf_rope,
    build_token_batches,
    calibration_tensor_ledger,
    chunked_causal_ce_hidden_gradient,
    deterministic_row_indices,
    load_json_object,
    safetensors_payload_bytes,
    tensor_magnitude_summary,
    validate_receipt_seal,
)


def _write_safetensors_fixture(path: Path, header: dict, payload: bytes) -> None:
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)


def test_safetensors_payload_bytes_validates_contiguous_offsets(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    _write_safetensors_fixture(
        path,
        {
            "a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        },
        b"12345678",
    )
    assert safetensors_payload_bytes(path) == 8

    broken = tmp_path / "broken.safetensors"
    _write_safetensors_fixture(
        broken,
        {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [1, 5]}},
        b"12345",
    )
    with pytest.raises(RuntimeError, match="non-contiguous"):
        safetensors_payload_bytes(broken)


class _ToyTokenizer:
    bos_token_id = 100
    eos_token_id = 101

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) - 96 for character in text]


def _write_arrow(path: Path, texts: list[str], source: str) -> None:
    table = pa.table(
        {
            "text": texts,
            "id": [f"{source}-{index}" for index in range(len(texts))],
            "source_ds": [source] * len(texts),
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def test_build_token_batches_round_robins_sorted_sources(tmp_path):
    second = tmp_path / "b.arrow"
    first = tmp_path / "a.arrow"
    _write_arrow(first, ["aa", "cc"], "a")
    _write_arrow(second, ["bb", "dd"], "b")

    payload, documents = build_token_batches(
        [second, first],
        _ToyTokenizer(),
        seq_len=4,
        batch_size=1,
        num_batches=1,
    )

    assert payload["input_ids"].tolist() == [[[100, 1, 1, 2]]]
    assert payload["target_ids"].tolist() == [[[1, 1, 2, 2]]]
    assert [document["source_file"] for document in documents[:2]] == [
        "a.arrow",
        "b.arrow",
    ]
    ledger = calibration_tensor_ledger(payload)
    assert ledger["input_ids"]["shape"] == [1, 1, 4]


def test_deterministic_row_indices_are_stable_bounded_and_sorted():
    first = deterministic_row_indices(100, 17, 42)
    second = deterministic_row_indices(100, 17, 42)

    assert first == second
    assert first == sorted(set(first))
    assert len(first) == 17
    assert min(first) >= 0 and max(first) < 100
    assert deterministic_row_indices(4, 99, 42) == [0, 1, 2, 3]


def test_tensor_magnitude_summary_exposes_range_and_tail_energy():
    summary = tensor_magnitude_summary(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    assert summary["elements"] == 4
    assert summary["abs_max"] == 3.0
    assert summary["rms"] == pytest.approx(math.sqrt(3.5))
    assert summary["zero_fraction"] == 0.25
    assert summary["top_1pct_energy_fraction"] == pytest.approx(9.0 / 14.0)
    assert summary["distribution_sampling"] == "exact"

    large = torch.arange(100, dtype=torch.float32)
    sampled = tensor_magnitude_summary(large, max_distribution_samples=4)
    assert sampled["elements"] == 100
    assert sampled["distribution_sample_elements"] == 4
    assert sampled["distribution_sampling"] == "deterministic-stratified-midpoint-v1"
    assert sampled["top_1pct_energy_fraction_is_sampled"] is True
    assert sampled["abs_max"] == 99.0
    assert sampled["rms"] == pytest.approx(float(torch.sqrt(large.square().mean())))


def test_layer_writer_flushes_forward_and_backward_evidence(tmp_path):
    writer = LayerCaptureWriter(
        tmp_path / "capture",
        batch_size=1,
        seq_len=4,
        sample_rows=2,
        sample_seed=7,
        num_layers=1,
        bindings={"fixture": "cpu"},
    )
    value = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    value.requires_grad_(True)
    writer.capture_forward(0, "fixture.ref", value)
    writer.register_backward(
        0,
        "fixture.grad_ref",
        value,
        flush_layer=True,
    )
    writer.flush_forward(0)

    value.square().sum().backward()
    receipt = writer.finalize(loss=1.25, extra={"test": True})

    validate_receipt_seal(receipt)
    assert receipt["geometry"]["sample_rows"] == 2
    assert set(receipt["files"]) == {
        "layer_00_forward.pt",
        "layer_00_backward.pt",
    }
    manifest = load_json_object(tmp_path / "capture/capture_manifest.json")
    validate_receipt_seal(manifest)
    forward = torch.load(
        tmp_path / "capture/layer_00_forward.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(forward["tensors"]) == {"fixture.ref"}
    assert forward["tensor_metadata"]["fixture.ref"]["magnitude"]["abs_max"] > 0


def test_tt_equivalent_hf_rope_reconstructs_interleaved_complex_rotation():
    sequence = 3
    interleaved = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.bfloat16
    ).expand(1, 1, sequence, 4).clone()
    half_split = torch.stack(
        (
            interleaved[..., 0],
            interleaved[..., 2],
            interleaved[..., 1],
            interleaved[..., 3],
        ),
        dim=-1,
    )
    angles = torch.tensor(
        [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]], dtype=torch.float32
    )
    frequencies = torch.polar(torch.ones_like(angles), angles)
    cos = torch.cat((frequencies.real, frequencies.real), dim=-1)
    sin = torch.cat((frequencies.imag, frequencies.imag), dim=-1)
    cos = cos.to(torch.bfloat16).unsqueeze(0)
    sin = sin.to(torch.bfloat16).unsqueeze(0)

    query, key = _tt_equivalent_hf_rope(
        half_split,
        half_split,
        cos,
        sin,
        torch.arange(sequence).unsqueeze(0),
        freqs_cis=frequencies,
    )
    complex_input = torch.view_as_complex(
        interleaved.float().reshape(1, 1, sequence, 2, 2)
    )
    expected = torch.view_as_real(
        complex_input * frequencies.view(1, 1, sequence, 2)
    ).flatten(-2).to(torch.bfloat16)

    torch.testing.assert_close(query, expected, rtol=0, atol=0)
    torch.testing.assert_close(key, expected, rtol=0, atol=0)


def test_tt_equivalent_hf_rope_uses_pinned_frequencies_not_rounded_carrier():
    query = torch.ones((1, 1, 2, 4), dtype=torch.bfloat16)
    frequencies = torch.polar(
        torch.ones((2, 2), dtype=torch.float32),
        torch.tensor([[0.0, 0.0], [0.3, 0.7]], dtype=torch.float32),
    )
    carrier_angles = torch.tensor(
        [[0.0, 0.0], [0.301, 0.699]], dtype=torch.float32
    )
    carrier = torch.polar(torch.ones_like(carrier_angles), carrier_angles)
    cos = torch.cat((carrier.real, carrier.real), dim=-1).to(torch.bfloat16)[None]
    sin = torch.cat((carrier.imag, carrier.imag), dim=-1).to(torch.bfloat16)[None]

    actual, _ = _tt_equivalent_hf_rope(
        query,
        query,
        cos,
        sin,
        torch.arange(2).unsqueeze(0),
        freqs_cis=frequencies,
    )
    interleaved = torch.stack((query[..., :2], query[..., 2:]), dim=-1).flatten(-2)
    complex_input = torch.view_as_complex(interleaved.float().reshape(1, 1, 2, 2, 2))
    expected = torch.view_as_real(
        complex_input * frequencies.view(1, 1, 2, 2)
    ).flatten(-2).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="unit circle"):
        _tt_equivalent_hf_rope(
            query,
            query,
            cos,
            torch.zeros_like(sin),
            torch.arange(2).unsqueeze(0),
            freqs_cis=frequencies,
        )


def test_chunked_ce_hidden_gradient_matches_direct_reference():
    torch.manual_seed(3)
    hidden = torch.randn(2, 3, 5)
    head = torch.nn.Linear(5, 11, bias=False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    targets = torch.randint(0, 11, (2, 3))

    loss, gradient = chunked_causal_ce_hidden_gradient(
        hidden,
        head,
        targets,
        chunk_tokens=2,
    )
    direct_hidden = hidden.detach().requires_grad_(True)
    direct_loss = torch.nn.functional.cross_entropy(
        head(direct_hidden).reshape(-1, 11),
        targets.reshape(-1),
    )
    direct_gradient = torch.autograd.grad(direct_loss, direct_hidden)[0]

    assert loss == pytest.approx(float(direct_loss.detach()), rel=1e-6)
    torch.testing.assert_close(gradient, direct_gradient)
