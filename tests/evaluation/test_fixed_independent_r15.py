from pathlib import Path
import importlib.util
import sys

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/evaluation/fixed_independent_r15/materialize_fixed_independent.py"
)
SPEC = importlib.util.spec_from_file_location("fixed_independent", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def document(source: str, sample_id: int, token_count: int):
    return MODULE.SelectedDocument(
        source=source,
        sample_id=sample_id,
        hf_global_id=sample_id,
        text_sha256=MODULE.sha256_bytes(f"text-{source}-{sample_id}".encode()),
        meta_sha256=MODULE.sha256_bytes(f"meta-{source}-{sample_id}".encode()),
        text_bytes=10,
        full_token_count=token_count,
        selected_tokens=list(range(token_count)),
        source_ordinal=sample_id,
        key=MODULE.shuffle_key(source, sample_id, f"{sample_id:064x}", MODULE.SEED),
    )


def test_geometry_and_exact_strata():
    assert MODULE.SEQUENCES == 768
    assert MODULE.STORED_WIDTH == 8193
    assert MODULE.STORED_TOKENS == 6_292_224
    assert MODULE.VALIDATION_TOKENS == 6_291_456
    assert MODULE.DCLM_TOKENS == 5_159_624
    assert MODULE.OLMO_TOKENS == 1_132_600
    assert MODULE.DCLM_TOKENS + MODULE.OLMO_TOKENS == MODULE.STORED_TOKENS


def test_deterministic_start_stays_inside_sealed_band():
    for source, spec in MODULE.SOURCE_SPECS.items():
        start = MODULE.deterministic_start(source, spec["samples"], MODULE.SEED)
        assert spec["samples"] * 90 // 100 <= start
        assert start < spec["samples"] * 95 // 100 + 1
        assert start == MODULE.deterministic_start(source, spec["samples"], MODULE.SEED)


def test_shuffle_key_binds_source_sample_text_and_seed():
    base = MODULE.shuffle_key("dclm", 1, "a" * 64, MODULE.SEED)
    assert len(base) == 64
    assert base != MODULE.shuffle_key("olmo-no-dclm", 1, "a" * 64, MODULE.SEED)
    assert base != MODULE.shuffle_key("dclm", 2, "a" * 64, MODULE.SEED)
    assert base != MODULE.shuffle_key("dclm", 1, "b" * 64, MODULE.SEED)
    assert base != MODULE.shuffle_key("dclm", 1, "a" * 64, MODULE.SEED + 1)


def test_small_pack_contract_with_temporary_geometry(monkeypatch):
    monkeypatch.setattr(MODULE, "SEQUENCES", 2)
    monkeypatch.setattr(MODULE, "STORED_WIDTH", 5)
    monkeypatch.setattr(MODULE, "STORED_TOKENS", 10)
    monkeypatch.setattr(MODULE, "VOCAB_SIZE", 1000)
    docs = [document("dclm", 0, 4), document("olmo-no-dclm", 1, 6)]
    packed, source_ledger, packing_ledger = MODULE.build_packed_tensor(docs)
    assert packed.shape == (2, 5)
    assert packed.dtype == torch.int32
    assert len(source_ledger) == 2
    assert len(packing_ledger) == 2
    assert [row["row"] for row in packing_ledger] == [0, 1]
    assert all(row["segments"] for row in packing_ledger)


def test_manifest_seal_rejects_mutation():
    geometry = {
        "world_size": MODULE.WORLD_SIZE,
        "sequences_per_rank": MODULE.SEQUENCES_PER_RANK,
        "sequences": MODULE.SEQUENCES,
        "stored_tokens_per_sequence": MODULE.STORED_WIDTH,
        "scored_tokens_per_sequence": MODULE.SEQUENCE_LENGTH,
        "stored_tokens": MODULE.STORED_TOKENS,
        "validation_tokens": MODULE.VALIDATION_TOKENS,
        "padding": False,
    }
    document_value = {
        "schema": MODULE.SCHEMA,
        "stream_id": MODULE.STREAM_ID,
        "claim": "fixed-independent-not-proven-held-out",
        "geometry": geometry,
        "stratification": {
            "selected_tokens": {
                "dclm": MODULE.DCLM_TOKENS,
                "olmo-no-dclm": MODULE.OLMO_TOKENS,
            }
        },
        "shards": [{"rank": rank} for rank in range(MODULE.WORLD_SIZE)],
    }
    sealed = MODULE.seal_document(document_value, "manifest_sha256")
    MODULE.validate_manifest(sealed)
    sealed["claim"] = "held-out"
    try:
        MODULE.validate_manifest(sealed)
    except RuntimeError:
        pass
    else:
        raise AssertionError("mutated manifest unexpectedly passed")
