#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from pathlib import Path

import pytest

import low_bits_training  # noqa

from low_bits_training.config import JobConfig
from low_bits_training.generate.generate import Generator

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cpu_generation_unit_test_model(
    config_and_checkpoint: tuple[JobConfig, Path],
    capsys: pytest.CaptureFixture,
    caplog: pytest.CaptureFixture,
    device="cpu",
):
    config, checkpoint = config_and_checkpoint
    generator = Generator(
        config_path=config,
        checkpoint_path=checkpoint,
        device=device,
        seed=33,
    )
    # Check that the model was converted correctly

    # NOTE: This checks both caplog and capsys. When using capsys, the correct string shows up in the logs,
    # but not in stdout, while without capsys, it shows up in stdout but not in the logs. No idea why,
    # but checking both works.
    capture_sys = capsys.readouterr().out
    capture_log = caplog.text

    exp_attn_found_str = "No attention modules found"
    kv_cache_success_str = "Successfully added KV caching to the model"
    assert (
        exp_attn_found_str not in capture_sys and exp_attn_found_str not in capture_log
    ), "Failed to find attention modules to replace"
    assert (
        kv_cache_success_str in capture_sys or kv_cache_success_str in capture_log
    ), "Failed to add KV caching to the model"

    # -- check that the model can generate --
    # Our tokenizer is a truncated test tokenizer so it can't tokenize actual words
    prompt = generator.tokenizer.decode([0, 1])

    out = generator.generate(prompt=prompt, max_new_tokens=5)

    assert out["metadata"]["generated_n_tokens"] == 5
    assert out["metadata"]["input_n_tokens"] == 3
    assert out["metadata"]["batch_size"] == 1
    assert out["responses"][0]["input_text"] == f"<|begin_of_text|>{prompt}"
    assert out["responses"][0][
        "output_text"
    ], "There should be something in the output - event if its noise"
    assert len(out["responses"]) == 1

    out = generator.generate(prompt=[prompt, prompt], max_new_tokens=5, batch_size=2)

    assert out["metadata"]["generated_n_tokens"] == 5
    assert out["metadata"]["input_n_tokens"] == 3
    assert out["metadata"]["batch_size"] == 2
    assert out["responses"][0]["input_text"] == f"<|begin_of_text|>{prompt}"
    assert out["responses"][0][
        "output_text"
    ], "There should be something in the output - event if its noise"
    assert len(out["responses"]) == 2
