from __future__ import annotations

import torch


WORLD_SIZE = 32
LOCALCTA_PRODUCERS = 108
MXFP4_PRODUCERS = 20
LOCALCTA_SEED_BASE = 0
MXFP4_SEED_BASE = 1 << 40
SUBSEQUENCE_BASE = 0


class _Logger:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def info(self, message: str, *args: object) -> None:
        self.events.append(message % args if args else message)

    def warning(self, message: str, *args: object) -> None:
        self.events.append(message % args if args else message)


class _Checkpointer:
    def __init__(self, payload: dict[str, dict[str, object]], events: list[str]) -> None:
        self.payload = payload
        self.events = events
        self.states: dict[str, object] = {}
        self.ft_manager = None

    def dcp_load(
        self,
        state_dict: dict[str, object],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        assert checkpoint_id == "/sealed/hybrid/step-1000"
        assert not from_hf and not from_quantized
        for key in ("localcta_sr_state", "mxfp4_sr_state"):
            self.events.append(f"load:{key}")
            state_dict[key].load_state_dict(self.payload[key])


def test_hybrid_restores_both_independent_sr_states_before_next_backward(
    monkeypatch,
) -> None:
    from low_bits_training.quantization import localcta_sr_state as localcta
    from low_bits_training.quantization import mxfp4_sr_state as mxfp4

    local_key = "layers.0.attention:qkv:sr:qkv_grad"
    mx_key = "layers.28.attention:qkv:sr:qkv_grad"
    source_local = localcta.LocalCTASRState(
        (local_key,),
        device="cpu",
        user_seed=LOCALCTA_SEED_BASE,
        user_subsequence_base=SUBSEQUENCE_BASE,
        training_steps=71526,
        gradient_accumulation_steps=4,
        rank=0,
        world_size=1,
    )
    source_mx = mxfp4.MXFP4SRState(
        (mx_key,),
        device="cpu",
        user_seed=MXFP4_SEED_BASE,
        user_subsequence_base=SUBSEQUENCE_BASE,
        training_steps=71526,
        gradient_accumulation_steps=4,
        rank=0,
        world_size=1,
    )
    source_local.get(local_key)[1].fill_(2 * localcta.SUBSEQUENCE_STRIDE)
    mx_seed, first_mx_subsequence = source_mx.reserve(mx_key)
    assert first_mx_subsequence == 0
    payload = {
        localcta.CHECKPOINT_KEY: source_local.state_dict(),
        mxfp4.CHECKPOINT_KEY: source_mx.state_dict(),
    }
    assert int(payload[localcta.CHECKPOINT_KEY]["seed_base"].item()) == 0
    assert int(payload[mxfp4.CHECKPOINT_KEY]["seed_base"].item()) == MXFP4_SEED_BASE
    assert int(payload[mxfp4.CHECKPOINT_KEY]["subsequence_base"].item()) == 0

    resumed_local = localcta.LocalCTASRState(
        (local_key,),
        device="cpu",
        user_seed=LOCALCTA_SEED_BASE,
        user_subsequence_base=SUBSEQUENCE_BASE,
        training_steps=71526,
        gradient_accumulation_steps=4,
        rank=0,
        world_size=1,
    )
    resumed_mx = mxfp4.MXFP4SRState(
        (mx_key,),
        device="cpu",
        user_seed=MXFP4_SEED_BASE,
        user_subsequence_base=SUBSEQUENCE_BASE,
        training_steps=71526,
        gradient_accumulation_steps=4,
        rank=0,
        world_size=1,
    )
    events: list[str] = []
    checkpointer = _Checkpointer(payload, events)
    logger = _Logger(events)
    monkeypatch.setattr(
        localcta, "checkpoint_localcta_sr_state_schema", lambda _: "v2"
    )
    monkeypatch.setattr(
        mxfp4, "checkpoint_mxfp4_sr_state_schema", lambda _: "v1"
    )
    localcta.register_with_checkpointer(checkpointer, resumed_local, logger)
    mxfp4.register_with_checkpointer(checkpointer, resumed_mx, logger)

    assert set(checkpointer.states) == {"localcta_sr_state", "mxfp4_sr_state"}
    assert checkpointer.states["localcta_sr_state"] is resumed_local
    assert checkpointer.states["mxfp4_sr_state"] is resumed_mx
    checkpointer.dcp_load(
        checkpointer.states,
        checkpoint_id="/sealed/hybrid/step-1000",
        from_hf=False,
        from_quantized=False,
    )
    events.append("next_backward")

    assert events.index("load:localcta_sr_state") < events.index("next_backward")
    assert events.index("load:mxfp4_sr_state") < events.index("next_backward")
    assert any(
        "Restored checkpointed MXFP4 SR ABI v1" in event
        and "before the next backward" in event
        for event in events
    )
    assert int(resumed_local.get(local_key)[1].item()) == (
        2 * localcta.SUBSEQUENCE_STRIDE
    )
    next_mx_seed, next_mx_subsequence = resumed_mx.reserve(mx_key)
    assert next_mx_seed == mx_seed
    assert next_mx_subsequence == mxfp4.SUBSEQUENCE_STRIDE
    assert resumed_local.user_subsequence_base == SUBSEQUENCE_BASE
    assert resumed_mx.user_subsequence_base == SUBSEQUENCE_BASE


def test_hybrid_world32_seed_namespaces_are_exact_unique_and_disjoint() -> None:
    from low_bits_training.quantization import localcta_sr_state as localcta
    from low_bits_training.quantization import mxfp4_sr_state as mxfp4

    local_keys = tuple(f"localcta-producer-{index:03d}" for index in range(LOCALCTA_PRODUCERS))
    mx_keys = tuple(f"mxfp4-producer-{index:03d}" for index in range(MXFP4_PRODUCERS))
    local_seeds: set[int] = set()
    mx_seeds: set[int] = set()

    for rank in range(WORLD_SIZE):
        local_state = localcta.LocalCTASRState(
            local_keys,
            device="cpu",
            user_seed=LOCALCTA_SEED_BASE,
            user_subsequence_base=SUBSEQUENCE_BASE,
            training_steps=71526,
            gradient_accumulation_steps=4,
            rank=rank,
            world_size=WORLD_SIZE,
        )
        mx_state = mxfp4.MXFP4SRState(
            mx_keys,
            device="cpu",
            user_seed=MXFP4_SEED_BASE,
            user_subsequence_base=SUBSEQUENCE_BASE,
            training_steps=71526,
            gradient_accumulation_steps=4,
            rank=rank,
            world_size=WORLD_SIZE,
        )
        for key in local_keys:
            seed, subsequence = (int(value.item()) for value in local_state.get(key))
            local_seeds.add(seed)
            assert subsequence == SUBSEQUENCE_BASE
        for key in mx_keys:
            seed, subsequence = mx_state.peek(key)
            mx_seeds.add(seed)
            assert subsequence == SUBSEQUENCE_BASE

    assert len(local_seeds) == WORLD_SIZE * LOCALCTA_PRODUCERS
    assert len(mx_seeds) == WORLD_SIZE * MXFP4_PRODUCERS
    assert local_seeds == set(range(1, WORLD_SIZE * LOCALCTA_PRODUCERS + 1))
    assert mx_seeds == set(
        range(MXFP4_SEED_BASE + 1, MXFP4_SEED_BASE + WORLD_SIZE * MXFP4_PRODUCERS + 1)
    )
    assert local_seeds.isdisjoint(mx_seeds)


def _expected_producer_keys(first_layer: int, last_layer: int) -> tuple[str, ...]:
    keys: list[str] = []
    for layer in range(first_layer, last_layer + 1):
        keys.extend(
            (
                f"layers.{layer}.attention:qkv:sr:qkv_grad",
                f"layers.{layer}.attention:wo:sr:wo_grad",
                f"layers.{layer}.feed_forward:sr:ffn_deriv_grad",
                f"layers.{layer}.feed_forward:sr:ffn_w2_grad",
            )
        )
    return tuple(sorted(keys))


def test_hybrid_mxfp4_sr_discovers_exact_tail_five_producer_manifest() -> None:
    from low_bits_training.quantization import mxfp4_sr_state as mxfp4

    attention_type = type("FusedAttentionMXFP4_TK", (torch.nn.Module,), {})
    ffn_type = type("FusedFeedForwardMXFP4_TK", (torch.nn.Module,), {})
    model = torch.nn.Module()
    for layer in range(28, 33):
        attention = attention_type()
        attention._lbt_debug_name = f"layers.{layer - 1}.attention"
        attention._force_wo_bf16 = False
        model.add_module(f"attn_{layer}", attention)
        ffn = ffn_type()
        ffn._lbt_debug_name = f"layers.{layer - 1}.feed_forward"
        model.add_module(f"ffn_{layer}", ffn)

    keys = mxfp4.discover_logical_keys((model,))
    assert keys == _expected_producer_keys(27, 31)


def test_hybrid_localcta_sr_discovers_exact_leading_27_producer_manifest() -> None:
    from low_bits_training.quantization import localcta_sr_state as localcta

    attention_type = type("FusedAttentionFP4_TK", (torch.nn.Module,), {})
    ffn_type = type("FusedFeedForwardFP4_TK", (torch.nn.Module,), {})
    model = torch.nn.Module()
    for layer in range(27):
        attention = attention_type()
        attention._lbt_debug_name = f"layers.{layer}.attention"
        model.add_module(f"attn_{layer}", attention)
        ffn = ffn_type()
        ffn._lbt_debug_name = f"layers.{layer}.feed_forward"
        model.add_module(f"ffn_{layer}", ffn)

    assert localcta.discover_logical_keys((model,)) == _expected_producer_keys(0, 26)


def test_hybrid_explicit_route_is_localcta_1_27_and_mxfp4_28_32(
    monkeypatch,
) -> None:
    from low_bits_training.quantization import mixed_fp4_converter as mixed

    monkeypatch.setattr(mixed, "_block_layer_indices", lambda _: list(range(32)))
    monkeypatch.setenv(
        "LBT_FP4_MIXED_LAYERS", "localcta:1-27;mxfp4:28-32"
    )
    monkeypatch.setenv("LBT_FP4_MIXED_POLICY", "tail_mxfp4")
    monkeypatch.setenv("LBT_FP4_MIXED_TAIL_LAYERS", "5")

    routes = mixed._build_layer_routes(object())

    assert routes == {
        **{index: "localcta" for index in range(27)},
        **{index: "mxfp4" for index in range(27, 32)},
    }
