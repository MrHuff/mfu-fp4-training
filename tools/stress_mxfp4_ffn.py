#!/usr/bin/env python3
"""Stress production-shape fused MXFP4 blocks alongside optional NCCL work."""

from __future__ import annotations

import argparse
import os
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=14336)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument(
        "--layer-kind",
        choices=("ffn", "qkv", "block"),
        default="ffn",
        help="Fused block to place inside the FSDP prefetch chain.",
    )
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cde-emit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exact-cde",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Carry the production exact C/D/E row-RMS partial between block layers.",
    )
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--fsdp-layers",
        type=int,
        default=0,
        help="Wrap this many fused FFN layers in composable FSDP2.",
    )
    parser.add_argument(
        "--fsdp-reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable FSDP reduce-scatter while retaining unshard/all-gather work.",
    )
    parser.add_argument(
        "--checkpoint-layer-frequency",
        type=int,
        default=0,
        help=(
            "Checkpoint every Nth fused layer before FSDP wrapping. A value equal "
            "to --fsdp-layers reproduces the production AC=1/N last-layer replay."
        ),
    )
    parser.add_argument(
        "--overlap-all-reduce-mib",
        type=int,
        default=0,
        help="Size of a rotating NCCL all-reduce launched after each backward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.exact_cde and (args.layer_kind != "block" or args.fsdp_layers < 2):
        raise ValueError(
            "--exact-cde requires --layer-kind block and at least 2 FSDP layers"
        )
    for name in ("FP4_MXFP4_ROOT", "FP4_MATMUL_ROOT", "FP4_MATMUL_GEMM_ROOT"):
        os.environ[name] = args.runtime_root

    import torch
    import torch.distributed as dist

    from low_bits_training.quantization import mxfp4_fused_linear

    FusedAttentionMXFP4_TK = mxfp4_fused_linear.FusedAttentionMXFP4_TK
    FusedFeedForwardMXFP4_TK = mxfp4_fused_linear.FusedFeedForwardMXFP4_TK
    _drain_mxfp4_fsdp_backward_prefetch = getattr(
        mxfp4_fused_linear,
        "_drain_mxfp4_fsdp_backward_prefetch",
        lambda _device: False,
    )
    from low_bits_training.quantization.mxfp4_tk_converter import (
        _FusedAttentionWrapper,
    )
    from torchtitan.models.attention import ScaledDotProductAttentionWrapper

    if args.distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device(args.device)
        rank = 0
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)

    class AttentionSpec(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_heads = 32
            self.n_kv_heads = 8
            self.head_dim = 128
            self.use_flex_attn = False
            self.inner_attention = ScaledDotProductAttentionWrapper()

    def make_attention_wrapper() -> _FusedAttentionWrapper:
        fused = FusedAttentionMXFP4_TK(
            dim=args.dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            device=device,
            dtype=torch.bfloat16,
        )
        return _FusedAttentionWrapper(AttentionSpec(), fused)

    if args.fsdp_layers:
        if not args.distributed:
            raise ValueError("--fsdp-layers requires --distributed")
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy

        class DrainBackwardPrefetch(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value: torch.Tensor) -> torch.Tensor:
                return value

            @staticmethod
            def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
                _drain_mxfp4_fsdp_backward_prefetch(grad.device)
                return grad

        class FusedLayer(torch.nn.Module):
            def __init__(self, *, has_next: bool) -> None:
                super().__init__()
                self.has_next = has_next
                if args.layer_kind == "ffn":
                    self.block = FusedFeedForwardMXFP4_TK(
                        dim=args.dim,
                        hidden_dim=args.hidden_dim,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                elif args.layer_kind == "qkv":
                    self.block = FusedAttentionMXFP4_TK(
                        dim=args.dim,
                        n_heads=32,
                        n_kv_heads=8,
                        head_dim=128,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                else:
                    self.attention = make_attention_wrapper()
                    self.feed_forward = FusedFeedForwardMXFP4_TK(
                        dim=args.dim,
                        hidden_dim=args.hidden_dim,
                        device=device,
                        dtype=torch.bfloat16,
                    )

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                if isinstance(value, tuple):
                    residual, cde_partial = value
                    value = (DrainBackwardPrefetch.apply(residual), cde_partial)
                else:
                    value = DrainBackwardPrefetch.apply(value)
                if args.layer_kind == "ffn":
                    result = self.block.forward_with_residual(
                        value,
                        residual=value if args.residual else None,
                        cde_emit=args.cde_emit,
                    )
                    return result[0] if args.cde_emit else result
                if args.layer_kind == "qkv":
                    q, k, v = self.block.forward_qkv(value, freqs_cis)
                    kv_repeat = q.shape[-1] // k.shape[-1]
                    mixed_qkv = q + k.repeat_interleave(kv_repeat, dim=-1)
                    mixed_qkv = mixed_qkv + v.repeat_interleave(
                        kv_repeat,
                        dim=-1,
                    )
                    return mixed_qkv
                if args.exact_cde:
                    if isinstance(value, tuple):
                        residual, cde_partial = value
                    else:
                        residual, cde_partial = value, None
                    attention_output = self.attention.forward_with_cde_partial(
                        residual,
                        freqs_cis,
                        None,
                        input_cde_partial=cde_partial,
                    )
                    if args.residual:
                        attention_output = attention_output + residual
                    return self.feed_forward.forward_with_residual(
                        attention_output,
                        residual=attention_output if args.residual else None,
                        cde_emit=self.has_next,
                    )
                attention_output = self.attention.forward_with_cde_partial(
                    value,
                    freqs_cis,
                    None,
                )
                if args.residual:
                    attention_output = attention_output + value
                result = self.feed_forward.forward_with_residual(
                    attention_output,
                    residual=attention_output if args.residual else None,
                    cde_emit=args.cde_emit,
                )
                return result[0] if args.cde_emit else result

        class FusedStack(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    FusedLayer(has_next=index + 1 < args.fsdp_layers)
                    for index in range(args.fsdp_layers)
                )
                if args.checkpoint_layer_frequency:
                    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                        checkpoint_wrapper,
                    )

                    frequency = args.checkpoint_layer_frequency
                    if frequency < 1:
                        raise ValueError(
                            "--checkpoint-layer-frequency must be non-negative"
                        )
                    for index, layer in enumerate(self.layers, start=1):
                        if index % frequency == 0:
                            self.layers[index - 1] = checkpoint_wrapper(
                                layer,
                                preserve_rng_state=False,
                                determinism_check="default",
                                early_stop=False,
                                debug=False,
                            )

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    value = layer(value)
                return value

        module = FusedStack()
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        preserve_input_dtype_mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}
        for layer in module.layers:
            layer_fsdp_config = fsdp_config
            if args.exact_cde:
                layer_fsdp_config = {
                    **fsdp_config,
                    "mp_policy": preserve_input_dtype_mp_policy,
                }
            fully_shard(
                layer,
                **layer_fsdp_config,
                reshard_after_forward=args.fsdp_reshard_after_forward,
            )
        fully_shard(module, **fsdp_config)
        if not args.gradient_sync:
            module.set_requires_gradient_sync(False)
        for index, layer in enumerate(module.layers[:-1]):
            layer.set_modules_to_forward_prefetch([module.layers[index + 1]])
        if args.fsdp_reshard_after_forward:
            reversed_layers = list(reversed(module.layers))
            for index, layer in enumerate(reversed_layers[:-1]):
                layer.set_modules_to_backward_prefetch([reversed_layers[index + 1]])
    else:
        if args.layer_kind == "ffn":
            module = FusedFeedForwardMXFP4_TK(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                device=device,
                dtype=torch.bfloat16,
            )
        elif args.layer_kind == "qkv":
            module = FusedAttentionMXFP4_TK(
                dim=args.dim,
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                device=device,
                dtype=torch.bfloat16,
            )
        else:
            module = torch.nn.ModuleDict(
                {
                    "attention": make_attention_wrapper(),
                    "feed_forward": FusedFeedForwardMXFP4_TK(
                        dim=args.dim,
                        hidden_dim=args.hidden_dim,
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                }
            )
    module.train()
    if args.layer_kind in ("qkv", "block"):
        if args.rows % args.seq_len:
            raise ValueError("--rows must be divisible by --seq-len for QKV stress")
        from torchtitan.models.llama3.model.model import precompute_freqs_cis

        batch_size = args.rows // args.seq_len
        freqs_cis = precompute_freqs_cis(128, args.seq_len).to(device)
        x_shape = (batch_size, args.seq_len, args.dim)
    else:
        freqs_cis = None
        x_shape = (args.rows, args.dim)
    x = torch.randn(*x_shape, device=device, dtype=torch.bfloat16)
    x.requires_grad_(True)

    comm_buffers: list[torch.Tensor] = []
    comm_handles = [None, None]
    if args.overlap_all_reduce_mib:
        if not args.distributed:
            raise ValueError("--overlap-all-reduce-mib requires --distributed")
        elements = args.overlap_all_reduce_mib * 1024 * 1024 // 2
        comm_buffers = [
            torch.ones(elements, device=device, dtype=torch.bfloat16),
            torch.ones(elements, device=device, dtype=torch.bfloat16),
        ]

    torch.cuda.synchronize(device)
    started = None
    measured = 0
    for iteration in range(args.warmup + args.iterations):
        slot = iteration % 2
        handle = comm_handles[slot]
        if handle is not None:
            handle.wait()
            comm_handles[slot] = None

        if args.fsdp_layers:
            result = module(x)
            output = result
        else:
            if args.layer_kind == "ffn":
                result = module.forward_with_residual(
                    x,
                    residual=x if args.residual else None,
                    cde_emit=args.cde_emit,
                )
                output = result[0] if args.cde_emit else result
            else:
                if args.layer_kind == "block":
                    output = module["attention"].forward_with_cde_partial(
                        x,
                        freqs_cis,
                        None,
                    )
                    if args.residual:
                        output = output + x
                    result = module["feed_forward"].forward_with_residual(
                        output,
                        residual=output if args.residual else None,
                        cde_emit=args.cde_emit,
                    )
                    output = result[0] if args.cde_emit else result
                else:
                    q, k, v = module.forward_qkv(x, freqs_cis)
                    kv_repeat = q.shape[-1] // k.shape[-1]
                    output = q + k.repeat_interleave(kv_repeat, dim=-1)
                    output = output + v.repeat_interleave(
                        kv_repeat,
                        dim=-1,
                    )
                    result = (q, k, v)
        loss = output.float().mean()
        loss.backward()

        if comm_buffers:
            comm_handles[slot] = dist.all_reduce(comm_buffers[slot], async_op=True)

        if iteration == args.warmup - 1:
            for pending in comm_handles:
                if pending is not None:
                    pending.wait()
            comm_handles = [None, None]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        elif iteration >= args.warmup:
            measured += 1

        if (iteration + 1) % args.report_every == 0:
            torch.cuda.synchronize(device)
            finite = bool(torch.isfinite(output).all().item())
            if not finite:
                raise RuntimeError(f"non-finite output at iteration {iteration + 1}")
            if rank == 0:
                print(
                    f"iteration={iteration + 1} measured={measured} "
                    f"loss={loss.item():.8f} finite={finite}",
                    flush=True,
                )

        module.zero_grad(set_to_none=True)
        x.grad = None
        del result, output, loss

    for handle in comm_handles:
        if handle is not None:
            handle.wait()
    torch.cuda.synchronize(device)
    assert started is not None
    elapsed = time.perf_counter() - started
    if rank == 0:
        print(
            f"PASS iterations={args.iterations} elapsed_s={elapsed:.3f} "
            f"ms_per_iteration={elapsed * 1000 / args.iterations:.3f}",
            flush=True,
        )
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
