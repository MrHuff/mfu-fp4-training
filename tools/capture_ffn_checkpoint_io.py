#!/usr/bin/env python3
"""Capture one converted FFN's real input and output gradient from a checkpoint."""

import argparse
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from low_bits_training.generate.generate import Generator
from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4_TK


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _find_ffn(model, layer):
    suffix = f"layers.{layer}.feed_forward"
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, FusedFeedForwardFP4_TK)
        and (name == suffix or name.endswith(f".{suffix}"))
    ]
    if len(matches) != 1:
        available = [
            name
            for name, module in model.named_modules()
            if isinstance(module, FusedFeedForwardFP4_TK)
        ]
        raise RuntimeError(
            f"expected one converted FFN for layer {layer}, found {matches}; "
            f"available={available}"
        )
    return matches[0]


def main():
    args = _parse_args()
    if args.seq_len <= 1:
        raise ValueError("seq-len must be greater than one")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = Generator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        deterministic=args.deterministic,
        add_kv_cache=False,
        device=args.device,
    )
    model = generator.model
    compute_dtype_name = generator.config.training.mixed_precision_param
    try:
        compute_dtype = getattr(torch, compute_dtype_name)
    except AttributeError as error:
        raise ValueError(
            f"unsupported mixed-precision parameter dtype: {compute_dtype_name}"
        ) from error
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != compute_dtype:
            parameter.data = parameter.data.to(dtype=compute_dtype)
        parameter.requires_grad_(False)
    model.train()
    module_name, module = _find_ffn(model, args.layer)

    capture = {}
    original = module.forward_with_residual

    def wrapped(self, x, residual=None, cde_row_rms_partial=None, cde_emit=False):
        capture["input"] = x.detach().clone()
        output = original(
            x,
            residual=residual,
            cde_row_rms_partial=cde_row_rms_partial,
            cde_emit=cde_emit,
        )
        primary = output[0] if isinstance(output, tuple) else output
        primary = primary.detach().requires_grad_(True)

        def save_upstream(gradient):
            capture["upstream"] = gradient.detach().clone()

        primary.register_hook(save_upstream)
        if isinstance(output, tuple):
            return (primary, *output[1:])
        return primary

    module.forward_with_residual = types.MethodType(wrapped, module)

    device = torch.device(args.device)
    if args.input_text:
        text = Path(args.input_text).read_text()
        token_ids = generator.tokenizer.encode(
            text,
            add_bos=True,
            add_eos=False,
            allowed_special="all",
        )
        required_tokens = args.batch_size * args.seq_len
        if len(token_ids) < required_tokens:
            raise ValueError(
                f"input text produced {len(token_ids)} tokens, "
                f"but {required_tokens} are required"
            )
        tokens = torch.tensor(
            token_ids[:required_tokens], device=device, dtype=torch.long
        ).view(args.batch_size, args.seq_len)
    else:
        vocab_size = generator.tokenizer.get_vocab_size()
        tokens = torch.randint(
            0,
            vocab_size,
            (args.batch_size, args.seq_len),
            device=device,
            dtype=torch.long,
        )
    labels = tokens.roll(-1, dims=1)
    labels[:, -1] = -100
    try:
        result = model(tokens, labels=labels)
    except TypeError:
        logits = model(tokens)
        result = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
    loss = result[0] if isinstance(result, tuple) else result
    if loss.numel() != 1:
        raise RuntimeError(f"expected a scalar loss, found shape {tuple(loss.shape)}")
    loss.backward()
    torch.cuda.synchronize(device)

    if capture.keys() != {"input", "upstream"}:
        raise RuntimeError(f"incomplete FFN capture: {sorted(capture)}")
    payload = {
        "input": capture["input"].reshape(-1, module.dim).cpu(),
        "upstream": capture["upstream"].reshape(-1, module.dim).cpu(),
        "layer": args.layer,
        "module_name": module_name,
        "loss": float(loss.detach()),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "seq_len": args.seq_len,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "input_shape": list(payload["input"].shape),
                "batch_size": args.batch_size,
                "layer": args.layer,
                "loss": payload["loss"],
                "module_name": module_name,
                "output": str(output_path),
                "upstream_shape": list(payload["upstream"].shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
