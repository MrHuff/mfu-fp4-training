import sys
import time

import torch


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "matmul"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    pre_sleep = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if pre_sleep > 0:
        time.sleep(pre_sleep)

    if mode == "matmul":
        x = torch.randn(n, n, device="cuda", dtype=torch.float16)
        y = torch.randn(n, n, device="cuda", dtype=torch.float16)
        z = x @ y
        torch.cuda.synchronize()
        print("matmul", tuple(z.shape), float(z[0, 0]))
        return

    if mode == "elementwise":
        x = torch.randn(n * n, device="cuda", dtype=torch.float32)
        y = torch.sin(x) + torch.cos(x)
        torch.cuda.synchronize()
        print("elementwise", tuple(y.shape), float(y[0]))
        return

    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
