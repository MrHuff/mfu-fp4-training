# MXFP4 GB200 selector evidence

## Acceptance

- Base LBT commit: `5903f5272ce7144f3ee2fea2ec157b904b125817`
- fp4_matmul commit: `0e9ab834519287a6c96cd723109146fd691c85cf`
- Immutable quant/GEMM root:
  `/tmp/fp4_matmul_volt_runtime_sm100_0e9ab834`
- Discovery quant/GEMM root:
  `/tmp/fp4_matmul_nemotron_h_8b_pure_tk_20260720` (clean at the
  pinned commit when each retained discovery worker ran)
- GEMM extension SHA256:
  `5308af6a7c559c95c61794bc234a9ca1c28e978d8b6b1f25ceb588fbd0793616`
- Quant extension SHA256:
  `48742f8bf31595eabdeec2fa33acf53f7c011f3ec2b6e38911b024b9f14f506e`
- Hardware: NVIDIA GB200, CUDA capability 10.0. Physical GPU 0 UUID
  `5a755dd4-1813-db08-ed72-5a7aca637ac2`; physical GPU 3 UUID
  `cc696c13-a01c-d220-ae39-838ee8bdde6b`.
- Acceptance threshold: config median at least 0.5% faster than its paired
  native entrypoint, at least two-thirds paired-trial wins, and bitwise BF16
  parity in both uncontended runs.
- Timing is native TK CUDA only. Each case/config ran in an isolated process;
  parent and worker `nvidia-smi pmon` guards rejected foreign GPU PIDs.

## Retained selectors

All retained rows use config 10 and had zero parity violations.

| Orientation | M | N | K | Discovery native/config ms | Win | Confirm native/config ms | Win | Paired wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 32768 | 4096 | 8192 | 0.340763 / 0.338148 | 0.773% | 0.361603 / 0.349495 | 3.464% | 8/9, 9/11 |
| dgrad | 32768 | 8192 | 4096 | 0.383414 / 0.378746 | 1.233% | 0.394922 / 0.376477 | 4.899% | 8/9, 8/11 |
| forward | 32768 | 21504 | 4096 | 1.117063 / 1.048907 | 6.498% | 1.143185 / 1.121280 | 1.954% | 9/9, 9/11 |
| dgrad | 32768 | 21504 | 4096 | 1.113767 / 1.051217 | 5.950% | 1.171649 / 1.084328 | 8.053% | 9/9, 8/11 |
| dgrad | 32768 | 4096 | 5120 | 0.225641 / 0.222310 | 1.498% | 0.224088 / 0.218886 | 2.376% | 9/9, 9/11 |
| dgrad | 32768 | 4096 | 6144 | 0.266846 / 0.258710 | 3.145% | 0.258681 / 0.252802 | 2.326% | 7/7, 9/11 |

Worker clock samples were P0 with 3996 MHz memory throughout:

| Shape (M,N,K) | Discovery physical GPU / SM MHz | Confirm physical GPU / SM MHz |
| --- | --- | --- |
| 32768,4096,8192 | 0 / 1425-2062 | 0 / 1402-2062 |
| 32768,8192,4096 | 3 / 1462-1950 | 0 / 1567-2062 |
| 32768,21504,4096 forward | 0 / 1485-1732 | 0 / 1552-1612 |
| 32768,21504,4096 dgrad | 0 / 1462-1687 | 0 / 1530-1612 |
| 32768,4096,5120 | 0 / 1492-2062 | 0 / 1515-2062 |
| 32768,4096,6144 | 0 / 1522-2062 | 0 / 1492-2062 |

## Raw artifacts

| Artifact | SHA256 |
| --- | --- |
| `/tmp/mxfp4_selector_0e9ab834_gpu0_pass1/nemotron_mamba_out_forward.json` | `0fae9c05df294e206d62ef5adb9985cd5b5d7d0c239ff7a9823084b3963ceb09` |
| `/tmp/mxfp4_selector_0e9ab834_gpu3_pass1/nemotron_mamba_out_dgrad.json` | `d85bf6159b3bfbf6bd73d6942ee39becac4817bc5f4a585463fd305a2d320a6a` |
| `/tmp/mxfp4_selector_0e9ab834_gpu0_pass1/nemotron_mlp_w13_forward.json` | `590b4d43cdafcc14823f4add58cead3a4cb3e94faf2f744f87d6833d50fade4b` |
| `/tmp/mxfp4_selector_0e9ab834_gpu0_pass1/nemotron_mlp_w2_dgrad.json` | `b19f5317200a24f152df4c83656f99e9d287f7daa9a7518f5f948c5aea0d340e` |
| `/tmp/mxfp4_selector_0e9ab834_gpu0_pass1/nemotron_qkv_dgrad.json` | `dd3ac7868cf0165045c3854dad3fc070534234ca0fab7df6043e33c4e2616146` |
| `/tmp/mxfp4_selector_0e9ab834_immutable_discovery/llama_qkv_dgrad.json` | `094950b855986878ea30b4488b62c954d5acc8d80d340b36cf15cfc0282d8f3f` |
| `/tmp/mxfp4_selector_0e9ab834_confirm1/config10.json` | `d1c397d1d4d6dd424cd9df86ac5cc29ed1d8e966afbcf07cee4214055c3db6f3` |

## Rejected selectors

No table row is retained for Nemotron Mamba-out wgrad, W13 batched forward,
W2 residual forward, W13 one-pass dgrad, W13 batched wgrad, W2 wgrad, or QKV
one-pass dgrad. No row is retained for Llama W13 dense/batched forward, W13
one-pass dgrad, or QKV one-pass dgrad.

The residual config-10 confirmation had only 6/11 paired wins. Other rejected
cases either did not beat native in discovery or lacked a second qualifying
run. Their selector architecture is not present in the backend.
