# localCTA v4 Bootstrap

`v3` replaces the `v1` prepared contract with a `v5`-style fast contract.

## `v1` failure mode

For one localCTA chunk with chunk amax `a_c` and outer numerator `K_cta`:

```text
S_c  = K_cta / a_c
sg_c = a_c / K_cta

M_b  = Q8(6 / (m_b * S_c))
sc_b = Q8(1 / M_b)
q_b  = Q4(x * M_b * S_c)
```

Raw reconstruction is:

```text
x_raw ~= q_b * sc_b * sg_c
```

Prepared `v1` replaces this with:

```text
sc_prepared = Q8(sc_b * sg_c)
x_prepared  ~= q_b * sc_prepared
```

In exact arithmetic:

```text
sc_b * sg_c ~= m_b / 6
```

So the prepared path is trying to store `Q8(m_b / 6)`. That is why tiny backward blocks die: the correction is quantized too early and underflows in `e4m3`.

## `v3` design target

`v3` keeps:

- FP4 payload `q_b`
- FP8 microscales `sc_b`
- FP32 outer scales `sg`

`v3` never computes or consumes `Q8(sc_b * sg)`.

The outer correction is GEMM-tile aligned and K-invariant:

- `row_sg_v3[row_tile]`
- `col_sg_v3[col_tile]`

First landing geometry is fixed:

- `Mb = 256`
- `Nb = 256`
- `Kb = 256`

That means each `v3` localCTA tile corresponds directly to one fast GEMM tile.

## Quantization contract

For each GEMM-aligned operand tile `t`:

```text
a_t  = max_{(i,j) in tile t} |x_ij|
S_t  = K_v3 / a_t
sg_t = a_t / K_v3

M_b  = Q8(6 / (m_b * S_t))
sc_b = Q8(1 / M_b)
q_b  = Q4(x * M_b * S_t)
```

Reconstruction is:

```text
x_v3 ~= q_b * sc_b * sg_t
```

The important invariant is:

- `sg_t` is constant across the full K reduction for that GEMM tile
- only `sc_b` varies inside the tile

This is what makes a fast `v5`-style consumer possible.

## Consumer contract

Fast `v3` GEMM consumes:

```text
q_fp4 + sc_fp8 + row_sg_fp32 + col_sg_fp32
```

The full K reduction runs exactly like regular fast NVFP4 GEMM.
After accumulation, the consumer applies:

```text
output_tile *= row_sg_v3[row_tile] * col_sg_v3[col_tile]
```

once per output tile, in `float`.

No scale rewriting in shared memory.
No per-K consumer accumulation.
No folded prepared scales.

## Implementation order

1. Regular GEMM
2. Grouped QKV
3. Batched FFN
4. Batched-accum FFN
5. Split2 / split3 specialized backward kernels only after the generic path is fast

`v1` prepared remains the performance and regression baseline until `v3` clears parity and MFU gates.
