from __future__ import annotations

import pytest


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    monkeypatch.setenv("USE_TK_GEMM_NOPDL", "0")
    monkeypatch.delenv("USE_TK_GEMM_CONFIG", raising=False)
    monkeypatch.delenv("USE_TK_GEMM_CONFIG_ID", raising=False)
    from low_bits_training.quantization import tk_gemm

    for name in tk_gemm._TK_V5_PDL_PRODUCTION_GEMM_CONFIG_ENVS.values():
        monkeypatch.delenv(name, raising=False)
    return tk_gemm


class _Shaped:
    def __init__(self, *shape: int):
        self.shape = shape


class _FakeTK:
    def __init__(self, *, localcta: bool = False):
        self._is_localcta = localcta
        self.calls: list[tuple[str, int | None]] = []

    def nvfp4_gemm(self, *args):
        self.calls.append(("pdl", None))
        return "pdl"

    def nvfp4_gemm_nopdl(self, *args):
        self.calls.append(("nopdl", None))
        return "nopdl"

    def nvfp4_gemm_config(self, *args):
        self.calls.append(("pdl_config", args[-1]))
        return "pdl_config"

    def nvfp4_gemm_config_nopdl(self, *args):
        self.calls.append(("nopdl_config", args[-1]))
        return "nopdl_config"


def _dispatch(tk_gemm, tk, shape: tuple[int, int, int]):
    m, n, k = shape
    return tk_gemm.tk_dispatch_gemm(
        tk,
        _Shaped(m, k // 2),
        None,
        None,
        _Shaped(n, k // 2),
        None,
        None,
        _Shaped(m, n),
    )


@pytest.mark.parametrize(
    ("shape", "config_id"),
    (
        ((32768, 14336, 4096), 12),
        ((4096, 14336, 32768), 28),
        ((4096, 4096, 24576), 12),
        ((24576, 21504, 4096), 12),
        ((24576, 18688, 4096), 12),
        ((24576, 4096, 18688), 12),
        ((24576, 4096, 8192), 28),
        ((24576, 8192, 4096), 27),
    ),
)
def test_exact_production_shape_uses_validated_pdl_config(
    monkeypatch, shape, config_id
) -> None:
    tk_gemm = _load(monkeypatch)
    tk = _FakeTK()

    assert _dispatch(tk_gemm, tk, shape) == "pdl_config"
    assert tk.calls == [("pdl_config", config_id)]


def test_nonmatching_shape_keeps_native_default(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    tk = _FakeTK()

    assert _dispatch(tk_gemm, tk, (32768, 4096, 4096)) == "pdl"
    assert tk.calls == [("pdl", None)]


def test_localcta_never_consumes_regular_v5_selector(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_GEMM_CONFIG", "7")
    tk = _FakeTK(localcta=True)

    assert _dispatch(tk_gemm, tk, (32768, 14336, 4096)) == "pdl"
    assert tk.calls == [("pdl", None)]


def test_exact_then_global_then_alias_override_precedence(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    shape = (32768, 14336, 4096)
    exact_env = tk_gemm._TK_V5_PDL_PRODUCTION_GEMM_CONFIG_ENVS[shape]
    monkeypatch.setenv("USE_TK_GEMM_CONFIG_ID", "9")
    monkeypatch.setenv("USE_TK_GEMM_CONFIG", "8")
    monkeypatch.setenv(exact_env, "7")

    assert tk_gemm.get_tk_gemm_config(shape, use_production_default=True) == 7
    monkeypatch.delenv(exact_env)
    assert tk_gemm.get_tk_gemm_config(shape, use_production_default=True) == 8
    monkeypatch.delenv("USE_TK_GEMM_CONFIG")
    assert tk_gemm.get_tk_gemm_config(shape, use_production_default=True) == 9


def test_explicit_negative_override_disables_selector(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    shape = (32768, 14336, 4096)
    exact_env = tk_gemm._TK_V5_PDL_PRODUCTION_GEMM_CONFIG_ENVS[shape]
    monkeypatch.setenv("USE_TK_GEMM_CONFIG", "8")
    monkeypatch.setenv(exact_env, "-1")

    assert tk_gemm.get_tk_gemm_config(shape, use_production_default=True) is None
    monkeypatch.delenv(exact_env)
    monkeypatch.setenv("USE_TK_GEMM_CONFIG", "-1")
    assert tk_gemm.get_tk_gemm_config(shape, use_production_default=True) is None


def test_nopdl_uses_explicit_nopdl_config_but_not_pdl_table(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    shape = (32768, 14336, 4096)
    exact_env = tk_gemm._TK_V5_PDL_PRODUCTION_GEMM_CONFIG_ENVS[shape]
    monkeypatch.setenv("USE_TK_GEMM_NOPDL", "1")
    tk = _FakeTK()

    assert _dispatch(tk_gemm, tk, shape) == "nopdl"
    monkeypatch.setenv(exact_env, "7")
    assert _dispatch(tk_gemm, tk, shape) == "nopdl_config"
    assert tk.calls == [("nopdl", None), ("nopdl_config", 7)]
