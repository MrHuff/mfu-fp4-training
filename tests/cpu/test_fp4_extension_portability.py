import sysconfig

import pytest

from low_bits_training.cce import backend as cce_backend
from low_bits_training.quantization import mxfp4_backend, tk_gemm


@pytest.mark.parametrize(
    "path",
    [
        "/runtime/_C_nv_gemm.cpython-312-aarch64-linux-gnu.so",
        "/runtime/_C_nv_gemm.cpython-312-x86_64-linux-gnu.so",
    ],
)
def test_extension_module_name_is_architecture_independent(path):
    assert cce_backend._module_name_from_path(path) == "_C_nv_gemm"
    assert mxfp4_backend._module_name_from_path(path) == "_C_nv_gemm"


def test_tk_loader_considers_the_native_python_extension_suffix():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert f"_C_nv_gemm{suffix}" in tk_gemm._extension_candidate_names(
        "_C_nv_gemm"
    )


def test_tk_loader_finds_native_localcta_extension(tmp_path):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extension = tmp_path / f"_C_nv_localcta_gemm_v3{suffix}"
    extension.touch()

    assert tk_gemm._find_extension_in_dirs(
        "_C_nv_localcta_gemm_v3", [str(tmp_path)]
    ) == str(extension)


def test_localcta_variant_spec_is_not_tied_to_arm(monkeypatch):
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    directory, module_name = tk_gemm._localcta_gemm_variant_spec()
    assert directory == "localCTA_epilogue_v3"
    assert module_name == "_C_nv_localcta_gemm_v3"
    assert "aarch64" not in module_name
