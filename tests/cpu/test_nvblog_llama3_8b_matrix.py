from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_nvblog_llama3_8b_matrix as matrix  # noqa: E402


def test_localcta_highwater_uses_shape_compatible_cde_policy():
    paper_cases = matrix.build_cases("8B_nvfp4_paper_proxy")
    blog_cases = matrix.build_cases("8B_llama3_blog")

    for name in (
        "nvfp4_localcta_v4_highwater",
        "nvfp4_localcta_v4_highwater_delayed",
    ):
        assert paper_cases[name].env["USE_FP4_CODA_EXACT_CDE"] == "1"
        assert paper_cases[name].env["USE_FP4_CODA_EXACT_CDE_WO"] == "1"
        assert blog_cases[name].env["USE_FP4_CODA_EXACT_CDE"] == "1"
        assert blog_cases[name].env["USE_FP4_CODA_EXACT_CDE_WO"] == "1"
        assert (
            paper_cases[name].env[
                "USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD"
            ]
            == "0"
        )
        assert (
            paper_cases[name].env["USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD"]
            == "0"
        )
        assert (
            blog_cases[name].env[
                "USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD"
            ]
            == "0"
        )
        assert (
            blog_cases[name].env["USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD"]
            == "0"
        )
        assert (
            paper_cases[name].env["USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO"]
            == "0"
        )
        assert (
            blog_cases[name].env["USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO"]
            == "0"
        )
