import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[2]
    / "low_bits_training"
    / "quantization"
    / "fused_te_linear.py"
)


def _method_args(class_name: str, method_name: str) -> set[str]:
    module = ast.parse(SOURCE.read_text())
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return {arg.arg for arg in method.args.args}


def _class_methods(class_name: str) -> set[str]:
    module = ast.parse(SOURCE.read_text())
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _method_source(class_name: str, method_name: str) -> str:
    module = ast.parse(SOURCE.read_text())
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.unparse(method)


def _function_source(function_name: str) -> str:
    module = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return ast.unparse(function)


def test_te_attention_accepts_mixed_backend_carrier_interface():
    assert {
        "x",
        "freqs_cis",
        "h_carrier",
        "cde_row_rms_partial",
    } <= _method_args("FusedAttentionFP4_TE", "forward_qkv")
    assert {
        "attn_output",
        "residual",
        "h_gamma",
        "cde_emit",
    } <= _method_args("FusedAttentionFP4_TE", "forward_wo")


def test_nvfp4_role_marker_supports_new_te_tensor_copy_contract():
    assert "copy" in _class_methods("_NVFP4RoleMarker")


def test_te_wo_accepts_direct_nhsd_attention_layout():
    source = _method_source("FusedAttentionFP4_TE", "forward_wo")
    assert "attn_output.dim() == 4" in source
    assert "attn_output.transpose(1, 2).contiguous()" in source


def test_fused_w2_silu_payload_waits_for_previous_consumers():
    state_source = _function_source("_get_ffn_localcta_bwd_state")
    backward_source = _function_source("_ffn_bwd_graphed")

    event_name = "w2_dgrad_silu_payload_ready_event"
    assert f"'{event_name}': torch.cuda.Event()" in state_source

    wait = backward_source.index(
        f"torch.cuda.current_stream().wait_event(localcta_state['{event_name}'])"
    )
    producer = backward_source.index("tk.nvfp4_w2_dgrad_silu_quant_gemm(")
    wgrad_done = backward_source.index(
        "torch.cuda.current_stream().wait_stream(wgrad_stream)", producer
    )
    record = backward_source.index(
        f"torch.cuda.current_stream().record_event(localcta_state['{event_name}'])"
    )

    assert wait < producer < wgrad_done < record


def test_localcta_w2_parameter_grad_uses_layer_owned_storage():
    helper_source = _function_source("_get_ffn_localcta_owned_grad_buffer")
    backward_source = _function_source("_ffn_bwd_graphed")

    assert "USE_TK_LOCALCTA_TRANSIENT_W2_GRAD" in helper_source
    assert "_ffn_localcta_owned_grad_cache" in helper_source
    assert "localcta_grad_w2 = _get_ffn_localcta_owned_grad_buffer" in backward_source
    assert "localcta_state['grad_w2']" not in backward_source
