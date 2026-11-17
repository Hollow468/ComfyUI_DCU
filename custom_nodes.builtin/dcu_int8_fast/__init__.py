import logging
import os
import sys
import types

import torch

logger = logging.getLogger(__name__)

# lmslim/lightop 已安装到本 venv（wheels/das18）。lmslim.quantize.quant_ops 会
# `from vllm.utils import direct_register_custom_op`，这里注入一个最小 vllm 包来
# 满足该引用，避免在 ComfyUI 进程中真正导入 vllm 带来的额外初始化。
_VLLM_STUB_INSTALLED = False


def _is_gfx936() -> bool:
    try:
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            return False
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return getattr(props, "gcnArchName", "").startswith("gfx936")
    except Exception:
        return False


def _install_vllm_stub() -> None:
    global _VLLM_STUB_INSTALLED
    if _VLLM_STUB_INSTALLED or "vllm.utils" in sys.modules:
        return

    os.environ.setdefault("ROCM_HOME", "/opt/dtk")

    vllm_pkg = types.ModuleType("vllm")
    vllm_utils = types.ModuleType("vllm.utils")

    def direct_register_custom_op(op_name, op_func, mutates_args=None, fake_impl=None,
                                  target_lib=None, dispatch_key=None, tags=()):
        import torch.library
        if mutates_args is None:
            mutates_args = []
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
        my_lib = target_lib
        my_lib.define(op_name + schema_str, tags=tags)
        my_lib.impl(op_name, op_func, dispatch_key=dispatch_key or "CUDA")
        if fake_impl is not None:
            my_lib._register_fake(op_name, fake_impl)

    vllm_utils.direct_register_custom_op = direct_register_custom_op
    sys.modules.setdefault("vllm", vllm_pkg)
    sys.modules.setdefault("vllm.utils", vllm_utils)
    vllm_pkg.utils = vllm_utils
    _VLLM_STUB_INSTALLED = True


def _dcu_int8_linear_chunk(x2d: torch.Tensor, weight: torch.Tensor, weight_scale_2d: torch.Tensor,
                           bias: torch.Tensor | None, out_dtype: torch.dtype,
                           convrot: bool, convrot_groupsize: int,
                           input_act: str | None, input_act_weight: torch.Tensor | None,
                           input_act_eps: float, residual2d: torch.Tensor | None,
                           residual_scale: torch.Tensor | None,
                           apply_input_act, apply_residual,
                           quantize_and_rotate_rowwise, per_token_quant_int8,
                           hipblaslt_w8a8_channelwise_gemm,
                           quantize_int8_rowwise, _build_hadamard, _rotate_activation) -> torch.Tensor:
    try:
        x2d = apply_input_act(x2d, input_act, input_act_weight, input_act_eps)
    except TypeError:
        x2d = apply_input_act(x2d, input_act)

    x2d = x2d.contiguous()
    if convrot:
        h = _build_hadamard(convrot_groupsize, device=x2d.device, dtype=x2d.dtype)
        if quantize_and_rotate_rowwise is not None:
            x8, x_scale = quantize_and_rotate_rowwise(x2d, h, convrot_groupsize)
            del x2d
        else:
            rot = _rotate_activation(x2d, h, convrot_groupsize).contiguous()
            del x2d
            x8, x_scale = per_token_quant_int8(rot)
            del rot
    else:
        x8, x_scale = quantize_int8_rowwise(x2d)
        del x2d

    m, k = x8.shape
    n = weight.shape[0]
    ok, y = hipblaslt_w8a8_channelwise_gemm(
        x8, weight, x_scale.contiguous(), weight_scale_2d,
        m, n, k, "NT", out_dtype, None,
    )
    del x8, x_scale
    if not ok:
        raise RuntimeError("hipBLASLt INT8 GEMM failed")

    if bias is not None:
        y = y + bias.to(device=y.device, dtype=y.dtype).reshape(1, -1)
    if residual2d is not None and apply_residual is not None:
        y = apply_residual(y, residual2d, residual_scale)
    return y


_OPS = None


def _get_ops():
    global _OPS
    if _OPS is None:
        _install_vllm_stub()
        from lmslim.layers.gemm.int8_utils import per_token_quant_int8
        from lmslim.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm
        try:
            from comfy_kitchen.backends.triton import quantize_and_rotate_rowwise
        except ImportError:
            quantize_and_rotate_rowwise = None
        try:
            from comfy_kitchen.backends._activations import apply_input_act, apply_residual
            has_new_act = True
        except ImportError:
            from comfy_kitchen.backends._activations import apply_input_act
            apply_residual = None
            has_new_act = False

        from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

        _OPS = (
            per_token_quant_int8,
            hipblaslt_w8a8_channelwise_gemm,
            quantize_and_rotate_rowwise,
            apply_input_act,
            apply_residual,
            has_new_act,
            quantize_int8_rowwise,
            _build_hadamard,
            _rotate_activation,
        )
    return _OPS


def dcu_int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                    bias: torch.Tensor | None = None, out_dtype: torch.dtype = torch.bfloat16,
                    convrot: bool = False, convrot_groupsize: int = 256,
                    input_act: str | None = None,
                    input_act_weight: torch.Tensor | None = None,
                    input_act_eps: float = 0.0,
                    residual: torch.Tensor | None = None,
                    residual_scale: torch.Tensor | None = None,
                    **kwargs) -> torch.Tensor:
    """gfx936 INT8 linear：comfy_kitchen triton 旋转量化 + hipBLASLt channelwise W8A8 GEMM。

    与 eager 版本语义一致：激活动态逐行量化，权重为 per-output-channel scale；
    convrot=True 时使用 comfy_kitchen triton quantize_and_rotate_rowwise 融合 Hadamard 旋转与量化。
    支持 comfy_kitchen 0.2.41+ 激活融合参数与输出残差累加。
    """
    (
        per_token_quant_int8,
        hipblaslt_w8a8_channelwise_gemm,
        quantize_and_rotate_rowwise,
        apply_input_act,
        apply_residual,
        has_new_act,
        quantize_int8_rowwise,
        _build_hadamard,
        _rotate_activation,
    ) = _get_ops()

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    m = x2d.shape[0]
    n = weight.shape[0]

    if weight.device != x.device or not weight.is_contiguous():
        weight = weight.to(device=x.device).contiguous()
    if (
        weight_scale.device == x.device
        and weight_scale.dtype == torch.float32
        and weight_scale.shape == (n, 1)
        and weight_scale.is_contiguous()
    ):
        weight_scale_2d = weight_scale
    else:
        weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
        if weight_scale.numel() not in (1, n):
            raise ValueError(
                f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)}"
            )
        if weight_scale.numel() == 1:
            weight_scale_2d = weight_scale.reshape(1).expand(n).reshape(n, 1).contiguous()
        else:
            weight_scale_2d = weight_scale.reshape(n, 1).contiguous()

    residual2d = residual.reshape(m, n) if (residual is not None and apply_residual is not None) else None
    act_weight = input_act_weight if has_new_act else None
    act_eps = input_act_eps if has_new_act else 0.0

    # 大激活（如 DiT MLP swiglu+convrot）按行分块，避免中间激活叠加导致 OOM
    chunk_rows = 32768
    if m > chunk_rows and x2d.numel() * x2d.element_size() > 512 * 1024 * 1024:
        out = torch.empty((m, n), dtype=out_dtype, device=x.device)
        for start in range(0, m, chunk_rows):
            end = min(start + chunk_rows, m)
            res_chunk = residual2d[start:end] if residual2d is not None else None
            out[start:end] = _dcu_int8_linear_chunk(
                x2d[start:end], weight, weight_scale_2d, bias, out_dtype,
                convrot, convrot_groupsize, input_act, act_weight, act_eps,
                res_chunk, residual_scale,
                apply_input_act, apply_residual,
                quantize_and_rotate_rowwise, per_token_quant_int8,
                hipblaslt_w8a8_channelwise_gemm,
                quantize_int8_rowwise, _build_hadamard, _rotate_activation,
            )
        return out.reshape(*orig_shape[:-1], n)

    result = _dcu_int8_linear_chunk(
        x2d, weight, weight_scale_2d, bias, out_dtype,
        convrot, convrot_groupsize, input_act, act_weight, act_eps,
        residual2d, residual_scale,
        apply_input_act, apply_residual,
        quantize_and_rotate_rowwise, per_token_quant_int8,
        hipblaslt_w8a8_channelwise_gemm,
        quantize_int8_rowwise, _build_hadamard, _rotate_activation,
    )
    return result.reshape(*orig_shape[:-1], n)


def _patch_comfy_kitchen() -> None:
    if not _is_gfx936():
        logger.info("dcu_int8_fast: current DCU is not gfx936, skip")
        return

    _get_ops()
    import comfy_kitchen as ck

    backend_name = "hygon"
    if ck.registry.is_available(backend_name):
        logger.info("dcu_int8_fast: '%s' backend already registered", backend_name)
        return

    module = types.SimpleNamespace(int8_linear=dcu_int8_linear)
    constraints = ck.registry.get_constraints("eager", "int8_linear")
    ck.registry.register(backend_name, module, {"int8_linear": constraints})
    # 启用并保留 triton 及其他已注册后端（如 hygon_triton），避免覆盖优先级列表
    ck.registry.enable("triton")
    cur_priority = list(getattr(ck.registry, "_priority", ["ascend", "cuda", "triton", "eager"]))
    new_priority = []
    for name in (backend_name, *cur_priority, "triton", "eager"):
        if name not in new_priority:
            new_priority.append(name)
    ck.registry.set_priority(new_priority)
    logger.info("dcu_int8_fast: registered '%s' backend (triton rot+quant / hipBLASLt W8A8) for gfx936 INT8",
                backend_name)


_patch_comfy_kitchen()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
