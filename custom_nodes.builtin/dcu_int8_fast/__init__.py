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


def dcu_int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                    bias: torch.Tensor | None = None, out_dtype: torch.dtype = torch.bfloat16,
                    convrot: bool = False, convrot_groupsize: int = 256,
                    input_act: str | None = None,
                    input_act_weight: torch.Tensor | None = None,
                    input_act_eps: float = 0.0,
                    residual: torch.Tensor | None = None,
                    residual_scale: torch.Tensor | None = None,
                    **kwargs) -> torch.Tensor:
    """gfx936 INT8 linear：lmslim per-token 量化 + hipBLASLt channelwise W8A8 GEMM。

    与 eager 版本语义一致：激活动态逐行量化，权重为 per-output-channel scale；
    convrot=True 时先对激活做 Hadamard 旋转。GEMM 使用海光 hipBLASLt INT8 kernel。
    支持 comfy_kitchen 0.2.41+ 激活融合参数与输出残差累加。
    """
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8
    from lmslim.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm
    try:
        from comfy_kitchen.backends._activations import apply_input_act, apply_residual
        x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
    except (ImportError, TypeError):
        from comfy_kitchen.backends._activations import apply_input_act
        apply_residual = None
        x = apply_input_act(x, input_act)

    from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    orig_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    if convrot:
        # Krea2/ConvRot 路径：Hadamard 旋转 + lmslim Triton 量化，和 GEMM 一起
        # 构成 gfx936 上最快的 INT8 线性实现。
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x2d = _rotate_activation(x2d, h, convrot_groupsize).contiguous()
        x8, x_scale = per_token_quant_int8(x2d)
    else:
        # 非 ConvRot 路径保持与 ComfyUI eager 完全一致的激化量化，只替换 GEMM。
        x8, x_scale = quantize_int8_rowwise(x2d)
    m, k = x2d.shape
    n = weight.shape[0]

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, n):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)}"
        )
    if weight_scale.numel() == 1:
        weight_scale = weight_scale.reshape(1).expand(n).reshape(n, 1).contiguous()
    else:
        weight_scale = weight_scale.reshape(n, 1)

    ok, y = hipblaslt_w8a8_channelwise_gemm(
        x8, weight, x_scale.contiguous(), weight_scale,
        m, n, k, "NT", out_dtype, None,
    )
    if not ok:
        raise RuntimeError("hipBLASLt INT8 GEMM failed")

    if bias is not None:
        y = y + bias.to(device=y.device, dtype=y.dtype).reshape(1, -1)
    
    result = y.reshape(*orig_shape[:-1], n)
    if residual is not None and apply_residual is not None:
        result = apply_residual(result, residual, residual_scale)
    return result


def _patch_comfy_kitchen() -> None:
    if not _is_gfx936():
        logger.info("dcu_int8_fast: current DCU is not gfx936, skip")
        return

    _install_vllm_stub()
    import lmslim  # noqa: F401
    import comfy_kitchen as ck

    backend_name = "hygon"
    if ck.registry.is_available(backend_name):
        logger.info("dcu_int8_fast: '%s' backend already registered", backend_name)
        return

    module = types.SimpleNamespace(int8_linear=dcu_int8_linear)
    constraints = ck.registry.get_constraints("eager", "int8_linear")
    ck.registry.register(backend_name, module, {"int8_linear": constraints})
    ck.registry.set_priority([backend_name, "eager"])
    logger.info("dcu_int8_fast: registered '%s' backend (lmslim/hipBLASLt W8A8) for gfx936 INT8",
                backend_name)


_patch_comfy_kitchen()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
