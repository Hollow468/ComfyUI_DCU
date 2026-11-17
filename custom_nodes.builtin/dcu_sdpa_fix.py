import glob
import logging
import os
import sysconfig
import weakref

import torch

# DCU（Hygon gfx936）flash-attention 修复（方案 C，真·根因修复）
#
# 背景：DCU 定制版 torch（2.9.0+das.opt1.dtk2604）的 HIP flash SDP 后端由 CUTLASSFA
# 适配器实现（torch/include/ATen/native/transformers/hip/cutlassfa_adapter.h），运行时会
# dlopen 一个名为 flash_attn_2_cuda*.so 的动态库（读环境变量 FA_SO_PATH，否则扫描
# site-packages 根目录），但该库未随 torch 分发 → 任何无 mask 的
# F.scaled_dot_product_attention(q,k,v) 一旦命中 flash 后端就抛：
#   RuntimeError: No matching libraries found for flash_attn_2_cuda*.so
#
# 现状：已从海光源（sourcefind.cn 下载站，AI生态包 /flash_attn/DAS1.8）获取预编译 wheel
#   flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
# 安装后 .so 位于 venv site-packages 根目录。start.sh 已 export FA_SO_PATH；
# 本节点作为兜底：若 FA_SO_PATH 未设置则按当前 venv 的 site-packages 自动定位，
# 并确保 flash SDP 后端开启。定位逻辑与部署位置无关（不再硬编码路径）。

_logger = logging.getLogger(__name__)
_CAUSAL_MASK_CACHE = None  # (weakref.ref(mask), is_causal: bool)


def _locate_fa_so():
    """按优先级定位 flash_attn_2_cuda*.so：环境变量 -> 当前 venv site-packages。"""
    env = os.environ.get("FA_SO_PATH")
    if env and os.path.isfile(env):
        return env
    for key in ("purelib", "platlib"):
        site = sysconfig.get_paths().get(key)
        if not site or not os.path.isdir(site):
            continue
        hits = glob.glob(os.path.join(site, "flash_attn_2_cuda*.so"))
        if hits:
            return hits[0]
    return None


def _is_causal_mask(mask: torch.Tensor, seq_q: int, seq_k: int) -> bool:
    """判断 4D mask 是否为标准上三角因果掩码（用于将显式 mask 转为 is_causal=True 走 FlashAttention）。"""
    global _CAUSAL_MASK_CACHE
    if mask.ndim != 4 or mask.shape != (1, 1, seq_q, seq_k) or seq_q != seq_k or seq_q <= 1:
        return False

    if _CAUSAL_MASK_CACHE is not None:
        cached_ref, cached_val = _CAUSAL_MASK_CACHE
        if cached_ref() is mask:
            return cached_val

    m2d = mask[0, 0]
    # O(1) 快速预筛：对角线与左下角应为 0，右上角应为极大负数
    if m2d[0, 0].item() != 0.0 or m2d[0, -1].item() >= -1e4 or m2d[-1, 0].item() != 0.0:
        _CAUSAL_MASK_CACHE = (weakref.ref(mask), False)
        return False

    triu = torch.triu(torch.ones((seq_q, seq_k), dtype=torch.bool, device=mask.device), diagonal=1)
    is_causal = bool(torch.where(triu, m2d < -1e4, m2d == 0).all().item())
    _CAUSAL_MASK_CACHE = (weakref.ref(mask), is_causal)
    return is_causal


def _patch_comfy_sdpa() -> None:
    """优化 ComfyUI SDPA：纯因果掩码转 is_causal=True + fp16 走 FlashAttention，回退 MATH 时按 Q 分块防 OOM。"""
    import comfy.ops

    if getattr(comfy.ops.scaled_dot_product_attention, "_dcu_sdpa_patched", False):
        return

    orig_sdpa = comfy.ops.scaled_dot_product_attention
    chunk_threshold_bytes = 256 * 1024 * 1024
    chunk_size = 512

    def dcu_scaled_dot_product_attention(q, k, v, *args, **kwargs):
        attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
        dropout_p = args[1] if len(args) > 1 else kwargs.get("dropout_p", 0.0)
        is_causal = args[2] if len(args) > 2 else kwargs.get("is_causal", False)
        enable_gqa = kwargs.get("enable_gqa", False)

        # 1. 若传入的是标准上三角因果 mask，转为 is_causal=True 并去掉 attn_mask 以启用 FlashAttention
        if (
            attn_mask is not None
            and not is_causal
            and q.is_cuda
            and _is_causal_mask(attn_mask, q.shape[-2], k.shape[-2])
        ):
            attn_mask = None
            is_causal = True
            if len(args) > 0:
                args = (None, *args[1:])
            else:
                kwargs["attn_mask"] = None
            if len(args) > 2:
                args = (*args[:2], True, *args[3:])
            else:
                kwargs["is_causal"] = True

        # 2. FlashAttention 不支持 float32：无 mask 时若设备支持 fp16 FlashAttention，则临时转 fp16 计算
        if (
            q.is_cuda
            and q.dtype == torch.float32
            and attn_mask is None
            and q.nelement() >= 1024 * 128
            and torch.backends.cuda.flash_sdp_enabled()
            and q.shape[-1] <= 256
        ):
            q16 = q.to(torch.float16)
            k16 = k.to(torch.float16)
            v16 = v.to(torch.float16)
            params = torch.backends.cuda.SDPAParams(q16, k16, v16, None, dropout_p, is_causal, enable_gqa)
            if torch.backends.cuda.can_use_flash_attention(params):
                return orig_sdpa(q16, k16, v16, *args, **kwargs).to(torch.float32)

        # 3. 仍需回退到 MATH 后端的长序列场景，沿 seq_q 分块避免 O(N^2) 显存爆炸
        if q.is_cuda and q.ndim == 4:
            est_bytes = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * q.element_size()
            if est_bytes > chunk_threshold_bytes and q.shape[2] > chunk_size:
                params = torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa)
                can_fast = (
                    torch.backends.cuda.can_use_flash_attention(params)
                    or torch.backends.cuda.can_use_efficient_attention(params)
                )
                if not can_fast:
                    seq_q = q.shape[2]
                    out = torch.empty(
                        (q.shape[0], q.shape[1], seq_q, v.shape[3]),
                        dtype=q.dtype,
                        device=q.device,
                    )
                    for start in range(0, seq_q, chunk_size):
                        end = min(start + chunk_size, seq_q)
                        q_chunk = q[:, :, start:end, :]
                        k_chunk, v_chunk = k, v
                        chunk_kwargs = dict(kwargs)
                        chunk_args = args
                        if is_causal:
                            k_chunk = k[:, :, :end, :]
                            v_chunk = v[:, :, :end, :]
                            causal_chunk = torch.full(
                                (1, 1, end - start, end),
                                float("-inf"),
                                dtype=q.dtype,
                                device=q.device,
                            ).triu_(diagonal=start + 1)
                            if len(chunk_args) > 0:
                                chunk_args = (causal_chunk, *chunk_args[1:])
                            else:
                                chunk_kwargs["attn_mask"] = causal_chunk
                            if len(chunk_args) > 2:
                                chunk_args = (*chunk_args[:2], False, *chunk_args[3:])
                            else:
                                chunk_kwargs["is_causal"] = False
                        elif attn_mask is not None and attn_mask.ndim >= 2 and attn_mask.shape[-2] == seq_q:
                            m_chunk = attn_mask[..., start:end, :]
                            if len(chunk_args) > 0:
                                chunk_args = (m_chunk, *chunk_args[1:])
                            else:
                                chunk_kwargs["attn_mask"] = m_chunk
                        out[:, :, start:end, :] = orig_sdpa(q_chunk, k_chunk, v_chunk, *chunk_args, **chunk_kwargs)
                    return out

        return orig_sdpa(q, k, v, *args, **kwargs)

    dcu_scaled_dot_product_attention._dcu_sdpa_patched = True
    comfy.ops.scaled_dot_product_attention = dcu_scaled_dot_product_attention
    _logger.info("dcu_sdpa_fix: patched comfy.ops.scaled_dot_product_attention (causal+fp16 flash & math chunking)")


_FA_SO = _locate_fa_so()
if _FA_SO:
    os.environ.setdefault("FA_SO_PATH", _FA_SO)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
    _logger.info(
        "dcu_sdpa_fix: FA_SO_PATH=%s (flash attention enabled)",
        os.environ["FA_SO_PATH"],
    )
    _patch_comfy_sdpa()
else:
    _logger.warning("dcu_sdpa_fix: flash_attn_2_cuda*.so not found")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
