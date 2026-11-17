import logging
import re
import types

import torch

_logger = logging.getLogger(__name__)
_GEP_NUW_RE = re.compile(r"(getelementptr\b(?:\s+inbounds)?)\s+(?:(?:nuw|nusw)\s+)+")


def _is_gfx936() -> bool:
    try:
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            return False
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return getattr(props, "gcnArchName", "").startswith("gfx936")
    except Exception:
        return False


def _patch_triton_llvm_gep_nuw() -> None:
    """Strip LLVM 19+ `nuw`/`nusw` modifiers on `getelementptr` before DTK LLVM-17 `libgcvm` parses LLIR."""
    try:
        from triton.backends.amd.compiler import HIPBackend
    except ImportError:
        return
    if getattr(HIPBackend.make_amdgcn, "_dcu_gep_nuw_patched", False):
        return

    orig_make_amdgcn = HIPBackend.make_amdgcn

    def _patched_make_amdgcn(src, metadata, options):
        if "getelementptr" in src and ("nuw" in src or "nusw" in src):
            src = _GEP_NUW_RE.sub(r"\1 ", src)
        return orig_make_amdgcn(src, metadata, options)

    _patched_make_amdgcn._dcu_gep_nuw_patched = True
    HIPBackend.make_amdgcn = staticmethod(_patched_make_amdgcn)


def _patch_vsa_triton() -> None:
    if not _is_gfx936():
        _logger.info("dcu_vsa_triton: current DCU is not gfx936, skip")
        return

    try:
        import triton  # noqa: F401
    except ImportError as exc:
        _logger.warning("dcu_vsa_triton: triton unavailable (%s), skip", exc)
        return

    _patch_triton_llvm_gep_nuw()

    import comfy_kitchen as ck
    from comfy_kitchen.constraints import (
        ExactDims,
        FunctionConstraints,
        ParamConstraint,
        sol_attn_common_call_rule,
    )

    from .triton_kernels import (
        triton_rms_rope_split_half,
        triton_sol_attn,
        triton_sol_attn_chunked,
    )

    def rms_rope_split_half(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor | None = None,
        epsilon: float = 1e-6,
        rot_dim: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return triton_rms_rope_split_half(
            q, k, freqs_cis, q_scale, k_scale=k_scale, epsilon=epsilon, rot_dim=rot_dim, inplace=False
        )

    def rms_rope_split_half_(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor | None = None,
        epsilon: float = 1e-6,
        rot_dim: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return triton_rms_rope_split_half(
            q, k, freqs_cis, q_scale, k_scale=k_scale, epsilon=epsilon, rot_dim=rot_dim, inplace=True
        )

    backend_name = "hygon_triton"
    if not ck.registry.is_available(backend_name):
        cuda_devices = frozenset({"cuda"})
        half_floats = frozenset({torch.bfloat16, torch.float16})
        rope_constraints = ck.registry.get_constraints("eager", "rms_rope_split_half")
        if rope_constraints is not None:
            rope_constraints = FunctionConstraints(
                params=rope_constraints.params,
                default_devices=cuda_devices,
                call_rules=rope_constraints.call_rules,
            )

        capabilities = {
            "sol_attn": FunctionConstraints(
                params={
                    "q": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(4),)),
                    "k": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(4),)),
                    "v": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(4),)),
                },
                default_devices=cuda_devices,
                call_rules=(sol_attn_common_call_rule,),
            ),
        }
        module_attrs = {
            "sol_attn": triton_sol_attn,
        }
        if rope_constraints is not None:
            capabilities["rms_rope_split_half"] = rope_constraints
            capabilities["rms_rope_split_half_"] = rope_constraints
            module_attrs["rms_rope_split_half"] = rms_rope_split_half
            module_attrs["rms_rope_split_half_"] = rms_rope_split_half_

        ck.registry.register(backend_name, types.SimpleNamespace(**module_attrs), capabilities)

        ck.registry.enable("triton")
        cur_priority = list(getattr(ck.registry, "_priority", ["ascend", "cuda", "triton", "eager"]))
        new_priority = []
        for name in ("hygon", backend_name, "triton", *cur_priority, "eager"):
            if name not in new_priority:
                new_priority.append(name)
        ck.registry.set_priority(new_priority)

    orig_is_available = ck.sol_attn_is_available

    def _dcu_sol_attn_is_available(device: torch.device | None = None) -> bool:
        if device is not None and device.type != "cuda":
            return False
        if _is_gfx936():
            return True
        return orig_is_available(device)

    ck.sol_attn_is_available = _dcu_sol_attn_is_available
    ck.sol_attn_chunked = triton_sol_attn_chunked

    _logger.info(
        "dcu_vsa_triton: registered '%s' (Sol-Attn/VSA + partial rms_rope_split_half) for gfx936",
        backend_name,
    )


_patch_vsa_triton()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
