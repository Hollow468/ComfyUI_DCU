import logging

# DCU 平台：海光 DAS 预编译 xformers（0.0.33+das.opt1.dtk2604）是纯 Python 桩，
# 没有编译好的 _C_hip.so，因此 xformers._has_cpp_library 为 False，
# 会被 ComfyUI（comfy/model_management.py:407）判定为不可用。
# 实测该桩可通过 triton 后端正常执行 memory_efficient_attention（DCU 上已验证），
# 故在 prestartup（早于 comfy.model_management import）阶段把它翻转为 True，
# 让 ComfyUI 走 xformers 注意力路径。
# 若后续有带 _C_hip.so 的完整版 xformers，本脚本可安全删除。
try:
    import xformers

    xformers._has_cpp_library = True
    logging.getLogger(__name__).info(
        "dcu_xformers: xformers._has_cpp_library forced True (triton backend), version %s",
        getattr(xformers, "__version__", "?"),
    )
except Exception as e:
    logging.getLogger(__name__).warning("dcu_xformers: failed to enable xformers: %s", e)
