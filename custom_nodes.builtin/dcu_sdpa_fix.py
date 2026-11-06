import glob
import logging
import os
import sysconfig

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


_FA_SO = _locate_fa_so()
if _FA_SO:
    os.environ.setdefault("FA_SO_PATH", _FA_SO)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
    logging.getLogger(__name__).info(
        "dcu_sdpa_fix: FA_SO_PATH=%s (flash attention enabled)",
        os.environ["FA_SO_PATH"],
    )
else:
    logging.getLogger(__name__).warning("dcu_sdpa_fix: flash_attn_2_cuda*.so not found")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
