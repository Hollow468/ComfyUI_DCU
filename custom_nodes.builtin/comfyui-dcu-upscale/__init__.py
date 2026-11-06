import copy

import torch
import folder_paths
import comfy.utils
import comfy.model_management
import comfy.model_patcher
from spandrel import ModelLoader, ImageModelDescriptor
from spandrel.__helpers.size_req import pad_tensor

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class UpscaleModelLoaderDCU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
            }
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "Load an upscale model (R-ESRGAN / Real-ESRGAN / AnimeSharp etc.) via spandrel, same behavior as the official UpscaleModelLoader."

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")

        out.patcher = comfy.model_patcher.CoreModelPatcher(
            out.model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        return (out,)


class UpscaleImageWithModelDCU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = (
        "Upscale an image with a loaded upscale model. "
        "Precision selectable (fp16 recommended on DCU, ~2x faster than fp32; fp32 for best quality). "
        "Uses adaptive tiling with automatic OOM fallback."
    )

    _warmed_keys = set()

    def _warmup(self, upscale_model, model, precision, device, tile_size):
        key = (precision, tile_size)
        if key in UpscaleImageWithModelDCU._warmed_keys:
            return
        # 惰性预热：首次使用该(精度,tile)组合时，按 tile_size 尺寸跑一次小图前向，
        # 触发 MIOpen 对实际 tile shape 的自动调优，避免首次正式推理被冷启动调优拖慢。
        # 只在本进程内生效，重启后重新预热；失败不影响正式推理。
        try:
            warm = torch.rand(1, 3, tile_size, tile_size, device=device)
            if precision == "fp16":
                warm = warm.half()
            with torch.inference_mode():
                self._forward(upscale_model, model, warm)
            torch.cuda.synchronize()
        except Exception:
            pass
        UpscaleImageWithModelDCU._warmed_keys.add(key)

    def _forward(self, upscale_model, model, a):
        did_pad, padded = pad_tensor(a, upscale_model.size_requirements)
        out = model(padded)
        out = out.clamp_(0, 1)
        if did_pad:
            out = out[..., : a.shape[2] * upscale_model.scale, : a.shape[3] * upscale_model.scale]
        return out

    def upscale(self, upscale_model, image, precision="fp16", tile_size=1024, overlap=32):
        device = upscale_model.patcher.load_device

        memory_required = (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        comfy.model_management.load_models_gpu([upscale_model.patcher], memory_required=memory_required)

        # fp16: deep-copy the (fp32) weights and cast once, keep the original untouched.
        if precision == "fp16":
            model = copy.deepcopy(upscale_model.patcher.model).half().to(device)
        else:
            model = upscale_model.patcher.model

        tile = max(int(tile_size), 128)
        overlap = max(int(overlap), 0)

        self._warmup(upscale_model, model, precision, device, tile)

        in_img = image.movedim(-1, -3).to(device)
        if precision == "fp16":
            in_img = in_img.half()

        output_device = comfy.model_management.intermediate_device()

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)
                if precision == "fp16":
                    s = comfy.utils.tiled_scale(
                        in_img, lambda a: self._forward(upscale_model, model, a).float(),
                        tile_x=tile, tile_y=tile, overlap=overlap,
                        upscale_amount=upscale_model.scale, pbar=pbar,
                        output_device=output_device,
                    )
                else:
                    s = comfy.utils.tiled_scale(
                        in_img, lambda a: self._forward(upscale_model, model, a),
                        tile_x=tile, tile_y=tile, overlap=overlap,
                        upscale_amount=upscale_model.scale, pbar=pbar,
                        output_device=output_device,
                    )
                oom = False
            except Exception as e:
                comfy.model_management.raise_non_oom(e)
                tile //= 2
                if tile < 128:
                    raise e

        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0).to(comfy.model_management.intermediate_dtype())
        return (s,)


NODE_CLASS_MAPPINGS["UpscaleModelLoaderDCU"] = UpscaleModelLoaderDCU
NODE_CLASS_MAPPINGS["UpscaleImageWithModelDCU"] = UpscaleImageWithModelDCU
NODE_DISPLAY_NAME_MAPPINGS["UpscaleModelLoaderDCU"] = "Load Upscale Model (DCU)"
NODE_DISPLAY_NAME_MAPPINGS["UpscaleImageWithModelDCU"] = "Upscale Image (using Model, DCU)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
