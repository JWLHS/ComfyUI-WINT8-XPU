"""
wint8_model_loader.py
─────────────────────
WINT8 Model Loader node for ComfyUI.

Loads an INT8-quantized diffusion model using Int8XPUOps
for per-row INT8 inference on Intel XPU (Arc A770).
"""

import logging
import folder_paths
import comfy.sd

log = logging.getLogger("WINT8-Loader")

NODE_NAME = "WINT8 Model Loader"


class WINT8ModelLoader:

    NAME = NODE_NAME
    CATEGORY = "WINT8"

    @classmethod
    def INPUT_TYPES(cls):
        from .wint8_model_quantizer import MODEL_TYPES
        return {
            "required": {
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "INT8 model produced by WINT8ModelQuantizer"},
                ),
                "model_type": (
                    MODEL_TYPES,
                    {
                        "default": "flux2",
                        "tooltip": "Must match the type used during quantization",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"

    def load_model(self, unet_name: str, model_type: str):
        from .wint8_xpu_ops import Int8XPUOps
        from .wint8_model_quantizer import _EXCLUSIONS

        Int8XPUOps.excluded_names = _EXCLUSIONS.get(model_type, [])
        Int8XPUOps._is_prequantized = False

        model_options = {"custom_operations": Int8XPUOps}

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(
                f"[WINT8 Loader] Model '{unet_name}' not found in diffusion_models."
            )

        log.info(
            f"[WINT8 Loader] Loading: {unet_name} (type={model_type})"
        )
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        log.info(
            f"[WINT8 Loader] Loaded '{unet_name}' | type={model_type} "
            f"| INT8 VRAM savings active"
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"WINT8ModelLoader": WINT8ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT8ModelLoader": "WINT8 Model Loader"}
