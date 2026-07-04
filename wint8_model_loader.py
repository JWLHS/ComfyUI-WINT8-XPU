"""
wint8_model_loader.py
─────────────────────
WINT8 Model Loader node for ComfyUI.

Loads an INT8-quantized diffusion model using Int8XPUOps
for per-row INT8 inference on Intel XPU (Arc A770).
Supports torch.compile acceleration modes for XPU/CUDA/ROCm.
"""

import logging
import folder_paths
import comfy.sd

log = logging.getLogger("WINT8-Loader")

NODE_NAME = "WINT8 Model Loader"

# ── Acceleration modes ───────────────────────────────────────────────────────

_ACCEL_MODES = ["python", "compile", "compile_freeze"]

_ACCEL_TOOLTIP = (
    "python: original dequant + F.linear path (works everywhere, no compilation).\n"
    "compile: torch.compile — ~20-40% faster, LoRA / bake-in fully functional.\n"
    "compile_freeze: max speed (~30-50%) but weights locked at compile time.\n"
    "If compile fails, falls back to python mode automatically."
)


def _apply_acceleration(model, mode: str):
    import torch

    if mode == "python":
        return model

    freezing = (mode == "compile_freeze")
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            import torch._inductor.config as inductor_config

            inductor_config.cpp_wrapper = False
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
            model = model.float()

        compiled = torch.compile(
            model,
            options={"freezing": True} if freezing else {}
        )
        log.info(f"[WINT8 Loader] torch.compile enabled (mode={mode})")
        return compiled
    except Exception as e:
        log.warning(
            f"[WINT8 Loader] torch.compile failed ({e}), "
            f"falling back to python mode"
        )
        return model


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
                "acceleration_mode": (
                    _ACCEL_MODES,
                    {
                        "default": "python",
                        "tooltip": _ACCEL_TOOLTIP,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"

    def load_model(self, unet_name: str, model_type: str, acceleration_mode: str = "python"):
        import torch
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

        # ── Acceleration ─────────────────────────────────────
        if acceleration_mode != "python":
            model.model.diffusion_model = _apply_acceleration(
                model.model.diffusion_model, acceleration_mode
            )

        # ── Mark model for LoRA reset on next load ───────────────
        object.__setattr__(model.model, '_lora_needs_reset', True)

        # ── Patch detach to clear _lora_entries before offload ──
        _orig_detach = model.detach
        def _detach_with_cleanup(unpatch_all=True):
            dm = model.model.diffusion_model
            while hasattr(dm, '_orig_mod'):
                dm = dm._orig_mod
            for module in dm.modules():
                if hasattr(module, '_lora_entries'):
                    object.__setattr__(module, '_lora_entries', {})
                bake_state = getattr(module, '_wint8_bake_state', None)
                if bake_state is not None and '_orig_weight' in bake_state:
                    module.weight.data.copy_(bake_state['_orig_weight'])
                object.__setattr__(module, '_wint8_bake_state', None)
            return _orig_detach(unpatch_all)
        object.__setattr__(model, 'detach', _detach_with_cleanup)

        log.info(
            f"[WINT8 Loader] Loaded '{unet_name}' | type={model_type} "
            f"| mode={acceleration_mode} | INT8 VRAM savings active"
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"WINT8ModelLoader": WINT8ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT8ModelLoader": "WINT8 Model Loader"}
