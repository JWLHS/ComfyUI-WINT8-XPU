"""
WINT8 Model Quantizer + Loader
───────────────────────────────
Standalone INT8 per-row model quantization & loading for ComfyUI.

Two nodes:
  WINT8ModelQuantizer  → offline per-row INT8 quantization
  WINT8ModelLoader     → load INT8 models with VRAM savings

Pure PyTorch — no CUDA, no external dependencies beyond ComfyUI itself.
Works on CPU / CUDA / XPU / any PyTorch backend.
"""

from .wint8_model_quantizer import (
    NODE_CLASS_MAPPINGS as _QUANT_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _QUANT_DISPLAY,
)

from .wint8_model_loader import (
    NODE_CLASS_MAPPINGS as _LOAD_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _LOAD_DISPLAY,
)

NODE_CLASS_MAPPINGS = {
    **_QUANT_MAPPINGS,
    **_LOAD_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_QUANT_DISPLAY,
    **_LOAD_DISPLAY,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
