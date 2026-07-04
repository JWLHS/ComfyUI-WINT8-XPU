"""
WINT8 Model Quantizer + Loader + LoRA
──────────────────────────────────────
Standalone INT8 per-row model quantization, loading & LoRA for ComfyUI.

Nodes:
  WINT8ModelQuantizer  → UNet BF16/FP16/FP8 → INT8
  WINT8ModelLoader     → load INT8 UNet
  WINT8LoRALoader      → single LoRA
  WINT8LoRAStack       → multi-LoRA stack (up to 5)

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

from .wint8_lora_loader import (
    NODE_CLASS_MAPPINGS as _LORA_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _LORA_DISPLAY,
)

from .wint8_lora_stack import (
    NODE_CLASS_MAPPINGS as _STACK_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _STACK_DISPLAY,
)

NODE_CLASS_MAPPINGS = {
    **_QUANT_MAPPINGS,
    **_LOAD_MAPPINGS,
    **_LORA_MAPPINGS,
    **_STACK_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_QUANT_DISPLAY,
    **_LOAD_DISPLAY,
    **_LORA_DISPLAY,
    **_STACK_DISPLAY,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
