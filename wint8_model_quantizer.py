"""
wint8_model_quantizer.py
────────────────────────
Standalone INT8 model quantizer node for ComfyUI.

Per-row INT8 quantization.  Auto-picks ctq (faster, learned rounding)
when available, falls back to built-in GPU-accelerated per-row quant.

Output format (compatible with convert_to_quant / ctq):
  - weight_scale → per-row (out_f, 1) float32
  - input_scale  → 1.0 float32
  - comfy_quant  → {"format": "int8_tensorwise", "per_row": true}
"""

import os
import json
import logging
import subprocess

import torch
import folder_paths
import comfy.utils

from .wint8_quarot import build_hadamard, rotate_weight

log = logging.getLogger("WINT8-Quantizer")

# ── Exclusion lists ──────────────────────────────────────────────────────────

_EXCLUSIONS = {
    "flux2": [
        "img_in", "time_in", "guidance_in", "txt_in", "final_layer",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation",
    ],
    "z-image": [
        "cap_embedder", "t_embedder", "x_embedder", "cap_pad_token",
        "context_refiner", "final_layer", "noise_refiner", "adaLN",
        "x_pad_token", "layers.0.",
        "cap_embedder.0", "attention_norm1", "attention_norm2",
        "ffn_norm1", "ffn_norm2", "k_norm", "q_norm",
    ],
    "chroma": [
        "distilled_guidance_layer", "final_layer", "img_in", "txt_in",
        "nerf_image_embedder", "nerf_blocks", "nerf_final_layer_conv",
        "__x0__",
    ],
    "wan": [
        "patch_embedding", "text_embedding", "time_embedding",
        "time_projection", "head", "img_emb", "motion_encoder",
    ],
    "ltx2": [
        "adaln_single", "audio_adaln_single", "audio_caption_projection",
        "audio_patchify_proj", "audio_proj_out", "audio_scale_shift_table",
        "av_ca_a2v_gate_adaln_single", "av_ca_audio_scale_shift_adaln_single",
        "av_ca_v2a_gate_adaln_single", "av_ca_video_scale_shift_adaln_single",
        "caption_projection", "patchify_proj", "proj_out", "scale_shift_table",
        "learnable_registers", "q_norm", "k_norm",
    ],
    "qwen": [
        "time_text_embed", "img_in", "norm_out", "proj_out", "txt_in",
        "norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm",
        "transformer_blocks.0.img_mod.1",
    ],
    "ernie": [
        "time", "x_embedder", "adaLN", "final", "text_proj",
        "norm", "layers.0.", "layers.35",
    ],
    "hidream": [
        "patch_embedding", "time_text_embed", "norm_out", "proj_out",
    ],
    "boogu": [
        "embed", "refine", "norm_out",
    ],
    "krea2": [
        "first", "last", "tmlp", "tproj", "txtfusion", "txtmlp",
    ],
    "ideogram4": [
        "embed_image_indicator", "t_embedding", "proj",
    ],
    "auto": [],
}

MODEL_TYPES = list(_EXCLUSIONS.keys())

QUANT_METHODS = ["auto", "ctq", "builtin"]

# ── ctq flag map ─────────────────────────────────────────────────────────────

_CTQ_FLAG_MAP = {
    "qwen":    "qwen",
    "z-image": "zimage",
    "flux2":   "flux2",
    "wan":     "wan",
    "ltx2":    "ltxv2",
    "chroma":  "distillation_large",
}

# ── Device helpers ────────────────────────────────────────────────────────────

def _get_available_devices() -> list[str]:
    choices = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        choices.append("xpu")
    if torch.cuda.is_available():
        choices.append("cuda")
    return choices


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_ctq() -> str | None:
    """Locate ctq in the .ext environment."""
    import sys
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    ctq_exe = os.path.join(scripts_dir, "ctq.exe")
    if os.path.isfile(ctq_exe):
        return ctq_exe
    try:
        subprocess.run(
            [sys.executable, "-m", "convert_to_quant", "-hf"],
            capture_output=True, text=True, timeout=10,
        )
        return f"{sys.executable} -m convert_to_quant"
    except Exception:
        return None


def _build_ctq_cmd(src_path: str, dst_path: str, model_type: str, ctq_bin: str) -> list[str]:
    cmd = [*ctq_bin.split(),
           "-i", src_path,
           "-o", dst_path,
           "--int8", "--scaling_mode", "row", "--simple",
           "--comfy_quant", "--save-quant-metadata", "--low-memory"]
    if model_type in _CTQ_FLAG_MAP:
        cmd.append(f"--{_CTQ_FLAG_MAP[model_type]}")
    else:
        patterns = _EXCLUSIONS.get(model_type, [])
        if patterns:
            regex = "(" + "|".join(patterns) + ")"
            cmd.extend(["--exclude-layers", regex])
    return cmd


# ── Exclusion helpers ─────────────────────────────────────────────────────────

def _is_excluded(key: str, model_type: str) -> bool:
    for pattern in _EXCLUSIONS.get(model_type, []):
        if pattern in key:
            return True
    return False


def _should_quantize(key: str, tensor: torch.Tensor, model_type: str) -> bool:
    if tensor.ndim != 2:
        return False
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32,
                             torch.float8_e4m3fn, torch.float8_e5m2):
        return False
    if _is_excluded(key, model_type):
        return False
    return True


class WINT8ModelQuantizer:

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("diffusion_models")
        if not files:
            files = ["none"]
        devices = _get_available_devices()
        device_default = "xpu" if "xpu" in devices else ("cuda" if "cuda" in devices else "cpu")
        return {
            "required": {
                "model_name": (files, {"tooltip": "Source model to quantize"}),
                "model_type": (MODEL_TYPES, {"default": "flux2"}),
                "quant_method": (QUANT_METHODS, {
                    "default": "auto",
                    "tooltip": "auto = ctq if available, fallback builtin | ctq = force ctq | builtin = built-in per-row",
                }),
                "enable_quarot": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Hadamard rotation (QuaRot/ConvRot) for better quality. Only used in builtin mode.",
                }),
                "group_size": ("INT", {
                    "default": 128, "min": 64, "max": 256, "step": 64,
                    "tooltip": "QuaRot group size. Recommended: 128.",
                }),
                "device": (devices, {
                    "default": device_default,
                    "tooltip": "Device used during quantization.",
                }),
                "output_filename": ("STRING", {
                    "default": "model_int8",
                    "tooltip": "Saved to ComfyUI/output/",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "quantize"
    OUTPUT_NODE = True
    CATEGORY = "WINT8"
    DESCRIPTION = "Quantize a diffusion model to per-row INT8."

    def quantize(
        self,
        model_name: str,
        model_type: str,
        quant_method: str,
        enable_quarot: bool,
        group_size: int,
        device: str,
        output_filename: str,
    ):
        src_path = folder_paths.get_full_path("diffusion_models", model_name)
        if src_path is None:
            raise FileNotFoundError(f"Model '{model_name}' not found.")

        output_dir = folder_paths.get_output_directory()
        dst_path = os.path.join(output_dir, f"{output_filename}.safetensors")

        # ── builtin — skip ctq entirely ──────────────────────────────
        if quant_method == "builtin":
            log.info("[WINT8 Quantizer] builtin mode selected, using built-in per-row quantization.")
            return self._quantize_builtin(
                model_name, model_type, enable_quarot, group_size, device, src_path, dst_path
            )

        # ── ctq / auto ───────────────────────────────────────────────
        ctq_bin = _find_ctq()
        if ctq_bin is None:
            if quant_method == "ctq":
                raise RuntimeError("[WINT8 Quantizer] ctq mode selected but ctq not found.")
            log.info("[WINT8 Quantizer] ctq not found, auto-fallback to built-in per-row quantization.")
            return self._quantize_builtin(
                model_name, model_type, enable_quarot, group_size, device, src_path, dst_path
            )

        cmd = _build_ctq_cmd(src_path, dst_path, model_type, ctq_bin)
        log.info(f"[WINT8 Quantizer] ctq ({quant_method}): {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            stderr = result.stderr[-800:] if len(result.stderr) > 800 else result.stderr
            if quant_method == "ctq":
                raise RuntimeError(f"ctq failed (code={result.returncode}):\n{stderr}")
            log.warning(
                f"[WINT8 Quantizer] ctq failed (code={result.returncode}), "
                f"auto-fallback to built-in per-row quantization.\n{stderr}"
            )
            return self._quantize_builtin(
                model_name, model_type, enable_quarot, group_size, device, src_path, dst_path
            )

        log.info(f"[WINT8 Quantizer] ctq done: {dst_path}")
        return ()

    def _quantize_builtin(self, model_name, model_type, enable_quarot, group_size, device, src_path, dst_path):
        dev = _resolve_device(device)
        log.info(f"[WINT8 Quantizer] Device: {dev}  (built-in per-row INT8)")

        sd = comfy.utils.load_torch_file(src_path, safe_load=True)
        log.info(f"[WINT8 Quantizer] Loaded {len(sd)} keys.")

        H = None
        quarot_applied = False

        # Boogu: override group_size to 32 for full QuaRot coverage
        if model_type == "boogu" and enable_quarot:
            group_size = 32
            log.info(f"[WINT8 Quantizer] Boogu detected: overriding group_size → 32 (full coverage)")

        if enable_quarot:
            H = build_hadamard(group_size, device=str(dev), dtype=torch.float32)
            log.info(f"[WINT8 Quantizer] QuaRot enabled, group_size={group_size}")

        quantized_count = 0
        excluded_count = 0
        total_before_bytes = 0
        total_after_bytes = 0

        for key in list(sd.keys()):
            tensor = sd[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            if not _should_quantize(key, tensor, model_type):
                if tensor.ndim == 2 and _is_excluded(key, model_type):
                    excluded_count += 1
                continue

            w = tensor.float().to(dev)

            layer_quarot = False
            if H is not None and w.shape[1] % group_size == 0:
                try:
                    w = rotate_weight(w, H, group_size=group_size)
                    layer_quarot = True
                    quarot_applied = True
                except ValueError:
                    pass

            # ── Per-row quantization ──────────────────────────────────
            amax = w.abs().amax(dim=1, keepdim=True)
            scale = (amax / 127.0).clamp(min=1e-8)
            q = (w / scale).round().clamp(-128, 127).to(torch.int8)

            base = key.rsplit(".weight", 1)[0]

            sd[key] = q.cpu()
            sd[f"{base}.weight_scale"] = scale.cpu()
            sd[f"{base}.comfy_quant"] = _make_comfy_quant(
                quarot=layer_quarot,
                group_size=group_size if layer_quarot else None,
            )
            sd[f"{base}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)

            total_before_bytes += tensor.numel() * tensor.element_size()
            total_after_bytes += q.numel() * 1 + scale.numel() * 4
            quantized_count += 1
            del w, q, scale

        if dev.type in ("xpu", "cuda"):
            try:
                (torch.xpu if dev.type == "xpu" else torch.cuda).empty_cache()
            except Exception:
                pass

        sd["int8_quantized"] = torch.tensor(1, dtype=torch.uint8)
        sd["int8_model_type"] = _str_to_uint8_tensor(model_type)

        log.info(f"[WINT8 Quantizer] Writing {dst_path} ...")
        comfy.utils.save_torch_file(sd, dst_path)

        mb_before = total_before_bytes / (1024 * 1024)
        mb_after = total_after_bytes / (1024 * 1024)
        log.info(
            f"\n{'='*60}\n"
            f"  WINT8 Quantization Complete (built-in · per-row)\n"
            f"  Model: {model_name} | Type: {model_type}\n"
            f"  Device: {dev} | QuaRot: {quarot_applied}\n"
            f"  Quantized: {quantized_count} | Excluded: {excluded_count}\n"
            f"  Weight: {mb_before:.1f} MB → {mb_after:.1f} MB "
            f"({100*mb_after/max(mb_before,1):.0f}%)\n"
            f"  Output: {dst_path}\n"
            f"  {'='*60}"
        )
        return ()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_comfy_quant(quarot: bool = False, group_size: int | None = None) -> torch.Tensor:
    payload = {"format": "int8_tensorwise", "per_row": True}
    if quarot and group_size:
        payload["convrot"] = True
        payload["convrot_groupsize"] = group_size
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


def _str_to_uint8_tensor(s: str) -> torch.Tensor:
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.uint8)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {"WINT8ModelQuantizer": WINT8ModelQuantizer}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT8ModelQuantizer": "WINT8 Model Quantizer"}
