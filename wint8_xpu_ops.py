"""
wint8_xpu_ops.py
────────────────
INT8 custom operations for Intel XPU (Arc A770).

Pure per-row INT8 inference:
  - weight_scale: (out_features, 1) or (out_features,)
  - dequant: w.float() * scale  (one broadcast multiply)
  - simple broadcast + F.linear

Triton kernel reserved for future Intel backend int8 dot support.
"""

import os
import json
import logging

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

log = logging.getLogger("WINT8-XPU")

# ═══════════════════════════════════════════════════════════════════════════════
# Environment setup
# ═══════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE = False


def _try_add_dll_search_paths():
    candidates = []
    try:
        import folder_paths
        base = folder_paths.base_path
        if base:
            candidates.append(os.path.join(base, ".ext", "Library", "bin"))
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        rel = os.path.normpath(os.path.join(here, "..", "..", "..", ".ext", "Library", "bin"))
        candidates.append(rel)
    except Exception:
        pass
    for p in candidates:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass


def _find_oneapi_2025():
    base = r"C:\Program Files (x86)\Intel\oneAPI\compiler"
    if not os.path.isdir(base):
        return None
    versions = []
    for e in os.listdir(base):
        full = os.path.join(base, e)
        if os.path.isdir(full) and e.startswith("2025."):
            if os.path.isfile(os.path.join(full, "bin", "icpx.exe")):
                versions.append(e)
    versions.sort(reverse=True)
    return os.path.join(base, versions[0]) if versions else None


def _patch_compilation_helper():
    global _TRITON_AVAILABLE
    try:
        from triton.backends.intel.driver import COMPILATION_HELPER
    except ImportError:
        return
    oneapi_dir = _find_oneapi_2025()
    if oneapi_dir is None:
        return
    level_zero_dir = r"C:\Program Files\LevelZeroSDK\1.28.2"
    triton_dir = os.path.dirname(triton.__file__)
    triton_inc = os.path.join(triton_dir, "backends", "intel", "include")
    triton_lib = os.path.join(triton_dir, "backends", "intel", "lib")
    COMPILATION_HELPER.include_dir = [
        triton_inc,
        os.path.join(oneapi_dir, "include"),
        os.path.join(oneapi_dir, "include", "sycl"),
        os.path.join(level_zero_dir, "include"),
    ]
    COMPILATION_HELPER.library_dir = [
        triton_lib,
        os.path.join(oneapi_dir, "lib"),
        os.path.join(level_zero_dir, "lib"),
    ]
    icpx_bin = os.path.join(oneapi_dir, "bin")
    if os.path.isdir(icpx_bin) and icpx_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = icpx_bin + os.pathsep + os.environ.get("PATH", "")
    log.info(f"WINT8-XPU: Triton compilation locked to {oneapi_dir}")


_try_add_dll_search_paths()
_patch_compilation_helper()

try:
    _TRITON_AVAILABLE = True
    log.info("WINT8-XPU: Triton XPU available")
except ImportError:
    log.info("WINT8-XPU: Triton not available")


# ═══════════════════════════════════════════════════════════════════════════════
# Triton kernel (reserved — Intel backend WIP for int8 dot)
# ═══════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:

    @triton.jit
    def _w8a8_blockwise_gemm_kernel(
        a_ptr, w_ptr, c_ptr,
        a_scale_ptr, w_scale_ptr, bias_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_k[None, :] < (K - k0)), other=0)
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < (K - k0)), other=0)
            acc += tl.dot(a, w)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk
        c = acc.to(tl.float32)
        k_blocks = K // BLOCK_SIZE
        act_scale_offs = offs_m * k_blocks + (k_blocks - 1)
        act_sc = tl.load(a_scale_ptr + act_scale_offs, mask=offs_m < M, other=1.0)
        scale_n_idx = offs_n // BLOCK_SIZE
        w_scale_ptrs = w_scale_ptr + (offs_m // BLOCK_SIZE)[:, None] * (N // BLOCK_SIZE) + scale_n_idx[None, :]
        w_sc = tl.load(w_scale_ptrs, mask=(offs_m < M) & (offs_n < N), other=1.0)
        c = c * act_sc[:, None] * w_sc
        if HAS_BIAS:
            c = c + tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI custom operations
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS = True
except ImportError:
    _COMFY_OPS = False

if _COMFY_OPS:

    class Int8XPUOps(manual_cast):
        excluded_names: list = []
        _is_prequantized: bool = False

        class Linear(manual_cast.Linear):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer("weight_scale", None)
                self._is_quantized = False
                self._use_quarot = False
                self._group_size = 128
                self._hadamard_H = None
                self.compute_dtype = torch.float16

            def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            ):
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                bias_key = prefix + "bias"
                meta_key = prefix + "comfy_quant"
                input_scale_key = prefix + "input_scale"

                weight_tensor = state_dict.pop(weight_key, None)
                weight_scale = state_dict.pop(scale_key, None)
                meta_raw = state_dict.pop(meta_key, None)
                state_dict.pop(input_scale_key, None)

                if weight_tensor is None:
                    missing_keys.append(weight_key)
                    self._is_quantized = False
                elif weight_tensor.dtype == torch.int8 and weight_scale is not None:
                    Int8XPUOps._is_prequantized = True
                    self._is_quantized = True
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                    self.register_buffer("weight_scale", weight_scale.float())

                    if meta_raw is not None:
                        try:
                            meta = json.loads(bytes(meta_raw.tolist()).decode("utf-8"))
                            is_rotated = meta.get("quarot", False) or meta.get("convrot", False)
                            if is_rotated:
                                self._use_quarot = True
                                gs = meta.get("group_size", meta.get("convrot_groupsize", 128))
                                self._group_size = gs
                                from .wint8_quarot import build_hadamard
                                self._hadamard_H = build_hadamard(
                                    gs, device="cpu", dtype=torch.float32
                                )
                        except Exception:
                            pass

                elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    self._is_quantized = False
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    self._is_quantized = False
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)

                bias_tensor = state_dict.pop(bias_key, None)
                self.bias = (
                    torch.nn.Parameter(bias_tensor, requires_grad=False)
                    if bias_tensor is not None else None
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                need_cast = (
                    self.comfy_cast_weights
                    or len(getattr(self, 'weight_function', [])) > 0
                    or len(getattr(self, 'bias_function', [])) > 0
                )

                if not self._is_quantized:
                    if need_cast:
                        weight, bias, offload_stream = cast_bias_weight(
                            self, x, offloadable=True,
                        )
                        out = F.linear(x, weight, bias)
                        uncast_bias_weight(self, weight, bias, offload_stream)
                        return out
                    return F.linear(x, self.weight, self.bias)

                # ── AIMDO cast in ──────────────────────────────────
                weight, bias, offload_stream = cast_bias_weight(
                    self, x, offloadable=True,
                )

                # ── Align weight_scale ─────────────────────────────
                w_scale = self.weight_scale
                if w_scale is not None and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)

                # ── Per-row dequant ────────────────────────────────
                x2 = x.reshape(-1, x.shape[-1])
                comp_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16

                if self._use_quarot and self._hadamard_H is not None:
                    try:
                        from .wint8_quarot import rotate_activation
                        x2 = rotate_activation(x2, self._hadamard_H, self._group_size)
                    except Exception:
                        pass

                if w_scale.ndim >= 1 and w_scale.shape[0] > 1:
                    w_dq = (weight.float() * w_scale.view(-1, 1)).to(comp_dtype)
                else:
                    w_dq = (weight.float() * w_scale).to(comp_dtype)

                b_dq = bias.to(device=x.device, dtype=comp_dtype) if bias is not None else None

                # ── LoRA (optional) ────────────────────────────────
                if need_cast:
                    for fn in getattr(self, 'weight_function', []):
                        w_dq = fn(w_dq)
                    for fn in getattr(self, 'bias_function', []):
                        if b_dq is not None:
                            b_dq = fn(b_dq)

                # ── Compute ────────────────────────────────────────
                out = F.linear(x2.to(comp_dtype), w_dq, b_dq)
                del w_dq

                # ── AIMDO cast out ─────────────────────────────────
                uncast_bias_weight(self, weight, bias, offload_stream)

                return out.reshape(*x.shape[:-1], out.shape[-1])

        class GroupNorm(manual_cast.GroupNorm):
            pass
        class LayerNorm(manual_cast.LayerNorm):
            pass
        class Conv2d(manual_cast.Conv2d):
            pass
        class Conv3d(manual_cast.Conv3d):
            pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            pass
        class Embedding(manual_cast.Embedding):
            pass

        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            raise ValueError(f"Int8XPUOps: unsupported conv dims: {dims}")

else:
    class Int8XPUOps:
        pass
