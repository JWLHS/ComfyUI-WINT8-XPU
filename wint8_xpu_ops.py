"""
wint8_xpu_ops.py
────────────────
INT8 custom operations for Intel XPU (Arc A770).

Pure per-row INT8 inference:
  - weight_scale: (out_features, 1) or (out_features,)
  - dequant: w.float() * scale
  - F.linear

LoRA: via _lora_entries dict {lora_name: [(A,B,multiplier[,start,end]), ...]}.
A = down projection (rank, in_f), B = up projection (out_f, rank).
Stored on XPU.  Forward computes delta = B @ A on-the-fly.

LoKr: entry = ("lokr", w1, w2, multiplier, factor[, start, end]).
Dynamic Kronecker expansion in forward pass.

Triton kernel reserved for future Intel backend int8 dot support.
"""
import json
import locale
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Float32 hard-enforce — Arc A770 has no fp64; inductor emits float64 by default.
# ═══════════════════════════════════════════════════════════════════════════════

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

# --- torch factory overrides ---
_orig_torch_zeros = torch.zeros
_orig_torch_ones = torch.ones
_orig_torch_empty = torch.empty

def _wint8_zeros(*a, **kw):
    kw.setdefault('dtype', torch.float32)
    return _orig_torch_zeros(*a, **kw)

def _wint8_ones(*a, **kw):
    kw.setdefault('dtype', torch.float32)
    return _orig_torch_ones(*a, **kw)

def _wint8_empty(*a, **kw):
    kw.setdefault('dtype', torch.float32)
    return _orig_torch_empty(*a, **kw)

torch.zeros = _wint8_zeros
torch.ones = _wint8_ones
torch.empty = _wint8_empty

# triton.language.zeros / tl.full monkey-patches intentionally omitted —
# they break Triton 3.7.x AST visitor.

# --- inductor async_compile.triton: float64 → float32 in generated kernel source ---
try:
    from torch._inductor.async_compile import triton as _inductor_triton
    _orig_ind_triton = _inductor_triton
    import torch._inductor.async_compile as _ac
    def _patched_triton(src, *a, **kw):
        if isinstance(src, str):
            src = src.replace('tl.float64', 'tl.float32')
        return _orig_ind_triton(src, *a, **kw)
    _ac.triton = _patched_triton
except Exception:
    pass


log = logging.getLogger("WINT8-XPU")

# ═══════════════════════════════════════════════════════════════════════════════
# Auto-configure oneAPI compiler and build hook for torch.compile / Triton
# ═══════════════════════════════════════════════════════════════════════════════


def _configure_oneapi_for_triton():
    """Find oneAPI DPC++ compiler via PATH, patch COMPILATION_HELPER,
    and install a build hook that forces all Triton kernel compilation
    through icpx.exe.
    """
    icpx = shutil.which("icpx.exe")
    if not icpx:
        log.warning("[WINT8] icpx.exe not found in PATH — Triton compile disabled")
        return None

    bin_dir = os.path.dirname(icpx)
    lib_dir = os.path.join(os.path.dirname(bin_dir), "lib")
    inc_dir = os.path.join(os.path.dirname(bin_dir), "include")

    if not os.path.isfile(icpx):
        return None

    if hasattr(os, "add_dll_directory"):
        for d in (bin_dir, lib_dir):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

    _patch_compilation_helper(bin_dir, lib_dir, inc_dir)
    _install_build_hook(icpx, lib_dir, inc_dir)

    log.info(f"[WINT8] oneAPI detected at {bin_dir}")
    return icpx


def _patch_compilation_helper(bin_dir, lib_dir, inc_dir):
    """Force Triton's COMPILATION_HELPER to use oneAPI paths."""
    try:
        import triton
        from triton.backends.intel.driver import COMPILATION_HELPER

        level_zero_dir = r"C:\Program Files\LevelZeroSDK\1.28.2"
        triton_root = os.path.dirname(os.path.dirname(triton.__file__))
        triton_inc = os.path.join(triton_root, "backends", "intel", "include")
        triton_lib = os.path.join(triton_root, "backends", "intel", "lib")

        COMPILATION_HELPER.include_dir = [
            triton_inc,
            os.path.join(inc_dir, "sycl"),
            inc_dir,
            os.path.join(level_zero_dir, "include"),
        ]
        COMPILATION_HELPER.library_dir = [
            triton_lib,
            lib_dir,
            os.path.join(level_zero_dir, "lib"),
        ]
        try:
            COMPILATION_HELPER.sycl_dir = [bin_dir]
        except AttributeError:
            pass

        log.info("[WINT8] Triton COMPILATION_HELPER patched")
    except Exception:
        pass


def _install_build_hook(icpx_path, lib_dir, inc_dir):
    """Replace triton.runtime.build._build with our own implementation.

    Hard-codes oneAPI paths, bypassing shutil.which() in child processes.
    Also patches triton.backends.intel.driver._build (import capture).
    """
    _ze_lib = r"C:\Program Files\LevelZeroSDK\1.28.2\lib"
    _ze_inc = r"C:\Program Files\LevelZeroSDK\1.28.2\include"

    try:
        import triton.runtime.build as build_mod
    except ImportError:
        log.warning("[WINT8] triton.runtime.build not available — build hook skipped")
        return

    if getattr(build_mod._build, '__wint8_hook__', False):
        log.info("[WINT8] build hook already installed, skipping")
        return

    def _wint8_build(name, src, srcdir, library_dirs, include_dirs,
                     libraries, ccflags=None):
        if ccflags is None:
            ccflags = []

        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        so = os.path.join(srcdir, f'{name}{suffix}')

        scheme = sysconfig.get_default_scheme()
        if scheme == 'posix_local':
            scheme = 'posix_prefix'
        py_inc = sysconfig.get_paths(scheme=scheme)["include"]
        inc_dirs = [srcdir, py_inc]
        inc_dirs += list(include_dirs)
        inc_dirs += [os.path.join(inc_dir, "sycl"), inc_dir, _ze_inc]

        import triton as _triton_mod
        _triton_inc = os.path.join(
            os.path.dirname(_triton_mod.__file__),
            "backends", "intel", "include")
        if os.path.isdir(_triton_inc):
            inc_dirs.append(_triton_inc)

        py_lib = os.path.abspath(
            os.path.join(sysconfig.get_paths(scheme=scheme)["stdlib"], "..", "libs")
        )
        lib_dirs = [lib_dir, _ze_lib, py_lib]
        lib_dirs += list(library_dirs)

        cmd = [icpx_path, src, "-O3", "-shared"]
        cmd += [f"-I{d}" for d in inc_dirs]
        cmd += [f"-L{d}" for d in lib_dirs]
        cmd += [f"-l{lib}" for lib in libraries]
        cmd += [
            "-fsycl", "-fno-sycl-id-queries-fit-in-int",
            "-Wno-deprecated-declarations",
            "-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING",
        ]
        cmd += ccflags
        for bad in ("-Wno-psabi", "/Zc:__cplusplus", "/std:c++17",
                    "/nologo", "/O2", "/LD", "/wd4996", "/MD", "/EHsc",
                    "/Fo", "/link", "/OUT:", "/IMPLIB:", "/PDB:",
                    "/LIBPATH:", ".lib"):
            cmd = [a for a in cmd if bad not in a]
        cmd += ["-o", so]

        _saved_include = os.environ.pop("INCLUDE", None)
        _saved_lib = os.environ.pop("LIB", None)
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            output = e.stdout.decode(
                locale.getpreferredencoding()
                if sys.platform == "win32" else ()
            )
            raise RuntimeError(output)
        finally:
            if _saved_include is not None:
                os.environ["INCLUDE"] = _saved_include
            if _saved_lib is not None:
                os.environ["LIB"] = _saved_lib
        return so

    _wint8_build.__wint8_hook__ = True
    build_mod._build = _wint8_build
    import triton.backends.intel.driver as driver_mod
    driver_mod._build = _wint8_build

    log.info("[WINT8] build hook installed — all Triton compilations will use icpx.exe")


_TRITON_COMPILER = _configure_oneapi_for_triton()


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

                # Conv2d layers use "kernel" instead of "weight"
                if weight_tensor is None:
                    kernel_key = prefix + "kernel"
                    weight_tensor = state_dict.pop(kernel_key, None)
                    if weight_tensor is not None:
                        weight_key = kernel_key
                        scale_key = prefix + "kernel_scale"
                        meta_key = prefix + "kernel_comfy_quant"
                        input_scale_key = prefix + "kernel_input_scale"

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

                # ── LoRA injection ────────────────────────────────
                lora_entries = getattr(self, '_lora_entries', None)
                if lora_entries is not None:
                    for entries_list in lora_entries.values():
                        for entry in entries_list:
                            if isinstance(entry[0], str) and entry[0] == "lokr":
                                _, w1, w2, multiplier, factor = entry[:5]
                                sl_start = entry[5] if len(entry) > 5 else None
                                sl_end   = entry[6] if len(entry) > 6 else None

                                out_f_w2, in_f_w2 = w2.shape
                                w1_d = w1.to(device=w_dq.device, dtype=comp_dtype)
                                w2_d = w2.to(device=w_dq.device, dtype=comp_dtype)
                                w1_exp = w1_d.repeat_interleave(out_f_w2 // factor, dim=0).repeat_interleave(in_f_w2 // factor, dim=1)
                                delta = (w1_exp * w2_d).mul_(multiplier)

                                if delta.shape[0] != w_dq.shape[0] or delta.shape[1] != w_dq.shape[1]:
                                    continue
                                if sl_start is not None:
                                    w_dq[sl_start:sl_end, :].add_(delta)
                                else:
                                    w_dq.add_(delta)
                                continue

                            A, B, multiplier = entry[:3]
                            sl_start = entry[3] if len(entry) > 3 else None
                            sl_end   = entry[4] if len(entry) > 4 else None

                            if A.shape[1] != w_dq.shape[1]:
                                continue

                            A_d = A.to(dtype=comp_dtype) if A.dtype != comp_dtype else A
                            B_d = B.to(dtype=comp_dtype) if B.dtype != comp_dtype else B
                            if A_d.device != w_dq.device:
                                A_d = A_d.to(device=w_dq.device)
                            if B_d.device != w_dq.device:
                                B_d = B_d.to(device=w_dq.device)

                            delta = (B_d @ A_d).mul_(multiplier)
                            if sl_start is not None:
                                w_dq[sl_start:sl_end, :].add_(delta)
                            else:
                                w_dq.add_(delta)

                b_dq = bias.to(device=x.device, dtype=comp_dtype) if bias is not None else None

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
