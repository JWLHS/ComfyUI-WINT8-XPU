

```markdown

抱歉，这是上传错误，但是最近我在研究新改动，无暇顾及，总是这是python实现int8方式的一种，原理和以下int4几乎一样。
# ComfyUI-WINT4-XPU-beta

> **INT4 per-row 模型量化 + LoRA 推理插件 — Intel Arc A770 16GB 优化，跨后端通用**

将扩散模型量化为 per-row INT4 (packed uint8)，显存节省 75%，支持 10 种 LoRA 格式叠加。
**Python 模式通用（XPU / CUDA / ROCm / CPU）。Triton 加速为 Intel XPU 专用。**

---

## 快速判断

| 你是 | 建议 |
|------|------|
| Intel Arc A770/A750 用户 | 可选 Triton compile 加速（首次等 3-5min，后续同参数秒出） |
| NVIDIA / AMD 用户 | 用 **python 模式**，零配置，开箱即用 |
| 不想折腾 Triton | 用 **python 模式**，跟旧版完全一样，量化器照样用 |

> **加速模式默认 python。** 除非你在 Intel Arc 上且已配好 PT 2.14 + oneAPI 2026.0，否则不要切换 compile/compile_freeze。
>
> **Triton 加速为 XPU 专用。** 其他显卡（NVIDIA / AMD / ROCm）若需 Triton 加速，请自行寻找适配自己硬件的方案。

---

## 目录

1. [功能概览](#功能概览)
2. [性能数据](#性能数据)
3. [安装](#安装)
4. [环境要求（Triton 加速）](#环境要求triton-加速仅-xpu)
5. [启动配置（Triton XPU 必须）](#启动配置triton-xpu-加速必须)
6. [加速模式详解](#加速模式详解)
7. [节点详解](#节点详解)
8. [快速开始](#快速开始)
9. [支持的 LoRA 格式](#支持的-lora-格式)
10. [支持的模型](#支持的模型)
11. [从 PT 2.12 迁移](#从-pt-212-迁移到-pt-214)
12. [已知限制](#已知限制)
13. [常见问题](#常见问题)
14. [v5 新增](#v5-新增)
15. [v6 新增](#v6-新增)
16. [Bug 修复记录](#bug-修复记录)
17. [v6 踩坑记录 — Triton compile 加速](#v6-踩坑记录--triton-compile-加速200-次尝试历时-3-天)
18. [文件清单](#文件清单)
19. [相关链接](#相关链接)
20. [鸣谢](#鸣谢)

---

## 功能概览

| 能力 | 状态 | 适用范围 |
|------|:--:|------|
| BF16/FP16/FP8/INT8 → INT4 量化 | ✅ | 通用 |
| QuaRot (Hadamard 旋转) 质量提升 | ✅ INT4 必须开启 | 通用 |
| INT4 UNet 推理 | ✅ | 通用 |
| 单 LoRA 加载 | ✅ | 通用 |
| 多 LoRA 叠加（串联 / Stack） | ✅ | 通用 |
| LyCORIS LoKr 格式 | ✅ | 通用 |
| ICLoRA / adaLN LoRA（bake-in） | ✅ | 通用 |
| QKV 融合模型 LoRA | ✅ | 通用 |
| 10 种 LoRA key 格式自动适配 | ✅ | 通用 |
| AIMDO DynamicVRAM 双路径 | ✅ | 通用 |
| Triton torch.compile 加速 | ✅ | **XPU 专用** |

---

## 性能数据

### 存储 & 显存（Krea2）

| 指标 | BF16 | INT8 | INT4 |
|------|:--:|:--:|:--:|
| UNet 存储 | 24 GB | 12 GB | **6 GB** |
| 推理显存 | ~24 GB | ~16-17 GB | **~8-9 GB** |
| A770 16GB 裸跑 | ❌ | ❌ | ✅ |

### Triton 加速（Arc A770, Boogu INT4, 8 steps）

| 模式 | 首次 | 同参数第二次 | 改分辨率后 |
|------|------|------|------|
| python | ~180s | ~180s | ~180s |
| compile | ~300s（含编译） | **~50s（3.6x）** | ~330s（重编译） |
| compile_freeze | ~300s | **~45s（4.0x）** | ~330s |

> **python 模式跨后端通用，所有显卡都能用。compile 仅在 Intel Arc + PT 2.14 + oneAPI 2026.0 上可用。**

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta.git
```

依赖同 INT8 插件，ComfyUI 启动时自动安装。

---

## 环境要求（Triton 加速，仅 XPU）

以下仅针对 **Intel Arc 显卡 + 开启 compile/compile_freeze 模式**。python 模式无需关注。

| 组件 | 版本 | 说明 |
|------|------|------|
| PyTorch | **2.14.0.dev+xpu** 以上 | nightly 版本，sycl9 运行时 |
| Triton | 3.7.2（随 PyTorch 安装） | |
| oneAPI | **2026.0** | ⚠️ 2025.3 存在 IGC 寄存器溢出 bug |
| LevelZero SDK | 1.28.2+ | |
| Python | 3.11+ | |

---

## 启动配置（Triton XPU 加速必须）

### 方式 A：绘世启动器（推荐）

在 `ComfyUI\.ext\Lib\site-packages\` 下新建两个文件：

**`wint4_env.pth`**
```
import wint4_env_setup
```

**`wint4_env_setup.py`**
```python
import os

_ONEAPI_BIN = r"C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\bin"
_ONEAPI_LIB = r"C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\lib"

_path = os.environ.get("PATH", "")
if _ONEAPI_BIN not in _path:
    os.environ["PATH"] = f"{_ONEAPI_BIN};{_ONEAPI_LIB};{_path}"

os.environ.setdefault("IGC_largeGRF", "1")
os.environ.setdefault("SYCL_ENABLE_DEFAULT_CONTEXTS", "1")
os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")
```

> 替换 `2026.0` 为你的实际 oneAPI 版本号。

### 方式 B：BAT 启动脚本（仅供参考）

```bat
@echo off
:: ── Triton 启用必要项（仅供参考，根据实际路径调整）：───────────
:: ① vcvarsall — 提供 MSVC 编译器
:: ② oneAPI 2026.0 PATH — icpx.exe 必须能在 PATH 中找到
:: ③ IGC_largeGRF=1 — 大寄存器文件
:: ④ SYCL_ENABLE_DEFAULT_CONTEXTS=1 — Level Zero 上下文
:: ⑤ PYTORCH_ENABLE_XPU_FALLBACK=1 — XPU 回退
:: ──────────────────────────────────────────────────────────────

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set "PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\bin;C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\lib;%PATH%"
set "IGC_largeGRF=1"
set "SYCL_ENABLE_DEFAULT_CONTEXTS=1"
set "PYTORCH_ENABLE_XPU_FALLBACK=1"

cd /d E:\ComfyUI
E:\ComfyUI\.ext\python.exe main.py --port 8189
```

---

## 加速模式详解

| 模式 | 速度 | 首次编译 | 改分辨率 | LoRA | 适用场景 | 适用范围 |
|------|:--:|:--:|:--:|:--:|------|------|
| **python** | 基准 | 0 | 无影响 | ✅ | 日常出图，终极兼容 | **通用** |
| **compile** | +20-40% | 3-5 min | 需重编译 | ✅ | 同参数批量、seedvr2 放大 | **XPU 专用** |
| **compile_freeze** | +30-50% | 3-5 min | 需重编译 | ❌ | 同参数批量，不换 LoRA | **XPU 专用** |

### 工作原理（compile 模式）

```
首次 forward:
    inductor trace 模型图（30-60s）
        │
        ▼
    IGC 编译 Triton kernel（200-250s）  ← 只用一次
        │
        ▼
    缓存持久化 → %USERPROFILE%\.triton\cache\

后续同参数 forward:
    缓存命中 → 跳过 IGC 编译 → 秒进
```

---

## 节点详解

### WINT4 Model Quantizer

**功能：** BF16/FP16/FP8/INT8 → INT4 packed uint8 + QuaRot

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `model_name` | — | 选择源模型 |
| `model_type` | `flux2` | 模型架构，控制排除列表 |
| `enable_quarot` | `False` | **INT4 必须开启**，Hadamard 旋转 |
| `group_size` | 128 | QuaRot 分组大小（Boogu 自动 32） |
| `device` | `xpu` | 量化计算设备 |
| `output_filename` | `model_int4` | 保存至 `output/`，支持子目录 |

### WINT4 Model Loader

**功能：** 加载 INT4 模型，可选加速模式

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `unet_name` | — | 选择量化后的模型 |
| `model_type` | `flux2` | **必须和量化时一致** |
| `acceleration_mode` | `python` | python / compile / compile_freeze |

### WINT4 LoRA Loader / LoRA Stack

单 LoRA 链式串联 / 多 LoRA 一次叠加（最多 5 个）。compile 模式下已适配 `_orig_mod` 穿透，LoRA 不受影响。

---

## 快速开始

```
1. 量化:
   WINT4ModelQuantizer
   ├─ model_name      = BF16/FP16 模型
   ├─ model_type      = krea2 / wan / ltx2 / flux2 / ...
   ├─ enable_quarot   = True  ← INT4 必须开启
   ├─ group_size      = 64（Boogu 自动 32）
   ├─ device          = xpu
   └─ output_filename = my_model_int4

2. 推理（无 LoRA）:
   WINT4ModelLoader → MODEL → KSampler → VAE Decode

3. 推理（单 LoRA）:
   WINT4ModelLoader → WINT4LoRALoader → KSampler

4. 推理（多 LoRA 串联）:
   WINT4ModelLoader → LoRA Loader (A) → LoRA Loader (B) → ...

5. 推理（多 LoRA Stack）:
   WINT4ModelLoader → WINT4LoRAStack（最多 5 槽）→ KSampler
```

LoRA 强度建议 **1.5-2.0x**（INT4 精度限制）。ICLoRA / adaLN LoRA 自动 bake-in。

---

## 支持的 LoRA 格式

| # | 格式 | 示例 |
|:-:|------|------|
| ① | Kohya 标准 | `diffusion_model.blocks.0.attn.wq.lora_B.weight` |
| ② | diffusers | `transformer.blocks.0.attn.to_q.lora_B.weight` |
| ③ | SimpleTuner lycoris | `lycoris_blocks_0_attn_wq.lora_down.weight` |
| ④ | bare | `blocks.0.attn.wq.lora_B.weight` |
| ⑤ | onetrainer | `transformer.text_fusion.layerwise_blocks.0...` |
| ⑥ | legacy ComfyUI | `lora_unet_blocks_0_attn_wq.lora_down.weight` |
| ⑦ | onetrainer alt | `lora_transformer_blocks_0_attn_wq.lora_down.weight` |
| ⑧ | BFL (Flux) | `single_blocks.0.attn.qkv.lora_A.weight` → 自动转换 |
| ⑨ | LyCORIS LoKr | `diffusion_model.blocks.0.attn.wq.lokr_w1` |
| ⑩ | LTX ICLoRA | `diffusion_model.adaln_single.*.lora_A.weight` → bake-in |

---

## 支持的模型

| model_type | 模型系列 | 验证 |
|:---|------|:--:|
| `flux2` | Flux2-Klein | ✅ |
| `qwen` | Qwen-Rapid / Qwen-EDIT / Qwen-AIO | ✅ |
| `z-image` | Z-Image | ✅ |
| `wan` | Wan 2.1/2.2 / SCAIL2 / WANanimate / Bernini / WANremix | ✅ |
| `ltx2` | LTX 2.3 | ✅ |
| `krea2` | Krea2 Turbo / Raw | ✅ |
| `boogu` | Boogu Base / Edit / Turbo（auto gs=32） | ✅ |
| `hidream` | HiDream | ✅ |
| `ernie` | ERNIE | ✅ |
| `ideogram4` | Ideogram 4 | ✅ |
| `chroma` | Chroma/Distillation | ✅ |
| `auto` | 通用（不排除任何层） | ⚠️ 谨慎使用 |

---

## 从 PT 2.12 迁移到 PT 2.14

```bat
:: 1. 卸载旧版 sycl8
E:\ComfyUI\.ext\python.exe -m pip uninstall -y torch torchaudio torchvision triton-xpu intel-sycl-rt intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-opencl-rt intel-openmp intel-pti mkl tbb tcmlib umf dpcpp-cpp-rt onemkl-sycl-blas onemkl-sycl-dft onemkl-sycl-lapack onemkl-sycl-rng onemkl-sycl-sparse onemkl-license

:: 2. 安装 PT 2.14 nightly
E:\ComfyUI\.ext\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

:: 3. 安装 oneAPI 2026.0（从 Intel 官网下载）
:: 4. 按上方「启动配置」操作
```

---

## 已知限制

1. 首次 compile 需等 3-5 分钟（IGC 编译）
2. 改分辨率触发重编译；提示词/步数/CFG/seed 不触发
3. compile_freeze 不支持 LoRA
4. Triton 加速仅测试 Intel Arc A770 16GB
5. PT 2.14 为 nightly dev，正式版可能有 API 变化

---

## 常见问题

**不想用 Triton？**
选 python 模式。量化器照样用，所有功能一应俱全。跟旧版一模一样。

**NVIDIA/AMD 能用吗？**
量化器和 python 推理模式通用。Triton compile 模式为 XPU 专用——本插件的 build hook 仅处理 Intel 路径。NVIDIA/AMD 显卡的 Triton 加速请自行寻找适配方案。

**重启后要重新编译吗？**
Triton kernel 缓存（`.triton\cache\`）跨 session 持久。Inductor 图需重 trace（几十秒），但比首次快得多。

**为什么改分辨率要重编译？**
Inductor 为每个 input shape 生成特化 kernel。提示词、步数、CFG 不改变 tensor shape，所以不触发。

**INT4 模型能用官方 Load Diffusion Model 加载吗？**
可以。ComfyUI 新版支持 INT4 格式，本插件量化器输出与原生协议对齐。遇到兼容性问题再切回 WINT4ModelLoader。

---

## v5 新增

| 功能 | 说明 |
|------|------|
| LTX2.3 完整支持 | 排除列表补全 + metadata 保留 |
| Wan 模型检测 fallback | `head.modulation` 缺失时从权重 shape 反推 config |
| FP8 清理修复 | 仅删除已被量化的 FP8 权重，其余转 FP16 |
| motion_encoder 保护 | Wan 排除列表加 `motion_encoder` |
| ICLoRA bake-in | 非量化层 LoRA 直接融合到 weight |
| Boogu group_size=32 | 自动覆盖，100% QuaRot 覆盖率 |
| 源 metadata 保留 | 透传原始 `config`，正确识别模型版本 |

## v6 新增

| 功能 | 说明 |
|------|------|
| Triton torch.compile 加速 | 三种模式，首次编译后同参数 4x 加速 |
| compile + LoRA 支持 | 穿透 OptimizedModule wrapper |
| INT8 Triton 同步移植 | build hook + float32 强制 + 电感静默 |
| oneAPI 2026.0 适配 | IGC 寄存器溢出修复 |
| 绘世启动器兼容 | `.pth` 注入方案 |
| 代码清理 | 删重复分支、死代码、DLL 诊断日志 |

---

## Bug 修复记录

| # | Bug | 修复 | 版本 |
|---|------|------|:--:|
| 28 | 多 LoRA 链式只有最后一个生效 | 删 `_lora_needs_reset=True`；prune 机制 | v4 |
| 29 | LoKr 格式匹配 0 层 | 解析 lokr_w1/w2；动态 Kronecker 展开 | v4 |
| 30 | LoKr 预计算 delta 撑爆显存 | 改动态展开 | v4 |
| 31 | 双采第一个 UNet 不卸载 | detach 清空 `_lora_entries` | v4 |
| 32 | LoKr delta shape 不匹配 | `_compute` shape guard | v4 |
| 33 | MPS 设备不支持 | 加 MPS 检测 | v4 |
| 34 | FP8 清理误删排除层 | 改为按 weight_scale 判断 + 转 FP16 | v5 |
| 35 | Wan 模型检测失败 | 5D patch_embedding fingerprint fallback | v5 |
| 36 | motion_encoder 被量化 | wan 排除列表 + kernel 回退 | v5 |
| 37 | LTX2.3 shape mismatch | 排除列表补全 + metadata 保留 | v5 |
| 38 | Boogu QuaRot 仅 27% 覆盖 | 自动 group_size=32 | v5.1 |
| 39 | ICLoRA adaLN 被跳过 | bake-in 到非量化层 | v5.1 |

---

## v6 踩坑记录 — Triton compile 加速（200+ 次尝试，历时 3 天）

| # | 问题 | 现象 | 根因 | 修复 |
|---|------|------|------|------|
| 40 | IGC 寄存器溢出 | `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE`，200 次尝试全部失败 | PT 2.12 + IGC 2025.3 无法处理 inductor 融合 kernel（9 合 1），Arc 128 寄存器饱和 | 升级 PT 2.14 + IGC 2026.0 |
| 41 | `tl.full` monkey-patch 崩溃 | `ValueError: Did you forget to add @triton.jit?` | Triton 3.7.2 AST visitor 要求原始函数签名，lambda 破坏了 `_semantic` | 删除 monkey-patch，仅靠字符串替换 |
| 42 | `dynamic=True` IGC 溢出 | 同 #40 | dynamic kernel 更复杂 | 弃用，后续手写 kernel |
| 43 | `max_fusion_size=1` 无效 | 改分辨率仍重编译 330s | inductor 缓存 key 含 input shape | 确认 inductor 设计如此 |
| 44 | compile + LoRA 失效 | `0 layers matched` | `OptimizedModule` wrapper 隐藏了 `_is_quantized` | 加 `while hasattr(dm, '_orig_mod')` 穿透 |
| 45 | 系统 Python 被 sycl8 污染 | venv 启动 `access violation` | 系统 Python site-packages 有 sycl8 全家桶 | 卸载 Intel 2025.3 |
| 46 | oneAPI `latest` 指向 2025.3 | `icpx.exe` 路径错误 | 符号链接未更新 | `mklink /D latest 2026.0` |
| 47 | 绘世启动器不传 PATH | `icpx.exe` 找不到 | 绘世 python 不从系统继承 PATH | build hook 硬编码 + `.pth` 注入 |
| 48 | MSVC INCLUDE/LIB 污染 | icpx 继承 MSVC 头文件 | `vcvarsall.bat` 变量被 subprocess 继承 | `os.environ.pop/restore` |
| 49 | `cpp_wrapper` 传 MSVC 参数 | `/nologo` `/O2` `/MD` 出现在 icpx 命令 | inductor 默认 C++ wrapper | `inductor_config.cpp_wrapper=False` |
| 50 | float64 不支持 | `Double type not supported` | Arc A770 无 fp64 | `set_default_dtype(float32)` + 字符串替换 |
| 51 | `sycl_functions.h` 找不到 | 头文件缺失 | COMPILATION_HELPER include_dirs 不全 | 补全 triton_inc 路径 |
| 52 | 改提示词疑似重编译 | 326s vs 214s | 排查时发现 inductor trace 不跨 session | 确认：同进程内改提示词不触发，重启才 trace |

---

## 文件清单

```
ComfyUI-WINT4-XPU-beta/
├── __init__.py
├── wint4_model_quantizer.py     # 量化器
├── wint4_model_loader.py        # 加载器（含 Triton 加速）
├── wint4_xpu_ops.py             # 推理 ops（含 oneAPI build hook）
├── wint4_lora_loader.py         # 单 LoRA（含 compile 穿透）
├── wint4_lora_stack.py          # 多 LoRA Stack
├── wint4_lora_common.py         # key 归一化 + BFL 转换
├── wint8_quarot.py              # Hadamard 旋转
├── check_int4.py                # 模型诊断脚本
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 相关链接

- **本插件**: https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta
- **INT8 插件**: https://github.com/JWLHS/ComfyUI-WINT8-XPU
- **convert-to-quant**: https://github.com/newgrit1004/convert-to-quant

---

## 鸣谢

本插件全部由 **DeepSeek V4 Pro** 完成。

参考项目：
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-WINT8-XPU](https://github.com/JWLHS/ComfyUI-WINT8-XPU)
- [convert-to-quant (ctq)](https://github.com/newgrit1004/convert-to-quant)
- [FLA - Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)（Triton XPU 经验参考）

---

## License

MIT
```

---
