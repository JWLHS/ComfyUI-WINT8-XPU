
# ComfyUI-WINT8-XPU

> **INT8 per-row 模型量化 + LoRA 推理插件 — Intel Arc A770 16GB 优化，跨后端通用**

将扩散模型量化为 per-row INT8，显存节省 50%。支持 10 种 LoRA 格式叠加、ctq 加速量化。
**Python 模式通用（XPU / CUDA / ROCm / CPU）。Triton 加速为 Intel XPU 专用。**
# 
#
#    注意！因添加triton功能，现在wint8 model loder节点已经无法使用官方默认lora加载节点
#        可选路径： wint8 model loder --- wint8 lora loder
#        或者选择：官方默认unet节点---官方默认加载lora  官方节点依然能够识别wint8量化节点量化的模型！
#  wint8:预量化模型，并非最佳参数，仅为测试时使用。
#  链接: https://pan.baidu.com/s/1X7HzxlO1Bmq6hQKl8YAVMw?pwd=x2k2 提取码: x2k2 

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

## 功能概览

| 能力 | 状态 | 适用范围 |
|------|:--:|------|
| BF16/FP16/FP8 → INT8 量化 | ✅ | 通用 |
| ctq 加速量化（推荐） | ✅ | 通用 |
| QuaRot (Hadamard 旋转) 质量提升 | ✅ 可选 | 通用 |
| INT8 UNet 推理 | ✅ | 通用 |
| 单 LoRA 加载 | ✅ v6 新增自定义节点 | 通用 |
| 多 LoRA 叠加（串联 / Stack） | ✅ v6 新增 | 通用 |
| LyCORIS LoKr 格式 | ✅ v6 新增 | 通用 |
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
| UNet 存储 | 24 GB | **12 GB** | 6 GB |
| 推理显存 | ~24 GB | **~16-17 GB** | ~8-9 GB |
| A770 16GB 裸跑 | ❌ | ⚠️ 需 AIMDO | ✅ |

> INT8 推理需 ~16-17GB，略超 A770 16GB。启用 AIMDO DynamicVRAM 后可在 A770 上运行。

### Triton 加速（Arc A770, INT8, 8 steps）

| 模式 | 首次 | 同参数第二次 | 改分辨率后 |
|------|------|------|------|
| python | ~基准 | ~基准 | ~基准 |
| compile | ~含编译 | **显著加速** | 需重编译 |
| compile_freeze | ~含编译 | **最大加速** | 需重编译 |

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/JWLHS/ComfyUI-WINT8-XPU.git
```

依赖同 ComfyUI 标准环境，启动时自动安装。

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

**`wint8_env.pth`**
```
import wint8_env_setup
```

**`wint8_env_setup.py`**
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

### WINT8 Model Quantizer

**功能：** BF16/FP16/FP8 → INT8 per-row 量化 + 可选 QuaRot

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `model_name` | — | 选择源模型 |
| `model_type` | `flux2` | 模型架构，控制排除列表 |
| `quant_method` | `auto` | auto / ctq / builtin。ctq 更快质量更好 |
| `enable_quarot` | `False` | Hadamard 旋转，提升质量（可选） |
| `group_size` | 128 | QuaRot 分组大小（Boogu 自动 32） |
| `device` | `xpu` | 量化计算设备 |
| `output_filename` | `model_int8` | 保存至 `output/`，支持子目录 |

### WINT8 Model Loader

**功能：** 加载 INT8 模型，可选加速模式

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `unet_name` | — | 选择量化后的模型 |
| `model_type` | `flux2` | **必须和量化时一致** |
| `acceleration_mode` | `python` | python / compile / compile_freeze |

### WINT8 LoRA Loader / LoRA Stack

单 LoRA 链式串联 / 多 LoRA 一次叠加（最多 5 个）。

> ⚠️ **v6 起使用自定义 LoRA 节点。** PT 2.14 + Triton monkey-patch 下官方 LoraLoader 可能不兼容。
> 请使用 `WINT8 LoRA Loader` 和 `WINT8 LoRA Stack` 替代官方节点。

---

## 快速开始

```
1. 量化:
   WINT8ModelQuantizer
   ├─ model_name      = BF16/FP16 模型
   ├─ model_type      = krea2 / wan / ltx2 / flux2 / ...
   ├─ quant_method    = ctq (推荐) 或 builtin
   ├─ enable_quarot   = True (可选, 提升质量)
   ├─ group_size      = 128 (Boogu 自动 32)
   ├─ device          = xpu
   └─ output_filename = my_model_int8

2. 推理（无 LoRA）:
   WINT8ModelLoader → MODEL → KSampler → VAE Decode

3. 推理（单 LoRA）:
   WINT8ModelLoader → WINT8LoRALoader → KSampler

4. 推理（多 LoRA 串联）:
   WINT8ModelLoader → LoRA Loader (A) → LoRA Loader (B) → ...

5. 推理（多 LoRA Stack）:
   WINT8ModelLoader → WINT8LoRAStack（最多 5 槽）→ KSampler
```

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
6. INT8 推理显存 ~16-17GB，A770 16GB 需配合 AIMDO offload

---

## 常见问题

**不想用 Triton？**
选 python 模式。量化器照样用，所有功能一应俱全。

**NVIDIA/AMD 能用吗？**
量化器和 python 推理模式通用。Triton compile 模式为 XPU 专用。

**重启后要重新编译吗？**
Triton kernel 缓存（`.triton\cache\`）跨 session 持久。Inductor 图需重 trace（几十秒），但比首次快得多。

**为什么改分辨率要重编译？**
Inductor 为每个 input shape 生成特化 kernel。提示词、步数、CFG 不改变 tensor shape。

**INT8 能用官方 LoraLoader 吗？**
v6 起建议使用自带 `WINT8 LoRA Loader`。PT 2.14 的 monkey-patch 可能导致官方 Loader 静默失效。

**ctq 和 builtin 有什么区别？**
ctq 使用 learned rounding，量化质量更高，但需要额外安装 `convert-to-quant`。builtin 纯 PyTorch，无外部依赖。

---

## v6 新增 (2026-07)

| 功能 | 说明 |
|------|------|
| **Triton torch.compile 加速** | 三种模式，首次编译后同参数加速 |
| **自定义 LoRA Loader / Stack** | 替代官方节点，兼容 PT 2.14 monkey-patch |
| **LoKr 支持** | 量化层动态 Kronecker 展开 |
| **LoRA cleanup** | detach 时自动清理 + bake-in 恢复 |
| **oneAPI 2026.0 适配** | IGC 寄存器溢出修复 |
| **绘世启动器兼容** | `.pth` 注入方案 |
| **代码同步 WINT4** | build hook + float32 强制 + 电感静默 |

---

## v6 踩坑记录 — Triton compile 加速（200+ 次尝试，历时 3 天）

| # | 问题 | 现象 | 根因 | 修复 |
|---|------|------|------|------|
| 40 | IGC 寄存器溢出 | `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` | PT 2.12 + IGC 2025.3 无法处理融合 kernel | 升级 PT 2.14 + IGC 2026.0 |
| 41 | `tl.full` monkey-patch 崩溃 | `ValueError` | Triton 3.7.2 AST visitor 不兼容 lambda | 删除 monkey-patch，仅靠字符串替换 |
| 42 | `dynamic=True` IGC 溢出 | 同 #40 | dynamic kernel 更大 | 弃用，后续手写 kernel |
| 43 | `max_fusion_size=1` 无效 | 改分辨率仍重编译 | inductor 缓存 key 含 input shape | 确认设计如此 |
| 44 | compile + LoRA 失效 | `0 layers matched` | OptimizedModule 隐藏属性 | `while hasattr(dm, '_orig_mod')` 穿透 |
| 45 | 系统 Python sycl8 污染 | access violation | 系统 site-packages 有旧 DLL | 卸载系统 Python 中的 Intel 包 |
| 46 | oneAPI `latest` → 2025.3 | icpx 路径错误 | 符号链接未更新 | `mklink /D latest 2026.0` |
| 47 | 绘世不传 PATH | icpx 找不到 | 便携版不继承系统 PATH | build hook 硬编码 + `.pth` 注入 |
| 48 | MSVC INCLUDE/LIB 污染 | 编译失败 | vcvarsall 变量被子进程继承 | subprocess 前后 pop/restore |
| 49 | cpp_wrapper 传 MSVC 参数 | `/nologo` 等标志出现 | inductor 默认 C++ wrapper | `cpp_wrapper=False` |
| 50 | float64 不支持 | Double type error | Arc 无 fp64 | `set_default_dtype(float32)` + 字符串替换 |
| 51 | sycl_functions.h 找不到 | 头文件缺失 | include_dirs 不全 | 补全 triton_inc 路径 |
| 52 | 官方 LoRA 静默失效 | 加载成功但无效果 | monkey-patch 干扰官方 weight_function | v6 新增自定义 LoRA Loader |

---

## 文件清单

```
ComfyUI-WINT8-XPU/
├── __init__.py                  # 节点注册
├── wint8_model_quantizer.py     # 量化器（ctq + builtin）
├── wint8_model_loader.py        # 加载器（含 Triton 加速）
├── wint8_xpu_ops.py             # 推理 ops（含 oneAPI build hook + LoRA 注入）
├── wint8_lora_loader.py         # 单 LoRA（含 bake-in + compile 穿透）
├── wint8_lora_stack.py          # 多 LoRA Stack
├── wint8_lora_common.py         # key 归一化 + BFL 转换
├── wint8_quarot.py              # Hadamard 旋转
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 相关链接

- **本插件**: https://github.com/JWLHS/ComfyUI-WINT8-XPU
- **INT4 插件**: https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta
- **convert-to-quant**: https://github.com/newgrit1004/convert-to-quant

---

## 鸣谢

本插件全部由 **DeepSeek V4 Pro** 完成。

参考项目：
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-WINT4-XPU-beta](https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta)
- [convert-to-quant (ctq)](https://github.com/newgrit1004/convert-to-quant)
- [FLA - Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

---

## License

MIT
```

---

