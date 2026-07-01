# WINT8 — Per-Row INT8 量化插件 for ComfyUI (Intel Arc XPU)

> Per-row INT8 model quantization & loading for ComfyUI on Intel Arc A770 / B580.

---

## 目录

1. [功能概览](#功能概览)
2. [安装](#安装)
3. [节点详解](#节点详解)
   - [WINT8 Model Quantizer](#wint8-model-quantizer)
   - [WINT8 Model Loader](#wint8-model-loader)
4. [量化方法对比](#量化方法对比)
5. [AIMDO DynamicVRAM 使用建议](#aimdo-dynamicvram-使用建议)
6. [完整工作流](#完整工作流)
7. [支持模型](#支持模型)
8. [排除列表说明](#排除列表说明)
9. [常见问题](#常见问题)
10. [已验证效果](#已验证效果)
11. [Bug 修复记录](#bug-修复记录)
12. [文件结构](#文件结构)
13. [v5.1 同步更新](#v51-同步更新-2026-07-02)
14. [关于 ComfyUI 原生加载器](#关于-comfyui-原生加载器)
15. [关于性能](#关于性能)

---

## 功能概览

将 **BF16 / FP16 / FP8** 扩散模型量化为 **per-row INT8**，实现：

| 指标 | 效果 |
|------|------|
| **显存** | 节省 ~50%（vs BF16） |
| **推理速度** | ≈ BF16 / FP16，无体感差异 |
| **画质** | 正常，无花屏、无偏色 |
| **加载** | < 2 秒（纯 tensor 赋值，无编译开销） |
| **LoRA** | 支持（权重建议调高 1.5-2× 以补偿 INT8 精度损失） |
| **AIMDO 兼容** | 兼容 DynamicVRAM，共享显存正常释放不泄漏 |

### 原理

```
量化阶段（离线）：             推理阶段（在线）：

weight_fp16 (out, in)         safetensors 加载
    │                             │
    ▼                             ▼
amax = |w|.max(dim=1)         nn.Parameter(int8)  +  weight_scale(float32)
scale = amax / 127               │
w_int8 = round(w / scale)        ▼
                            cast_bias_weight (AIMDO 搬到 XPU)
保存：w_int8 + scale                │
                                ▼
                            反量化：w_fp = w_int8.float() × scale
                                │
                                ▼
                            F.linear(x, w_fp, bias)
                                │
                                ▼
                            uncast_bias_weight (AIMDO 搬回 CPU)
```

每行权重独立计算 scale，反量化仅一次 broadcast 乘法，无额外内存开销。

---

## 安装

### 方法一：git clone（推荐）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/JWLHS/ComfyUI-WINT8-XPU.git
```

### 方法二：手动下载

1. 打开 https://github.com/JWLHS/ComfyUI-WINT8-XPU
2. 点击绿色 **"Code"** → **"Download ZIP"**
3. 解压到 `ComfyUI/custom_nodes/ComfyUI-WINT8-XPU/`

### 依赖

`requirements.txt` 中已声明，ComfyUI 启动时自动安装：

```
convert-to-quant    # ctq CLI（可选——不装也能用内置量化）
safetensors         # 模型文件读写
```

> **注意：** `convert-to-quant` 是可选的。如果未安装，插件自动 fallback 到内置量化，功能完全正常。

---

## 节点详解

---

### WINT8 Model Quantizer

**位置：** `WINT8` 类别 → `WINT8 Model Quantizer`

**功能：** 将 BF16/FP16/FP8 模型量化为 per-row INT8，输出 `.safetensors` 文件。

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `model_name` | 下拉 | — | 选择 `models/diffusion_models/` 下的模型 |
| `model_type` | 下拉 | `flux2` | 模型架构，控制排除列表 |
| `quant_method` | 下拉 | `auto` | 量化方法：`auto` / `ctq` / `builtin` |
| `enable_quarot` | 开关 | `False` | Hadamard 正交旋转，提升质量（仅 `builtin`） |
| `group_size` | 整数 | `128` | QuaRot 分组大小，64-256，步长 64 |
| `device` | 下拉 | `xpu` | 量化计算设备 |
| `output_filename` | 文本 | `model_int8` | 输出文件名，保存至 `ComfyUI/output/` |

#### `quant_method` 详解

| 选项 | 行为 | 适用场景 |
|------|------|------|
| **`auto`** | 优先调用 ctq → ctq 失败自动 fallback 到内置量化 | 日常使用，最省心 |
| **`ctq`** | 强制使用 ctq → 失败直接报错 | 需要 learned rounding 时 |
| **`builtin`** | 直接用内置 per-row INT8，**可配合 QuaRot** | ctq 报错时，或需要 QuaRot 时 |

#### `enable_quarot`（QuaRot / ConvRot）

勾选后对每组权重应用 Hadamard 正交旋转：

- **原理：** 浮点 outlier 被均匀分散到整行 → 量化误差更小 → 细节保留更好
- **仅在 `builtin` 模式下生效**（ctq 模式由 ctq 自己的 `--convrot` 控制）
- `group_size` 必须是 **2 的幂**，且需整除权重的 `in_features`
- 量化时旋转权重（离线），推理时旋转激活（在线）— 两处由插件自动处理
- **Boogu 自动覆盖：** `model_type=="boogu"` 时自动强制 `group_size=32`，覆盖率 27%→100%

#### 量化速度参考（A770 XPU）

| 模型 | 大小 (BF16) | 量化耗时 |
|------|:---:|:---:|
| Krea2 (12B) | ~24 GB | ~2-3 分钟 |
| Flux2-Klein (9B) | ~18 GB | ~1.5-2 分钟 |
| Qwen (7B) | ~14 GB | ~1 分钟 |

#### FP8 → INT8 转换

如果输入模型是 FP8（`Float8_e4m3fn`），内置量化会自动转为 float32 再量化，输出 INT8。文件体积变化极小（仅增加 per-row scale，约 0.1%）。目的不是省体积，而是用 per-row scale 替代 FP8 的全局 scale 提升质量。

---

### WINT8 Model Loader

**位置：** `WINT8` 类别 → `WINT8 Model Loader`

**功能：** 加载量化后的 INT8 模型，输出 `MODEL` 直接接入采样流程。

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `unet_name` | 下拉 | — | 选择量化后的 `.safetensors` 文件 |
| `model_type` | 下拉 | `flux2` | **必须和量化时一致** |

#### 内部推理流程

```
加载 safetensors
    │
    ▼
Int8XPUOps.Linear._load_from_state_dict:
    - INT8 权重  → nn.Parameter（AIMDO 可见，支持延迟加载）
    - weight_scale → register_buffer
    - 解析元数据（QuaRot 标记 / group_size）
    │
    ▼
推理 forward（逐层）:
    cast_bias_weight(self, x)          ← AIMDO 把 weight/bias 从 CPU 搬到 XPU
        │
        ▼
    weight_scale 对齐到 x.device
        │
        ▼
    反量化: w_dq = weight.float() × w_scale  → comp_dtype
        │
        ▼
    (可选) QuaRot 旋转激活
        │
        ▼
    F.linear(x, w_dq, b_dq)
        │
        ▼
    uncast_bias_weight(...)            ← AIMDO 把 weight/bias 搬回 CPU + 释放显存
```

> **和 BF16 原生路径完全对称** — 只是中间多了反量化 + 可选 QuaRot。

---

## 量化方法对比

| | ctq | builtin | builtin + QuaRot |
|---|---|---|---|
| **格式** | per-row INT8 | per-row INT8 | per-row INT8 |
| **舍入方式** | Learned rounding | 简单 round | 简单 round + Hadamard |
| **质量** | 最高 | 良好 | 优秀 |
| **依赖** | `convert-to-quant` | 无 | 无（scipy 可选） |
| **模型兼容** | 部分模型有 bug（见 FAQ） | 全部 | 全部（需 in_f 整除 group_size） |
| **推理速度** | 一致 | 一致 | 略有 overhead（Hadamard 旋转） |

### 推荐

| 场景 | 推荐配置 |
|------|------|
| ctq 可用且模型兼容 | `quant_method = auto` |
| ctq 报错（如 Qwen FP8 相关） | `quant_method = builtin` |
| 追求最高质量 | `quant_method = builtin` + `enable_quarot = True` |

---

## AIMDO DynamicVRAM 使用建议

### INT8 模型 — 推荐关 AIMDO

**节点设置：** `XPU AIMDO Status` → `Enable_DynamicVRAM = OFF`

| 模型 | 显存 (INT8) | A770 16GB 是否够 |
|------|:---:|:---:|
| Krea2 (12B) 1080×1920 | ~12 GB | ✅ 够 |
| Flux2-Klein (9B) | ~9 GB | ✅ 够 |
| Qwen (7B) | ~7 GB | ✅ 够 |
| Z-Image | ~10 GB | ✅ 够 |

> **INT8 模型显存已减半，大多数情况 16GB A770 够用。关 AIMDO 可避免共享显存管理开销，推理速度略快。**

### BF16 原版模型 — 必须开 AIMDO

原版 BF16 模型（~24 GB）远超 16 GB 显存，必须开 DynamicVRAM 做延迟加载。

### 如果 INT8 + AIMDO 同时开

**完全兼容**（Bug #9 已修复）。forward 走完整的 `cast_bias_weight` → 反量化 → 计算 → `uncast_bias_weight` 闭环，共享显存用完立即释放，不累积、不泄漏。

---

## 完整工作流

### 第一步：量化模型

```
┌─────────────────────────────────────┐
│  WINT8 Model Quantizer              │
│                                     │
│  model_name      = 选择 BF16 模型   │
│  model_type      = flux2 / qwen ... │
│  quant_method    = builtin          │
│  enable_quarot   = True（可选）     │
│  group_size      = 128              │
│  device          = xpu              │
│  output_filename = my_model_int8    │
│                                     │
│  → 执行                             │
└─────────────────────────────────────┘
```

输出：`ComfyUI/output/my_model_int8.safetensors`

### 第二步：构建推理工作流

```
┌──────────────────────────┐
│  WINT8 Model Loader      │
│  unet_name  = 刚量化的文件│
│  model_type = 和量化一致  │
│              ↓ MODEL      │
└──────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  XPU AIMDO Status    │
│  Enable_DynamicVRAM  │
│  = OFF（推荐）       │
└──────────────────────┘
         │
         ▼
┌──────────────────────┐
│  KSampler            │
│  → VAE Decode        │
│  → 出图              │
└──────────────────────┘
```

### 第三步（可选）：连接 LoRA

```
WINT8 Model Loader → MODEL
    │
    ├──→ LoRA Loader → MODEL
    │         │
    └──→ KSampler ←──┘
```

> LoRA 生效，但信号偏弱（INT8 精度限制），建议将 LoRA 权重调至 **1.5-2.0×**。

---

## 支持模型

| model_type | 含义 | ctq 支持 |
|:---|------|:---:|
| `flux2` | Flux2-Klein 系列 | ✅ 原生 flag |
| `qwen` | Qwen-Rapid 系列 | ✅ 原生 flag |
| `z-image` | Z-Image 系列 | ✅ 原生 flag |
| `wan` | Wan 视频模型 | ✅ 原生 flag |
| `ltx2` | LTX2 视频模型 | ✅ 原生 flag |
| `chroma` | Chroma/Distillation | ✅ 原生 flag |
| `krea2` | Krea2 图像模型 | ✅ 正则排除 |
| `boogu` | Boogu 系列 | ✅ 正则排除 |
| `ernie` | ERNIE 系列 | ✅ 正则排除 |
| `hidream` | HiDream 系列 | ✅ 正则排除 |
| `ideogram4` | Ideogram 4 系列 | ✅ 正则排除 |
| `auto` | 通用（不排除任何层） | ❌ 无 |

---

## 排除列表说明

以下类型的层保持原精度，不量化（和 `convert_to_quant` 行为一致）：

| 层类型 | 示例 key | 原因 |
|------|------|------|
| 输入投影 | `img_in`, `txt_in` | 第一层对精度敏感 |
| 输出投影 | `final_layer`, `proj_out`, `head` | 最后一层影响最终输出 |
| 时间嵌入 | `time_in`, `t_embedder`, `time_projection` | 小参数，量化无益 |
| Norm 层 | `adaLN`, `norm_out`, `norm_q`, `norm_k` | 非 2D 权重 |
| 小 Embedding | `patch_embedding`, `text_embedding` | 参数量极小 |
| 特殊注册层 | `learnable_registers`, `q_norm`, `k_norm` | 非标准 Linear，量化会坏 |
| 视频编码器 | `motion_encoder` | 非 Transformer 层，保持原精度 |

---

## 常见问题

### Q: ctq 量化 Qwen 模型时报错 `"sum_cpu" not implemented for 'Float8_e4m3fn'`？

**A:** 这是 ctq 的 bug——它在 `--int8` 模式下对 Qwen 错误调用了 FP8 路径。解决方案：选择 `quant_method = builtin`。

### Q: 加载时看到 `[WARNING] Missing weight for layer model.lm_head`？

**A:** 正常。`lm_head` 是语言模型头，图像生成时不参与计算。忽略即可，不影响画质。

### Q: 量化后文件体积和原模型一模一样？

**A:** 检查日志中的 `Quantized:` 数字。如果为 0：源模型可能已是 INT8/FP8——`_should_quantize` 不匹配。确保源模型是 BF16/FP16/FP8，且 `model_type` 正确。

### Q: INT8 模型 + AIMDO DynamicVRAM 共享显存一直涨？

**A:** 请更新到最新版（Bug #9 已修复）。确保 `wint8_xpu_ops.py` 的 forward 中走的是 `cast_bias_weight` / `uncast_bias_weight` 完整闭环。

### Q: QuaRot 勾上了但没有效果？

**A:** QuaRot 仅在 `quant_method = builtin` 时生效。检查：
1. `enable_quarot = True`
2. `quant_method = builtin`
3. 权重的 `in_features` 能被 `group_size` 整除

### Q: 原生 Load Diffusion Model 能加载 INT8 模型吗？

**A:** 新版 ComfyUI 可以。核心 `ops.py` 已内置 `int8_tensorwise` 格式支持（含 `convrot`/QuaRot 协议）。如果你的 ComfyUI 版本较新，**优先尝试原生加载器**。遇到兼容性问题再切回 `WINT8 Model Loader`。本插件的量化器产生的格式与原生协议完全对齐。

---

## 已验证效果

| 测试项 | 配置 | 结果 |
|------|------|------|
| Krea2 INT8 1080×1920 显存 | A770 16GB | ~12 GB（vs BF16 ~24 GB） |
| 推理速度 | 同上 | ≈ BF16，无体感差异 |
| 画质 | 多个 prompt 对比 | 正常，无花屏/偏色 |
| LoRA | 权重 2.0× | 生效 |
| AIMDO DynamicVRAM | INT8 模型 | 共享显存正常释放，不累积 |
| 多次推理 | 连续 3 轮 | 显存稳定，不泄漏 |
| 加载速度 | safetensors → MODEL | < 2 秒 |
| FP8 → INT8 转换 | builtin 模式 | 正常 |

---

## Bug 修复记录

本插件历经 9 轮 bug 修复，已稳定：

| # | Bug 现象 | 根因 | 修复方式 |
|---|------|------|------|
| 1 | 花屏 | 量化器 key 用下划线，加载器用点分隔找不到 scale | `rsplit(".weight", 1)[0]` + `.weight_scale` |
| 2 | 精度/偏色 | 排除列表不完整 | 从 ctq `constants.py` 同步 |
| 3 | 元数据不识别 | ctq 和加载器键名不同 | 双读 `group_size`/`block_size`、`convrot`/`quarot` |
| 4 | 缺失 `input_scale` | ctq 块状格式需要但未写入 | 量化器写入；加载器消费不报错 |
| 5 | unexpected key 警告 | `_build_qlinear` 子模块自动注册 | `object.__setattr__` 绕过 |
| 6 | bias device mismatch | AIMDO 把 bias 移 CPU，forward 未搬回 | `.to(device=x.device, dtype=comp_dtype)` |
| 7 | 循环导入 | loader 顶层 import 触发 ops 初始化 | import 移至方法内 |
| 8 | ops 文件被 loader 覆盖 | 误把 loader 代码写入 ops | 恢复 `Int8XPUOps` 类 |
| 9 | AIMDO 共享显存泄漏 | INT8 forward 绕过 `cast_bias_weight`/`uncast_bias_weight` | 走 AIMDO 完整闭环 |

---

## 文件结构

```
ComfyUI-WINT8-XPU/
├── __init__.py                  # ComfyUI 节点注册
├── wint8_model_quantizer.py     # 量化节点（auto / ctq / builtin + QuaRot）
├── wint8_model_loader.py        # 加载节点
├── wint8_xpu_ops.py             # 推理 ops（AIMDO 闭环 + per-row 反量化 + LoRA）
├── wint8_quarot.py              # Hadamard 正交旋转（离线权重 + 在线激活）
├── requirements.txt             # 依赖声明
├── README.md                    # 本文档
├── LICENSE                      # MIT License
└── .gitignore
```

---

## v5.1 同步更新 (2026-07-02)

从 WINT4 v5.1 回移植以下修复：

### Bug 修复

| Bug | 修复 | 文件 |
|------|------|------|
| Conv2d 层 `kernel` 后缀不匹配 | `_load_from_state_dict` 加 kernel 回退，`weight` 找不到时尝试 `kernel` | `wint8_xpu_ops.py` |
| Wan `motion_encoder` 被误量化 | 排除列表补全 | `wint8_model_quantizer.py` |
| LTX2.3 `learnable_registers` / `q_norm` / `k_norm` 被误量化 | 排除列表补全 | `wint8_model_quantizer.py` |

### 新增

| 功能 | 说明 |
|------|------|
| Boogu gs=32 自动覆盖 | `model_type=="boogu"` 且启用 QuaRot 时自动将 group_size 强制为 32，QuaRot 覆盖率 27%→100% |

---

## 关于 ComfyUI 原生加载器

ComfyUI 核心 `ops.py` 已内置 `int8_tensorwise` 量化格式支持（含 `convrot` QuaRot 协议）。**如果你的 ComfyUI 版本较新，可以直接用原生 `Load Diffusion Model` 节点加载 INT8 模型，无需使用本插件的 `WINT8 Model Loader`。**

本插件仍然提供：
- **WINT8 Model Quantizer** — 离线量化节点（原生不提供）
- **WINT8 Model Loader** — 兼容旧版 ComfyUI（原生加载器不支持 INT8 时使用）

> 建议：优先尝试原生 `Load Diffusion Model`；遇到兼容性问题再切回 `WINT8 Model Loader`。

---

## 关于性能

本插件为 **纯 Python 实现**，无自定义 CUDA/SYCL kernel：

- **推理速度** ≈ BF16/FP16，无体感差异（反量化仅为一次 broadcast 乘法）
- **无额外加速** — 不包含 Triton/oneAPI/SYCL 加速路径（Intel XPU Triton 后端尚未成熟）
- **适合 XPU 用户**（Intel Arc A770/B580）— ComfyUI 原生加载器在 XPU 上可正确加载 INT8 权重，本插件量化器产生兼容格式
- 量化器输出格式与 ComfyUI 原生 `int8_tensorwise` 协议完全对齐，不绑定本插件

> 如果你的硬件是 NVIDIA/CUDA，建议使用其他已集成了 CUDA kernel 的 INT8 方案（如 `bitsandbytes`、`torchao`）。本插件的优势在于 XPU 兼容性和格式与 ComfyUI 原生协议完全对齐。

---

## License

MIT © 2026 JWLHS

---

## 相关链接

- 仓库地址：https://github.com/JWLHS/ComfyUI-WINT8-XPU
- 问题反馈：https://github.com/JWLHS/ComfyUI-WINT8-XPU/issues
- ComfyUI：https://github.com/comfyanonymous/ComfyUI
