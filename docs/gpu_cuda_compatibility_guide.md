# NVIDIA 显卡 / 驱动 / CUDA 适配关系指南

> 面向场景：买了一张 RTX 3060 (12G)，想跑 PyTorch 2.11（使用 CUDA 12.8 / cu128 版本），应该装什么驱动？

---

## 1. 三者的关系（一句话版）

```
显卡硬件 ← 显卡驱动（提供底层支持） ← CUDA Toolkit（编译/运行环境） ← PyTorch（应用层）
```

- **显卡硬件**（如 RTX 3060）决定了它属于哪个架构（Ampere），以及支持哪些计算特性。
- **显卡驱动**是硬件和 CUDA 之间的桥梁。**驱动向后兼容**：新驱动可以运行旧 CUDA 应用。
- **CUDA Toolkit** 是 PyTorch 等框架依赖的并行计算平台。每个 CUDA 版本要求一个**最低驱动版本**。
- **PyTorch** 针对特定 CUDA 版本预编译好（如 cu128 表示 CUDA 12.8），你只需装对应的 wheel 包。

> **你不需要手动安装 CUDA Toolkit**。PyTorch 的 wheel 自带了 CUDA runtime。你只需保证：
> 1. 有 NVIDIA 显卡（硬件满足）
> 2. 驱动版本 >= PyTorch 所用 CUDA 版本的最低要求

---

## 2. RTX 3060 的硬件能力

| 项目 | 值 |
|---|---|
| 架构 | **Ampere**（安培） |
| Compute Capability | **8.6**（即 sm_86） |
| 支持的 CUDA 版本 | CUDA **11.1 ~ 13.x 全系列** |
| 最新显卡驱动支持 | **R580 / R595 / R610 全系列均支持，Ampere 架构未被弃用** |
| Tensor Core | 第 3 代（支持 FP16 / INT8 混合精度） |
| 显存 | 12 GB GDDR6 |

### 那么最新驱动还能用吗？—— 能，RTX 3060（Ampere）没有被抛弃

NVIDIA 驱动会逐步停止对老旧 GPU 架构的支持。目前已被停止支持的架构：

| 架构 | Compute Capability | 最后支持的驱动分支 |
|---|---|---|
| Kepler | sm_30/35/37 | R470（已 EOL） |
| Maxwell | sm_50/52/53 | **R580**（仍有 LTS 支持到 2028 年） |
| Pascal | sm_60/61 | **R580**（同上） |
| Volta | sm_70 | **R580**（同上） |

而 **Ampere（RTX 3060 所在的架构，sm_86）** 在所有当前主流驱动分支中均受支持：

| 驱动分支 | 类型 | 发布日 | 支持 Ampere 吗？ |
|---|---|---|---|
| **R580** | LTS（长期支持，至 2028 年 6 月） | 2025 年 8 月 | ✅ |
| **R595** | 当前 Production Branch（至 2027 年 3 月） | 2026 年 3 月 | ✅ |
| **R610** | 最新 Beta/NFB | 2026 年 5 月 | ✅ |

数据来源：R595 GeForce 驱动官方产品列表明确包含 `GeForce RTX 3060 (12GB & 8GB)`；NVIDIA Data Center 驱动 R595 发布说明中明确支持 `NVIDIA Ampere architecture`。

**结论：** RTX 3060 不仅可以跑所有现代 CUDA 版本，**也完全兼容所有最新的显卡驱动**（R580 / R595 / R610），无需担心被驱动弃用的问题。

---

## 3. 为什么本项目的 PyTorch 用 cu128？

看 `pyproject.toml`（第 368-374 行）：

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
```

项目默认使用 **CUDA 12.8 (cu128)** 的 PyTorch。这是 PyTorch 2.11 中的稳定版 CUDA 后端。

根据 PyTorch 2.11 官方发布信息（2026 年 3 月），cu128 对 GPU 架构的支持情况：

| CUDA 版本 | PyTorch 2.11 中的定位 | 支持的 GPU 架构（含 RTX 3060？） |
|---|---|---|
| **12.8 (cu128)** | Stable（稳定版） | Turing(7.5), **Ampere 8.6 ✅ (RTX 3060)**, Hopper(9.0), Blackwell(10.0/12.0) |
| **13.0** | Stable（默认，从 2.11 起成为 PyPI 默认） | 同上 |

**cu128 完整覆盖 Ampere 架构（sm_86），RTX 3060 完美支持。**

### 安装命令

```bash
# CUDA 12.8（推荐，与项目一致）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 13.0（PyTorch 2.11 默认，不用加 index-url）
pip install torch torchvision
```

---

## 4. 我该装什么显卡驱动？

你只需要满足 PyTorch 所用的 CUDA 版本的**最低驱动要求**。因为**驱动向下兼容**，装个较新的驱动就一劳永逸。

### CUDA 12.8 (cu128) 的驱动要求

数据来源：[NVIDIA CUDA 12.8 Release Notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/) 及 [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/support-matrix.html)。

| 平台 | CUDA 12.8 官方最低要求 | GeForce 下载页实际可选的最老版本（你看到的） |
|---|---|---|
| **Linux** | ≥ 570.26.06 | ≥ 591.44（远超要求 ✅） |
| **Windows** | ≥ 570.65 | ≥ 591.44（远超要求 ✅） |

> **实际上你完全不用担心。** NVIDIA GeForce 驱动下载页能选到的最老版本（591.44）已经远高于 CUDA 12.8 的最低驱动要求（570.26/570.65）。这意味着随便选一个版本都能正常工作，没有选错的风险。

### 实际操作建议

**对于你来说，最省心的做法：**

| 你的系统 | 建议 |
|---|---|
| **Linux (Ubuntu/Debian 等)** | 装 **R570+ 系列驱动**（≥ 570.26.06），或直接装最新稳定版（当前 R595 系列）。建议用 `sudo ubuntu-drivers autoinstall` 或从 NVIDIA 官网下载。 |
| **Windows** | 去 [NVIDIA 驱动下载页](https://www.nvidia.cn/geforce/drivers/results/) 下载驱动。至于选 **Game Ready** 还是 **Studio Driver**，见下方说明。 |

### Game Ready 还是 Studio Driver？—— 深度学习选 **Studio Driver**

两者在 CUDA 计算层面**没有区别**——PyTorch/TensorFlow 在两种驱动下的运行速度和兼容性完全一样。区别在于更新策略和测试周期：

| 对比项 | Game Ready 驱动 | Studio Driver |
|---|---|---|
| 更新频率 | **约每月一次**，紧跟新游戏发布 | **约每季度一次**，节奏更慢 |
| 测试重点 | 新游戏首发优化、DLSS 等游戏特性 | 创意软件（Adobe、Blender 等）的 ISV 认证测试 |
| 稳定性 | 可能偶有 regression（回退 bug） | **经过更长时间的回归测试，更稳定** |
| CUDA / 深度学习 | CUDA 计算栈完全一致 ✅ | CUDA 计算栈完全一致 ✅ |
| 适合人群 | 需要第一时间玩新游戏的用户 | 工作站、开发者、追求稳定而非最新特性的用户 |

**建议：** 如果你这台机器**专门跑深度学习 / lerobot 数据采集**，选 **Studio Driver**——它更新少、经过更充分的稳定性测试，不会在训练中途因为驱动更新带来意外问题。

如果你同时也在这台机器上打最新的 3A 游戏，那选 **Game Ready** 也无妨——它对 PyTorch 没有负面影响。

> ⚠️ 无论选哪种，确保版本 ≥ R570（Linux ≥ 570.26.06 / Windows ≥ 570.65）即可满足 CUDA 12.8 的要求。

> **RTX 3060 在 Linux 和 Windows 下都完全受支持**，不用担心驱动不支持的问题。

### 检查当前驱动版本

```bash
# Linux
nvidia-smi

# Windows（PowerShell）
nvidia-smi
```

输出第一行会显示 `Driver Version`，例如 `Driver Version: 570.86.15`。

---

## 5. 常见疑问（FAQ）

### Q: 我装完驱动后 `nvidia-smi` 显示 CUDA 13.x / 13.2，不是 12.8，怎么办？

**什么都不用做，这是正常的。** 看这张图理清关系：

```
nvidia-smi 显示的 CUDA Version = 驱动内置的 CUDA Driver 最高支持版本
                                    ↓
                          PyTorch cu128 = 自己带了 CUDA 12.8 运行时库
                                    ↓
                          PyTorch 通过驱动提供的接口在 GPU 上执行计算
```

`nvidia-smi` 显示的是驱动的能力上限，不代表你装了 CUDA 12.8 或 CUDA 13.x。PyTorch cu128 自带 CUDA 12.8 的库，驱动只要版本够新（≥ 570.26），就能正确运行这些库。

**一句话：装好驱动后，直接 `pip install torch --index-url https://download.pytorch.org/whl/cu128` 即可，`nvidia-smi` 的数字不用管。**

### Q: 我需要单独装 CUDA Toolkit 吗？

**不需要。** PyTorch 的 CUDA wheel（如 cu128）已经包含了运行所需的 CUDA runtime。你只需装显卡驱动即可。

如果你之后需要从源码编译 CUDA 代码，才需要安装完整的 CUDA Toolkit。

### Q: "CUDA 12.8" 和显卡驱动里显示的 "CUDA Version" 是什么关系？

`nvidia-smi` 输出的 "CUDA Version" 表示**你的驱动最高支持到哪个 CUDA 版本**。例如驱动显示 `CUDA Version: 12.8`，意思是它能运行 CUDA 12.8 及以下所有的 CUDA 程序。

驱动的 CUDA Version >= PyTorch 的 CUDA 版本，就能正常工作。

### Q: 如果我装了最新驱动（比如 R595 或 R610，支持 CUDA 13.x），还能跑 PyTorch cu128 吗？

**能。** 驱动完全向下兼容。CUDA 12.8 的程序可以在 R595/R610 驱动下正常运行。

### Q: 以后会有新的驱动停止支持 RTX 3060 吗？

**短期内不会。** 根据 NVIDIA 的历史记录，一个架构从首发到被驱动弃用通常有 **6~8 年**的生命周期。

| GPU 架构 | 首发年份 | 驱动弃用年份 | 跨度 |
|---|---|---|---|
| Maxwell | 2014 | 2025 (R580) | ~11 年 |
| Pascal | 2016 | 2025 (R580) | ~9 年 |
| Volta | 2017 | 2025 (R580) | ~8 年 |
| **RTX 3060 (Ampere)** | **2021** | **？** | **至少到 2029+** |

RTX 3060 发布于 2021 年，Ampere 架构被弃用至少还要等 4~5 年。在此之前，你尽管放心安装最新驱动。

---

## 6. 总结：针对你的场景

> 买 RTX 3060 (12G) → 装 PyTorch 2.11 cu128 → 跑 lerobot（数据采集/训练/评估）

| 步骤 | 操作 |
|---|---|
| ① 买显卡 | RTX 3060 12G（Compute Capability 8.6） |
| ② 装驱动 | Linux / Windows: 去 NVIDIA 官网下载 **Studio 驱动**（选你能选到的最新或较新版本即可，最老的 591.44 也远超 CUDA 12.8 要求） |
| ③ 装 PyTorch | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| ④ 装 lerobot | `pip install -e ".[core_scripts]"` |
| ⑤ 验证 | `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"` 应输出 `True` 和 `NVIDIA GeForce RTX 3060` |

### 一句话总结

**RTX 3060 + PyTorch 2.11 cu128 完全兼容。NVIDIA 官网能下载到的最老驱动（591.44）已远超 CUDA 12.8 要求，随便选一个版本都能正常工作。不需要装 CUDA Toolkit。**
