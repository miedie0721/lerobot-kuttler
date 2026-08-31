# HIL-SERL 真机强化学习

## `train_config_hilserl_so101.json` 超参数全景

### 分组表（★ = 对训练/效果影响大）

**环境与机器人**
| 参数 | 当前值 | 作用 | 调参提示 |
|------|--------|------|---------|
| `env.fps` | 30 | 控制频率 | 低 → 动作粗；高 → 每步位移小 |
| `env.processor.reset.control_time_s` | 15.0 | ★ 单集时长上限 | 任务来不及做就加大；`terminate_on_success` 会自动提前结束 |
| `fixed_reset_joint_positions` | [-5.407,...] | ★ 每集起点 | 必须与任务起始位一致 |
| `end_effector_step_sizes` | 0.002×3 | ★ 动作→位移换算 | 太小机械臂几乎不动；太大一出界就失败 |
| `end_effector_bounds` | min/max | 末端安全边界 | 缩得越紧学习越快，但要覆盖任务范围 |

**数据质量**
| 参数 | 当前值 | 作用 | 调参提示 |
|------|--------|------|---------|
| `image_preprocessing.crop_params_dict` | front:[36,19,77,90] side:[2,8,56,65] | ★★ 图像 ROI | **已填 ROI**（裁剪到工作区）；若不裁剪 → 模型看全图含背景，视觉学习慢 |
| `resize_size` | [128,128] | 输入分辨率 | 128 是性价比折中 |
| `observation.display_cameras` | true | 调试显示 | 正式训练改 false |

**策略网络**
| 参数 | 当前值 | 作用 | 调参提示 |
|------|--------|------|---------|
| `vision_encoder_name` | lerobot/resnet10 | ★ 视觉编码器 | resnet10 轻量；换大的更强但慢 |
| `freeze_vision_encoder` | false | ★ 是否冻结视觉 | false=端到端微调（推荐但需足够数据） |
| `device / storage_device` | cuda / cpu | 计算/存储设备 | CPU 存储已改好，避免爆显存 |

**经验池**
| 参数 | 当前值 | 作用 | 调参提示 |
|------|--------|------|---------|
| `online_buffer_capacity` | 12000 | ★ 在线数据容量 | ≈6.7 分钟数据后开始 FIFO 丢弃 |
| `offline_buffer_capacity` | 3200 | ★ demo 容量 | 要能装下全部 demo（3200 步≈11 集×300步） |
| `online_ratio` | 0.5 | ★★ demo:在线采样比例 | 高→稳但学得死；低→探索多 |
| `online_step_before_learning` | 100 | 热身步数 | 前期攒数据再开始学 |

**SAC 算法**
| 参数 | 当前值 | 作用 | 调参提示 |
|------|--------|------|---------|
| `utd_ratio` | 4 | ★★ 每步更新次数 | 1=标准SAC；4=更省样本；20=RLPD 激进值 |
| `batch_size` | 32 | 采样批次 | 大→梯度稳但吃显存 |
| `discount` | 0.99 | 折扣因子 | 0.95 更短视；0.99 更看远期 |
| `actor_lr/critic_lr` | 0.0003 | ★ 学习率 | 太大震荡；太小学不动 |
| `critic_target_update_weight` | 0.005 | 软更新 τ | 越小目标网络越稳越慢 |
| `temperature_init` | 1.0 | 初始熵 | 大→探索多 |
| `num_critics` | 2 | 双 Q 数量 | 默认 2；可选 `num_subsample_critics`（当前 null 不启用，实现在 sac_algorithm.py L295-302，高 UTD 下子采样防过拟合） |

**训练基础设施**
| 参数 | 当前值 | 作用 | 调参提示 |
|------|--------|------|---------|
| `online_steps` | 100000 | ★ 总训练步数 | 30fps 下 ≈55 分钟 |
| `policy_parameters_push_frequency` | 4 | ★ 参数推送频率（**单位：秒**） | 每 4 秒推一次最新参数；越大→策略更新到真机越滞后 |
| `save_freq` | 1000 | 存 checkpoint | 小→存得勤 |

### 调参优先级（先调这些）
```
第1档（改配置立刻见效）：crop_params_dict（填 ROI）、display_cameras=false
第2档（影响学习速度）：online_ratio、utd_ratio
第3档（影响任务可达）：control_time_s、end_effector_step_sizes、fixed_reset_joint_positions
第4档（网络容量）：vision_encoder、freeze_vision_encoder
第5档（锦上添花）：batch_size、learning rate、discount
```

---

## 诊断与优化实战

### 第一步：先诊断，不要急着调参

**A. 确认 reward 通路（最重要）**
- 训练中按右键 → 那一帧 `REWARD` 应为 1.0
- 若 `reward_classifier=null`，**唯一奖励来自你的右键**——如果没按过右键，全程 reward=0，策略永远学不到"成功长什么样"
- 检查方法：learner 日志/断点看 batch 里 `REWARD.max()`

**B. 确认训练真的在更新**
- wandb（或tensorboard）看 `loss`、`Q value` 是否在变化
- 若 `utd_ratio=4` 生效，训练线程应明显占用 GPU

**C. 确认数据在流动**
- `intervention_rate`：若你干预过多（>50%），说明策略太差，可能 demo 不足或 reward 没给对
- 离线 buffer 是否成功装载 demo（`initialize_offline_replay_buffer` 日志）

**D. 确认探索范围合理**
- `end_effector_bounds` 是否覆盖任务区域（用 `lerobot-find-joint-limits` 实测）
- `crop_params_dict` 是否为空（空 = 模型看整张图，视觉学习慢）

**E. 确认训练时长**
- 20 分钟 @ 30fps ≈ 36000 步，而 `online_steps=100000`。**20 分钟可能只是刚开始**
- 但"完全没起色"通常不是时间问题，而是上面的 reward/数据问题

### 第二步：优化决策树（按症状）

```
症状1：策略完全不动 / 原地抖
  ├─ end_effector_step_sizes 太小 → 调到 0.005~0.01
  ├─ action 被 bounds 截死 → 检查 end_effector_bounds
  └─ 电机没跟上（max_relative_target）→ 检查串口/速度

症状2：动但乱冲、出界
  ├─ 缩小 end_effector_bounds（贴合任务工作区）
  ├─ 降低 step_sizes
  └─ 提高 deadzone 减少噪声抖动

症状3：几乎不干预但成功率为 0
  ├─ reward 没给对（右键没触发 REWARD=1）→ 优先排查
  ├─ demo 太少/质量差 → 重录 20-30 集，覆盖任务全流程
  └─ control_time_s 太短，任务做不完

症状4：干预率高（策略老跑偏）
  ├─ 增加 demo 数量、保证每集都成功
  ├─ 提高 online_ratio 里的 demo 占比（0.5→0.7）
  ├─ crop ROI 对准工作区
  └─ utd_ratio 4→8，让 demo 被更多次学习

症状5：有进步但慢、不稳定
  ├─ utd_ratio 提到 8~20
  ├─ 降低 lr（0.0003→0.0001）求稳
  └─ freeze_vision_encoder=true 先训稳定再解冻
```

### 第三步：低成本快速实验清单（一次只改一个变量）
1. 填 `crop_params_dict` ROI ✅ 必做
2. `display_cameras` → false
3. 重新录 demo：**20-30 集、每集都成功、覆盖任务全流程**（离线 buffer 12000 步足够装）
4. 训练中**勤按右键**（做对就按），确保 reward 非零
5. 观察干预率：目标 <30%

---

## 参考资料汇总

| 资料 | 链接/位置 | 阶段 |
|------|-----------|------|
| HIL-SERL 论文 | https://arxiv.org/abs/2410.21845 | 2 |
| SERL 论文 | https://arxiv.org/abs/2401.16013 | 1 |
| SAC 论文 | https://arxiv.org/abs/1801.01290 | 0 |
| RLPD 论文 | https://arxiv.org/abs/2207.13586 | 1 |
| 官方文档（英文） | `docs/source/hilserl.mdx` | 2 |
| 官方文档（中文） | `docs/source/hilserl_zh.mdx` | 2 |
| 你的操作笔记 | `docs/hilserl.txt` | — |
| 训练配置 | `src/lerobot/configs/train_config_hilserl_so101.json` | 4 |
| 采集配置 | `src/lerobot/configs/env_config_so101_spacemouse.json` | 4 |

---

## 学习完成自检清单

- [ ] 能写出 SAC 的 critic/actor/temperature 损失函数
- [ ] 能解释 RLPD 为什么让 UTD 到 20 有效
- [ ] 能画出 actor-learner 架构与 gRPC 数据流
- [ ] 能说清 SpaceMouse 左右键在训练中对应什么训练信号
- [ ] 能在代码中定位 `InterventionActionProcessorStep` 的 reward/done 处理
- [ ] 能说出当前 JSON 中 5 个最影响效果的参数及调法
- [ ] 面对"效果差"，能按 A→E 顺序完成诊断
