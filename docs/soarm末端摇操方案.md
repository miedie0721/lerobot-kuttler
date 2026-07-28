键盘摇操（SO-101 5 轴，以 shoulder_pan 为基坐标系）

[键盘事件]
    ↓
① 读当前关节角 q = (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
    ↓
② a/d/q/e 直接更新（不经过 IK）：

    a/d → sp_new = shoulder_pan ± Δsp
    q/e → wr_new = wrist_roll ± Δwr

③ 建立肩部坐标系（由 shoulder_pan 角度决定）：

    原点是 shoulder_pan 关节在 base_link 下的位置：(0.0388, 0, 0.0624)
    X 轴 = [cos(sp_new), sin(sp_new), 0]        ← 臂前方
    Y 轴 = [-sin(sp_new), cos(sp_new), 0]       ← 臂左边
    Z 轴 = [0, 0, 1]                            ← 竖直
    Y 轴上的坐标永远 = 0（不可以侧移）

④ 在肩部坐标系里算目标位姿：

    FK 当前 EE 在 base_link 的位置 → 转到肩部系 → (r, 0, z)

    w/s → r_target = r ± Δr        ← 在肩部 X 轴伸缩
    j/u → z_target = z ± Δz        ← 在肩部 Z 轴升降
    Y = 0                          ← 在肩部 Y 轴始终为 0

    姿态 = R_down（竖直向下，固定）

    T_target_sp = 肩部坐标下的 4×4 位姿
    = [ 1  0  0  r_target ]
      [ 0  1  0    0      ]
      [ 0  0  1  z_target ]
      [ 0  0  0    1      ]    合并姿态后得到完整 T_target_sp

⑤ T_target_sp → 转回世界坐标 → T_target_base

    T_target_base = T_shoulder_frame @ T_target_sp
    （T_shoulder_frame = shoulder_pan 在 base 下的 4×4 位姿）

⑥ IK(T_target_base, q_init=[shoulder_lift, elbow_flex, wrist_flex],
        joint_names=["shoulder_lift", "elbow_flex", "wrist_flex"])
      → [sl_target, ef_target, wf_target]    ← 只解这 3 个关节

⑦ 合成为电机指令：

    shoulder_pan  ← sp_new       (a/d, 直接)
    shoulder_lift ← sl_target    (IK)
    elbow_flex    ← ef_target    (IK)
    wrist_roll    ← wr_new       (q/e, 直接)
    gripper       ← 合/开        (i/k, 直接)

为了便于复位，设置了reset键（P键）：
按 P 键（仅仅按一次P键就能来到目标state）  
目标state：{"shoulder_pan":-16.44,"shoulder_lift":-26.95,"elbow_flex":35.21,"wrist_flex":91.16,"wrist_roll":6.81,"gripper":1.0}