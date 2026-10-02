# TRAIN-01-A 实际 checkpoint 验证（2026-10-02）

这些报告来自已训练两轮的历史 smoke 权重重新进行的 **本地 CPU 推理**，不是本轮新训练或棋力验收。原 PT / NPZ 权重 SHA、实际模型配置、源码提交、案例/参考/模型哈希及首次失败见 report.json。

- 855 个公开状态、20,073 个合法候选；13 级、10 类牌型、两副百搭、四座位和 31 个达到 512 token 的历史截断案例。
- NumPy/Node tokens 与 404 维特征完全一致，精确 argmax 855/855；最大 Q 误差 1.94312e-7。
- 新执行 PyTorch 2.14.0 / NumPy 2.5.2 CPU 核对，PT/NPZ 推理权重逐张量完全相同；最大 Q 误差 2.42196e-7，精确 argmax 855/855。两份报告使用同一案例及 NumPy 参考。parity-locked-report.json 的 torchRerun=false 只限定该 Node 程序，新的 Torch 证据另见 torch-report.json。
- 真实 worker 完成 8 次单局和 2 次多轮比赛；后者分别 2/4 轮，零超时、非法动作或兜底。每角色仅一个发牌组、对手 baseline，用于生命周期，不能评棋力。
- 原首次运行已完成出牌，但多轮记录落盘哈希失败。修复的是游戏通用 trace JSON 表示，旧原始结果保留，新冻结计划重新验证。未改模型动作或换种子追分。
- 实際 4 项 Node 回归（0 跳过）、17 项 Python unittest 和 22 项游戏评估回归通过。独立上游脚本函数不属于该 unittest 数字。CI 的 synthetic 权重另作实现回归，不冒充该模型。

原始数据、两次参考、PT/NPZ/Node 权重、旧失败与修复后原始 trace、日志和逐文件 inventory 保存在 ~/D/code/guandan_local_artifacts/train-01-a；Git 仅保存小型报告。Node 差分延迟 p50 4.68ms / p95 10.27ms 包含分类与断言，仅针对 d_model=16、1 block 小模型，不能推断正式网络或线上速度。

学习进还贡、严格训练恢复、预算停止、正式训练与独立棋力/真人验收仍待执行。
