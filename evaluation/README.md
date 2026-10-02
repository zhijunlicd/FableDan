# 真实模型的独立评估适配

本目录属于 FableDan，供游戏 EVAL-03 外部 model adapter 使用。游戏仓库不复制训练或推理代码；没有修改游戏默认，也不以 smoke 模型证明棋力。

## 推理与信息边界

`adapter.mjs` 提供 `createEvaluationBot` 生命周期；Node 直接运行 `FableDanModel`，无需 Python 常驻进程。模型只看本人手牌、公开动作及合法动作，原输入不包含其他手牌。候选顺序不改变；选最大实际 Q 值，近并列时保留首个。贡还牌阶段显式使用固定合法动作策略，模型尚未学习进还贡。

`encoding.mjs` 将权威 TypeScript 分类映射到 guandan-suits-v1（404 特征、103 tokens、512 历史上限）。包括实体花色、两副百搭、公开贡还牌、已完成座位；长历史截断保留原 Python 行为。`inference.mjs` 复现 NumPy 的 Float32 RMSNorm、RoPE、因果 attention、SwiGLU 和 Q/hand MLP。

权重从 `model_torch.export_npz` 的 NPZ 导出成 `*.node.json`，包含版本/形状/原 NPZ SHA-256；拒绝不兼容配置、非法维度、非有限输入/权重。模型 JSON 在 `modelFiles` 中冻结。权威 classifier 通过 `options.rulesEntry` 加载，完整 `options.rulesFiles` 哈希也在工厂启动核对；这些 JS 文件必须全部已出现在评估 manifest 的 runtimeFiles（run_validation.mjs 自动验证）。不接受任意未冻结外部分类器。

runtimeFiles 声明 evaluation/adapter.mjs、encoding.mjs、inference.mjs；源码需先提交。modelFiles 的路径相对 FableDan 根目录，不是其他仓库或临时目录。转换产物、决策数据、NPZ、检查点均保持在 Git 外。

## 本地验证

设置 Python 路径（NumPy 必须可用）和游戏路径（已安装/构建的 EVAL-03 提交）：

```sh
export PYTHON_BIN=/path/to/python-with-numpy
export GAME_ROOT=/absolute/Guandan_opencode-worktree
export NPZ=/absolute/trained/checkpoint.npz
mkdir -p runs/eval-inference
PYTHONPATH=. "$PYTHON_BIN" evaluation/export_model.py --source "$NPZ" --output runs/eval-inference/model.node.json
node evaluation/generate_cases.mjs "$GAME_ROOT" runs/eval-inference/cases.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. "$PYTHON_BIN" evaluation/reference.py --cases runs/eval-inference/cases.json --model "$NPZ" --out runs/eval-inference/reference.jsonl
node evaluation/verify_parity.mjs "$GAME_ROOT" runs/eval-inference/cases.json runs/eval-inference/reference.jsonl runs/eval-inference/model.node.json runs/eval-inference/parity-report.json
FABLEDAN_MODEL=runs/eval-inference/model.node.json node --test evaluation/adapter.test.mjs
node evaluation/run_validation.mjs "$GAME_ROOT" runs/eval-inference/model.node.json runs/eval-inference/worker
```

输入状态来自独立 TS 发牌与合法动作，也含历史小手牌的边界夹具；Python `app_adapter` / `encode_decision` / `NumpyModel` 作独立参考。tokens 与特征要求完全一致，预测容差预设为 `2e-5 + 2e-4*abs(reference)`；精确 argmax 和数值近并列分别记录。Python/Node 完成不代表本轮重新执行 PyTorch，更不代表策略强度。

小模型可以验证真实 worker、单局和多轮生命周期；正式网络规模的速度和棋力需后续重新测量。CPU 实现保留清楚的模型边界，若大模型达不到预算，再更换推理后端并通过同一数值差分。
