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

输入状态来自独立 TS 发牌与合法动作，也含历史小手牌的边界夹具；Python `app_adapter` / `encode_decision` / `NumpyModel` 作独立参考。tokens 与特征要求完全一致，预测容差预设为 `2e-5 + 2e-4*abs(reference)`；精确 argmax 和数值近并列分别记录。这一步只证明 Python/Node 一致；PyTorch 必须另行执行以下检查，不根据历史导出报告冒充新验收。两类结果均不证明策略强度。

小模型可以验证真实 worker、单局和多轮生命周期；正式网络规模的速度和棋力需后续重新测量。CPU 实现保留清楚的模型边界，若大模型达不到预算，再更换推理后端并通过同一数值差分。


## PyTorch 原检查点核对

在本仓库 `.venv` 安装 `requirements.lock.txt`，使用 CPU 即可，不租 GPU：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python evaluation/verify_torch.py --checkpoint /absolute/original.pt --npz "$NPZ" --cases runs/eval-inference/cases.json --out runs/eval-inference/torch-report.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

逐张量要求 PT 与 NPZ 权重完全相同，逐候选核对 PyTorch 与 NumPy Q 值和精确 argmax。Node 差分使用同一环境生成的 NumPy 参考；两份报告共同构成三方验证。unittest 不执行 `test_all.py` 中的独立脚本函数，不能称为所有上游测试。

## 持续验证与真实结果

默认 GitHub Actions 使用 independently authored 的合成输入，覆盖 13 级、10 牌型与历史截断，执行编码/数值差分、全部 4 项 Node 回归和 6 项既有 Python 规则/比赛回归。合成输入只测试表示与推理，不保证穷举合法动作或模拟真实对局；默认 CI 不执行游戏 worker，也不替代本地 855 状态的真实权威规则验证。CI 通过 `make_fixture.py` 生成确定性的 **未训练** Transformer 权重，不依赖本机私人 checkpoint，也不把它当训练候选。

本地实际 smoke checkpoint 的新验证报告见 `evaluation/validation/`。它仅训练两轮，足以验证真实格式与生命周期，不能判断棋力，也没有加入游戏产品。首次失败与修复后运行分别保留；记录修复针对多轮轨迹 JSON 哈希，没有改变模型动作。

`run_validation.mjs` 的输出目录必须全新；不要覆盖失败或首次尝试。评估协议及其完整运行依赖由游戏 EVAL-03 冻结，游戏 PR #32 合并前请使用对应已构建功能分支。


游戏仓库为私有仓库，FableDan 的默认 GITHUB_TOKEN 无权跨仓库取代码。默认 CI 无须跨仓库凭据或复制案例，独立编写的 synthetic_cases.mjs 不读取私有游戏或其生成数据。完整游戏差分与 worker 请使用前面的本地命令；远程集成待另行配置授权，尚未执行。首次远程 CI 在私有仓库 checkout 失败，未执行测试，记录保留在 PR 历史。
