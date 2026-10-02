# Guandan_opencode 集成分支

本仓库是学习型机器人训练代码的主来源，分支 `codex/guandan-integration`。保留原上游 README、许可证和代码历史。对应游戏仓库为同级 `Guandan_opencode`，方案存于同级 `guandan_ai_design`，首期云预算 US$1,000。

## 已实现

- 实际花色手牌/出牌特征、公开花色历史、版本化检查点。
- 候选动作枚举不同自然牌、百搭牌消耗选择；同花色同点数双副牌副本允许合并。
- 与游戏一致的牌型优先级：不允许三带二带王、双百搭对子按级牌、歧义顺子取游戏认可的序列。
- 有限轮数 DMC 自对弈、NumPy 导出、JSONL 推理服务。
- 规则夹具来自游戏 TypeScript 分类器独立枚举小手牌所有子集，涵盖 13 个级别、双百搭、同花顺及炸弹。测试候选覆盖、牌型大小和跟牌集合。

## 本地验证

在本仓库根目录执行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m unittest discover -s tests -p 'test_guandan.py'
.venv/bin/python -m unittest discover -s tests -p 'test_rule_parity.py'
.venv/bin/python tests/test_all.py
.venv/bin/python smoke.py --out runs/smoke --rounds 2
```

原上游训练入口保留供后续改造，未完成预算硬停止、完整随机状态恢复和独立评估前，不应用其启动付费训练。

## 独立评估，不复制到游戏

游戏的旧 training 副本和专用接入已撤除。训练代码只在此仓库维护，不再执行历史 sync_fork.py。真实模型接入新的 EVAL-03 外部评估入口，见 [evaluation/README.md](evaluation/README.md)。是否正式集成取决于训练、独立评估和真人验收；当前默认 strategy-v1 保持。

历史夹具来自固定 TypeScript 规则生成器。原 training/scripts 路径属于已删除的历史原型；本轮的权威公开决策生成器位于 evaluation/generate_cases.mjs，可直接使用游戏 rules/dist。

## 仍未完成

扩大轮次状态转移覆盖与边界测试、学习进还贡、对手池、预算停止和严格恢复、独立配对棋力评估。小手牌集合通过不等于完整环境规则已经对齐；当前 smoke 模型仅验证技术链路，不代表强机器人。


## 轮次与比赛状态回归

新增 `fabledan/match.py` 管理两队等级、上局名次、下局进贡模式、首家及过 A。单局环境修正首家、双贡同点数顺序、结束时机与剩余名次顺序。训练脚本目前仍使用单局团队收益；比赛状态组件尚未切换生产训练目标。

独立 TypeScript 夹具验证：16 局、2,023 次逐步动作；208 组进还贡场景（78 组抗贡）；156 组升级与过 A 场景。回放逐步核对手牌、行动者、已完成状态及待压牌型，并核对最终名次。

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

夹具生成器在游戏仓库 `training/scripts/export_round_fixtures.mjs`、`export_tribute_fixtures.mjs` 和 `export_advancement_fixtures.mjs`，更新后把相应 JSON 复制回本仓库 `fixtures/`，无需同步到游戏。进还贡动作仍由固定策略选择；这些测试不代表模型已经学会进还贡，也不是全部游戏状态的穷举证明。
