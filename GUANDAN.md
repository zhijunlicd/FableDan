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

## 同步到游戏

游戏仓库的 `training/` 是可独立运行的固定副本，不能在两处分别修改训练代码。改动本仓库后，在游戏根目录运行：

```sh
python3 training/scripts/sync_fork.py
python3 training/scripts/sync_fork.py --check
```

同步清单记录 fork 地址、基准提交、分支、工作区是否含未提交改动，以及每个文件的 SHA-256；未提交版本不能冒充已发布提交。同步不会执行提交或推送。

规则夹具由游戏仓库 `training/scripts/export_rule_fixtures.mjs` 生成，再复制到本仓库 `fixtures/rule-subsets.json`。原始判定来自 TypeScript，而不是 Python 生成器自身。

## 仍未完成

扩大轮次状态转移覆盖与边界测试、学习进还贡、对手池、预算停止和严格恢复、独立配对棋力评估。小手牌集合通过不等于完整环境规则已经对齐；当前 smoke 模型仅验证技术链路，不代表强机器人。


## 轮次与比赛状态回归

新增 `fabledan/match.py` 管理两队等级、上局名次、下局进贡模式、首家及过 A。单局环境修正首家、双贡同点数顺序、结束时机与剩余名次顺序。训练脚本目前仍使用单局团队收益；比赛状态组件尚未切换生产训练目标。

独立 TypeScript 夹具验证：16 局、2,023 次逐步动作；208 组进还贡场景（78 组抗贡）；156 组升级与过 A 场景。回放逐步核对手牌、行动者、已完成状态及待压牌型，并核对最终名次。

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

夹具生成器在游戏仓库 `training/scripts/export_round_fixtures.mjs`、`export_tribute_fixtures.mjs` 和 `export_advancement_fixtures.mjs`，更新后把相应 JSON 复制回本仓库 `fixtures/`，再同步游戏副本。进还贡动作仍由固定策略选择；这些测试不代表模型已经学会进还贡，也不是全部游戏状态的穷举证明。
