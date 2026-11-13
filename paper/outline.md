# Manuscript outline (PCL-WM, EAAI, v3 rebuild)

Working dir: `C:\Users\danta\Doubao\chats\2026-11-07\new-chat\paper`

## 依据

- 用户附件：`1-s2.0-S0952197625003471-main.pdf`（EAAI 148 (2025) 110347）在路由阶段登记为**版式基准（格式参考）**，不是事实来源：
  公式右侧连续编号、三线表、图注 Fig. n. 粗体 + 说明、单栏 3.3 in / 双栏 6.8 in。本轮该文件在本机已不可读，版式按第二轮已落地的同一套规则沿用。
- 数据来源：`exp/results_v2/*.json`（37 组）、`exp/data_eff.json`、`exp/mpc_D1.json`、`exp/digest.txt`。**正文所有数值必须与 digest 一致**，不得出现旧稿的 41.6–57.8%、β 0.96–1.02 等已废弃数值。
- 论证依据：`experience/latent-world-model-partial-obs-eval.md`（本机，已读）。

## 叙事（v2 结论，用户已确认）

主线从"辛推进带来线性误差增长"改为**结构先验的分解式评估**：

1. 潜空间世界模型在部分观测下可与直接模型竞争，但同分布上无结构优势。
2. 硬辛推进（固定步长 Störmer–Verlet）在离散学习型潜空间中**有害**：0.793→0.347（2.3×），β 1.33→0.99。
3. 真正有效的是 port-Hamiltonian 的 PSD 耗散 R=LLᵀ 与 gauge 固定；在 60 条轨迹 + 初始状态偏移下比无结构潜模型低约 15%（0.581 对 0.681）。
4. 闭环比开环放大差异，但 MLP 仍略优（0.0574 对 0.0680）。
5. HNN 无观测→状态映射，在部分观测下失效（D2 2.441、D9 4.582），可支撑"编码必要性"。

## 结构（英文，Elsevier 体例，目标 9000–12000 词）

| 节 | 内容 | 关键证据 |
|---|---|---|
| Highlights | 5 条 | 结论 1–3 |
| Abstract | 250 词 | 主表 + OOD |
| Nomenclature | 19 行 | 符号表 |
| 1 Introduction | 问题、缺口、贡献 3 条 | 文献 [1]–[22] |
| 2 Related work | 2.1 潜空间世界模型；2.2 结构化物理先验；2.3 结构保持积分器；2.4 学习模型 + MPC | 逐条对应真实文献 |
| 3 Method | 3.1 设定；3.2 潜空间分解；3.3 port-Hamiltonian 残余块；3.4 gauge；3.5 证书与误差分解（Eq. 13–16）；3.6 训练目标 | Eq. (1)–(18) |
| 4 Protocol | 4.1 四个工程域（Table 1）；4.2 部分观测与历史窗口；4.3 基线；4.4 指标；4.5 训练与硬件 | Table 1 |
| 5 Results | 5.1 长时程精度（Table 2, Fig. 2）；5.2 误差增长（Table 3, Fig. 3）；5.3 消融（Table 4, Fig. 4）；5.4 数据效率与分布外（Table 5, Fig. 5）；5.5 闭环 MPC（Table 6, Fig. 6）；5.6 能量一致性（Table 7）；5.7 成本（Table 8） | digest |
| 6 Discussion | 辛性为何失效（三条机理）；该保留什么；与文献关系；局限 | 5.3 + 5.4 |
| 7 Conclusion | 4 段 | — |
| 后置 | CRediT / Declaration / Funding / Acknowledgements / References (23 条，全部经检索核验) | — |

## 图表清单（全部由真实数据重算）

- Fig. 1 架构示意（重绘）
- Fig. 2 误差增长 D1/D2（`figs_v2/v2_err_growth.png`）
- Fig. 3 β 排序（`v2_beta.png`）
- Fig. 4 消融排序（`v2_ablation.png`）
- Fig. 5 数据效率/OOD（`v2_data_eff.png`）
- Fig. 6 闭环 MPC（新绘，MLP 0.0574 / PCL 0.0680）
- Table 1 域与传感器；Table 2 主精度；Table 3 β；Table 4 消融；Table 5 数据效率/OOD；Table 6 MPC；Table 7 能量偏差；Table 8 成本

## 格式（沿用第二轮 EAAI 版式）

A4，四边 2.5 cm；Times New Roman；正文 10.5 pt 两端对齐；H1 12 pt 粗体；公式段落左对齐 + 两个制表位（8.0 cm CENTER / 15.9 cm RIGHT），内容 `\t 公式图 \t (n)`；三线表（top/bottom 1 pt，表头下 0.5 pt，无竖线）；图注 9 pt 单倍行距 `Fig. n.` 粗体；参考文献悬挂 2 字符。
