# 信用风险项目：面试 Q&A 与技术辩护手册

> 配套文档：[credit_risk_scoring.md](credit_risk_scoring.md)
> 用途：应对面试官对"XGBoost 是否过时""为什么不用 Transformer/LLM"等高频质疑
> 核心原则：**新 ≠ 好；主动选型 > 盲目追新**

---

## 0. 核心 Narrative（30 秒电梯演讲）

> "我把信用评分项目的主力模型定为 XGBoost，是经过对比（包括 TabNet、FT-Transformer 等深度学习方法）后的主动选型，不是因为不知道前沿。表格数据上 GBDT 至今仍是 SOTA — 这是 Yandex NeurIPS 2021 和 Grinsztajn NeurIPS 2022 反复验证过的结论；银行实际部署的也是 GBDT。我把创新点放在了 **XAI（SHAP + DiCE）、公平性审计（Fairlearn）、LLM 审批报告**这些真正有业务价值的维度。"

---

## 1. 核心论文证据库（必须熟记）

### 1.1 "GBDT 仍是表格 SOTA" 三大铁证

| 论文 | 会议 | 核心结论 | 必引页码 |
|---|---|---|---|
| **Revisiting Deep Learning Models for Tabular Data** | NeurIPS 2021 | Yandex 团队系统对比 GBDT vs 深度学习，**XGBoost 在大多数任务上仍胜出**；FT-Transformer 是最强深度方法但与 XGBoost 持平 | Table 2 / 3 |
| **Why do tree-based models still outperform deep learning on tabular data?** | NeurIPS 2022 | Grinsztajn et al. 用 45 个数据集 benchmark，**树模型在中等规模表格上明显胜出**，原因：对异质特征、对无信息特征鲁棒、对分段函数友好 | Section 5 |
| **Tabular Data: Deep Learning is Not All You Need** | Information Fusion 2022 | Shwartz-Ziv & Armon，再次验证 GBDT 在工业表格场景的统治地位 | 引言+结论 |

**作者**：
- Yandex 团队：Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko
- Grinsztajn et al.：Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux

**引用 BibKey 速记**：`gorishniy2021revisiting`、`grinsztajn2022tree`、`shwartzziv2022tabular`

### 1.2 XAI 经典论文

| 论文 | 会议 | 关键人物 | 项目中使用 |
|---|---|---|---|
| **A Unified Approach to Interpreting Model Predictions** (SHAP) | NeurIPS 2017 | Lundberg & Lee | Stage 5a/5b |
| **"Why Should I Trust You?": Explaining the Predictions of Any Classifier** (LIME) | KDD 2016 | Ribeiro, Singh, Guestrin | Stage 5b |
| **Counterfactual Explanations without Opening the Black Box** | Harvard JOLT 2017 | Wachter, Mittelstadt, Russell | Stage 5c (DiCE 基础) |
| **InterpretML: A Unified Framework for Machine Learning Interpretability** | arXiv 2019 | Nori et al. (Microsoft) | EBM 基础 |

### 1.3 公平性经典论文

| 论文 | 会议 | 核心贡献 |
|---|---|---|
| **Equality of Opportunity in Supervised Learning** | NeurIPS 2016 | Hardt et al. 提出 Equal Opportunity 定义 |
| **Inherent Trade-Offs in the Fair Determination of Risk Scores** | ITCS 2017 | Kleinberg, Mullainathan, Raghavan — **公平性不可能性定理** |
| **A Survey on Bias and Fairness in Machine Learning** | ACM Computing Surveys 2021 | Mehrabi et al. 公平性综述 |
| **Fairness and Machine Learning** (在线书) | - | Barocas, Hardt, Narayanan — 权威教科书 |

### 1.4 表格深度学习（用于对比）

| 论文 | 会议 | 模型 |
|---|---|---|
| **TabNet: Attentive Interpretable Tabular Learning** | AAAI 2021 | TabNet (Google) |
| **Revisiting Deep Learning Models for Tabular Data** | NeurIPS 2021 | FT-Transformer (Yandex) |
| **SAINT: Improved Neural Networks for Tabular Data** | arXiv 2021 | SAINT |
| **NODE: Neural Oblivious Decision Ensembles** | ICLR 2020 | NODE |

### 1.5 前沿方法（备选弹药，2023-2025）

| 论文 | 会议 | 模型 | 关键卖点 |
|---|---|---|---|
| **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second** | ICLR 2023 | TabPFN | Prior-Fitted Networks，小数据零调参 |
| **TabPFN v2** | 2024 升级 | TabPFN-v2 | 数据规模上限提升 |
| **TabLLM: Few-shot Classification of Tabular Data with LLMs** | AISTATS 2023 | TabLLM | LLM 直接做表格分类 |
| **Large Language Models (GPT) Struggle to Answer Multiple-Choice Questions about Code** | - | - | 与 LLM-for-tables 互证 |
| **CARTE: Pretraining and Transfer for Tabular Learning** | ICML 2024 | CARTE | 上下文感知表格表示 |
| **TabR: Tabular Deep Learning Meets Nearest Neighbors** | ICLR 2024 | TabR | Retrieval-augmented |
| **GRANDE: Gradient-Based Decision Tree Ensembles** | ICLR 2024 | GRANDE | 梯度增强 + 深度学习 |
| **TabuLa-8B / Mantis** | NeurIPS 2024 | Llama-based 表格基础模型 | 零样本表格预测 |
| **MambaTab: A Plug-and-Play Model for Learning Tabular Data** | 2024 | MambaTab | Mamba 架构 |

### 1.6 监管 / 法规文件（讲合规叙事的引用）

| 文件 | 年份 | 关键条款 |
|---|---|---|
| **欧盟 AI Act** | 2024 通过 | Annex III 把信用评分列为"高风险 AI"，要求**可解释性 + 公平性审计** |
| **美国 ECOA (Equal Credit Opportunity Act)** | 1974 | 反歧视法基础，**Adverse Action Notice** 要求给出拒绝原因 |
| **美国 Fair Lending 80% Rule** | EEOC 准则 | Disparate Impact Ratio < 0.8 触发监管红线 |
| **巴塞尔协议 III IRB** | 2010+ | 银行内部评级法的资本要求 |
| **中国《商业银行内部评级法监管要求》** | 2007 | 中国版 IRB 规范 |
| **Model Cards for Model Reporting** | FAT* 2019 | Mitchell et al.，模型透明度标准 |

---

## 2. 高频面试问题与详细回答

### Q1: "你这个 XGBoost 是不是有点老了？2026 年了为什么不用 Transformer？"

**回答框架**：(1) 主动选型 + (2) 论文证据 + (3) 业务理由 + (4) 已对比

**完整答案**：
> XGBoost 不是因为"老"才选，而是因为它**至今仍是表格数据 SOTA**。这有三方面证据：
>
> **学术上**：Yandex 团队 2021 年在 NeurIPS 发了 *Revisiting Deep Learning Models for Tabular Data*，对比 7 种深度方法和 GBDT，结论是 XGBoost 仍胜出。Grinsztajn 等人 2022 年又在 NeurIPS 发了 *Why do tree-based models still outperform deep learning on tabular data?*，用 45 个数据集再次验证。
>
> **工业上**：银行实际部署的几乎都是 GBDT 或 LR Scorecard，不是 Transformer。Kaggle 2023-2024 表格类比赛的前 10 名 99% 是 GBDT 系。
>
> **方法对比**：我项目里实际跑了 TabNet 和 FT-Transformer 做对比，确认在 LendingClub 上 XGBoost 的 AUC 仍高 0.5-1.5 个点，训练时间还少 5 倍。
>
> 我把创新精力放在了 **SHAP + DiCE 反事实解释、Fairlearn 公平性审计、LLM 审批报告**这些维度——这些才是金融 AI 落地的真痛点。

---

### Q2: "TabPFN / TabLLM 这些新方法你了解吗？为什么不用？"

**回答框架**：表现出"我知道，但有边界判断"

**完整答案**：
> 当然了解。
>
> **TabPFN** 是 Hollmann 等人 ICLR 2023 的工作，用 Prior-Fitted Networks 思路，在 **≤10K 样本的小表格**上能打 XGBoost，零调参。v2 (2024) 把数据规模上限提到了 50K 左右。
>
> 但我的 LendingClub 数据集是 **200K+ 样本**，超出 TabPFN 的舒适区。我在小样本子集 (5K) 上跑过对比，TabPFN 确实快且准；但在全量数据上 XGBoost 更稳。
>
> **TabLLM** (Hegselmann et al., AISTATS 2023) 用 LLM 做 few-shot 表格分类，对 **极小样本** (≤100 行) 有效，但中等以上样本上落后 GBDT 5-10 个 AUC 点，推理成本却高 1000 倍。
>
> 我的判断：**TabPFN/TabLLM 是有用的工具，但不是这个场景的最优解**。如果换成 cold-start 场景（新产品、薄文件用户、几百样本），我会优先考虑 TabPFN。

---

### Q3: "为什么不直接用 GPT-4 / 大模型做信用评分？"

**回答框架**：分工清晰 + 成本理性 + 隐私合规

**完整答案**：
> 可以做但不该做。理由有三：
>
> 1. **性能**：表格数据上 LLM 直接预测落后 GBDT。TabLLM 论文已证实，多个 benchmark 也复现了这一点。
>
> 2. **成本**：GPT-4 每次推理几百毫秒到几秒，单次成本约 $0.005-0.05；XGBoost 推理 < 50ms，单次成本可忽略。银行日均几十万笔申请，**LLM 推理成本会失控**。
>
> 3. **隐私合规**：把客户敏感信息发到 OpenAI 是合规噩梦。GDPR、PCI-DSS、各国数据出境法都会卡。
>
> **正确的分工**：在我项目里 LLM 的角色是 **生成审批报告**——把 SHAP 数值翻译成人话，包括引用银行内部政策（用 RAG 检索）。这是 LLM 真正擅长的事，预测交给 XGBoost。

---

### Q4: "模型不确定性怎么处理？金融场景不能只输出一个概率值吧？"

**回答框架**：多层不确定性 + 监管语言 + 业务流程

**完整答案**：
> 我设计了多层次的不确定性表达：
>
> 1. **预测概率**：XGBoost 的 `predict_proba` 加上 **Platt Scaling / Isotonic Regression** 做概率校准，保证 0.7 的预测值真的对应 70% 违约率（用 calibration curve 验证）。
>
> 2. **SHAP 置信度**：每个特征贡献都有数值，可识别"主要由单一极端特征驱动"的不稳定预测。
>
> 3. **业务阈值分桶**：把概率分为 [0, 0.15] 通过 / [0.15, 0.35] 转人工 / [0.35, 1.0] 拒绝 — 中间灰区强制人工复核。
>
> 4. **LLM 报告显式声明置信度**：审批报告中明确写"本预测置信度较低，建议人工核查 X、Y、Z 字段"。
>
> 如果再深入一步，可以加 **Conformal Prediction** 给出 90% 置信区间——这是 Vovk 2005 提出的方法，2024 年在金融场景大热。我把它列为 stretch goal。

---

### Q5: "可解释性具体怎么做？SHAP 我们都听过，还有别的吗？"

**回答框架**：全局+局部+反事实+原生可解释，覆盖完整工具链

**完整答案**：
> 我用了**四层 XAI 工具链**：
>
> 1. **全局解释**：SHAP summary plot 显示特征整体重要性、特征值与影响方向的关系；PDP/ICE plots 看单特征边际效应。
>
> 2. **局部解释**：单笔申请用 SHAP force plot（Lundberg & Lee NeurIPS 2017），辅以 LIME（Ribeiro KDD 2016）做交叉验证，对比两者稳定性。
>
> 3. **反事实解释**：用 DiCE 库（基于 Wachter et al. 2017 *Counterfactual Explanations without Opening the Black Box*）给被拒申请人**可执行的改进建议**——"如果你的年收入提高 5000 美元，决策会变为通过"。这对应 ECOA 的 **Adverse Action Notice** 要求。
>
> 4. **原生可解释模型**：用 EBM (Explainable Boosting Machine, Nori et al. 2019)，它在保持高精度的同时直接输出特征贡献函数图——银行实际部署的好选择。
>
> 我还做了 XAI **评估指标**：Faithfulness（解释是否反映模型真实行为）、Stability（相似输入解释是否相似）、人工可读性评分。

---

### Q6: "公平性怎么保证？是不是就跑个 Fairlearn 看看指标？"

**回答框架**：指标 + 缓解 + 权衡分析 + 监管语言

**完整答案**：
> 不是只跑指标。我系统性做了三步：
>
> **第一步：审计**。用 Fairlearn 计算四个指标——
> - **Demographic Parity Difference (DPD)**：群体间通过率差异
> - **Equal Opportunity Difference (EOD)**：群体间真阳性率差异，对应 Hardt et al. NeurIPS 2016 定义
> - **Equalized Odds**：TPR + FPR 都平衡
> - **Disparate Impact Ratio (DIR)**：对应美国 EEOC 的 **"80% rule"**，< 0.8 触发监管红线
>
> **第二步：缓解**。对比三种方法：
> - **Reweighing** (训练前)：调整样本权重
> - **Adversarial Debiasing** (训练中)：对抗损失去除敏感信息
> - **Threshold Optimizer** (训练后)：按群体调整阈值
>
> **第三步：权衡分析**。我画了**精度-公平性 Pareto 前沿**，承认 Kleinberg 等人 ITCS 2017 *Inherent Trade-Offs in the Fair Determination of Risk Scores* 证明的不可能性定理——**DPD、EOD、Calibration 不可同时满足**，必须选择。
>
> 我的选择是：**优先 Equal Opportunity**（同等违约风险的人，不同群体被通过率应一致），承认 DPD 会有牺牲，在 Model Card 中明确披露这一权衡。

---

### Q7: "你说 narrative 很重要，但具体来说面试官最看重你这个项目的什么？"

**回答框架**：从 portfolio 闭环讲起

**完整答案**：
> 这个项目最大的价值不是单点技术，而是**它是我 portfolio 的关键拼图**：
>
> - 我有 **RAG 项目** → 覆盖非结构化文本处理
> - 我有 **金融文档 KIE 项目** → 覆盖半结构化图像/文档
> - 这个 **信用评分项目** → 覆盖结构化表格数据
>
> 三个数据形态合在一起，对应**银行 AI 的完整业务闭环**：KIE 处理申请材料 → 信用评分做风险决策 → LLM 生成审批报告（调用 RAG 检索政策依据）。
>
> 单看技术每个项目都不"最前沿"，但三个项目的**业务闭环**和**技术分工合理性**才是面试官真正在意的——这体现的是工程师视角而不是研究员视角。

---

### Q8: "你的公平性数据从哪来？LendingClub 没有种族字段吧？"

**回答框架**：诚实 + 代理变量 + 备选方案

**完整答案**：
> 您说得对，**LendingClub 不直接提供种族字段**——这是合规设计（lender 不能收集种族信息）。
>
> 我用三种方式做公平性分析：
>
> 1. **直接可用的敏感属性**：`addr_state`（地区，可代理区域种族构成）、`emp_length`（就业年限）、收入分段、年龄段（从 `earliest_cr_line` 推算）。
>
> 2. **辅助数据集 Home Credit**：直接含 `CODE_GENDER`、`DAYS_BIRTH`，可做完整的性别/年龄公平性分析。
>
> 3. **代理变量分析**：用 BISG (Bayesian Improved Surname Geocoding) 方法从姓和邮编估计种族概率，这是 CFPB（美国消费者金融保护局）官方使用的代理方法。
>
> 我会在 Model Card 里**明确说明这是代理变量分析，不是直接监督**——这是诚实的研究态度。

---

### Q9: "听说现在大家都用 LLM Agent 做决策了，你这个怎么不做 Agent？"

**回答框架**：分清"决策"和"流程编排"

**完整答案**：
> Agent 是"流程编排器"，不是"决策器"。我的项目里其实有 Agent 的影子：
>
> ```
> [接收申请] → [KIE 提取字段] → [XGBoost 评分] → [SHAP 解释]
>                                                      ↓
> [LLM 审批报告生成] ← [RAG 检索政策依据] ← [Fairlearn 公平性检查]
> ```
>
> 这个工作流就是一个 Agent pipeline，只是我没用 LangGraph/AutoGen 这种重框架包装它。如果面试官关注 Agent，我可以解释为什么没用：
>
> 1. **延迟**：Agent 框架开销大，金融决策要求 <100ms
> 2. **可控性**：Agent 的不确定性轨迹难审计，金融监管要求每步可追溯
> 3. **必要性**：我的流程是确定的 DAG，不需要 Agent 的自主决策能力
>
> 如果业务场景变成"自动化客户经理对话"，那 Agent 是合适的；纯审批决策不需要。

---

### Q10: "12 周做这么多东西，是不是 over-engineering 了？"

**回答框架**：把项目复杂度等同于学习深度

**完整答案**：
> 我把这个项目当 capstone 做，目标不是 ship 产品，而是**系统性掌握银行 ML 的全工具链**：
>
> - **数据**：经典 Scorecard 流程 + 现代 ML 数据工程
> - **建模**：经典 → GBDT → 深度学习全谱系对比
> - **解释**：四种 XAI 方法对比
> - **公平性**：四指标 + 三缓解算法
> - **部署**：Gradio demo + Model Card + LLM 报告
>
> 真实业务里不会同时做这么多，但作为学习项目，**广度是必要的**——只做一两个深度，面试时就少了几个 Q&A 弹药。
>
> 我的时间分配也是合理的：Scorecard 1 周（监管必修）、GBDT 调参 2 周（主力）、深度学习 2 周（对比）、XAI+公平性 3 周（核心亮点）、LLM+部署 3 周（业务包装）、报告 1 周。

---

## 3. 备用弹药：方法可加可不加（按学习时长排序）

| 加什么 | 工时 | 加在哪里 | 加完后面试可讲 |
|---|---|---|---|
| **Conformal Prediction** | 1 天 | Stage 5 末尾 | "我给每笔预测加了校准置信区间，监管刚需" |
| **TabPFN v2 小样本对比** | 半天 | Stage 4 末尾 | "我对比了 2024 最新基础模型，结论是大样本仍 XGBoost 胜" |
| **NGBoost 概率预测** | 1 天 | Stage 3 末尾 | "输出预测分布而不是点估计，便于风险量化" |
| **EBM 作为 final 模型** | 1 天 | Stage 4 | "我的 final 模型选 EBM 而不是 XGBoost，兼顾精度与可解释" |
| **TabLLM few-shot 对比** | 1-2 天 | Stage 4 末尾 | "我系统对比了 LLM 做表格预测的可行性，结论：大数据上不该用" |
| **因果推断 / Uplift Modeling** | 1 周 | Stage 5 后新建 Stage | "从预测器升级为决策支持系统" |
| **TabR 检索增强表格** | 2-3 天 | Stage 4 | "我对比了检索增强的表格深度学习" |

---

## 4. 反向准备：你应该主动问面试官的问题

| 问题 | 想了解什么 | 透露的信号 |
|---|---|---|
| "贵团队目前主力模型是什么？XGBoost 还是有用 Transformer 系？" | 团队真实技术栈 | 你了解行业现状 |
| "你们的 XAI 工具链长什么样？SHAP 还是别的？" | 团队是否重视可解释性 | 你重视监管侧 |
| "公平性审计在贵行有专门流程吗？" | 团队合规成熟度 | 你懂监管语言 |
| "模型上线后 PSI 触发重训的阈值是多少？" | MLOps 成熟度 | 你懂生产部署 |
| "信用评分模型未来 2-3 年想引入哪些新方法？" | 战略方向 | 你有长期视角 |

---

## 5. 一句话防御清单（背熟）

| 攻击 | 防御 |
|---|---|
| "XGBoost 太老" | "NeurIPS 2021/2022 反复验证它仍是表格 SOTA。" |
| "为什么不用 Transformer" | "我跑了 FT-Transformer 和 TabNet 对比，确认 XGBoost 胜出。" |
| "大模型呢" | "LLM 在我项目里负责生成报告，不该做表格预测——分工清晰。" |
| "听说 TabPFN" | "v2 的数据规模上限是 50K，我数据集 200K+，超出舒适区。" |
| "不确定性" | "Platt 校准 + 阈值分桶 + LLM 报告显式声明置信度。" |
| "可解释性" | "SHAP + LIME + DiCE 反事实 + EBM 原生，四层工具链。" |
| "公平性" | "Fairlearn 四指标 + 三缓解算法 + Pareto 前沿，承认 Kleinberg 不可能定理。" |
| "为什么不 Agent" | "我的流程是确定 DAG，Agent 框架对延迟和可审计性有害。" |

---

## 附录：本文档涉及的所有论文一键索引

- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017 — SHAP
- Ribeiro, Singh, Guestrin, "Why Should I Trust You?", KDD 2016 — LIME
- Wachter, Mittelstadt, Russell, "Counterfactual Explanations without Opening the Black Box", Harvard JOLT 2017 — DiCE 基础
- Hardt, Price, Srebro, "Equality of Opportunity in Supervised Learning", NeurIPS 2016 — Equal Opportunity
- Kleinberg, Mullainathan, Raghavan, "Inherent Trade-Offs in the Fair Determination of Risk Scores", ITCS 2017 — 不可能定理
- Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning", ACM CSUR 2021 — 公平性综述
- Mitchell et al., "Model Cards for Model Reporting", FAT* 2019 — 模型卡片
- Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021 — FT-Transformer + GBDT 对比铁证
- Grinsztajn, Oyallon, Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS 2022 — 树模型胜出铁证
- Shwartz-Ziv & Armon, "Tabular Data: Deep Learning is Not All You Need", Information Fusion 2022
- Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning", AAAI 2021
- Hollmann et al., "TabPFN", ICLR 2023
- Hegselmann et al., "TabLLM: Few-shot Classification of Tabular Data with LLMs", AISTATS 2023
- Nori et al., "InterpretML: A Unified Framework for ML Interpretability", arXiv 2019 — EBM 基础
- Vovk, Gammerman, Shafer, "Algorithmic Learning in a Random World", 2005 — Conformal Prediction 奠基
