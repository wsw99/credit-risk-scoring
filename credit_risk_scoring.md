# 项目方案：可解释 + 公平的个人消费贷款违约预测系统

> 项目类型：深度探索项目（10-12 周）
> 硬件：AWS T4 GPU（16GB 显存）
> 行业对标：银行零售信贷、互联网信贷、消费金融公司
> 核心能力：表格 ML 建模 + **可解释 AI (XAI)** + **算法公平性** + **LLM 审批报告生成**
>
> Portfolio 互补叙事：RAG (文本) + KIE (图像/文档) + **本项目 (结构化表格数据)**，覆盖银行 AI 的三种数据形态。

---

## 1. 项目概述

**项目目标**：构建一个端到端的个人消费贷款违约预测系统，不只追求 AUC 指标，更强调：
1. **可解释性**：每个决策都能讲清楚为什么（监管要求）
2. **公平性**：检测并缓解性别、年龄等敏感属性上的偏差
3. **LLM 自动化报告**：把 SHAP 解释转成人话审批结论（与现有 RAG 项目联动）

**典型业务场景**：
- 银行个人消费贷款审批
- 信用卡分期违约预警
- 互联网消费金融（蚂蚁、京东、消费金融公司）
- 巴塞尔协议 IRB 模型搭建参考

**输入输出**：
- 输入：申请人结构化特征（年收入、债务比、信贷历史、贷款用途、雇佣信息...）
- 输出：
```json
{
  "decision": "REJECT",
  "default_probability": 0.34,
  "credit_score": 580,
  "key_drivers": [
    {"feature": "dti", "value": 32.5, "impact": "+0.18", "direction": "increases risk"},
    {"feature": "annual_inc", "value": 35000, "impact": "+0.12", "direction": "below threshold"},
    {"feature": "fico_score", "value": 620, "impact": "+0.08", "direction": "marginal"}
  ],
  "fairness_check": {"passed": true, "audit_id": "..."},
  "llm_report": "申请人 DTI 偏高（32.5%），主要由于...建议先优化债务结构..."
}
```

**最终交付物**：完整训练/推理代码、模型对比实验表、XAI 分析报告、公平性审计报告、Gradio 审批 demo（含 LLM 报告生成）、技术报告、5 分钟演示视频。

---

## 2. 相关项目和参考

### 2.1 顶校课程 / 作业参考

| 来源 | 内容 | 可学习点 |
|---|---|---|
| **Stanford CS229 (ML)** | 经典分类模型、特征工程 | 项目报告结构 |
| **Stanford STATS 315A/B** | 现代回归与分类（Hastie/Tibshirani） | LR + Logistic 严谨建模 |
| **Berkeley INDENG 290 / DATA 100** | 数据科学项目模板 | 完整 pipeline 范式 |
| **CMU 10-708 PGM** | 概率图模型，可用于不确定性建模 | 贝叶斯方法补充 |
| **Stanford CS329T (Trustworthy ML)** ⭐ | XAI / Fairness 系统课程 | **本项目核心方法论** |
| **MIT 6.S898 Deep Learning** | 表格深度学习专题 | TabNet/FT-Transformer 参考 |
| **Harvard / Stanford "AI Ethics" 课程** | 公平性、偏见、监管 | 公平性 narrative |

### 2.2 开源项目参考

| 项目 | 说明 | 复用度 |
|---|---|---|
| **[scikit-learn](https://github.com/scikit-learn/scikit-learn)** ⭐⭐⭐⭐⭐ | LR、特征处理、评测 | 全程使用 |
| **[XGBoost / LightGBM / CatBoost](https://github.com/microsoft/LightGBM)** ⭐⭐⭐⭐⭐ | 主流树模型 | 主力模型 |
| **[feature-engine](https://github.com/feature-engine/feature_engine)** ⭐⭐⭐⭐ | 银行级特征工程 | WOE/IV 等监管标准 |
| **[optbinning](https://github.com/guillermo-navas-palencia/optbinning)** ⭐⭐⭐⭐⭐ | **专为信用评分设计**，WOE 分箱、Scorecard 自动化 | Stage 2 核心 |
| **[SHAP](https://github.com/shap/shap)** ⭐⭐⭐⭐⭐ | Shapley value 解释 | XAI 主力 |
| **[LIME](https://github.com/marcotcr/lime)** ⭐⭐⭐⭐ | 局部解释 | XAI 对比 |
| **[DiCE](https://github.com/interpretml/DiCE)** ⭐⭐⭐⭐ | 反事实解释 | 给出"为什么被拒+怎么做能通过" |
| **[InterpretML](https://github.com/interpretml/interpret)** ⭐⭐⭐⭐ | EBM 可解释树模型 | 同时高精度+高可解释 |
| **[Fairlearn (微软)](https://github.com/fairlearn/fairlearn)** ⭐⭐⭐⭐⭐ | 公平性评测 + 缓解算法 | **公平性核心** |
| **[AIF360 (IBM)](https://github.com/Trusted-AI/AIF360)** ⭐⭐⭐⭐ | 公平性工具箱 | 公平性对比 |
| **[pytorch-tabular](https://github.com/manujosephv/pytorch_tabular)** ⭐⭐⭐⭐ | TabNet/FT-Transformer/NODE 集成 | Stage 4 深度学习 |
| **[Kaggle Home Credit 顶级 solutions](https://www.kaggle.com/c/home-credit-default-risk/discussion)** ⭐⭐⭐⭐⭐ | 第 1-10 名公开方案 | 工程经验源 |
| **[LangChain / LlamaIndex](https://github.com/langchain-ai/langchain)** ⭐⭐⭐ | LLM 报告生成 | Stage 6 模块 |

### 2.3 论文与基准

**信用评分经典**：
- Hand & Henley (1997) — 银行信用评分综述
- Thomas et al. (2002) — Credit Scoring and Its Applications 标准教材
- Lessmann et al. (2015) — 主流 ML 信用模型 benchmark

**树模型**：
- XGBoost (KDD 2016)
- LightGBM (NeurIPS 2017)
- CatBoost (NeurIPS 2018)

**表格深度学习**：
- TabNet (AAAI 2021)
- FT-Transformer (NeurIPS 2021)
- NODE (ICLR 2020)
- SAINT (2021)
- **"Revisiting Deep Learning Models for Tabular Data"** (NeurIPS 2021) — Yandex，结论：树模型仍最强

**XAI**：
- SHAP (Lundberg & Lee, NeurIPS 2017)
- LIME (Ribeiro et al., KDD 2016)
- DiCE / Counterfactual Explanations (Wachter et al. 2017)
- "A Unified Approach to Interpreting Model Predictions" — SHAP 论文必读

**公平性**：
- Hardt et al. (NeurIPS 2016) — Equal Opportunity
- Barocas et al. — "Fairness and Machine Learning" 在线书
- Mehrabi et al. (ACM Computing Surveys 2021) — Bias 综述

**金融 ML 监管文献**：
- 巴塞尔协议 III IRB 文档
- 欧盟 AI Act（2024 通过）— 信用评分被列为高风险 AI
- 美国 ECOA (Equal Credit Opportunity Act) — 反歧视法
- 中国《商业银行内部评级法监管要求》

### 2.4 数据集

| 数据集 | 规模 | 标注 | 推荐用途 |
|---|---|---|---|
| **LendingClub Loan Data** ⭐⭐⭐⭐⭐ | 2007-2018 约 220万笔贷款 | 违约标签 + 150+ 特征 + 发放时间 | **主力数据集**，规模大、时间维度全 |
| **Kaggle Home Credit Default Risk** ⭐⭐⭐⭐⭐ | 30万 + 申请人，7 张关联表 | 违约标签 | 多表关联 + 大规模特征工程练习 |
| **Give Me Some Credit (Kaggle)** ⭐⭐⭐ | 15万 笔贷款 | 90 天逾期 | 入门 + 快速实验 |
| **German Credit Dataset (UCI)** ⭐⭐⭐ | 1000 笔贷款 | 违约标签 | 经典教学，太小，仅做快速 sanity check |
| **Taiwan Credit Card Default (UCI)** ⭐⭐⭐ | 3万信用卡客户 | 下月违约 | 信用卡场景补充 |
| **PKDD'99 Financial Dataset** ⭐⭐ | 8张表的银行数据 | 贷款状态 | 关联建模练习 |
| **FICO Explainable ML Challenge** ⭐⭐⭐⭐ | HELOC 数据（脱敏） | 违约标签 | **XAI 基准数据集** |

**关键选择**：
- **主力训练集**：LendingClub（规模大、时间维度长，适合做时间漂移分析）
- **进阶训练集**：Home Credit（多表关联、特征工程复杂度高）
- **XAI 专项**：FICO HELOC（业内 XAI 基准）
- **快速验证**：Give Me Some Credit

**敏感属性获取（公平性分析必需）**：
- LendingClub 含 `addr_state`（地区，可代理种族构成）、`emp_length`
- 性别需用 `emp_title` 文本推断（合规风险，仅用于偏差检测）
- Home Credit 含 `CODE_GENDER`、`DAYS_BIRTH` 直接可用

---

## 3. 可行性评估

### 3.1 技术要求

- **必备**：Python ≥ 3.10、pandas、numpy、scikit-learn、XGBoost/LightGBM
- **银行特征工程**：optbinning、feature-engine、category_encoders
- **深度学习**：PyTorch ≥ 2.0、pytorch-tabular
- **XAI**：shap、lime、dice-ml、interpret
- **公平性**：fairlearn、aif360
- **LLM**：transformers (本地) + openai / anthropic API
- **部署**：Gradio + HuggingFace Spaces / Streamlit
- **实验跟踪**：MLflow 或 Weights & Biases

### 3.2 数据需求

- **下载量**：LendingClub ~2GB、Home Credit ~700MB、其他 < 100MB
- **存储**：建议 50GB EBS（够用）
- **清洗工作**：LendingClub 缺失值很多，需仔细处理；Home Credit 7 表关联需聚合特征
- **特征工程**：手工特征（WOE 编码、聚合特征、比率特征）→ 自动特征（featuretools）

### 3.3 模型路线（详见第 4 节）

| 路线 | 是否作 baseline | 是否作 final |
|---|---|---|
| 银行 Scorecard (LR + WOE) | ✅ | ✅（监管对标） |
| Logistic Regression | ✅ | baseline |
| Random Forest | ✅ | baseline |
| **XGBoost / LightGBM / CatBoost** | ✅ | ✅（主推荐） |
| Stacking / Blending | ❌ | ✅（性能上限） |
| TabNet | ❌ | ✅（深度学习对比） |
| FT-Transformer | ❌ | ✅（深度学习对比） |
| EBM (InterpretML) | ❌ | ✅（高精度+高可解释） |

### 3.4 算力评估（AWS T4，16GB 显存）

| 任务 | 估计时间 | 备注 |
|---|---|---|
| 数据清洗 + EDA | CPU 4-8 小时 | 一次性 |
| Scorecard (LR + WOE) | CPU 30 分钟 | T4 不需要 |
| XGBoost/LightGBM 全数据集 | CPU/GPU 1-3 小时 | GPU 加速 5-10x |
| 超参搜索 (Optuna 100 trials) | T4 GPU 8-16 小时 | 主要算力消耗 |
| TabNet 训练 | T4 4-8 小时 | batch=2048 |
| FT-Transformer 训练 | T4 6-12 小时 | batch=512 |
| SHAP 全数据集计算 | CPU 1-4 小时 | 用 TreeSHAP 加速 |
| LLM 报告生成 (本地 7B) | T4 INT4 ~2s/份报告 | 或用 API |

**T4 总成本**：
- GPU 时间：60-120 小时（树模型 + 深度学习 + 调参）
- AWS T4 Spot ~$0.16/h；On-Demand ~$0.53/h
- **预算：Spot $10-20；On-Demand $30-60**（比 OCR 项目便宜！）
- LLM API（如用 GPT-4o-mini 生成报告）：500 份 × $0.005 = $2.5

### 3.5 时间成本（10-12 周）

总投入：每周 15-20 小时，**总计 150-240 小时**。

### 3.6 风险与备选

| 风险 | 应对 |
|---|---|
| LendingClub 数据下载源不稳定 | Kaggle 镜像、archive.org 备份 |
| 数据质量差影响所有模型 | 严格 EDA + 数据卡片记录决策 |
| 树模型已经接近上限，深度学习提升小 | 这本身就是结论；正确诚实报告即可 |
| 公平性指标互相冲突 | 选 2-3 个主要指标，承认权衡 |
| LLM 报告幻觉 | 用 RAG 锚定到 SHAP 数值，加置信度过滤 |
| Home Credit 多表关联工作量大 | 先在 LendingClub 完成主线，Home Credit 作扩展 |

---

## 4. 深度探索路线（6 阶段）

### Stage 1：数据准备 + EDA + 评测协议（Week 1-2）

**目标**：建立稳定的数据 pipeline 与评测框架。

**任务**：
- 下载 LendingClub 2007-2018 全量数据
- 数据清洗：
  - 标签定义（"Charged Off" / "Default" = 违约 1；"Fully Paid" = 0；进行中样本剔除）
  - 缺失值策略（缺失率 > 80% 字段删除；20-80% 单独建标志位）
  - **数据泄漏防范**（强制移除发放后才能观察到的特征：还款记录、回收金额等）
- 时间分割：
  - 训练集：2014-2016
  - 验证集：2017
  - 测试集：2018（**严格 out-of-time**，模拟真实部署）
- 实现统一评测脚本：
  - **AUC** (核心指标)
  - **KS 统计量** (银行业标准指标)
  - **Precision/Recall @ 多个阈值**
  - **Gini 系数** (= 2·AUC - 1)
  - **业务指标**：在固定通过率下的违约率、坏账损失
  - **校准曲线**：预测概率与实际违约率的偏差

**可调参数**：标签定义边界、特征筛选阈值

**复用关系**：所有后续 stage 共用这套数据/评测

**评估**：用一个简单 LR 跑通端到端，确认评测脚本正确

### Stage 2：经典银行 Scorecard（Week 2-3）

**目标**：建立"监管对标"的可解释 baseline。

**这一步与所有学生项目都不同——这是银行真实在用的方法。**

**方法**：
- **WOE 编码 (Weight of Evidence)**：用 optbinning 对每个特征自动分箱
- **IV 筛选 (Information Value)**：IV > 0.1 的特征保留
- **逐步回归** + 多重共线性检查 (VIF)
- **Scorecard 转换**：把 LR 系数转成 300-850 分制
- **PSI 监控 (Population Stability Index)**：检测时间漂移

**关键技术**：
- 单调性约束（更高收入 → 更低违约概率，符合业务常识）
- 业务规则审核（每个特征的 WOE 单调性需要业务可解释）

**评估**：
- LendingClub test set AUC、KS
- 业务可解释性：每条规则能写成"如果 X 且 Y，则评分 +Z 分"
- PSI < 0.1 视为稳定，0.1-0.25 警告，> 0.25 重训

**复用关系**：Scorecard 是 monitoring 阶段的标杆；XAI 解释能与 Scorecard 系数对照验证

### Stage 3：树模型 + 超参优化（Week 3-5）

**目标**：建立性能上限的强基线，对比深度学习是否值得。

**子阶段**：

**3a. 基础树模型**
- Random Forest（baseline）
- XGBoost
- LightGBM
- CatBoost
- 默认参数 vs 调参对比

**3b. 超参优化**
- Optuna 100-200 trials
- 关键参数：learning_rate、max_depth、num_leaves、min_child_samples、subsample
- 早停 + 5-fold cross-validation

**3c. 集成**
- Stacking：LR/XGBoost/LightGBM/CatBoost → Meta-LR
- Blending：加权平均
- 对比单模型与集成的提升

**3d. 类别不平衡处理**
- 违约样本仅 ~15-20%，对比：
  - 不处理
  - class_weight 调整
  - scale_pos_weight (XGBoost)
  - SMOTE 过采样
  - Focal Loss

**评估**：
- 各模型 AUC、KS、PR-AUC
- 训练时间、推理速度
- 特征重要性（gain-based）

**复用关系**：选最优单模型进 Stage 5 做 XAI 分析；选集成模型作 Stage 6 部署

### Stage 4：表格深度学习对比（Week 5-7）

**目标**：诚实评估深度学习相对树模型的真实价值。

**已知结论**（Yandex NeurIPS 2021）：在大多数表格数据上，**XGBoost 仍是冠军**。但深度学习在以下场景有优势：高维稀疏特征、需要 embedding、多模态融合。

**子实验**：

**4a. TabNet**
- 用 pytorch_tabnet
- 优势：自带特征选择注意力，可解释性 inherent
- 调参：n_d、n_a、n_steps、gamma

**4b. FT-Transformer**
- 用 rtdl-revisiting-models 官方实现
- 调参：token_size、n_blocks、attention_dropout

**4c. NODE / SAINT（可选）**
- 时间紧可砍

**4d. 工业级深度模型 EBM (Explainable Boosting Machine)**
- InterpretML 提供
- 兼顾精度与可解释性，**银行实际部署的好选择**
- 自动给出特征贡献图

**4e. 与树模型直接对比**
- 同样数据、同样评测
- 报告：AUC 差异、训练成本差异、推理速度差异
- **诚实结论**：如果树模型赢了，明确说出来——这是一个有价值的负面结果

**评估对比表**：

| 模型 | AUC | KS | 训练时间 | 推理 ms | 参数量 | 可解释性 |
|---|---|---|---|---|---|---|
| LR + WOE | | | | | | ⭐⭐⭐⭐⭐ |
| XGBoost | | | | | | ⭐⭐⭐ |
| LightGBM | | | | | | ⭐⭐⭐ |
| CatBoost | | | | | | ⭐⭐⭐ |
| Stacking | | | | | | ⭐⭐ |
| TabNet | | | | | | ⭐⭐⭐⭐ |
| FT-Transformer | | | | | | ⭐⭐ |
| EBM | | | | | | ⭐⭐⭐⭐⭐ |

### Stage 5：XAI + 公平性核心（Week 7-9）

**目标**：把模型从"黑盒预测"转为"可审计、可解释、可监管"的系统。**这是本项目的最大亮点**。

**5a. 全局解释**
- 特征重要性（XGBoost gain / Permutation importance）
- SHAP summary plot
- Partial Dependence Plots (PDP) / ICE plots
- Feature interaction analysis

**5b. 局部解释（单笔贷款决策）**
- SHAP force plot：单条预测的特征贡献
- LIME：本地线性近似
- 对比 SHAP vs LIME 的稳定性

**5c. 反事实解释 (DiCE)** ⭐
- "如果年收入 +$10K，决策会变吗？"
- 给被拒申请人**可执行的改进建议**
- 这是面向终端用户的关键 narrative

**5d. XAI 评估指标**
- Faithfulness（解释是否真实反映模型）
- Stability（相似输入解释是否相似）
- Comprehensibility（人工可读性）

**5e. 公平性审计**

敏感属性候选：
- 性别（Home Credit 直接有；LendingClub 需从 `emp_title` 推断或用 Home Credit 子项目）
- 年龄段
- 收入分段
- 地区（代理种族）

公平性指标（用 fairlearn）：
- **Demographic Parity Difference (DPD)**：通过率在群体间的差异
- **Equal Opportunity Difference (EOD)**：真阳性率在群体间的差异
- **Equalized Odds**：TPR + FPR 都平衡
- **Disparate Impact Ratio**：< 0.8 触发监管红线（美国"80% rule"）

**5f. 公平性缓解**
- Reweighing（训练前）
- Adversarial Debiasing（训练中）
- Threshold Optimizer（训练后）
- 对比"精度-公平性"权衡曲线

**评估**：
- 公平性指标变化（缓解前 vs 后）
- AUC 损失（公平性的代价）
- 业务影响（通过率变化）

**复用关系**：Stage 6 的 LLM 报告需要 SHAP 数值；公平性结果会写入审批报告的"合规检查"字段

### Stage 6：LLM 审批报告 + Demo + 部署（Week 9-12）

**目标**：把技术成果包装成业务可用的系统。

**6a. LLM 审批报告生成模块** ⭐（与 RAG portfolio 联动）

**架构**：
```
模型预测 (概率 + SHAP)
   ↓
[结构化模板] 提取 top-3 风险驱动因子 + 反事实建议
   ↓
[LLM Prompt] "你是信贷分析师，请基于以下数据生成审批意见..."
   ↓
[RAG 检索] 政策文档、监管要求、行业基准（与现有 RAG 项目复用）
   ↓
[输出] 自然语言审批报告 + 引用政策条款
```

**对比实验**：
- 本地 Qwen2.5-7B-Instruct（T4 INT4）
- GPT-4o-mini API
- Claude 3.5 Haiku API
- 评估：信息准确性（是否复述了正确的 SHAP 数值）、可读性（人工评分）、监管语言合规性、成本

**6b. Gradio Web Demo**
- 上传 CSV 或填写表单 → 获得：
  - 违约概率 + 信用分
  - 决策（通过/拒绝/转人工）
  - SHAP 可视化（瀑布图 + 力图）
  - 反事实建议（"如果...你将通过"）
  - 公平性审计标签
  - **LLM 生成的审批报告**（中文+英文双语）
- 后台审计日志

**6c. 模型监控仪表盘**
- PSI 时间漂移
- 各群体公平性指标实时监控
- 模型性能衰减预警

**6d. 部署优化**
- ONNX 导出 + 量化
- 推理速度：目标 < 50ms/申请

**6e. 技术报告 + GitHub**
- 报告（15-20 页）
  - 含监管合规章节、公平性权衡分析、模型卡片 (Model Card)
- GitHub repo
- 数据卡片 (Data Card) + 模型卡片 (Model Card)
- 5 分钟演示视频

---

## 5. 实验设计

### 5.1 主模型对比表（必做）

| 模型 | LendingClub Test AUC | LendingClub Test KS | Home Credit AUC | 推理 ms | 训练 GPU·h |
|---|---|---|---|---|---|
| LR + WOE (Scorecard) | | | | | |
| Logistic Regression (raw) | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |
| LightGBM | | | | | |
| CatBoost | | | | | |
| Stacking | | | | | |
| TabNet | | | | | |
| FT-Transformer | | | | | |
| EBM | | | | | |

### 5.2 XAI 评估对比

| 解释方法 | Faithfulness | Stability | 人工评分（1-5） | 计算时间 ms |
|---|---|---|---|---|
| SHAP (TreeSHAP) | | | | |
| LIME | | | | |
| DiCE (反事实) | | | | |
| EBM 原生 | | | | |
| TabNet 原生 attention | | | | |

### 5.3 公平性审计表

| 模型 | DPD | EOD | DIR | AUC 损失 | 通过率变化 |
|---|---|---|---|---|---|
| XGBoost 原始 | | | | - | - |
| XGBoost + Reweighing | | | | | |
| XGBoost + Adversarial | | | | | |
| XGBoost + Threshold Optimizer | | | | | |

### 5.4 LLM 报告评估

| 模型 | 准确性（数值复述对错率） | 可读性 (1-5) | 合规语言评分 | 成本 / 报告 |
|---|---|---|---|---|
| Qwen2.5-7B 本地 | | | | $0 |
| GPT-4o-mini | | | | $0.005 |
| Claude 3.5 Haiku | | | | $0.003 |

### 5.5 消融实验（至少 4 组）

1. **WOE 编码效果**：原始特征 vs WOE 编码（在 LR 和 XGBoost 上）
2. **类别不平衡处理**：4 种策略对比
3. **时间漂移**：训练于 2014-2016，分别测于 2017、2018 的 AUC
4. **公平性权衡**：精度-公平性 Pareto 前沿

### 5.6 鲁棒性测试

- 特征缺失（随机 mask 10%/30%/50%）下的预测稳定性
- 输入噪声（数值特征 ±10% 扰动）
- 不同子群体（不同年龄、收入段）的稳定性

---

## 6. 项目里程碑

| 周次 | 目标 | 输出 |
|---|---|---|
| **Week 1** | 数据下载 + EDA + 文献调研 | LendingClub 数据卡片、EDA 笔记本 |
| **Week 2** | 评测协议 + 数据清洗 pipeline + LR baseline | 评测脚本、清洗代码、LR 跑通 |
| **Week 3** | Scorecard (WOE + IV + 单调性) | Stage 2 完整 baseline |
| **Week 4** | XGBoost / LightGBM / CatBoost 训练 | Stage 3a 结果 |
| **Week 5** | 超参优化 + Stacking + 不平衡处理 | Stage 3 完整对比 |
| **Week 6** | TabNet + FT-Transformer + EBM | Stage 4 深度学习对比 |
| **Week 7** | SHAP / LIME / DiCE 解释分析 | Stage 5a-d XAI 报告 |
| **Week 8** | 公平性审计 + 缓解算法 | Stage 5e/5f 公平性报告 |
| **Week 9** | LLM 报告生成模块 + 三模型对比 | Stage 6a 完成 |
| **Week 10** | Gradio Demo + 模型监控仪表盘 | Stage 6b/6c 完成 |
| **Week 11** | 部署优化 + 鲁棒性测试 + 误差分析 | 完整鲁棒性报告 |
| **Week 12** | 报告 + GitHub + 视频 + Model Card | 全部交付物 |

**弹性策略**：
- 砍 4c (NODE/SAINT) 节省 0.5 周
- 砍 6c (监控仪表盘) 节省 0.5 周
- 第二数据集 (Home Credit) 列为 stretch goal

---

## 7. 最终交付物

1. **GitHub 仓库**：
   - `README.md`（项目介绍、Model Card、Data Card）
   - `environment.yml`
   - `notebooks/01_eda.ipynb`、`02_scorecard.ipynb`、`03_models.ipynb`、`04_xai.ipynb`、`05_fairness.ipynb`、`06_llm_report.ipynb`
   - `src/`（模块化代码）
   - `configs/`（所有实验 YAML config）
   - `results/`（实验结果、可视化、模型权重）
   - `tests/`（pytest 单元测试，体现工程素养）

2. **Gradio Web Demo**（HuggingFace Spaces 部署）
   - 实时审批流程演示
   - SHAP 可视化
   - LLM 报告生成

3. **技术报告**（15-20 页 PDF）
   - 含监管合规章节、公平性权衡分析

4. **Model Card + Data Card**（参考 Mitchell et al. 2019）
   - 模型适用范围、局限性、偏差风险

5. **演示视频**（5 分钟）

6. **PPT slides**（最终汇报）

---

## 8. Portfolio 三件套联动叙事（重要）

> **数据形态全覆盖：**
> - **RAG 项目**：处理银行政策、合规手册等**非结构化文本**
> - **金融文档 KIE**：处理收据、发票等**半结构化图像/文档**
> - **信用评分 + XAI**：处理客户特征、交易数据等**结构化表格**

> **业务流程全闭环：**
> - **KIE 项目**：申请材料数字化（输入处理）
> - **信用评分项目**：风险决策 + LLM 审批报告（核心决策）
> - **RAG 项目**：政策检索、客户问答（知识服务）

> **技术深度全维度：**
> - **多模态预训练**（LayoutLMv3 in KIE）
> - **经典 ML + 深度学习对比**（本项目）
> - **可解释 AI + 公平性 + LLM**（本项目核心亮点 + 与 RAG 联动）

**在简历/面试时的一句话总结**：
> "我系统性构建了银行 AI 三件套：用 RAG 做知识检索，用 KIE 做文档自动化，用可解释信用评分模型做风险决策，三个项目通过 LLM 报告生成形成业务闭环。"

---

## 9. 行业叙事点（写报告/简历/面试用）

1. **监管对标**：Scorecard + WOE + PSI 是银行真实在用的方法（区别于"刷 AUC"的学生项目）
2. **可解释性是稀缺技能**：欧盟 AI Act、巴塞尔协议都要求模型可解释，能讲监管语言
3. **公平性合规**：DIR 80% 规则、ECOA 反歧视法的工程实现
4. **决策可追溯**：每笔审批有完整的 SHAP 解释 + LLM 报告 + 审计日志
5. **业务价值**：通过率、坏账率、客户改进建议（不只是 AUC）
6. **跨技术栈集成**：传统 ML + 深度学习 + LLM 三栈协同

---

## 10. 关键风险与诚实说明

- ⚠️ **诚实评估"树模型 vs 深度学习"**：在表格数据上 XGBoost 大概率仍最强，这本身是有价值的结论，不要为了 narrative 强行让深度学习赢
- ⚠️ **公平性指标互相冲突**：DPD、EOD、Calibration 不可同时满足（Kleinberg 2017 不可能性定理），必须选择并解释
- ⚠️ **LendingClub 数据有时间漂移**：2014 模型在 2018 表现下降是正常现象，需正确归因
- ⚠️ **LLM 报告可能幻觉**：必须用 SHAP 数值做事实锚定，不能让 LLM 凭空生成数字
- ⚠️ **公平性缓解会牺牲精度**：诚实报告 Pareto 前沿
- ⚠️ **特征数据泄漏陷阱**：LendingClub 大量字段是贷款发放后才有的，必须严格剔除
- ⚠️ **样本选择偏差**：训练数据只包含"被批准过的贷款"，未通过的人不在数据中，违约预测有 survivor bias

---

## 附录：参考链接快查

### 数据集
- LendingClub: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Home Credit: https://www.kaggle.com/competitions/home-credit-default-risk
- Give Me Some Credit: https://www.kaggle.com/competitions/GiveMeSomeCredit
- FICO HELOC: https://community.fico.com/s/explainable-machine-learning-challenge

### 核心工具
- optbinning: https://github.com/guillermo-navas-palencia/optbinning
- SHAP: https://github.com/shap/shap
- DiCE: https://github.com/interpretml/DiCE
- Fairlearn: https://github.com/fairlearn/fairlearn
- InterpretML (EBM): https://github.com/interpretml/interpret
- pytorch-tabular: https://github.com/manujosephv/pytorch_tabular

### 参考阅读
- "Fairness and Machine Learning" book: https://fairmlbook.org/
- Google Model Cards: https://modelcards.withgoogle.com/about
- EU AI Act 信用评分章节: https://artificialintelligenceact.eu/
