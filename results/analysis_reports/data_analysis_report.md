# LendingClub 信用评分 — 数据分析报告

> 生成日期: 2026-05-19
> 数据集: [LendingClub Accepted Loans 2007-2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

## 1. 数据总览

### 1.1 原始数据

| 指标 | 值 |
|---|---|
| 总贷款笔数 | 2,260,701 |
| 原始特征数 | 151 |
| 时间跨度 | 2007年6月 → 2018年12月 |
| 数据来源 | LendingClub 公开发布的已获批贷款数据 |

### 1.2 目标变量 (`loan_status`)

原始数据包含 9 种贷款状态：

| 状态 | 数量 | 占比 | 处理方式 |
|---|---|---|---|
| Fully Paid | 1,076,751 | 47.6% | → **Good (0)** |
| Charged Off | 268,559 | 11.9% | → **Bad (1)** |
| Current | 878,317 | 38.9% | 排除（无最终结果） |
| Late (31-120 days) | 21,467 | 0.9% | 排除（尚未最终违约） |
| Late (16-30 days) | 4,349 | 0.2% | 排除 |
| In Grace Period | 8,436 | 0.4% | 排除 |
| Default | 40 | <0.01% | → **Bad (1)** |
| 其他（政策例外） | 2,749 | 0.1% | 排除 |

**目标定义**: `is_bad = 1` 当 `loan_status ∈ {Charged Off, Default}`，`is_bad = 0` 当 `loan_status = Fully Paid`。所有中间状态（Current, Late, Grace）全部排除。

**为什么排除 Current/Late？**
- Current：贷款仍在还款中，不知道最终会不会违约 → 标签不确定
- Late (16-30/31-120 days)：逾期但可能恢复，不等于最终违约 → 标签模糊
- 包含这些样本会导致错误标签，模型学到错误信号

---

## 2. 数据样例

从原始数据中选取 5 笔不同状态的贷款，展示各维度特征的典型值：

| 维度 | 列名 | 样例1 | 样例2 | 样例3 | 样例4 | 样例5 |
|---|---|---|---|---|---|---|
| **标签** | `loan_status` | **Fully Paid** | **Charged Off** | Current | Late (31-120) | In Grace Period |
| | → 最终处理 | ✅ Good (0) | ✅ Bad (1) | ❌ 排除 | ❌ 排除 | ❌ 排除 |
| **数值特征** | `loan_amnt` | $3,600 | $18,000 | $35,000 | $16,000 | $16,000 |
| | `int_rate` | 13.99% | 19.48% | 14.85% | 14.85% | 8.49% |
| **分类特征** | `grade` | C | E | C | C | B |
| **非特征** | `emp_title` | leadman | Software Manager | Information Systems Officer | District Sales Leader | Supervisor |
| **数据泄露** | `total_pymnt` | $4,421 | $9,452 | $31,464 | $14,157 | $12,430 |
| | `recoveries` | $0 | $1,618 | $0 | $0 | $0 |
| **高缺失** | `annual_inc_joint` | NaN | NaN | NaN | NaN | NaN |
| | `member_id` | NaN | NaN | NaN | NaN | NaN |

**观察**：
- `emp_title` 是自由文本（"leadman", "Information Systems Officer"），高基数、格式不统一 → **非特征，已移除**
- `total_pymnt` 是还款总额，只有贷款结清后才知道 → **数据泄露，已移除**
- `recoveries` 是催收追回金额，只有违约后才有 → **数据泄露，已移除**
- `annual_inc_joint` 和 `member_id` 几乎全为空（~95%+） → **高缺失，已移除**
- Current / Late / Grace 状态的贷款没有最终结果 → **标签不确定，已排除**

---

## 3. 数据清洗

### 3.1 清洗流水线

数据清洗分两阶段：**全局清洗**（在完整数据集上）和 **逐 split 清洗**（切分后对每个 train/val/test 独立检查）。

| 步骤 | 阶段 | 方法 | 移除 | 原因 |
|---|---|---|---|---|
| 1. 标签定义 | 全局 | 仅保留 Fully Paid / Charged Off / Default | 879K 行 | 排除 Current/Late/Grace 等无最终结果的贷款 |
| 2. ID/制品列 | 全局 | 按列名删除 | 7 列 | `id`, `member_id`, `url`, `desc`, `emp_title`, `title`, `zip_code` |
| 3. 高缺失率 | 全局 | NaN 率 > 80%（在完整数据上） | 38 列 | `hardship_*`, `sec_app_*`, `annual_inc_joint` 等 |
| 4. 数据泄露 | 全局 | 贷款发放后才可观测 | 31 列 | `total_pymnt`, `recoveries`, `last_pymnt_*`, `settlement_*` 等 |
| 5. 时间切分 | 全局 | 按 issue_d 切分为 Train(2007-2014) / Val(2015) / Test(2016) | — | 严格 Out-of-Time Split |
| 6. 逐 split 清洗 | 每 split | NaN 率 ≥ 80% **或** 常数率 ≥ 99% 在任一 split 中 | 14 NaN + 9 常数 | `open_acc_6m` 等 LendingClub 2015+ 新增字段；`pymnt_plan` 等近常数列 |

> **步骤 6 的设计动机**：全局 80% 阈值会漏掉在 train 中 100% 为 NaN 但 val/test 中有值的特征（因为全局 NaN 率 ≈ 40%，远低于 80%）。这些特征在全局检查中"合法存活"，但在 train 中无法提供任何信号。逐 split 检查独立评估每个数据集，确保 train/val/test 中的特征质量一致。

**总计**：原始 151 列 → 全局清洗后 92 列 → 逐 split 清洗后 **68 列**（65 特征 + `is_bad` + `issue_d` + `loan_status`）。

### 3.2 清洗后数据

| 指标 | 值 |
|---|---|
| 有效贷款笔数 | 1,119,711 |
| 总列数 | 68 |
| 可用特征数 | 65（55 数值 + 10 分类） |
| 总体违约率 | 19.92% |

---

## 4. 数据切分方案

### 4.1 切分方法: 严格时间切分 (Out-of-Time Split)

```
Train: 2007年 → 2014年 (8年，451,060笔)
Val:   2015年        (1年，375,546笔)
Test:  2016年        (1年，293,105笔)
```

### 4.2 为什么用时间切分而不是随机切分？

| | 随机切分 (Random Split) | 时间切分 (Temporal Split) |
|---|---|---|
| **做法** | 打乱所有样本，随机分配 70/15/15 | 按贷款发放年份切分 |
| **AUC 典型值** | 0.78-0.80 (偏高) | 0.71-0.74 (真实) |
| **问题** | 同一时期的相似贷款分到 train 和 test → **数据泄露** | 模拟真实部署：用过去预测未来 |
| **银行实际使用** | 不使用 | **使用** |

随机切分会把 2015 年的贷款分到 train 和 test，模型见过"经济环境相似"的样本后再在 test 上预测，AUC 虚高。**这是学生项目最常见的错误之一。**

### 3.3 为什么选 2007-2014 做 Train、2016 做 Test？

1. **贷款成熟度**: LendingClub 贷款期限为 36 或 60 个月。2016 年发放的贷款，最晚 2021 年到期。数据收集截止于 2018 年 Q3，2016 年的贷款有至少 20 个月的观察期——足够大部分违约行为发生。
2. **数据量平衡**: Train 45 万、Val 37 万、Test 29 万，三个集合都足够大。
3. **与社区实践对齐**: Kaggle 上常见的 LendingClub 项目也以 2014-2015 作为切分点。

### 3.4 切分后数据统计

| 数据集 | 年份 | 样本数 | 占比 | 违约率 |
|---|---|---|---|---|
| **Train** | 2007-2014 | 451,060 | 40.3% | 16.96% |
| **Val** | 2015 | 375,546 | 33.5% | 20.19% |
| **Test** | 2016 | 293,105 | 26.2% | 23.29% |

> 违约率逐年上升（17% → 20% → 23%），反映了 LendingClub 在后期放宽了信贷标准，这是真实的时间漂移信号。

---

## 4. 基准模型结果 (Benchmark)

### 4.1 Logistic Regression（62 特征，排除 grade/sub_grade 以与 Stage 2 对齐）

| C | Train AUC | Val AUC | Test AUC |
|---|---|---|---|
| 0.01 | 0.7031 | 0.7336 | 0.7108 |
| 0.10 | 0.7031 | 0.7336 | 0.7108 |
| 1.00 | 0.7031 | 0.7336 | 0.7108 |
| 10.00 | 0.7031 | 0.7336 | 0.7108 |

> C 参数对结果几乎无影响，说明 L2 正则化在此特征规模下饱和。排除 grade/sub_grade 是为了与 Stage 2 评分卡做公平对比（Stage 2 不使用 LendingClub 内部评级）。

**测试集基准: AUC = 0.7108, KS = 0.307**

### 4.2 与社区结果对比

| 来源 | 切分方式 | LR AUC | XGBoost AUC |
|---|---|---|---|
| **本项目 Stage 1** | 时间切分 2007-2014/2015/2016 | **0.711** | 待跑 |
| 本项目 Stage 2 | 时间切分, WOE 编码 | **0.704** | — |
| [sjagannathan17](https://github.com/sjagannathan17/Lending-Club-ML-Analysis) | 10-fold 随机 CV | 0.78 | **0.80** |
| [chetan7659](https://github.com/chetan7659/Financial-Risk-Intelligence-System-Loan-Default-Prediction-) | 未详述 | 0.72 | 0.73 |
| 文献一般范围 | 随机切分 | 0.68-0.72 | 0.73-0.78 |

本项目 LR 0.711 处于时间切分的合理范围内。**后续 XGBoost/LightGBM 的目标是将 Test AUC 从 0.711 提升到 0.73-0.75。**

---

## 5. 关键发现与风险

### 5.1 数据特征

- **Top 5 最有预测力的单特征**: `int_rate` (AUC 0.673), `fico_range_low` (0.588), `fico_range_high` (0.588), `acc_open_past_24mths` (0.569), `dti` (0.569)
- `grade` 和 `sub_grade` 是 LendingClub 自己的内部评级，预测力很强但在真实银行场景中不可用（相当于用了别人的模型输出做特征）
- 多个特征高度相关（如 `fico_range_low` ↔ `fico_range_high` 相关系数 1.0），需要去重或在模型中选择一个

### 5.2 残余风险

| 风险 | 严重性 | 缓解 |
|---|---|---|
| **幸存者偏差** | 高 | 训练数据只有获批贷款，被拒申请人的特征不在数据中；在报告和面试中需明确说明 |
| **时间漂移** | 中 | Test 违约率 (23.3%) 明显高于 Train (17.0%)；通过 PSI 监控跟踪 |
| **地域偏差** | 中 | `addr_state` 作为地域代理变量可能导致公平性问题；Stage 5 公平性审计中检查 |
| **部分特征缺失** | 低 | `mths_since_last_delinq` (51% 缺失) 等需要填补策略 |

---

## 6. 下一步

- **Stage 2**: 经典银行评分卡（WOE 编码 + 逻辑回归） — 对标监管要求，AUC 0.704
- **Stage 3**: XGBoost / LightGBM / CatBoost + Optuna 调参 — 性能基准
- **目标**: Test AUC 达到 0.73-0.75，KS 达到 0.30-0.35

---

## 附录: 数据文件清单

| 文件 | 路径 | 说明 |
|---|---|---|
| 原始数据 | `data/raw/wordsforthewise_lending-club/` | LendingClub 原始 CSV (2007-2018) |
| 清洗后 Train | `data/processed/train.parquet` | 2007-2014, 451K 笔 |
| 清洗后 Val | `data/processed/val.parquet` | 2015, 375K 笔 |
| 清洗后 Test | `data/processed/test.parquet` | 2016, 293K 笔 |
| 分析图表 | `results/analysis_reports/*.png` | 清洗前后对比图 (10张) |
