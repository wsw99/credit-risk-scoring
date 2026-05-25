# Stage 3 报告：树模型与超参数优化

## 概述

Stage 3 在 Stage 2 WOE 评分卡（AUC=0.704, KS=0.295）基础上，引入四种树模型（Random Forest、XGBoost、LightGBM、CatBoost），通过基线对比、Optuna 超参数优化和集成学习，进一步提升信用风险预测性能。

数据划分：训练集 2014-2016（451K）、验证集 2017（376K）、测试集 2018（293K），与 Stage 2 保持一致。

---

## 1. 基线模型对比

使用各模型的合理默认参数训练，不做超参数调优。

| 模型 | Test AUC | Test KS | Test Gini | Test PR-AUC | 训练时间 |
|------|----------|---------|-----------|-------------|----------|
| CatBoost | 0.7194 | 0.3189 | 0.4387 | 0.4292 | 258.6s |
| XGBoost | 0.7191 | 0.3163 | 0.4382 | 0.4294 | 110.9s |
| LightGBM | 0.7188 | 0.3180 | 0.4375 | 0.4281 | 26.2s |
| Random Forest | 0.7134 | 0.3111 | 0.4269 | 0.4212 | 57.8s |

**关键发现：**
- 所有树模型均显著优于 Stage 2 评分卡（AUC 提升 1.0-1.5 个百分点，KS 提升 2-2.5 个百分点）
- CatBoost、XGBoost、LightGBM 三者性能几乎持平（AUC 差异 < 0.001），但 LightGBM 训练速度碾压（26s vs CatBoost 259s，约 10 倍）
- Random Forest 明显弱于 Boosting 类模型，Gini 差异约 1 个百分点
- **LightGBM 是最优基线模型**：性能与最好模型持平，速度远超其他

---

## 2. Optuna 超参数优化

对 XGBoost 和 LightGBM 各进行 50 次 Optuna 搜索（TPE sampler, MedianPruner）。

### XGBoost 最优参数

| 参数 | 最优值 |
|------|--------|
| learning_rate | 0.0101 |
| max_depth | 6 |
| min_child_weight | 5 |
| subsample | 0.677 |
| colsample_bytree | 0.857 |
| reg_alpha | 0.817 |
| reg_lambda | 0.723 |
| scale_pos_weight | 1.57 |

验证集 AUC: **0.7421** | 测试集 AUC: **0.7202**（+0.001 vs 基线）

### LightGBM 最优参数

| 参数 | 最优值 |
|------|--------|
| learning_rate | 0.0136 |
| num_leaves | 34 |
| min_child_samples | 29 |
| subsample | 0.870 |
| colsample_bytree | 0.660 |
| reg_alpha | 0.148 |
| reg_lambda | 0.025 |

验证集 AUC: **0.7417** | 测试集 AUC: **0.7184**（-0.0004 vs 基线）

**关键发现：**
- Optuna 优化的提升幅度有限（XGB +0.001, LGB 微跌），说明**默认参数已经接近最优**
- 优化后 XGBoost（AUC=0.7202）成为最佳单模型
- Optuna 搜索到的参数倾向于更强的正则化（高 reg_alpha、低 learning_rate），符合风控场景对过拟合的敏感性
- 50 次试验的搜索曲线在约 20 次后趋于平缓，说明搜索空间不需要更大

---

## 3. 集成学习：Stacking 与 Blending

使用 XGBoost (Optuna) + LightGBM (Optuna) + CatBoost (Baseline) 作为基模型。

| 方法 | Test AUC | Test KS | Test Gini | Test PR-AUC |
|------|----------|---------|-----------|-------------|
| Blending（等权） | **0.7204** | 0.3195 | 0.4407 | 0.4309 |
| Stacking（LR 元学习器） | 0.7202 | 0.3191 | 0.4404 | 0.4308 |

**关键发现：**
- 集成方法略微超越单模型，但提升幅度极小（Blending +0.0002 vs 最佳 XGB）
- Blending（简单加权平均）反而略优于 Stacking，说明简单的等权集成已足够
- 三个 Boosting 模型相关性较高，限制了集成的多样性收益
- **集成不是 Stage 3 的主要收益来源**——从评分卡到树模型的跃升才是核心提升

---

## 4. 最终模型对比

| 排名 | 模型 | Test AUC | Test KS | Test Gini |
|------|------|----------|---------|-----------|
| 1 | Blending (XGB+LGB+CB) | 0.7204 | 0.3195 | 0.4407 |
| 2 | Stacking (XGB+LGB+CB) | 0.7202 | 0.3191 | 0.4404 |
| 3 | XGBoost (Optuna) | 0.7202 | 0.3178 | 0.4403 |
| 4 | CatBoost (Baseline) | 0.7194 | 0.3189 | 0.4387 |
| 5 | XGBoost (Baseline) | 0.7191 | 0.3163 | 0.4382 |
| 6 | LightGBM (Baseline) | 0.7188 | 0.3180 | 0.4375 |
| 7 | LightGBM (Optuna) | 0.7184 | 0.3176 | 0.4369 |
| 8 | Random Forest (Baseline) | 0.7134 | 0.3111 | 0.4269 |
| — | **Stage 2 WOE 评分卡** | **0.7040** | **0.2946** | **0.4079** |

**对比 Stage 2：**
- 最佳树模型 AUC 提升 **+0.0164**（0.7040 → 0.7204），相对提升 **2.3%**
- KS 提升 **+0.0249**（0.2946 → 0.3195），相对提升 **8.5%**
- Gini 提升 **+0.0328**（0.4079 → 0.4407），相对提升 **8.0%**

---

## 5. 结论与建议

### 核心结论

1. **树模型全面超越评分卡。** Boosting 类模型（XGB/LGB/CB）将 AUC 从 0.704 提升至 0.719-0.720，KS 从 0.295 提升至 0.318-0.319，在所有指标上均显著优于 Stage 2 的 WOE + 逻辑回归方案。

2. **LightGBM 性价比最高。** 训练仅需 26 秒，性能与 CatBoost/XGBoost 基本持平（AUC 差距 < 0.001），适合快速迭代和部署。

3. **Optuna 优化收益有限。** 默认参数已接近最优，50 次搜索带来的提升仅 0.001 AUC。在时间有限的情况下，使用合理默认参数即可，无需反复调优。

4. **集成提升空间小。** 三个 Boosting 模型高度相关，Stacking/Blending 仅带来 0.0002-0.0012 的边际改善。单独部署一个调优后的 XGBoost 或 LightGBM 已足够。

### 推荐方案

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 追求极致性能 | Blending (XGB+LGB+CB) | AUC 最高 0.7204 |
| 快速部署/上线 | LightGBM (Baseline) | 训练 26s，AUC 0.7188，性能损失 < 0.2% |
| 兼顾性能与速度 | XGBoost (Optuna) | AUC 0.7202，训练约 2min |

### 后续方向（Stage 4+）

- **深度学习模型**：TabNet、FT-Transformer 可能在特征交互上进一步突破
- **特征工程深化**：历史行为特征、还款时序特征的构建空间较大
- **模型校准**：树模型的概率输出需要 Platt Scaling 或 Isotonic Regression 校准
- **决策阈值优化**：结合业务损失函数（误拒成本 vs 坏账成本）确定最优 cutoff
