# Project Proposal: Explainable & Fair Personal Loan Default Prediction System

> Project type: Deep exploration project (10-12 weeks)
> Hardware: AWS T4 GPU (16GB VRAM)
> Industry target: Bank retail credit, internet lending, consumer finance
> Core capabilities: Tabular ML modeling + **Explainable AI (XAI)** + **Algorithmic Fairness** + **LLM-generated approval reports**
>
> Portfolio narrative: RAG (text) + KIE (image/document) + **This project (structured tabular data)** — covering all three data modalities of bank AI.

---

## 1. Project Overview

**Goal**: Build an end-to-end personal loan default prediction system that pursues not just AUC but also:
1. **Explainability**: every decision must be defensible (regulatory requirement)
2. **Fairness**: detect and mitigate bias across protected attributes (gender, age, etc.)
3. **LLM automated reporting**: turn SHAP explanations into human-readable approval reports (interoperates with the existing RAG project)

**Target business scenarios**:
- Bank personal consumer loan underwriting
- Credit card installment default early warning
- Internet consumer finance (Ant Group, JD, consumer finance companies)
- Basel III IRB internal-rating-model reference build

**Input / Output**:
- Input: applicant structured features (annual income, DTI, credit history, loan purpose, employment...)
- Output:
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
  "llm_report": "Applicant has high DTI (32.5%), mostly driven by... Recommend restructuring debt first..."
}
```

**Final deliverables**: Reproducible training/inference code, model-comparison tables, XAI analysis report, fairness audit report, Gradio approval demo (with LLM report generation), technical report, 5-minute demo video.

---

## 2. Related Projects and References

### 2.1 University Course / Assignment References

| Source | Content | Takeaways |
|---|---|---|
| **Stanford CS229 (ML)** | Classical classification, feature engineering | Project report structure |
| **Stanford STATS 315A/B** | Modern regression and classification (Hastie/Tibshirani) | Rigorous LR/Logistic modeling |
| **Berkeley INDENG 290 / DATA 100** | Data science project templates | End-to-end pipeline paradigm |
| **CMU 10-708 PGM** | Probabilistic graphical models, useful for uncertainty | Bayesian supplement |
| **Stanford CS329T (Trustworthy ML)** ⭐ | XAI / Fairness systems course | **Core methodology for this project** |
| **MIT 6.S898 Deep Learning** | Tabular deep learning topics | TabNet / FT-Transformer reference |
| **Harvard / Stanford "AI Ethics"** | Fairness, bias, regulation | Fairness narrative |

### 2.2 Open-Source Project References

| Project | Description | Reuse value |
|---|---|---|
| **[scikit-learn](https://github.com/scikit-learn/scikit-learn)** ⭐⭐⭐⭐⭐ | LR, feature processing, evaluation | Used throughout |
| **[XGBoost / LightGBM / CatBoost](https://github.com/microsoft/LightGBM)** ⭐⭐⭐⭐⭐ | Mainstream tree models | Main models |
| **[feature-engine](https://github.com/feature-engine/feature_engine)** ⭐⭐⭐⭐ | Bank-grade feature engineering | WOE/IV regulatory standards |
| **[optbinning](https://github.com/guillermo-navas-palencia/optbinning)** ⭐⭐⭐⭐⭐ | **Purpose-built for credit scoring**, WOE binning, scorecard automation | Stage 2 core |
| **[SHAP](https://github.com/shap/shap)** ⭐⭐⭐⭐⭐ | Shapley-value explanations | XAI main |
| **[LIME](https://github.com/marcotcr/lime)** ⭐⭐⭐⭐ | Local explanations | XAI comparison |
| **[DiCE](https://github.com/interpretml/DiCE)** ⭐⭐⭐⭐ | Counterfactual explanations | "Why rejected and how to qualify" |
| **[InterpretML](https://github.com/interpretml/interpret)** ⭐⭐⭐⭐ | EBM interpretable boosting | High accuracy + high interpretability |
| **[Fairlearn (Microsoft)](https://github.com/fairlearn/fairlearn)** ⭐⭐⭐⭐⭐ | Fairness metrics + mitigation | **Fairness core** |
| **[AIF360 (IBM)](https://github.com/Trusted-AI/AIF360)** ⭐⭐⭐⭐ | Fairness toolkit | Fairness comparison |
| **[pytorch-tabular](https://github.com/manujosephv/pytorch_tabular)** ⭐⭐⭐⭐ | TabNet/FT-Transformer/NODE | Stage 4 deep learning |
| **[Kaggle Home Credit top solutions](https://www.kaggle.com/c/home-credit-default-risk/discussion)** ⭐⭐⭐⭐⭐ | Top 1-10 public solutions | Engineering experience |
| **[LangChain / LlamaIndex](https://github.com/langchain-ai/langchain)** ⭐⭐⭐ | LLM report generation | Stage 6 module |

### 2.3 Papers and Benchmarks

**Classical credit scoring**:
- Hand & Henley (1997) — Bank credit scoring survey
- Thomas et al. (2002) — *Credit Scoring and Its Applications*, standard textbook
- Lessmann et al. (2015) — Mainstream ML credit model benchmark

**Tree models**:
- XGBoost (KDD 2016)
- LightGBM (NeurIPS 2017)
- CatBoost (NeurIPS 2018)

**Tabular deep learning**:
- TabNet (AAAI 2021)
- FT-Transformer (NeurIPS 2021)
- NODE (ICLR 2020)
- SAINT (2021)
- **"Revisiting Deep Learning Models for Tabular Data"** (NeurIPS 2021) — Yandex; conclusion: tree models still dominate

**XAI**:
- SHAP (Lundberg & Lee, NeurIPS 2017)
- LIME (Ribeiro et al., KDD 2016)
- DiCE / Counterfactual Explanations (Wachter et al. 2017)
- "A Unified Approach to Interpreting Model Predictions" — SHAP paper must-read

**Fairness**:
- Hardt et al. (NeurIPS 2016) — Equal Opportunity
- Barocas et al. — "Fairness and Machine Learning" online book
- Mehrabi et al. (ACM Computing Surveys 2021) — Bias survey

**Financial ML regulatory documents**:
- Basel III IRB documentation
- EU AI Act (2024 enacted) — credit scoring listed as high-risk AI
- US ECOA (Equal Credit Opportunity Act) — anti-discrimination
- China "Commercial Bank Internal-Ratings-Based Approach Regulations"

### 2.4 Datasets

| Dataset | Scale | Annotation | Recommended use |
|---|---|---|---|
| **LendingClub Loan Data** ⭐⭐⭐⭐⭐ | 2007-2018, ~2.2M loans | Default labels + 150+ features + issue date | **Main dataset**, large scale + time axis |
| **Kaggle Home Credit Default Risk** ⭐⭐⭐⭐⭐ | 300K+ applicants, 7 related tables | Default labels | Multi-table joins + heavy feature engineering |
| **Give Me Some Credit (Kaggle)** ⭐⭐⭐ | 150K loans | 90-day delinquency | Entry-level / quick experiments |
| **German Credit Dataset (UCI)** ⭐⭐⭐ | 1000 loans | Default labels | Classic teaching set, too small — sanity check only |
| **Taiwan Credit Card Default (UCI)** ⭐⭐⭐ | 30K credit-card customers | Next-month default | Credit card supplement |
| **PKDD'99 Financial Dataset** ⭐⭐ | 8-table bank data | Loan status | Relational modeling practice |
| **FICO Explainable ML Challenge** ⭐⭐⭐⭐ | HELOC (anonymized) | Default labels | **XAI benchmark dataset** |

**Key choices**:
- **Main training set**: LendingClub (large + multi-year, good for time-drift analysis)
- **Advanced training set**: Home Credit (multi-table joins, harder feature engineering)
- **XAI dedicated**: FICO HELOC (industry XAI benchmark)
- **Quick validation**: Give Me Some Credit

**Protected-attribute acquisition (required for fairness analysis)**:
- LendingClub includes `addr_state` (region, proxy for race composition), `emp_length`
- Gender inferred from `emp_title` text (compliance-sensitive, used only for bias detection)
- Home Credit directly provides `CODE_GENDER`, `DAYS_BIRTH`

---

## 3. Feasibility Assessment

### 3.1 Technical Requirements

- **Must-have**: Python ≥ 3.10, pandas, numpy, scikit-learn, XGBoost/LightGBM
- **Bank feature engineering**: optbinning, feature-engine, category_encoders
- **Deep learning**: PyTorch ≥ 2.0, pytorch-tabular
- **XAI**: shap, lime, dice-ml, interpret
- **Fairness**: fairlearn, aif360
- **LLM**: transformers (local) + openai / anthropic API
- **Deployment**: Gradio + HuggingFace Spaces / Streamlit
- **Experiment tracking**: MLflow or Weights & Biases

### 3.2 Data Requirements

- **Download**: LendingClub ~2GB, Home Credit ~700MB, others <100MB
- **Storage**: 50GB EBS recommended (sufficient)
- **Cleaning**: LendingClub has heavy missingness; Home Credit 7-table joins need aggregate features
- **Feature engineering**: hand-crafted (WOE, aggregates, ratios) → automated (featuretools)

### 3.3 Model Routes (see Section 4)

| Route | Baseline? | Final? |
|---|---|---|
| Bank Scorecard (LR + WOE) | ✅ | ✅ (regulatory alignment) |
| Logistic Regression | ✅ | baseline |
| Random Forest | ✅ | baseline |
| **XGBoost / LightGBM / CatBoost** | ✅ | ✅ (main pick) |
| Stacking / Blending | ❌ | ✅ (performance ceiling) |
| TabNet | ❌ | ✅ (DL comparison) |
| FT-Transformer | ❌ | ✅ (DL comparison) |
| EBM (InterpretML) | ❌ | ✅ (high accuracy + high interpretability) |

### 3.4 Compute Assessment (AWS T4, 16GB VRAM)

| Task | Estimated time | Notes |
|---|---|---|
| Data cleaning + EDA | CPU 4-8 h | One-off |
| Scorecard (LR + WOE) | CPU 30 min | T4 not needed |
| XGBoost/LightGBM full | CPU/GPU 1-3 h | GPU 5-10× speedup |
| Hyperparam search (Optuna 100 trials) | T4 8-16 h | Main compute |
| TabNet training | T4 4-8 h | batch=2048 |
| FT-Transformer training | T4 6-12 h | batch=512 |
| SHAP on full dataset | CPU 1-4 h | TreeSHAP accelerated |
| LLM report generation (local 7B) | T4 INT4 ~2s/report | Or API |

**T4 total cost**:
- GPU hours: 60-120 (trees + DL + tuning)
- AWS T4 Spot ~$0.16/h; On-Demand ~$0.53/h
- **Budget: Spot $10-20; On-Demand $30-60** (cheaper than OCR project!)
- LLM API (e.g. GPT-4o-mini for reports): 500 × $0.005 = $2.5

### 3.5 Time Cost (10-12 weeks)

Total: 15-20 hours/week, **150-240 hours total**.

### 3.6 Risks and Backup Plans

| Risk | Mitigation |
|---|---|
| LendingClub download source unstable | Kaggle mirror, archive.org backup |
| Poor data quality affects all models | Strict EDA + decisions logged on data card |
| Tree models near ceiling, DL gain small | That's itself a valid finding; report honestly |
| Conflicting fairness metrics | Pick 2-3 primary, acknowledge trade-off |
| LLM report hallucinations | Anchor with SHAP values via RAG, filter low-confidence |
| Home Credit multi-table workload heavy | Finish LendingClub main line first; Home Credit as extension |

---

## 4. Deep Exploration Route (6 Stages)

### Stage 1: Data Prep + EDA + Evaluation Protocol (Week 1-2)

**Goal**: Build stable data pipeline + evaluation framework.

**Tasks**:
- Download full LendingClub 2007-2018
- Cleaning:
  - Label definition ("Charged Off" / "Default" = 1; "Fully Paid" = 0; in-progress dropped)
  - Missingness strategy (>80% missing dropped; 20-80% flagged)
  - **Leakage prevention** (drop features observable only after loan issuance: repayment history, recovery amount, etc.)
- Temporal split:
  - Train: 2014-2016
  - Val: 2017
  - Test: 2018 (**strict out-of-time**, mirrors real deployment)
- Unified evaluation script:
  - **AUC** (core)
  - **KS statistic** (bank-industry standard)
  - **Precision/Recall @ multiple thresholds**
  - **Gini coefficient** (= 2·AUC - 1)
  - **Business metrics**: default rate at fixed approval rate, expected loss
  - **Calibration curve**: predicted vs actual default rate

**Tunable parameters**: label boundary definitions, feature selection thresholds

**Reuse**: All later stages share this data/evaluation

**Evaluation**: Run a simple LR end-to-end; confirm the evaluation script is correct

### Stage 2: Classical Bank Scorecard (Week 2-3)

**Goal**: Establish an "interpretable + regulator-aligned" baseline.

**This step differs from typical student projects — it is what banks actually deploy.**

**Method**:
- **WOE encoding (Weight of Evidence)**: optbinning auto-binning per feature
- **IV filtering (Information Value)**: keep features with IV > 0.1
- **Stepwise regression** + multicollinearity check (VIF)
- **Scorecard transform**: convert LR coefficients to 300-850 credit-score scale
- **PSI monitoring (Population Stability Index)**: detect time drift

**Key techniques**:
- Monotonicity constraints (higher income → lower default — business-sensible)
- Business-rule review (each feature's WOE direction must be interpretable)

**Evaluation**:
- AUC, KS on LendingClub test set
- Business interpretability: each rule writable as "if X and Y, add Z points"
- PSI < 0.1 stable, 0.1-0.25 warning, > 0.25 retrain

**Reuse**: Scorecard is the monitoring benchmark; XAI explanations can be cross-checked against Scorecard coefficients.

### Stage 3: Tree Models + Hyperparameter Optimization (Week 3-5)

**Goal**: Establish strong performance baseline; benchmark deep learning's value.

**Sub-stages**:

**3a. Base tree models**
- Random Forest (baseline)
- XGBoost
- LightGBM
- CatBoost
- Default params vs tuned

**3b. Hyperparameter optimization**
- Optuna 100-200 trials
- Key params: learning_rate, max_depth, num_leaves, min_child_samples, subsample
- Early stopping + 5-fold CV

**3c. Ensembling**
- Stacking: LR/XGBoost/LightGBM/CatBoost → Meta-LR
- Blending: weighted average
- Compare single models vs ensembles

**3d. Class imbalance handling**
- Default ~15-20%, compare:
  - No handling
  - class_weight adjustment
  - scale_pos_weight (XGBoost)
  - SMOTE oversampling
  - Focal Loss

**Evaluation**:
- AUC, KS, PR-AUC per model
- Training time, inference speed
- Feature importance (gain-based)

**Reuse**: Best single model → Stage 5 XAI; best ensemble → Stage 6 deployment.

### Stage 4: Tabular Deep Learning Comparison (Week 5-7)

**Goal**: Honestly evaluate deep learning's value vs tree models.

**Known result** (Yandex NeurIPS 2021): on most tabular data, **XGBoost still wins**. But deep learning has advantages in high-dim sparse features, embedding needs, or multimodal fusion.

**Sub-experiments**:

**4a. TabNet**
- Use pytorch_tabnet
- Strength: built-in attention-based feature selection, inherent interpretability
- Tuning: n_d, n_a, n_steps, gamma

**4b. FT-Transformer**
- Use rtdl-revisiting-models official implementation
- Tuning: token_size, n_blocks, attention_dropout

**4c. NODE / SAINT (optional)**
- Drop if time-constrained

**4d. Industrial-grade EBM (Explainable Boosting Machine)**
- Provided by InterpretML
- Balances accuracy and interpretability — **a strong real-deployment choice**
- Outputs per-feature contribution plots automatically

**4e. Direct comparison with tree models**
- Same data, same eval
- Report: AUC delta, training cost delta, inference speed delta
- **Honest conclusion**: if trees win, state it clearly — this is a valuable negative result

**Comparison table**:

| Model | AUC | KS | Train time | Inference ms | Params | Interpretability |
|---|---|---|---|---|---|---|
| LR + WOE | | | | | | ⭐⭐⭐⭐⭐ |
| XGBoost | | | | | | ⭐⭐⭐ |
| LightGBM | | | | | | ⭐⭐⭐ |
| CatBoost | | | | | | ⭐⭐⭐ |
| Stacking | | | | | | ⭐⭐ |
| TabNet | | | | | | ⭐⭐⭐⭐ |
| FT-Transformer | | | | | | ⭐⭐ |
| EBM | | | | | | ⭐⭐⭐⭐⭐ |

### Stage 5: XAI + Fairness Core (Week 7-9)

**Goal**: Transform the model from "black-box predictor" to "auditable, explainable, regulatable system". **This is the project's biggest highlight.**

**5a. Global explanations**
- Feature importance (XGBoost gain / Permutation importance)
- SHAP summary plot
- Partial Dependence Plots (PDP) / ICE plots
- Feature interaction analysis

**5b. Local explanations (single decisions)**
- SHAP force plot: per-prediction feature contributions
- LIME: local linear approximation
- Compare SHAP vs LIME stability

**5c. Counterfactual explanations (DiCE)** ⭐
- "If annual income were +$10K, would the decision change?"
- Provide **actionable improvement advice** for rejected applicants
- Crucial narrative for end users

**5d. XAI evaluation metrics**
- Faithfulness (does the explanation reflect the model?)
- Stability (similar inputs → similar explanations?)
- Comprehensibility (human-readable?)

**5e. Fairness audit**

Candidate protected attributes:
- Gender (Home Credit direct; LendingClub via `emp_title` inference or use Home Credit subproject)
- Age band
- Income band
- Region (proxy for race)

Fairness metrics (via fairlearn):
- **Demographic Parity Difference (DPD)**: approval-rate gap across groups
- **Equal Opportunity Difference (EOD)**: TPR gap across groups
- **Equalized Odds**: TPR + FPR both balanced
- **Disparate Impact Ratio**: < 0.8 triggers regulatory red line (US "80% rule")

**5f. Fairness mitigation**
- Reweighing (pre-training)
- Adversarial Debiasing (in-training)
- Threshold Optimizer (post-training)
- Compare "accuracy-fairness" trade-off curve

**Evaluation**:
- Fairness metric changes (before vs after mitigation)
- AUC loss (cost of fairness)
- Business impact (approval-rate change)

**Reuse**: Stage 6 LLM report needs SHAP values; fairness results go into report's "compliance check" field.

### Stage 6: LLM Approval Report + Demo + Deployment (Week 9-12)

**Goal**: Wrap technical results into a business-ready system.

**6a. LLM approval report module** ⭐ (RAG portfolio linkage)

**Architecture**:
```
Model prediction (probability + SHAP)
   ↓
[Structured template] Extract top-3 risk drivers + counterfactual advice
   ↓
[LLM Prompt] "You are a credit analyst; generate an approval opinion..."
   ↓
[RAG retrieval] Policy docs, regulations, industry benchmarks (reuse existing RAG project)
   ↓
[Output] Natural-language approval report + policy citations
```

**Comparison**:
- Local Qwen2.5-7B-Instruct (T4 INT4)
- GPT-4o-mini API
- Claude 3.5 Haiku API
- Evaluate: factual accuracy (faithful to SHAP), readability (human score), regulatory-language compliance, cost

**6b. Gradio Web Demo**
- Upload CSV or fill form → output:
  - Default probability + credit score
  - Decision (approve / reject / manual review)
  - SHAP visualization (waterfall + force plot)
  - Counterfactual suggestion ("if... you would qualify")
  - Fairness audit label
  - **LLM-generated approval report** (bilingual EN/CN)
- Audit log on backend

**6c. Model monitoring dashboard**
- PSI time drift
- Real-time group-level fairness metrics
- Performance-decay warning

**6d. Deployment optimization**
- ONNX export + quantization
- Inference speed: target <50ms/application

**6e. Technical report + GitHub**
- Report (15-20 pages)
  - Includes regulatory chapter, fairness trade-off analysis, Model Card
- GitHub repo
- Data Card + Model Card
- 5-minute demo video

---

## 5. Experiment Design

### 5.1 Main Model Comparison Table (Required)

| Model | LendingClub Test AUC | LendingClub Test KS | Home Credit AUC | Inference ms | Training GPU·h |
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

### 5.2 XAI Evaluation Comparison

| Method | Faithfulness | Stability | Human score (1-5) | Compute time ms |
|---|---|---|---|---|
| SHAP (TreeSHAP) | | | | |
| LIME | | | | |
| DiCE (counterfactual) | | | | |
| EBM native | | | | |
| TabNet native attention | | | | |

### 5.3 Fairness Audit Table

| Model | DPD | EOD | DIR | AUC loss | Approval-rate change |
|---|---|---|---|---|---|
| XGBoost vanilla | | | | - | - |
| XGBoost + Reweighing | | | | | |
| XGBoost + Adversarial | | | | | |
| XGBoost + Threshold Optimizer | | | | | |

### 5.4 LLM Report Evaluation

| Model | Accuracy (numeric recall error rate) | Readability (1-5) | Compliance language score | Cost / report |
|---|---|---|---|---|
| Qwen2.5-7B local | | | | $0 |
| GPT-4o-mini | | | | $0.005 |
| Claude 3.5 Haiku | | | | $0.003 |

### 5.5 Ablations (At Least 4)

1. **WOE encoding effect**: raw features vs WOE encoded (on LR and XGBoost)
2. **Class-imbalance handling**: 4 strategies comparison
3. **Time drift**: train 2014-2016, test 2017 vs 2018
4. **Fairness trade-off**: accuracy-fairness Pareto frontier

### 5.6 Robustness Tests

- Feature missing (random mask 10%/30%/50%) prediction stability
- Input noise (numeric ±10% perturbation)
- Different sub-populations (age, income bands) stability

---

## 6. Project Milestones

| Week | Goal | Output |
|---|---|---|
| **Week 1** | Data download + EDA + literature review | LendingClub data card, EDA notebook |
| **Week 2** | Evaluation protocol + data cleaning pipeline + LR baseline | Eval script, cleaning code, LR runs end-to-end |
| **Week 3** | Scorecard (WOE + IV + monotonicity) | Stage 2 complete baseline |
| **Week 4** | XGBoost / LightGBM / CatBoost training | Stage 3a results |
| **Week 5** | Hyperparam tuning + Stacking + imbalance handling | Stage 3 full comparison |
| **Week 6** | TabNet + FT-Transformer + EBM | Stage 4 DL comparison |
| **Week 7** | SHAP / LIME / DiCE explanations | Stage 5a-d XAI report |
| **Week 8** | Fairness audit + mitigation | Stage 5e/5f fairness report |
| **Week 9** | LLM report module + three-model comparison | Stage 6a done |
| **Week 10** | Gradio Demo + monitoring dashboard | Stage 6b/6c done |
| **Week 11** | Deployment optimization + robustness + error analysis | Full robustness report |
| **Week 12** | Report + GitHub + video + Model Card | All deliverables |

**Elastic strategy**:
- Drop 4c (NODE/SAINT) → save 0.5 week
- Drop 6c (monitoring dashboard) → save 0.5 week
- Second dataset (Home Credit) → stretch goal

---

## 7. Final Deliverables

1. **GitHub repository**:
   - `README.md` (intro, Model Card, Data Card)
   - `environment.yml`
   - `notebooks/01_eda.ipynb`, `02_scorecard.ipynb`, `03_models.ipynb`, `04_xai.ipynb`, `05_fairness.ipynb`, `06_llm_report.ipynb`
   - `src/` (modular code)
   - `configs/` (all experiment YAMLs)
   - `results/` (experiment results, plots, weights)
   - `tests/` (pytest unit tests, demonstrates engineering rigor)

2. **Gradio Web Demo** (HuggingFace Spaces deployment)
   - Real-time approval workflow demo
   - SHAP visualization
   - LLM report generation

3. **Technical report** (15-20 page PDF)
   - Includes regulatory chapter, fairness trade-off analysis

4. **Model Card + Data Card** (per Mitchell et al. 2019)
   - Model scope, limitations, bias risk

5. **Demo video** (5 minutes)

6. **Slides** (final presentation)

---

## 8. Portfolio Three-Project Linkage Narrative (Important)

> **Full data-modality coverage:**
> - **RAG project**: handles bank policies, compliance manuals — **unstructured text**
> - **Financial document KIE**: handles receipts, invoices — **semi-structured image/document**
> - **Credit scoring + XAI**: handles customer features, transactions — **structured tabular**

> **Full business closed-loop:**
> - **KIE project**: application materials digitalization (input processing)
> - **Credit scoring project**: risk decision + LLM approval report (core decision)
> - **RAG project**: policy retrieval, customer Q&A (knowledge service)

> **Full technical depth coverage:**
> - **Multimodal pretraining** (LayoutLMv3 in KIE)
> - **Classical ML + Deep Learning comparison** (this project)
> - **Explainable AI + Fairness + LLM** (this project's core highlight + RAG linkage)

**One-line summary for resume/interview**:
> "I systematically built a bank-AI trifecta: RAG for knowledge retrieval, KIE for document automation, and an explainable credit scoring model for risk decisions, with LLM-generated approval reports closing the business loop."

---

## 9. Industry Narrative Points (For Report / Resume / Interview)

1. **Regulatory alignment**: Scorecard + WOE + PSI are what banks actually use (vs "AUC-grinding" student projects)
2. **Explainability is a scarce skill**: EU AI Act and Basel III both require interpretable models — speak the regulator's language
3. **Fairness compliance**: engineered implementation of the 80% DIR rule, ECOA anti-discrimination
4. **Decision traceability**: every approval has full SHAP explanation + LLM report + audit log
5. **Business value**: approval rate, default rate, actionable improvement advice (not just AUC)
6. **Cross-stack integration**: classical ML + deep learning + LLM, three stacks collaborating

---

## 10. Key Risks and Honest Notes

- ⚠️ **Honestly evaluate "trees vs deep learning"**: on tabular data XGBoost will likely still win — that's a valuable finding, don't force DL to win for narrative
- ⚠️ **Fairness metrics conflict**: DPD, EOD, Calibration cannot be simultaneously satisfied (Kleinberg 2017 impossibility theorem); must choose and explain
- ⚠️ **LendingClub has time drift**: 2014 model degrades on 2018 — that's expected, attribute correctly
- ⚠️ **LLM hallucinations**: must anchor to SHAP values, do not let the LLM invent numbers
- ⚠️ **Fairness mitigation costs accuracy**: report Pareto frontier honestly
- ⚠️ **Feature data leakage traps**: many LendingClub fields are post-issuance; must be strictly excluded
- ⚠️ **Sample selection bias**: training data contains only approved loans; rejected applicants are missing — survivor bias in default prediction

---

## Appendix: Reference Links

### Datasets
- LendingClub: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Home Credit: https://www.kaggle.com/competitions/home-credit-default-risk
- Give Me Some Credit: https://www.kaggle.com/competitions/GiveMeSomeCredit
- FICO HELOC: https://community.fico.com/s/explainable-machine-learning-challenge

### Core tools
- optbinning: https://github.com/guillermo-navas-palencia/optbinning
- SHAP: https://github.com/shap/shap
- DiCE: https://github.com/interpretml/DiCE
- Fairlearn: https://github.com/fairlearn/fairlearn
- InterpretML (EBM): https://github.com/interpretml/interpret
- pytorch-tabular: https://github.com/manujosephv/pytorch_tabular

### Recommended reading
- "Fairness and Machine Learning" book: https://fairmlbook.org/
- Google Model Cards: https://modelcards.withgoogle.com/about
- EU AI Act credit-scoring section: https://artificialintelligenceact.eu/
