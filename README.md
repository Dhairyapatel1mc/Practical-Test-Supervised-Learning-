<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24,30&height=220&section=header&text=📡%20Customer%20Churn%20Predictor&fontSize=46&fontColor=ffffff&fontAlignY=38&desc=Supervised%20Learning%20Showdown%20%7C%20KNN%20·%20Naive%20Bayes%20·%20SVM%20·%20Decision%20Tree&descAlignY=58&descSize=17&animation=fadeIn" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=20&pause=1000&color=00D4AA&center=true&vCenter=true&multiline=true&width=900&height=80&lines=📡+Predicting+Which+Telecom+Subscribers+Will+Leave;🔵+KNN+%7C+🎲+Naive+Bayes+%7C+🧭+SVM+%7C+🌳+Decision+Tree;🎯+Recall-First+Selection+%7C+Deployed%3A+Decision+Tree+%3D+83.7%25+Recall" alt="Typing SVG"/>
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/imbalanced--learn-SMOTE-0A7E8C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-Math-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-✅%20Completed-2ECC71?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Deployed-Decision%20Tree-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Churn%20Recall-83.7%25-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Customers-7%2C043%20Records-00D4AA?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-orange?style=for-the-badge"/>
</p>

<br/>

<blockquote>
<i>📡 A telecom retention pipeline — four supervised classifiers tuned and compared head-to-head on churn Recall, with a leak-free SMOTE workflow and a business-ready recommendation.</i>
</blockquote>

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/GitHub-@Dhairyapatel1mc-181717?style=flat-square&logo=github&logoColor=white"/>
</a>
&nbsp;&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/LinkedIn-ghost--patel-0077B5?style=flat-square&logo=linkedin&logoColor=white"/>
</a>
&nbsp;&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Instagram-@ghost__6927-E4405F?style=flat-square&logo=instagram&logoColor=white"/>
</a>

<br/><br/>

</div>

---

<a name="toc"></a>
## 📋 Table of Contents

<div align="center">

| # | Section | Jump |
|:---:|---------|:---:|
| 01 | 🎯 Project Overview | [→](#overview) |
| 02 | 📊 Dataset at a Glance | [→](#dataset) |
| 03 | 🗂️ Dataset Schema | [→](#schema) |
| 04 | 🔍 EDA Findings | [→](#eda) |
| 05 | 🧠 Theory — Key Concepts | [→](#theory) |
| 06 | ⚙️ ML Pipeline Architecture | [→](#pipeline) |
| 07 | 🤖 Model Results | [→](#models) |
| 08 | ⚖️ SMOTE vs Class Weights | [→](#smote) |
| 09 | 🏆 Performance Leaderboard | [→](#leaderboard) |
| 10 | 🔬 Error Analysis & Feature Importance | [→](#errors) |
| 11 | 📈 Visualizations | [→](#charts) |
| 12 | 📁 Project Structure | [→](#structure) |
| 13 | 🚀 Quick Start | [→](#quickstart) |
| 14 | 📋 Final Report | [→](#report) |
| 15 | 👤 Author & Contact | [→](#author) |

</div>

---

<a name="overview"></a>
## 🎯 Project Overview

<table>
<tr>
<td width="60%">

You are a **Junior Data Scientist** at a Jio/Airtel-style telecom. Subscribers leave every month and nobody can say who is next. Winning a new customer (**CAC**) costs far more than keeping one, and every churner takes their future revenue (**CLV**) with them.

**Task:** 🎯 predict `Churn` (Yes = 1 / No = 0) and recommend one model for the retention team.

**What this pipeline delivers:**
- ✅ Full EDA with a high-risk segment check
- ✅ Leak-free preprocessing: split → scale → SMOTE (train only)
- ✅ 4 classifiers: KNN, Gaussian Naive Bayes, RBF-SVM, Decision Tree
- ✅ Hyper-parameter tuning for k, C and tree depth
- ✅ SMOTE vs `class_weight='balanced'` comparison
- ✅ Recall-first model selection, error analysis and a saved sklearn pipeline

> **Why Recall?** A **False Negative** (a churner we missed) is a silently lost customer. A **False Positive** is only a wasted retention call. Recall on the churn class leads; Precision is kept in check because the team can only call ~500 customers a day.

</td>
<td width="40%" align="center">

| Field | Value |
|-------|-------|
| 📡 Project | Customer Churn Predictor |
| 🏢 Domain | Telecom / Retention Analytics |
| 🎯 Task | Binary Classification |
| 🏆 Deployed | **Decision Tree (depth 4)** |
| 📈 Churn Recall | **0.8374** |
| 🎯 AUC-ROC | **0.8566** |
| 👥 Dataset | 7,043 customers |
| ⏱️ Duration | 6 Hours |

</td>
</tr>
</table>

[⬆ Back to top](#toc)

---

<a name="dataset"></a>
## 📊 Dataset at a Glance

<div align="center">

| 👥 Customers | 🔢 Columns | ❌ Churned | ✅ Retained | ⚖️ Imbalance |
|:---:|:---:|:---:|:---:|:---:|
| **7,043** | **21** | **1,842 (26.15%)** | **5,201 (73.85%)** | **2.82 : 1** |

</div>

```
CHURN DISTRIBUTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retained (0): ████████████████████████████████ 73.85% (5,201)
Churned  (1): ███████████                       26.15% (1,842)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train: 5,634  |  Test: 1,409  |  Split: 80/20 stratified
Train after SMOTE: 4,161 vs 4,161  |  Test untouched: 1,040 vs 369
```

> ℹ️ **Data note:** the Kaggle download was not reachable while building the notebook, so `WA_Fn-UseC_-Telco-Customer-Churn.csv` was regenerated with the **exact IBM Telco schema** (7,043 × 21, 11 blank `TotalCharges` rows where `tenure = 0`). To run on the original data, drop the Kaggle file in place — the code is unchanged.

[⬆ Back to top](#toc)

---

<a name="schema"></a>
## 🗂️ Dataset Schema

| Group | Columns | Notes |
|-------|---------|-------|
| 👤 Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Yes/No → 1/0 |
| 📞 Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | "No internet/phone service" → 0 |
| 📝 Account | `Contract`, `PaperlessBilling`, `PaymentMethod` | One-hot encoded |
| 💰 Billing | `tenure`, `MonthlyCharges`, `TotalCharges` | 11 blanks → median 1,312.28 |
| 🎯 **Target** | `Churn` | **1 = left, 0 = stayed** |

**Engineered features**

| Feature | Definition |
|---------|------------|
| `tenure_group` | New (≤12) · Mid (13–36) · Senior (37–60) · Loyal (61–72), ordinal-encoded |
| `num_services` | Count of add-on services subscribed (0–6) |
| `AutoPay` | 1 if payment method is automatic (43% of customers) |

`customerID` is dropped → **29 model features** after encoding.

[⬆ Back to top](#toc)

---

<a name="eda"></a>
## 🔍 EDA Findings

| Insight | Number |
|---------|:------:|
| Churn — **Month-to-month** contract | **41.94%** |
| Churn — One-year contract | 10.83% |
| Churn — Two-year contract | 3.21% |
| Churn — tenure 0–12 months | **46.83%** |
| Churn — tenure 49–72 months | 4.97% |
| Avg monthly charge — churners vs stayers | 83.96 vs 61.71 |
| `corr(tenure, TotalCharges)` | 0.80 |

```
HIGH-RISK SEGMENT: Month-to-month + MonthlyCharges > median + tenure ≤ 12
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1,003 customers  →  71.0% churn
Rest of the base →  18.7% churn   (overall 26.2%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

<div align="center">
  <img src="charts/03_churn_by_segment.png" width="90%" alt="Churn by contract and tenure"/>
</div>

[⬆ Back to top](#toc)

---

<a name="theory"></a>
## 🧠 Theory — Key Concepts

<details>
<summary><b>🔷 Q1 — Why is churn prediction valuable? (CAC vs CLV)</b></summary>

<br>

> **Churn** is a paying subscriber leaving the service. **CAC** (acquisition cost of a *new* subscriber) is typically several times the cost of *retaining* one, and **CLV** (lifetime value) is lost the moment a customer leaves. Predicting churn lets a limited retention budget go to high-risk customers only.

</details>

<details>
<summary><b>🔷 Q2 — Confusion matrix & the cost of errors</b></summary>

<br>

```
                      Predicted: Stay      Predicted: Churn
Actual: Stay        TN  (fine)             FP  (wasted offer)
Actual: Churn       FN  (LOST CUSTOMER)    TP  (saved by call)
```

$$\text{Precision}=\frac{TP}{TP+FP}\qquad \text{Recall}=\frac{TP}{TP+FN}\qquad F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

> FN costs the customer's whole CLV; FP costs one call or coupon. **Recall on the churn class is the primary metric.**

</details>

<details>
<summary><b>🔷 Q3 — The four algorithms at a glance</b></summary>

<br>

| Algorithm | Core idea | Strength | Weakness |
|-----------|-----------|----------|----------|
| **KNN** | Vote of the *k* nearest customers | No training, simple | Slow at scoring, scale-sensitive |
| **Gaussian NB** | Bayes' rule + feature independence | Very fast, high recall | Independence assumption broken here |
| **RBF-SVM** | Max-margin boundary in a kernel space | Captures non-linear patterns | Black box, slowest (~9 s) |
| **Decision Tree** | Axis-aligned if/else splits | Fully interpretable | Overfits when deep |

$$P(\text{churn}\mid x)\;\propto\;P(\text{churn})\prod_{j}P(x_j\mid \text{churn})$$

> **Independence check:** `tenure`, `MonthlyCharges`, `TotalCharges` and `num_services` are clearly correlated (TotalCharges ≈ tenure × MonthlyCharges), so NB is over-confident — yet it still ranks customers well.

</details>

<details>
<summary><b>🔷 Q4 — SMOTE and why the order matters</b></summary>

<br>

> **SMOTE** creates *synthetic* minority samples by interpolating between neighbouring churners, balancing the training set.

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])      # fit on TRAIN only
X_sm, y_sm = SMOTE(random_state=RS).fit_resample(X_tr, y_tr)  # TRAIN only
```

```
LEAK-FREE ORDER:  split  →  scale (fit on train)  →  SMOTE (train only)
Test set stays original and unaugmented.
```

</details>

[⬆ Back to top](#toc)

---

<a name="pipeline"></a>
## ⚙️ ML Pipeline Architecture

```
╔════════════════════════════════════════════════════════════════════════╗
║                CUSTOMER CHURN — SUPERVISED PIPELINE                    ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ┌─────────────┐   ┌──────────────┐   ┌───────────────────────────┐   ║
║  │ Telco CSV   │ → │ EDA &        │ → │ Clean + Feature Eng +     │   ║
║  │ 7,043 × 21  │   │ Segment Check│   │ Encoding (29 features)    │   ║
║  └─────────────┘   └──────────────┘   └─────────────┬─────────────┘   ║
║                                                     │                  ║
║                 ┌───────────────────────────────────▼───────────────┐  ║
║                 │ Stratified 80/20 split → Scaler (train) → SMOTE   │  ║
║                 └───────────────────────┬───────────────────────────┘  ║
║                                         │                              ║
║  ┌─────────────┬──────────────┬─────────┴─────┬─────────────────────┐  ║
║  │ 🔵 KNN      │ 🎲 NAIVE     │ 🧭 SVM        │ 🌳 DECISION TREE    │  ║
║  │ tune k      │   BAYES      │ tune C        │ tune max_depth      │  ║
║  │ (SMOTE)     │ (SMOTE)      │ (SMOTE in CV) │ (class_weight)      │  ║
║  └──────┬──────┴──────┬───────┴───────┬───────┴──────────┬──────────┘  ║
║         │             │               │                  │             ║
║  ┌──────▼─────────────▼───────────────▼──────────────────▼──────────┐  ║
║  │ Recall-first comparison → Error analysis → churn_model.pkl       │  ║
║  └──────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════════════╝
```

[⬆ Back to top](#toc)

---

<a name="models"></a>
## 🤖 Model Results

### 🔵 KNN — tune *k* on a validation split

| | Accuracy | Precision | Recall | F1 | AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| k = 5 (baseline) | 0.7495 | 0.5144 | 0.7751 | 0.6184 | 0.8138 |
| **k = 15 (tuned)** | 0.7459 | 0.5089 | **0.8482** | 0.6362 | 0.8447 |

```
VALIDATION F1 vs k:  1→0.517  3→0.584  5→0.602  7→0.608  9→0.611  11→0.610  15→0.631 ← best
```

### 🎲 Gaussian Naive Bayes

| Accuracy | Precision | Recall | F1 | AUC |
|:---:|:---:|:---:|:---:|:---:|
| 0.7672 | 0.5358 | 0.8320 | 0.6518 | 0.8600 |

> Trained on SMOTE data → priors forced to 0.5 / 0.5 (true prior is 0.739 / 0.261).

### 🧭 RBF-SVM — tune *C* with 3-fold CV (SMOTE inside each fold)

| | Accuracy | Precision | Recall | F1 | AUC | Train time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| C = 1 (baseline) | 0.7963 | 0.5876 | 0.7453 | 0.6571 | 0.8535 | 8.19 s |
| **C = 0.1 (tuned)** | 0.7700 | 0.5404 | **0.8157** | 0.6501 | 0.8659 | 9.42 s |

```
CV F1 vs C:  0.1→0.661 ← best   1→0.658   10→0.630   100→0.555   (high C overfits)
```

### 🌳 Decision Tree — tune `max_depth` with 5-fold CV

| | Accuracy | Precision | Recall | F1 | AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| depth 5 (baseline) | 0.7466 | 0.5094 | 0.8780 | 0.6448 | 0.8587 |
| **depth 4 (tuned)** | 0.7615 | 0.5282 | 0.8374 | **0.6478** | 0.8566 |

```
CV F1 vs depth:  3→0.638  4→0.644 ← best  5→0.640  6→0.632  7→0.619  8→0.627  None→0.507
```

> Unlimited depth collapses to F1 0.507 — a textbook overfit. The **root split is `Contract_Month-to-month`**.

<div align="center">
  <img src="charts/10_decision_tree_structure.png" width="95%" alt="Decision tree structure"/>
</div>

[⬆ Back to top](#toc)

---

<a name="smote"></a>
## ⚖️ SMOTE vs Class Weights

Compared on the Decision Tree (the better of SVM / Tree by recall):

| Imbalance strategy | Recall | Precision | F1 |
|--------------------|:------:|:---------:|:--:|
| SMOTE-augmented data | 0.8320 | 0.5302 | 0.6477 |
| **`class_weight="balanced"`** (original data) | **0.8374** | 0.5282 | 0.6478 |

> Practically a tie (Δ recall = 0.0054). `class_weight` wins on simplicity — no synthetic rows, no extra dependency, and no leakage risk.

[⬆ Back to top](#toc)

---

<a name="leaderboard"></a>
## 🏆 Performance Leaderboard

<div align="center">

| Rank | 🏅 Model | Accuracy | Precision | **Recall** | F1 | AUC-ROC | Train (s) |
|:----:|----------|:--------:|:---------:|:----------:|:--:|:-------:|:---------:|
| 🥇 | **KNN (k=15)** | 0.7459 | 0.5089 | **0.8482** | 0.6362 | 0.8447 | 0.0048 |
| 🥈 | **Decision Tree (d=4)** ⭐ | 0.7615 | 0.5282 | **0.8374** | 0.6478 | 0.8566 | 0.0131 |
| 🥉 | Naive Bayes | 0.7672 | 0.5358 | 0.8320 | 0.6518 | 0.8600 | 0.0097 |
| 4 | SVM (C=0.1) | 0.7700 | 0.5404 | 0.8157 | 0.6501 | 0.8659 | 9.4204 |

</div>

```
CHURN RECALL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNN           ████████████████████████████████████ 0.8482 🏆 best recall
Decision Tree ███████████████████████████████████  0.8374 ⭐ deployed
Naive Bayes   ██████████████████████████████████   0.8320
SVM           ████████████████████████████████     0.8157 (best AUC 0.8659)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

<div align="center">
  <img src="charts/14_model_comparison.png" width="95%" alt="ROC curves and precision vs recall"/>
</div>

**Selection rule:** Recall comes first. Models within **2 recall-points** of the best (84.8%) with Precision ≥ 45% count as equally good at catching churners — among them we deploy the **most interpretable and fastest** one → **Decision Tree**.

[⬆ Back to top](#toc)

---

<a name="errors"></a>
## 🔬 Error Analysis & Feature Importance

### Deployed model on 1,409 test customers

| | Predicted Stay | Predicted Churn |
|---|:---:|:---:|
| **Actual Stay** | 764 (TN) | 276 (FP) |
| **Actual Churn** | **60 (FN)** | **309 (TP)** |

> Flags **585 customers (41.5%)** and catches **309 of 369** real churners.

### Who does the model miss? (60 false negatives)

| Profile | Missed churners | All churners | All customers |
|---------|:---------------:|:------------:|:-------------:|
| Avg tenure (months) | 24.75 | 14.79 | 29.60 |
| Avg MonthlyCharges | 69.46 | 84.20 | 68.65 |
| % Month-to-month | **40.0%** | 90.24% | 56.07% |

> The tree misses churners who **don't look like the typical high-risk profile** — longer tenure, lower bills, and less often on month-to-month contracts.

### Top 5 split features

| # | Feature | Importance | Action |
|:-:|---------|:----------:|--------|
| 1 | `Contract_Month-to-month` | 0.531 | Offer incentives to move to 1/2-year contracts |
| 2 | `InternetService_No` | 0.241 | Phone-only customers — very low-churn segment |
| 3 | `InternetService_DSL` | 0.082 | Lower-risk internet segment |
| 4 | `tenure` | 0.075 | Run a 90-day onboarding / welcome programme |
| 5 | `MonthlyCharges` | 0.041 | Plan right-sizing or loyalty discounts |

<div align="center">
  <img src="charts/13_dt_feature_importance.png" width="55%" alt="Decision tree feature importance"/>
</div>

[⬆ Back to top](#toc)

---

<a name="charts"></a>
## 📈 Visualizations

| # | Chart File | Description |
|:-:|-----------|-------------|
| 01 | `01_eda_univariate.png` | Target balance + tenure / charges histograms |
| 02 | `02_eda_categorical.png` | Contract, internet service and payment counts |
| 03 | `03_churn_by_segment.png` | Churn rate by contract and tenure bucket |
| 04 | `04_charges_correlation.png` | MonthlyCharges boxplot by churn + numeric correlation heatmap |
| 05 | `05_knn_k_tuning.png` | Validation F1 vs *k* |
| 06 | `06_knn_cm_roc.png` | KNN confusion matrix + ROC |
| 07 | `07_naive_bayes_cm_roc.png` | Naive Bayes confusion matrix + ROC |
| 08 | `08_svm_c_tuning.png` | CV F1 vs *C* |
| 09 | `09_svm_cm_roc.png` | SVM confusion matrix + ROC |
| 10 | `10_decision_tree_structure.png` | Fitted tree, root = `Contract_Month-to-month` |
| 11 | `11_dt_depth_tuning.png` | CV F1 vs `max_depth` |
| 12 | `12_dt_cm_roc.png` | Decision Tree confusion matrix + ROC |
| 13 | `13_dt_feature_importance.png` | Top-15 feature importances |
| 14 | `14_model_comparison.png` | ROC overlay + Precision vs Recall bars |

[⬆ Back to top](#toc)

---

<a name="structure"></a>
## 📁 Project Structure

```
📡 Practical-Test-Supervised-Learning-/
│
├── 📓 CustomerChurn_SupervisedLearning.ipynb   ← Pre-executed notebook (all outputs)
├── 📊 WA_Fn-UseC_-Telco-Customer-Churn.csv     ← Dataset (7,043 × 21)
├── 📈 charts/                                  ← 14 exported figures
└── 📄 README.md                                ← This file
```

[⬆ Back to top](#toc)

---

<a name="quickstart"></a>
## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Dhairyapatel1mc/Practical-Test-Supervised-Learning-.git
cd Practical-Test-Supervised-Learning-
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook CustomerChurn_SupervisedLearning.ipynb
```

### 4. Score New Customers

```python
import joblib

pipe = joblib.load('churn_model.pkl')          # StandardScaler + Decision Tree
proba = pipe.predict_proba(X_new)[:, 1]        # X_new = same 29 encoded columns
print(["CHURN" if p > 0.5 else "Retained" for p in proba])
```

[⬆ Back to top](#toc)

---

<a name="report"></a>
## 📋 Final Report

### Recommendations

| Goal | Model | Why |
|------|-------|-----|
| 🎯 Deploy for retention team | **Decision Tree (depth 4)** | 83.7% recall, readable rules, 0.01 s training |
| 🏅 Max churn catch-rate | KNN (k=15) | Best recall 84.8%, but slow scoring and opaque |
| 📈 Best ranking quality | SVM (C=0.1) | Highest AUC 0.8659, but a black box at ~9 s training |

### Business Interpretation

```
CLASSIFICATION INSIGHT:
  The tree catches 83.7% of churners (309 of 369) at 52.8% precision.
  → About 1 in 2 flagged customers truly leaves — acceptable for a cheap offer.
  → Target Month-to-month, high-bill, first-12-month customers first.

WHERE IT FAILS:
  60 missed churners skew longer-tenure and lower-bill.
  → Add loyalty/engagement signals to catch "quiet" churners.
```

### Design Decisions

- **Split before scaling/SMOTE** — the exam lists them earlier, but doing so leaks test information; the order here avoids it.
- **SMOTE inside CV folds** for SVM tuning — validation folds stay untouched.
- **Recall-first, then interpretability** — a missed churner costs more than a wasted offer.

### Future Work

- Extend the KNN grid beyond k = 15 (the best value sits at the grid edge).
- Try ensembles (Random Forest, XGBoost) and threshold tuning against the 500-calls/day budget.
- Re-run on the original Kaggle file to confirm the findings.

[⬆ Back to top](#toc)

---

<a name="author"></a>
## 👤 Author & Contact

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=6,11,20&height=120&section=header&text=Ghost%20(Patel%20Dhairya)&fontSize=30&fontColor=ffffff&desc=B.Tech%20AI%20%7C%20Data%20Science%20%7C%20RWSkill&descSize=14&animation=fadeIn" width="70%"/>

<br/>

| | |
|:---:|---|
| 👤 **Name** | Ghost (Patel Dhairya Rakesh Kumar) |
| 🎓 **Program** | B.Tech Artificial Intelligence — Gandhinagar University |
| 🏫 **Institute** | Red and White Skill Education (RWSkill) |
| 🐙 **GitHub** | [@Dhairyapatel1mc](https://github.com/Dhairyapatel1mc) |
| 💼 **LinkedIn** | [ghost-patel](https://linkedin.com/in/ghost-patel) |
| 📸 **Instagram** | [@ghost_6927](https://instagram.com/ghost_6927) |

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/Follow%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>
&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/Connect%20on%20LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>
&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Follow%20on%20Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>
</a>

</div>

---

## 🛠️ Full Tech Stack

<div align="center">

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **Models** | KNeighborsClassifier, GaussianNB, SVC (RBF), DecisionTreeClassifier |
| **Imbalance** | imbalanced-learn SMOTE, `class_weight='balanced'` |
| **Tuning** | Validation split, `cross_val_score`, imblearn Pipeline |
| **Preprocessing** | StandardScaler, ColumnTransformer, stratified `train_test_split` |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | joblib sklearn Pipeline (`churn_model.pkl`) |
| **Notebook** | Jupyter (pre-executed, all outputs visible) |

</div>

---

<div align="center">

**[⬆ Back to Top](#toc)**

<br/>

```
⭐ If this project helped you, please star the repository!
```

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24,30&height=100&section=footer&animation=fadeIn" width="100%"/>

*Built with precision by Ghost (Patel Dhairya) · Red and White Skill Education (RWSkill) · 2026*

</div>
