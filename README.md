<div align="center">

# 📉 Telco Customer Churn Prediction
### Multi-Algorithm Classification Showdown: KNN · Naive Bayes · SVM · Decision Tree

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-0A7E8C)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Best Recall](https://img.shields.io/badge/Churn%20Recall-78.3%25-brightgreen)
![Status](https://img.shields.io/badge/status-complete-success)

*Predict which telecom subscribers are about to leave, and decide which model the retention team should deploy.*

</div>

---

## 📌 Table of Contents
1. [Business Problem](#-business-problem)
2. [Dataset](#-dataset)
3. [Approach](#-approach)
4. [Results](#-results)
5. [Key Insights & Recommendations](#-key-insights--recommendations)
6. [Project Structure](#-project-structure)
7. [Getting Started](#-getting-started)
8. [Using the Saved Model](#-using-the-saved-model)
9. [Design Decisions](#-design-decisions)
10. [Future Work](#-future-work)

---

## 🎯 Business Problem
A telecom company (think Jio or Airtel) loses paying subscribers every month and cannot predict who is about to leave. Winning a new customer (**CAC**) costs far more than keeping an existing one, and every churner takes their future revenue (**CLV**) with them.

This project builds and compares four supervised classifiers and recommends one for production.

> **Why Recall?** A **False Negative** (a churner we missed) is a silently lost customer. A **False Positive** is only a wasted retention call. So the primary metric is **Recall on the churn class**, while keeping Precision in check because the retention team can only call ~500 customers a day.

## 📂 Dataset
| | |
|---|---|
| **Name** | IBM Telco Customer Churn |
| **Source** | [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Size** | 7,043 customers × 21 columns |
| **Target** | `Churn` (Yes = 1, No = 0), about **26.5%** churn (moderately imbalanced) |
| **Features** | Demographics, services, contract, billing |

## 🛠 Approach

```
Raw CSV
  │
  ├─ 1. Problem framing & theory notes (CAC vs CLV, confusion-matrix costs, SMOTE)
  ├─ 2. EDA: univariate, bivariate, correlation, high-risk segment check
  ├─ 3. Preprocessing & feature engineering
  │      • drop customerID, median-fill 11 blank TotalCharges (new customers, tenure = 0)
  │      • tenure_group (New / Mid / Senior / Loyal), num_services, AutoPay
  │      • binary mapping + one-hot encoding
  │      • stratified 80/20 split → StandardScaler (fit on train only) → SMOTE (train only)
  ├─ 4. KNN (tune k) & Gaussian Naive Bayes
  ├─ 5. SVM (tune C) & Decision Tree (tune depth) + class_weight vs SMOTE
  ├─ 6. Comparison: metrics table, ROC curves, Precision-vs-Recall chart
  ├─ 7. Error analysis on False Negatives + feature importance
  └─ 8. Single sklearn Pipeline → churn_model.pkl
```

## 📊 Results
Evaluated on a held-out, **unaugmented** test set (1,409 customers), churn class metrics, sorted by Recall:

| Model | Accuracy | Precision | **Recall** | F1 | AUC-ROC | Train time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive Bayes | 0.738 | 0.504 | **0.789** | 0.615 | 0.819 | 0.004 s |
| 🏆 **Decision Tree** (depth 3, balanced) | 0.746 | 0.514 | **0.783** | **0.621** | 0.816 | 0.010 s |
| KNN (k = 15) | 0.786 | 0.617 | 0.513 | 0.561 | 0.826 | 0.004 s |
| SVM (RBF, C = 10) | 0.784 | 0.617 | 0.492 | 0.548 | 0.810 | 3.5 s |

**🏆 Recommended model: Decision Tree**
- Recall within 0.5 pt of Naive Bayes, but with better Precision, F1 and accuracy.
- Naive Bayes outputs near 0/1 probabilities, so it can't rank customers for a capacity-limited team.
- KNN and SVM miss about half of all churners, which is unacceptable when False Negatives cost more.
- Fully interpretable rules and millisecond training and scoring.

**`class_weight='balanced'` vs SMOTE** (Decision Tree): identical churn Recall (0.783), but `class_weight` has the better AUC (0.816 vs 0.807) and uses no synthetic data, so it was chosen.

## 💡 Key Insights & Recommendations

| # | Churn signal | Evidence | Suggested action |
|:-:|---|---|---|
| 1 | **Month-to-month contract** | 42.7% churn vs 2.8% on 2-year (≈15×) | Discounts and perks for 1–2 year contracts |
| 2 | **Fiber-optic internet** | 41.9% churn vs 19.0% on DSL | Review pricing and service quality |
| 3 | **Short tenure (≤ 12 months)** | 47.4% churn vs 17.1% after | Onboarding and early-life care programme |

- 🎯 **High-risk segment:** month-to-month + above-median bill + tenure ≤ 12 months → **812 customers with 70% churn** (vs 26.5% overall).
- 🔍 **Error analysis:** the 81 missed churners are *loyal, longer-contract* customers (avg tenure 33.8 vs 16.5 months; only 44% month-to-month vs 88%). The model finds the classic churner well but needs extra signals (usage drops, complaints, contract expiry) to catch these.

## 📁 Project Structure
```
telco-churn-supervised-learning/
├── CustomerChurn_SupervisedLearning.ipynb   # fully executed notebook (Steps 1–8)
├── churn_model.pkl                          # saved end-to-end pipeline
├── telco_churn_processed.csv                # cleaned, encoded dataset (unscaled)
├── summary_report.md                        # 400–500 word business summary
├── requirements.txt
└── README.md
```
> `churn_model.pkl`, `summary_report.md` and `requirements.txt` are generated when you run the notebook.

## 🚀 Getting Started
```bash
git clone https://github.com/<your-username>/telco-churn-supervised-learning.git
cd telco-churn-supervised-learning
pip install -r requirements.txt

# Download WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle and place it next to the notebook
jupyter notebook CustomerChurn_SupervisedLearning.ipynb   # Run All
```
If the CSV is missing, the notebook automatically falls back to a public mirror of the same IBM dataset.

## 🧪 Using the Saved Model
The pipeline takes a **raw customer row** (same columns as the original CSV) and returns churn probability. Feature engineering, scaling and the model are all inside.

```python
import joblib, pandas as pd

# engineer_raw must be defined first: run the notebook's Step 3 cell,
# or copy that function into your script, because the pipeline references it.
pipe = joblib.load("churn_model.pkl")

customers = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv").head(5)
proba = pipe.predict_proba(customers)[:, 1]
print(pd.DataFrame({"customerID": customers["customerID"], "churn_probability": proba.round(3)}))
```

## 🧠 Design Decisions
- **No data leakage:** the split comes first; the scaler and SMOTE are fit on training data only, and the test set stays original.
- **Honest tuning:** KNN's k is tuned on a validation split from the training set, SVM's C and the tree's depth by cross-validation, and the test set is touched only for final evaluation.
- **One source of truth:** all preprocessing lives in a single `engineer_raw()` function, reused in the deployable pipeline, so the notebook and production can't drift apart.
- **Metric aligned to cost:** the model is chosen on churn Recall with Precision as a guardrail, not on accuracy (always predicting "No" already gives ~73.5%).

## 🔭 Future Work
- Richer features: usage trends, complaints, support-call history, contract-expiry dates
- Ensembles (Random Forest, XGBoost/LightGBM) with probability calibration
- Threshold tuning against real retention cost and benefit numbers
- Real-time scoring API (FastAPI) with drift monitoring

---

<div align="center">

**Built with ❤️ for the Red & White Skill Education practical exam (Supervised Learning, Set B)**

⭐ If you found this useful, consider starring the repo!

</div>
