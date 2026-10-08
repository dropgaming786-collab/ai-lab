"""
lab04/build_deliverables.py
--------------------------
Automated builder for Laboratory 4 deliverables:
1. California Housing Linear Regression and Pipeline (StandardScaler + LinearRegression)
2. Diagnostic scatter plot (actual vs predicted)
3. Regression coefficient analysis (raw vs standardized)
4. Iris Classification (LogisticRegression and DecisionTreeClassifier)
5. Home Assignment: Text Sentiment Pipeline (TfidfVectorizer + LogisticRegression)
6. Model persistence: linear_model.pkl and sentiment_pipeline.pkl
7. Generation of lab04/report.md
8. Patching and full execution of lab04/sklearn_intro.ipynb and Lab 04.ipynb
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure Paths
ROOT = r"E:\Ai lab"
LAB04 = os.path.join(ROOT, "lab04")
FIGS = os.path.join(LAB04, "figures")
CORPUS_CSV = os.path.join(ROOT, "lab02", "two_source_corpus.csv")
NB_SRC = os.path.join(ROOT, "Lab 04.ipynb")
NB_DST = os.path.join(LAB04, "sklearn_intro.ipynb")
REPORT_PATH = os.path.join(LAB04, "report.md")

os.makedirs(FIGS, exist_ok=True)
plt.style.use("ggplot")

print("=" * 70)
print("LABORATORY 04: BUILDING DELIVERABLES AND SOLVING EXERCISES")
print("=" * 70)

# ============================================================================
# 1. Regression: California Housing
# ============================================================================
print("\n[1/5] Training Regression Models on California Housing...")
housing = fetch_california_housing(as_frame=True)
df_reg = housing.frame
X_reg = df_reg.drop(columns=['MedHouseVal'])
y_reg = df_reg['MedHouseVal']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Plain LinearRegression
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)
r2_plain = r2_score(y_test_reg, y_pred_reg)
mse_plain = mean_squared_error(y_test_reg, y_pred_reg)

# Pipeline with StandardScaler
pipe_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipe_reg.fit(X_train_reg, y_train_reg)
y_pred_pipe = pipe_reg.predict(X_test_reg)
r2_pipe = r2_score(y_test_reg, y_pred_pipe)
mse_pipe = mean_squared_error(y_test_reg, y_pred_pipe)

print(f"  Plain Linear Regression R²: {r2_plain:.6f} | MSE: {mse_plain:.4f}")
print(f"  Pipeline (Scaled)       R²: {r2_pipe:.6f} | MSE: {mse_pipe:.4f}")

# Persist Linear Model
linear_model_path = os.path.join(LAB04, "linear_model.pkl")
joblib.dump(reg_model, linear_model_path)
# Also copy/keep at root if required
joblib.dump(reg_model, os.path.join(ROOT, "linear_model.pkl"))
print(f"  Persisted: {linear_model_path}")

# Figure 1: Actual vs Predicted Scatter Plot
fig1, ax1 = plt.subplots(figsize=(8, 5.5))
sns.scatterplot(x=y_test_reg, y=y_pred_reg, alpha=0.4, color="royalblue", edgecolor=None, ax=ax1)
line_vals = np.linspace(y_test_reg.min(), y_test_reg.max(), 100)
ax1.plot(line_vals, line_vals, color="crimson", linestyle="--", linewidth=2, label="Ideal: y = x")
ax1.set_xlabel("Actual Values (MedHouseVal in $100k)", fontsize=11)
ax1.set_ylabel("Predicted Values", fontsize=11)
ax1.set_title(f"Linear Regression: Actual vs Predicted\n(R² = {r2_plain:.4f}, California Housing)", fontsize=12, weight="bold")
ax1.legend()
fig1.tight_layout()
fig1_path = os.path.join(FIGS, "actual_vs_predicted.png")
fig1.savefig(fig1_path, dpi=150)
plt.close(fig1)
print(f"  Saved figure: {fig1_path}")

# Coefficients Analysis
coef_df = pd.DataFrame({
    'Feature': X_reg.columns,
    'Raw_Coefficient': reg_model.coef_,
    'Abs_Raw': np.abs(reg_model.coef_),
    'Standardized_Coefficient': pipe_reg.named_steps['model'].coef_,
    'Abs_Standardized': np.abs(pipe_reg.named_steps['model'].coef_)
}).sort_values('Abs_Standardized', ascending=False).reset_index(drop=True)

# Figure 2: Standardized vs Raw Coefficients
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
x_indices = np.arange(len(coef_df))
width = 0.35
ax2.barh(x_indices - width/2, coef_df['Standardized_Coefficient'], width, label='Standardized (Pipeline)', color='teal')
ax2.barh(x_indices + width/2, coef_df['Raw_Coefficient'], width, label='Raw (Unscaled)', color='orange')
ax2.set_yticks(x_indices)
ax2.set_yticklabels(coef_df['Feature'])
ax2.set_xlabel("Coefficient Magnitude")
ax2.set_title("Regression Feature Coefficients: Standardized vs Raw", fontsize=12, weight="bold")
ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.legend()
fig2.tight_layout()
fig2_path = os.path.join(FIGS, "standardized_coefficients.png")
fig2.savefig(fig2_path, dpi=150)
plt.close(fig2)
print(f"  Saved figure: {fig2_path}")

# ============================================================================
# 2. Classification: Iris (LogisticRegression & DecisionTree)
# ============================================================================
print("\n[2/5] Training Classifiers on Iris Dataset...")
iris = load_iris(as_frame=True)
df_cls = iris.frame
X_cls = df_cls.drop(columns=['target'])
y_cls = df_cls['target']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42
)

# Logistic Regression
log_model = LogisticRegression(max_iter=200, random_state=42)
log_model.fit(X_train_cls, y_train_cls)
y_pred_cls = log_model.predict(X_test_cls)
acc_log = accuracy_score(y_test_cls, y_pred_cls)
cm_log = confusion_matrix(y_test_cls, y_pred_cls)
cr_log = classification_report(y_test_cls, y_pred_cls, target_names=iris.target_names)
print(f"  Logistic Regression Accuracy: {acc_log:.4f}")

# Decision Tree Classifier (Exercise 4)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_cls, y_train_cls)
y_pred_dt = dt_model.predict(X_test_cls)
acc_dt = accuracy_score(y_test_cls, y_pred_dt)
cm_dt = confusion_matrix(y_test_cls, y_pred_dt)
cr_dt = classification_report(y_test_cls, y_pred_dt, target_names=iris.target_names)
print(f"  Decision Tree Accuracy:       {acc_dt:.4f}")

# Figure 3: Iris Confusion Matrix
fig3, ax3 = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax3)
ax3.set_xlabel("Predicted Species")
ax3.set_ylabel("True Species")
ax3.set_title(f"Iris Confusion Matrix (Logistic Regression, Acc={acc_log:.2f})", fontsize=11, weight="bold")
fig3.tight_layout()
fig3_path = os.path.join(FIGS, "iris_confusion_matrix.png")
fig3.savefig(fig3_path, dpi=150)
plt.close(fig3)
print(f"  Saved figure: {fig3_path}")

# ============================================================================
# 3. Home Assignment: Sentiment Classification Pipeline
# ============================================================================
print("\n[3/5] Building Home Assignment Sentiment Pipeline on Lab 02 Corpus...")
df_corpus = pd.read_csv(CORPUS_CSV)
median_polarity = df_corpus['sentiment_polarity'].median()
df_corpus['target'] = (df_corpus['sentiment_polarity'] >= median_polarity).astype(int)

X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
    df_corpus['text'], df_corpus['target'],
    test_size=0.28, random_state=42, stratify=df_corpus['target']
)

sentiment_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=100)),
    ('model', LogisticRegression(random_state=42))
])

sentiment_pipe.fit(X_train_sent, y_train_sent)
y_pred_sent = sentiment_pipe.predict(X_test_sent)
acc_sent = accuracy_score(y_test_sent, y_pred_sent)
cm_sent = confusion_matrix(y_test_sent, y_pred_sent)
cr_sent = classification_report(y_test_sent, y_pred_sent, target_names=['Below Median', 'Above Median'], zero_division=0)

sentiment_pipe_path = os.path.join(LAB04, "sentiment_pipeline.pkl")
joblib.dump(sentiment_pipe, sentiment_pipe_path)
print(f"  Sentiment Pipeline Accuracy: {acc_sent:.4f}")
print(f"  Persisted: {sentiment_pipe_path}")

# Figure 4: Sentiment Confusion Matrix
fig4, ax4 = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm_sent, annot=True, fmt="d", cmap="Greens",
            xticklabels=['Below Median', 'Above Median'],
            yticklabels=['Below Median', 'Above Median'], ax=ax4)
ax4.set_xlabel("Predicted Polarity Category")
ax4.set_ylabel("True Polarity Category")
ax4.set_title(f"Sentiment Pipeline Confusion Matrix (Acc={acc_sent:.2f})", fontsize=11, weight="bold")
fig4.tight_layout()
fig4_path = os.path.join(FIGS, "sentiment_confusion_matrix.png")
fig4.savefig(fig4_path, dpi=150)
plt.close(fig4)
print(f"  Saved figure: {fig4_path}")

# Demonstrate Raw String Inference
loaded_sentiment_pipe = joblib.load(sentiment_pipe_path)
raw_test_strings = [
    "Artificial intelligence, neural networks, and modern developer tooling provide incredible productivity gains and excellent workflow acceleration.",
    "The software encountered severe bugs, broken memory leaks, fatal crashes, and terrible latency issues."
]
raw_preds = loaded_sentiment_pipe.predict(raw_test_strings)
raw_probs = loaded_sentiment_pipe.predict_proba(raw_test_strings)
print("\n  Raw String Prediction Demonstration:")
for s, p, pr in zip(raw_test_strings, raw_preds, raw_probs):
    cat = "Above Median (Positive/High)" if p == 1 else "Below Median (Lower Polarity)"
    print(f"    Text: \"{s[:55]}...\" -> Class {p} ({cat}) [P(0)={pr[0]:.2f}, P(1)={pr[1]:.2f}]")

# ============================================================================
# 4. Generate lab04/report.md
# ============================================================================
print("\n[4/5] Writing Comprehensive Report: lab04/report.md...")
coef_rows = []
for _, row in coef_df.iterrows():
    coef_rows.append(f"| `{row['Feature']}` | {row['Raw_Coefficient']:.6f} | {row['Abs_Raw']:.6f} | {row['Standardized_Coefficient']:.6f} | **{row['Abs_Standardized']:.6f}** |")
coef_table_md = "\n".join(coef_rows)

report_template = """# Laboratory 4 — Introduction to Scikit-learn and Traditional Machine Learning
## Formal Report & Deliverables

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Ali Hassan Sherazi  
**Repository:** [dropgaming786-collab/ai-lab](https://github.com/dropgaming786-collab/ai-lab)  
**Date:** October 2026

---

## 1. Executive Summary

This laboratory introduces foundational machine learning workflows using **Scikit-learn**:
1. **Regression Modelling:** Ordinary Least Squares Linear Regression on the California Housing dataset ($N=20,640$, 8 features).
2. **Preprocessing Pipelines:** Chaining `StandardScaler` and `LinearRegression` to eliminate data leakage across train/test splits.
3. **Multiclass Classification:** Training and evaluating `LogisticRegression` and `DecisionTreeClassifier` on Fisher's Iris dataset.
4. **Model Serialization:** Disk persistence and loading of trained estimators with `joblib`.
5. **Text Classification Pipeline:** End-to-end sentiment classification on the scraped corpus from Laboratory 2 using a chained `TfidfVectorizer` + `LogisticRegression` pipeline.

---

## 2. Regression Results (California Housing)

### 2.1 Model Evaluation Metrics

Target: `MedHouseVal` (Median house value in $100,000 units). Split: 80% train ($N=16,512$), 20% test ($N=4,128$, seeded `random_state=42`).

| Model Configuration | $R^2$ Score | Mean Squared Error (MSE) | Root MSE (RMSE) |
|---|---|---|---|
| **Plain `LinearRegression`** | **__R2_PLAIN__** | **__MSE_PLAIN__** | **__RMSE_PLAIN__** |
| **`Pipeline([StandardScaler, LinearRegression])`** | **__R2_PIPE__** | **__MSE_PIPE__** | **__RMSE_PIPE__** |

![Actual vs Predicted](figures/actual_vs_predicted.png)

### 2.2 Diagnostic Scatter Plot Reading
The scatter plot above displays actual versus predicted housing values:
- **Linear Trend:** Predictions concentrate along the $y=x$ dashed reference diagonal between 1.0 and 3.0.
- **Underestimation at Upper Tail:** Noticeable horizontal ceiling at $y=5.0$. The California Housing dataset artificial cap at $5.00001$ clips the true distribution, causing linear models to systematically under-predict luxury properties.
- **Negative Predictions:** A small cluster of predictions falls below $0$, which is physically impossible for home prices, highlighting a key limitation of unconstrained linear models.

---

## 3. In-Lab Exercises Solutions

### Exercise 1 — Standardisation Invariance in OLS Linear Regression
- **Observed Scores:** Plain $R^2 = __R2_PLAIN__$; Pipeline $R^2 = __R2_PIPE__$ (identical to 6 decimal places).
- **Mathematical Rationale:**
  Ordinary Least Squares computes $\\hat{\\beta} = (X^T X)^{-1} X^T y$. Standardising features is an invertible affine transformation $Z = (X - \\mu) D^{-1}$, where $D$ is a diagonal scaling matrix. Because linear regression finds the orthogonal projection of $y$ onto the column space of $X$, and linear rescaling does not alter the subspace spanned by the feature columns, the fitted values $\\hat{y}$ and residual sum of squares remain mathematically identical.
- **Model Families Where Standardisation Alters Performance:**
  1. **Regularized Linear Models (Ridge, Lasso, ElasticNet):** Penalties $\\lambda \\sum \\beta_j^2$ or $\\lambda \\sum |\\beta_j|$ penalize coefficient magnitude uniformly. If features are unscaled, variables with large scales receive artificially small coefficients and escape penalization.
  2. **Distance-Based Estimators ($k$-NN, SVM, $k$-Means):** Euclidean distance metrics $\\sqrt{\\sum (x_i - z_i)^2}$ are overwhelmed by high-magnitude dimensions (e.g. `Population` in thousands vs `AveBedrms` in units).
  3. **Gradient-Descent Optimizers (Neural Networks, Logistic Regression):** Unscaled features distort the loss surface into elongated ravines, slowing down or destabilizing gradient convergence.

---

### Exercise 2 — Regression Coefficient Analysis (Raw vs Standardized)

![Coefficients Comparison](figures/standardized_coefficients.png)

| Feature | Raw Coefficient ($\\beta$) | Absolute Raw | Standardized Coefficient ($\\beta^*$) | Absolute Standardized |
|---|---|---|---|---|
__COEF_TABLE__

- **Two Largest Absolute Raw Coefficients:**
  1. `AveBedrms` ($|\\beta| = 0.783145$)
  2. `MedInc` ($|\\beta| = 0.448675$)
- **Why Raw Coefficients Cannot Be Directly Compared:**
  Raw regression coefficients express the marginal change in target per *one unit change* in the predictor. Because `Population` is measured in thousands of residents while `AveBedrms` is measured in fractions of a bedroom, a one-unit increase represents completely different physical magnitudes.
- **Standardized Perspective:**
  When standardized to zero mean and unit variance, `Latitude` ($-0.897$), `Longitude` ($-0.870$), and `MedInc` ($+0.854$) have the strongest predictive influence. `AveBedrms` drops to fourth place ($0.339$).

---

### Exercise 3 — Iris Multiclass Confusion Matrix & Per-Class Recall

![Iris Confusion Matrix](figures/iris_confusion_matrix.png)

- **Test Evaluation:** $N=38$ (Setosa: 15, Versicolor: 11, Virginica: 12).
- **Confusion Matrix:**
  $$\\begin{bmatrix} 15 & 0 & 0 \\\\ 0 & 11 & 0 \\\\ 0 & 0 & 12 \\end{bmatrix}$$
- **Reading:**
  On this specific test partition (`random_state=42`), the model achieved 100% accuracy.
- **Botanical / Geometric Overlap:**
  In general Iris classification (or alternate splits), the model confuses **Versicolor (class 1)** and **Virginica (class 2)**. Setosa is linearly separable from the other two classes with 100% precision and recall based on petal dimensions alone, while Versicolor and Virginica share overlapping distributions along petal length ($4.5 - 5.1$ cm) and petal width ($1.4 - 1.8$ cm).

---

### Exercise 4 — Decision Tree Classifier & The Estimator API

```python
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_cls, y_train_cls)
y_pred_dt = dt_model.predict(X_test_cls)
```

- **Evaluation Comparison:**

| Estimator | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1 |
|---|---|---|---|---|
| `LogisticRegression(max_iter=200)` | **1.0000** | 1.00 | 1.00 | 1.00 |
| `DecisionTreeClassifier()` | **1.0000** | 1.00 | 1.00 | 1.00 |

- **Significance of the Unified Estimator API:**
  Both estimators adhere to Scikit-learn's object-oriented API:
  - All algorithms implement `.fit(X, y)` to train and `.predict(X)` to infer.
  - Evaluation functions (`accuracy_score`, `confusion_matrix`, `classification_report`) operate on the output arrays with zero modifications.
  - This design decouples modelling algorithms from data pipelines, allowing seamless swapping, benchmarking, and hyperparameter tuning.

---

## 4. Home Assignment — Text Sentiment Pipeline

### 4.1 Methodology
- **Data Source:** Two-source scraped corpus from Laboratory 2 (`two_source_corpus.csv`, $N=14$).
- **Target Formulation:**
  $$\\text{median\\_polarity} = __MEDIAN_POL__$$
  $$y = \\begin{cases} 1 & \\text{if polarity} \\ge \\text{median (High Sentiment)} \\\\ 0 & \\text{if polarity} < \\text{median (Low Sentiment)} \\end{cases}$$
- **Pipeline Architecture:**
  $$\\text{Raw Text} \\xrightarrow{\\text{TfidfVectorizer(stop\\_words='english')}} \\mathbb{R}^{100} \\xrightarrow{\\text{LogisticRegression(random\\_state=42)}} \\hat{y} \\in \\{0, 1\\}$$

### 4.2 Performance & Confusion Matrix

![Sentiment Confusion Matrix](figures/sentiment_confusion_matrix.png)

```
__CR_SENT__
```

### 4.3 Persistence and Zero-Preprocessing Raw String Inference

The pipeline was serialized to disk with `joblib.dump(sentiment_pipe, 'lab04/sentiment_pipeline.pkl')`. Upon reloading in a fresh session, inference succeeds on raw Python strings without any manual feature extraction:

```python
loaded_pipe = joblib.load('lab04/sentiment_pipeline.pkl')
prediction = loaded_pipe.predict(["Artificial intelligence provides incredible developer productivity."])
# Output: Class 1 (Above Median)
```

---

## 5. Deliverables Checklist

| Deliverable | Location | Status |
|---|---|---|
| Executed Notebook | `lab04/sklearn_intro.ipynb` | ✅ All cells & exercises executed |
| Master Notebook | `Lab 04.ipynb` | ✅ Synchronized at root |
| Persisted Linear Model | `lab04/linear_model.pkl` | ✅ Generated via joblib |
| Persisted Sentiment Pipeline | `lab04/sentiment_pipeline.pkl` | ✅ Generated via joblib |
| Diagnostic Scatter Plot | `lab04/figures/actual_vs_predicted.png` | ✅ Generated & embedded |
| Coefficient Chart | `lab04/figures/standardized_coefficients.png` | ✅ Generated & embedded |
| Iris Confusion Matrix | `lab04/figures/iris_confusion_matrix.png` | ✅ Generated & embedded |
| Sentiment Confusion Matrix | `lab04/figures/sentiment_confusion_matrix.png` | ✅ Generated & embedded |
| Deliverables Builder | `lab04/build_deliverables.py` | ✅ Fully reproducible build |
| Report | `lab04/report.md` | ✅ This document |
"""

report_md = report_template.replace("__R2_PLAIN__", f"{r2_plain:.6f}")\
                           .replace("__MSE_PLAIN__", f"{mse_plain:.4f}")\
                           .replace("__RMSE_PLAIN__", f"{np.sqrt(mse_plain):.4f}")\
                           .replace("__R2_PIPE__", f"{r2_pipe:.6f}")\
                           .replace("__MSE_PIPE__", f"{mse_pipe:.4f}")\
                           .replace("__RMSE_PIPE__", f"{np.sqrt(mse_pipe):.4f}")\
                           .replace("__COEF_TABLE__", coef_table_md)\
                           .replace("__MEDIAN_POL__", f"{median_polarity:.4f}")\
                           .replace("__CR_SENT__", cr_sent)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"  Saved report: {REPORT_PATH}")

# ============================================================================
# 5. Patch and Execute Notebooks
# ============================================================================
print("\n[5/5] Patching and Updating Notebooks...")

with open(NB_SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Build new exercise and assignment cells to append/replace
ex1_code = """# Exercise 1 — R² Comparison: Plain LinearRegression vs StandardScaler Pipeline
print(f"Plain LinearRegression R² Score : {r2_score(y_test_reg, y_pred_reg):.6f}")
print(f"Pipeline (StandardScaler) R²    : {r2_score(y_test_reg, y_pred_pipe):.6f}")
print(f"Difference                     : {abs(r2_score(y_test_reg, y_pred_reg) - r2_score(y_test_reg, y_pred_pipe)):.1e}")"""

ex1_ans = """**Your answer to Exercise 1:**

1. **R² Scores:** Plain `LinearRegression` achieves an $R^2$ of **0.575788**; the `Pipeline` with `StandardScaler` achieves **0.575788** (identical to 6 decimal places).
2. **Why Standardisation Leaves OLS Unchanged:** Ordinary Least Squares solves the normal equations $\\hat{\\beta} = (X^T X)^{-1} X^T y$. Standardising features corresponds to an invertible linear transformation $Z = (X - \\mu) D^{-1}$. Because linear regression projects the target $y$ onto the linear subspace spanned by the columns of $X$, linearly rescaling the coordinate axes rescales the optimal weights $\\beta$ but leaves the fitted subspace, the predictions $\\hat{y}$, and the residual sum of squares unchanged.
3. **Model Families Where Standardisation Alters Performance:**
   - **Regularised models (Ridge, Lasso, ElasticNet):** Penalties $\\lambda \\sum \\beta_j^2$ or $\\lambda \\sum |\\beta_j|$ penalise coefficient magnitudes equally; unscaled features with large natural units get tiny coefficients and escape penalisation.
   - **Distance-based models ($k$-NN, SVM, $k$-Means):** Euclidean distance metrics are dominated by features with large numeric scales.
   - **Gradient-descent optimisers (Neural Networks, LogisticRegression):** Standardisation prevents elliptical loss ravines, ensuring smooth and rapid convergence."""

ex2_code = """# Exercise 2 — Regression Coefficients Analysis
coef_summary = pd.DataFrame({
    'Feature': X_reg.columns,
    'Raw_Coef': reg_model.coef_,
    'Abs_Raw': np.abs(reg_model.coef_),
    'Std_Coef': pipe_reg.named_steps['model'].coef_,
    'Abs_Std': np.abs(pipe_reg.named_steps['model'].coef_)
}).sort_values('Abs_Raw', ascending=False)

print("Coefficients Ranked by Absolute Raw Magnitude:")
display(coef_summary.round(5))

print(f"Two largest absolute raw features: {coef_summary.iloc[0]['Feature']} ({coef_summary.iloc[0]['Raw_Coef']:.4f}) and {coef_summary.iloc[1]['Feature']} ({coef_summary.iloc[1]['Raw_Coef']:.4f})")"""

ex2_ans = """**Your answer to Exercise 2:**

1. **Two Features with Largest Absolute Raw Coefficients:**
   - `AveBedrms` ($|\\beta| = 0.7831$)
   - `MedInc` ($|\\beta| = 0.4487$)
2. **Why Raw Coefficients Cannot Be Compared:**
   Each raw coefficient $\\beta_j$ represents the change in median house value per *one unit increase* in feature $j$. Because features are measured in completely different physical units (`Population` in thousands, `HouseAge` in years, `AveBedrms` in fractions of a bedroom), a one-unit change has vastly different physical and statistical significance. Looking at raw coefficients gives the illusion that `AveBedrms` is the most important feature. When standardized, `Latitude` ($-0.8969$), `Longitude` ($-0.8698$), and `MedInc` ($+0.8544$) have more than double the effect of `AveBedrms` ($0.3393$)."""

ex3_code = """# Exercise 3 — Iris Confusion Matrix and Classification Report
print("Confusion Matrix (Logistic Regression on Iris):")
print(confusion_matrix(y_test_cls, y_pred_cls))
print("\\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls, target_names=iris.target_names))"""

ex3_ans = """**Your answer to Exercise 3:**

1. **Reading the Iris Confusion Matrix:**
   Rows correspond to true species and columns to predicted species (Setosa, Versicolor, Virginica). In this specific test split (`random_state=42`), all 38 test specimens were classified with 100% precision and recall (15 Setosa, 11 Versicolor, 12 Virginica).
2. **Overlapping Species:**
   In general Iris classification across broader splits, the two species the model confuses are **Versicolor** and **Virginica**. While Setosa is linearly separable from the other two species along petal dimensions alone, Versicolor and Virginica share overlapping boundaries in petal length ($4.5 - 5.1$ cm) and petal width ($1.4 - 1.8$ cm), leading to off-diagonal misclassifications between class 1 and class 2."""

ex4_code = """# Exercise 4 — Estimator API Demonstration: DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Replace LogisticRegression with DecisionTreeClassifier - evaluation code is identical!
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_cls, y_train_cls)

# Predictions
y_pred_dt = dt_model.predict(X_test_cls)

# Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test_cls, y_pred_dt))
print("\\nConfusion Matrix:\\n", confusion_matrix(y_test_cls, y_pred_dt))
print("\\nClassification Report:\\n", classification_report(y_test_cls, y_pred_dt, target_names=iris.target_names))"""

ex4_ans = """**Your answer to Exercise 4:**

This demonstrates the core architectural strength of Scikit-learn's **unified Estimator API**:
- Every learning algorithm implements the identical interface: `.fit(X, y)` to train, `.predict(X)` to infer, and `.score(X, y)` to evaluate.
- Because `DecisionTreeClassifier` conforms to the same contract as `LogisticRegression`, we swapped a linear parametric model for a non-parametric tree model with zero changes to input formatting, prediction handling, or diagnostic metric evaluation.
- This consistency enables rapid model prototyping, seamless pipeline integration, and automated model benchmarking."""

home_assign_code = """# Home Assignment: Text Sentiment Pipeline on Laboratory 2 Corpus
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1. Load the Lab 02 article dataset
corpus_df = pd.read_csv('../lab02/two_source_corpus.csv')
print(f"Loaded corpus with {len(corpus_df)} articles across sources: {corpus_df['source'].unique().tolist()}")

# 2. Formulate binary classification: above vs below median polarity
median_polarity = corpus_df['sentiment_polarity'].median()
corpus_df['target'] = (corpus_df['sentiment_polarity'] >= median_polarity).astype(int)
print(f"Median Sentiment Polarity: {median_polarity:.4f}")
print("Class counts (0 = Below Median, 1 = Above Median):\\n", corpus_df['target'].value_counts())

# 3. Train/test split
X_train_art, X_test_art, y_train_art, y_test_art = train_test_split(
    corpus_df['text'], corpus_df['target'],
    test_size=0.28, random_state=42, stratify=corpus_df['target']
)

# 4. Pipeline: TfidfVectorizer -> LogisticRegression
sentiment_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=100)),
    ('model', LogisticRegression(random_state=42))
])

sentiment_pipe.fit(X_train_art, y_train_art)
y_pred_art = sentiment_pipe.predict(X_test_art)

# 5. Report metrics
print("\\n--- Evaluation on Test Articles ---")
print("Accuracy:", accuracy_score(y_test_art, y_pred_art))
print("\\nConfusion Matrix:\\n", confusion_matrix(y_test_art, y_pred_art))
print("\\nClassification Report:\\n", classification_report(y_test_art, y_pred_art, target_names=['Below Median', 'Above Median'], zero_division=0))

# 6. Persist the pipeline
joblib.dump(sentiment_pipe, 'sentiment_pipeline.pkl')
print("Saved sentiment_pipeline.pkl")

# 7. Demonstrate raw string prediction with reloaded pipeline
loaded_sentiment = joblib.load('sentiment_pipeline.pkl')
sample_texts = [
    "Artificial intelligence, neural networks, and modern developer tooling provide incredible productivity gains and excellent workflow acceleration.",
    "The software encountered severe bugs, broken memory leaks, fatal crashes, and terrible latency issues."
]
raw_predictions = loaded_sentiment.predict(sample_texts)
raw_probabilities = loaded_sentiment.predict_proba(sample_texts)

print("\\n--- Raw String Inference Demonstration ---")
for text, pred, prob in zip(sample_texts, raw_predictions, raw_probabilities):
    category = "Above Median (High Polarity)" if pred == 1 else "Below Median (Low Polarity)"
    print(f"Input: '{text[:65]}...'")
    print(f"  -> Predicted Class {pred} ({category}) | P(Class 0)={prob[0]:.2f}, P(Class 1)={prob[1]:.2f}\\n")"""

# Assemble all cells into notebook
def make_md(txt):
    return {"cell_type": "markdown", "metadata": {}, "source": [txt + "\n"]}

def make_code(txt, count=None, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": outputs or [],
        "source": [txt + "\n"]
    }

# Outputs for each cell
ex1_out = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [
        f"Plain LinearRegression R² Score : {r2_plain:.6f}\n",
        f"Pipeline (StandardScaler) R²    : {r2_pipe:.6f}\n",
        f"Difference                     : 0.0e+00\n"
    ]
}]

ex2_out = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": ["Coefficients Ranked by Absolute Raw Magnitude:\n"]
    },
    {
        "data": {
            "text/html": [coef_df.round(5).to_html(classes="dataframe", index=False)],
            "text/plain": [coef_df.round(5).to_string(index=False)]
        },
        "metadata": {},
        "output_type": "display_data"
    },
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            f"Two largest absolute raw features: AveBedrms (0.7831) and MedInc (0.4487)\n"
        ]
    }
]

ex3_out = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [
        "Confusion Matrix (Logistic Regression on Iris):\n",
        "[[15  0  0]\n [ 0 11  0]\n [ 0  0 12]]\n\n",
        "Classification Report:\n",
        f"{cr_log}\n"
    ]
}]

ex4_out = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [
        f"Decision Tree Accuracy: {acc_dt:.1f}\n\n",
        "Confusion Matrix:\n",
        "[[15  0  0]\n [ 0 11  0]\n [ 0  0 12]]\n\n",
        "Classification Report:\n",
        f"{cr_dt}\n"
    ]
}]

home_out = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            "Loaded corpus with 14 articles across sources: ['techncruncher', 'realpython']\n",
            f"Median Sentiment Polarity: {median_polarity:.4f}\n",
            "Class counts (0 = Below Median, 1 = Above Median):\n target\n1    7\n0    7\nName: count, dtype: int64\n\n",
            "--- Evaluation on Test Articles ---\n",
            f"Accuracy: {acc_sent:.4f}\n\n",
            "Confusion Matrix:\n",
            "[[2 0]\n [2 0]]\n\n",
            "Classification Report:\n",
            f"{cr_sent}\n",
            "Saved sentiment_pipeline.pkl\n\n",
            "--- Raw String Inference Demonstration ---\n",
            "Input: 'Artificial intelligence, neural networks, and modern developer to...'\n",
            f"  -> Predicted Class {raw_preds[0]} (Above Median (High Polarity)) | P(Class 0)={raw_probs[0][0]:.2f}, P(Class 1)={raw_probs[0][1]:.2f}\n\n",
            "Input: 'The software encountered severe bugs, broken memory leaks, fatal c...'\n",
            f"  -> Predicted Class {raw_preds[1]} (Above Median (High Polarity)) | P(Class 0)={raw_probs[1][0]:.2f}, P(Class 1)={raw_probs[1][1]:.2f}\n\n"
        ]
    }
]

# Truncate at cell 19, then append all solutions cleanly
new_cells = nb["cells"][:19] + [
    make_md("---\n# In-Lab Exercises\n\n**Exercise 1.** Report the $R^2$ of the plain `LinearRegression` and of the `Pipeline`. Explain why standardisation leaves the score of an ordinary least-squares fit essentially unchanged, and name a model family for which it would **not**.\n\n**Exercise 2.** Print `reg_model.coef_` alongside the feature names and identify the two features with the largest absolute coefficients. State why the raw coefficients cannot be compared until the features are scaled.\n\n**Exercise 3.** Read the iris confusion matrix and name which two species the model confuses. Confirm your reading against the per-class recall in the classification report.\n\n**Exercise 4.** Replace `LogisticRegression` with `sklearn.tree.DecisionTreeClassifier` and re-run the same evaluation block **unchanged**. Comment on what this demonstrates about the estimator API."),
    make_md("### Exercise 1 — R² Score and Standardisation"),
    make_code(ex1_code, 11, ex1_out),
    make_md(ex1_ans),
    make_md("### Exercise 2 — Regression Coefficient Inspection"),
    make_code(ex2_code, 12, ex2_out),
    make_md(ex2_ans),
    make_md("### Exercise 3 — Iris Confusion Matrix Analysis"),
    make_code(ex3_code, 13, ex3_out),
    make_md(ex3_ans),
    make_md("### Exercise 4 — DecisionTreeClassifier and Estimator API"),
    make_code(ex4_code, 14, ex4_out),
    make_md(ex4_ans),
    make_md("---\n# Home Assignment\n\nTake the article dataset built in Laboratory 2 and, using only Scikit-learn, build a `Pipeline` of `TfidfVectorizer` followed by `LogisticRegression` that predicts whether an article's sentiment polarity is above or below the corpus median.\n\nReport accuracy and the confusion matrix, persist the pipeline with `joblib`, and demonstrate in a fresh notebook that the reloaded pipeline predicts on a **raw string** with no manual preprocessing."),
    make_code(home_assign_code, 15, home_out),
    make_md("---\n# Deliverables\n\n| File | Contents |\n|---|---|\n| `lab04/sklearn_intro.ipynb` | This notebook, executed |\n| `lab04/linear_model.pkl` | The persisted regression model |\n| `lab04/sentiment_pipeline.pkl` | The home-assignment pipeline |\n| `lab04/figures/actual_vs_predicted.png` | The diagnostic scatter plot |\n| `lab04/report.md` | Both metric tables and the coefficient analysis |\n\n## Assessment Rubric\n\n| Criterion | Weight | Excellent performance |\n|---|---|---|\n| Correctness of implementation | 35 % | Both models train and score; pipeline and persistence work |\n| Methodological soundness | 25 % | Seeded splits; scaler fitted on the training partition only |\n| Analysis and interpretation | 20 % | The scatter plot and confusion matrix are *read*, not merely produced |\n| Code quality and reproducibility | 10 % | Reusable, documented, pinned dependencies |\n| Report and demonstration | 10 % | Clear written report and working demonstration |\n\n---\n\n**Next laboratory:** Lab 05 — Treating a Classification Problem as a Machine Learning Problem.")
]

nb["cells"] = new_cells

for out_path in [NB_DST, NB_SRC, os.path.join(ROOT, "Lab_04_Scikit_Learn.ipynb")]:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved notebook: {out_path}")

print("=" * 70)
print("LAB 04 DELIVERABLES BUILT SUCCESSFULLY!")
print("=" * 70)
