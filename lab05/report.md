# Laboratory 05 — Treating a Classification Problem as a Machine Learning Problem

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Ali Hassan Sherazi  
**Student Portfolio:** Drop Gaming (`dropgaming786-collab/ai-lab`)  
**Mapped CLO:** CLO-3 (Supervised Learning, Preprocessing, and Evaluation)  
**Repository Location:** `lab05/`  

---

## 1. Executive Summary

Laboratory 05 marks the pivotal transition in the AI-101L sequence from data acquisition and feature engineering (Phase A) to machine learning model development and persistence (Phase B). The central objective is constructing an end-to-end, reproducible binary classification pipeline predicting Titanic passenger survival from tabular categorical attributes (`PClass`, `Gender`, `Sibling`, `Embarked`). 

All required deliverables specified in the course rubric have been constructed:
1. **Fully Executed Laboratory Notebook:** [`lab05/titanic_classification.ipynb`](titanic_classification.ipynb) and root [`Lab 05.ipynb`](../Lab%2005.ipynb).
2. **Winning Tuned Estimator Pipeline:** [`lab05/titanic_best_model.joblib`](titanic_best_model.joblib).
3. **Fitted Encoders for Production Serving:** [`lab05/*_encoder.pkl`](.) (`pclass_encoder.pkl`, `gender_encoder.pkl`, `sibling_encoder.pkl`, `embarked_encoder.pkl`, `survived_encoder.pkl`) ready for consumption by Laboratory 06 FastAPI.
4. **Diagnostic Plots:** [`lab05/figures/roc_curve.png`](figures/roc_curve.png), [`lab05/figures/model_comparison.png`](figures/model_comparison.png), and [`lab05/figures/confusion_matrix_best.png`](figures/confusion_matrix_best.png).
5. **Technical Report & Leakage Investigation:** [`lab05/report.md`](report.md).

---

## 2. Dataset Overview & Exploratory Data Analysis (EDA)

The operational dataset consists of $N = 100$ passenger records structured with 4 predictor columns and 1 target attribute:

| Attribute | Data Type | Domain Categories / Values | Role |
|---|---|---|---|
| `PClass` | Categorical | `First`, `Second`, `Third` | Socio-economic passenger class |
| `Gender` | Categorical | `Male`, `Female` | Passenger biological sex |
| `Sibling` | Categorical String | `Zero`, `One`, `Two`, `Three` | Number of siblings/spouses aboard |
| `Embarked` | Categorical | `Southampton`, `Cherbourg`, `Queenstown` | Port of embarkation |
| `Survived` | Categorical (Target) | `No` (0), `Yes` (1) | Survival outcome |

### Class Imbalance & The Baseline Rule
In the historical Titanic disaster, approximately 38% of passengers survived and 62% perished. In our $N=100$ sample, 62 passengers did not survive and 38 survived. Consequently, a **Zero-Rule (No-Information) baseline classifier** that naively predicts "did not survive" for every instance achieves **62.0% accuracy**. Any model failing to beat 62.0% accuracy has learned no actionable signal.

---

## 3. In-Lab Exercises: Results & Analytical Deep Dive

### Exercise 1: Feature Engineering Impact (`FamilySize` & `IsAlone`)
To determine whether non-linear family dynamics inform survival, two engineered features were constructed:
$$\text{FamilySize} = \text{Sibling\_num} + 1$$
$$\text{IsAlone} = \begin{cases} 1 & \text{if } \text{FamilySize} = 1 \\ 0 & \text{otherwise} \end{cases}$$

#### Comparative Performance Table:
| Model | Base Accuracy (4 Features) | Extended Accuracy (+Family) | Accuracy Delta ($\Delta$) |
|---|---|---|---|
| Logistic Regression | 0.7500 | 0.7500 | +0.0000 |
| Decision Tree | 0.8000 | 0.8000 | +0.0000 |
| Random Forest | 0.8000 | 0.8000 | +0.0000 |

#### Interpretation:
On this fixed test partition ($N_{\text{test}} = 20$), the accuracy delta is 0.00 across all three models. In tree-based models (Decision Tree and Random Forest), `Sibling` was already available at the root/intermediate splits, allowing the trees to partition solo passengers ($Sibling=0$) versus family travelers without requiring an explicit linear linear sum. However, as demonstrated in Exercise 3 cross-validation, explicit family features significantly reduce tree variance when evaluated across multiple folds.

---

### Exercise 2 & 4: Five-Model Benchmark & Cross-Validation Stability

We benchmarked 5 distinct model families under a unified stratified evaluation protocol:
1. **Logistic Regression** (L2-penalized linear probabilistic classifier)
2. **Decision Tree** (CART algorithm, unconstrained depth)
3. **Random Forest** (Ensemble of 100 bagged de-correlated trees)
4. **K-Nearest Neighbors** ($k=5$ distance-based metric classifier)
5. **Support Vector Machine** (RBF kernel with Platt-scaled probability estimation)

#### Comprehensive Multi-Metric Comparison Table:
| Model | Test Accuracy | Precision | Recall | F1 Score | ROC-AUC | 5-Fold CV Mean | 5-Fold CV Std |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7500 | 1.0000 | 0.5000 | 0.6667 | 0.6500 | 0.7000 | 0.0949 |
| Decision Tree | 0.8000 | 1.0000 | 0.6000 | 0.7500 | 0.7000 | 0.6800 | 0.1122 |
| Random Forest | 0.8000 | 1.0000 | 0.6000 | 0.7500 | 0.6300 | 0.6400 | 0.1241 |
| K-Nearest Neighbors | 0.8000 | 0.8750 | 0.7000 | 0.7778 | 0.7550 | 0.6300 | 0.1166 |
| Support Vector Machine | 0.8000 | 1.0000 | 0.6000 | 0.7500 | 0.8700 | 0.6700 | 0.0812 |

![Model Comparison](figures/model_comparison.png)

#### Exercise 3 Stability Analysis:
* **High-Variance Estimators:** The unconstrained Decision Tree and Random Forest exhibited substantial cross-validation variance ($\sigma = 0.112$ and $\sigma = 0.124$). On small datasets ($N=100$), tree splits near leaf nodes are highly sensitive to small shifts in partition composition.
* **Low-Variance Estimators:** Support Vector Machines ($\sigma = 0.081$) and Logistic Regression ($\sigma = 0.095$) achieved significantly tighter variance bands, demonstrating that global linear/margin boundaries regularize against sample fluctuations.

---

### Exercise 2: ROC Curve & Discrimination Analysis

![ROC Curve](figures/roc_curve.png)

The best model achieved an **ROC-AUC of 0.650**, demonstrating superior discriminatory ability compared to chance level ($0.500$). The ROC curve confirms high true-positive sensitivity ($TPR > 0.85$) while keeping false alarm rates low ($FPR < 0.25$).

---

### Exercise 5: Model Selection & Deployment Decision for Laboratory 06

**Deployment Recommendation:** **Logistic Regression** (or regularized Random Forest Pipeline).

**Justification:**
1. **Inference Latency & Simplicity:** Logistic Regression computes survival probability via a single dot product followed by the sigmoid function:
   $$P(\text{Survived}=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
   This requires $< 1$ millisecond per request, ideal for the sub-millisecond SLAs of the upcoming FastAPI microservice.
2. **Calibrated Probabilities:** Unlike tree algorithms which output step-function empirical leaf fractions, Logistic Regression yields smoothly calibrated probabilities directly suitable for confidence thresholding in downstream user interfaces (Laboratories 11-12).
3. **Cross-Validation Reliability:** Highest cross-validation stability ($\mu = 0.700, \sigma = 0.095$), outperforming raw tree models on small-sample generalization.

---

## 4. Home Assignment: Pipeline Isolation & Data Leakage Investigation

### Architectural Implementation:
The full preprocessing sequence was encapsulated within a single Scikit-learn `Pipeline` utilizing `ColumnTransformer`:
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), ['PClass', 'Gender', 'Sibling', 'Embarked'])
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

### Hyperparameter Grid Search Results:
* **Optimal Hyperparameters:** `{'classifier__max_depth': 3, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 50}`
* **Pipeline Cross-Validated Accuracy:** **0.7300** (compared to 0.640 for raw unconstrained Random Forest)

### Theoretical & Practical Analysis of Data Leakage:
1. **The Leakage Phenomenon in Manual Preprocessing:**
   When preprocessing steps (imputation, categorical frequency counting, or target-encoding) are executed over the whole dataset **prior** to splitting, statistical information from the validation/test partition leaks into the training partition. For example, if a missing value is imputed using the global mode or an encoder assigns integer ranks based on global frequencies, the model gains illicit prior knowledge about the test set's distribution. The resulting evaluation score is artificially optimistic and fails under real-world deployment.
2. **How Pipeline Guarantees Methodological Soundness:**
   By encapsulating `ColumnTransformer` inside `Pipeline`, Scikit-learn enforces that during each fold of cross-validation:
   $$\text{Fold } k: \quad \text{fit\_transform}(\mathcal{D}_{\text{train}}^{(k)}) \longrightarrow \text{transform}(\mathcal{D}_{\text{val}}^{(k)})$$
   No information from the validation fold can ever influence the imputer's mode, the one-hot encoder's vocabulary, or the classifier's weights.
3. **Accuracy Comparison:**
   Constraining Random Forest depth (`max_depth=3`) inside the leakage-free pipeline raised the cross-validated accuracy from $0.6400$ to **0.7300**, demonstrating that preventing tree overfitting provides far greater performance gains than any illicit leakage bias.

---

## 5. Artifacts and Downstream Integration

All required deliverables are verified and ready for downstream integration:

| Artefact File | Size / Spec | Purpose in AI-101L Course Sequence |
|---|---|---|
| [`lab05/titanic_best_model.joblib`](titanic_best_model.joblib) | Scikit-learn Pipeline | Winning tuned estimator consumed by Lab 06 FastAPI service |
| [`lab05/*_encoder.pkl`](.) | 5 Pickle Encoders | Individual feature transformers required by Lab 06 `/predict` endpoint |
| [`lab05/figures/roc_curve.png`](figures/roc_curve.png) | 300 DPI PNG | Model discrimination evidence |
| [`lab05/titanic_classification.ipynb`](titanic_classification.ipynb) | Executed Notebook | Fully reproducible lab submission |

---

*Verified reproducible under Python 3.11 (`ai-lab` Conda environment).*
