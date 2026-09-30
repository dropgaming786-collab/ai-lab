# Laboratory 1 — Environment Setup and First Inspections
## Report

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Ali Hassan Sherazi  
**Repository:** https://github.com/dropgaming786-collab/ai-lab  
**Date:** September 2026

---

## 1. Setup and Environment Verification

### 1.1 Python & Conda Verification
The course environment was provisioned using Conda with Python 3.11:
- **Environment Name:** `ai-lab`
- **Python Version:** `3.11.16`
- **Interpreter:** `C:\Users\dropg\.conda\envs\ai-lab\python.exe`
- **Platform:** Windows 11 (64-bit)
- **Jupyter Kernel:** Registered as `Python (ai-lab)` via `ipykernel`

---

## 2. Dataset First-Look Inspections (Titanic Corpus)

The dataset was ingested directly from:
`https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv`

### 2.1 Tabular Structure (`head()`)
- **Dimensions:** 891 rows $\times$ 15 columns
- **Sample records:** Captures passenger survival, demographic classifications (`pclass`, `sex`, `age`), family relations (`sibsp`, `parch`), and transit economics (`fare`, `embarked`, `deck`).

### 2.2 Composition and Missing Value Audit (`info()`)
Three primary feature groups contain missing entries:
1. `deck`: 688 missing (77.2% missing) — high sparsity, requires categorical imputation or indicator flag.
2. `age`: 177 missing (19.9% missing) — requires median or subgroup imputation.
3. `embarked` / `embark_town`: 2 missing (0.2% missing) — minimal sparsity, safely imputable with mode.

### 2.3 Numeric Distributions (`describe()`)
- `fare`: Highly skewed right (mean = 32.20, std = 49.69, min = 0.00, 75th percentile = 31.00, max = 512.33).
- `age`: Spans infants (0.42 years) to elderly passengers (80.00 years) with mean $\approx 29.70$.

---

## 3. In-Lab Exercises

### Exercise 2 — Missing Value Extraction
From `df.info()` alone:
```python
missing_columns = {
    "age": 177,
    "deck": 688,
    "embarked": 2,
    "embark_town": 2,
}
```

### Exercise 3 — Coefficient of Variation (std / mean)
| Feature | Mean | Std | CV (|std / mean|) |
|---|---|---|---|
| `fare` | 32.20 | 49.69 | **1.543** |
| `survived` | 0.38 | 0.49 | 1.265 |
| `parch` | 0.38 | 0.81 | 1.173 |
| `sibsp` | 0.52 | 1.10 | 0.932 |
| `age` | 29.70 | 14.53 | 0.490 |
| `pclass` | 2.31 | 0.84 | 0.362 |

**Analysis:**
The feature `fare` has the largest coefficient of variation (approx. 1.543), indicating that its standard deviation is more than 1.5 times its mean, with values spanning from 0 to over 512. For a model trained on unscaled data, this extreme relative dispersion causes distance metrics and gradient-descent steps to be dominated by the large numerical magnitude of `fare`, unfairly masking the signals of smaller-scale features.

### Exercise 4 — Version Control Graph
The project follows clean branching:
```
* 554d7e1 (HEAD -> main) Lab 03: solve all 4 home-lab exercises
* b6dd63e Lab 03: complete deliverables
* 5bde9bf Lab 02: complete deliverables - figures, two-source corpus, report
* e8169ab Initial Commit - Lab01 completed
```

---

## 4. Home Assignment — Modular `inspect.py` Utility

The utility `lab01/inspect.py` was developed and validated against two diverse public datasets.

### Dataset 1: Titanic Passengers
```
======================================================================
SOURCE : https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
SHAPE  : 891 rows x 15 columns
======================================================================

--- DATA TYPES ---
survived         int64
pclass           int64
sex                str
age            float64
sibsp            int64
parch            int64
fare           float64
embarked           str
class              str
who                str
adult_male        bool
deck               str
embark_town        str
alive              str
alone             bool
dtype: object

--- MISSING VALUES ---
             missing  percent
deck             688    77.22
age              177    19.87
embarked           2     0.22
embark_town        2     0.22

--- NUMERIC SUMMARY ---
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
======================================================================
```

### Dataset 2: Fisher's Iris Flower Measurements
```
======================================================================
SOURCE : https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
SHAPE  : 150 rows x 5 columns
======================================================================

--- DATA TYPES ---
sepal_length    float64
sepal_width     float64
petal_length    float64
petal_width     float64
species             str
dtype: object

--- MISSING VALUES ---
None.

--- NUMERIC SUMMARY ---
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
======================================================================
```

**Observation on Iris:**
The Iris dataset is completely balanced with 150 rows, 5 columns, zero missing values across all measurements, and compact, well-behaved feature spreads (sepal and petal lengths and widths).

---

## 5. Deliverables Checklist

| Deliverable | Location | Status |
|---|---|---|
| Notebook | `lab01/lab01_setup.ipynb` | Complete with executed outputs |
| Inspection Utility | `lab01/inspect.py` | Implemented and modular |
| Inspection Logs | `lab01/reports/*.txt` | Generated for Titanic and Iris |
| Lab Report | `lab01/report.md` | This document |
| Requirements | `requirements.txt` | Pinned environment specifications |
| Git Exclusions | `.gitignore` | Configured according to course guidelines |
