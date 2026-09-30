# Comprehensive AI Lab Portfolio Guide & Implementation Reference

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Ali Hassan Sherazi  
**Repository:** [dropgaming786-collab/ai-lab](https://github.com/dropgaming786-collab/ai-lab)  
**Environment:** Python 3.11 (`ai-lab` Conda Environment)

---

## 1. Executive Summary & Purpose

This document serves as the complete technical manual and project guide for the entire laboratory portfolio developed in this repository. It covers every laboratory session from foundational environment configuration to multimodal data feature engineering:
- **Lab 01:** Environment Setup (Anaconda, Jupyter, Git), Data Inspection Engine (`inspect.py`), and Statistical Diagnostics.
- **Lab 02:** Polite Web Scraping, Linguistic & Sentiment Feature Extraction (spaCy + TextBlob), Exploratory Data Analysis, and Two-Source Comparative Scraping.
- **Lab 03:** Multimodal Feature Engineering — Bag-of-Words (BoW) & N-Grams for Text, Image Preprocessing Pipelines (Grayscale & Flattening), Curated 20-Sample Two-Class Image Dataset, and Dimensionality Diagnostics.

All source code, notebooks, dataset generators, figure renderers, and persistent model artefacts are organized modularly inside their respective laboratory directories.

---

## 2. Global Repository Layout

```
ai-lab/
│
├── .gitignore                      # Course-standard exclusions (data, environments, checkpoints)
├── requirements.txt                # Pinned dependencies from the ai-lab environment
│
├── docs/
│   └── GUIDE.md                    # This master technical guide
│
├── lab01/                          # LABORATORY 01: Setup & Data Inspection
│   ├── lab01_setup.ipynb           # Fully executed notebook with all outputs
│   ├── inspect.py                  # Standalone data inspection CLI utility
│   ├── report.md                   # Formal report with statistical & Git graph analysis
│   └── reports/                    # Generated inspection logs for public datasets
│       ├── titanic_inspection.txt
│       └── iris_inspection.txt
│
├── lab02/                          # LABORATORY 02: Web Scraping & Linguistic EDA
│   ├── scraping_eda.ipynb          # Executed notebook with all exercises & home tasks
│   ├── articles.csv                # Core scraped corpus with spaCy + TextBlob features
│   ├── articles_raw.csv            # Raw HTML text cache
│   ├── two_source_corpus.csv       # 14-article corpus across two distinct blogs
│   ├── build_deliverables.py       # Automated build script for Lab 02 figures & report
│   ├── report.md                   # Full comparative and linguistic analysis report
│   └── figures/                    # High-resolution exported EDA charts
│       ├── 01_title_length_dist.png
│       ├── 02_sentiment_polarity_dist.png
│       ├── 03_pairplot.png
│       ├── 04_tfidf_terms.png
│       └── 05_home_assignment_comparison.png
│
├── lab03/                          # LABORATORY 03: Text & Image Feature Engineering
│   ├── text_image_features.ipynb   # Executed 38-cell notebook (Part A, Part B, 4 Exercises)
│   ├── images.jpg                  # Reference test image
│   ├── image_utils.py              # Reusable image_to_features() utility function
│   ├── dataset.npz                 # Persistent binary dataset (X: 20x4096, y: 20)
│   ├── build_deliverables.py       # Deliverables builder for core tasks & home assignment
│   ├── solve_exercises.py          # Automated solver for all 4 home-lab exercises
│   ├── report.md                   # Comprehensive feature engineering report
│   ├── data/                       # 20 curated sample images across 2 classes
│   │   ├── cats/                   # 10 cat images (64x64)
│   │   └── dogs/                   # 10 dog images (64x64)
│   └── figures/                    # Exported diagnostic figures
│       ├── 01_bow_matrix.png
│       ├── 02_image_pipeline.png
│       ├── 03_home_assignment_classes.png
│       ├── E1_bigram_matrix.png
│       ├── E2_fourth_document.png
│       └── E3_resize_comparison.png
│
├── Lab_01_Environment_Setup.ipynb  # Root master copy of Lab 01
├── Lab_02_Web_Scraping_and_EDA.ipynb # Root master copy of Lab 02
└── Lab_03_Text_and_Image_Features.ipynb # Root master copy of Lab 03
```

---

## 3. Laboratory 01 Deep Dive: Environment Setup & Data Inspection

### 3.1 Architecture & Objectives
Lab 01 establishes the foundational engineering stack required for scientific Python computing:
1. **Isolated Conda Environment:** Python 3.11 environment (`ai-lab`) registered to `ipykernel`.
2. **First-Look Data Inspection Methodology:** Establishing the mandatory sequence (`head()`, `info()`, `describe()`, and explicit null quantification) before running any downstream transformations.
3. **Version Control Lifecycle:** Configuring local and remote Git tracking, branch management, merge resolution, and graphical history logging.

### 3.2 In-Lab Exercises Solutions
- **Exercise 1 (Kernel Verification):**
  The environment was verified directly via Python runtime inspection:
  ```python
  import sys, platform
  print("Python version   :", sys.version.split()[0])
  print("Interpreter path :", sys.executable)
  print("Running in ai-lab:", "ai-lab" in sys.executable)
  ```
- **Exercise 2 (Missing Value Audit from `info()`):**
  Identified the three columns with null values in the Titanic dataset ($N=891$):
  - `age`: 714 non-null $\rightarrow$ **177 missing** (19.9%)
  - `deck`: 203 non-null $\rightarrow$ **688 missing** (77.2%)
  - `embarked` / `embark_town`: 889 non-null $\rightarrow$ **2 missing** (0.2%)
- **Exercise 3 (Coefficient of Variation Analysis):**
  Computed $CV = |\frac{\sigma}{\mu}|$ across all numeric columns:
  - `fare` exhibited the highest ratio ($CV \approx 1.549$, $\mu=32.20, \sigma=49.69$).
  - **Significance:** In models sensitive to euclidean distance or gradient magnitudes (e.g., k-NN, SVM, unregularized neural networks), features with disproportionately large relative dispersion dominate gradient updates and distance metrics, arbitrarily dwarfing subtle features. Feature scaling (e.g., standardisation or MinMax normalization) is essential.
- **Exercise 4 (Git Graph Inspection):**
  Demonstrated feature branch creation (`git checkout -b lab01/exercises`), committed atomic updates, merged cleanly to `main`, and recorded the output graph.

### 3.3 Home Assignment: `lab01/inspect.py`
A standalone, modular data diagnostic utility was engineered to accept either a local path or an HTTP/HTTPS URL. It outputs:
- File source and tabular dimension dimensions ($M \times N$)
- Datatypes per column
- Sorted missing value tally with exact percentage impact
- Five-number distribution summary for numeric fields
- Tested and verified on two public datasets:
  1. *Titanic Passenger Roster* (`raw.githubusercontent.com/.../titanic.csv`)
  2. *Fisher's Iris Flower Measurements* (`raw.githubusercontent.com/.../iris.csv`)

---

## 4. Laboratory 02 Deep Dive: Web Scraping & Linguistic EDA

### 4.1 Architecture & Objectives
Lab 02 introduces end-to-end data acquisition and unstructured text quantification:
1. **Polite Web Scraping:** Using `requests` and `BeautifulSoup` with browser emulation headers, bounded timeouts, and inter-request throttles (`time.sleep(1)`).
2. **Linguistic Feature Engineering:** Parsing article bodies with spaCy (`en_core_web_sm`) to compute token counts, sentence counts, named entities, and noun densities.
3. **Sentiment Quantisation:** Leveraging TextBlob to compute unbounded Polarity $[-1.0, +1.0]$ and Subjectivity $[0.0, 1.0]$.
4. **Information Retrieval via TF-IDF:** Converting raw strings into informative term-weight matrices.

### 4.2 In-Lab Exercises Solutions
- **Exercise 1 (HTML Selector Analysis):**
  Evaluated custom blog URLs against Blogger standard tags (`h3.post-title` and `div.post-body.entry-content`). Non-standard blogs (e.g., WordPress or static docs) failed to match because their main article bodies are wrapped in semantic `<article>` tags or `<div class="entry-content">` without `post-body`.
- **Exercise 2 (Extended Linguistic Features):**
  Extended the feature extraction pipeline with:
  - `num_verbs`: Count of tokens where `token.pos_ == 'VERB'`
  - `avg_sentence_length`: $\frac{\text{num\_tokens}}{\text{num\_sentences}}$
  Rendered a comprehensive pairwise scatter matrix incorporating the new syntax dimensions.
- **Exercise 3 (KDE Small-Sample Critique):**
  Explained why a Kernel Density Estimate (KDE) over $N=5$ points cannot represent a population: Gaussian smoothing kernels place artificial probability mass beyond the empirical range, creating the false illusion of a continuous, multimodal population when the sample variance is underpowered. Instead, raw data points must be displayed directly via strip plots or rug plots.
- **Exercise 4 (TF-IDF Stop-Word Contrast):**
  Compared $K=20$ terms extracted with `stop_words='english'` versus `stop_words=None`. Without stop-word pruning, syntactic function words (*the, to, and, of, is, in*) dominate total frequency, drowning out domain-specific semantic keywords (*ai, content, tools, prompts*).

### 4.3 Home Assignment: Two-Source Comparative Scraping
- Scraped 14 total articles across two distinct web domains:
  - **Source A (Techncruncher):** Tech-commercial blogging
  - **Source B (Real Python):** In-depth technical developer tutorials
- Extracted and engineered all 10 linguistic metrics into `lab02/two_source_corpus.csv`.
- Generated `lab02/figures/05_home_assignment_comparison.png` visualizing the comparative distributions of sentiment polarity and noun density.

---

## 5. Laboratory 03 Deep Dive: Text & Image Multimodal Features

### 5.1 Architecture & Objectives
Lab 03 examines the transformation of qualitative textual strings and 2D pixel arrays into dense and sparse numerical feature vectors suitable for machine learning algorithms.

### 5.2 Part A — Bag-of-Words (BoW) Text Transformation
- **Vectorization Pipeline:** `CountVectorizer` converts an $N$-document collection into an $N \times V$ Document-Term Matrix (DTM).
- **Expanded Corpus ($N=40$ Documents):**
  The corpus was expanded by adding 37 distinct documents (lines) to the original 3, forming a rich 40-sentence dataset.
  - **Feature Space:** $40 \text{ documents} \times 210 \text{ unique vocabulary terms}$.
  - **Total Matrix Cells:** $40 \times 210 = 8,400$.
  - **Zero Cells:** $8,083$ cells.
  - **Sparsity:** **$96.2\%$** zero entries, demonstrating how BoW sparsity rapidly climbs toward enterprise levels ($>99\%$) as corpus diversity grows.
- **Silent Design Decisions:**
  1. Automated lowercase conversion eliminates case sensitivity.
  2. The default token pattern `(?u)\b\w\w+\b` drops all single-character tokens (e.g., `"a"` is silently discarded).

### 5.3 Part B — Image Feature Engineering
- **Grayscale Dimensionality Reduction:** Collapses three RGB channels into one luminance channel using standard weighted perceptual luminance ($Y \approx 0.299R + 0.587G + 0.114B$).
  - For a $148 \times 148$ image: $148 \times 148 \times 3 = 65,712$ features $\rightarrow 148 \times 148 \times 1 = 21,904$ features (exact $3\times$ compression).
- **Pixel Flattening:** Serializing the 2D grid into a 1D vector $\mathbb{R}^{21904}$.
  - **Trade-off:** Completely destroys 2D spatial locality. Pixel $(r, c)$ is mapped to index $r \cdot W + c$, removing neighbor proximity that Convolutional Neural Networks (CNNs) preserve.

### 5.4 Home-Lab Exercises Solutions (All 4 Solved)
1. **Exercise 1 (N-Gram Expansion):**
   - Configured `CountVectorizer(ngram_range=(1,2))`.
   - The vocabulary expanded from **11 unigrams** to **24 features** (+13 bigrams: *day is, good the, is good, is great, is shining, is wonderful, shining today, sun is, sunny day, the sun, the weather, weather is, wonderful day*).
   - Saved heatmap to `lab03/figures/E1_bigram_matrix.png`.
2. **Exercise 2 (Document Length Bias & 4th Repetitive Document):**
   - Added `"Sun sun sun sun sun."` as Document 4.
   - Matrix expanded to $4 \times 11$. Document 4 registered a count of **5** for `sun`, completely dominating unigram frequency despite containing zero semantic depth.
   - Proved why TF-IDF length normalization is mandatory. Saved heatmap to `lab03/figures/E2_fourth_document.png`.
3. **Exercise 3 (Resolution vs. Information Loss):**
   - Evaluated downsampling on `images.jpg`:
     - Original: $739 \times 415 \rightarrow 306,685$ features
     - $64 \times 64 \rightarrow 4,096$ features
     - $32 \times 32 \rightarrow 1,024$ features
   - Detail analysis: At $32 \times 32$, fine textural gradients, sharp edge boundaries, and typography are obliterated. Only coarse macro-intensity blocks survive. Saved visual comparison to `lab03/figures/E3_resize_comparison.png`.
4. **Exercise 4 (`image_to_features` Standardizer):**
   - Implemented in `lab03/image_utils.py`:
     ```python
     def image_to_features(path: str, size: tuple = (64, 64)) -> np.ndarray:
         img = Image.open(path).convert("RGB").resize(size)
         gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
         return gray.flatten()
     ```
   - Verified that regardless of input resolution (e.g., $415 \times 739$ or $64 \times 64$), every processed output is guaranteed to be a normalized float vector of length **4,096**.

### 5.5 Home Assignment: 20-Sample Two-Class Image Dataset
- Curated 20 balanced image samples across two classes (`cats` and `dogs`) into `lab03/data/`.
- Executed batch feature extraction, generating feature matrix $X \in \mathbb{R}^{20 \times 4096}$ and target vector $y \in \{0, 1\}^{20}$.
- Saved to persistent binary format: `lab03/dataset.npz`.
- **Curse of Dimensionality Assessment:**
  With 20 samples in a 4,096-dimensional space, the feature-to-sample ratio is $205:1$. In this sparse hypercube, pairwise Euclidean distances converge, causing k-NN classifiers to fail and linear classifiers to memorize training noise (severe overfitting). Recommended remedies: PCA compression to 10–30 components, gathering $\ge 50,000$ samples, or using convolutional kernels.

---

## 6. Verification and Reproduction Instructions

To execute and verify the complete workflow on any machine running Windows, macOS, or Linux:

### 1. Environment Activation
```powershell
conda activate ai-lab
```

### 2. Run Lab 01 Data Inspection
```powershell
python "lab01/inspect.py"
```

### 3. Build Lab 02 Deliverables & Two-Source Scrape
```powershell
python "lab02/build_deliverables.py"
```

### 4. Build Lab 03 Deliverables & Exercises
```powershell
python "lab03/build_deliverables.py"
python "lab03/solve_exercises.py"
```

---

*This guide was generated to accompany commit updates to the `ai-lab` repository portfolio.*
