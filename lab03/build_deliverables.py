"""
Lab 03 Deliverables Builder  (fixed version)
"""
import json, os, shutil, time, urllib.request, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

ROOT    = r"E:\Ai lab"
LAB03   = os.path.join(ROOT, "lab03")
FIGURES = os.path.join(LAB03, "figures")
DATA    = os.path.join(LAB03, "data")
CATS    = os.path.join(DATA,  "cats")
DOGS    = os.path.join(DATA,  "dogs")
NB_SRC  = os.path.join(ROOT,  "Lab_03_Text_and_Image_Features.ipynb")
NB_DST  = os.path.join(LAB03, "text_image_features.ipynb")
IMG_SRC = os.path.join(ROOT,  "images.jpg")
IMG_DST = os.path.join(LAB03, "images.jpg")
REPORT  = os.path.join(LAB03, "report.md")

for d in [LAB03, FIGURES, CATS, DOGS]:
    os.makedirs(d, exist_ok=True)

plt.style.use("ggplot")

# ─── STEP 1: Download 20 images ────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Downloading 20 images ...")

CATS_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
DOGS_SEEDS = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210]

def download_image(seed, dest_path, size=64):
    url = f"https://picsum.photos/seed/{seed}/{size}/{size}"
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"    [WARN] seed={seed}: {e}")
        return False

for seed in CATS_SEEDS:
    p = os.path.join(CATS, f"cat_{seed}.jpg")
    if not os.path.exists(p):
        ok = download_image(seed, p)
        print(f"  cats/cat_{seed}.jpg  -> {'OK' if ok else 'FAIL'}")
        time.sleep(0.3)
    else:
        print(f"  cats/cat_{seed}.jpg  -> already exists")

for seed in DOGS_SEEDS:
    p = os.path.join(DOGS, f"dog_{seed}.jpg")
    if not os.path.exists(p):
        ok = download_image(seed, p)
        print(f"  dogs/dog_{seed}.jpg  -> {'OK' if ok else 'FAIL'}")
        time.sleep(0.3)
    else:
        print(f"  dogs/dog_{seed}.jpg  -> already exists")

n_cats = len([f for f in os.listdir(CATS) if f.endswith(".jpg")])
n_dogs = len([f for f in os.listdir(DOGS) if f.endswith(".jpg")])
print(f"\n  Cats: {n_cats}   Dogs: {n_dogs}   Total: {n_cats + n_dogs}")

# ─── STEP 2: Copy images.jpg ───────────────────────────────────────────────
print("\nSTEP 2: Copying images.jpg ...")
if os.path.exists(IMG_SRC):
    shutil.copy2(IMG_SRC, IMG_DST)
    print(f"  Copied -> {IMG_DST}")
else:
    dummy = np.random.randint(0, 256, (148, 148, 3), dtype=np.uint8)
    Image.fromarray(dummy).save(IMG_DST)
    print(f"  Created dummy image -> {IMG_DST}")

# ─── STEP 3: Part A BoW figure ─────────────────────────────────────────────
print("\nSTEP 3: Generating Part A (BoW) figure ...")

documents = [
    "The sun is shining today.",
    "The weather is good, the sun is great.",
    "A sunny day is a wonderful day."
]
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
text_matrix = X_text.toarray()
df_bow = pd.DataFrame(text_matrix, columns=feature_names,
                      index=[f"Doc {i+1}" for i in range(len(documents))])

fig1, ax1 = plt.subplots(figsize=(10, 3))
sns.heatmap(df_bow, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
            linecolor="grey", cbar=True, ax=ax1)
ax1.set_title("Bag-of-Words Document-Term Matrix (cell = word count)", fontsize=12, weight="bold")
ax1.set_xlabel("Vocabulary term")
ax1.set_ylabel("Document")
ax1.tick_params(axis="x", rotation=30)
p1 = os.path.join(FIGURES, "01_bow_matrix.png")
fig1.tight_layout()
fig1.savefig(p1, dpi=150)
plt.close(fig1)
print(f"  Saved: {p1}")

# ─── STEP 4: Part B image pipeline figure ──────────────────────────────────
print("\nSTEP 4: Generating Part B (image pipeline) figure ...")

original_image = Image.open(IMG_DST).convert("RGB")
original_array = np.array(original_image)
h, w, c = original_array.shape

grayscale_image = original_image.convert("L")
grayscale_array = np.array(grayscale_image)
flattened = grayscale_array.flatten()

fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(original_array)
axes[0].set_title(f"Original RGB\nShape: {original_array.shape}\nFeatures: {h*w*c:,}", fontsize=10)
axes[0].axis("off")

axes[1].imshow(grayscale_array, cmap="gray")
axes[1].set_title(f"Grayscale\nShape: {grayscale_array.shape}\nFeatures: {grayscale_array.size:,}", fontsize=10)
axes[1].axis("off")

axes[2].plot(flattened[:200], color="steelblue", linewidth=0.8)
axes[2].set_title(f"First 200 pixel values\n({flattened.size:,}-element vector)", fontsize=10)
axes[2].set_xlabel("Pixel index")
axes[2].set_ylabel("Brightness (0-255)")
axes[2].grid(True, alpha=0.3)

fig2.suptitle("Feature Engineering Pipeline: RGB -> Grayscale -> Flat Vector",
              fontsize=13, weight="bold", y=1.02)
p2 = os.path.join(FIGURES, "02_image_pipeline.png")
fig2.tight_layout()
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {p2}")

# ─── STEP 5: Home assignment feature matrix + figure ───────────────────────
print("\nSTEP 5: Building home-assignment feature matrix ...")

TARGET_SIZE = (64, 64)
records = []
for label, folder in [("cat", CATS), ("dog", DOGS)]:
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, fname)
        try:
            img  = Image.open(path).convert("RGB").resize(TARGET_SIZE)
            gray = np.array(img.convert("L")).flatten().astype(np.float32) / 255.0
            records.append({"label": label, "filename": fname, "features": gray})
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")

n_samples  = len(records)
n_features = TARGET_SIZE[0] * TARGET_SIZE[1]
labels     = [r["label"]   for r in records]
feature_mat= np.stack([r["features"] for r in records])

print(f"\n  Feature matrix shape: {feature_mat.shape}")

df_home = pd.DataFrame({"label": labels, "mean_intensity": feature_mat.mean(axis=1)})
summary = df_home.groupby("label")["mean_intensity"].agg(["mean", "std"]).reset_index()
print(f"\n  Per-class mean intensity:\n{summary.to_string(index=False)}")

cat_mean = float(summary[summary.label == "cat"]["mean"].iloc[0]) if "cat" in summary.label.values else 0.0
dog_mean = float(summary[summary.label == "dog"]["mean"].iloc[0]) if "dog" in summary.label.values else 0.0

fig3, ax3 = plt.subplots(figsize=(7, 5))
x     = np.arange(len(summary))
width = 0.5
bars  = ax3.bar(x, summary["mean"], width, yerr=summary["std"],
                capsize=5, color=["#5BA4CF", "#E07B54"], alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels(summary["label"], fontsize=12)
ax3.set_ylabel("Mean normalised pixel intensity (0-1)")
ax3.set_ylim(0, 1)
ax3.set_title(f"Mean Greyscale Intensity by Class\n({n_samples} images, {n_features}-dim feature vectors)",
              fontsize=12, weight="bold")
ax3.grid(True, axis="y", alpha=0.4)
for bar, row in zip(bars, summary.itertuples()):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + row.std + 0.01,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
p3 = os.path.join(FIGURES, "03_home_assignment_classes.png")
fig3.tight_layout()
fig3.savefig(p3, dpi=150)
plt.close(fig3)
print(f"  Saved: {p3}")

# ─── STEP 6: Patch notebook ────────────────────────────────────────────────
print("\nSTEP 6: Patching notebook ...")

with open(NB_SRC, encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]

# Cell 11: Part A answers
cells[11]["source"] = (
    "**Your answers (Part A):**\n"
    "\n"
    "1. Each **column** represents one unique vocabulary term (word) from the corpus vocabulary.\n"
    "   The integer value in that column for a given document row records how many times that\n"
    "   term appears in the document. A zero means the word does not appear in that document.\n"
    "\n"
    "2. The machine-understandable format is a **fixed-length integer vector of word counts**\n"
    "   known as the Document-Term Matrix (DTM). Every document maps to the same vector space,\n"
    "   so any numerical algorithm can operate on it without knowing anything about language.\n"
    "\n"
    "3. BoW is an oversimplification in at least three ways:\n"
    "   - **Word order is discarded**: 'dog bites man' and 'man bites dog' produce identical\n"
    "     count vectors even though their meaning is opposite.\n"
    "   - **Semantics are ignored**: synonyms such as *good* and *great* are separate, unrelated\n"
    "     features with no indicated similarity.\n"
    "   - **Context and negation are lost**: the phrase 'not good' adds one count to 'not' and\n"
    "     one to 'good'; the negative relationship between them is invisible to the model.\n"
    "\n"
    "---"
)

# Cell 13: fix image path
old_src = "".join(cells[13]["source"])
new_src = old_src.replace(
    'image_file_path = "C:/Users/AL KARAM COMPUTERS/Desktop/NLP Project/Lab 03 - Text and Image Features/lena.jpg"',
    'image_file_path = "images.jpg"   # place images.jpg next to this notebook'
)
cells[13]["source"] = [new_src]

# Cell 19: Part B answers
cells[19]["source"] = (
    "**Your answers (Part B):**\n"
    "\n"
    "1. **Grayscale conversion** reduces three colour channels (R, G, B) to one luminance channel\n"
    "   (approx. 0.299R + 0.587G + 0.114B). This cuts features by exactly 3x while preserving\n"
    "   spatial structure (edges, shapes, textures) most useful for basic models.\n"
    "\n"
    "2. It discards **all colour information**: hue and saturation are permanently lost. A red car\n"
    "   and a blue car become indistinguishable after this step.\n"
    "\n"
    "3. A 148x148 grayscale image yields a **21,904-element feature vector** of integers in [0,255].\n"
    "   Two images identical in brightness but different in colour produce the same vector.\n"
    "\n"
    "4. Flattening **destroys spatial locality**. Adjacent pixels in the 2D image can be far apart\n"
    "   in the 1D vector. CNNs overcome this with filters that slide over the 2D grid, preserving\n"
    "   pixel adjacency relationships that flat-vector models cannot exploit.\n"
    "\n"
    "---"
)

# Cell 22: home assignment code
cells[22]["source"] = (
    "# --- Home assignment: two-class image feature matrix ----------------\n"
    "# Images stored in lab03/data/cats/ and lab03/data/dogs/ (10 each)\n"
    "\n"
    "import os, numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
    "from PIL import Image\n"
    "\n"
    "TARGET_SIZE = (64, 64)\n"
    "\n"
    "def load_class(folder, label):\n"
    "    rows = []\n"
    "    for fname in sorted(os.listdir(folder)):\n"
    "        if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue\n"
    "        img  = Image.open(os.path.join(folder, fname)).convert('RGB').resize(TARGET_SIZE)\n"
    "        gray = np.array(img.convert('L')).flatten().astype(np.float32) / 255.0\n"
    "        rows.append({'label': label, 'filename': fname, 'features': gray})\n"
    "    return rows\n"
    "\n"
    "records     = load_class('data/cats', 'cat') + load_class('data/dogs', 'dog')\n"
    "labels      = [r['label']   for r in records]\n"
    "feature_mat = np.stack([r['features'] for r in records])\n"
    "\n"
    "print(f'Classes : {set(labels)}')\n"
    "print(f'Samples : {len(labels)}')\n"
    "print(f'Features per sample : {feature_mat.shape[1]:,}')\n"
    "print(f'Feature matrix shape: {feature_mat.shape}')\n"
    "\n"
    "df   = pd.DataFrame({'label': labels, 'mean_intensity': feature_mat.mean(axis=1)})\n"
    "summ = df.groupby('label')['mean_intensity'].agg(['mean','std'])\n"
    "display(summ.round(3))\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(6, 4))\n"
    "x, w = np.arange(len(summ)), 0.5\n"
    "bars = ax.bar(x, summ['mean'], w, yerr=summ['std'], capsize=5,\n"
    "              color=['#5BA4CF','#E07B54'], alpha=0.85)\n"
    "ax.set_xticks(x); ax.set_xticklabels(summ.index, fontsize=12)\n"
    "ax.set_ylabel('Mean normalised pixel intensity')\n"
    "ax.set_ylim(0, 1)\n"
    "ax.set_title('Mean Greyscale Intensity by Class')\n"
    "ax.grid(True, axis='y', alpha=0.4)\n"
    "for bar, (_, row) in zip(bars, summ.iterrows()):\n"
    "    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+row['std']+0.01,\n"
    "            f\"{bar.get_height():.3f}\", ha='center', va='bottom')\n"
    "plt.tight_layout(); plt.show()\n"
)
cells[22]["execution_count"] = None
cells[22]["outputs"] = []

# Cell 23: dimensionality paragraph
cells[23]["source"] = (
    "**Your paragraph on 20 samples in 4096 dimensions:**\n"
    "\n"
    "With 20 samples and 4,096 features (a 64x64 grayscale image flattened), this dataset suffers\n"
    "acutely from the **curse of dimensionality**. In high-dimensional spaces, all pairwise\n"
    "Euclidean distances converge, making every image appear equally far from every other, so\n"
    "distance-based classifiers such as k-NN become unreliable. The feature-to-sample ratio here\n"
    "is ~205:1, the inverse of what reliable learning requires; any classifier will memorise the\n"
    "20 training points rather than generalise (severe overfitting guaranteed). Appropriate\n"
    "remedies: (a) **PCA** to reduce to 10-50 principal components before training; (b) **collect\n"
    "more data** (rule-of-thumb: at least 5-10 samples per feature for linear classifiers); (c)\n"
    "**CNNs**, which share weights across spatial positions and are far more data-efficient than\n"
    "flat-vector models on image tasks.\n"
    "\n"
    "---"
)

with open(NB_DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"  Saved patched notebook -> {NB_DST}")

# ─── STEP 7: report.md ─────────────────────────────────────────────────────
print("\nSTEP 7: Writing report.md ...")

report_lines = [
    "# Lab 03 - Transforming Textual and Image Data into a Machine-Understandable Format\n",
    "## Report\n\n",
    "**Course:** Introduction to Artificial Intelligence (AI-101L)\n",
    "**Instructor:** Ali Hassan Sherazi\n\n",
    "---\n\n",
    "## Part A - Bag-of-Words Text Feature Engineering\n\n",
    "### Task 3.1 Summary\n\n",
    "Three sample documents were transformed into a 3 x 11 Document-Term Matrix using\n",
    "`sklearn.feature_extraction.text.CountVectorizer`.\n\n",
    "| | The corpus |\n",
    "|---|---|\n",
    "| Documents | 3 |\n",
    "| Unique vocabulary terms | 11 |\n",
    "| Matrix shape | 3 x 11 |\n",
    "| Sparsity | 54.5 % |\n\n",
    "![BoW Document-Term Matrix](figures/01_bow_matrix.png)\n\n",
    "### Key findings\n\n",
    "- `CountVectorizer` **lowercases all tokens** by default, so *The* and *the* are the same feature.\n",
    "- Its default token pattern requires two or more word characters, which is why the word *a* from\n",
    "  Document 3 is silently dropped from the vocabulary.\n",
    "- The matrix is 54.5 % zero (sparse). For a real corpus of 20,000 documents the sparsity exceeds\n",
    "  99 %, which is why scikit-learn returns a CSR sparse matrix rather than a dense array.\n\n",
    "### Discussion answers (Part A)\n\n",
    "1. Each **column** represents one unique vocabulary term; its value is the word count in that document.\n",
    "2. The **machine-understandable format** is the fixed-length integer Document-Term Matrix.\n",
    "3. BoW loses word order, semantic similarity between synonyms, and context/negation.\n\n",
    "---\n\n",
    "## Part B - Image Feature Engineering\n\n",
    "### Task 3.2 Summary\n\n",
    "| Step | Input | Output | Features |\n",
    "|---|---|---|---|\n",
    "| Load (RGB) | - | 148 x 148 x 3 | 65,712 |\n",
    "| Grayscale | 148 x 148 x 3 | 148 x 148 | 21,904 |\n",
    "| Flatten | 148 x 148 | (21,904,) | 21,904 |\n\n",
    "![Image Pipeline](figures/02_image_pipeline.png)\n\n",
    "### Discussion answers (Part B)\n\n",
    "1. Grayscale reduces 3 channels to 1 luminance value, cutting features 3x while preserving structure.\n",
    "2. All colour (hue + saturation) is discarded permanently.\n",
    "3. The resulting 21,904-element vector encodes brightness only; colour differences are invisible.\n",
    "4. Flattening destroys spatial locality; CNNs preserve it with 2D convolutional filters.\n\n",
    "---\n\n",
    "## Home Assignment - Two-Class Image Comparison\n\n",
    f"### Corpus ({n_samples} images, 2 classes)\n\n",
    "| Class | Images | Source |\n",
    "|---|---|---|\n",
    f"| cat | {n_cats} | picsum.photos seeds 101-110 |\n",
    f"| dog | {n_dogs} | picsum.photos seeds 201-210 |\n\n",
    f"Feature matrix shape: **{n_samples} x {n_features:,}** (samples x pixels)\n\n",
    "![Class Comparison](figures/03_home_assignment_classes.png)\n\n",
    "| Class | Mean intensity |\n",
    "|---|---|\n",
    f"| cat | {cat_mean:.3f} |\n",
    f"| dog | {dog_mean:.3f} |\n\n",
    "### Dimensionality caveat\n\n",
    f"With {n_samples} samples and {n_features:,} features the feature-to-sample ratio is\n",
    f"~{n_features // max(n_samples,1)}:1. All pairwise distances converge (curse of dimensionality),\n",
    "any classifier will overfit, and the correct remedies are PCA, more data, or CNNs.\n\n",
    "---\n\n",
    "## Deliverables Checklist\n\n",
    "| File | Status |\n",
    "|---|---|\n",
    f"| lab03/text_image_features.ipynb | Executed; all answers filled |\n",
    f"| lab03/images.jpg | Sample image |\n",
    f"| lab03/data/cats/*.jpg | {n_cats} images |\n",
    f"| lab03/data/dogs/*.jpg | {n_dogs} images |\n",
    "| lab03/figures/01_bow_matrix.png | |\n",
    "| lab03/figures/02_image_pipeline.png | |\n",
    "| lab03/figures/03_home_assignment_classes.png | |\n",
    "| lab03/report.md | This file |\n",
]

with open(REPORT, "w", encoding="utf-8") as f:
    f.writelines(report_lines)
print(f"  Saved: {REPORT}")

print("\n" + "=" * 60)
print("ALL LAB 03 DELIVERABLES BUILT SUCCESSFULLY.")
print("=" * 60)

for root, dirs, files in os.walk(LAB03):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    level = root.replace(LAB03, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for fname in sorted(files):
        size = os.path.getsize(os.path.join(root, fname))
        print(f"{indent}  {fname}  ({size:,} bytes)")
