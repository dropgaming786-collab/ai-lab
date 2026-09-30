"""
Lab 03 - Solve all Home-Lab Exercises + patch notebook + push to GitHub
"""

import json, os, shutil, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

ROOT   = r"E:\Ai lab"
LAB03  = os.path.join(ROOT, "lab03")
FIGS   = os.path.join(LAB03, "figures")
NB_DST = os.path.join(LAB03, "text_image_features.ipynb")
IMG    = os.path.join(LAB03, "images.jpg")

os.makedirs(FIGS, exist_ok=True)
plt.style.use("ggplot")

# ─── Exercise 1: BoW with bigrams ─────────────────────────────────────────
print("="*60)
print("EXERCISE 1: bigram CountVectorizer")

documents = [
    "The sun is shining today.",
    "The weather is good, the sun is great.",
    "A sunny day is a wonderful day."
]

# Unigram baseline
vec1 = CountVectorizer()
X1   = vec1.fit_transform(documents)
n_uni = X1.shape[1]

# Bigram
vec2 = CountVectorizer(ngram_range=(1,2))
X2   = vec2.fit_transform(documents)
n_bi  = X2.shape[1]
bigrams_only = sorted(set(vec2.get_feature_names_out()) - set(vec1.get_feature_names_out()))

print(f"  Unigrams only : {n_uni} features")
print(f"  Unigrams+Bigrams : {n_bi} features")
print(f"  New bigrams added: {bigrams_only}")

# Figure E1 - bigram matrix heatmap
df_bi = pd.DataFrame(X2.toarray(),
                     columns=vec2.get_feature_names_out(),
                     index=[f"Doc {i+1}" for i in range(len(documents))])
fig1, ax1 = plt.subplots(figsize=(14, 3))
import seaborn as sns
sns.heatmap(df_bi, annot=True, fmt="d", cmap="Oranges", linewidths=0.4,
            linecolor="grey", cbar=False, ax=ax1, annot_kws={"size": 8})
ax1.set_title(f"Bigram BoW Matrix  ({n_bi} features vs {n_uni} unigrams)",
              fontsize=12, weight="bold")
ax1.tick_params(axis="x", rotation=45, labelsize=7)
ax1.set_ylabel("Document")
fig1.tight_layout()
p_e1 = os.path.join(FIGS, "E1_bigram_matrix.png")
fig1.savefig(p_e1, dpi=150)
plt.close(fig1)
print(f"  Saved: {p_e1}")

ex1_answer = (
    "**Exercise 1 — Bigram vocabulary:**\n\n"
    f"With `CountVectorizer(ngram_range=(1,2))` the vocabulary grows from **{n_uni} unigrams** to\n"
    f"**{n_bi} features** — an increase of {n_bi - n_uni} new bigram tokens.\n\n"
    "New bigrams added: `" + "`, `".join(bigrams_only) + "`.\n\n"
    "The growth occurs because every adjacent pair of tokens in each document is added as a new\n"
    "feature in addition to the individual words. For a document with $k$ tokens after tokenisation,\n"
    "up to $k-1$ new bigrams can be created. Bigrams capture short-range word-order information\n"
    "that the unigram model discards — for example the bigram `is good` conveys a positive\n"
    "relationship that the individual tokens `is` and `good` do not. However, they also expand\n"
    "the vocabulary rapidly, increasing sparsity further.\n"
)

# ─── Exercise 2: 4th document with "sun" × 5 ──────────────────────────────
print("\nEXERCISE 2: 4th document — raw counts favour long documents")

documents4 = documents + ["Sun sun sun sun sun."]
vec3 = CountVectorizer()
X3   = vec3.fit_transform(documents4)
mat3 = X3.toarray()
feat3 = vec3.get_feature_names_out()

df4 = pd.DataFrame(mat3, columns=feat3,
                   index=[f"Doc {i+1}" for i in range(len(documents4))])
print(f"  New matrix shape: {df4.shape}")
print(f"  Doc 4 'sun' count: {mat3[3, list(feat3).index('sun') if 'sun' in feat3 else 0]}")

fig2, ax2 = plt.subplots(figsize=(12, 4))
sns.heatmap(df4, annot=True, fmt="d", cmap="Blues", linewidths=0.4,
            linecolor="grey", cbar=True, ax=ax2, annot_kws={"size": 9})
ax2.set_title("BoW with 4th Document (sun x5) — Raw Counts Dominate",
              fontsize=12, weight="bold")
ax2.tick_params(axis="x", rotation=35, labelsize=9)
fig2.tight_layout()
p_e2 = os.path.join(FIGS, "E2_fourth_document.png")
fig2.savefig(p_e2, dpi=150)
plt.close(fig2)
print(f"  Saved: {p_e2}")

ex2_answer = (
    "**Exercise 2 — Adding a 4th document with *sun* repeated 5 times:**\n\n"
    "Adding `\"Sun sun sun sun sun.\"` changes the matrix shape from (3,11) to (4,11).\n"
    "Document 4 gets a count of **5** for the term `sun`, compared to 1 in Doc 1 and Doc 2.\n\n"
    "![4th document matrix](figures/E2_fourth_document.png)\n\n"
    "**Why raw counts favour long documents:** A document that simply repeats the same word\n"
    "many times scores higher for that term than a longer, richer document where the word\n"
    "appears fewer times. If you compute cosine similarity or dot-product similarity on raw\n"
    "count vectors, the repetitive document will appear more 'related' to other documents\n"
    "mentioning `sun` than it actually is. The standard remedy is **TF-IDF normalisation**,\n"
    "which divides the raw count by the document length (TF) and by how common the word is\n"
    "across the whole corpus (IDF), making the representation length-invariant.\n"
)

# ─── Exercise 3: resize to 64×64 and 32×32 ────────────────────────────────
print("\nEXERCISE 3: Resize image to 64x64 and 32x32")

orig = Image.open(IMG).convert("RGB")
orig_arr = np.array(orig.convert("L"))

img64 = orig.resize((64,  64)).convert("L")
img32 = orig.resize((32,  32)).convert("L")

arr64 = np.array(img64)
arr32 = np.array(img32)

vec_orig = orig_arr.flatten()
vec64    = arr64.flatten()
vec32    = arr32.flatten()

print(f"  Original: {orig_arr.shape}  -> {len(vec_orig):,} features")
print(f"  64x64   : {arr64.shape}  -> {len(vec64):,} features")
print(f"  32x32   : {arr32.shape} -> {len(vec32):,} features")

fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))
axes3[0].imshow(orig_arr, cmap="gray")
axes3[0].set_title(f"Original ({orig_arr.shape[0]}x{orig_arr.shape[1]})\n{len(vec_orig):,} features", fontsize=10)
axes3[0].axis("off")
axes3[1].imshow(arr64, cmap="gray")
axes3[1].set_title(f"64x64 resize\n{len(vec64):,} features", fontsize=10)
axes3[1].axis("off")
axes3[2].imshow(arr32, cmap="gray")
axes3[2].set_title(f"32x32 resize\n{len(vec32):,} features", fontsize=10)
axes3[2].axis("off")
fig3.suptitle("Image Resize Comparison: Feature Count vs Detail", fontsize=12, weight="bold")
fig3.tight_layout()
p_e3 = os.path.join(FIGS, "E3_resize_comparison.png")
fig3.savefig(p_e3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved: {p_e3}")

ex3_answer = (
    f"**Exercise 3 — Resizing to 64x64 and 32x32:**\n\n"
    f"| Version | Shape | Flattened vector length |\n"
    f"|---|---|---|\n"
    f"| Original | {orig_arr.shape[0]}x{orig_arr.shape[1]} | {len(vec_orig):,} |\n"
    f"| 64x64 resize | 64x64 | {len(vec64):,} |\n"
    f"| 32x32 resize | 32x32 | {len(vec32):,} |\n\n"
    f"![Resize comparison](figures/E3_resize_comparison.png)\n\n"
    f"**What is no longer visible at 32x32:** Fine-grained texture, thin edges, small text,\n"
    f"and subtle intensity gradients are lost. The 32x32 image retains only the coarsest\n"
    f"block-level structure — general regions of light and dark — because downsampling\n"
    f"averages neighbouring pixels together, discarding high-frequency detail. At 64x64 the\n"
    f"main shapes are still recognisable but fine details are already softened. This\n"
    f"illustrates the resolution-vs-dimensionality trade-off: smaller images are cheaper to\n"
    f"process and generalise better with limited data, but they permanently discard fine detail.\n"
)

# ─── Exercise 4: image_to_features function ───────────────────────────────
print("\nEXERCISE 4: image_to_features function")

# Write image_utils.py
utils_path = os.path.join(LAB03, "image_utils.py")
utils_code = '''"""
image_utils.py
--------------
Utility for converting any image to a fixed-length grayscale feature vector.
"""
import numpy as np
from PIL import Image


def image_to_features(path: str, size: tuple = (64, 64)) -> np.ndarray:
    """Load an image, resize it, convert to grayscale and flatten.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.
    size : tuple of int, optional
        Target (width, height) in pixels.  Default is (64, 64).

    Returns
    -------
    np.ndarray, shape (size[0] * size[1],), dtype float32
        Normalised pixel values in the range [0, 1].
    """
    img = Image.open(path).convert("RGB").resize(size)
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    return gray.flatten()
'''

with open(utils_path, "w", encoding="utf-8") as f:
    f.write(utils_code)
print(f"  Saved: {utils_path}")

# Verify same vector length for 3 different images
import sys
sys.path.insert(0, LAB03)
from image_utils import image_to_features

# Use our 3 downloaded images of different original sizes
test_imgs = [
    os.path.join(LAB03, "images.jpg"),
    os.path.join(LAB03, "data", "cats", "cat_101.jpg"),
    os.path.join(LAB03, "data", "dogs", "dog_201.jpg"),
]

print("\n  Verification — same output length for 3 different input images:")
lengths = []
for p in test_imgs:
    native = Image.open(p).size
    v = image_to_features(p, size=(64, 64))
    lengths.append(len(v))
    print(f"    {os.path.basename(p):25s}  native={native}  ->  vector len={len(v)}")

assert len(set(lengths)) == 1, "Vector lengths differ!"
print(f"\n  All vectors have length {lengths[0]}  [assertion passed]")

ex4_answer = (
    "**Exercise 4 — `image_to_features(path, size)` function:**\n\n"
    "The function is implemented in `image_utils.py`:\n\n"
    "```python\n"
    "def image_to_features(path: str, size: tuple = (64, 64)) -> np.ndarray:\n"
    '    """Load, resize, grayscale, flatten -> fixed-length float32 vector."""\n'
    "    img = Image.open(path).convert('RGB').resize(size)\n"
    "    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0\n"
    "    return gray.flatten()\n"
    "```\n\n"
    "**Verification — same vector length for 3 differently-sized input images:**\n\n"
    "| Image | Native size | Output vector length |\n"
    "|---|---|---|\n"
)
for p, native_len in zip(test_imgs, lengths):
    native = Image.open(p).size
    ex4_answer += f"| `{os.path.basename(p)}` | {native} | **{native_len:,}** |\n"
ex4_answer += (
    "\nAll three return vectors of length **4,096** regardless of original size,\n"
    "because `resize(size)` forces every image to 64x64 pixels before flattening.\n"
    "This is the key guarantee that allows a model to process a batch of images with\n"
    "different native resolutions — every sample ends up with the same number of features.\n"
)

# ─── Patch the notebook ────────────────────────────────────────────────────
print("\nPatching notebook with exercise answers ...")

with open(NB_DST, encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]

# Cell 20 is the Home-Lab Exercises question cell — add answer cells after it
# Build answer cells
def make_md_cell(source_str, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [source_str]
    }

def make_code_cell(source_str, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [source_str]
    }

# Find cell 20 (Home-Lab Exercises markdown)
# Insert answer cells after it
ex_code_e1 = (
    "# Exercise 1: BoW with bigrams\n"
    "from sklearn.feature_extraction.text import CountVectorizer\n"
    "import pandas as pd\n\n"
    "documents = [\n"
    '    "The sun is shining today.",\n'
    '    "The weather is good, the sun is great.",\n'
    '    "A sunny day is a wonderful day."\n'
    "]\n\n"
    "vec_uni = CountVectorizer()\n"
    "vec_bi  = CountVectorizer(ngram_range=(1,2))\n"
    "X_uni = vec_uni.fit_transform(documents)\n"
    "X_bi  = vec_bi.fit_transform(documents)\n\n"
    "print(f'Unigrams only     : {X_uni.shape[1]} features')\n"
    "print(f'Unigrams+Bigrams  : {X_bi.shape[1]} features')\n"
    "print(f'New bigrams added : {X_bi.shape[1] - X_uni.shape[1]}')\n"
    "print()\n"
    "new_bigrams = sorted(set(vec_bi.get_feature_names_out()) - set(vec_uni.get_feature_names_out()))\n"
    "print('New bigram tokens:', new_bigrams)\n"
)

ex_code_e2 = (
    "# Exercise 2: Add 4th document repeating 'sun' x5\n"
    "documents4 = documents + ['Sun sun sun sun sun.']\n"
    "vec4 = CountVectorizer()\n"
    "X4   = vec4.fit_transform(documents4)\n"
    "mat4 = X4.toarray()\n"
    "feat4 = vec4.get_feature_names_out()\n\n"
    "df4 = pd.DataFrame(mat4, columns=feat4,\n"
    "                   index=[f'Doc {i+1}' for i in range(len(documents4))])\n"
    "print('Updated Document-Term Matrix (4 docs):')\n"
    "display(df4)\n"
    "print(f\"\\nDoc 4 'sun' count: {mat4[3, list(feat4).index('sun')]}\")\n"
    "print('Observe: Doc 4 dominates the sun column despite being semantically vacuous.')\n"
)

ex_code_e3 = (
    "# Exercise 3: Resize to 64x64 and 32x32\n"
    "from PIL import Image\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n\n"
    "img_path = 'images.jpg'\n"
    "orig = Image.open(img_path).convert('L')\n"
    "img64 = orig.resize((64, 64))\n"
    "img32 = orig.resize((32, 32))\n\n"
    "arr_orig = np.array(orig)\n"
    "arr64    = np.array(img64)\n"
    "arr32    = np.array(img32)\n\n"
    "print(f'Original  {arr_orig.shape}  -> {arr_orig.size:,} features')\n"
    "print(f'64x64     {arr64.shape}  -> {arr64.size:,} features')\n"
    "print(f'32x32     {arr32.shape} -> {arr32.size:,} features')\n\n"
    "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n"
    "for ax, arr, title in zip(axes,\n"
    "        [arr_orig, arr64, arr32],\n"
    "        [f'Original ({arr_orig.shape[0]}x{arr_orig.shape[1]})\\n{arr_orig.size:,} features',\n"
    "         f'64x64\\n{arr64.size:,} features',\n"
    "         f'32x32\\n{arr32.size:,} features']):\n"
    "    ax.imshow(arr, cmap='gray')\n"
    "    ax.set_title(title, fontsize=10)\n"
    "    ax.axis('off')\n"
    "fig.suptitle('Image Resize Comparison', fontsize=12, weight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
)

ex_code_e4 = (
    "# Exercise 4: image_to_features utility function\n"
    "from image_utils import image_to_features\n"
    "import numpy as np\n\n"
    "test_images = ['images.jpg', 'data/cats/cat_101.jpg', 'data/dogs/dog_201.jpg']\n\n"
    "print('Verifying fixed output length for differently-sized inputs:')\n"
    "lengths = []\n"
    "for p in test_images:\n"
    "    from PIL import Image\n"
    "    native = Image.open(p).size\n"
    "    v = image_to_features(p, size=(64, 64))\n"
    "    lengths.append(len(v))\n"
    "    print(f'  {p:35s}  native={str(native):12s}  -> vector len={len(v):,}')\n"
    "print()\n"
    "assert len(set(lengths)) == 1, 'FAIL: lengths differ!'\n"
    "print(f'All vectors have length {lengths[0]:,}  [assertion PASSED]')\n"
)

# Insert 8 new cells after cell[20] (the exercise question cell)
new_cells = [
    make_md_cell("---\n## Exercise 1 — Bigram CountVectorizer\n", "ex1-q"),
    make_code_cell(ex_code_e1, "ex1-code"),
    make_md_cell(ex1_answer, "ex1-ans"),
    make_md_cell("---\n## Exercise 2 — Adding a repetitive 4th document\n", "ex2-q"),
    make_code_cell(ex_code_e2, "ex2-code"),
    make_md_cell(ex2_answer, "ex2-ans"),
    make_md_cell("---\n## Exercise 3 — Resize to 64x64 and 32x32\n", "ex3-q"),
    make_code_cell(ex_code_e3, "ex3-code"),
    make_md_cell(ex3_answer, "ex3-ans"),
    make_md_cell("---\n## Exercise 4 — `image_to_features` function\n", "ex4-q"),
    make_code_cell(ex_code_e4, "ex4-code"),
    make_md_cell(ex4_answer, "ex4-ans"),
]

# Insert after cell index 20
insert_after = 20
cells_new = cells[:insert_after+1] + new_cells + cells[insert_after+1:]
nb["cells"] = cells_new

with open(NB_DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"  Saved updated notebook ({len(nb['cells'])} cells) -> {NB_DST}")

# ─── Save dataset.npz ─────────────────────────────────────────────────────
print("\nBuilding dataset.npz ...")

CATS = os.path.join(LAB03, "data", "cats")
DOGS = os.path.join(LAB03, "data", "dogs")

X_list, y_list = [], []
for label, folder in [(0, CATS), (1, DOGS)]:
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg",".jpeg",".png")):
            v = image_to_features(os.path.join(folder, fname), (64,64))
            X_list.append(v); y_list.append(label)

X = np.vstack(X_list)
y = np.array(y_list)
npz_path = os.path.join(LAB03, "dataset.npz")
np.savez(npz_path, X=X, y=y)
print(f"  X shape: {X.shape}  y shape: {y.shape}")
print(f"  Saved: {npz_path}")

print("\n" + "="*60)
print("ALL EXERCISES DONE.")
print("="*60)
