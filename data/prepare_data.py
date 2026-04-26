"""
Data Preparation Pipeline
EMU414 - Mathematical Programming and Computer Applications
Hacettepe University, Spring 2025-2026

Steps:
  1. Load top-10k English word list from  data/pop10000.xlsx
       columns: 'Popülerlik Sırası' (rank)  and  'Kelime' (word)
  2. Select 100 words via stratified sampling across 10 frequency bands
  3. Compute frequency weight F_w = 1/rank  (normalised to [0,1])
  4. Load pre-trained GloVe embeddings  (glove.6B.100d.txt)
  5. Compute 100×100 pairwise cosine distance matrix
  6. Save data/words_100.csv  and  data/distance_matrix.csv

Requirements:
  pip install numpy pandas openpyxl

  GloVe file (glove.6B.100d.txt) — place in the data/ directory:
  Download: https://nlp.stanford.edu/projects/glove/  → glove.6B.zip
"""

import os
import sys
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
TOP_K       = 10_000    # source vocabulary size
N_WORDS     = 100       # words to select for the problem instance
N_BANDS     = 10        # stratified bands  →  N_WORDS/N_BANDS per band
RANDOM_SEED = 42

HERE          = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE   = os.path.join(HERE, "pop10000.xlsx")   # ← local data source
GLOVE_FILE    = os.path.join(HERE, "glove.6B.100d.txt")
OUT_WORDS     = os.path.join(HERE, "words_100.csv")
OUT_DIST      = os.path.join(HERE, "distance_matrix.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Top-10k word list  (from pop10000.xlsx)
# ─────────────────────────────────────────────────────────────────────────────
def load_top10k() -> list[tuple[int, str]]:
    """
    Reads data/pop10000.xlsx.
    Expected columns: 'Popülerlik Sırası' (rank int) and 'Kelime' (word str).
    Returns [(rank, word), ...] sorted by rank ascending, length == TOP_K.
    """
    if not os.path.exists(SOURCE_FILE):
        sys.exit(
            f"[ERROR] Source file not found: {SOURCE_FILE}\n"
            "  Place pop10000.xlsx in the data/ directory."
        )

    df = pd.read_excel(SOURCE_FILE, engine="openpyxl")

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    rank_col = "Popülerlik Sırası"
    word_col = "Kelime"

    if rank_col not in df.columns or word_col not in df.columns:
        sys.exit(
            f"[ERROR] Expected columns '{rank_col}' and '{word_col}' in pop10000.xlsx.\n"
            f"  Found: {df.columns.tolist()}"
        )

    df = df[[rank_col, word_col]].dropna()
    df[rank_col] = df[rank_col].astype(int)
    df = df.sort_values(rank_col).reset_index(drop=True)
    df = df.head(TOP_K)

    word_list = list(zip(df[rank_col].tolist(), df[word_col].astype(str).tolist()))
    print(f"  Source: pop10000.xlsx  ({len(word_list)} words loaded)")
    return word_list


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Stratified word selection
# ─────────────────────────────────────────────────────────────────────────────
def select_words(word_list: list[tuple[int, str]],
                 n: int = N_WORDS,
                 n_bands: int = N_BANDS,
                 seed: int = RANDOM_SEED) -> list[tuple[int, str]]:
    """
    Divide word_list into n_bands equal frequency bands.
    Sample n/n_bands words uniformly from each band.
    Returns list of (rank, word) tuples, length == n.
    """
    rng       = np.random.default_rng(seed)
    band_size = len(word_list) // n_bands
    per_band  = n // n_bands
    selected  = []

    for b in range(n_bands):
        band   = word_list[b * band_size : (b + 1) * band_size]
        idxs   = rng.choice(len(band), size=per_band, replace=False)
        for i in sorted(idxs):
            selected.append(band[i])

    return selected[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Frequency weights   F_w = 1/rank,  normalised to [0,1]
# ─────────────────────────────────────────────────────────────────────────────
def compute_weights(ranks: list[int]) -> list[float]:
    """
    F_w = 1 / rank  →  normalised so that the highest-rank word has weight 1.
    Words with rank 1 (most frequent) get F = 1.0.
    """
    raw    = [1.0 / r for r in ranks]
    max_fw = max(raw)
    return [fw / max_fw for fw in raw]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Embeddings  (GloVe if available, else character n-gram TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────
def load_glove(path: str, vocab: set[str]) -> dict[str, np.ndarray]:
    """Parse GloVe text file; return {word: vector} for words in vocab."""
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if word in vocab:
                embeddings[word] = np.array(parts[1:], dtype=np.float32)
    return embeddings


def ngram_embeddings(words: list[str]) -> dict[str, np.ndarray]:
    """
    Fallback when GloVe is unavailable.
    Represent each word as a character n-gram TF-IDF vector (sklearn).
    Captures orthographic + morphological similarity — no download needed.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    mat = vec.fit_transform(words).toarray().astype(np.float32)
    return {w: mat[i] for i, w in enumerate(words)}


def get_embeddings(words: list[str]) -> tuple[dict[str, np.ndarray], str]:
    """
    Return (embeddings_dict, source_label).
    Tries GloVe first; falls back to character n-gram TF-IDF.
    """
    if os.path.exists(GLOVE_FILE):
        print(f"  Source: GloVe  ({GLOVE_FILE})")
        emb = load_glove(GLOVE_FILE, set(words))
        missing = [w for w in words if w not in emb]
        if missing:
            print(f"  [WARN] {len(missing)} words missing from GloVe → zero vector")
        return emb, "GloVe 6B 100d"
    else:
        print("  [INFO] GloVe file not found — using character n-gram TF-IDF (sklearn)")
        print("         (To use GloVe later: place glove.6B.100d.txt in the data/ folder)")
        return ngram_embeddings(words), "char n-gram TF-IDF (2-4)"


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Cosine distance matrix
# ─────────────────────────────────────────────────────────────────────────────
def build_distance_matrix(words: list[str],
                           embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """
    N×N cosine distance matrix.
    D_wv = 1 - cosine_similarity(e_w, e_v)  in [0, 2].
    Words missing from embeddings dict get a zero vector.
    """
    dim  = next(iter(embeddings.values())).shape[0]
    vecs = np.array([embeddings.get(w, np.zeros(dim)) for w in words],
                    dtype=np.float32)

    norms     = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1.0, norms)
    vecs_norm = vecs / norms

    sim  = vecs_norm @ vecs_norm.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DATA PREPARATION PIPELINE")
    print("=" * 60)

    # ── Step 1 ───────────────────────────────────────────────────────────────
    print("\nStep 1 | Loading top-10k word list ...")
    word_list = load_top10k()
    print(f"         {len(word_list)} words loaded.")

    # ── Step 2 ───────────────────────────────────────────────────────────────
    print(f"\nStep 2 | Stratified sampling  "
          f"({N_BANDS} bands × {N_WORDS // N_BANDS} words, seed={RANDOM_SEED}) ...")
    selected = select_words(word_list)
    ranks    = [r for r, _ in selected]
    words    = [w for _, w in selected]
    print(f"         Selected {len(words)} words.")
    print(f"         Sample: {words[:8]} ...")

    # ── Step 3 ───────────────────────────────────────────────────────────────
    print("\nStep 3 | Computing frequency weights  F_w = (1/rank) / max(1/rank) ...")
    freq_weights = compute_weights(ranks)
    print(f"         Weight range: [{min(freq_weights):.6f}, {max(freq_weights):.6f}]")

    # ── Step 4 ───────────────────────────────────────────────────────────────
    print(f"\nStep 4 | Loading word embeddings ...")
    embeddings, emb_source = get_embeddings(words)
    found = sum(1 for w in words if w in embeddings)
    print(f"         Embedding source : {emb_source}")
    print(f"         Vectors found    : {found}/{N_WORDS}")

    # ── Step 5 ───────────────────────────────────────────────────────────────
    print("\nStep 5 | Computing 100×100 cosine distance matrix ...")
    dist = build_distance_matrix(words, embeddings)
    print(f"         Shape: {dist.shape}")
    nonzero = dist[dist > 0]
    print(f"         Distance range (excl. diagonal): "
          f"[{nonzero.min():.4f}, {nonzero.max():.4f}]")
    print(f"         Mean distance: {nonzero.mean():.4f}")

    # ── Step 6 — Save ────────────────────────────────────────────────────────
    print("\nStep 6 | Saving output files ...")

    words_df = pd.DataFrame({
        "rank":         ranks,
        "word":         words,
        "freq_weight":  freq_weights,
    })
    words_df.to_csv(OUT_WORDS, index=False)
    print(f"         Saved: words_100.csv  ({len(words_df)} rows)")

    dist_df = pd.DataFrame(dist, index=words, columns=words)
    dist_df.to_csv(OUT_DIST)
    print(f"         Saved: distance_matrix.csv  ({dist.shape[0]}x{dist.shape[1]})")

    print("\n" + "=" * 60)
    print("  Done. You can now run model.py.")
    print("=" * 60)


if __name__ == "__main__":
    main()
