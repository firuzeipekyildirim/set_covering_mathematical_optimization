"""
Data Preparation Pipeline
EMU414 - Mathematical Programming and Computer Applications
Hacettepe University, Spring 2025-2026

Steps:
  1. Load a top-10k English word frequency list
  2. Select 100 words (stratified by frequency band)
  3. Load pre-trained GloVe embeddings (glove.6B.100d.txt)
  4. Compute pairwise cosine distance matrix
  5. Save words_100.csv and distance_matrix.csv

Requirements:
  - glove.6B.100d.txt in the data/ directory
    Download: https://nlp.stanford.edu/projects/glove/
  - pip install numpy pandas
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_K = 10000               # size of the source word list
N_WORDS = 100               # words to select for the problem instance
N_BANDS = 10                # stratified sampling: N_WORDS / N_BANDS per band
RANDOM_SEED = 42

GLOVE_FILE = os.path.join(os.path.dirname(__file__), "glove.6B.100d.txt")
OUT_WORDS = os.path.join(os.path.dirname(__file__), "words_100.csv")
OUT_DIST = os.path.join(os.path.dirname(__file__), "distance_matrix.csv")


def load_top10k_words():
    """
    Returns a list of (rank, word) for the top TOP_K English words.

    We use a simple built-in frequency list derived from common corpora.
    Replace this with your own source list if preferred (e.g., wordfreq library).
    """
    try:
        from wordfreq import top_n_list
        words = top_n_list("en", TOP_K)
        return [(i + 1, w) for i, w in enumerate(words)]
    except ImportError:
        pass

    # Fallback: load from a plain text file (one word per line, most frequent first)
    fallback_path = os.path.join(os.path.dirname(__file__), "top10k_words.txt")
    if os.path.exists(fallback_path):
        with open(fallback_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return [(i + 1, w) for i, w in enumerate(lines[:TOP_K])]

    raise FileNotFoundError(
        "Could not find a top-10k word source. Either install wordfreq "
        "(pip install wordfreq) or place top10k_words.txt in the data/ directory."
    )


def select_words(word_list, n=N_WORDS, n_bands=N_BANDS, seed=RANDOM_SEED):
    """Stratified sampling: sample n/n_bands words from each frequency band."""
    rng = np.random.default_rng(seed)
    band_size = len(word_list) // n_bands
    per_band = n // n_bands
    selected = []
    for b in range(n_bands):
        band = word_list[b * band_size: (b + 1) * band_size]
        chosen = rng.choice(len(band), size=per_band, replace=False)
        for idx in sorted(chosen):
            selected.append(band[idx])
    return selected[:n]


def load_glove(path, vocab):
    """Load GloVe vectors for the given vocabulary. Returns {word: np.array}."""
    vocab_set = set(vocab)
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab_set:
                embeddings[word] = np.array(parts[1:], dtype=np.float32)
    return embeddings


def cosine_distance(a, b):
    """Cosine distance = 1 - cosine_similarity. Range [0, 2]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


def build_distance_matrix(words, embeddings):
    """Build the N×N cosine distance matrix. Missing words get zero vector."""
    n = len(words)
    dim = next(iter(embeddings.values())).shape[0]
    vecs = np.array([embeddings.get(w, np.zeros(dim)) for w in words])

    # Normalize rows for fast cosine distance
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs_norm = vecs / norms

    # Cosine similarity matrix, then convert to distance
    sim = vecs_norm @ vecs_norm.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def main():
    print("Step 1: Loading top-10k word list...")
    word_list = load_top10k_words()
    print(f"  Loaded {len(word_list)} words.")

    print(f"Step 2: Selecting {N_WORDS} words (stratified, seed={RANDOM_SEED})...")
    selected = select_words(word_list)
    ranks = [r for r, w in selected]
    words = [w for r, w in selected]
    print(f"  Selected: {words[:5]} ... (and {len(words)-5} more)")

    print("Step 3: Loading GloVe embeddings...")
    if not os.path.exists(GLOVE_FILE):
        raise FileNotFoundError(
            f"GloVe file not found: {GLOVE_FILE}\n"
            "Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ "
            "and extract glove.6B.100d.txt into the data/ directory."
        )
    embeddings = load_glove(GLOVE_FILE, words)
    coverage = sum(1 for w in words if w in embeddings)
    print(f"  Found embeddings for {coverage}/{N_WORDS} words.")

    print("Step 4: Computing cosine distance matrix...")
    dist = build_distance_matrix(words, embeddings)
    print(f"  Distance matrix shape: {dist.shape}")
    print(f"  Min distance: {dist[dist > 0].min():.4f}, Max: {dist.max():.4f}")

    print("Step 5: Saving output files...")
    words_df = pd.DataFrame({"rank": ranks, "word": words})
    words_df.to_csv(OUT_WORDS, index=False)
    print(f"  Saved: {OUT_WORDS}")

    dist_df = pd.DataFrame(dist, index=words, columns=words)
    dist_df.to_csv(OUT_DIST)
    print(f"  Saved: {OUT_DIST}")

    print("\nDone. You can now run model.py.")


if __name__ == "__main__":
    main()
