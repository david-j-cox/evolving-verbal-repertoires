"""Phoneme inventory with articulatory feature vectors and similarity matrix.

36 phonemes: 24 consonants + 11 vowels + 1 silence token.
Each phoneme has an articulatory feature vector enabling graded similarity.
"""

from __future__ import annotations

import numpy as np

from .config import PhonologyConfig

# ---------------------------------------------------------------------------
# Phoneme inventory
# ---------------------------------------------------------------------------

# Consonants: (symbol, voiced, manner, place)
#   voiced: 0=voiceless, 1=voiced
#   manner: stop=0, fricative=1, affricate=2, nasal=3, liquid=4, glide=5
#   place:  bilabial=0, labiodental=1, dental=2, alveolar=3, postalveolar=4,
#           palatal=5, velar=6, glottal=7
_CONSONANTS: list[tuple[str, int, int, int]] = [
    ("P",  0, 0, 0),  # p
    ("B",  1, 0, 0),  # b
    ("T",  0, 0, 3),  # t
    ("D",  1, 0, 3),  # d
    ("K",  0, 0, 6),  # k
    ("G",  1, 0, 6),  # g
    ("F",  0, 1, 1),  # f
    ("V",  1, 1, 1),  # v
    ("TH", 0, 1, 2),  # θ (thin)
    ("DH", 1, 1, 2),  # ð (this)
    ("S",  0, 1, 3),  # s
    ("Z",  1, 1, 3),  # z
    ("SH", 0, 1, 4),  # ʃ (ship)
    ("ZH", 1, 1, 4),  # ʒ (measure)
    ("CH", 0, 2, 4),  # tʃ (chip)
    ("JH", 1, 2, 4),  # dʒ (judge)
    ("M",  1, 3, 0),  # m
    ("N",  1, 3, 3),  # n
    ("NG", 1, 3, 6),  # ŋ (sing)
    ("L",  1, 4, 3),  # l
    ("R",  1, 4, 3),  # r
    ("W",  1, 5, 0),  # w
    ("Y",  1, 5, 5),  # j (yes)
    ("HH", 0, 1, 7),  # h
]

# Vowels: (symbol, height, backness, rounded)
#   height:    high=0, mid=1, low=2
#   backness:  front=0, central=1, back=2
#   rounded:   0=unrounded, 1=rounded
_VOWELS: list[tuple[str, int, int, int]] = [
    ("IY", 0, 0, 0),  # iː (beat)
    ("IH", 0, 0, 0),  # ɪ  (bit)
    ("EH", 1, 0, 0),  # ɛ  (bet)
    ("AE", 2, 0, 0),  # æ  (bat)
    ("AH", 1, 1, 0),  # ʌ  (but)
    ("UH", 0, 2, 1),  # ʊ  (book)
    ("UW", 0, 2, 1),  # uː (boot)
    ("OW", 1, 2, 1),  # oʊ (boat)
    ("AO", 2, 2, 1),  # ɔː (bought)
    ("AA", 2, 2, 0),  # ɑː (bot)
    ("ER", 1, 1, 0),  # ɝ  (bird)
]

SILENCE = "_"


def _build_inventory() -> tuple[list[str], np.ndarray]:
    """Build phoneme list and articulatory feature matrix.

    Feature vector layout (11 dimensions):
      [0]     is_consonant (1) vs vowel (0)
      [1]     voiced
      [2-7]   manner one-hot (6 categories) — consonants only
      [8-10]  place (3 normalized dims: bilabial..glottal) — consonants only
        OR
      [2-3]   height (2 dims) — vowels only
      [4-5]   backness (2 dims) — vowels only
      [6]     rounded — vowels only
      [7-10]  zeros — vowels only

    Silence gets a zero vector.
    """
    n_manner = 6
    n_place = 8
    feat_dim = 11  # is_consonant, voiced, 6 manner, 3 place-projection

    symbols: list[str] = []
    features: list[np.ndarray] = []

    # Consonants
    for sym, voiced, manner, place in _CONSONANTS:
        v = np.zeros(feat_dim)
        v[0] = 1.0  # is_consonant
        v[1] = float(voiced)
        v[2 + manner] = 1.0  # manner one-hot
        # Place: project onto 3 dims via normalized position
        place_norm = place / (n_place - 1)  # 0..1
        v[8] = 1.0 - place_norm          # front-ness
        v[9] = place_norm                 # back-ness
        v[10] = 0.5 if place in (3, 4) else 0.0  # coronal marker
        symbols.append(sym)
        features.append(v)

    # Vowels
    for sym, height, backness, rounded in _VOWELS:
        v = np.zeros(feat_dim)
        v[0] = 0.0  # vowel
        v[1] = 1.0  # vowels are all voiced
        # Pack vowel features into [2..6]
        v[2] = height / 2.0       # height normalized
        v[3] = 1.0 - height / 2.0
        v[4] = backness / 2.0     # backness normalized
        v[5] = 1.0 - backness / 2.0
        v[6] = float(rounded)
        symbols.append(sym)
        features.append(v)

    # Silence
    symbols.append(SILENCE)
    features.append(np.zeros(feat_dim))

    return symbols, np.array(features, dtype=np.float64)


# Module-level constants built once at import time.
PHONEMES, FEATURES = _build_inventory()
N_PHONEMES = len(PHONEMES)
PHONEME_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(PHONEMES)}
SILENCE_IDX = PHONEME_TO_IDX[SILENCE]
FEAT_DIM = FEATURES.shape[1]


def _cosine_sim_matrix(M: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix, NaN-safe for zero vectors."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    U = M / norms
    S = U @ U.T
    # Silence (zero vector) should have 0 similarity to everything.
    zero_mask = (np.linalg.norm(M, axis=1) < 1e-12)
    S[zero_mask, :] = 0.0
    S[:, zero_mask] = 0.0
    return S


# Precomputed phoneme-to-phoneme similarity (O(1) lookup).
SIMILARITY_MATRIX: np.ndarray = _cosine_sim_matrix(FEATURES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PhonologySystem:
    """Provides phoneme similarity operations for the simulation."""

    def __init__(self, config: PhonologyConfig | None = None):
        self.symbols = PHONEMES
        self.n_phonemes = N_PHONEMES
        self.features = FEATURES
        self.similarity = SIMILARITY_MATRIX
        self.silence_idx = SILENCE_IDX

    def phoneme_similarity(self, a: int, b: int) -> float:
        """Similarity between two phonemes by index."""
        return float(self.similarity[a, b])

    def word_similarity(
        self,
        emitted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Graded position-by-position similarity between two phoneme sequences.

        Both arrays should be 1-D integer arrays of phoneme indices,
        of the same length (padded with silence).
        Returns mean per-position similarity in [0, 1].
        """
        # Vectorized: look up all position similarities at once.
        sims = self.similarity[emitted, target]
        # Mask positions where both are silence (padding).
        active = ~((emitted == self.silence_idx) & (target == self.silence_idx))
        n_active = active.sum()
        if n_active == 0:
            return 1.0
        return float(sims[active].sum() / n_active)

    def encode_phonemes(self, indices: np.ndarray) -> np.ndarray:
        """Encode a phoneme index sequence as concatenated feature vectors.

        Returns a 1-D vector of shape (len(indices) * FEAT_DIM,).
        """
        return self.features[indices].ravel()
