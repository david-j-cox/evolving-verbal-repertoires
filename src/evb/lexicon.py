"""Target lexicon: ~35 early-childhood words as phoneme sequences.

Each word is stored as a padded phoneme-index array.  Semantic categories
provide structured context feature vectors; intraverbal pairs and mandable
subsets support the four operant types.
"""

from __future__ import annotations

import numpy as np

from .config import LexiconConfig
from .phonology import (
    PHONEME_TO_IDX,
    SILENCE_IDX,
    N_PHONEMES,
    FEAT_DIM,
    PhonologySystem,
)

# ---------------------------------------------------------------------------
# Word definitions: (spelling, phoneme sequence, category, mandable)
# ---------------------------------------------------------------------------

_WORD_DEFS: list[tuple[str, list[str], str, bool]] = [
    # People
    ("mama",    ["M", "AA", "M", "AA"],       "people",  False),
    ("dada",    ["D", "AA", "D", "AA"],       "people",  False),
    ("baby",    ["B", "EH", "B", "IY"],       "people",  False),
    # Animals
    ("dog",     ["D", "AO", "G"],             "animal",  False),
    ("cat",     ["K", "AE", "T"],             "animal",  False),
    ("bird",    ["B", "ER", "D"],             "animal",  False),
    ("fish",    ["F", "IH", "SH"],            "animal",  False),
    ("duck",    ["D", "AH", "K"],             "animal",  False),
    # Food/drink (mandable)
    ("milk",    ["M", "IH", "L", "K"],        "food",    True),
    ("water",   ["W", "AO", "T", "ER"],       "food",    True),
    ("juice",   ["JH", "UW", "S"],            "food",    True),
    ("cookie",  ["K", "UH", "K", "IY"],       "food",    True),
    ("banana",  ["B", "AH", "N", "AE", "N"],  "food",    True),
    # Toys/objects (mandable)
    ("ball",    ["B", "AO", "L"],             "toy",     True),
    ("book",    ["B", "UH", "K"],             "toy",     True),
    ("shoe",    ["SH", "UW"],                 "toy",     True),
    ("cup",     ["K", "AH", "P"],             "object",  True),
    ("hat",     ["HH", "AE", "T"],            "object",  False),
    ("nose",    ["N", "OW", "Z"],             "body",    False),
    ("eye",     ["AA", "IY"],                 "body",    False),
    # Actions
    ("go",      ["G", "OW"],                  "action",  False),
    ("up",      ["AH", "P"],                  "action",  True),
    ("down",    ["D", "AA", "N"],             "action",  False),
    ("eat",     ["IY", "T"],                  "action",  True),
    ("open",    ["OW", "P", "AH", "N"],       "action",  True),
    # Social words
    ("hi",      ["HH", "AA", "IY"],           "social",  False),
    ("bye",     ["B", "AA", "IY"],            "social",  False),
    ("no",      ["N", "OW"],                  "social",  False),
    ("yes",     ["Y", "EH", "S"],             "social",  False),
    ("more",    ["M", "AO", "R"],             "social",  True),
    ("please",  ["P", "L", "IY", "Z"],        "social",  True),
    # Descriptors
    ("hot",     ["HH", "AA", "T"],            "descriptor", False),
    ("cold",    ["K", "OW", "L", "D"],        "descriptor", False),
    ("big",     ["B", "IH", "G"],             "descriptor", False),
    ("wet",     ["W", "EH", "T"],             "descriptor", False),
]

# Intraverbal pairs (stimulus → conventional response).
_INTRAVERBAL_PAIRS: list[tuple[str, str]] = [
    ("dog",  "cat"),
    ("cat",  "dog"),
    ("hot",  "cold"),
    ("cold", "hot"),
    ("hi",   "bye"),
    ("bye",  "hi"),
    ("up",   "down"),
    ("down", "up"),
    ("yes",  "no"),
    ("no",   "yes"),
    ("mama", "dada"),
    ("dada", "mama"),
    ("big",  "baby"),
    ("more", "please"),
]

# ---------------------------------------------------------------------------
# Semantic category → base feature vector index ranges
# ---------------------------------------------------------------------------

_CATEGORIES = [
    "people", "animal", "food", "toy", "object",
    "body", "action", "social", "descriptor",
]


# ---------------------------------------------------------------------------
# Lexicon class
# ---------------------------------------------------------------------------

class Lexicon:
    """Target lexicon for the simulation."""

    def __init__(self, config: LexiconConfig, rng: np.random.Generator):
        self.config = config
        self.max_len = config.max_word_len
        self.n_words = len(_WORD_DEFS)

        # Parse word definitions.
        self.word_names: list[str] = []
        self.word_phonemes: np.ndarray = np.full(
            (self.n_words, self.max_len), SILENCE_IDX, dtype=np.int32
        )
        self.categories: list[str] = []
        self.mandable: np.ndarray = np.zeros(self.n_words, dtype=bool)

        for i, (name, phones, cat, mand) in enumerate(_WORD_DEFS):
            self.word_names.append(name)
            self.categories.append(cat)
            self.mandable[i] = mand
            for j, p in enumerate(phones[: self.max_len]):
                self.word_phonemes[i, j] = PHONEME_TO_IDX[p]

        self.word_to_idx: dict[str, int] = {
            n: i for i, n in enumerate(self.word_names)
        }

        # Build semantic context features.
        self.context_features = self._build_context_features(rng)

        # Build intraverbal mapping: stimulus word idx → response word idx.
        self.intraverbal_map: dict[int, int] = {}
        for stim_name, resp_name in _INTRAVERBAL_PAIRS:
            if stim_name in self.word_to_idx and resp_name in self.word_to_idx:
                self.intraverbal_map[self.word_to_idx[stim_name]] = (
                    self.word_to_idx[resp_name]
                )

        # MO category assignments (which MO dimension each word satisfies).
        self.mo_assignments = self._assign_mo_dims(config, rng)

    def _build_context_features(self, rng: np.random.Generator) -> np.ndarray:
        """Build context feature vectors with category structure.

        Same-category words share a base vector (scaled by category_overlap),
        plus a unique random component.
        """
        dim = self.config.context_dim
        overlap = self.config.category_overlap

        # Random base vector per category.
        cat_bases: dict[str, np.ndarray] = {}
        for cat in _CATEGORIES:
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v) + 1e-12
            cat_bases[cat] = v

        features = np.zeros((self.n_words, dim))
        for i in range(self.n_words):
            base = cat_bases[self.categories[i]]
            unique = rng.standard_normal(dim)
            unique /= np.linalg.norm(unique) + 1e-12
            combined = overlap * base + (1.0 - overlap) * unique
            combined /= np.linalg.norm(combined) + 1e-12
            features[i] = combined

        return features

    def _assign_mo_dims(
        self, config: LexiconConfig, rng: np.random.Generator
    ) -> np.ndarray:
        """Assign each mandable word to an MO dimension.

        Non-mandable words get -1.
        """
        from .config import EnvironmentConfig
        n_mo = 5  # default; will be overridden in environment
        assignments = np.full(self.n_words, -1, dtype=np.int32)
        mandable_idxs = np.where(self.mandable)[0]
        for idx in mandable_idxs:
            assignments[idx] = rng.integers(0, n_mo)
        return assignments

    def get_mandable_indices(self) -> np.ndarray:
        return np.where(self.mandable)[0]
