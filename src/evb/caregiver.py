"""Adaptive child-directed speech (CDS) caregiver.

Selects words biased toward the child's zone of proximal development.
Evaluates child emissions with graded phoneme similarity.
Different contingencies per operant type.
"""

from __future__ import annotations

import numpy as np

from .config import CaregiverConfig, ReinforcementConfig
from .phonology import PhonologySystem, SIMILARITY_MATRIX, SILENCE_IDX
from .lexicon import Lexicon
from .reinforcement import ShapingController, ReinforcementSchedule


def _weighted_sample(probs: np.ndarray, u: float) -> int:
    """Cumulative sampling without numpy.choice overhead."""
    c = 0.0
    for i in range(len(probs) - 1):
        c += probs[i]
        if u < c:
            return i
    return len(probs) - 1


class Caregiver:
    """Caregiver that models words and evaluates child emissions."""

    def __init__(
        self,
        config: CaregiverConfig,
        reinforcement_config: ReinforcementConfig,
        phonology: PhonologySystem,
        lexicon: Lexicon,
        rng: np.random.Generator,
    ):
        self.config = config
        self.phonology = phonology
        self.lexicon = lexicon
        self.rng = rng

        self.shaping = ShapingController(reinforcement_config, lexicon.n_words)
        self.schedule = ReinforcementSchedule(reinforcement_config)

        # Per-word selection weights (adapted over time).
        self.word_weights = np.ones(lexicon.n_words)
        # Track per-word similarity for ZPD adaptation.
        self._word_similarity_ema = np.zeros(lexicon.n_words)

        # Pre-cache for select_word.
        self._mandable_indices = lexicon.get_mandable_indices()
        self._iv_words = np.array(list(lexicon.intraverbal_map.keys()), dtype=np.int32)
        self._cached_default_probs = self.word_weights / self.word_weights.sum()
        self._cached_iv_probs = (
            self.word_weights[self._iv_words]
            / (self.word_weights[self._iv_words].sum() + 1e-12)
            if len(self._iv_words) > 0 else np.array([])
        )

        # Direct references for hot loop.
        self._sim_matrix = SIMILARITY_MATRIX
        self._silence_idx = SILENCE_IDX

    def select_word(self, trial_type: str, mo_state: np.ndarray | None = None) -> int:
        """Select a target word for the current trial."""
        if trial_type == "mand" and mo_state is not None:
            mandable = self._mandable_indices
            if len(mandable) == 0:
                return int(self.rng.integers(0, self.lexicon.n_words))
            mo_vals = mo_state[self.lexicon.mo_assignments[mandable]]
            mo_vals = np.maximum(mo_vals, 0.01)
            combined = mo_vals * self.word_weights[mandable]
            combined /= combined.sum() + 1e-12
            return int(mandable[_weighted_sample(combined, self.rng.random())])

        if trial_type == "intraverbal" and len(self._iv_words) > 0:
            return int(self._iv_words[_weighted_sample(self._cached_iv_probs, self.rng.random())])

        return _weighted_sample(self._cached_default_probs, self.rng.random())

    def model_word(self, word_idx: int) -> np.ndarray:
        """Produce the target phoneme sequence for the child to hear."""
        return self.lexicon.word_phonemes[word_idx].copy()

    def evaluate(
        self,
        word_idx: int,
        emitted: np.ndarray,
        step: int,
    ) -> tuple[bool, float, float]:
        """Evaluate a child emission against the target word.

        Returns (reinforced, reward_magnitude, similarity).
        """
        target = self.lexicon.word_phonemes[word_idx]

        # Inline word similarity for speed.
        sims = self._sim_matrix[emitted, target]
        active = ~((emitted == self._silence_idx) & (target == self._silence_idx))
        n = active.sum()
        similarity = float(sims[active].sum() / n) if n > 0 else 1.0

        # Update similarity EMA for ZPD tracking.
        self._word_similarity_ema[word_idx] += 0.01 * (
            similarity - self._word_similarity_ema[word_idx]
        )

        # Shaping controller decides if this meets threshold.
        reinforced, reward = self.shaping.evaluate(
            word_idx, similarity, self.rng
        )

        # Schedule may withhold reinforcement (intermittent).
        if reinforced and not self.schedule.should_deliver(step, self.rng):
            reinforced = False
            reward = 0.0

        return reinforced, reward, similarity

    def adapt_word_selection(self) -> None:
        """Periodically adapt word selection weights toward ZPD."""
        cfg = self.config
        sim = self._word_similarity_ema

        # Vectorized weight computation.
        weights = np.where(
            (sim >= cfg.zpd_low) & (sim <= cfg.zpd_high),
            1.0 + cfg.zpd_bias,
            np.where(sim > cfg.zpd_high, 0.5, 0.8),
        )
        self.word_weights = weights

        # Update cached probability distributions.
        total = weights.sum() + 1e-12
        self._cached_default_probs = weights / total
        if len(self._iv_words) > 0:
            iv_w = weights[self._iv_words]
            self._cached_iv_probs = iv_w / (iv_w.sum() + 1e-12)
