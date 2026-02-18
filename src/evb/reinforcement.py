"""Shaping controller and reinforcement schedule.

ShapingController: per-word adaptive thresholds that ratchet upward.
ReinforcementSchedule: CRF → intermittent thinning.
"""

from __future__ import annotations

import numpy as np

from .config import ReinforcementConfig


class ShapingController:
    """Per-word adaptive thresholds for successive approximation.

    Each word maintains a running best similarity and a threshold set
    just below it.  The threshold ratchets upward as the learner improves,
    implementing successive approximation.
    """

    def __init__(self, config: ReinforcementConfig, n_words: int):
        self.config = config
        self.n_words = n_words
        # Per-word best-so-far similarity (EMA-smoothed).
        self.best_similarity = np.full(n_words, config.initial_threshold)
        # Per-word current threshold.
        self.thresholds = np.full(n_words, config.initial_threshold)

    def evaluate(
        self,
        word_idx: int,
        similarity: float,
        rng: np.random.Generator,
    ) -> tuple[bool, float]:
        """Evaluate an emission and decide whether to reinforce.

        Returns (reinforced: bool, reward_magnitude: float).
        Reward magnitude is proportional to similarity (graded, not binary).
        """
        cfg = self.config

        # Update best-so-far (EMA).
        self.best_similarity[word_idx] = (
            (1 - cfg.threshold_ratchet_rate) * self.best_similarity[word_idx]
            + cfg.threshold_ratchet_rate * similarity
        )

        # Ratchet threshold up toward best - margin, but never down.
        target_threshold = max(
            self.best_similarity[word_idx] - cfg.threshold_margin,
            cfg.initial_threshold,
        )
        target_threshold = min(target_threshold, cfg.max_threshold)
        self.thresholds[word_idx] = max(
            self.thresholds[word_idx], target_threshold
        )

        # Decide reinforcement.
        meets_threshold = similarity >= self.thresholds[word_idx]

        if meets_threshold:
            # False negative: miss a good emission.
            if rng.random() < cfg.false_negative_rate:
                return False, 0.0
            return True, similarity  # graded reward
        else:
            # False positive: reinforce a sub-threshold emission.
            if rng.random() < cfg.false_positive_rate:
                return True, similarity * 0.5  # reduced magnitude
            return False, 0.0


class ReinforcementSchedule:
    """CRF → intermittent thinning schedule.

    During CRF phase, every qualifying emission is reinforced.
    After CRF, reinforcement probability thins linearly to a floor.
    """

    def __init__(self, config: ReinforcementConfig):
        self.config = config

    def should_deliver(self, step: int, rng: np.random.Generator) -> bool:
        """Whether reinforcement should be delivered at this step.

        This is applied *after* ShapingController says the emission qualifies.
        """
        cfg = self.config
        if step < cfg.crf_duration:
            return True  # CRF: always deliver

        elapsed = step - cfg.crf_duration
        progress = min(elapsed / max(cfg.thinning_duration, 1), 1.0)
        p_reinforce = 1.0 - progress * (1.0 - cfg.min_reinforcement_prob)

        return rng.random() < p_reinforce
