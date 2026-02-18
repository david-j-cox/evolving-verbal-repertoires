"""Trial scheduling, motivating operations, and environmental contingencies.

Trial types shift over development: heavy echoic early, increasing mand
and intraverbal later.  Motivating operations fluctuate; successful mands
produce satiation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvironmentConfig

TRIAL_TYPES = ["echoic", "tact", "mand", "intraverbal"]
_TRIAL_TYPE_INDICES = {t: i for i, t in enumerate(TRIAL_TYPES)}


@dataclass
class Trial:
    """A single interaction trial."""
    trial_type: str  # one of TRIAL_TYPES
    target_word_idx: int
    context_features: np.ndarray
    mo_state: np.ndarray
    # For intraverbal: the stimulus word that prompts a response.
    stimulus_word_idx: int | None = None


class Environment:
    """Manages trial generation, MO dynamics, and schedule progression."""

    def __init__(self, config: EnvironmentConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        self.initial_probs = np.array(config.initial_trial_probs, dtype=np.float64)
        self.final_probs = np.array(config.final_trial_probs, dtype=np.float64)
        self._delta_probs = self.final_probs - self.initial_probs
        self._shift_inv = 1.0 / max(config.trial_shift_duration, 1)

        # Cache trial probs (recomputed periodically, not every step).
        self._cached_probs = self.initial_probs.copy()
        self._cache_step = -1

        # Motivating operations: deprivation level per dimension.
        self.mo_state = np.ones(config.n_mo_dims)

    def get_trial_probs(self, step: int) -> np.ndarray:
        """Linearly interpolate trial type probabilities (cached)."""
        progress = min(step * self._shift_inv, 1.0)
        probs = self.initial_probs + progress * self._delta_probs
        return probs

    def sample_trial_type(self, step: int) -> int:
        """Return trial type as integer index (0-3) for speed."""
        # Recompute cached probs every 1000 steps.
        cache_key = step // 1000
        if cache_key != self._cache_step:
            self._cached_probs = self.get_trial_probs(step)
            self._cached_probs /= self._cached_probs.sum()
            self._cache_step = cache_key
        # Manual cumulative sampling (faster than rng.choice for 4 categories).
        u = self.rng.random()
        c = 0.0
        for i in range(3):
            c += self._cached_probs[i]
            if u < c:
                return i
        return 3

    def fluctuate_mo(self) -> None:
        """Random MO fluctuation: deprivation tends to increase over time."""
        mask = self.rng.random(self.config.n_mo_dims) < self.config.mo_fluctuation_rate
        self.mo_state[mask] = np.minimum(1.0, self.mo_state[mask] + 0.1)

    def satiate(self, mo_dim: int) -> None:
        """Reduce deprivation after successful mand delivery."""
        if 0 <= mo_dim < self.config.n_mo_dims:
            self.mo_state[mo_dim] = max(
                0.0, self.mo_state[mo_dim] - self.config.satiation_amount
            )
