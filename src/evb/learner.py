"""Child learner agent: 4 verbal operant systems + receptive system.

Each operant system has a factored per-position policy:
  W[pos] @ input_features → phoneme logits at position pos.
REINFORCE with baseline and entropy bonus.

Babbling emerges from high initial softmax temperature.
"""

from __future__ import annotations

import numpy as np

from .config import LearnerConfig
from .phonology import N_PHONEMES, FEAT_DIM, PhonologySystem


def _softmax_2d(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Batched softmax over axis=1 with temperature.  Shape: (positions, phonemes)."""
    z = logits / max(temperature, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class OperantSystem:
    """One verbal operant (echoic, tact, mand, or intraverbal).

    Weight tensor W of shape (max_len, n_phonemes, input_dim).
    All positions computed in a single batched matmul.
    """

    def __init__(
        self,
        name: str,
        input_dim: int,
        max_len: int,
        config: LearnerConfig,
        rng: np.random.Generator,
    ):
        self.name = name
        self.input_dim = input_dim
        self.max_len = max_len
        self.config = config
        self.n_phonemes = N_PHONEMES

        # Stacked weight tensor: (max_len, n_phonemes, input_dim).
        self.W = rng.normal(0, 0.01, size=(max_len, N_PHONEMES, input_dim))
        self.baseline = 0.0

    def sample(
        self,
        input_features: np.ndarray,
        temperature: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a phoneme sequence from the policy.

        Returns (phoneme_indices [max_len], probs [max_len, n_phonemes]).
        """
        # Batched logits: (max_len, n_phonemes)
        logits = self.W @ input_features  # (max_len, n_phonemes)
        probs = _softmax_2d(logits, temperature)

        # Sample from each position's categorical distribution.
        cumprobs = probs.cumsum(axis=1)
        u = rng.random(self.max_len)[:, None]
        indices = (cumprobs < u).sum(axis=1).astype(np.int32)
        # Clamp edge cases without calling np.clip (avoid overhead).
        indices[indices >= self.n_phonemes] = self.n_phonemes - 1

        return indices, probs

    def update(
        self,
        input_features: np.ndarray,
        chosen_indices: np.ndarray,
        probs: np.ndarray,
        reward: float,
    ) -> None:
        """Vectorized REINFORCE update with baseline and entropy bonus."""
        cfg = self.config
        advantage = reward - self.baseline
        self.baseline = (
            (1 - cfg.baseline_lr) * self.baseline + cfg.baseline_lr * reward
        )

        # One-hot: (max_len, n_phonemes)
        onehot = np.zeros((self.max_len, self.n_phonemes))
        onehot[np.arange(self.max_len), chosen_indices] = 1.0

        # Policy gradient: (max_len, n_phonemes)
        grad_log_pi = onehot - probs

        # Entropy gradient: pushes toward uniform.
        log_probs = np.log(probs + 1e-12)
        # Per-position: -(p * (log p - sum(p * log p)))
        weighted_log = (probs * log_probs).sum(axis=1, keepdims=True)
        entropy_grad = -(probs * (log_probs - weighted_log))

        # Combined gradient: (max_len, n_phonemes)
        total_grad = advantage * grad_log_pi + cfg.entropy_bonus * entropy_grad

        # Weight update: outer product across all positions at once.
        # total_grad: (max_len, n_phonemes), input_features: (input_dim,)
        # → dW: (max_len, n_phonemes, input_dim)
        self.W += cfg.learning_rate * np.einsum(
            "mp,d->mpd", total_grad, input_features
        )


class ReceptiveSystem:
    """Respondent (Hebbian) conditioning: maps heard phoneme sequences → meaning.

    Association matrix A: (n_words, context_dim). Updated via Hebbian rule
    when the learner hears a word in a context.
    """

    def __init__(
        self,
        n_words: int,
        context_dim: int,
        config: LearnerConfig,
    ):
        self.n_words = n_words
        self.context_dim = context_dim
        self.config = config
        self.A = np.zeros((n_words, context_dim))

    def update(self, word_idx: int, context: np.ndarray) -> None:
        """Hebbian update: strengthen association between heard word and context."""
        lr = self.config.receptive_lr
        self.A[word_idx] += lr * (context - self.A[word_idx])

    def comprehension_score(self, word_idx: int, context: np.ndarray) -> float:
        """How well the learner maps this word to this context (cosine sim)."""
        a = self.A[word_idx]
        norm_a = np.linalg.norm(a)
        norm_c = np.linalg.norm(context)
        if norm_a < 1e-12 or norm_c < 1e-12:
            return 0.0
        return float(np.dot(a, context) / (norm_a * norm_c))


class Learner:
    """Complete child learner with 4 verbal operant systems + receptive system."""

    def __init__(
        self,
        config: LearnerConfig,
        phonology: PhonologySystem,
        n_words: int,
        max_word_len: int,
        context_dim: int,
        rng: np.random.Generator,
    ):
        self.config = config
        self.phonology = phonology
        self.n_words = n_words
        self.max_len = max_word_len
        self.rng = rng

        self.temperature = config.initial_temperature
        self.step_count = 0

        # Input dimensions for each operant type.
        echoic_dim = max_word_len * FEAT_DIM
        tact_dim = context_dim
        mand_dim = context_dim
        intraverbal_dim = max_word_len * FEAT_DIM

        self.echoic = OperantSystem(
            "echoic", echoic_dim, max_word_len, config, rng
        )
        self.tact = OperantSystem(
            "tact", tact_dim, max_word_len, config, rng
        )
        self.mand = OperantSystem(
            "mand", mand_dim, max_word_len, config, rng
        )
        self.intraverbal = OperantSystem(
            "intraverbal", intraverbal_dim, max_word_len, config, rng
        )

        self.operants: dict[str, OperantSystem] = {
            "echoic": self.echoic,
            "tact": self.tact,
            "mand": self.mand,
            "intraverbal": self.intraverbal,
        }

        self.receptive = ReceptiveSystem(n_words, context_dim, config)

    def emit(
        self, operant_name: str, input_features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Emit a phoneme sequence from the specified operant system."""
        system = self.operants[operant_name]
        return system.sample(input_features, self.temperature, self.rng)

    def learn(
        self,
        operant_name: str,
        input_features: np.ndarray,
        chosen_indices: np.ndarray,
        probs: np.ndarray,
        reward: float,
    ) -> None:
        """Apply REINFORCE update to the relevant operant system."""
        if reward > 0:
            self.operants[operant_name].update(
                input_features, chosen_indices, probs, reward
            )

    def hear(self, word_idx: int, context: np.ndarray) -> None:
        """Process a heard word in context (receptive/respondent learning)."""
        self.receptive.update(word_idx, context)

    def decay_temperature(self) -> None:
        """Decay babbling temperature by one step."""
        self.step_count += 1
        self.temperature = max(
            self.config.min_temperature,
            self.config.initial_temperature
            * np.exp(-self.config.temperature_decay * self.step_count),
        )
