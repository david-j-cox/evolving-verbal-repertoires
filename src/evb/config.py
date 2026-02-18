"""Nested dataclass configuration for all simulation components."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class PhonologyConfig:
    """Parameters for the phoneme inventory and similarity computation."""
    # No user-tunable params currently — phoneme inventory is fixed.
    # Placeholder for future extensions (e.g., subset inventories).
    pass


@dataclass
class LexiconConfig:
    """Parameters for the target lexicon."""
    max_word_len: int = 5
    # Dimensionality of semantic context feature vectors.
    context_dim: int = 20
    # How much same-category words share context features (0–1).
    category_overlap: float = 0.4


@dataclass
class ReinforcementConfig:
    """Parameters for the shaping controller and reinforcement schedule."""
    # Initial shaping threshold (similarity needed for reinforcement).
    initial_threshold: float = 0.2
    # Maximum shaping threshold.
    max_threshold: float = 0.85
    # How fast thresholds ratchet up (EMA smoothing of best similarity).
    threshold_ratchet_rate: float = 0.01
    # Offset below best-so-far similarity to set the threshold.
    threshold_margin: float = 0.10
    # False-positive reinforcement probability (noise).
    false_positive_rate: float = 0.05
    # False-negative probability (miss a good emission).
    false_negative_rate: float = 0.05
    # CRF duration (steps before thinning begins).
    crf_duration: int = 50_000
    # Minimum reinforcement probability after thinning.
    min_reinforcement_prob: float = 0.3
    # Steps over which thinning occurs (after crf_duration).
    thinning_duration: int = 200_000


@dataclass
class LearnerConfig:
    """Parameters for the child learner agent."""
    # Initial softmax temperature (high = babbling).
    initial_temperature: float = 5.0
    # Final softmax temperature floor.
    min_temperature: float = 0.3
    # Temperature decay rate per step (exponential).
    temperature_decay: float = 5e-6
    # REINFORCE learning rate.
    learning_rate: float = 0.05
    # Baseline learning rate (running average of reward).
    baseline_lr: float = 0.005
    # Entropy bonus coefficient (encourages exploration).
    entropy_bonus: float = 0.01
    # Receptive system (Hebbian) learning rate.
    receptive_lr: float = 0.1


@dataclass
class CaregiverConfig:
    """Parameters for the adaptive CDS caregiver."""
    # How strongly caregiver biases toward partially-learned words (ZPD).
    zpd_bias: float = 0.6
    # Range of similarity considered "zone of proximal development".
    zpd_low: float = 0.3
    zpd_high: float = 0.7
    # Frequency of caregiver word selection adaptation (steps).
    adaptation_interval: int = 5000


@dataclass
class EnvironmentConfig:
    """Parameters for trial scheduling and motivating operations."""
    # Initial trial type probabilities: [echoic, tact, mand, intraverbal].
    initial_trial_probs: list[float] = field(
        default_factory=lambda: [0.50, 0.25, 0.15, 0.10]
    )
    # Final trial type probabilities.
    final_trial_probs: list[float] = field(
        default_factory=lambda: [0.10, 0.30, 0.30, 0.30]
    )
    # Steps over which trial probabilities shift from initial to final.
    trial_shift_duration: int = 300_000
    # Number of MO dimensions (independent deprivation states).
    n_mo_dims: int = 5
    # MO fluctuation rate (probability of change per step).
    mo_fluctuation_rate: float = 0.01
    # Satiation amount after successful mand.
    satiation_amount: float = 0.5


@dataclass
class SimulationConfig:
    """Top-level configuration containing all sub-configs."""
    seed: int = 42
    n_steps: int = 500_000
    log_every: int = 1000

    phonology: PhonologyConfig = field(default_factory=PhonologyConfig)
    lexicon: LexiconConfig = field(default_factory=LexiconConfig)
    reinforcement: ReinforcementConfig = field(default_factory=ReinforcementConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    caregiver: CaregiverConfig = field(default_factory=CaregiverConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> SimulationConfig:
        d = json.loads(s)
        return cls(
            seed=d["seed"],
            n_steps=d["n_steps"],
            log_every=d["log_every"],
            phonology=PhonologyConfig(**d["phonology"]),
            lexicon=LexiconConfig(**d["lexicon"]),
            reinforcement=ReinforcementConfig(**d["reinforcement"]),
            learner=LearnerConfig(**d["learner"]),
            caregiver=CaregiverConfig(**d["caregiver"]),
            environment=EnvironmentConfig(**d["environment"]),
        )
