"""Main simulation loop.

Per trial: environment generates trial → caregiver may model → child emits
phoneme sequence → caregiver evaluates → operant update → temperature decay
→ periodic caregiver adaptation.

Logging snapshots every N steps with per-operant similarity, reinforcement
rates, thresholds, babble temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SimulationConfig
from .phonology import PhonologySystem, SIMILARITY_MATRIX, SILENCE_IDX
from .lexicon import Lexicon
from .learner import Learner
from .caregiver import Caregiver
from .environment import Environment, TRIAL_TYPES


@dataclass
class SimulationLog:
    """Accumulated simulation metrics."""
    steps: list[int] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)

    # Per-operant running metrics (windowed averages at each log point).
    similarity_by_operant: dict[str, list[float]] = field(
        default_factory=lambda: {
            "echoic": [], "tact": [], "mand": [], "intraverbal": [],
        }
    )
    reinforcement_rate_by_operant: dict[str, list[float]] = field(
        default_factory=lambda: {
            "echoic": [], "tact": [], "mand": [], "intraverbal": [],
        }
    )
    overall_similarity: list[float] = field(default_factory=list)
    overall_reinforcement_rate: list[float] = field(default_factory=list)

    # Per-word best similarity at each log point.
    word_similarity_snapshots: list[np.ndarray] = field(default_factory=list)

    # Shaping thresholds at each log point.
    threshold_snapshots: list[np.ndarray] = field(default_factory=list)

    # Trial type counts per window.
    trial_type_counts: dict[str, list[int]] = field(
        default_factory=lambda: {
            "echoic": [], "tact": [], "mand": [], "intraverbal": [],
        }
    )


def _fast_word_similarity(
    emitted: np.ndarray,
    target: np.ndarray,
    sim_matrix: np.ndarray,
    silence_idx: int,
) -> float:
    """Inlined word similarity for the hot loop."""
    sims = sim_matrix[emitted, target]
    active = ~((emitted == silence_idx) & (target == silence_idx))
    n = active.sum()
    if n == 0:
        return 1.0
    return sims[active].sum() / n


def run_simulation(
    config: SimulationConfig,
    progress_callback: callable | None = None,
) -> tuple[SimulationLog, Learner, Lexicon]:
    """Run the full simulation and return logs + trained learner.

    Parameters
    ----------
    config : SimulationConfig
        Full simulation configuration.
    progress_callback : callable, optional
        Called as ``progress_callback(step, n_steps)`` every *log_every* steps.
        Useful for driving progress bars in UIs.
    """
    rng = np.random.default_rng(config.seed)

    # Build components.
    phonology = PhonologySystem(config.phonology)
    lexicon = Lexicon(config.lexicon, rng)
    learner = Learner(
        config.learner,
        phonology,
        lexicon.n_words,
        config.lexicon.max_word_len,
        config.lexicon.context_dim,
        rng,
    )
    caregiver = Caregiver(
        config.caregiver,
        config.reinforcement,
        phonology,
        lexicon,
        rng,
    )
    environment = Environment(config.environment, rng)

    log = SimulationLog()

    # Pre-cache frequently accessed arrays for the hot loop.
    sim_matrix = SIMILARITY_MATRIX
    silence_idx = SILENCE_IDX
    word_phonemes = lexicon.word_phonemes
    context_features = lexicon.context_features
    iv_map = lexicon.intraverbal_map
    mo_assignments = lexicon.mo_assignments
    adapt_interval = config.caregiver.adaptation_interval
    log_every = config.log_every
    n_steps = config.n_steps

    # Pre-encode all word phoneme sequences for echoic/intraverbal input.
    features_arr = phonology.features
    max_len = config.lexicon.max_word_len
    encoded_words = np.zeros((lexicon.n_words, max_len * features_arr.shape[1]))
    for i in range(lexicon.n_words):
        encoded_words[i] = features_arr[word_phonemes[i]].ravel()

    # Operant systems indexed by trial type index (0-3).
    operant_names = TRIAL_TYPES  # ["echoic", "tact", "mand", "intraverbal"]
    operant_systems = [learner.operants[n] for n in operant_names]

    # Temperature decay precomputation.
    temp_init = config.learner.initial_temperature
    temp_decay = config.learner.temperature_decay
    temp_min = config.learner.min_temperature

    # Windowed accumulators for logging.
    window_sim = [[] for _ in range(4)]
    window_rein = [[] for _ in range(4)]
    window_trial_counts = [0, 0, 0, 0]

    for step in range(1, n_steps + 1):
        # --- MO fluctuation ---
        environment.fluctuate_mo()

        # --- Trial generation ---
        trial_type_idx = environment.sample_trial_type(step)
        target_idx = caregiver.select_word(
            operant_names[trial_type_idx], environment.mo_state
        )
        context = context_features[target_idx]

        # --- Build input features for the operant ---
        if trial_type_idx == 0:  # echoic
            input_features = encoded_words[target_idx]
            learner.receptive.update(target_idx, context)
        elif trial_type_idx == 1:  # tact
            input_features = context
            learner.receptive.update(target_idx, context)
        elif trial_type_idx == 2:  # mand
            input_features = context
        else:  # intraverbal
            stim_idx = target_idx
            if stim_idx in iv_map:
                target_idx = iv_map[stim_idx]
            input_features = encoded_words[stim_idx]

        # --- Child emits ---
        system = operant_systems[trial_type_idx]
        emitted, probs = system.sample(input_features, learner.temperature, rng)

        # --- Caregiver evaluates (shaping + schedule) ---
        reinforced, reward, similarity = caregiver.evaluate(
            target_idx, emitted, step
        )

        # --- Operant update ---
        if reward > 0:
            system.update(input_features, emitted, probs, reward)

        # --- Mand-specific: satiation if reinforced ---
        if trial_type_idx == 2 and reinforced:
            mo_dim = mo_assignments[target_idx]
            if mo_dim >= 0:
                environment.satiate(mo_dim)

        # --- Temperature decay ---
        learner.step_count = step
        learner.temperature = max(
            temp_min, temp_init * np.exp(-temp_decay * step)
        )

        # --- Accumulate window stats ---
        window_sim[trial_type_idx].append(similarity)
        window_rein[trial_type_idx].append(float(reinforced))
        window_trial_counts[trial_type_idx] += 1

        # --- Periodic caregiver adaptation ---
        if step % adapt_interval == 0:
            caregiver.adapt_word_selection()

        # --- Logging ---
        if step % log_every == 0:
            log.steps.append(step)
            log.temperature.append(learner.temperature)

            all_sims: list[float] = []
            all_reins: list[float] = []

            for i in range(4):
                op = operant_names[i]
                sims = window_sim[i]
                reins = window_rein[i]
                avg_sim = float(np.mean(sims)) if sims else 0.0
                avg_rein = float(np.mean(reins)) if reins else 0.0
                log.similarity_by_operant[op].append(avg_sim)
                log.reinforcement_rate_by_operant[op].append(avg_rein)
                log.trial_type_counts[op].append(window_trial_counts[i])
                all_sims.extend(sims)
                all_reins.extend(reins)

            log.overall_similarity.append(
                float(np.mean(all_sims)) if all_sims else 0.0
            )
            log.overall_reinforcement_rate.append(
                float(np.mean(all_reins)) if all_reins else 0.0
            )

            log.word_similarity_snapshots.append(
                caregiver._word_similarity_ema.copy()
            )
            log.threshold_snapshots.append(
                caregiver.shaping.thresholds.copy()
            )

            # Reset window.
            window_sim = [[] for _ in range(4)]
            window_rein = [[] for _ in range(4)]
            window_trial_counts = [0, 0, 0, 0]

            if progress_callback is not None:
                progress_callback(step, n_steps)

    return log, learner, lexicon
