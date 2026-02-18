"""Post-hoc analysis metrics.

Acquisition curves, word emergence detection, confusion matrices,
phoneme accuracy, developmental milestones.
"""

from __future__ import annotations

import numpy as np

from .phonology import PhonologySystem, N_PHONEMES, SILENCE_IDX
from .lexicon import Lexicon
from .learner import Learner
from .simulation import SimulationLog


def acquisition_step(
    log: SimulationLog,
    threshold: float = 0.5,
) -> dict[str, int | None]:
    """Find the step at which each operant's mean similarity first exceeds threshold.

    Returns dict mapping operant name → step (or None if never reached).
    """
    result: dict[str, int | None] = {}
    for op in ["echoic", "tact", "mand", "intraverbal"]:
        sims = log.similarity_by_operant[op]
        found = None
        for i, s in enumerate(sims):
            if s >= threshold:
                found = log.steps[i]
                break
        result[op] = found
    return result


def word_emergence(
    log: SimulationLog,
    lexicon: Lexicon,
    threshold: float = 0.5,
) -> dict[str, int | None]:
    """Find the step at which each word's similarity EMA first exceeds threshold.

    Returns dict mapping word name → step (or None).
    """
    result: dict[str, int | None] = {}
    snapshots = log.word_similarity_snapshots
    for w in range(lexicon.n_words):
        found = None
        for i, snap in enumerate(snapshots):
            if snap[w] >= threshold:
                found = log.steps[i]
                break
        result[lexicon.word_names[w]] = found
    return result


def confusion_matrix(
    learner: Learner,
    lexicon: Lexicon,
    phonology: PhonologySystem,
    operant_name: str = "echoic",
    n_samples: int = 10,
) -> np.ndarray:
    """Build a word×word confusion matrix.

    For each target word, sample n_samples emissions from the specified operant
    and compute similarity to every target word.  Returns (n_words, n_words)
    matrix where entry [i, j] = avg similarity of emissions targeting word i
    to the phoneme sequence of word j.
    """
    n_words = lexicon.n_words
    C = np.zeros((n_words, n_words))
    system = learner.operants[operant_name]

    for i in range(n_words):
        # Build input features.
        if operant_name in ("echoic", "intraverbal"):
            input_features = phonology.encode_phonemes(lexicon.word_phonemes[i])
        else:
            input_features = lexicon.context_features[i]

        for _ in range(n_samples):
            emitted, _ = system.sample(
                input_features, learner.temperature, learner.rng
            )
            for j in range(n_words):
                target = lexicon.word_phonemes[j]
                C[i, j] += phonology.word_similarity(emitted, target)

    C /= n_samples
    return C


def phoneme_accuracy_by_position(
    learner: Learner,
    lexicon: Lexicon,
    phonology: PhonologySystem,
    operant_name: str = "echoic",
    n_samples: int = 20,
) -> np.ndarray:
    """Compute per-position phoneme accuracy across all words.

    Returns (n_words, max_len) matrix of per-position similarity.
    """
    n_words = lexicon.n_words
    max_len = lexicon.max_len
    accuracy = np.zeros((n_words, max_len))
    system = learner.operants[operant_name]

    for i in range(n_words):
        if operant_name in ("echoic", "intraverbal"):
            input_features = phonology.encode_phonemes(lexicon.word_phonemes[i])
        else:
            input_features = lexicon.context_features[i]

        for _ in range(n_samples):
            emitted, _ = system.sample(
                input_features, learner.temperature, learner.rng
            )
            for pos in range(max_len):
                accuracy[i, pos] += phonology.phoneme_similarity(
                    emitted[pos], lexicon.word_phonemes[i, pos]
                )

    accuracy /= n_samples
    return accuracy


def developmental_milestones(
    log: SimulationLog,
    lexicon: Lexicon,
    similarity_thresholds: list[float] | None = None,
) -> dict[str, dict[str, int | None]]:
    """Report when the learner hits various milestones.

    Returns nested dict: {milestone_name: {detail: step}}.
    """
    if similarity_thresholds is None:
        similarity_thresholds = [0.3, 0.5, 0.7]

    milestones: dict[str, dict[str, int | None]] = {}

    # Operant acquisition milestones.
    for thresh in similarity_thresholds:
        key = f"operant_sim>{thresh:.1f}"
        milestones[key] = acquisition_step(log, threshold=thresh)

    # Word emergence milestones.
    for thresh in similarity_thresholds:
        key = f"word_sim>{thresh:.1f}"
        emergence = word_emergence(log, lexicon, threshold=thresh)
        # Summarize: count of emerged words at end.
        n_emerged = sum(1 for v in emergence.values() if v is not None)
        first_word = min(
            ((k, v) for k, v in emergence.items() if v is not None),
            key=lambda x: x[1],
            default=(None, None),
        )
        milestones[key] = {
            "n_words_emerged": n_emerged,
            "first_word": first_word[0],
            "first_word_step": first_word[1],
        }

    return milestones


def receptive_similarity_matrix(learner: Learner) -> np.ndarray:
    """Cosine similarity of learned receptive representations.

    Returns (n_words, n_words) matrix.
    """
    A = learner.receptive.A
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    U = A / norms
    return U @ U.T
