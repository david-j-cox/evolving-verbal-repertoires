"""Diagnostic plots for the evolving verbal behavior simulation.

Dashboard with 6 panels + individual plot functions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .lexicon import Lexicon
from .learner import Learner
from .phonology import PhonologySystem
from .simulation import SimulationLog
from . import analysis


def dashboard(
    log: SimulationLog,
    learner: Learner,
    lexicon: Lexicon,
    phonology: PhonologySystem,
) -> plt.Figure:
    """6-panel diagnostic dashboard."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    steps = np.array(log.steps)

    # Panel 1: Acquisition by operant type.
    ax1 = fig.add_subplot(gs[0, 0])
    for op in ["echoic", "tact", "mand", "intraverbal"]:
        ax1.plot(steps, log.similarity_by_operant[op], label=op)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean similarity")
    ax1.set_title("Acquisition by operant type")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1)

    # Panel 2: Reinforcement rate by operant type.
    ax2 = fig.add_subplot(gs[0, 1])
    for op in ["echoic", "tact", "mand", "intraverbal"]:
        ax2.plot(steps, log.reinforcement_rate_by_operant[op], label=op)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reinforcement rate")
    ax2.set_title("Reinforcement rate")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)

    # Panel 3: Babble temperature.
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, log.temperature, color="tab:red")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Temperature")
    ax3.set_title("Babble temperature")

    # Panel 4: Shaping thresholds (mean + range).
    ax4 = fig.add_subplot(gs[1, 0])
    thresholds = np.array(log.threshold_snapshots)
    ax4.plot(steps, thresholds.mean(axis=1), label="Mean threshold")
    ax4.fill_between(
        steps,
        thresholds.min(axis=1),
        thresholds.max(axis=1),
        alpha=0.2,
        label="Range",
    )
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Threshold")
    ax4.set_title("Shaping thresholds")
    ax4.legend(fontsize=8)

    # Panel 5: Per-word similarity heatmap.
    ax5 = fig.add_subplot(gs[1, 1])
    word_sims = np.array(log.word_similarity_snapshots)  # (n_logs, n_words)
    # Subsample for readability.
    n_ticks = min(50, len(steps))
    step_stride = max(1, len(steps) // n_ticks)
    im = ax5.imshow(
        word_sims[::step_stride].T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax5.set_xlabel("Log point")
    ax5.set_ylabel("Word index")
    ax5.set_title("Per-word similarity over time")
    plt.colorbar(im, ax=ax5, shrink=0.8)

    # Panel 6: Receptive semantic network similarity.
    ax6 = fig.add_subplot(gs[1, 2])
    rec_sim = analysis.receptive_similarity_matrix(learner)
    im6 = ax6.imshow(rec_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax6.set_xlabel("Word")
    ax6.set_ylabel("Word")
    ax6.set_title("Receptive similarity")
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    # Label ticks with word names if not too many.
    if lexicon.n_words <= 40:
        ax6.set_xticks(range(lexicon.n_words))
        ax6.set_xticklabels(lexicon.word_names, rotation=90, fontsize=5)
        ax6.set_yticks(range(lexicon.n_words))
        ax6.set_yticklabels(lexicon.word_names, fontsize=5)

    fig.suptitle("Evolving Verbal Behavior — Simulation Dashboard", fontsize=14)
    return fig


def plot_acquisition_curves(log: SimulationLog) -> plt.Figure:
    """Individual plot: acquisition curves per operant type."""
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.array(log.steps)
    for op in ["echoic", "tact", "mand", "intraverbal"]:
        ax.plot(steps, log.similarity_by_operant[op], label=op)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean similarity")
    ax.set_title("Acquisition curves by operant type")
    ax.legend()
    ax.set_ylim(0, 1)
    return fig


def plot_confusion_heatmap(
    learner: Learner,
    lexicon: Lexicon,
    phonology: PhonologySystem,
    operant_name: str = "echoic",
    n_samples: int = 10,
) -> plt.Figure:
    """Confusion heatmap: what does the learner produce for each target?"""
    C = analysis.confusion_matrix(
        learner, lexicon, phonology, operant_name, n_samples
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(C, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Compared-to word")
    ax.set_ylabel("Target word")
    ax.set_title(f"Confusion matrix ({operant_name})")
    plt.colorbar(im, ax=ax)
    if lexicon.n_words <= 40:
        ax.set_xticks(range(lexicon.n_words))
        ax.set_xticklabels(lexicon.word_names, rotation=90, fontsize=6)
        ax.set_yticks(range(lexicon.n_words))
        ax.set_yticklabels(lexicon.word_names, fontsize=6)
    return fig


def plot_shaping_trajectory(
    log: SimulationLog,
    lexicon: Lexicon,
    word_names: list[str] | None = None,
) -> plt.Figure:
    """Shaping trajectory for specific words."""
    if word_names is None:
        word_names = ["mama", "ball", "dog", "milk", "no"]

    fig, ax = plt.subplots(figsize=(10, 5))
    steps = np.array(log.steps)
    word_sims = np.array(log.word_similarity_snapshots)

    for name in word_names:
        if name in lexicon.word_to_idx:
            idx = lexicon.word_to_idx[name]
            ax.plot(steps, word_sims[:, idx], label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Similarity (EMA)")
    ax.set_title("Shaping trajectories")
    ax.legend()
    ax.set_ylim(0, 1)
    return fig


def plot_phoneme_accuracy(
    learner: Learner,
    lexicon: Lexicon,
    phonology: PhonologySystem,
    operant_name: str = "echoic",
    n_samples: int = 20,
) -> plt.Figure:
    """Per-position phoneme accuracy heatmap."""
    accuracy = analysis.phoneme_accuracy_by_position(
        learner, lexicon, phonology, operant_name, n_samples
    )
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(accuracy, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Position")
    ax.set_ylabel("Word")
    ax.set_title(f"Phoneme accuracy by position ({operant_name})")
    plt.colorbar(im, ax=ax)
    if lexicon.n_words <= 40:
        ax.set_yticks(range(lexicon.n_words))
        ax.set_yticklabels(lexicon.word_names, fontsize=6)
    ax.set_xticks(range(lexicon.max_len))
    return fig
