"""Streamlit browser UI for the Evolving Verbal Behavior simulation.

Launch with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from evb import (
    SimulationConfig,
    run_simulation,
    SimulationLog,
    Learner,
    Lexicon,
    PhonologySystem,
    analysis,
    visualization,
)
from evb.config import (
    LexiconConfig,
    ReinforcementConfig,
    LearnerConfig,
    CaregiverConfig,
    EnvironmentConfig,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Evolving Verbal Behavior",
    page_icon="🗣️",
    layout="wide",
)

st.title("Evolving Verbal Behavior")
st.caption("Phoneme-level simulation of Skinner's verbal operants")

# ---------------------------------------------------------------------------
# Helper: normalize 4 probabilities
# ---------------------------------------------------------------------------

def _normalize4(vals: list[float]) -> list[float]:
    s = sum(vals)
    if s == 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [v / s for v in vals]


# ---------------------------------------------------------------------------
# Sidebar — parameter controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Parameters")

    # --- Config import/export ---
    with st.expander("Config Import / Export"):
        uploaded = st.file_uploader("Import config JSON", type=["json"])
        if uploaded is not None:
            try:
                cfg_json = uploaded.read().decode("utf-8")
                st.session_state["imported_config"] = SimulationConfig.from_json(cfg_json)
                st.success("Config loaded!")
            except Exception as e:
                st.error(f"Failed to parse config: {e}")

    # Use imported config as defaults if available.
    defaults = st.session_state.get("imported_config", SimulationConfig())

    # --- Simulation ---
    with st.expander("Simulation", expanded=True):
        seed = st.number_input("Seed", value=defaults.seed, step=1)
        n_steps = st.number_input(
            "Steps", value=defaults.n_steps, min_value=1000, step=1000,
        )
        log_every = st.number_input(
            "Log every", value=defaults.log_every, min_value=100, step=100,
        )

    # --- Lexicon ---
    with st.expander("Lexicon"):
        max_word_len = st.number_input(
            "Max word length", value=defaults.lexicon.max_word_len,
            min_value=2, max_value=10,
        )
        context_dim = st.number_input(
            "Context dim", value=defaults.lexicon.context_dim,
            min_value=5, max_value=100,
        )
        category_overlap = st.slider(
            "Category overlap", 0.0, 1.0, defaults.lexicon.category_overlap,
        )

    # --- Reinforcement & Shaping ---
    with st.expander("Reinforcement & Shaping"):
        initial_threshold = st.slider(
            "Initial threshold", 0.0, 1.0,
            defaults.reinforcement.initial_threshold,
        )
        max_threshold = st.slider(
            "Max threshold", 0.0, 1.0,
            defaults.reinforcement.max_threshold,
        )
        threshold_ratchet_rate = st.slider(
            "Ratchet rate", 0.0, 1.0,
            defaults.reinforcement.threshold_ratchet_rate,
        )
        threshold_margin = st.slider(
            "Threshold margin", 0.0, 0.5,
            defaults.reinforcement.threshold_margin,
        )
        false_positive_rate = st.slider(
            "False positive rate", 0.0, 0.3,
            defaults.reinforcement.false_positive_rate,
        )
        false_negative_rate = st.slider(
            "False negative rate", 0.0, 0.3,
            defaults.reinforcement.false_negative_rate,
        )
        crf_duration = st.number_input(
            "CRF duration", value=defaults.reinforcement.crf_duration,
            min_value=0, step=10000,
        )
        min_reinforcement_prob = st.slider(
            "Min reinforcement prob", 0.0, 1.0,
            defaults.reinforcement.min_reinforcement_prob,
        )
        thinning_duration = st.number_input(
            "Thinning duration",
            value=defaults.reinforcement.thinning_duration,
            min_value=0, step=10000,
        )

    # --- Learner ---
    with st.expander("Learner"):
        initial_temperature = st.number_input(
            "Initial temperature",
            value=defaults.learner.initial_temperature,
            min_value=0.1, step=0.5, format="%.1f",
        )
        min_temperature = st.number_input(
            "Min temperature",
            value=defaults.learner.min_temperature,
            min_value=0.1, step=0.1, format="%.1f",
        )
        temperature_decay = st.number_input(
            "Temperature decay",
            value=defaults.learner.temperature_decay,
            min_value=0.0, step=1e-6, format="%.2e",
        )
        learning_rate = st.number_input(
            "Learning rate",
            value=defaults.learner.learning_rate,
            min_value=0.001, step=0.01, format="%.3f",
        )
        baseline_lr = st.number_input(
            "Baseline LR",
            value=defaults.learner.baseline_lr,
            min_value=0.001, step=0.01, format="%.3f",
        )
        entropy_bonus = st.number_input(
            "Entropy bonus",
            value=defaults.learner.entropy_bonus,
            min_value=0.0, step=0.01, format="%.3f",
        )
        receptive_lr = st.number_input(
            "Receptive LR",
            value=defaults.learner.receptive_lr,
            min_value=0.001, step=0.01, format="%.3f",
        )

    # --- Caregiver ---
    with st.expander("Caregiver"):
        zpd_bias = st.slider(
            "ZPD bias", 0.0, 1.0, defaults.caregiver.zpd_bias,
        )
        zpd_low = st.slider(
            "ZPD low", 0.0, 1.0, defaults.caregiver.zpd_low,
        )
        zpd_high = st.slider(
            "ZPD high", 0.0, 1.0, defaults.caregiver.zpd_high,
        )
        adaptation_interval = st.number_input(
            "Adaptation interval",
            value=defaults.caregiver.adaptation_interval,
            min_value=100, step=1000,
        )

    # --- Environment ---
    with st.expander("Environment"):
        st.markdown("**Initial trial probabilities** *(auto-normalized)*")
        itp = defaults.environment.initial_trial_probs
        it_echo = st.slider("Echoic (initial)", 0.0, 1.0, itp[0], key="it_echo")
        it_tact = st.slider("Tact (initial)", 0.0, 1.0, itp[1], key="it_tact")
        it_mand = st.slider("Mand (initial)", 0.0, 1.0, itp[2], key="it_mand")
        it_iv = st.slider("Intraverbal (initial)", 0.0, 1.0, itp[3], key="it_iv")
        initial_trial_probs = _normalize4([it_echo, it_tact, it_mand, it_iv])

        st.markdown("**Final trial probabilities** *(auto-normalized)*")
        ftp = defaults.environment.final_trial_probs
        ft_echo = st.slider("Echoic (final)", 0.0, 1.0, ftp[0], key="ft_echo")
        ft_tact = st.slider("Tact (final)", 0.0, 1.0, ftp[1], key="ft_tact")
        ft_mand = st.slider("Mand (final)", 0.0, 1.0, ftp[2], key="ft_mand")
        ft_iv = st.slider("Intraverbal (final)", 0.0, 1.0, ftp[3], key="ft_iv")
        final_trial_probs = _normalize4([ft_echo, ft_tact, ft_mand, ft_iv])

        trial_shift_duration = st.number_input(
            "Trial shift duration",
            value=defaults.environment.trial_shift_duration,
            min_value=0, step=10000,
        )
        n_mo_dims = st.number_input(
            "MO dimensions",
            value=defaults.environment.n_mo_dims,
            min_value=1, max_value=20,
        )
        mo_fluctuation_rate = st.slider(
            "MO fluctuation rate", 0.0, 0.1,
            defaults.environment.mo_fluctuation_rate,
        )
        satiation_amount = st.slider(
            "Satiation amount", 0.0, 1.0,
            defaults.environment.satiation_amount,
        )

    # --- Export current config ---
    current_config = SimulationConfig(
        seed=seed,
        n_steps=n_steps,
        log_every=log_every,
        lexicon=LexiconConfig(
            max_word_len=max_word_len,
            context_dim=context_dim,
            category_overlap=category_overlap,
        ),
        reinforcement=ReinforcementConfig(
            initial_threshold=initial_threshold,
            max_threshold=max_threshold,
            threshold_ratchet_rate=threshold_ratchet_rate,
            threshold_margin=threshold_margin,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            crf_duration=crf_duration,
            min_reinforcement_prob=min_reinforcement_prob,
            thinning_duration=thinning_duration,
        ),
        learner=LearnerConfig(
            initial_temperature=initial_temperature,
            min_temperature=min_temperature,
            temperature_decay=temperature_decay,
            learning_rate=learning_rate,
            baseline_lr=baseline_lr,
            entropy_bonus=entropy_bonus,
            receptive_lr=receptive_lr,
        ),
        caregiver=CaregiverConfig(
            zpd_bias=zpd_bias,
            zpd_low=zpd_low,
            zpd_high=zpd_high,
            adaptation_interval=adaptation_interval,
        ),
        environment=EnvironmentConfig(
            initial_trial_probs=initial_trial_probs,
            final_trial_probs=final_trial_probs,
            trial_shift_duration=trial_shift_duration,
            n_mo_dims=n_mo_dims,
            mo_fluctuation_rate=mo_fluctuation_rate,
            satiation_amount=satiation_amount,
        ),
    )

    st.download_button(
        "Export config JSON",
        data=current_config.to_json(),
        file_name="evb_config.json",
        mime="application/json",
    )

# ---------------------------------------------------------------------------
# Main area — Run simulation
# ---------------------------------------------------------------------------

# Estimated run time (rough heuristic: ~150us/step).
est_seconds = n_steps * 150e-6
if est_seconds < 60:
    est_str = f"~{est_seconds:.0f}s"
elif est_seconds < 3600:
    est_str = f"~{est_seconds / 60:.1f}min"
else:
    est_str = f"~{est_seconds / 3600:.1f}hr"

st.markdown(f"**{n_steps:,} steps** — estimated run time: {est_str}")

# Detect stale results.
if "last_config_json" in st.session_state:
    if current_config.to_json() != st.session_state["last_config_json"]:
        st.warning("Parameters have changed since the last run. Results below are stale.")

# Run button.
if st.button("Run Simulation", type="primary"):
    progress_bar = st.progress(0, text="Starting simulation...")

    def _progress(step: int, total: int) -> None:
        frac = step / total
        progress_bar.progress(frac, text=f"Step {step:,} / {total:,}")

    with st.spinner("Running simulation..."):
        log, learner, lexicon = run_simulation(current_config, progress_callback=_progress)

    progress_bar.progress(1.0, text="Done!")

    # Store results in session state.
    phonology = PhonologySystem(current_config.phonology)
    st.session_state["results"] = {
        "log": log,
        "learner": learner,
        "lexicon": lexicon,
        "phonology": phonology,
        "config": current_config,
    }
    st.session_state["last_config_json"] = current_config.to_json()
    st.rerun()


# ---------------------------------------------------------------------------
# Results display (persists in session_state)
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.info("Configure parameters in the sidebar, then click **Run Simulation**.")
    st.stop()

res = st.session_state["results"]
log: SimulationLog = res["log"]
learner: Learner = res["learner"]
lexicon: Lexicon = res["lexicon"]
phonology: PhonologySystem = res["phonology"]
run_config: SimulationConfig = res["config"]

tab_dash, tab_analysis, tab_plots, tab_raw = st.tabs(
    ["Dashboard", "Analysis", "Individual Plots", "Raw Data"]
)

# --- Tab 1: Dashboard ---
with tab_dash:
    fig = visualization.dashboard(log, learner, lexicon, phonology)
    st.pyplot(fig)
    plt.close(fig)

# --- Tab 2: Analysis ---
with tab_analysis:
    import pandas as pd

    st.subheader("Developmental milestones")
    milestones = analysis.developmental_milestones(log, lexicon)
    st.json(milestones)

    st.subheader("Operant acquisition steps")
    for thresh in [0.3, 0.5, 0.7]:
        acq = analysis.acquisition_step(log, threshold=thresh)
        st.markdown(f"**Threshold {thresh}:** " + ", ".join(
            f"{op}={step if step else 'N/A'}" for op, step in acq.items()
        ))

    st.subheader("Word emergence")
    emergence = analysis.word_emergence(log, lexicon, threshold=0.5)
    df_emergence = pd.DataFrame([
        {"Word": word, "Step emerged": step if step else "N/A"}
        for word, step in sorted(
            emergence.items(), key=lambda x: (x[1] is None, x[1])
        )
    ])
    st.dataframe(df_emergence, use_container_width=True)

# --- Tab 3: Individual Plots ---
with tab_plots:
    plot_type = st.selectbox("Plot type", [
        "Acquisition curves",
        "Confusion heatmap",
        "Shaping trajectory",
        "Phoneme accuracy",
    ])

    if plot_type == "Acquisition curves":
        fig = visualization.plot_acquisition_curves(log)
        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Confusion heatmap":
        operant = st.selectbox(
            "Operant", ["echoic", "tact", "mand", "intraverbal"],
            key="confusion_op",
        )
        n_samples = st.slider("Samples per word", 1, 50, 10, key="confusion_n")
        fig = visualization.plot_confusion_heatmap(
            learner, lexicon, phonology, operant, n_samples,
        )
        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Shaping trajectory":
        available_words = lexicon.word_names
        default_words = [w for w in ["mama", "ball", "dog", "milk", "no"]
                         if w in available_words]
        selected_words = st.multiselect(
            "Words to plot", available_words, default=default_words,
        )
        if selected_words:
            fig = visualization.plot_shaping_trajectory(log, lexicon, selected_words)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Select at least one word.")

    elif plot_type == "Phoneme accuracy":
        operant = st.selectbox(
            "Operant", ["echoic", "tact", "mand", "intraverbal"],
            key="phoneme_op",
        )
        n_samples = st.slider("Samples per word", 1, 50, 20, key="phoneme_n")
        fig = visualization.plot_phoneme_accuracy(
            learner, lexicon, phonology, operant, n_samples,
        )
        st.pyplot(fig)
        plt.close(fig)

# --- Tab 4: Raw Data ---
with tab_raw:
    import pandas as pd

    st.subheader("Configuration")
    st.json(json.loads(run_config.to_json()))

    st.subheader("Simulation log")
    # Build a flat dataframe from the log.
    log_data = {"step": log.steps, "temperature": log.temperature}
    for op in ["echoic", "tact", "mand", "intraverbal"]:
        log_data[f"sim_{op}"] = log.similarity_by_operant[op]
        log_data[f"rein_{op}"] = log.reinforcement_rate_by_operant[op]
        log_data[f"trials_{op}"] = log.trial_type_counts[op]
    log_data["overall_sim"] = log.overall_similarity
    log_data["overall_rein"] = log.overall_reinforcement_rate
    df_log = pd.DataFrame(log_data)
    st.dataframe(df_log, use_container_width=True)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Save results")

default_label = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_label = st.text_input("Run label", value=default_label)

if st.button("Save Results"):
    out_dir = Path("research_output") / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config JSON.
    (out_dir / "config.json").write_text(run_config.to_json())

    # Dashboard figure.
    fig = visualization.dashboard(log, learner, lexicon, phonology)
    fig.savefig(out_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual plots.
    fig = visualization.plot_acquisition_curves(log)
    fig.savefig(out_dir / "acquisition_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = visualization.plot_confusion_heatmap(learner, lexicon, phonology, "echoic")
    fig.savefig(out_dir / "confusion_echoic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = visualization.plot_phoneme_accuracy(learner, lexicon, phonology, "echoic")
    fig.savefig(out_dir / "phoneme_accuracy_echoic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Milestones JSON.
    milestones = analysis.developmental_milestones(log, lexicon)
    (out_dir / "milestones.json").write_text(json.dumps(milestones, indent=2))

    # Word emergence CSV.
    import pandas as pd
    emergence = analysis.word_emergence(log, lexicon, threshold=0.5)
    df_em = pd.DataFrame([
        {"word": w, "step_emerged": s} for w, s in emergence.items()
    ])
    df_em.to_csv(out_dir / "word_emergence.csv", index=False)

    # Full simulation log CSV.
    log_data = {"step": log.steps, "temperature": log.temperature}
    for op in ["echoic", "tact", "mand", "intraverbal"]:
        log_data[f"sim_{op}"] = log.similarity_by_operant[op]
        log_data[f"rein_{op}"] = log.reinforcement_rate_by_operant[op]
        log_data[f"trials_{op}"] = log.trial_type_counts[op]
    log_data["overall_sim"] = log.overall_similarity
    log_data["overall_rein"] = log.overall_reinforcement_rate
    pd.DataFrame(log_data).to_csv(out_dir / "simulation_log.csv", index=False)

    st.success(f"Saved to `{out_dir}/`")
