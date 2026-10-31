"""Experiment design and hypothesis testing dashboard page."""

import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.experiment_design import (
    run_home_advantage_experiment,
    run_model_calibration_experiment,
)

st.set_page_config(page_title="Pitchline — Experiments", layout="wide", page_icon="🧪")

st.title("🧪 Designed Experiments")
st.caption(
    "Two formal hypothesis tests run against the real 1990-2026 dataset — each with an "
    "explicit null hypothesis, a named statistical test, and a reported p-value."
)

with st.spinner("Running experiments against the real combined dataset..."):
    home_adv = run_home_advantage_experiment()
    calibration, buckets = run_model_calibration_experiment()

tab1, tab2 = st.tabs(["Home Advantage Test", "Model Calibration Test"])

with tab1:
    st.subheader(home_adv.experiment_name)
    st.markdown(f"**H₀:** {home_adv.null_hypothesis}")
    st.markdown(f"**H₁:** {home_adv.alternative_hypothesis}")
    st.markdown(f"**Test used:** {home_adv.test_used}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test statistic (z)", home_adv.test_statistic)
    with col2:
        st.metric("p-value", f"{home_adv.p_value:.2e}")
    with col3:
        st.metric("Sample size", home_adv.sample_size)

    if home_adv.significant:
        st.success(f"**Statistically significant** at α = 0.05. {home_adv.effect_description}")
    else:
        st.info(f"Not statistically significant at α = 0.05. {home_adv.effect_description}")

    st.caption(
        "Only non-neutral-venue matches are included — home advantage structurally cannot "
        "apply at a neutral venue, so including those matches would bias the test toward "
        "finding no effect."
    )

with tab2:
    st.subheader(calibration.experiment_name)
    st.markdown(f"**H₀:** {calibration.null_hypothesis}")
    st.markdown(f"**H₁:** {calibration.alternative_hypothesis}")
    st.markdown(f"**Test used:** {calibration.test_used}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test statistic (χ²)", calibration.test_statistic)
    with col2:
        st.metric("p-value", f"{calibration.p_value:.2e}")
    with col3:
        st.metric("Sample size", calibration.sample_size)

    if calibration.significant:
        st.warning(
            f"**Statistically significant deviation from perfect calibration** at α = 0.05. "
            f"{calibration.effect_description}"
        )
    else:
        st.success(f"No significant deviation from calibration detected. {calibration.effect_description}")

    st.divider()
    st.subheader("Predicted vs Observed Win Rate by Probability Bucket")
    fig = px.scatter(
        buckets, x="mean_predicted", y="observed_win_rate", size="n_matches",
        labels={"mean_predicted": "Model's predicted win probability", "observed_win_rate": "Real observed win rate"},
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points on the dashed line would indicate perfect calibration. Bubble size reflects how many real matches fell into that bucket.")
