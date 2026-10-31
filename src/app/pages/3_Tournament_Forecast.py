"""Monte Carlo tournament forecast simulation dashboard page."""

import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.monte_carlo_simulation import simulate_group_outcomes

st.set_page_config(page_title="Pitchline — Tournament Forecast", layout="wide", page_icon="🎲")

st.title("🎲 Monte Carlo Tournament Forecast")
st.caption(
    "The same forward-simulation method used in demand and price-elasticity forecasting: "
    "run the real remaining fixtures forward thousands of times with randomized outcomes "
    "drawn from the model's real probability distribution, then report the spread of outcomes."
)

n_sims = st.slider("Number of simulations", min_value=500, max_value=20_000, value=5_000, step=500)

with st.spinner(f"Running {n_sims:,} simulations of the real remaining group fixtures..."):
    results = simulate_group_outcomes(n_simulations=n_sims, seed=42)

st.caption(
    f"Simulated {n_sims:,} times using only real, currently-scheduled group-stage fixtures — "
    "no invented matchups."
)

st.divider()

groups = sorted(results["group"].unique())
selected_group = st.selectbox("Select a group", groups)

group_data = results[results["group"] == selected_group].sort_values("pct_advance", ascending=False)

fig = px.bar(
    group_data, x="team", y=["pct_finish_1st", "pct_finish_2nd"],
    barmode="stack",
    labels={"value": "Probability", "team": "Team", "variable": "Finish"},
    title=f"{selected_group} — Probability of Advancing",
)
fig.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    group_data[["team", "pct_finish_1st", "pct_finish_2nd", "pct_advance", "pct_eliminated"]],
    use_container_width=True,
    column_config={
        "pct_finish_1st": st.column_config.ProgressColumn("1st Place %", min_value=0, max_value=1, format="%.1f%%"),
        "pct_finish_2nd": st.column_config.ProgressColumn("2nd Place %", min_value=0, max_value=1, format="%.1f%%"),
        "pct_advance": st.column_config.ProgressColumn("Advance %", min_value=0, max_value=1, format="%.1f%%"),
        "pct_eliminated": st.column_config.ProgressColumn("Eliminated %", min_value=0, max_value=1, format="%.1f%%"),
    },
)

st.divider()
st.subheader("All Groups — Advancement Probability")
fig2 = px.bar(
    results.sort_values(["group", "pct_advance"], ascending=[True, False]),
    x="team", y="pct_advance", color="group",
    labels={"pct_advance": "Probability of Advancing"},
)
fig2.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-45, height=450)
st.plotly_chart(fig2, use_container_width=True)
