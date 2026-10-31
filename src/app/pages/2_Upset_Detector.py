"""Upset / anomaly detector dashboard page."""

import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.upset_detector import detect_upsets, summarize_upsets

st.set_page_config(page_title="Pitchline — Upset Detector", layout="wide", page_icon="🚨")

st.title("🚨 Upset & Statement Win Detector")
st.caption(
    "Same anomaly-detection pattern used in fraud and sensor monitoring: define an expected "
    "baseline from the model, then flag real results that deviate significantly from it."
)

with st.spinner("Re-computing ELO baseline across 36 years of real history..."):
    results = detect_upsets(real_results_year=2026)
    summary = summarize_upsets(results)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Matches Analyzed", summary["total_matches"])
with col2:
    st.metric("Total Upsets", summary["upsets_detected"])
with col3:
    st.metric("Underdog Wins", summary["underdog_wins"], help="Winner had under 25% pre-match win probability")
with col4:
    st.metric("Favorites Held to Draw", summary["favorites_held_to_draw"], help="Heavy favorite (75%+) only drew")

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Biggest Underdog Win")
    if summary["biggest_upset"]:
        u = summary["biggest_upset"]
        st.success(
            f"**{u['winner']}** won despite only a **{u['winner_pre_match_prob']:.1%}** "
            f"pre-match win probability — {u['home_team']} {u['home_score']}-{u['away_score']} {u['away_team']}"
        )
    else:
        st.info("No outright underdog wins yet.")

with col_b:
    st.subheader("Biggest Favorite Held to a Draw")
    if summary["biggest_draw_upset"]:
        d = summary["biggest_draw_upset"]
        st.warning(
            f"**{d['favorite_held_to_draw']}** was a heavy favorite but only drew — "
            f"{d['home_team']} {d['home_score']}-{d['away_score']} {d['away_team']}"
        )
    else:
        st.info("No favorites held to a surprise draw yet.")

st.divider()
st.subheader("Pre-Match Win Probability vs Real Outcome")
st.caption("Every real match, plotted by how confident the model was before kickoff. Points far from the model's expectation are flagged.")

results["winner_label"] = results["winner"].fillna("Draw")
fig = px.scatter(
    results, x="date", y="winner_pre_match_prob", color="is_upset",
    hover_data=["home_team", "away_team", "home_score", "away_score"],
    labels={"winner_pre_match_prob": "Winner's pre-match win probability", "date": "Date"},
    color_discrete_map={True: "#D85A30", False: "#378ADD"},
)
fig.add_hline(y=0.25, line_dash="dash", line_color="gray", annotation_text="Upset threshold (25%)")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Full Match Log")
st.dataframe(
    results[["date", "home_team", "away_team", "home_score", "away_score",
             "winner", "winner_pre_match_prob", "is_upset", "is_statement_win"]],
    use_container_width=True,
)
