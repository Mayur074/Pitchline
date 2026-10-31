"""
"What happened in this match without watching it" — an at-a-glance summary
page built entirely from real goal-event data (scorer, minute, penalty/own
goal flags) pulled live from the openfootball/worldcup.json feed.

This deliberately does NOT show a pitch heatmap of player positions, since
we don't have that data and faking it would contradict everything else this
project stands for. What it shows instead is real: the momentum timeline
(running scoreline across the match) and the goal-by-goal event log.

Run with: streamlit run src/app/match_summary_page.py
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.match_summary import list_available_completed_matches, fetch_match_summary

st.set_page_config(page_title="Pitchline — Match at a Glance", layout="wide", page_icon="⚽")

st.title("⚽ Match at a Glance")
st.caption(
    "Catch up on any completed World Cup 2026 match in seconds — built from real goal "
    "scorer and minute data, not a simulation."
)

with st.spinner("Loading completed matches..."):
    available = list_available_completed_matches()

if available.empty:
    st.warning("No completed matches found in the live feed yet.")
    st.stop()

available["label"] = (
    available["date"] + "  —  " +
    available["home_team"] + " " + available["home_score"].astype(str) +
    "-" + available["away_score"].astype(str) + " " + available["away_team"] +
    "  (" + available["stage"] + ")"
)

selected_label = st.selectbox(
    "Pick a match",
    options=available["label"].tolist()[::-1],  # most recent first
)

selected_row = available[available["label"] == selected_label].iloc[0]
summary = fetch_match_summary(selected_row["home_team"], selected_row["away_team"], date=selected_row["date"])

if summary is None:
    st.error("Couldn't load details for this match.")
    st.stop()

# --- Header scoreline ---
col1, col2, col3 = st.columns([3, 1, 3])
with col1:
    st.markdown(f"## {summary.home_team}")
with col2:
    st.markdown(f"## {summary.home_score} — {summary.away_score}")
with col3:
    st.markdown(f"## {summary.away_team}")

st.caption(f"{summary.stage}  •  {summary.ground}  •  {summary.date}")
if summary.half_time_score:
    st.caption(f"Half-time: {summary.half_time_score[0]} — {summary.half_time_score[1]}")

st.divider()

# --- Momentum timeline chart ---
st.subheader("Match Momentum")
st.caption("The running scoreline across the match — built from real goal events, minute by minute.")

minutes = [p["minute"] for p in summary.momentum_timeline]
home_scores = [p["home_score"] for p in summary.momentum_timeline]
away_scores = [p["away_score"] for p in summary.momentum_timeline]

# extend the line to minute 90 (or later if there were stoppage-time goals)
final_minute = max(90.0, max(minutes) if minutes else 90.0)
minutes.append(final_minute)
home_scores.append(home_scores[-1] if home_scores else 0)
away_scores.append(away_scores[-1] if away_scores else 0)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=minutes, y=home_scores, mode="lines+markers", name=summary.home_team,
    line=dict(shape="hv", width=3), marker=dict(size=8),
))
fig.add_trace(go.Scatter(
    x=minutes, y=away_scores, mode="lines+markers", name=summary.away_team,
    line=dict(shape="hv", width=3), marker=dict(size=8),
))
fig.update_layout(
    xaxis_title="Minute", yaxis_title="Goals",
    height=350, hovermode="x unified",
    yaxis=dict(dtick=1),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Goal-by-goal event log ---
st.subheader("Goal Timeline")

if not summary.goal_events:
    st.info("No goals in this match — a scoreless draw.")
else:
    for event in summary.goal_events:
        tags = []
        if event.is_penalty:
            tags.append("🎯 Penalty")
        if event.is_own_goal:
            tags.append("⚠️ Own Goal")
        tag_str = "  ·  ".join(tags)

        side = "🏠" if event.team == summary.home_team else "✈️"
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 2])
            with c1:
                st.markdown(f"**{event.minute_display}'**")
            with c2:
                st.markdown(f"{side} **{event.scorer}** ({event.team})")
            with c3:
                if tag_str:
                    st.caption(tag_str)

st.divider()
st.caption(
    "Data source: openfootball/worldcup.json (public domain, no API key required). "
    "All scorers, minutes, and scorelines shown are real."
)
