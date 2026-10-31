"""Data quality monitoring dashboard page."""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.data_quality import run_all_quality_checks

st.set_page_config(page_title="Pitchline — Data Quality", layout="wide", page_icon="✅")

st.title("✅ Data Quality Monitoring")
st.caption(
    "The same validation pattern used in a production dbt project or data pipeline: explicit, "
    "automated checks for completeness, validity, uniqueness, business rules, and freshness — "
    "run against the real combined dataset."
)

with st.spinner("Running quality checks against the real dataset..."):
    report = run_all_quality_checks()

passed = int(report["passed"].sum())
total = len(report)

col1, col2 = st.columns(2)
with col1:
    st.metric("Checks Passed", f"{passed} / {total}")
with col2:
    status = "🟢 All checks passing" if passed == total else "🔴 Failures detected"
    st.metric("Overall Status", status)

st.divider()

for category in report["category"].unique():
    st.subheader(category)
    category_checks = report[report["category"] == category]
    for _, row in category_checks.iterrows():
        icon = "✅" if row["passed"] else "❌"
        with st.container(border=True):
            st.markdown(f"{icon} **{row['check_name']}**")
            st.caption(row["detail"])

st.divider()
st.caption(
    "These checks run against the same real, combined 1990-2026 World Cup dataset used "
    "throughout this project — not a synthetic sample."
)
