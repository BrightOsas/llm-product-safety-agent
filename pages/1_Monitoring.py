"""
pages/1_Monitoring.py

Monitoring dashboard, built on the interaction/feedback data logged by
app.py to monitoring/monitoring.db. Reachable from the Streamlit app's
sidebar page navigation ("Monitoring").

Charts:
1. Questions per day (volume over time)
2. Feedback breakdown (thumbs up / down / no feedback)
3. Average response time per day
4. Most frequently used tools
5. Response time distribution
6. Feedback rate (% of answers that received any feedback) over time

Run as part of the app: streamlit run app.py, then open "Monitoring" in
the sidebar. Can also be run standalone: streamlit run pages/1_Monitoring.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from monitoring.db import fetch_all_interactions  # noqa: E402

st.set_page_config(page_title="Monitoring — Is My Stuff Safe?", page_icon="📊", layout="wide")
st.title("📊 Monitoring Dashboard")

interactions = fetch_all_interactions()

if not interactions:
    st.info("No interactions logged yet. Ask a few questions in the main app first.")
    st.stop()

df = pd.DataFrame(interactions)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
df["date"] = df["timestamp"].dt.date
df["tool_calls_parsed"] = df["tool_calls"].apply(lambda x: json.loads(x) if x else [])

# ---- top-line metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total questions", len(df))
col2.metric("Avg response time", f"{df['response_time_seconds'].mean():.1f}s")
thumbs_up = (df["feedback"] == 1).sum()
thumbs_down = (df["feedback"] == -1).sum()
col3.metric("👍 Thumbs up", int(thumbs_up))
col4.metric("👎 Thumbs down", int(thumbs_down))

st.divider()

# ---- Chart 1: questions per day ----
st.subheader("1. Questions per day")
daily_counts = df.groupby("date").size().rename("questions")
st.bar_chart(daily_counts)

# ---- Chart 2: feedback breakdown ----
st.subheader("2. Feedback breakdown")
feedback_labels = df["feedback"].map({1: "👍 Thumbs up", -1: "👎 Thumbs down"}).fillna("No feedback")
feedback_counts = feedback_labels.value_counts()
st.bar_chart(feedback_counts)

# ---- Chart 3: average response time per day ----
st.subheader("3. Average response time per day")
avg_response_time = df.groupby("date")["response_time_seconds"].mean().rename("avg_seconds")
st.line_chart(avg_response_time)

# ---- Chart 4: tool usage frequency ----
st.subheader("4. Tool usage frequency")
tool_counter = Counter()
for calls in df["tool_calls_parsed"]:
    for c in calls:
        tool_counter[c["name"]] += 1
if tool_counter:
    tool_df = pd.Series(tool_counter).rename("calls")
    st.bar_chart(tool_df)
else:
    st.caption("No tool calls logged yet.")

# ---- Chart 5: response time distribution ----
st.subheader("5. Response time distribution")
bins = pd.cut(df["response_time_seconds"], bins=10)
hist = bins.value_counts().sort_index()
hist.index = hist.index.astype(str)
st.bar_chart(hist)

# ---- Chart 6: feedback rate over time ----
st.subheader("6. Feedback rate over time (% of questions rated)")
daily_feedback_rate = df.groupby("date").apply(
    lambda g: (g["feedback"].notna().sum() / len(g)) * 100
).rename("feedback_rate_pct")
st.line_chart(daily_feedback_rate)

st.divider()
st.subheader("Recent interactions")
st.dataframe(
    df[["timestamp", "question", "response_time_seconds", "feedback"]]
    .sort_values("timestamp", ascending=False)
    .head(20),
    use_container_width=True,
)
