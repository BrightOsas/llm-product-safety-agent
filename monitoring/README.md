# Monitoring

Every question asked in the app is logged to `monitoring.db` (SQLite):
question, answer, tool calls used, response time, and timestamp. Each
answer has 👍/👎 feedback buttons in the app, logged against that
interaction.

[Streamlit App Monitoring](https://llm-appuct-safety-agent-6h8zxlpdke5yxc287qi4dv.streamlit.app/Monitoring)

## Dashboard

The **"Monitoring"** page (`pages/1_Monitoring.py`, in the Streamlit
sidebar) shows:

1. Questions per day
2. Feedback breakdown (👍 / 👎 / no feedback)
3. Average response time per day
4. Tool usage frequency
5. Response time distribution
6. Feedback rate over time

## Data location

`monitoring/monitoring.db` — created automatically on first run, not
committed to version control (local/session data).
