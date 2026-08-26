"""
app.py

Streamlit chat interface for "Is My Stuff Safe?".

Every answer is logged to monitoring/monitoring.db (question, answer,
tool calls, response time), and each answer has thumbs up/down feedback
buttons whose result is also logged.

Run with:
    streamlit run app.py
"""

import time

import streamlit as st
from dotenv import load_dotenv

from agent import RecallAgent
from ingest import build_index, build_vector_index, load_documents
from monitoring.db import log_feedback, log_interaction
from rag_helper import RAGBase
from tools import RecallTools

load_dotenv()

st.set_page_config(page_title="Is My Stuff Safe?", page_icon="🏠")


@st.cache_resource(show_spinner="Loading recall knowledge base...")
def load_agent() -> RecallAgent:
    docs = load_documents()
    index = build_index(docs)
    vector_index = build_vector_index(docs)
    rag = RAGBase(index=index, vector_index=vector_index, retrieval_mode="hybrid", use_reranking=True)
    tools = RecallTools(rag=rag, documents=docs)
    return RecallAgent(recall_tools=tools)


st.title("🏠 Is My Stuff Safe?")
st.caption(
    "Describe a product you own, ask about a category, or compare two brands. "
    "Answers are grounded in official CPSC recall data."
)

with st.expander("Example questions"):
    st.markdown(
        "- *I have a Cosco Rock 'N Roller baby stroller from around 2005, is it safe?*\n"
        "- *Is my COSORI air fryer model CP158-AF safe to use?*\n"
        "- *I have a COSORI air fryer model CS158-AF, was it recalled?*\n"
        "- *Have there been any recalls for space heaters?*\n"
        "- *What's the hazard associated with recalled pressure cookers?*\n"
        "- *Compare recall history between Graco and Chicco for car seats.*\n"
        "- *What recalls have there been for cribs?*\n"
        "- *Is my Instant Pot pressure cooker safe to still use?*\n"
        "- *What's the hazard associated with recalled baby swings?*\n"
        "- *I'm buying a used stroller secondhand, how do I check if it was recalled?*"
    )

agent = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {role, content, interaction_id?, feedback_given?}

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "interaction_id" in msg:
            col1, col2, _ = st.columns([1, 1, 8])
            already_voted = msg.get("feedback_given") is not None
            with col1:
                if st.button("👍", key=f"up_{i}", disabled=already_voted):
                    log_feedback(msg["interaction_id"], 1)
                    st.session_state.messages[i]["feedback_given"] = 1
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{i}", disabled=already_voted):
                    log_feedback(msg["interaction_id"], -1)
                    st.session_state.messages[i]["feedback_given"] = -1
                    st.rerun()
            if already_voted:
                st.caption("Thanks for the feedback!" if msg["feedback_given"] == 1 else "Thanks — noted.")

if question := st.chat_input("Ask about a product, category, or brand..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking recall records..."):
            start = time.time()
            result = agent.ask(question)
            elapsed = time.time() - start

        st.markdown(result["answer"])
        if result["tool_calls"]:
            with st.expander("Tools used"):
                for tc in result["tool_calls"]:
                    st.code(f"{tc['name']}({tc['arguments']})", language="python")

        interaction_id = log_interaction(
            question=question,
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            response_time_seconds=elapsed,
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "interaction_id": interaction_id,
    })
    st.rerun()