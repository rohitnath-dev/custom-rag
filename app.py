import os
import time
import streamlit as st
from rag import create_rag_app

st.set_page_config(
    page_title="Assistant",
    page_icon="◈",
    layout="centered"
)

identifier = st.secrets.get("BUSINESS_ID", os.getenv("BUSINESS_ID"))

if not identifier:
    st.error("Application configuration is missing.")
    st.stop()

rag_app = create_rag_app(identifier)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type a message...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            result = rag_app.invoke({
                "question": user_input,
                "docs": [],
                "relevant_docs": [],
                "context": "",
                "answer": "",
            })

            answer = result["answer"]
            displayed_text = ""

            for character in answer:
                displayed_text += character
                placeholder.markdown(displayed_text)
                time.sleep(0.01)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception:
            answer = "Unable to process the request."
            placeholder.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })