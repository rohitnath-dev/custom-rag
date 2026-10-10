import os
import json
import streamlit as st
from pathlib import Path
from rag import create_rag_app


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

identifier = st.secrets.get(
    "BUSINESS_ID",
    os.getenv("BUSINESS_ID", ""),
)

if not identifier:
    st.error("Application configuration is missing.")
    st.stop()

try:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        configurations = json.load(file)

    if identifier not in configurations:
        raise ValueError("Unknown business configuration.")

    business_config = configurations[identifier]
    ui_config = business_config.get("ui", {})

except (OSError, ValueError, json.JSONDecodeError):
    st.error("Unable to load the application configuration.")
    st.stop()


st.set_page_config(
    page_title=ui_config.get("page_title", "Business Assistant"),
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding: 1.5rem 1.25rem 1rem;
    }

    .welcome-section {
        min-height: 65vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 1rem 0;
        box-sizing: border-box;
    }

    .chat-header {
        width: 100%;
        font-size: clamp(1.5rem, 5vw, 2.2rem);
        font-weight: 650;
        line-height: 1.3;
        margin-bottom: 0.8rem;
        overflow-wrap: anywhere;
    }

    .chat-subtitle {
        max-width: 600px;
        width: 100%;
        font-size: clamp(0.95rem, 3vw, 1.1rem);
        line-height: 1.65;
        color: var(--text-color);
        opacity: 0.75;
        overflow-wrap: anywhere;
    }

    [data-testid="stChatMessage"] {
        padding: 0.75rem;
        border-radius: 12px;
        overflow-wrap: anywhere;
    }

    [data-testid="stChatInput"] {
        max-width: 100%;
    }

    @media (max-width: 600px) {
        .block-container {
            padding: 0.75rem 0.75rem 1rem;
        }

        .welcome-section {
            min-height: 55vh;
            padding: 1rem 0.25rem;
        }

        .chat-header {
            font-size: clamp(1.4rem, 6vw, 1.8rem);
        }

        .chat-subtitle {
            font-size: 0.95rem;
            line-height: 1.6;
        }

        [data-testid="stChatMessage"] {
            padding: 0.55rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <section class="welcome-section">
        <div class="chat-header">
            {ui_config.get("welcome_title", "Business Assistant")}
        </div>
        <div class="chat-subtitle">
            {ui_config.get("subtitle", "How can we help you?")}
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


try:
    rag_app = create_rag_app(identifier)
except Exception as error:
    print(f"RAG INITIALIZATION ERROR: {error!r}", flush=True)
    st.error("The assistant is temporarily unavailable.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input(
    ui_config.get("input_placeholder", "Ask a question...")
)


if user_input and user_input.strip():
    user_input = user_input.strip()

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = rag_app.invoke(
                    {
                        "question": user_input,
                        "docs": [],
                        "context": "",
                        "answer": "",
                    }
                )

                answer = result.get("answer", "").strip()

                if not answer:
                    answer = "No response was generated. Please try again."

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as error:
            print(f"CHATBOT ERROR: {error!r}", flush=True)

            answer = (
                "The assistant could not process your request. "
                "Please try again later."
            )

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )