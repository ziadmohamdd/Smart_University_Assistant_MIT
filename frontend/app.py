"""SmartUniversityAssistant - Streamlit frontend.

A simple chat-style UI that sends questions to the FastAPI backend (via
api_client.py) and displays grounded answers with their cited sources.

This file contains ONLY UI logic - all HTTP communication with the backend
goes through api_client.py, which reads the backend URL from frontend/.env.
"""

import streamlit as st

import api_client

st.set_page_config(
    page_title="SmartUniversityAssistant",
    page_icon="🎓",
    layout="centered",
)


# ---------------------------------------------------------------------
# Sidebar: backend status
# ---------------------------------------------------------------------
with st.sidebar:
    st.subheader("Backend status")

    try:
        health = api_client.check_health()
        status = health.get("status", "unknown")

        if status == "healthy":
            st.success("Backend online")
        else:
            st.warning(f"Backend status: {status}")

        retrieval_ready = health.get("retrieval_ready")
        generation_ready = health.get("generation_ready")

        if retrieval_ready is not None:
            st.caption(("✅" if retrieval_ready else "❌") + " Document retrieval")
        if generation_ready is not None:
            st.caption(("✅" if generation_ready else "❌") + " Answer generation")

    except api_client.APIConfigError:
        st.error("Configuration error. Check frontend/.env (API_BASE_URL).")
    except api_client.APIClientError:
        st.error("Backend is unreachable.")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🎓 SmartUniversityAssistant")
st.caption(
    "Ask a question and get an answer grounded in the university's "
    "document collection, with sources cited below each response."
)


# ---------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}


def render_message(role: str, content: str, sources: list[str] | None = None) -> None:
    """Render a single chat bubble, with sources shown under an assistant answer."""
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- {source}")


# Replay the conversation so far
for message in st.session_state.messages:
    render_message(
        message["role"],
        message["content"],
        message.get("sources"),
    )


# ---------------------------------------------------------------------
# Friendly error messages
# ---------------------------------------------------------------------
def friendly_error_message(exc: Exception) -> str:
    """Map api_client exceptions to a short, non-technical message for the user."""
    if isinstance(exc, api_client.APIConfigError):
        return (
            "⚙️ The app isn't configured correctly (missing or invalid backend URL). "
            "Please check frontend/.env and restart the app."
        )
    if isinstance(exc, api_client.APIConnectionError):
        return (
            "🔌 I can't reach the backend right now. Please make sure the "
            "server is running and try again."
        )
    if isinstance(exc, api_client.APITimeoutError):
        return "⏱️ That took too long to answer. Please try again in a moment."
    if isinstance(exc, api_client.APIHTTPError):
        if exc.status_code == 503:
            return (
                "🚧 The assistant is temporarily unavailable (a required "
                "service isn't ready yet). Please try again shortly."
            )
        if exc.status_code == 422:
            return "❓ That question wasn't valid. Please rephrase it and try again."
        return "⚠️ The backend ran into a problem answering that question."
    if isinstance(exc, api_client.APIInvalidResponseError):
        return "⚠️ The backend sent back something unexpected. Please try again."
    if isinstance(exc, api_client.APIClientError):
        return "⚠️ Something went wrong while contacting the backend."
    return "⚠️ An unexpected error occurred."


# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------
question = st.chat_input("Ask a question about the university...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message("user", question)

    with st.chat_message("assistant"):
        with st.spinner("Looking that up..."):
            try:
                result = api_client.ask_question(question)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except (ValueError, api_client.APIClientError) as exc:
                error_message = friendly_error_message(exc)
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
