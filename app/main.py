import streamlit as st

from rag import RAGPipeline


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# ==========================================================
# TITLE
# ==========================================================

st.title("📚 RAG Chatbot")

st.write(
    "Ask questions about the PDFs stored in the data folder."
)

# ==========================================================
# INITIALIZE RAG
# ==========================================================

if "rag_pipeline" not in st.session_state:

    with st.spinner(
        "Initializing RAG Pipeline..."
    ):

        st.session_state.rag_pipeline = RAGPipeline()

        st.session_state.rag_pipeline.load_vector_database()

# ==========================================================
# CHAT HISTORY
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==========================================================
# DISPLAY PREVIOUS MESSAGES
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# ==========================================================
# CHAT INPUT
# ==========================================================

query = st.chat_input(
    "Ask a question about your documents..."
)

# ==========================================================
# PROCESS QUESTION
# ==========================================================

if query:

    # ------------------------------------------------------
    # Display User Question
    # ------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # ------------------------------------------------------
    # Generate Answer
    # ------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            answer = (
                st.session_state
                .rag_pipeline
                .ask(
                    query,
                    chat_history=st.session_state.messages
                )
            )

        st.markdown(answer)


    # ------------------------------------------------------
    # Save Answer
    # ------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("Chat Controls")

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.write(
        "📄 PDFs are loaded from:"
    )

    st.code("data/")