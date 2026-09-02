import json
import os
import uuid

import streamlit as st
import streamlit.components.v1 as components

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
    "Ask questions about the documents in the data folder, "
    "or upload your own PDFs / CSVs / Excel files below."
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
# COPY-TO-CLIPBOARD HELPER
# ==========================================================


def render_copy_button(text, key):
    """
    Renders a small 'Copy' button under an assistant message.
    Streamlit has no native copy button for markdown text, so
    this injects a tiny HTML/JS snippet that uses the browser's
    clipboard API.
    """

    safe_text = json.dumps(text)

    html = f"""
    <div style="margin-top: -8px;">
        <button id="copy-btn-{key}" onclick="
            navigator.clipboard.writeText({safe_text});
            const btn = document.getElementById('copy-btn-{key}');
            const original = btn.innerText;
            btn.innerText = 'Copied!';
            setTimeout(() => {{ btn.innerText = original; }}, 1500);
        "
        style="
            font-size: 12px;
            padding: 2px 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
            background-color: #f6f6f6;
            cursor: pointer;
        ">
            📋 Copy
        </button>
    </div>
    """

    components.html(html, height=40)


# ==========================================================
# DISPLAY PREVIOUS MESSAGES
# ==========================================================

for i, message in enumerate(st.session_state.messages):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            render_copy_button(
                message["content"],
                key=message.get("id", i)
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

        answer_id = str(uuid.uuid4())

        render_copy_button(answer, key=answer_id)


    # ------------------------------------------------------
    # Save Answer
    # ------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "id": answer_id
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

    # ------------------------------------------------------
    # FILE UPLOAD
    # ------------------------------------------------------

    st.header("📤 Upload Documents")

    st.caption(
        "Upload PDF, CSV, or Excel files. They're added to the "
        "same knowledge base as the existing files in data/ — "
        "answers can draw on both."
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Add to knowledge base"):

        data_folder = st.session_state.rag_pipeline.data_folder

        os.makedirs(data_folder, exist_ok=True)

        saved_names = []

        for uploaded_file in uploaded_files:

            dest_path = os.path.join(
                data_folder,
                uploaded_file.name
            )

            with open(dest_path, "wb") as f:

                f.write(uploaded_file.getbuffer())

            saved_names.append(uploaded_file.name)

        with st.spinner(
            "Rebuilding knowledge base with new file(s)..."
        ):

            # load_vector_database() re-hashes the data folder,
            # notices it changed, and rebuilds the FAISS index
            # automatically (see RAGPipeline.database_is_current).
            st.session_state.rag_pipeline.load_vector_database()

        st.success(
            f"Added: {', '.join(saved_names)}. "
            "You can now ask questions about them."
        )

        st.rerun()

    st.divider()

    st.write(
        "📄 Documents are loaded from:"
    )

    st.code("data/")
