import os
import streamlit as st

from rag import RAGPipeline

st.set_page_config(
    page_title="PDF RAG Chatbot",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")

st.write(
    "Upload a PDF and ask questions from it."
)

# --------------------------------------------------------

if "rag" not in st.session_state:

    st.session_state.rag = RAGPipeline()

# --------------------------------------------------------

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_pdf is not None:

    os.makedirs("data", exist_ok=True)

    pdf_path = os.path.join(
        "data",
        uploaded_pdf.name
    )

    with open(pdf_path, "wb") as file:

        file.write(
            uploaded_pdf.getbuffer()
        )

    st.success("PDF Uploaded Successfully")

    if st.button("Build Knowledge Base"):

        with st.spinner("Building Vector Database..."):

            st.session_state.rag.build_vector_database(
                pdf_path
            )

            st.session_state.rag.load_database()

        st.success("Knowledge Base Created")

# --------------------------------------------------------

st.header("Ask Question")

query = st.text_input(
    "Enter your question"
)

if st.button("Generate Answer"):

    if query.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner("Generating Answer..."):

            answer = st.session_state.rag.ask(
                query
            )

        st.subheader("Answer")

        st.write(answer)