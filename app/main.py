import streamlit as st
from rag import RAGPipeline


# -------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG PDF Chatbot")

st.markdown("""
Ask questions from the PDFs available inside the **data/** folder.
""")

# -------------------------------------------------------
# Load Pipeline
# -------------------------------------------------------

if "rag_pipeline" not in st.session_state:

    with st.spinner("Initializing RAG Pipeline..."):

        rag = RAGPipeline()

        rag.load_vector_database()

        st.session_state.rag_pipeline = rag

    st.success("Knowledge Base Ready ✅")

# -------------------------------------------------------
# User Question
# -------------------------------------------------------

query = st.text_input(
    "Ask your question"
)

# -------------------------------------------------------
# Generate Answer
# -------------------------------------------------------

if st.button("Generate Answer"):

    if query.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching Documents..."):

            answer = st.session_state.rag_pipeline.ask(query)

        st.subheader("Answer")

        st.write(answer)
