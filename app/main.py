import json
import os
import uuid

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from rag import RAGPipeline
from utils import load_pdf


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# GLOBAL STYLE
# ==========================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif;
        }

        :root {
            --brand-primary: #4F46E5;
            --brand-primary-dark: #3730A3;
            --brand-accent: #06B6D4;
            --brand-bg: #F5F6FB;
            --brand-surface: #FFFFFF;
            --brand-border: #E5E7EB;
            --brand-text: #1F2333;
            --brand-muted: #6B7280;
        }

        .stApp {
            background: linear-gradient(180deg, #F5F6FB 0%, #EEF0FA 100%);
        }

        /* ---- Force readable text everywhere in the main content
               area, regardless of the user's local Streamlit theme
               (light/dark). Without this, a "Dark" theme setting can
               make text render white-on-white against our custom
               light backgrounds -- e.g. text typed into the chat
               input becoming invisible. Sidebar is re-overridden
               below (its own dark background needs light text). ---- */
        .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stMarkdown, .stMarkdown p {
            color: var(--brand-text) !important;
        }

        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInputTextArea"],
        textarea {
            color: var(--brand-text) !important;
            background-color: #FFFFFF !important;
            caret-color: var(--brand-text) !important;
            -webkit-text-fill-color: var(--brand-text) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--brand-muted) !important;
            opacity: 1 !important;
            -webkit-text-fill-color: var(--brand-muted) !important;
        }
        [data-testid="stChatInput"] {
            background-color: #FFFFFF !important;
            border: 1px solid var(--brand-border) !important;
        }

        /* ---- Hide default Streamlit chrome ---- */
        #MainMenu, footer, header {visibility: hidden;}

        /* ---- Hero header ---- */
        .app-hero {
            background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-accent) 100%);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
        }
        .app-hero h1 {
            color: white;
            font-size: 30px;
            font-weight: 800;
            margin: 0 0 6px 0;
            letter-spacing: -0.5px;
        }
        .app-hero p {
            color: rgba(255,255,255,0.9);
            font-size: 15px;
            margin: 0;
            font-weight: 400;
        }
        .app-hero .badge {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            color: white;
            font-size: 12px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 999px;
            margin-bottom: 10px;
            letter-spacing: 0.3px;
        }

        /* ---- Sidebar (kept minimal) ---- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%);
        }
        [data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.15);
        }

        /* ---- Buttons ---- */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            border: none;
            background: var(--brand-primary);
            color: white;
            transition: all 0.15s ease-in-out;
        }
        .stButton>button:hover {
            background: var(--brand-primary-dark);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
        }
        [data-testid="stSidebar"] .stButton>button {
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.25);
        }
        [data-testid="stSidebar"] .stButton>button:hover {
            background: rgba(255,255,255,0.22);
        }

        /* ---- File chip buttons inside chat (small, pill-shaped) ---- */
        .file-chip-row .stButton>button {
            background: #EEF2FF;
            color: var(--brand-primary-dark) !important;
            border: 1px solid #C7D2FE;
            border-radius: 999px;
            font-size: 12.5px;
            padding: 2px 12px;
            font-weight: 600;
        }
        .file-chip-row .stButton>button:hover {
            background: #E0E7FF;
            box-shadow: none;
            transform: none;
        }

        /* ---- Chat bubbles ---- */
        [data-testid="stChatMessage"] {
            background: var(--brand-surface);
            border: 1px solid var(--brand-border);
            border-radius: 14px;
            padding: 6px 4px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
        }

        /* ---- Chat input ---- */
        [data-testid="stChatInput"] {
            border-radius: 14px;
        }

        /* ---- Tabs ---- */
        button[data-baseweb="tab"] {
            font-weight: 600;
            font-size: 15px;
        }

        /* ---- Expanders / cards ---- */
        [data-testid="stExpander"] {
            background: var(--brand-surface);
            border: 1px solid var(--brand-border);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
        }

        /* ---- Dataframes ---- */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--brand-border);
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# HERO / TITLE
# ==========================================================

st.markdown(
    """
    <div class="app-hero">
        <div class="badge">📚 RAG · AI DOCUMENT ASSISTANT</div>
        <h1>DocuMind AI</h1>
        <p>Ask questions about your PDFs, spreadsheets and CSVs —
        every sheet and row is searched, not just the highlights.</p>
    </div>
    """,
    unsafe_allow_html=True
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
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# Tracks which file-preview chips are currently expanded,
# keyed by a unique id per (message, file).
if "open_previews" not in st.session_state:

    st.session_state.open_previews = set()

# ==========================================================
# COPY-TO-CLIPBOARD HELPER
# ==========================================================


def render_copy_button(text, key):
    """
    Renders a small 'Copy' button under an assistant message.
    Streamlit has no native copy button for markdown text, so
    this injects a tiny HTML/JS snippet that uses the browser's
    clipboard API.

    IMPORTANT: the click handler is attached via addEventListener
    in a <script> block rather than an inline onclick="..."
    attribute. Answer text can contain double quotes, and an
    inline onclick attribute is itself delimited by double
    quotes -- json.dumps(text) would then prematurely close the
    attribute and corrupt the HTML, which is why the button
    previously failed to render (it showed up as raw leftover
    text instead of a button).
    """

    safe_key = "".join(
        ch if ch.isalnum() else "_"
        for ch in str(key)
    )

    safe_text = json.dumps(text)

    html = f"""
    <div style="margin-top: -8px;">
        <button id="copy-btn-{safe_key}" style="
            font-size: 12px;
            padding: 2px 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
            background-color: #f6f6f6;
            cursor: pointer;
        ">📋 Copy</button>
    </div>
    <script>
        (function () {{
            var text = {safe_text};
            var btn = document.getElementById("copy-btn-{safe_key}");
            if (btn) {{
                btn.addEventListener("click", function () {{
                    navigator.clipboard.writeText(text);
                    var original = btn.innerText;
                    btn.innerText = "Copied!";
                    setTimeout(function () {{
                        btn.innerText = original;
                    }}, 1500);
                }});
            }}
        }})();
    </script>
    """

    components.html(html, height=40)


# ==========================================================
# TABLE PREVIEW HELPER
#
# Reads a CSV/XLSX/XLS file and returns {sheet_label: dataframe}.
# A CSV always yields a single "Sheet1" entry; an Excel workbook
# yields one entry per sheet, so multi-sheet files are previewed
# (and searched, see rag.py::load_tables) in full, not just the
# first sheet.
# ==========================================================


def read_table_sheets(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    sheets = {}

    if extension == ".csv":

        sheets["Sheet1"] = pd.read_csv(file_path)

    elif extension in (".xlsx", ".xls"):

        excel_file = pd.ExcelFile(file_path)

        for sheet_name in excel_file.sheet_names:

            sheets[sheet_name] = pd.read_excel(
                file_path,
                sheet_name=sheet_name
            )

    return sheets


def render_file_preview(file_path):
    """
    Renders the actual content of a file: a table for CSV/Excel
    (every sheet, in its own tab), or an extracted text preview
    for PDFs.
    """

    extension = os.path.splitext(file_path)[1].lower()

    try:

        if extension == ".pdf":

            text = load_pdf(file_path)

            st.caption(f"{len(text):,} characters extracted")

            st.text_area(
                "Extracted text",
                text[:5000] + ("..." if len(text) > 5000 else ""),
                height=260,
                label_visibility="collapsed"
            )

        else:

            sheets = read_table_sheets(file_path)

            if len(sheets) == 1:

                df = next(iter(sheets.values()))

                st.caption(
                    f"{len(df)} rows × {len(df.columns)} columns"
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    height=300
                )

            else:

                sheet_tabs = st.tabs(list(sheets.keys()))

                for sheet_tab, (sheet_name, df) in zip(
                    sheet_tabs,
                    sheets.items()
                ):

                    with sheet_tab:

                        st.caption(
                            f"{len(df)} rows × "
                            f"{len(df.columns)} columns"
                        )

                        st.dataframe(
                            df,
                            use_container_width=True,
                            height=300
                        )

    except Exception as e:

        st.error(f"Could not open this file: {e}")


def render_file_chips(file_paths, chip_id_prefix):
    """
    Renders each file as a small clickable chip. Clicking a chip
    toggles an inline preview of that file's actual content open
    or closed, right underneath the chips.
    """

    if not file_paths:
        return

    st.markdown('<div class="file-chip-row">', unsafe_allow_html=True)

    cols = st.columns(len(file_paths))

    for col, file_path in zip(cols, file_paths):

        file_name = os.path.basename(file_path)

        extension = os.path.splitext(file_path)[1].lower()

        icon = "📄" if extension == ".pdf" else "📊"

        preview_key = f"{chip_id_prefix}::{file_path}"

        with col:

            if st.button(
                f"{icon} {file_name}",
                key=f"chip_{preview_key}",
                help="Click to open this file"
            ):

                if preview_key in st.session_state.open_previews:
                    st.session_state.open_previews.discard(preview_key)
                else:
                    st.session_state.open_previews.add(preview_key)

    st.markdown('</div>', unsafe_allow_html=True)

    for file_path in file_paths:

        preview_key = f"{chip_id_prefix}::{file_path}"

        if preview_key in st.session_state.open_previews:

            with st.container(border=True):

                render_file_preview(file_path)


def save_files_to_knowledge_base(uploaded_files):
    """
    Saves attached files into data/ and rebuilds the knowledge
    base (vector index + direct table search) so they're
    immediately searchable. Returns the list of saved file paths.
    """

    data_folder = st.session_state.rag_pipeline.data_folder

    os.makedirs(data_folder, exist_ok=True)

    saved_paths = []

    for uploaded_file in uploaded_files:

        dest_path = os.path.join(
            data_folder,
            uploaded_file.name
        )

        with open(dest_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        saved_paths.append(dest_path)

    with st.spinner(
        "Adding file(s) to the knowledge base..."
    ):

        # load_vector_database() re-hashes the data folder,
        # notices it changed, and rebuilds the FAISS index
        # automatically (see RAGPipeline.database_is_current).
        st.session_state.rag_pipeline.load_vector_database()

    return saved_paths


# ==========================================================
# CHAT-CORNER FILE ATTACH
#
# st.chat_input's built-in accept_file param (paperclip icon
# inside the input box) only exists on newer Streamlit versions.
# On older versions it raises TypeError, so we try it and fall
# back to a small "📎" popover (or expander, on even older
# versions without st.popover) placed right next to the chat
# input -- still a corner attach control, just not built in.
# ==========================================================


def render_attach_fallback():
    """
    Renders a small '📎' control next to the chat input that opens
    a file uploader, for Streamlit versions without native
    accept_file support in st.chat_input. Returns newly-selected
    UploadedFile objects (already de-duplicated against files
    processed earlier in this session).
    """

    uploader_kwargs = dict(
        label="Attach files",
        type=["pdf", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="fallback_attach_uploader"
    )

    try:

        with st.popover("📎", use_container_width=False):

            picked = st.file_uploader(**uploader_kwargs)

    except Exception:

        with st.expander("📎 Attach a file", expanded=False):

            picked = st.file_uploader(**uploader_kwargs)

    if not picked:
        return []

    if "processed_attachment_ids" not in st.session_state:
        st.session_state.processed_attachment_ids = set()

    new_files = [
        f for f in picked
        if f.file_id not in st.session_state.processed_attachment_ids
    ]

    for f in new_files:
        st.session_state.processed_attachment_ids.add(f.file_id)

    return new_files


# ==========================================================
# MAIN TABS
# ==========================================================

chat_tab, data_tab = st.tabs(
    ["💬 Chat", "📊 Data Preview"]
)

# ==========================================================
# CHAT TAB
# ==========================================================

with chat_tab:

    # ------------------------------------------------------
    # DISPLAY PREVIOUS MESSAGES
    # ------------------------------------------------------

    for i, message in enumerate(st.session_state.messages):

        with st.chat_message(
            message["role"]
        ):

            if message["content"]:

                st.markdown(
                    message["content"]
                )

            render_file_chips(
                message.get("files", []),
                chip_id_prefix=f"msg{i}"
            )

            if message["role"] == "assistant" and message["content"]:

                render_copy_button(
                    message["content"],
                    key=message.get("id", i)
                )

    # ------------------------------------------------------
    # CHAT INPUT (with attach-file option in the corner)
    #
    # We try Streamlit's native accept_file param first, which
    # puts a paperclip button built directly into the chat input
    # box (Streamlit >= ~1.41). If this Streamlit install is older
    # and doesn't support that param, it raises TypeError -- we
    # catch that once and fall back to a small "📎" control next
    # to the input instead. Either way, attaching happens from the
    # same corner the user types in.
    # ------------------------------------------------------

    query_text = ""
    attached_files = []

    if "chat_attach_supported" not in st.session_state:
        st.session_state.chat_attach_supported = True

    if st.session_state.chat_attach_supported:

        try:

            chat_value = st.chat_input(
                "Ask a question, or attach a PDF / CSV / Excel file...",
                accept_file="multiple",
                file_type=["pdf", "csv", "xlsx", "xls"]
            )

        except TypeError:

            st.session_state.chat_attach_supported = False

            chat_value = None

        else:

            if chat_value:

                query_text = (chat_value.text or "").strip()

                attached_files = chat_value.files or []

    if not st.session_state.chat_attach_supported:

        input_col, attach_col = st.columns([0.93, 0.07])

        with attach_col:

            attached_files = render_attach_fallback()

        with input_col:

            typed = st.chat_input(
                "Ask a question, or attach a file via the 📎 button..."
            )

            if typed:

                query_text = typed.strip()

    # ------------------------------------------------------
    # PROCESS INPUT
    # ------------------------------------------------------

    if query_text or attached_files:

        saved_paths = []

        if attached_files:

            saved_paths = save_files_to_knowledge_base(
                attached_files
            )

        # ---- Display the user's turn ----

        with st.chat_message("user"):

            if query_text:
                st.markdown(query_text)

            render_file_chips(
                saved_paths,
                chip_id_prefix=f"live_user_{uuid.uuid4().hex[:8]}"
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query_text,
                "files": saved_paths
            }
        )

        # ---- Generate / confirm ----

        with st.chat_message("assistant"):

            if query_text:

                with st.spinner(
                    "Searching documents and generating answer..."
                ):

                    answer = (
                        st.session_state
                        .rag_pipeline
                        .ask(
                            query_text,
                            chat_history=st.session_state.messages
                        )
                    )

            elif saved_paths:

                # Files attached with no question -- just confirm
                # ingestion instead of spending an LLM call.
                file_names = ", ".join(
                    os.path.basename(p) for p in saved_paths
                )

                answer = (
                    f"Added **{file_names}** to the knowledge base. "
                    "Ask me anything about it whenever you're ready."
                )

            else:

                answer = "Please enter a question or attach a file."

            st.markdown(answer)

            answer_id = str(uuid.uuid4())

            render_copy_button(answer, key=answer_id)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "id": answer_id,
                "files": []
            }
        )

# ==========================================================
# DATA PREVIEW TAB
#
# Shows every CSV / Excel file currently in data/ as an actual
# table, so uploads are visibly confirmed -- not just listed by
# filename. Every sheet of a workbook gets its own tab.
# ==========================================================

with data_tab:

    st.subheader("📊 Spreadsheet & CSV data")

    st.caption(
        "Every file below is fully indexed for search — including "
        "every sheet in multi-sheet Excel workbooks, not just the "
        "first one."
    )

    all_source_files = (
        st.session_state
        .rag_pipeline
        .find_source_files()
    )

    table_files = [
        f for f in all_source_files
        if os.path.splitext(f)[1].lower() in (".csv", ".xlsx", ".xls")
    ]

    if not table_files:

        st.info(
            "No spreadsheet data yet. Attach a CSV or Excel file "
            "from the chat box below to see it previewed here."
        )

    else:

        for file_path in table_files:

            file_name = os.path.basename(file_path)

            try:

                sheets = read_table_sheets(file_path)

            except Exception as e:

                st.error(f"Could not read {file_name}: {e}")

                continue

            with st.expander(
                f"📄 {file_name}  ·  {len(sheets)} sheet(s)",
                expanded=False
            ):

                if len(sheets) == 1:

                    df = next(iter(sheets.values()))

                    st.caption(
                        f"{len(df)} rows × {len(df.columns)} columns"
                    )

                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=320
                    )

                else:

                    sheet_tabs = st.tabs(list(sheets.keys()))

                    for sheet_tab, (sheet_name, df) in zip(
                        sheet_tabs,
                        sheets.items()
                    ):

                        with sheet_tab:

                            st.caption(
                                f"{len(df)} rows × "
                                f"{len(df.columns)} columns"
                            )

                            st.dataframe(
                                df,
                                use_container_width=True,
                                height=320
                            )


# ==========================================================
# SIDEBAR -- kept intentionally minimal
# ==========================================================

with st.sidebar:

    st.markdown("### 🧭 DocuMind AI")

    st.caption("AI-powered document & spreadsheet assistant")

    st.divider()

    st.write("📄 Documents are loaded from:")

    st.code("data/")

    loaded_files = (
        st.session_state
        .rag_pipeline
        .find_source_files()
    )

    st.caption(f"{len(loaded_files)} file(s) currently indexed.")