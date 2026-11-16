import streamlit as st
from src.config import TOP_K_RESULTS, DATA_DIR, EXCLUDED_DIRS, CHROMA_DB_DIR
from src.rag_chain import ask_question

# 1. Configure the Streamlit web page
st.set_page_config(page_title="Document RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 Technical Document RAG Assistant")
st.caption(f"Knowledge Base: `{DATA_DIR.name}` | Grounded answers with source & page citations")

if not CHROMA_DB_DIR.exists():
    st.warning("Vector database not found. Run `python ingest.py` first to index your documents.")
    st.stop()

# 2. Sidebar for live RAG hyperparameter tuning & category filtering
with st.sidebar:
    st.header("⚙️ Retrieval Settings")

    categories = ["All"]
    if DATA_DIR.exists():
        categories += sorted(
            d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in EXCLUDED_DIRS
        )

    selected_category = st.selectbox("Filter by Category / Section", options=categories)
    top_k = st.slider("Number of chunks to retrieve (Top-K)", min_value=1, max_value=15, value=TOP_K_RESULTS)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


def render_sources(sources):
    """Render the expandable source citations dropdown for an assistant message."""
    if not sources:
        return
    with st.expander(f"📚 View Retrieved Sources ({len(sources)} full pages)"):
        for i, doc in enumerate(sources, start=1):
            file_name = doc.metadata.get("file_name", "Unknown")
            category = doc.metadata.get("category", "General")
            page_num = doc.metadata.get("page_number", "?")

            st.markdown(f"**Source {i}: `{file_name}`** (Category: *{category}*, Page: **{page_num}**)")
            st.caption(doc.page_content)
            st.divider()


# 3. Initialize and display chat history (including saved sources!)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"])

# 4. Handle new user question
if user_query := st.chat_input("Ask a question..."):
    # Display the user's question and save to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve sources and stream the LLM response live
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = ask_question(
                user_query,
                k=top_k,
                category=selected_category,
                chat_history=st.session_state.messages[:-1],
            )
            sources = result["sources"]

        # Stream tokens live onto the screen as the LLM generates them!
        answer = st.write_stream(result["stream"])
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})