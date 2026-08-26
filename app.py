import streamlit as st
import sys
sys.path.insert(0, '.')
from rag_helpdesk import index_documents, rag_query, CLAUDE_API_KEY
import anthropic

st.set_page_config(
    page_title="Enterprise IT Helpdesk Agent",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 Enterprise IT Helpdesk Agent")
st.caption("Powered by RAG + ChromaDB + Claude AI | Built by Swapnil Joshi")
st.divider()

# Index documents on startup
with st.spinner("Loading knowledge base..."):
    index_documents()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm your Enterprise IT Helpdesk assistant. How can I help you today?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your IT question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response = rag_query(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})