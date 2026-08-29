import streamlit as st
import os

# ── Google OAuth SSO ────────────────────────────────────────────
if not st.experimental_user.is_logged_in:
    st.title("🔐 Enterprise IT Helpdesk — Secure Login")
    st.write("Please sign in with your Google account to access the helpdesk.")
    st.button("Sign in with Google", on_click=st.login)
    st.stop()

# Show logged in user
user_email = st.experimental_user.email
user_name = st.experimental_user.name

# ── Sidebar — user info + logout ────────────────────────────────
with st.sidebar:
    st.write(f"👤 {user_name}")
    st.write(f"📧 {user_email}")
    st.button("Logout", on_click=st.logout)

# Load secrets in Streamlit context first
if "CLAUDE_API_KEY" in st.secrets:
    os.environ["CLAUDE_API_KEY"] = st.secrets["CLAUDE_API_KEY"]
if "AIRTABLE_TOKEN" in st.secrets:
    os.environ["AIRTABLE_TOKEN"] = st.secrets["AIRTABLE_TOKEN"]

# ── Now import agents ───────────────────────────────────────────
from v3_agent import create_context, router_agent, knowledge_agent, ticketing_agent

# Index documents on startup if collection is empty
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="it_helpdesk_kb",
    embedding_function=embedding_fn
)

if collection.count() == 0:
    with open("EnterpriseITHelpdesk_KnowledgeBase_SSJoshi.txt", "r") as f:
        content = f.read()
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

# Now import agents — they'll find keys in os.environ
from v3_agent import create_context, router_agent, knowledge_agent, ticketing_agent

st.set_page_config(
    page_title="Enterprise IT Helpdesk V3",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise IT Helpdesk — Multi-Agent V3")
st.caption(f"Powered by Multi-Agent RAG + ChromaDB + Claude AI + Airtable | Built by Swapnil Joshi | Logged in as: {user_name} ({user_email})")
st.divider()

# Agent status indicators
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔀 Router Agent", "Active")
with col2:
    st.metric("📚 Knowledge Agent", "Active")
with col3:
    st.metric("🎫 Ticketing Agent", "Active")

st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm your Multi-Agent IT Helpdesk assistant. I can answer IT questions or log support tickets automatically. How can I help you today?"
    })

if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False
if "pending_issue" not in st.session_state:
    st.session_state.pending_issue = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your IT question or describe your issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # Check if awaiting confirmation
        if st.session_state.awaiting_confirmation:
            confirmation_words = ["yes", "y", "yes please", "confirm", "go ahead", "proceed", 
                      "sure", "ok", "okay", "correct", "right", "yep", "yeah", 
                      "absolutely", "definitely", "please do", "do it"]
            if any(word in prompt.lower().strip() for word in confirmation_words):
                # User confirmed — create ticket
                st.info("🔀 Router → 🎫 Ticketing Agent")
                context = create_context(st.session_state.pending_issue, 
                                       [m for m in st.session_state.messages])
                context["intent"] = "TICKET"
                context["confidence"] = 1.0
                context = ticketing_agent(context)
                response = context["response"]
                st.session_state.awaiting_confirmation = False
                st.session_state.pending_issue = None
            else:
                # User said something else — reset and reprocess
                st.session_state.awaiting_confirmation = False
                st.session_state.pending_issue = None
                response = "No problem! How else can I help you?"
            
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
        
        else:
            with st.spinner("Multi-agent system processing..."):
                # Create shared context
                context = create_context(prompt, 
                    [m for m in st.session_state.messages[:-1]])
                
                # Router Agent
                context = router_agent(context)
                intent = context["intent"]
                confidence = context["confidence"]
                
                TICKET_CONFIDENCE_THRESHOLD = 0.90
                
                if context["intent"] == "KNOWLEDGE":
                    st.info("🔀 Router → 📚 Knowledge Agent")
                    context = knowledge_agent(context)
                    response = context["response"]
                    
                elif context["intent"] == "TICKET" and context["confidence"] >= TICKET_CONFIDENCE_THRESHOLD:
                    st.info("🔀 Router → 🎫 Ticketing Agent")
                    context = ticketing_agent(context)
                    response = context["response"]
                    
                else:
                    # CLARIFY or low confidence TICKET
                    if context["intent"] == "TICKET":
                        # Ask for confirmation before creating ticket
                        response = f"I want to make sure I create the right ticket. Could you confirm — shall I log a support ticket for: '{prompt}'? (Reply 'yes' to confirm)"
                        st.session_state.awaiting_confirmation = True
                        st.session_state.pending_issue = prompt
                    else:
                        response = "I want to make sure I help you correctly. Are you looking for troubleshooting guidance, or would you like me to create a support ticket? Please clarify and I'll take the right action."
                    
                    st.info("🔀 Router → ❓ Clarification needed")
                    
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })