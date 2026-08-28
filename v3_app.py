import streamlit as st
from v3_agent import create_context, router_agent, knowledge_agent, ticketing_agent

st.set_page_config(
    page_title="Enterprise IT Helpdesk V3",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise IT Helpdesk — Multi-Agent V3")
st.caption("Powered by Multi-Agent RAG + ChromaDB + Claude AI + Airtable | Built by Swapnil Joshi")
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
        with st.spinner("Multi-agent system processing..."):
            # Create shared context
            context = create_context(prompt)
            
            # Router Agent
            context = router_agent(context)
            intent = context["intent"]
            
            # Route to correct agent
            if intent == "KNOWLEDGE":
                st.info("🔀 Router → 📚 Knowledge Agent")
                context = knowledge_agent(context)
            else:
                st.info("🔀 Router → 🎫 Ticketing Agent")
                context = ticketing_agent(context)
            
            response = context["response"]
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })