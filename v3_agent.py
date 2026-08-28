import os
import anthropic
import chromadb
import requests
from chromadb.utils import embedding_functions

# ── Load API Key ────────────────────────────────────────────────
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
else:
    try:
        import streamlit as st
        CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
    except Exception:
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# ── Shared Context ──────────────────────────────────────────────
def create_context(user_message):
    return {
        "user_message": user_message,
        "intent": None,
        "retrieved_chunks": None,
        "ticket_id": None,
        "response": None
    }

# ══════════════════════════════════════════════════════════════
# AGENT 1 — ROUTER / ORCHESTRATOR
# Responsibility: Classify user intent — route to correct agent
# Never answers directly — only decides WHO should answer
# ══════════════════════════════════════════════════════════════
def router_agent(context):
    print("\n🔀 Router Agent activated")
    print(f"   Classifying: '{context['user_message']}'")

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"""You are a router for an IT helpdesk system.
Classify the following user message into exactly one category:

KNOWLEDGE - if the user is asking a question that can be answered from IT documentation
TICKET - if the user needs help that requires logging a support ticket or escalation

Reply with ONLY the single word: KNOWLEDGE or TICKET

User message: {context['user_message']}"""
        }]
    )
    
    intent = message.content[0].text.strip().upper()
    
    # Validate intent
    if intent not in ["KNOWLEDGE", "TICKET"]:
        intent = "TICKET"  # Default to ticket if unclear
    
    context["intent"] = intent
    print(f"   Intent classified as: {intent}")
    return context

# ══════════════════════════════════════════════════════════════
# AGENT 2 — KNOWLEDGE AGENT
# Responsibility: Answer IT questions using RAG pipeline
# Receives: context with KNOWLEDGE intent
# Returns: grounded answer from ChromaDB + Claude
# ══════════════════════════════════════════════════════════════
def knowledge_agent(context):
    print("\n📚 Knowledge Agent activated")
    print(f"   Searching knowledge base for: '{context['user_message']}'")
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name="it_helpdesk_kb",
        embedding_function=embedding_fn
    )
    
    # Semantic search — retrieve top 3 relevant chunks
    results = collection.query(
        query_texts=[context["user_message"]], 
        n_results=3
    )
    chunks = results["documents"][0]
    context["retrieved_chunks"] = chunks
    print(f"   Retrieved {len(chunks)} relevant chunks from ChromaDB")
    
    # Claude grounded generation
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    context_text = "\n\n".join(chunks)
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""You are an Enterprise IT Helpdesk specialist.
Answer the user's question using ONLY the context below.
If the answer is not in the context, say: 'I don't have that information. Please contact IT support at swapniljoshi1729@gmail.com'

Context:
{context_text}

User Question: {context['user_message']}

Answer:"""
        }]
    )
    
    context["response"] = message.content[0].text
    print(f"   ✅ Knowledge Agent response generated")
    return context

# ══════════════════════════════════════════════════════════════
# AGENT 3 — TICKETING AGENT
# Responsibility: Log support ticket to Airtable via REST API
# Receives: context with TICKET intent
# Returns: confirmation with ticket details
# ══════════════════════════════════════════════════════════════
def ticketing_agent(context):
    print("\n🎫 Ticketing Agent activated")
    print(f"   Logging ticket for: '{context['user_message']}'")
    
    # Airtable configuration
    AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
    AIRTABLE_BASE_ID = "appeA7AAUGMuiGmvp"
    AIRTABLE_TABLE = "Tickets"
    
    # Prepare ticket data
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "records": [{
            "fields": {
                "Title": context["user_message"],
                "Status": "Open",
                "Urgency": "Medium"
            }
        }]
    }
    
    # Make API call to Airtable
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        record = response.json()["records"][0]
        ticket_id = record["id"]
        context["ticket_id"] = ticket_id
        context["response"] = f"""✅ Support ticket logged successfully!

**Ticket ID:** {ticket_id}
**Issue:** {context['user_message']}
**Status:** Open
**Priority:** Medium

Our IT team will follow up at:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190
🕐 Monday–Friday, 9AM–6PM IST"""
        print(f"   ✅ Ticket created: {ticket_id}")
    else:
        context["response"] = """⚠️ Could not log ticket automatically.
Please contact IT support directly:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190"""
        print(f"   ⚠️ Airtable API error: {response.status_code}")
    
    return context

# ── Entry point for testing ─────────────────────────────────────
if __name__ == "__main__":
    print("🚀 V3 Multi-Agent IT Helpdesk")
    print("=" * 40)
    
    while True:
        user_input = input("\nYour message (or 'quit'): ")
        if user_input.lower() == "quit":
            break
            
        # Create shared context
        context = create_context(user_input)
        
                # Step 1 — Router classifies intent
        context = router_agent(context)
        
        # Step 2 — Route to correct agent
        if context["intent"] == "KNOWLEDGE":
            context = knowledge_agent(context)
            print(f"\n🤖 Answer:\n{context['response']}")
        else:
            context = ticketing_agent(context)
            print(f"\n🎫 Response:\n{context['response']}")