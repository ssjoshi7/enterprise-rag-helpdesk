import os
import anthropic
import chromadb
import requests
from chromadb.utils import embedding_functions

# ── Load API Key ────────────────────────────────────────────────
import os
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
else:
    try:
        import streamlit as st
        CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
        AIRTABLE_TOKEN = st.secrets["AIRTABLE_TOKEN"]
    except Exception:
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
        AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")

# ── Shared Context ──────────────────────────────────────────────
def create_context(user_message, conversation_history=None):
    return {
        "user_message": user_message,
        "conversation_history": conversation_history or [],
        "intent": None,
        "confidence": None,
        "reason": None,
        "retrieved_chunks": None,
        "answerability": None,
        "ticket_confirmed": False,
        "ticket_id": None,
        "response": None
    }

# ══════════════════════════════════════════════════════════════
# AGENT 1 — ROUTER / ORCHESTRATOR (V3.1)
# Responsibility: Classify intent WITH confidence score
# Returns: KNOWLEDGE, TICKET, or CLARIFY
# Never answers directly — only decides WHO should handle
# ══════════════════════════════════════════════════════════════
def router_agent(context):
    print("\n🔀 Router Agent activated")
    print(f"   Classifying: '{context['user_message']}'")

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""You are a router for an IT helpdesk system.
Classify the following user message and return a JSON response.

Rules:
- KNOWLEDGE: User is asking a question answerable from IT documentation
- TICKET: User explicitly wants to create a ticket or report an issue requiring action
- CLARIFY: Intent is ambiguous — cannot confidently determine if knowledge or action needed

Important: Never default to TICKET just because you are uncertain. Use CLARIFY for uncertainty.

Respond with ONLY this JSON format:
{{"intent": "KNOWLEDGE|TICKET|CLARIFY", "confidence": 0.0-1.0, "reason": "brief explanation"}}

Conversation history:
{chr(10).join([f"{m['role'].upper()}: {m['content']}" for m in context['conversation_history'][-3:]])}

Latest message: {context['user_message']}"""
        }]
    )
    
    import json
    response_text = message.content[0].text.strip()
    
    try:
        # Clean response in case Claude adds markdown
        clean_response = response_text.replace("```json", "").replace("```", "").strip()
        routing = json.loads(clean_response)
        intent = routing.get("intent", "CLARIFY").upper()
        confidence = routing.get("confidence", 0.5)
        reason = routing.get("reason", "")
        
        # Validate intent
        if intent not in ["KNOWLEDGE", "TICKET", "CLARIFY"]:
            intent = "CLARIFY"
            
    except Exception:
        intent = "CLARIFY"
        confidence = 0.0
        reason = "Could not parse routing decision"
    
    context["intent"] = intent
    context["confidence"] = confidence
    context["reason"] = reason
    
    print(f"   Intent: {intent} | Confidence: {confidence:.0%} | Reason: {reason}")
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
    
        # Use original issue from history for better semantic search
    search_query = context["user_message"]
    if context["conversation_history"] and len(context["conversation_history"]) > 2:
        # Find first substantive user message
        for msg in reversed(context["conversation_history"]):
            if msg["role"] == "user" and len(msg["content"]) > 15 and "guidance" not in msg["content"].lower() and "ticket" not in msg["content"].lower():
                search_query = msg["content"]
                break
    
    print(f"   Searching for: '{search_query}'")
    
    # Semantic search — retrieve top 3 relevant chunks
    results = collection.query(
        query_texts=[search_query], 
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
   # AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
    AIRTABLE_BASE_ID = "appeA7AAUGMuiGmvp"
    AIRTABLE_TABLE = "Tickets"

        # Extract actual issue from conversation history
    actual_issue = context["user_message"]
    
    meta_phrases = [
    "create a ticket", 
    "support ticket", 
    "log a ticket", 
    "this concern", 
    "my concern",
    "this issue",
    "for this issue",
    "please create"
]
    is_meta_request = any(phrase in context["user_message"].lower() for phrase in meta_phrases)
    
    # Check if current message contains a specific issue description
    has_specific_issue = len(context["user_message"].split()) > 6  # More than 6 words = specific
    
    if is_meta_request and not has_specific_issue and context["conversation_history"]:
        for msg in reversed(context["conversation_history"]):
            if msg["role"] == "user":
                is_also_meta = any(phrase in msg["content"].lower() for phrase in meta_phrases)
                if not is_also_meta and len(msg["content"]) > 10:
                    actual_issue = msg["content"]
                    break
    
    print(f"   Actual issue identified: '{actual_issue}'")
    
    # Prepare ticket data
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "records": [{
            "fields": {
                "Title": actual_issue,
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
**Issue:** {actual_issue}
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
    print("🚀 V3.1 Multi-Agent IT Helpdesk")
    print("=" * 40)
    
    conversation_history = []
    
    while True:
        user_input = input("\nYour message (or 'quit'): ")
        if user_input.lower() == "quit":
            break
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Create context with history
        context = create_context(user_input, conversation_history)
        
        # Step 1 — Router
        context = router_agent(context)
        
                # Step 2 — Route to correct agent
        # TICKET requires 90%+ confidence — external actions need explicit intent
        TICKET_CONFIDENCE_THRESHOLD = 0.90

        if context["intent"] == "KNOWLEDGE":
            context = knowledge_agent(context)
            print(f"\n🤖 Answer:\n{context['response']}")
        elif context["intent"] == "TICKET" and context["confidence"] >= TICKET_CONFIDENCE_THRESHOLD:
            context = ticketing_agent(context)
            print(f"\n🎫 Response:\n{context['response']}")
        else:
            # Low confidence TICKET or CLARIFY — ask for clarification
            if context["intent"] == "TICKET":
                clarify_response = f"I want to make sure I create the right ticket. Could you confirm — shall I log a support ticket for: '{context['user_message']}'?"
            else:
                clarify_response = "I want to make sure I help you correctly. Are you looking for troubleshooting guidance, or would you like me to create a support ticket? Please clarify and I'll take the right action."
            print(f"\n❓ Clarification needed:\n{clarify_response}")
            conversation_history.append({
                "role": "assistant",
                "content": clarify_response
            })