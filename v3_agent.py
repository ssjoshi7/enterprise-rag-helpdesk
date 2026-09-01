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
        "retrieval_log": [],
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
    
    # ── Initialize ChromaDB with failure handling ───────────────
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        collection = chroma_client.get_or_create_collection(
            name="it_helpdesk_kb",
            embedding_function=embedding_fn
        )
    except Exception as e:
        print(f"   ❌ ChromaDB connection failed: {e}")
        context["response"] = """Our knowledge base is temporarily unavailable.

Please contact IT Support directly:
📧 Email: swapniljoshi1729@gmail.com
📞 Phone: +91 9371615190
🕐 Available: Monday–Friday, 9AM–6PM IST"""
        context["retrieval_log"] = [{"error": "ChromaDB connection failed", "detail": str(e)}]
        return context
    
    # ── Smart search query from conversation history ────────────
    search_query = context["user_message"]
    if context["conversation_history"] and len(context["conversation_history"]) > 2:
        for msg in reversed(context["conversation_history"]):
            if msg["role"] == "user" and len(msg["content"]) > 15 and "guidance" not in msg["content"].lower() and "ticket" not in msg["content"].lower():
                search_query = msg["content"]
                break
    
    print(f"   Searching for: '{search_query}'")
    
    # ── Semantic search with failure handling ───────────────────
    try:
        results = collection.query(
            query_texts=[search_query],
            n_results=3,
            include=["documents", "distances"]
        )
        chunks = results["documents"][0]
        distances = results["distances"][0]
    except Exception as e:
        print(f"   ❌ ChromaDB query failed: {e}")
        context["response"] = """Unable to search the knowledge base at this time.

Please contact IT Support directly:
📧 Email: swapniljoshi1729@gmail.com
📞 Phone: +91 9371615190
🕐 Available: Monday–Friday, 9AM–6PM IST"""
        context["retrieval_log"] = [{"error": "ChromaDB query failed", "detail": str(e)}]
        return context
    
    # ── Retrieval Logging ───────────────────────────────────────
    similarity_scores = [round(1 / (1 + d), 3) for d in distances]
    
    print(f"\n   📊 Retrieval Log:")
    print(f"   Query: '{search_query}'")
    for i, (chunk, score) in enumerate(zip(chunks, similarity_scores)):
        print(f"   Chunk {i+1} | Score: {score} | Preview: '{chunk[:60]}...'")
    
    context["retrieval_log"] = [
        {"chunk_index": i+1, "similarity_score": score, "chunk_preview": chunk[:100]}
        for i, (chunk, score) in enumerate(zip(chunks, similarity_scores))
    ]
    
    # ── Similarity Threshold ────────────────────────────────────
    SIMILARITY_THRESHOLD = 0.5
    
    strong_chunks = [
        chunk for chunk, score
        in zip(chunks, similarity_scores)
        if score >= SIMILARITY_THRESHOLD
    ]
    
    print(f"\n   🎯 Threshold: {SIMILARITY_THRESHOLD}")
    print(f"   Chunks above threshold: {len(strong_chunks)}/{len(chunks)}")
    
    # ── Fallback if no chunks pass threshold ────────────────────
    if not strong_chunks:
        print(f"   ⚠️ No chunks above threshold — escalating")
        context["response"] = """I wasn't able to find sufficiently relevant information in the knowledge base for your query.

Please contact IT Support directly:
📧 Email: swapniljoshi1729@gmail.com
📞 Phone: +91 9371615190
🕐 Available: Monday–Friday, 9AM–6PM IST"""
        context["retrieval_log"].append({"threshold_result": "FAILED — escalated"})
        return context
    
    # ── Claude grounded generation with retry ───────────────────
    print(f"   ✅ {len(strong_chunks)} strong chunks passed to Claude")
    
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    context_text = "\n\n".join(strong_chunks)
    
    MAX_RETRIES = 2
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"   🔄 Claude generation attempt {attempt}/{MAX_RETRIES}")
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
            context["retrieval_log"].append({
                "threshold_result": "PASSED",
                "chunks_used": len(strong_chunks),
                "claude_attempt": attempt
            })
            print(f"   ✅ Knowledge Agent response generated on attempt {attempt}")
            return context
            
        except Exception as e:
            print(f"   ⚠️ Claude attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                print(f"   ❌ All Claude attempts failed — escalating")
                context["response"] = """I was able to find relevant information but encountered an error generating the response.

Please contact IT Support directly:
📧 Email: swapniljoshi1729@gmail.com
📞 Phone: +91 9371615190
🕐 Available: Monday–Friday, 9AM–6PM IST"""
                context["retrieval_log"].append({
                    "threshold_result": "PASSED",
                    "claude_error": str(e),
                    "final_status": "FAILED — escalated after retries"
                })
    
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
    
    # ── Airtable configuration ──────────────────────────────────
    # AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
    AIRTABLE_BASE_ID = "appeA7AAUGMuiGmvp"
    AIRTABLE_TABLE = "Tickets"
    MAX_RETRIES = 2
    REQUEST_TIMEOUT = 10  # seconds

    # ── Extract actual issue from conversation history ──────────
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
    has_specific_issue = len(context["user_message"].split()) > 6

    if is_meta_request and not has_specific_issue and context["conversation_history"]:
        for msg in reversed(context["conversation_history"]):
            if msg["role"] == "user":
                is_also_meta = any(phrase in msg["content"].lower() for phrase in meta_phrases)
                if not is_also_meta and len(msg["content"]) > 10:
                    actual_issue = msg["content"]
                    break

    print(f"   Actual issue identified: '{actual_issue}'")

    # ── Prepare ticket payload ──────────────────────────────────
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

    # ── API call with retry logic ───────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"   🔄 Airtable API attempt {attempt}/{MAX_RETRIES}")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            # ── Validate response status ────────────────────────
            if response.status_code == 200:
                try:
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
                    return context

                except (KeyError, IndexError, ValueError) as e:
                    print(f"   ⚠️ Malformed Airtable response: {e}")
                    if attempt == MAX_RETRIES:
                        context["response"] = """⚠️ Ticket was submitted but response was unexpected.
Please contact IT support to confirm:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190"""
                    continue

            elif response.status_code == 401:
                print(f"   ❌ Airtable authentication failed — invalid token")
                context["response"] = """⚠️ Unable to log ticket — authentication error.
Please contact IT support directly:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190"""
                return context

            elif response.status_code == 422:
                print(f"   ❌ Airtable rejected payload — malformed data")
                context["response"] = """⚠️ Unable to log ticket — invalid ticket data.
Please contact IT support directly:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190"""
                return context

            else:
                print(f"   ⚠️ Airtable returned status {response.status_code}")
                if attempt == MAX_RETRIES:
                    context["response"] = f"""⚠️ Could not log ticket automatically (Error {response.status_code}).
Please contact IT support directly:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190"""

        except requests.exceptions.Timeout:
            print(f"   ⚠️ Airtable request timed out on attempt {attempt}")
            if attempt == MAX_RETRIES:
                context["response"] = """⚠️ Ticket logging timed out after multiple attempts.
Please contact IT support directly:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190"""

        except requests.exceptions.ConnectionError:
            print(f"   ⚠️ Airtable connection error on attempt {attempt}")
            if attempt == MAX_RETRIES:
                context["response"] = """⚠️ Unable to reach ticketing system.
Please contact IT support directly:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190"""

        except Exception as e:
            print(f"   ⚠️ Unexpected error on attempt {attempt}: {e}")
            if attempt == MAX_RETRIES:
                context["response"] = """⚠️ An unexpected error occurred while logging your ticket.
Please contact IT support directly:
📧 swapniljoshi1729@gmail.com
📞 +91 9371615190"""

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