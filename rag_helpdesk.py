import os
import json
import anthropic
import chromadb
from chromadb.utils import embedding_functions

# ── Load API Key ────────────────────────────────────────────────
import os
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

KB_FILE = "EnterpriseITHelpdesk_KnowledgeBase_SSJoshi.jsonl"
COLLECTION_NAME = "it_helpdesk_kb"

# ── Initialize ChromaDB ─────────────────────────────────────────
print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# ── Load and index JSONL KB ─────────────────────────────────────
def index_documents():
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks. Skipping indexing.")
        return

    print(f"Loading JSONL knowledge base: {KB_FILE}")
    
    documents = []
    ids = []
    metadatas = []

    with open(KB_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            documents.append(record["content"])
            ids.append(record["id"])
            metadatas.append({
                "category": record["category"],
                "subcategory": record["subcategory"],
                "source": record["source"],
                "version": record["version"],
                "last_updated": record["last_updated"],
                "review_status": record["review_status"]
            })

    print(f"Indexing {len(documents)} documents with metadata...")
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ Indexed {len(documents)} documents into ChromaDB with full metadata!")

# ── Category Detection ──────────────────────────────────────────
def detect_category(query):
    """
    Detect query category using Claude.
    Returns category string or None if ambiguous.
    """
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"""Classify this IT helpdesk query into exactly one category.
Available categories: PASSWORD, JIRA, SALESFORCE, VPN, EMAIL, HARDWARE, SOFTWARE, ESCALATION

Rules:
- If query clearly maps to one category → return that category name only
- If query is ambiguous or spans multiple categories → return NONE
- Return ONLY the category name or NONE. Nothing else.

Query: {query}"""
            }]
        )
        
        result = message.content[0].text.strip().upper()
        valid_categories = ["PASSWORD", "JIRA", "SALESFORCE", "VPN", "EMAIL", "HARDWARE", "SOFTWARE", "ESCALATION"]
        
        if result in valid_categories:
            print(f"   🏷️ Category detected: {result}")
            return result
        else:
            print(f"   🏷️ Category: AMBIGUOUS — using global search")
            return None
            
    except Exception as e:
        print(f"   ⚠️ Category detection failed: {e}")
        return None

# ── Reranking — Claude as reranker ─────────────────────────────
def rerank_chunks(query, chunks, metadatas):
    """
    Use Claude to rerank retrieved chunks by true relevance.
    Returns reranked chunks and metadatas in order of relevance.
    """
    if len(chunks) <= 1:
        return chunks, metadatas
    
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    
    # Build numbered chunk list for Claude
    chunks_text = "\n\n".join([
        f"Chunk {i+1} [{metadatas[i]['category']} — {metadatas[i]['subcategory']}]:\n{chunk}"
        for i, chunk in enumerate(chunks)
    ])
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""You are a relevance judge for an IT helpdesk system.
Given a user query and a list of knowledge base chunks, rank the chunks from most to least relevant.

User Query: {query}

Chunks:
{chunks_text}

Return ONLY a comma-separated list of chunk numbers in order of relevance.
Example: 3,1,2
Nothing else."""
            }]
        )
        
        ranking_text = message.content[0].text.strip()
        ranking = [int(x.strip()) - 1 for x in ranking_text.split(",")]
        
        # Reorder chunks and metadatas based on ranking
        reranked_chunks = [chunks[i] for i in ranking if i < len(chunks)]
        reranked_metadatas = [metadatas[i] for i in ranking if i < len(metadatas)]
        
        print(f"   🔄 Reranked order: {ranking_text}")
        return reranked_chunks, reranked_metadatas
        
    except Exception as e:
        print(f"   ⚠️ Reranking failed: {e} — using original order")
        return chunks, metadatas

# ── Semantic retrieval with optional metadata filtering ─────────
def retrieve(query, top_k=5):
    """
    Smart retrieval:
    - Detect category from query
    - If high confidence category → filter by metadata first
    - If ambiguous → global semantic search
    - Always fall back to global if filtered results insufficient
    """
    print(f"\n   🔎 Starting retrieval for: '{query}'")
    
    # Step 1 — Detect category
    detected_category = detect_category(query)
    
    # Step 2 — Try filtered retrieval if category detected
    if detected_category:
        try:
            filtered_results = collection.query(
                query_texts=[query],
                n_results=min(top_k, 3),
                include=["documents", "distances", "metadatas"],
                where={"category": {"$eq": detected_category}}
            )
            
            filtered_docs = filtered_results["documents"][0]
            filtered_distances = filtered_results["distances"][0]
            filtered_metas = filtered_results["metadatas"][0]
            
            # Check if filtered results are good enough
            if filtered_docs and len(filtered_docs) > 0:
                similarity_scores = [round(1 / (1 + d), 3) for d in filtered_distances]
                best_score = max(similarity_scores)
                
                if best_score >= 0.4:
                    print(f"   ✅ Using filtered results for category: {detected_category}")
                    return {
                        "documents": filtered_docs,
                        "distances": filtered_distances,
                        "metadatas": filtered_metas,
                        "retrieval_mode": f"filtered:{detected_category}"
                    }
                else:
                    print(f"   ⚠️ Filtered results weak — falling back to global search")
        except Exception as e:
            print(f"   ⚠️ Filtered retrieval failed: {e} — falling back to global")
    
    # Step 3 — Global semantic search fallback
    print(f"   🌐 Using global semantic search")
    global_results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )
    
    return {
        "documents": global_results["documents"][0],
        "distances": global_results["distances"][0],
        "metadatas": global_results["metadatas"][0],
        "retrieval_mode": "global"
    }

# ── Generate grounded answer via Claude ────────────────────────
def ask_claude(query, context_chunks):
    context = "\n\n".join(context_chunks)
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""You are an Enterprise IT Helpdesk assistant.
Answer the user's question using ONLY the context below.
If the answer is not in the context, say: 'I don't have information on that. Please contact IT support at swapniljoshi1729@gmail.com'

Context:
{context}

User Question: {query}

Answer:"""
            }
        ]
    )
    return message.content[0].text

# ── Main RAG pipeline ───────────────────────────────────────────
def rag_query(query):
    print(f"\n🔍 Query: {query}")
    
    results = retrieve(query)
    
    documents = results["documents"]
    distances = results["distances"]
    metadatas = results["metadatas"]
    retrieval_mode = results.get("retrieval_mode", "global")
    
    # Similarity scores
    similarity_scores = [round(1 / (1 + d), 3) for d in distances]
    
    print(f"\n📊 Retrieval Log — Mode: {retrieval_mode}")
    for i, (doc, score, meta) in enumerate(zip(documents, similarity_scores, metadatas)):
        print(f"   Chunk {i+1} | Score: {score} | Category: {meta['category']} | Subcategory: {meta['subcategory']}")
    
        # Threshold filter
    SIMILARITY_THRESHOLD = 0.5
    strong_chunks = [
        doc for doc, score in zip(documents, similarity_scores)
        if score >= SIMILARITY_THRESHOLD
    ]
    strong_metas = [
        meta for meta, score in zip(metadatas, similarity_scores)
        if score >= SIMILARITY_THRESHOLD
    ]
    
    print(f"📚 {len(strong_chunks)}/{len(documents)} chunks passed threshold {SIMILARITY_THRESHOLD}")
    
    if not strong_chunks:
        return "I don't have sufficient information to answer that. Please contact IT support at swapniljoshi1729@gmail.com"
    
    # Reranking — only if more than 1 chunk passed threshold
    if len(strong_chunks) > 1:
        print(f"   🔄 Reranking {len(strong_chunks)} chunks...")
        strong_chunks, strong_metas = rerank_chunks(query, strong_chunks, strong_metas)
    
    # Send top 2 reranked chunks to Claude — not all
    final_chunks = strong_chunks[:2]
    final_metas = strong_metas[:2]
    
    print(f"   ✅ Sending top {len(final_chunks)} reranked chunks to Claude")
    for i, (chunk, meta) in enumerate(zip(final_chunks, final_metas)):
        print(f"   Final Chunk {i+1} | Category: {meta['category']} | Subcategory: {meta['subcategory']}")
    
    answer = ask_claude(query, final_chunks)
    
    print(f"\n🤖 Answer:\n{answer}")
    return answer

# ── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    index_documents()
    print("\n✅ RAG Helpdesk ready!\n")
    while True:
        query = input("Your question (or 'quit'): ")
        if query.lower() == "quit":
            break
        rag_query(query)