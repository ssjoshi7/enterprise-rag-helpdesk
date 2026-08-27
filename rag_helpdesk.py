import os
import anthropic
import chromadb
from chromadb.utils import embedding_functions

# ── Configuration ──────────────────────────────────────────────
import os

# Try Streamlit secrets first, fall back to .env
if "CLAUDE_API_KEY" in os.environ:
    CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
else:
    try:
        import streamlit as st
        CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

KB_FILE = "EnterpriseITHelpdesk_KnowledgeBase_SSJoshi.txt"
COLLECTION_NAME = "it_helpdesk_kb"

# ── Initialize ChromaDB with built-in embeddings ────────────────
print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# ── Load and chunk text file ────────────────────────────────────
def load_and_chunk_txt(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    return chunks

# ── Index documents into ChromaDB ──────────────────────────────
def index_documents():
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks. Skipping indexing.")
        return
    print(f"Loading document: {KB_FILE}")
    chunks = load_and_chunk_txt(KB_FILE)
    print(f"Chunked into {len(chunks)} pieces. Embedding now...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    print(f"✅ Indexed {len(chunks)} chunks into ChromaDB!")

# ── Semantic retrieval ──────────────────────────────────────────
def retrieve(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]

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
    chunks = retrieve(query)
    print(f"📚 Retrieved {len(chunks)} relevant chunks")
    answer = ask_claude(query, chunks)
    print(f"\n🤖 Answer:\n{answer}")
    return answer

# ── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    index_documents()
    print("\n✅ RAG Helpdesk ready! Ask your IT questions.\n")
    while True:
        query = input("Your question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break
        rag_query(query)