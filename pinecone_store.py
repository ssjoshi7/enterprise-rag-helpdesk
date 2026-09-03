# ── pinecone_store.py — Persistent Cloud Vector Store ───────────
# V4.5 — Replace ChromaDB local with Pinecone cloud storage

import os
import json
from pinecone import Pinecone
from chromadb.utils import embedding_functions

# ── Initialize embedding function ──────────────────────────────
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# ── Initialize Pinecone ─────────────────────────────────────────
def get_pinecone_index():
    """Get Pinecone index — persistent cloud vector store."""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            import streamlit as st
            api_key = st.secrets.get("PINECONE_API_KEY")
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index("it-helpdesk-kb")
        print("   ✅ Pinecone index connected!")
        return index
    except Exception as e:
        print(f"   ❌ Pinecone connection failed: {e}")
        return None

# ── Index documents into Pinecone ──────────────────────────────
def index_documents_pinecone(kb_file):
    """
    Index JSONL knowledge base into Pinecone.
    Stores vectors with full metadata.
    """
    index = get_pinecone_index()
    if not index:
        return False

    # Check if already indexed
    stats = index.describe_index_stats()
    if stats.total_vector_count > 0:
        print(f"   ✅ Pinecone already has {stats.total_vector_count} vectors — skipping indexing")
        return True

    print(f"   📄 Loading JSONL KB: {kb_file}")
    
    documents = []
    ids = []
    metadatas = []
    contents = []

    with open(kb_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            contents.append(record["content"])
            ids.append(record["id"])
            metadatas.append({
                "category": record["category"],
                "subcategory": record["subcategory"],
                "source": record["source"],
                "version": record["version"],
                "last_updated": record["last_updated"],
                "review_status": record["review_status"],
                "content": record["content"]  # store content in metadata for retrieval
            })

    print(f"   🔢 Generating embeddings for {len(contents)} documents...")
    embeddings = embedding_fn(contents)

    # Prepare vectors for Pinecone
    vectors = []
    for i, (id_, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
        vectors.append({
            "id": id_,
            "values": embedding,
            "metadata": metadata
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"   ✅ Upserted batch {i//batch_size + 1}")

    print(f"   🎉 Successfully indexed {len(vectors)} documents to Pinecone!")
    return True

# ── Semantic retrieval from Pinecone ────────────────────────────
def retrieve_pinecone(query, top_k=5, category_filter=None):
    """
    Retrieve relevant chunks from Pinecone.
    Supports optional metadata filtering.
    """
    index = get_pinecone_index()
    if not index:
        return None

    # Generate query embedding
    query_embedding = embedding_fn([query])[0].tolist()

    # Build filter if category provided
    filter_dict = None
    if category_filter:
        filter_dict = {"category": {"$eq": category_filter}}

    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

        documents = [match.metadata["content"] for match in results.matches]
        scores = [match.score for match in results.matches]
        metadatas = [{
            "category": match.metadata.get("category", ""),
            "subcategory": match.metadata.get("subcategory", ""),
            "source": match.metadata.get("source", ""),
            "version": match.metadata.get("version", ""),
            "last_updated": match.metadata.get("last_updated", ""),
            "review_status": match.metadata.get("review_status", "")
        } for match in results.matches]

        return {
            "documents": documents,
            "scores": scores,
            "metadatas": metadatas,
            "retrieval_mode": f"pinecone:{category_filter or 'global'}"
        }

    except Exception as e:
        print(f"   ❌ Pinecone query failed: {e}")
        return None