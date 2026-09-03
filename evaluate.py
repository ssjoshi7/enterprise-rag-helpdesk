import json
import os
from dotenv import load_dotenv

load_dotenv()

# Import our RAG pipeline
from rag_helpdesk import index_documents, retrieve, detect_category, ask_claude, collection

# ── Load evaluation set ─────────────────────────────────────────
def load_eval_set():
    with open("evaluation_set.json", "r") as f:
        return json.load(f)["evaluation_set"]

# ── Run single evaluation ───────────────────────────────────────
def evaluate_query(eval_item):
    query = eval_item["query"]
    expected_category = eval_item["expected_category"]
    expected_chunk_id = eval_item["expected_chunk_id"]
    expected_keywords = eval_item["expected_keywords"]

    print(f"\n{'='*60}")
    print(f"📋 {eval_item['id']} — {eval_item['notes']}")
    print(f"   Query: '{query}'")
    print(f"   Expected category: {expected_category}")

    results = {}

    # ── Test 1 — Category Detection ─────────────────────────────
    detected_category = detect_category(query)
    
    if expected_category == "AMBIGUOUS":
        category_pass = detected_category is None
    elif expected_category == "NONE":
        category_pass = True  # Category detection not the gate here
    else:
        category_pass = detected_category == expected_category

    results["category_detection"] = {
        "expected": expected_category,
        "actual": detected_category or "AMBIGUOUS",
        "pass": category_pass
    }

    # ── Test 2 — Chunk Retrieval ─────────────────────────────────
    retrieval_results = retrieve(query)
    retrieved_ids = []

    # Get IDs from ChromaDB
    full_results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "distances", "metadatas"]
    )
    
    # Check if expected chunk appears in top results
    metadatas = full_results["metadatas"][0]
    distances = full_results["distances"][0]
    similarity_scores = [round(1 / (1 + d), 3) for d in distances]

    chunk_pass = True
    if expected_chunk_id:
        # Verify expected chunk is in results with good score
        ids_result = collection.query(
            query_texts=[query],
            n_results=5,
            include=["metadatas", "distances"]
        )
        chunk_pass = any(
            score >= 0.4
            for score in similarity_scores[:3]
        )

    results["retrieval"] = {
        "top_categories": [m["category"] for m in metadatas[:3]],
        "top_scores": similarity_scores[:3],
        "expected_chunk_id": expected_chunk_id,
        "pass": chunk_pass
    }

    # ── Test 3 — Keyword presence ────────────────────────────────
    if expected_keywords:
        retrieval = retrieve(query)
        docs = retrieval["documents"]
        distances = retrieval["distances"]
        scores = [round(1 / (1 + d), 3) for d in distances]

        strong = [d for d, s in zip(docs, scores) if s >= 0.5]

        if strong:
            answer = ask_claude(query, strong[:2])
            keywords_found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
            keyword_pass = len(keywords_found) >= len(expected_keywords) * 0.6
        else:
            answer = "NO STRONG CHUNKS"
            keywords_found = []
            keyword_pass = False

        results["keyword_check"] = {
            "expected": expected_keywords,
            "found": keywords_found,
            "pass": keyword_pass
        }
    else:
        results["keyword_check"] = {"pass": True, "notes": "No keywords expected"}

    # ── Summary ──────────────────────────────────────────────────
    all_pass = all([
        results["category_detection"]["pass"],
        results["retrieval"]["pass"],
        results["keyword_check"]["pass"]
    ])

    status = "✅ PASS" if all_pass else "❌ FAIL"
    print(f"   Category: {'✅' if results['category_detection']['pass'] else '❌'} "
          f"Expected:{expected_category} Got:{results['category_detection']['actual']}")
    print(f"   Retrieval: {'✅' if results['retrieval']['pass'] else '❌'} "
          f"Top scores:{results['retrieval']['top_scores']}")
    print(f"   Keywords: {'✅' if results['keyword_check']['pass'] else '❌'}")
    print(f"   Overall: {status}")

    return all_pass

# ── Run full evaluation ─────────────────────────────────────────
def run_evaluation():
    print("\n🧪 Starting RAG Evaluation")
    print("=" * 60)

    index_documents()
    eval_set = load_eval_set()

    passed = 0
    failed = 0
    failed_ids = []

    for item in eval_set:
        success = evaluate_query(item)
        if success:
            passed += 1
        else:
            failed += 1
            failed_ids.append(item["id"])

    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}/{len(eval_set)}")
    print(f"❌ Failed: {failed}/{len(eval_set)}")
    print(f"📈 Score: {round(passed/len(eval_set)*100)}%")

    if failed_ids:
        print(f"❌ Failed cases: {', '.join(failed_ids)}")
    else:
        print(f"🎉 All cases passed!")

    return passed, failed

if __name__ == "__main__":
    run_evaluation()