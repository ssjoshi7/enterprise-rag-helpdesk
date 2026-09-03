# ── metrics.py — Observability & Operations ─────────────────────
# V4.4 — Track system behavior for monitoring and improvement

import json
import os
from datetime import datetime

METRICS_FILE = "metrics.json"

# ── Load existing metrics ───────────────────────────────────────
def load_metrics():
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"interactions": []}
    return {"interactions": []}

# ── Save metrics ────────────────────────────────────────────────
def save_metrics(metrics):
    try:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"   ⚠️ Metrics save failed: {e}")

# ── Log single interaction ──────────────────────────────────────
def log_interaction(context, latency_ms=None, user_email=None, user_role=None):
    """
    Log every interaction with full context for observability.
    """
    metrics = load_metrics()

    interaction = {
        "timestamp": datetime.now().isoformat(),
        "user_email": user_email or "unknown",
        "user_role": user_role or "unknown",
        "query": context.get("user_message", ""),
        "intent": context.get("intent", "unknown"),
        "confidence": context.get("confidence", 0),
        "workflow_state": context.get("workflow_state", "unknown"),
        "workflow_history": context.get("workflow_history", []),
        "retrieval_log": context.get("retrieval_log", []),
        "ticket_id": context.get("ticket_id"),
        "duplicate_detected": context.get("duplicate_check") is not None,
        "latency_ms": latency_ms,
        "error_state": context.get("error_state")
    }

    metrics["interactions"].append(interaction)
    save_metrics(metrics)
    print(f"   📊 Interaction logged — intent: {interaction['intent']} | state: {interaction['workflow_state']}")

# ── Compute summary stats ───────────────────────────────────────
def get_summary_stats():
    metrics = load_metrics()
    interactions = metrics.get("interactions", [])

    if not interactions:
        return None

    total = len(interactions)

    # Route distribution
    knowledge_count = sum(1 for i in interactions if i.get("intent") == "KNOWLEDGE")
    ticket_count = sum(1 for i in interactions if i.get("intent") == "TICKET")
    clarify_count = sum(1 for i in interactions if i.get("intent") == "CLARIFY")

    # Success/failure
    success_count = sum(1 for i in interactions if i.get("workflow_state") == "SUCCESS")
    failed_count = sum(1 for i in interactions if i.get("workflow_state") == "FAILED")

    # Duplicate detection
    duplicate_count = sum(1 for i in interactions if i.get("duplicate_detected"))

    # Average latency
    latencies = [i.get("latency_ms") for i in interactions if i.get("latency_ms")]
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else 0

    # Retrieval success rate
    retrieval_interactions = [i for i in interactions if i.get("intent") == "KNOWLEDGE"]
    retrieval_success = sum(
        1 for i in retrieval_interactions
        if i.get("workflow_state") == "SUCCESS"
    )
    retrieval_rate = round(retrieval_success / len(retrieval_interactions) * 100) if retrieval_interactions else 0

    return {
        "total_interactions": total,
        "route_distribution": {
            "KNOWLEDGE": knowledge_count,
            "TICKET": ticket_count,
            "CLARIFY": clarify_count
        },
        "success_rate": round(success_count / total * 100) if total else 0,
        "failed_count": failed_count,
        "duplicate_count": duplicate_count,
        "avg_latency_ms": avg_latency,
        "retrieval_success_rate": retrieval_rate,
        "clarification_rate": round(clarify_count / total * 100) if total else 0,
        "recent_interactions": interactions[-10:]
    }