from reddit_source import build_evidence_stream

from evidence_engine import (
    enrich_relevance,
    enrich_with_stance,
    enrich_ranking
)

from confidence_engine import compute_confidence_from_evidence
from timeline_visualizer import render_timeline
from query_engine import generate_search_queries

from state_store import TemporalEvidenceStore

from graph_engine import (
    build_evidence_graph,
    graph_confidence_adjustment,
    apply_temporal_weighting
)

from temporal_engine import compute_temporal_trend

from graph_visualizer import render_graph
from heatmap_visualizer import render_subreddit_heatmap


# =========================================================
# QUERY INTELLIGENCE LAYER (NEW CRITICAL FIX)
# =========================================================

STOPWORDS = {
    "i", "heard", "think", "is", "are", "the", "a", "an", "really",
    "very", "so", "just", "like", "that", "this", "to", "of", "in",
    "on", "for", "with", "and", "or", "it", "at", "as", "be"
}


def clean_query(q: str) -> str:
    """
    Removes noise words and stabilizes query structure.
    """
    words = q.lower().split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered).strip()


def score_query(q: str) -> float:
    """
    Heuristic scoring: encourages crisis-relevant signal.
    """
    score = 0.0

    # length reward (too short = bad signal)
    if 3 <= len(q.split()) <= 10:
        score += 0.3

    # crisis keywords boost
    crisis_keywords = {
        "fire", "wildfire", "evacuation", "explosion",
        "shooting", "flood", "earthquake", "crash",
        "spreading", "damage", "emergency", "alert"
    }

    for k in crisis_keywords:
        if k in q.lower():
            score += 0.1

    # penalty for repetition artifacts
    tokens = q.split()
    if len(tokens) != len(set(tokens)):
        score -= 0.2

    return score


def refine_queries(queries):
    """
    Filters + ranks generated queries.
    """
    cleaned = [clean_query(q) for q in queries]
    scored = [(q, score_query(q)) for q in cleaned]

    scored.sort(key=lambda x: x[1], reverse=True)

    # keep top diverse queries
    final = []
    seen = set()

    for q, _ in scored:
        if q and q not in seen:
            final.append(q)
            seen.add(q)

        if len(final) >= 6:
            break

    return final


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline():

    print("\n========================================")
    print("     LIVE CRISIS EVIDENCE ENGINE")
    print("        (STEP 10: FULL TEMPORAL SYSTEM)")
    print("========================================\n")

    # =====================================================
    # 1. USER INPUT
    # =====================================================

    user_input = input("Enter a claim or topic: ")

    # =====================================================
    # 2. GENERATE QUERIES
    # =====================================================

    raw_queries = generate_search_queries(user_input)
    queries = refine_queries(raw_queries)

    print("\n🧠 Refined Search Queries:\n")

    for q in queries:
        print(f" - {q}")

    print()

    # =====================================================
    # 3. BUILD EVIDENCE STREAM
    # =====================================================

    evidence = []

    for q in queries:

        print(f"🔎 Searching Reddit for: {q}")

        try:
            batch = build_evidence_stream(q)
            if batch:
                evidence.extend(batch)

        except Exception as e:
            print(f"[Search failed] {q}")
            print(e)

    # =====================================================
    # 4. DEDUPLICATION
    # =====================================================

    deduped = {}
    for ev in evidence:
        key = ev.text.strip().lower()
        if key not in deduped:
            deduped[key] = ev

    evidence = list(deduped.values())

    if not evidence:
        print("\nNo evidence found.\n")
        return

    print(f"\n✅ Retrieved {len(evidence)} unique evidence items.\n")

    # =====================================================
    # 5. ENRICHMENT PIPELINE
    # =====================================================

    evidence = enrich_relevance(user_input, evidence)

    evidence = [
        e for e in evidence
        if e.relevance_score >= 0.35
    ]

    if not evidence:
        print("\nNo sufficiently relevant evidence found.\n")
        return

    print(f"✅ Retained {len(evidence)} relevant evidence items.\n")

    evidence = enrich_with_stance(user_input, evidence)
    evidence = enrich_ranking(user_input, evidence)

    evidence.sort(
        key=lambda x: getattr(x, "ranking_score", 0.0),
        reverse=True
    )

    # =====================================================
    # 6. MEMORY STORE
    # =====================================================

    store = TemporalEvidenceStore()
    store.add_batch(evidence)
    all_evidence = store.get_all()

    # =====================================================
    # 7. DEBUG VIEW
    # =====================================================

    print("\n================ RAW EVIDENCE SAMPLE ================\n")

    for i, ev in enumerate(evidence[:10]):
        print(f"{i+1}. [{ev.source_type}]")
        print(f"   text: {ev.text}")
        print(f"   subreddit: {ev.subreddit}")
        print(f"   relevance_score: {ev.relevance_score:.3f}")
        print(f"   stance_confidence: {getattr(ev, 'stance_confidence', 0.0):.3f}")
        print(f"   ranking_score: {getattr(ev, 'ranking_score', 0.0):.3f}")
        print(f"   stance: {ev.stance}")
        print("--------------------------------------------------")

    # =====================================================
    # 8. STANCE BREAKDOWN
    # =====================================================

    support = [e for e in evidence if e.stance == "SUPPORT"]
    contradict = [e for e in evidence if e.stance == "CONTRADICT"]
    neutral = [e for e in evidence if e.stance == "NEUTRAL"]

    print("\n================ STANCE BREAKDOWN ================\n")
    print(f"SUPPORT ({len(support)})")
    print(f"CONTRADICT ({len(contradict)})")
    print(f"NEUTRAL ({len(neutral)})")

    # =====================================================
    # 9. CONFIDENCE
    # =====================================================

    base_confidence, explanation = compute_confidence_from_evidence(user_input, evidence)

    # =====================================================
    # 10. GRAPH
    # =====================================================

    G = build_evidence_graph(all_evidence)

    temporal_signal = compute_temporal_trend(all_evidence)
    G = apply_temporal_weighting(G, temporal_signal)

    graph_signal = graph_confidence_adjustment(G)

    if len(G.nodes) > 1:
        render_graph(G, filename="graph_latest.html")
    else:
        print("\n[Graph skipped: not enough structure]\n")

    # =====================================================
    # 11. HEATMAP
    # =====================================================

    print("\n================ CROWD ACTIVITY HEATMAP ================\n")
    render_subreddit_heatmap(evidence)

    # =====================================================
    # 12. FINAL SCORE
    # =====================================================

    final_confidence = max(
        0.0,
        min(1.0, base_confidence + (graph_signal * 0.2))
    )

    store.add_confidence_snapshot(final_confidence)
    render_timeline(store.get_timeline())

    # =====================================================
    # 13. OUTPUT
    # =====================================================

    print("\n================ CONFIDENCE RESULT ================\n")
    print(f"Base Confidence: {base_confidence:.3f}")
    print(f"Graph Signal: {graph_signal:.3f}")

    print("\n================ FINAL CONFIDENCE ================\n")
    print(f"{final_confidence:.3f}")

    print("\n================ CLAIM ACCURACY SUMMARY ================\n")
    print(f"Claim: {user_input}")
    print(f"Claim Accuracy / Trust Score: {final_confidence:.3f}")

    print("\nExplanation:\n")
    print(explanation)

    print("\n================ SUMMARY ================\n")
    print(f"Total Evidence: {len(evidence)}")
    print(f"Graph Nodes: {G.number_of_nodes()}")
    print(f"Graph Edges: {G.number_of_edges()}")

    return {
        "evidence": evidence,
        "graph": G,
        "events": store.get_all(),
        "base_confidence": base_confidence,
        "graph_signal": graph_signal,
        "final_confidence": final_confidence,
    }


if __name__ == "__main__":
    run_pipeline()