import networkx as nx


# =========================================================
# GRAPH CONSTRUCTION (BELIEF NETWORK)
# =========================================================

def build_evidence_graph(evidence_list):
    """
    Builds a belief interaction graph.

    Nodes:
        Individual evidence claims

    Edges:
        SUPPORT relationships
        CONTRADICTION relationships
    """

    G = nx.Graph()

    # =====================================================
    # ADD NODES
    # =====================================================

    for i, ev in enumerate(evidence_list):

        node_id = getattr(ev, "source_id", f"node_{i}")

        G.add_node(
            node_id,
            text=getattr(ev, "text", ""),
            stance=getattr(ev, "stance", "NEUTRAL"),
            relevance=getattr(ev, "relevance_score", 0.0),
            confidence=getattr(ev, "confidence_score", 0.0),
            subreddit=getattr(ev, "subreddit", "unknown"),
            stability=getattr(ev, "stance_confidence", 0.0),
            ranking_score=getattr(ev, "ranking_score", 0.0),
        )

    # =====================================================
    # ADD EDGES (BELIEF RELATIONSHIPS)
    # =====================================================

    n = len(evidence_list)

    for i in range(n):
        for j in range(i + 1, n):

            a = evidence_list[i]
            b = evidence_list[j]

            # -------------------------------------------------
            # SAFETY EXTRACTION
            # -------------------------------------------------

            a_rel = getattr(a, "relevance_score", 0.0)
            b_rel = getattr(b, "relevance_score", 0.0)

            a_stance = getattr(a, "stance", None)
            b_stance = getattr(b, "stance", None)

            # -------------------------------------------------
            # FILTER 1: IGNORE WEAK SIGNALS
            # -------------------------------------------------

            if a_rel < 0.25 or b_rel < 0.25:
                continue

            # -------------------------------------------------
            # FILTER 2: IGNORE LOW-CONFIDENCE STANCE
            # -------------------------------------------------

            a_conf = getattr(a, "stance_confidence", 0.0)
            b_conf = getattr(b, "stance_confidence", 0.0)

            if a_conf < 0.45 or b_conf < 0.45:
                continue

            # -------------------------------------------------
            # FILTER 3: IGNORE NEUTRAL-NEUTRAL EDGES
            # -------------------------------------------------

            if a_stance == "NEUTRAL" and b_stance == "NEUTRAL":
                continue

            # -------------------------------------------------
            # FILTER 4: PREVENT GIANT FULLY-CONNECTED MESH
            # -------------------------------------------------

            relevance_gap = abs(a_rel - b_rel)

            if relevance_gap < 0.03:
                continue

            # -------------------------------------------------
            # EDGE WEIGHT
            # -------------------------------------------------

            weight = (a_rel + b_rel) / 2

            # slight confidence amplification
            weight *= (1.0 + ((a_conf + b_conf) / 2) * 0.25)

            # clamp
            weight = max(0.0, min(1.0, weight))

            # -------------------------------------------------
            # CONTRADICTION EDGE
            # -------------------------------------------------

            if (
                    a_stance != b_stance and
                    "NEUTRAL" not in (a_stance, b_stance)
            ):

                G.add_edge(
                    getattr(a, "source_id", f"a_{i}"),
                    getattr(b, "source_id", f"b_{j}"),
                    type="CONTRADICT",
                    weight=weight
                )

            # -------------------------------------------------
            # SUPPORT EDGE
            # -------------------------------------------------

            elif (
                    a_stance == b_stance and
                    a_stance != "NEUTRAL"
            ):

                G.add_edge(
                    getattr(a, "source_id", f"a_{i}"),
                    getattr(b, "source_id", f"b_{j}"),
                    type="SUPPORT",
                    weight=weight
                )

    return G


# =========================================================
# GRAPH-BASED CONFIDENCE SIGNAL
# =========================================================

def graph_confidence_adjustment(G):
    """
    Computes overall belief pressure from graph structure.
    """

    support_weight = 0.0
    contradiction_weight = 0.0

    for _, _, data in G.edges(data=True):

        edge_type = data.get("type")
        weight = data.get("weight", 0.0)

        if edge_type == "SUPPORT":
            support_weight += weight

        elif edge_type == "CONTRADICT":
            contradiction_weight += weight

    total = support_weight + contradiction_weight

    if total <= 1e-8:
        return 0.0

    support_ratio = support_weight / total
    contradiction_ratio = contradiction_weight / total

    signal = support_ratio - contradiction_ratio

    return float(max(-1.0, min(1.0, signal)))


# =========================================================
# TEMPORAL WEIGHTING (TREND AMPLIFICATION)
# =========================================================

def apply_temporal_weighting(G, temporal_signal):
    """
    Boosts or dampens edge strengths based on temporal trend signals.
    """

    for _, _, data in G.edges(data=True):

        key = data.get("type", "SUPPORT")

        trend_boost = (
                temporal_signal.get(key)
                or temporal_signal.get(f"reddit_{key}")
                or 0.0
        )

        base_weight = data.get("weight", 0.0)

        adjusted = base_weight * (1.0 + trend_boost)

        data["weight"] = max(0.0, min(1.0, adjusted))

    return G


# =========================================================
# GRAPH STABILITY
# =========================================================

def compute_graph_stability(G):
    """
    Measures structural stability of belief network.

    More contradiction density -> lower stability.
    """

    edge_count = G.number_of_edges()

    if edge_count == 0:
        return 1.0

    contradiction_edges = sum(
        1
        for _, _, d in G.edges(data=True)
        if d.get("type") == "CONTRADICT"
    )

    contradiction_ratio = contradiction_edges / edge_count

    stability = 1.0 - contradiction_ratio

    return float(max(0.0, min(1.0, stability)))