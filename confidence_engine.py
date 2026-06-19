# confidence_engine.py


# =========================================================
# STEP 6: STANCE-AWARE CONFIDENCE MODEL
# =========================================================

def _extract_claim_facets(user_text):
    text = user_text.lower()

    return {
        "casualties": any(word in text for word in ["casualty", "casualties", "death", "deaths", "dead", "injury", "injuries", "fatal", "fatalities", "wounded", "killed", "hurt"]),
        "evacuation": any(word in text for word in ["evacuat", "shelter", "order", "warning", "relief"]),
        "fire": any(word in text for word in ["fire", "wildfire", "smoke", "contain", "blaze"]),
        "flood": any(word in text for word in ["flood", "water", "inundat", "overflow", "river"]),
        "health": any(word in text for word in ["water", "contamin", "illness", "outbreak", "boil"]),
        "transport": any(word in text for word in ["road", "highway", "rail", "transit", "bridge", "traffic", "closure"]),
        "quake": any(word in text for word in ["earthquake", "quake", "aftershock", "tsunami"]),
    }


def _evidence_matches_facets(evidence_text, facets):
    text = evidence_text.lower()

    if facets["casualties"]:
        support_terms = ["casualty", "casualties", "death", "deaths", "injur", "fatal", "wounded", "killed", "hospital"]
        contradiction_terms = ["no casualties", "no deaths", "no injuries", "no one injured", "not injured", "none reported"]
        if any(term in text for term in contradiction_terms):
            return "CONTRADICT", 1.35
        if any(term in text for term in support_terms):
            return "SUPPORT", 1.15
        return "NEUTRAL", 0.65

    if facets["evacuation"]:
        support_terms = ["evacuat", "shelter", "order", "warning", "leave", "displace"]
        contradiction_terms = ["safe to stay", "remain indoors", "lifted", "no evacuation", "not evacuating"]
        if any(term in text for term in contradiction_terms):
            return "CONTRADICT", 1.2
        if any(term in text for term in support_terms):
            return "SUPPORT", 1.1
        return "NEUTRAL", 0.7

    if facets["fire"]:
        support_terms = ["fire", "smoke", "contain", "blaze", "burn", "evacuat", "damage"]
        contradiction_terms = ["contained", "not spreading", "no fire", "out", "safe"]
        if any(term in text for term in contradiction_terms):
            return "CONTRADICT", 1.1
        if any(term in text for term in support_terms):
            return "SUPPORT", 1.05
        return "NEUTRAL", 0.75

    if facets["flood"]:
        support_terms = ["flood", "water", "river", "overflow", "inundat", "damage"]
        contradiction_terms = ["stable", "passed", "dried", "no flooding", "safe"]
        if any(term in text for term in contradiction_terms):
            return "CONTRADICT", 1.1
        if any(term in text for term in support_terms):
            return "SUPPORT", 1.05
        return "NEUTRAL", 0.75

    if facets["health"]:
        support_terms = ["contamin", "advisory", "illness", "outbreak", "boil water", "unsafe"]
        contradiction_terms = ["safe to drink", "no unusual illness", "negative", "clear"]
        if any(term in text for term in contradiction_terms):
            return "CONTRADICT", 1.1
        if any(term in text for term in support_terms):
            return "SUPPORT", 1.05
        return "NEUTRAL", 0.75

    if facets["transport"]:
        support_terms = ["closed", "suspended", "delay", "traffic", "closure", "avoid", "blocked"]
        contradiction_terms = ["open", "normal", "cleared", "unaffected", "reopened"]
        if any(term in text for term in contradiction_terms):
            return "CONTRADICT", 1.1
        if any(term in text for term in support_terms):
            return "SUPPORT", 1.05
        return "NEUTRAL", 0.75

    if facets["quake"]:
        support_terms = ["earthquake", "quake", "aftershock", "tsunami", "damage", "shaking"]
        contradiction_terms = ["no damage", "no casualties", "none reported", "minor"]
        if any(term in text for term in contradiction_terms):
            return "CONTRADICT", 1.1
        if any(term in text for term in support_terms):
            return "SUPPORT", 1.05
        return "NEUTRAL", 0.8

    return None, 1.0


def _source_weight(source_type):
    if source_type == "reddit_comment":
        return 0.7
    if source_type == "reddit_post":
        return 1.0
    return 0.85


def _evidence_direction(user_text, ev, facets):
    facet_label, facet_multiplier = _evidence_matches_facets(getattr(ev, "text", ""), facets)

    source_type = getattr(ev, "source_type", "reddit_post")
    stance_weight = max(getattr(ev, "stance_confidence", 0.5), 0.35)
    relevance = max(getattr(ev, "relevance_score", 0.0), 0.1)
    weight = relevance * _source_weight(source_type) * stance_weight * facet_multiplier

    if facet_label == "SUPPORT" or getattr(ev, "stance", None) == "SUPPORT":
        direction = "SUPPORT"
    elif facet_label == "CONTRADICT" or getattr(ev, "stance", None) == "CONTRADICT":
        direction = "CONTRADICT"
    else:
        direction = "NEUTRAL"

    return direction, weight


def _summarize_examples(items, limit=2):
    lines = []

    for direction, weight, ev in items[:limit]:
        text = getattr(ev, "text", "").replace("\n", " ").strip()
        if len(text) > 180:
            text = text[:177] + "..."

        source_type = getattr(ev, "source_type", "reddit_post")
        subreddit = getattr(ev, "subreddit", "unknown")
        lines.append(
            f"- {direction} ({source_type}, r/{subreddit}, weight {weight:.2f}): {text}"
        )

    return lines


def _build_holistic_explanation(user_text, evidence_list, support_ratio, contradict_ratio, neutral_ratio, confidence):
    facets = _extract_claim_facets(user_text)

    scored = []
    for ev in evidence_list:
        direction, weight = _evidence_direction(user_text, ev, facets)
        scored.append((direction, weight, ev))

    support_items = sorted([item for item in scored if item[0] == "SUPPORT"], key=lambda x: x[1], reverse=True)
    contradict_items = sorted([item for item in scored if item[0] == "CONTRADICT"], key=lambda x: x[1], reverse=True)
    neutral_items = sorted([item for item in scored if item[0] == "NEUTRAL"], key=lambda x: x[1], reverse=True)

    if confidence >= 0.75:
        verdict = "Likely true."
    elif confidence >= 0.55:
        verdict = "Mixed evidence."
    elif confidence >= 0.35:
        verdict = "Likely false."
    else:
        verdict = "Very likely false."

    facet_labels = []
    for label, enabled in facets.items():
        if enabled:
            facet_labels.append(label)

    if facet_labels:
        facet_line = ", ".join(facet_labels)
    else:
        facet_line = "general crisis relevance"

    explanation_lines = [
        f"Conclusion: {verdict}",
        f"Claim focus: {facet_line}",
        f"Support signal: {support_ratio:.3f}",
        f"Contradiction signal: {contradict_ratio:.3f}",
        f"Neutral signal: {neutral_ratio:.3f}",
    ]

    if contradict_items and confidence < 0.55:
        explanation_lines.append(
            "Why: the strongest claim-specific Reddit items mostly contradict the user claim, especially where posts explicitly say no casualties or no injuries were reported."
        )
    elif support_items and confidence >= 0.55:
        explanation_lines.append(
            "Why: the strongest claim-specific Reddit items mostly reinforce the user claim with direct mentions of the event details."
        )
    else:
        explanation_lines.append(
            "Why: the evidence is split between direct matches, indirect matches, and unrelated discussion, so the score remains cautious."
        )

    if support_items:
        explanation_lines.append("Strongest supporting evidence:")
        explanation_lines.extend(_summarize_examples(support_items, limit=2))

    if contradict_items:
        explanation_lines.append("Strongest contradicting evidence:")
        explanation_lines.extend(_summarize_examples(contradict_items, limit=2))

    if neutral_items:
        explanation_lines.append("Most relevant neutral evidence:")
        explanation_lines.extend(_summarize_examples(neutral_items, limit=1))

    explanation_lines.append(
        f"Final confidence = {confidence:.3f} after combining relevance, source reliability, stance confidence, and claim-specific facet matches."
    )

    return "\n".join(explanation_lines)


def compute_confidence_from_evidence(user_text, evidence_list):
    """
    Computes a global confidence score for a user claim
    based on ALL Reddit evidence.
    """

    support_score = 0.0
    contradict_score = 0.0
    neutral_score = 0.0

    total_weight = 0.0
    facets = _extract_claim_facets(user_text)

    # =====================================================
    # AGGREGATE SIGNALS
    # =====================================================

    for ev in evidence_list:

        source_type = getattr(ev, "source_type", "reddit_post")

        source_weight = 1.0
        if source_type == "reddit_comment":
            source_weight = 0.7
        elif source_type == "reddit_post":
            source_weight = 1.0
        else:
            source_weight = 0.85

        weight = max(ev.relevance_score, 0.1) * source_weight
        stance_weight = max(getattr(ev, "stance_confidence", 0.5), 0.35)
        weight *= stance_weight

        facet_label, facet_multiplier = _evidence_matches_facets(getattr(ev, "text", ""), facets)
        weight *= facet_multiplier
        total_weight += weight

        if facet_label == "SUPPORT" or ev.stance == "SUPPORT":
            support_score += weight

        elif facet_label == "CONTRADICT" or ev.stance == "CONTRADICT":
            contradict_score += weight

        else:
            neutral_score += weight * 0.15  # weak signal

    if total_weight == 0:
        return 0.5, "No evidence available → neutral confidence"

    # =====================================================
    # NORMALIZE
    # =====================================================

    support_ratio = support_score / total_weight
    contradict_ratio = contradict_score / total_weight

    neutral_ratio = neutral_score / total_weight

    # =====================================================
    # FINAL CONFIDENCE LOGIC
    # =====================================================

    # base confidence starts conservative and only rises with strong support
    confidence = 0.4

    # SUPPORT pushes up confidence
    confidence += support_ratio * 0.5

    # CONTRADICTION pushes down confidence
    confidence -= contradict_ratio * 0.95

    # neutral slightly stabilizes
    confidence += neutral_ratio * 0.03

    # clamp
    confidence = max(0.0, min(1.0, confidence))

    # =====================================================
    # EXPLANATION (AUDITABLE)
    # =====================================================

    explanation = _build_holistic_explanation(
        user_text,
        evidence_list,
        support_ratio,
        contradict_ratio,
        neutral_ratio,
        confidence,
    )

    return round(confidence, 3), explanation


# =========================================================
# BATCH WRAPPER (OPTIONAL FUTURE USE)
# =========================================================

def compute_and_attach_confidence(evidence_list):

    confidence, explanation = compute_confidence_from_evidence(evidence_list)

    return confidence, explanation