from typing import List
from data_models import EvidenceClaim

from semantic_engine import (
    compute_relevance as semantic_relevance,
    compute_stance as semantic_stance,
    compute_ranking_score as semantic_ranking
)

# =========================================================
# STEP 7: SEMANTIC RELEVANCE (EMBEDDING + SIMILARITY MODEL)
# =========================================================

def enrich_relevance(user_text: str, evidence_list: List[EvidenceClaim]):
    """
    Computes semantic relevance using embedding-based similarity.
    Mutates EvidenceClaim.relevance_score in-place.
    """

    return semantic_relevance(user_text, evidence_list)


# =========================================================
# STEP 7: NLI-BASED STANCE DETECTION (TRANSFORMER MODEL)
# =========================================================

def enrich_with_stance(user_text: str, evidence_list: List[EvidenceClaim]):
    """
    Computes stance using transformer-based NLI model.
    Mutates EvidenceClaim.stance in-place.
    """

    return semantic_stance(user_text, evidence_list)


# =========================================================
# STEP 7.5: OPTIONAL RANKING LAYER (ML FUSION SIGNAL)
# =========================================================

def enrich_ranking(user_text: str, evidence_list: List[EvidenceClaim]):
    """
    Computes combined ranking score from:
    - relevance (embedding similarity)
    - stance confidence (NLI strength)

    This will be used later for:
    - graph weighting
    - confidence scoring
    - evidence filtering
    """

    return semantic_ranking(user_text, evidence_list)


# =========================================================
# LEGACY FUNCTIONS (DEPRECATED SAFETY FALLBACKS)
# =========================================================

def detect_stance(user_text: str, evidence_text: str) -> str:
    """
    Deprecated: replaced by transformer-based NLI model.
    Always returns NEUTRAL for compatibility.
    """
    return "NEUTRAL"


def compute_relevance_legacy(user_text: str, evidence_text: str) -> float:
    """
    Deprecated: replaced by embedding similarity model.
    Always returns 0.0 for compatibility.
    """
    return 0.0