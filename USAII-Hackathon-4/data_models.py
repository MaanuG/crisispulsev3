from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

# =========================================================
# CORE DATA STRUCTURE: EVIDENCE CLAIM
# =========================================================

@dataclass
class EvidenceClaim:
    """
    A single atomic piece of information extracted from Reddit
    or future sources (news, APIs, etc.).
    """

    text: str
    source_type: str          # "reddit_post" | "reddit_comment" | "news" | etc.
    source_id: str

    subreddit: Optional[str] = None
    parent_id: Optional[str] = None

    # =====================================================
    # STEP 5+ (semantic enrichment fields)
    # =====================================================

    relevance_score: float = 0.0
    stance: Optional[str] = None   # "SUPPORT" | "CONTRADICT" | "NEUTRAL"

    # =====================================================
    # STEP 6+ (confidence + reasoning)
    # =====================================================

    confidence_score: float = 0.0
    explanation: str = ""
    timestamp: datetime = datetime.utcnow()


# =========================================================
# OPTIONAL: CLAIM (HIGHER-LEVEL USER INPUT STRUCTURE)
# =========================================================

@dataclass
class UserClaim:
    """
    Represents the user's input statement/query.
    This is what evidence is evaluated against.
    """

    text: str
    entity: Optional[str] = None


# =========================================================
# HELPERS
# =========================================================

def to_dict_list(items: List[EvidenceClaim]) -> List[Dict]:
    """
    Converts EvidenceClaim objects into dictionaries for pipeline processing.
    """
    return [item.__dict__ for item in items]