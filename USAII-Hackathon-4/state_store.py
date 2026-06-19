from typing import List, Dict, Any
from data_models import EvidenceClaim
from datetime import datetime


class TemporalEvidenceStore:
    """
    Stores:
    1. Evidence history (graph memory)
    2. Confidence timeline (for Step 11.2 visualization)
    """

    def __init__(self):
        self.history: List[EvidenceClaim] = []
        self.confidence_history: List[Dict[str, Any]] = []

    # =====================================================
    # EVIDENCE MEMORY
    # =====================================================

    def add_batch(self, new_evidence: List[EvidenceClaim]):
        self.history.extend(new_evidence)

    def get_all(self):
        return self.history

    def get_recent(self, window=50):
        return self.history[-window:]

    # =====================================================
    # CONFIDENCE TIMELINE (STEP 11.2 CORE)
    # =====================================================

    def add_confidence_snapshot(self, value: float):
        self.confidence_history.append({
            "timestamp": datetime.now(),
            "confidence": value
        })

    def get_timeline(self):
        return self.confidence_history