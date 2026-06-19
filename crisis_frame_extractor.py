# crisis_frame_extractor.py

import re
from typing import Dict, List


def extract_entities(text: str) -> List[str]:
    """
    Very simple entity extraction (upgrade later with spaCy if needed)
    """
    # crude but effective starter approach
    candidates = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", text)
    return list(set(candidates))


def infer_event_type(text: str) -> str:
    """
    Lightweight heuristic classifier (NO keyword lists per crisis)
    """
    t = text.lower()

    if any(w in t for w in ["fire", "wildfire", "burn", "smoke"]):
        return "fire_event"

    if any(w in t for w in ["crash", "accident", "collision"]):
        return "transport_accident"

    if any(w in t for w in ["earthquake", "quake", "seismic"]):
        return "earthquake"

    if any(w in t for w in ["protest", "riot", "march"]):
        return "civil_unrest"

    if any(w in t for w in ["scam", "fraud", "hack", "breach"]):
        return "cyber_incident"

    return "unknown_event"


def infer_urgency(text: str) -> str:
    t = text.lower()

    if any(w in t for w in ["breaking", "urgent", "emergency", "evacuation"]):
        return "high"

    if any(w in t for w in ["reportedly", "heard", "allegedly"]):
        return "medium"

    return "low"


def extract_crisis_frame(text: str) -> Dict:
    return {
        "raw_text": text,
        "event_type": infer_event_type(text),
        "entities": extract_entities(text),
        "urgency": infer_urgency(text)
    }