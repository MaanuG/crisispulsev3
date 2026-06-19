from typing import List
import re


def generate_search_queries(user_input: str) -> List[str]:
    """
    Converts user input into structured crisis search queries.
    Output is ALWAYS a list of strings (safe + pipeline compatible).
    """

    text = user_input.lower()

    # -----------------------------------------
    # STEP 1: extract key entities (simple heuristic)
    # -----------------------------------------

    words = re.findall(r"[a-zA-Z]+", text)

    stopwords = {
        "i", "heard", "is", "are", "the", "a", "an",
        "really", "very", "so", "just", "that", "this",
        "of", "in", "on", "for", "with", "to", "be",
        "as", "it", "its", "was", "were", "causing", "cause",
        "huge", "big", "major", "severe", "extreme"
    }

    keywords = [w for w in words if w not in stopwords]

    base = " ".join(keywords).strip()

    # -----------------------------------------
    # STEP 2: targeted expansion templates
    # -----------------------------------------

    crisis_modifiers = {
        "fire": ["smoke", "evacuation", "containment", "crews"],
        "wildfire": ["smoke", "evacuation", "containment", "crews"],
        "earthquake": ["casualties", "damage", "aftershock", "rescue"],
        "quake": ["casualties", "damage", "aftershock", "rescue"],
        "flood": ["evacuation", "damage", "warning", "rescue"],
        "storm": ["evacuation", "damage", "warning", "power outage"],
        "typhoon": ["evacuation", "damage", "warning", "power outage"],
        "hurricane": ["evacuation", "damage", "warning", "power outage"],
        "cyclone": ["evacuation", "damage", "warning", "power outage"],
        "accident": ["closure", "delay", "traffic", "injuries"],
        "explosion": ["injuries", "damage", "casualties", "evacuation"],
        "shooting": ["injuries", "casualties", "police", "lockdown"],
        "water": ["advisory", "contamination", "boil water", "health"],
        "landslide": ["rescue", "road closed", "damage", "casualties"],
        "outbreak": ["cases", "hospital", "advisory", "deaths"],
    }

    templates = [base]

    if len(base.split()) >= 2:
        templates.append(f'"{base}"')

    if base:
        templates.append(f"{base} update")
        templates.append(f"{base} news")

    for keyword, modifiers in crisis_modifiers.items():
        if keyword in keywords:
            for modifier in modifiers:
                templates.append(f"{base} {modifier}")

    if len(templates) == 1:
        templates.append(f"{base} latest")

    # -----------------------------------------
    # STEP 3: deduplicate + clean
    # -----------------------------------------

    cleaned = []
    seen = set()

    for q in templates:
        q = q.strip()

        if len(q.split()) < 2:
            continue

        if q not in seen:
            cleaned.append(q)
            seen.add(q)

    return cleaned[:6]