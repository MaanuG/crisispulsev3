from collections import defaultdict
import numpy as np


def compute_temporal_trend(evidence_list):
    """
    Measures whether a claim is gaining or losing momentum.
    """

    trend = defaultdict(list)

    for ev in evidence_list:

        key = ev.source_type + "_" + ev.stance

        trend[key].append(ev.relevance_score)

    trend_scores = {}

    for k, values in trend.items():

        if len(values) < 2:
            trend_scores[k] = 0.0
            continue

        # slope proxy: recent vs older
        mid = len(values) // 2

        early = np.mean(values[:mid])
        late = np.mean(values[mid:])

        trend_scores[k] = float(late - early)

    return trend_scores