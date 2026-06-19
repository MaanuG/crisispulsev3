from collections import Counter
import matplotlib.pyplot as plt


# =========================================================
# SUBREDDIT ACTIVITY HEATMAP
# =========================================================

def render_subreddit_heatmap(evidence):
    """
    Shows where discussion / signals are concentrated across Reddit.
    This is your "crowd intensity" layer.
    """

    if not evidence:
        print("[Heatmap skipped: no evidence]")
        return

    # -----------------------------------------------------
    # COUNT POSTS PER SUBREDDIT
    # -----------------------------------------------------

    counts = Counter()

    for e in evidence:
        subreddit = getattr(e, "subreddit", "unknown")
        counts[subreddit] += 1

    if not counts:
        print("[Heatmap skipped: no subreddit data]")
        return

    # -----------------------------------------------------
    # SORT FOR CLEAN VISUALIZATION
    # -----------------------------------------------------

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    labels = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    plt.title("Subreddit Activity Heatmap (Crowd Signal Intensity)")
    plt.xlabel("Subreddit")
    plt.ylabel("Number of Evidence Items")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.show()