from sklearn.cluster import DBSCAN
import numpy as np


def cluster_events(embeddings):

    clustering = DBSCAN(
        eps=0.3,
        min_samples=2,
        metric="cosine"
    )

    return clustering.fit_predict(embeddings)


def group_events(evidence, embeddings):

    labels = cluster_events(embeddings)

    events = {}

    for i, label in enumerate(labels):

        if label == -1:
            continue

        if label not in events:
            events[label] = []

        events[label].append(evidence[i])

    return events