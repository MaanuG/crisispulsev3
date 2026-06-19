from reddit_source import build_evidence_stream


def fetch_all_sources(query: str):
    """
    Unified ingestion layer.
    Later: add news API, RSS, Twitter, etc.
    """

    reddit_evidence = build_evidence_stream(query)

    # placeholder for future sources
    news_evidence = []

    return reddit_evidence + news_evidence