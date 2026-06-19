import praw
from data_models import EvidenceClaim
import re


# =========================================================
# REDDIT CLIENT
# =========================================================

reddit = praw.Reddit(
    client_id="4hNxpEgMZrRN0FlQ-xfA4w",
    client_secret="tg5C4_6V5biEnZBzhl0lIfhxhxiRUA",
    user_agent="windows:MySentimentExample:v1.0 (by u/Previous-Abies-7216)"
)



# =========================================================
# 1. FETCH POSTS
# =========================================================

def fetch_posts(query: str, limit: int = 20):

    return reddit.subreddit("all").search(
        query,
        sort="relevance",
        limit=limit
    )


def _tokenize(text: str):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def post_matches_query(post, query: str) -> bool:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return True

    post_text = f"{getattr(post, 'title', '')} {getattr(post, 'selftext', '')}".lower()
    if query.lower() in post_text:
        return True

    post_tokens = _tokenize(post_text)
    overlap = len(query_tokens & post_tokens)

    if len(query_tokens) <= 2:
        return overlap >= 1

    return overlap >= max(2, len(query_tokens) // 3)


# =========================================================
# 2. CONVERT POST → EVIDENCE CLAIMS
# =========================================================

def post_to_evidence(post):

    evidence = []

    # -------------------------
    # POST AS CLAIM
    # -------------------------
    evidence.append(
        EvidenceClaim(
            text=post.title,
            source_type="reddit_post",
            source_id=post.id,
            subreddit=str(post.subreddit)
        )
    )

    # -------------------------
    # COMMENTS AS CLAIMS
    # -------------------------
    post.comments.replace_more(limit=0)

    comments = [c for c in post.comments.list() if hasattr(c, "body")]
    comments.sort(key=lambda c: getattr(c, "score", 0), reverse=True)

    for comment in comments[:5]:

        evidence.append(
            EvidenceClaim(
                text=comment.body,
                source_type="reddit_comment",
                source_id=comment.id,
                subreddit=str(post.subreddit),
                parent_id=post.id
            )
            )
    return evidence


# =========================================================
# 3. BUILD FULL EVIDENCE STREAM
# =========================================================

def build_evidence_stream(query: str):

    posts = fetch_posts(query)

    all_evidence = []

    for post in posts:
        if not post_matches_query(post, query):
            continue

        all_evidence.extend(post_to_evidence(post))

    return all_evidence