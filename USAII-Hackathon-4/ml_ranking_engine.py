from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_relevance(user_text: str, evidence_text: str) -> float:
    """
    Semantic similarity ranking (NOT keyword overlap)
    """

    emb1 = model.encode(user_text, convert_to_tensor=True)
    emb2 = model.encode(evidence_text, convert_to_tensor=True)

    return float(util.cos_sim(emb1, emb2)[0][0])