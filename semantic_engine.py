from typing import List
from data_models import EvidenceClaim
import numpy as np
import re
import torch

# =========================================================
# ML MODELS (LAZY LOADED)
# =========================================================

_embedding_model = None
_nli_model = None
_nli_tokenizer = None


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

def _load_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embedding_model


# =========================================================
# LOAD NLI MODEL (BART MNLI)
# =========================================================

def _load_nli_model():
    global _nli_model, _nli_tokenizer

    if _nli_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        candidate_models = [
            "microsoft/deberta-base-mnli",
            "facebook/bart-large-mnli",
        ]

        last_error = None

        for model_name in candidate_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)

                _nli_tokenizer = tokenizer
                _nli_model = model
                break

            except Exception as error:
                last_error = error

        if _nli_model is None or _nli_tokenizer is None:
            raise RuntimeError("Unable to load an NLI model for stance scoring") from last_error

    return _nli_model


def _predict_stance_scores(user_text: str, evidence_text: str):
    model = _load_nli_model()
    tokenizer = _nli_tokenizer

    inputs = tokenizer(
        user_text,
        evidence_text,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits[0], dim=-1).cpu().numpy()

    label_map = {
        str(v).upper(): int(k)
        for k, v in model.config.id2label.items()
    }

    contradiction_idx = label_map.get("CONTRADICTION", 0)
    neutral_idx = label_map.get("NEUTRAL", 1)
    entailment_idx = label_map.get("ENTAILMENT", 2)

    return {
        "CONTRADICT": float(probabilities[contradiction_idx]),
        "NEUTRAL": float(probabilities[neutral_idx]),
        "SUPPORT": float(probabilities[entailment_idx]),
    }


# =========================================================
# STEP 1: SEMANTIC RELEVANCE (EMBEDDINGS)
# =========================================================

def compute_relevance(user_text: str, evidence_list: List[EvidenceClaim]):
    """
    Cosine similarity via sentence embeddings.
    """

    model = _load_embedding_model()

    query_vec = model.encode(user_text, normalize_embeddings=True)

    query_tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))
    query_tokens = {token for token in query_tokens if len(token) > 2}

    texts = [e.text for e in evidence_list]
    vectors = model.encode(texts, normalize_embeddings=True)

    for ev, vec in zip(evidence_list, vectors):
        embedding_score = float(np.dot(query_vec, vec))

        evidence_tokens = set(re.findall(r"[a-z0-9]+", ev.text.lower()))
        evidence_tokens = {token for token in evidence_tokens if len(token) > 2}

        if query_tokens:
            lexical_overlap = len(query_tokens & evidence_tokens) / len(query_tokens)
        else:
            lexical_overlap = 0.0

        score = (0.7 * embedding_score) + (0.3 * lexical_overlap)

        # clamp to [0,1]
        ev.relevance_score = max(0.0, min(1.0, score))

    return evidence_list


# =========================================================
# STEP 2: NLI STANCE DETECTION (FIXED LABEL MAPPING)
# =========================================================

def compute_stance(user_text: str, evidence_list: List[EvidenceClaim]):
    query_tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))
    query_tokens = {token for token in query_tokens if len(token) > 2}

    high_contradiction_threshold = 0.42
    high_support_threshold = 0.45

    for ev in evidence_list:

        try:
            evidence_text = ev.text.lower()
            evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_text))
            evidence_tokens = {token for token in evidence_tokens if len(token) > 2}

            overlap = len(query_tokens & evidence_tokens)
            stance_scores = _predict_stance_scores(user_text, ev.text)

            support_score = stance_scores["SUPPORT"]
            contradict_score = stance_scores["CONTRADICT"]
            neutral_score = stance_scores["NEUTRAL"]

            if overlap == 0 and max(support_score, contradict_score) < high_support_threshold:
                ev.stance = "NEUTRAL"
                ev.stance_confidence = float(neutral_score)

            elif contradict_score >= max(support_score, neutral_score, high_contradiction_threshold):
                ev.stance = "CONTRADICT"
                ev.stance_confidence = float(contradict_score)

            elif support_score >= max(contradict_score, neutral_score, high_support_threshold):
                ev.stance = "SUPPORT"
                ev.stance_confidence = float(support_score)

            else:
                ev.stance = "NEUTRAL"
                ev.stance_confidence = float(max(neutral_score, min(0.6, 0.35 + 0.03 * overlap)))

        except Exception:
            ev.stance = "NEUTRAL"
            ev.stance_confidence = 0.0

    return evidence_list


# =========================================================
# STEP 3: RANKING SCORE (FUSION LAYER)
# =========================================================

def compute_ranking_score(user_text: str, evidence_list: List[EvidenceClaim]):
    """
    Combines relevance + stance confidence.
    """

    for ev in evidence_list:

        stance_conf = getattr(ev, "stance_confidence", 0.5)

        ev.ranking_score = (
                0.7 * ev.relevance_score +
                0.3 * stance_conf
        )

        ev.ranking_score = float(max(0.0, min(1.0, ev.ranking_score)))

    return evidence_list