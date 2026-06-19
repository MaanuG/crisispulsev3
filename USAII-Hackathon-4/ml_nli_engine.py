from transformers import pipeline

# Load once (important for performance)
nli_model = pipeline(
    "text-classification",
    model="roberta-large-mnli"
)


def get_stance_nli(user_text: str, evidence_text: str):
    """
    Transformer-based Natural Language Inference:
    returns SUPPORT / CONTRADICT / NEUTRAL
    """

    result = nli_model({
        "text": evidence_text,
        "text_pair": user_text
    })[0]

    label = result["label"]
    score = result["score"]

    if label == "ENTAILMENT":
        return "SUPPORT", score
    elif label == "CONTRADICTION":
        return "CONTRADICT", score
    else:
        return "NEUTRAL", score