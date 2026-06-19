from keybert import KeyBERT
import spacy

# =========================================================
# LOAD MODELS
# =========================================================

_nlp = spacy.load("en_core_web_sm")
_kw_model = KeyBERT()


# =========================================================
# CLAIM PARSER
# =========================================================

def parse_claim(user_text: str):

    doc = _nlp(user_text)

    # =====================================================
    # NAMED ENTITIES
    # =====================================================

    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    # =====================================================
    # KEYPHRASE EXTRACTION
    # =====================================================

    keywords = _kw_model.extract_keywords(
        user_text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=10
    )

    phrases = [k[0] for k in keywords]

    # =====================================================
    # IMPORTANT TOKENS
    # =====================================================

    important_tokens = []

    for token in doc:

        if token.is_stop:
            continue

        if token.is_punct:
            continue

        if len(token.text) < 3:
            continue

        if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
            important_tokens.append(token.lemma_.lower())

    important_tokens = list(set(important_tokens))

    return {
        "entities": entities,
        "phrases": phrases,
        "keywords": important_tokens
    }