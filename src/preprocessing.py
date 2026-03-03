# src/preprocessing.py
import re
import unicodedata
from typing import List
import spacy

nlp = spacy.load("pt_core_news_sm", disable=["parser","ner"])

URL_OR_EMAIL = re.compile(r"(https?://\S+|\S+@\S+\.\S+)")

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def remove_noise(text: str) -> str:
    text = URL_OR_EMAIL.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[_\-]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    text = normalize_unicode(text)
    text = text.lower()
    text = remove_noise(text)
    return text

def preprocess_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    text = clean_text(text)
    doc = nlp(text)
    tokens = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if remove_stopwords and tok.is_stop:
            continue
        if lemmatize:
            lemma = tok.lemma_.strip()
            if lemma:
                tokens.append(lemma)
        else:
            tokens.append(tok.text)
    return " ".join(tokens)

def batch_preprocess(texts: List[str], **kwargs) -> List[str]:
    return [preprocess_text(t, **kwargs) for t in texts]

if __name__ == "__main__":
    print(preprocess_text("Olá, favor enviar o contrato para fulano@example.com. Prazo 10/03/2026"))