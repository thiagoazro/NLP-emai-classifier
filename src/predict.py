# src/predict_pipeline.py
import joblib
import pandas as pd
from pathlib import Path

SENT_MODEL = "models/sentiment.joblib"

OUT = Path("data/final_triage.csv")

def load_models():
    sentiment = joblib.load(SENT_MODEL)
    return sentiment

def extract_entities(nlp, text):
    if not nlp:
        return []
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char})
    return ents

def main(in_csv="data/unified_clean.csv", out_csv=OUT):
    df = pd.read_csv(in_csv)
    sentiment = load_models()
    df['text_clean'] = df.get('text_clean', df['text'])
    preds_sent = sentiment.predict(df['text_clean'].tolist())
    probs_sent = sentiment.predict_proba(df['text_clean'].tolist())
    df['sentiment'] = preds_sent
    df['sentiment_scores'] = [dict(zip(sentiment.classes_, p)) for p in probs_sent]
    df.to_csv(out_csv, index=False)
    print("predictions saved to", out_csv)

if __name__ == "__main__":
    main()