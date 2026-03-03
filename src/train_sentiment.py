# src/train_sentiment.py
import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

_NEGATIVE_KEYWORDS = ['reclamação', 'defeito', 'incorreto', 'problema', 'cancelamento', 'promoção não aplicada', 'ruim', 'insatisfeito', 'devolução', 'atraso', 'cobrança indevida', 'erro', 'demora', 'atendimento ruim']
_POSITIVE_KEYWORDS = ['parabenizo', 'positiva', 'agradável', 'gostei', 'elogio', 'satisfação', 'correto', 'resolvido', 'promoção aplicada', 'bom', 'satisfeito']

def _infer_label(subject: str) -> str:
    s = str(subject).lower()
    if any(k in s for k in _NEGATIVE_KEYWORDS):
        return 'negativo'
    if any(k in s for k in _POSITIVE_KEYWORDS):
        return 'positivo'
    return 'neutro'

def load_data(path: str):
    df = pd.read_csv(path)
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
    if 'label' not in df.columns:
        print("[INFO] Coluna 'label' não encontrada. Inferindo a partir de text_clean...")
        df['label'] = df['text_clean'].apply(_infer_label)
        df.to_csv(path, index=False)
        print(f"[INFO] CSV atualizado com coluna 'label' salvo em {path}")
        print(df['label'].value_counts().to_string())
    return df

def main(data_path, out_model):
    df = load_data(data_path)
    df['text_clean'] = df.get('text_clean', df['text'])
    X = df['text_clean'].astype(str).tolist()
    y = df['label'].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, digits=4))
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(pipe, out_model)
    print(f"Modelo salvo em {out_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV com colunas text,label")
    parser.add_argument("--out", default="models/sentiment.joblib", help="path para salvar modelo")
    args = parser.parse_args()
    main(args.data, args.out)