# src/clean_and_annotate.py
import pandas as pd
from pathlib import Path
from preprocessing import batch_preprocess

IN = Path("data/unified_inbox.csv")
OUT = Path("data/unified_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main(in_path=IN, out_path=OUT):
    df = pd.read_csv(in_path, dtype=str)
    df['subject'] = df['subject'].fillna("")
    df['text'] = df['text'].fillna("")
    # combina assunto + corpo para enriquecer o contexto NLP
    combined = (df['subject'] + ". " + df['text']).astype(str).tolist()
    df['text_clean'] = batch_preprocess(combined)
    df.to_csv(out_path, index=False)
    print("clean saved to", out_path)

if __name__ == "__main__":
    main()