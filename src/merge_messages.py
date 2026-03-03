# src/merge_messages.py
from pathlib import Path
import pandas as pd
from datetime import datetime, UTC

STAGING = Path("data/staging/email")
OUT_CSV = Path("data/unified_inbox.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

EXPECTED_COLS = ["id", "channel", "from", "to", "subject", "text", "received_at", "message_id"]

def load_all_staging():
    rows = []
    for p in STAGING.glob("*.csv"):
        try:
            df_email = pd.read_csv(p, dtype=str)
            if df_email.empty:
                continue
            for _, row in df_email.iterrows():
                rows.append({
                    "id":          row.get("id", ""),
                    "channel":     row.get("channel", "email"),
                    "from":        row.get("from", ""),
                    "to":          row.get("to", ""),
                    "subject":     row.get("subject", ""),
                    "text":        row.get("text", ""),
                    "received_at": row.get("received_at", datetime.now(UTC).isoformat()),
                    "message_id":  row.get("message_id", ""),
                    "raw_file":    str(p),
                })
        except Exception as e:
            print("erro lendo", p, e)
    return rows

def deduplicate(df):
    return df.drop_duplicates(subset=["id"])

def main():
    rows = load_all_staging()
    df = pd.DataFrame(rows)
    if df.empty:
        print("nenhuma mensagem no staging.")
        return
    df = deduplicate(df)
    df.to_csv(OUT_CSV, index=False)
    print("unified csv salvo em", OUT_CSV)

if __name__ == "__main__":
    main()