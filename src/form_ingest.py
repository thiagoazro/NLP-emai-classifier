# src/form_ingest.py
"""
Ingestão de respostas do Typeform para o pipeline NLP.
Substitui email_ingest_imap.py — grava no mesmo staging com o mesmo schema.

Variáveis de ambiente necessárias (.env):
    TYPEFORM_TOKEN   = Personal Access Token do Typeform
    TYPEFORM_FORM_ID = ID do formulário (aparece na URL do form)

Refs dos campos no Typeform (ajuste no .env conforme o que você configurou):
    FIELD_NOME      = ref do campo Nome      (default: "nome")
    FIELD_EMAIL     = ref do campo Email     (default: "email")
    FIELD_ASSUNTO   = ref do campo Assunto   (default: "assunto")
    FIELD_MENSAGEM  = ref do campo Mensagem  (default: "mensagem")

Execução:
    python src/form_ingest.py
    python src/form_ingest.py --max 200
    python src/form_ingest.py --reset-cursor   # reingere tudo desde o início
"""
import os
import uuid
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# --- configuração ---
STAGING     = Path("data/staging/email")
CURSOR_FILE = Path("data/staging/.typeform_cursor")   # guarda o token da última resposta processada
STAGING.mkdir(parents=True, exist_ok=True)

TOKEN   = os.getenv("TYPEFORM_TOKEN")
FORM_ID = os.getenv("TYPEFORM_FORM_ID")

# refs dos campos — devem bater com o que está configurado no painel do Typeform
FIELD_NOME     = os.getenv("FIELD_NOME",     "nome")
FIELD_EMAIL    = os.getenv("FIELD_EMAIL",    "email")
FIELD_ASSUNTO  = os.getenv("FIELD_ASSUNTO",  "assunto")
FIELD_MENSAGEM = os.getenv("FIELD_MENSAGEM", "mensagem")

BASE_URL = "https://api.typeform.com"


# --- helpers ---

def get_answer(answers: list, field_ref: str) -> str:
    """Extrai o valor de uma resposta pelo ref do campo."""
    for ans in answers:
        if ans.get("field", {}).get("ref") != field_ref:
            continue
        ans_type = ans.get("type", "")
        if ans_type == "text":
            return ans.get("text", "")
        elif ans_type == "choice":
            return ans.get("choice", {}).get("label", "")
        elif ans_type == "choices":
            labels = ans.get("choices", {}).get("labels", [])
            return ", ".join(labels)
        elif ans_type == "email":
            return ans.get("email", "")
        elif ans_type == "number":
            return str(ans.get("number", ""))
        else:
            return str(ans.get(ans_type, ""))
    return ""


def load_cursor() -> str | None:
    """Carrega o response_id da última resposta processada (cursor de paginação)."""
    if CURSOR_FILE.exists():
        val = CURSOR_FILE.read_text().strip()
        return val or None
    return None


def save_cursor(response_id: str):
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURSOR_FILE.write_text(response_id)


def fetch_responses(after_token: str = None, page_size: int = 100) -> dict:
    """
    Busca respostas do Typeform.
    after_token: response_id da última resposta já processada (cursor).
                 Typeform retorna apenas respostas DEPOIS desse token.
    """
    if not TOKEN:
        raise RuntimeError("Defina TYPEFORM_TOKEN no .env")
    if not FORM_ID:
        raise RuntimeError("Defina TYPEFORM_FORM_ID no .env")

    params = {
        "page_size": min(page_size, 1000),
        "sort": "submitted_at,asc",
    }
    if after_token:
        params["after"] = after_token

    resp = requests.get(
        f"{BASE_URL}/forms/{FORM_ID}/responses",
        headers={"Authorization": f"Bearer {TOKEN}"},
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# --- ingestão principal ---

def ingest(max_responses: int = 100, reset_cursor: bool = False):
    after = None if reset_cursor else load_cursor()

    if reset_cursor:
        print("Cursor resetado — reingerindo todas as respostas.")
    elif after:
        print(f"Buscando respostas após o token: {after}")
    else:
        print("Primeira execução — buscando todas as respostas disponíveis.")

    data  = fetch_responses(after_token=after, page_size=max_responses)
    items = data.get("items", [])
    total = data.get("total_items", len(items))

    if not items:
        print("Nenhuma resposta nova encontrada.")
        return

    print(f"{len(items)} resposta(s) encontrada(s) (total no form: {total})")

    ingested       = 0
    last_token     = None

    for item in items:
        try:
            response_id  = item.get("response_id", str(uuid.uuid4()))
            submitted_at = item.get("submitted_at", datetime.now(timezone.utc).isoformat())
            answers      = item.get("answers", [])

            nome     = get_answer(answers, FIELD_NOME)
            email    = get_answer(answers, FIELD_EMAIL)
            assunto  = get_answer(answers, FIELD_ASSUNTO)
            mensagem = get_answer(answers, FIELD_MENSAGEM)

            # texto principal para o NLP — junta assunto + mensagem
            text = f"{assunto}: {mensagem}".strip() if mensagem else assunto

            row = {
                "id":          f"form-{response_id}",
                "channel":     "formulario",
                "from":        f"{nome} <{email}>" if email else nome,
                "to":          "",
                "subject":     assunto,
                "text":        text,
                "received_at": submitted_at,
                "message_id":  email,         # email fica em message_id
            }

            fname = STAGING / f"form-{response_id}.csv"
            pd.DataFrame([row]).to_csv(fname, index=False, encoding="utf-8")
            ingested  += 1
            last_token = response_id

        except Exception as e:
            print(f"  Erro na resposta {item.get('response_id', '?')}: {e}")
            continue

    if last_token:
        save_cursor(last_token)

    print(f"Ingestão concluída: {ingested} resposta(s) salvas em {STAGING}")


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingere respostas do Typeform para o pipeline NLP.")
    parser.add_argument("--max",          type=int,            default=100,  help="máximo de respostas por execução (default: 100)")
    parser.add_argument("--reset-cursor", action="store_true",               help="ignora o cursor e reingere tudo desde o início")
    args = parser.parse_args()

    ingest(max_responses=args.max, reset_cursor=args.reset_cursor)
