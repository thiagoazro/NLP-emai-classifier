#!/usr/bin/env python3
"""
Lê data/llm_replies.csv e envia cada linha ao Slack como mensagem estruturada.

Colunas esperadas do CSV (geradas pela pipeline completa):
  id, channel, from, to, subject, text, received_at,
  sentiment, entities, priority, alert, explain_snippet,
  reply_subject, reply_body, reply_explain

Uso:
  source .venv/bin/activate
  SLACK_BOT_TOKEN=xoxb-... SLACK_CHANNEL_ID=C123... python src/send_to_slack_bot.py
  # ou passando CSV explícito:
  python src/send_to_slack_bot.py data/llm_replies.csv --limit 5 --dry
"""

import os
import time
import argparse
from pathlib import Path

import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from tenacity import retry, wait_exponential, stop_after_attempt

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL_ID")

if not SLACK_TOKEN or not SLACK_CHANNEL:
    raise RuntimeError("Defina SLACK_BOT_TOKEN e SLACK_CHANNEL_ID no ambiente ou .env")

client = WebClient(token=SLACK_TOKEN)

# Emojis por alerta/sentimento para facilitar triagem visual
# Valores de 'alert' gerados por postprocessing.py
_ALERT_EMOJI = {
    "possivel_cancelamento": ":x:",
    "legal_risk": ":scales:",
    "cliente_risco_valor": ":moneybag:",
    "prazo_proximo": ":hourglass_flowing_sand:",
    "insatisfeito": ":anger:",
}
_PRIORITY_EMOJI = {
    "alta": ":red_circle:",
    "media": ":yellow_circle:",
    "baixa": ":large_green_circle:",
}
_SENT_EMOJI = {
    "negativo": ":red_circle:",
    "neutro": ":yellow_circle:",
    "positivo": ":large_green_circle:",
}

DEFAULT_CSV = Path(__file__).parent.parent / "data" / "llm_replies.csv"


def _get(row, col, default="-"):
    val = row.get(col, default)
    return val if val and val != "nan" else default


@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
def _post(channel: str, blocks: list, fallback: str):
    try:
        return client.chat_postMessage(channel=channel, blocks=blocks, text=fallback)
    except SlackApiError as e:
        print(f"[SlackApiError] {e.response['error']}  status={getattr(e.response, 'status_code', '?')}")
        raise


def build_blocks(row) -> list:
    email_id    = _get(row, "id")
    sender      = _get(row, "from")
    recipient   = _get(row, "to")
    subject     = _get(row, "subject", "(sem assunto)")
    received    = _get(row, "received_at")
    body        = _get(row, "text", "")
    alert       = _get(row, "alert", "")
    priority    = _get(row, "priority", "baixa")
    sentiment   = _get(row, "sentiment")
    entities    = _get(row, "entities", "")
    r_subject   = _get(row, "reply_subject", "")
    r_body      = _get(row, "reply_body", "(sem resposta)")
    r_explain   = _get(row, "reply_explain", "")

    alert_emoji    = _ALERT_EMOJI.get(alert.lower(), ":email:")
    priority_emoji = _PRIORITY_EMOJI.get(priority.lower(), ":white_circle:")
    sent_emoji     = _SENT_EMOJI.get(sentiment.lower(), ":white_circle:")

    alert_label = alert.replace("_", " ").capitalize() if alert and alert != "-" else "Sem alerta"

    # ── cabeçalho ──────────────────────────────────────────────────────────────
    header_text = (
        f"{alert_emoji} *{alert_label}*  {priority_emoji} Prioridade: *{priority.capitalize()}*  {sent_emoji} *{sentiment.capitalize()}*\n"
        f":id: `{email_id}`  •  :clock3: {received}"
    )

    # ── corpo do e-mail ────────────────────────────────────────────────────────
    meta_text = (
        f":bust_in_silhouette: *De:* {sender}\n"
        f":inbox_tray: *Para:* {recipient}\n"
        f":envelope: *Assunto:* {subject}"
    )

    body_preview = body[:800] + ("…" if len(body) > 800 else "")

    # ── resposta proposta ──────────────────────────────────────────────────────
    reply_header = f":robot_face: *Resposta sugerida*"
    if r_subject:
        reply_header += f"  —  _{r_subject}_"

    reply_preview = r_body[:1200] + ("…" if len(r_body) > 1200 else "")

    blocks = [
        # cabeçalho colorido via context
        {"type": "header", "text": {"type": "plain_text", "text": f"{alert_emoji} Triagem de E-mail — Prioridade {priority.capitalize()}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": header_text}},
        {"type": "divider"},
        # metadados do e-mail
        {"type": "section", "text": {"type": "mrkdwn", "text": meta_text}},
        # corpo
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Mensagem original:*\n{body_preview}"}},
    ]

    # entidades (se houver algo além de '[]')
    if entities and entities not in ("-", "[]", ""):
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f":label: *Entidades detectadas:* {entities}"}]
        })

    blocks += [
        {"type": "divider"},
        # resposta LLM
        {"type": "section", "text": {"type": "mrkdwn", "text": f"{reply_header}\n{reply_preview}"}},
    ]

    # explicação do modelo (se disponível)
    if r_explain and r_explain != "-":
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f":bulb: *Raciocínio:* {r_explain[:300]}"}]
        })

    return blocks


def send_batch(df: pd.DataFrame, channel: str, limit=None, dry=False, delay=0.8) -> int:
    sent = 0
    for _, row in df.iterrows():
        if limit and sent >= limit:
            break

        blocks = build_blocks(row)
        fallback = f"[{_get(row,'alert','')}|{_get(row,'priority','')}] {_get(row,'subject')} — {_get(row,'from')}"

        if dry:
            print(f"[dry] {fallback}")
            sent += 1
            continue

        try:
            resp = _post(channel, blocks, fallback)
            print(f"sent  id={_get(row,'id')}  ts={resp['ts']}")
            sent += 1
            time.sleep(delay)
        except Exception as e:
            print(f"failed  id={_get(row,'id')}  err={e}")

    return sent


def main():
    parser = argparse.ArgumentParser(description="Envia llm_replies.csv ao Slack")
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(DEFAULT_CSV),
        help=f"CSV de respostas LLM (default: {DEFAULT_CSV})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Máximo de e-mails a enviar")
    parser.add_argument("--dry", action="store_true", help="Simula sem enviar ao Slack")
    parser.add_argument("--delay", type=float, default=0.8, help="Segundos entre mensagens")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"Carregados {len(df)} registros de {csv_path}")

    n = send_batch(df, SLACK_CHANNEL, limit=args.limit, dry=args.dry, delay=args.delay)
    print(f"Concluído — mensagens enviadas: {n}")


if __name__ == "__main__":
    main()
