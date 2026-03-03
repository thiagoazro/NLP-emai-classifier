#!/usr/bin/env python3
# src/generate_llm_replies.py
"""
Gera rascunhos de resposta usando LLM para cada linha de data/final_triage_enriched.csv
Saída: data/llm_replies.csv  (colunas originais + reply_subject + reply_body + reply_explain)
Requisitos:
  pip install openai pandas python-dotenv tqdm
Ambiente:
  OPENAI_API_KEY, OPENAI_MODEL (opcional)
Usage:
  python src/generate_llm_replies.py --in data/labeled_emails.csv --out data/llm_replies.csv --workers 20
"""
import os, argparse, json, threading
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

try:
    import openai
except Exception:
    openai = None

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PROMPT_TEMPLATE = """Você é um assistente especialista em atendimento ao cliente da empresa X.
Receba a mensagem do cliente abaixo, o tipo previsto e o sentimento. Produza:
1) Um assunto de resposta curto (uma linha).
2) Um corpo de resposta formal em português do brasil, porém empático, com máximo 6-8 linhas (não mais que ~120-200 palavras),
   pedindo confirmação quando necessário e oferecendo o próximo passo (ex: abrir ticket, solicitar dados).
3) Uma breve justificativa (1-2 frases) explicando por que essa resposta é adequada (para revisão humana).
Regras:
- Não invente dados pessoais ou promessas de reembolso sem confirmação. Se o cliente menciona cobrança, peça dados (com segurança).
- Se prioridade = ALTA, inclua frase: "Estaremos priorizando seu atendimento."
- Use tom corporativo e empático; mantenha conformidade com LGPD: não peça documentos sensíveis por canais inseguros.
Retorne um JSON com chaves: subject, body, explain.
Mensagem do cliente:
{message}
Metadados:
- tipo previsto: {type}
- sentimento previsto: {sentiment}
- prioridade: {priority}

Gere apenas o JSON válido sem texto adicional.
"""

def safe_parse_json(s):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
    return None

def call_openai(prompt, model, api_key, max_tokens=512, temperature=0.2):
    if openai is None:
        raise RuntimeError("openai não instalado. Execute: pip install openai")
    # Compatível com openai v0.x e v1.x
    if hasattr(openai, "OpenAI"):
        # v1.x
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Você gera respostas JSON para rascunhos de atendimento em português do Brasil."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content
    else:
        # v0.x
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Você gera respostas JSON para rascunhos de atendimento em português do Brasil."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp["choices"][0]["message"]["content"]

def process_row(idx, row, model, api_key):
    message = row.get('text') or row.get('text_clean') or ""
    sentiment = row.get('sentiment') or row.get('label') or ''
    email_type = row.get('type', '')
    priority = row.get('priority', 'NORMAL')
    if not priority or str(priority).strip() == '':
        priority = 'NORMAL'

    prompt = PROMPT_TEMPLATE.format(
        message=message,
        type=email_type,
        sentiment=sentiment,
        priority=priority
    )
    try:
        txt = call_openai(prompt, model=model, api_key=api_key)
        parsed = safe_parse_json(txt)
        if parsed:
            subj = parsed.get("subject", "").strip()
            body = parsed.get("body", "").strip()
            explain = parsed.get("explain", "").strip()
        else:
            subj = ""
            body = txt.strip()
            explain = "Resposta não parseada automaticamente; revisar."
    except Exception as e:
        subj = ""
        body = f"[ERRO AO GERAR RESPOSTA: {e}]"
        explain = "Erro ao chamar LLM"

    return idx, subj, body, explain

def main(in_path, out_path, model, api_key, workers, batch):
    df = pd.read_csv(in_path)
    if 'text' not in df.columns:
        raise RuntimeError(f"{in_path} precisa conter coluna 'text'.")

    # Retomada: se o CSV de saída já existe, carrega progresso anterior
    if os.path.exists(out_path):
        df_prev = pd.read_csv(out_path)
        for col in ['reply_subject', 'reply_body', 'reply_explain']:
            if col in df_prev.columns:
                df[col] = df_prev[col].fillna("")
        print(f"Retomando progresso de {out_path}")
    else:
        df['reply_subject'] = ""
        df['reply_body'] = ""
        df['reply_explain'] = ""

    for col in ['reply_subject', 'reply_body', 'reply_explain']:
        df[col] = df[col].fillna("")

    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    pending_idx = df.index[df['reply_body'] == ""].tolist()
    print(f"Total: {len(df)} | Já processados: {len(df) - len(pending_idx)} | Pendentes: {len(pending_idx)}")

    lock = threading.Lock()
    save_counter = [0]

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_row, idx, df.loc[idx], model, api_key): idx
                for idx in pending_idx
            }
            with tqdm(total=len(pending_idx), desc="Gerando rascunhos") as pbar:
                for future in as_completed(futures):
                    idx, subj, body, explain = future.result()
                    with lock:
                        df.at[idx, 'reply_subject'] = subj
                        df.at[idx, 'reply_body'] = body
                        df.at[idx, 'reply_explain'] = explain
                        save_counter[0] += 1
                        pbar.update(1)
                        if save_counter[0] % batch == 0:
                            df.to_csv(out_path, index=False, encoding="utf-8")
    except KeyboardInterrupt:
        print("\nInterrompido. Salvando progresso...")

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"LLM replies salvo em {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="data/final_triage_enriched.csv")
    p.add_argument("--out", dest="out_path", default="data/llm_replies.csv")
    p.add_argument("--model", dest="model", default=None)
    p.add_argument("--key", dest="key", default=None)
    p.add_argument("--workers", type=int, default=20, help="chamadas paralelas simultâneas")
    p.add_argument("--batch", type=int, default=50, help="salvar a cada N respostas")
    args = p.parse_args()
    main(args.in_path, args.out_path, args.model, args.key, args.workers, args.batch)
