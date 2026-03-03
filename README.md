# NLP Email Classifier

Pipeline de NLP para triagem automática de mensagens recebidas via Typeform. O sistema coleta respostas, classifica sentimento, atribui prioridade, gera rascunhos de resposta com LLM e entrega um resumo estruturado no Slack — tudo de forma automática via GitHub Actions.

---

## Visão geral

```
Typeform → Ingestão → Merge → Limpeza NLP → Treinamento → Predição → Priorização → LLM Replies → Slack
```

| Etapa | Script | Entrada | Saída |
|---|---|---|---|
| 1. Ingestão | `form_ingest.py` | API Typeform | `data/staging/email/*.csv` |
| 2. Merge | `merge_messages.py` | `data/staging/email/` | `data/unified_inbox.csv` |
| 3. Limpeza | `clean_and_annotate.py` | `unified_inbox.csv` | `data/unified_clean.csv` |
| 4. Treinamento | `train_sentiment.py` | `unified_clean.csv` | `models/sentiment.joblib` |
| 5. Predição | `predict.py` | `unified_clean.csv` | `data/final_triage.csv` |
| 6. Priorização | `postprocessing.py` | `final_triage.csv` | `data/final_triage_enriched.csv` |
| 7. Respostas LLM | `llm_generate_replies.py` | `final_triage_enriched.csv` | `data/llm_replies.csv` |
| 8. Slack | `send_to_slack_bot.py` | `llm_replies.csv` | mensagens no Slack |

---

## Estrutura do projeto

```
.
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml   # execução diária automática (08:00 UTC)
├── data/
│   ├── staging/
│   │   ├── email/               # CSVs brutos por resposta do Typeform
│   │   └── .typeform_cursor     # cursor de paginação (último response_id processado)
│   ├── unified_inbox.csv        # todas as mensagens unificadas
│   ├── unified_clean.csv        # mensagens pré-processadas pelo NLP
│   ├── final_triage.csv         # mensagens + sentimento previsto
│   ├── final_triage_enriched.csv# mensagens + prioridade + alertas
│   └── llm_replies.csv          # mensagens + rascunhos de resposta
├── models/
│   ├── sentiment.joblib         # modelo TF-IDF + Regressão Logística
│   └── type.joblib              # (reservado para classificação de tipo)
├── src/
│   ├── preprocessing.py         # normalização, remoção de ruído, lemmatização (spaCy)
│   ├── form_ingest.py           # coleta respostas do Typeform com paginação por cursor
│   ├── merge_messages.py        # consolida staging em unified_inbox.csv
│   ├── clean_and_annotate.py    # aplica preprocessing e salva unified_clean.csv
│   ├── train_sentiment.py       # treina o classificador de sentimento
│   ├── predict.py               # aplica o modelo e grava final_triage.csv
│   ├── postprocessing.py        # regras de negócio: prioridade e alertas
│   ├── llm_generate_replies.py  # gera rascunhos via OpenAI GPT (paralelo)
│   └── send_to_slack_bot.py     # envia mensagens estruturadas ao Slack
├── requirements.txt
└── .env                         # variáveis locais (não versionar)
```

---

## Requisitos

- Python 3.11+
- Conta Typeform com Personal Access Token
- Chave de API OpenAI
- Bot do Slack com permissão `chat:write`

---

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

---

## Configuração

Crie o arquivo `.env` na raiz do projeto com as variáveis abaixo:

```env
# Typeform
TYPEFORM_TOKEN=tfp_...          # Personal Access Token (admin.typeform.com/account#/section/tokens)
TYPEFORM_FORM_ID=XXXXXXXX       # ID do formulário (aparece na URL do form)

# Refs dos campos no Typeform (ajuste conforme o seu form)
FIELD_NOME=nome
FIELD_EMAIL=email
FIELD_ASSUNTO=assunto
FIELD_MENSAGEM=mensagem

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini        # opcional; padrão: gpt-4o-mini

# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C0XXXXXXXXX
```

Para uso no GitHub Actions, cadastre cada variável acima como **Secret** em
`Settings → Secrets and variables → Actions`.

---

## Executando a pipeline localmente

Execute cada etapa na ordem ou rode o pipeline completo:

```bash
# 1. Ingestão
python src/form_ingest.py

# 2. Merge
python src/merge_messages.py

# 3. Limpeza
PYTHONPATH=src python src/clean_and_annotate.py

# 4. Treinamento
python src/train_sentiment.py --data data/unified_clean.csv --out models/sentiment.joblib

# 5. Predição
PYTHONPATH=src python src/predict.py

# 6. Priorização
python src/postprocessing.py

# 7. Respostas LLM
python src/llm_generate_replies.py

# 8. Envio ao Slack (dry run para testar sem enviar)
python src/send_to_slack_bot.py --dry
```

---

## Detalhes de cada módulo

### `form_ingest.py` — Ingestão Typeform

Consulta a API do Typeform e salva cada resposta como um CSV individual em `data/staging/email/`. Utiliza um cursor persistido em `data/staging/.typeform_cursor` para processar apenas respostas novas a cada execução.

```bash
python src/form_ingest.py                 # busca respostas novas
python src/form_ingest.py --max 200       # limita a 200 respostas
python src/form_ingest.py --reset-cursor  # reinicia do zero (reingere tudo)
```

**Schema do CSV gerado:**

| Coluna | Descrição |
|---|---|
| `id` | `form-{response_id}` |
| `channel` | `formulario` |
| `from` | `Nome <email>` |
| `subject` | Campo "Assunto" do form |
| `text` | `assunto: mensagem` (texto principal para NLP) |
| `received_at` | Timestamp de envio (ISO 8601) |
| `message_id` | E-mail do remetente |

---

### `preprocessing.py` — Pré-processamento NLP

Funções de limpeza de texto usando spaCy (`pt_core_news_sm`):

- Normalização Unicode (NFKC)
- Remoção de URLs, e-mails, tags HTML e separadores
- Lowercase
- Remoção de stopwords
- Lematização

```python
from preprocessing import preprocess_text, batch_preprocess

preprocess_text("Favor enviar o contrato para fulano@example.com até 10/03.")
# → "favor enviar contrato prazo"
```

---

### `train_sentiment.py` — Treinamento do modelo

Treina um classificador **TF-IDF (1-2 gramas) + Regressão Logística** com balanceamento de classes.

**Labels:** `positivo`, `negativo`, `neutro`

Se a coluna `label` não existir no CSV, o script infere automaticamente as labels por palavras-chave:

- **Negativo:** `reclamação`, `defeito`, `problema`, `cancelamento`, `atraso`, `erro`, `devolução`...
- **Positivo:** `elogio`, `satisfação`, `parabenizo`, `gostei`, `resolvido`...
- **Neutro:** demais casos

```bash
python src/train_sentiment.py --data data/unified_clean.csv --out models/sentiment.joblib
```

O modelo serializado é salvo com `joblib` em `models/sentiment.joblib`.

---

### `postprocessing.py` — Priorização e alertas

Aplica regras de negócio sobre o resultado da predição para determinar prioridade e categoria de alerta:

| Condição | Prioridade | Alerta |
|---|---|---|
| Menciona cancelamento/rescisão | Alta | `possivel_cancelamento` |
| Menciona Procon/advogado/ação judicial | Alta | `legal_risk` |
| Sentimento negativo + valor ≥ R$1.000 | Alta | `cliente_risco_valor` |
| Sentimento negativo + data/prazo mencionado | Alta | `prazo_proximo` |
| Sentimento negativo (demais) | Média | `insatisfeito` |
| Sentimento neutro ou positivo | Baixa | — |

---

### `llm_generate_replies.py` — Geração de respostas com LLM

Gera rascunhos de resposta em português para cada mensagem usando a API OpenAI (GPT-4o-mini por padrão). Executa em paralelo com `ThreadPoolExecutor` e suporta retomada de progresso.

**Colunas geradas:**

| Coluna | Descrição |
|---|---|
| `reply_subject` | Assunto sugerido para a resposta |
| `reply_body` | Corpo da resposta (≤ 200 palavras, tom empático e formal) |
| `reply_explain` | Justificativa do modelo (para revisão humana) |

```bash
python src/llm_generate_replies.py --workers 20   # 20 chamadas paralelas
python src/llm_generate_replies.py --dry-run      # simula sem chamar a API
```

---

### `send_to_slack_bot.py` — Envio ao Slack

Lê `data/llm_replies.csv` e publica cada mensagem como bloco Slack estruturado com:

- Cabeçalho com emoji de prioridade e alerta
- Metadados do e-mail (remetente, assunto, data)
- Preview da mensagem original
- Rascunho de resposta gerado pelo LLM
- Raciocínio do modelo (contextual)

```bash
python src/send_to_slack_bot.py                    # envia tudo
python src/send_to_slack_bot.py --limit 5 --dry    # testa sem enviar
python src/send_to_slack_bot.py --delay 1.5        # ajusta intervalo entre mensagens
```

---

## Automação — GitHub Actions

O workflow `.github/workflows/daily_pipeline.yml` executa a pipeline completa **todos os dias às 08:00 UTC** (05:00 BRT). Pode ser acionado manualmente via `workflow_dispatch`.

**Etapas do workflow:**

1. Checkout do repositório
2. Setup Python 3.11 + cache de pip
3. Instalação de dependências
4. Download do modelo spaCy (`pt_core_news_sm`)
5. Validação das credenciais Typeform (verifica HTTP 200 na API)
6. Execução sequencial das 8 etapas da pipeline

**Secrets necessários no repositório:**

```
TYPEFORM_TOKEN
TYPEFORM_FORM_ID
FIELD_NOME
FIELD_EMAIL
FIELD_ASSUNTO
FIELD_MENSAGEM
OPENAI_API_KEY
OPENAI_MODEL
SLACK_BOT_TOKEN
SLACK_CHANNEL_ID
```

---

## Stack tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| NLP / ML | scikit-learn, spaCy (`pt_core_news_sm`) |
| LLM | OpenAI API (GPT-4o-mini) |
| Notificação | Slack SDK |
| Ingestão | Typeform API |
| Orquestração | GitHub Actions |
| Dados | pandas, CSV |
| Serialização de modelo | joblib |

---

## Dependências principais

```
pandas, numpy, python-dotenv, requests, tqdm
scikit-learn, joblib, spacy, dateparser
slack-sdk, tenacity
openai
pytest
```

Instale com:

```bash
pip install -r requirements.txt
```
