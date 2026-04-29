# 🏥 MedAgent — Medical Document Intelligence System

> Upload a medical lab report PDF → get plain-language explanations, severity classifications, doctor questions, action plan, and real-world agent actions (Google Calendar, Gmail, Slack).

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![BioBERT](https://img.shields.io/badge/BioBERT-F1%3D87.01%25-green)](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.3--70B-orange)](https://groq.com)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal)](https://fastapi.tiangolo.com)

---

## What is MedAgent?

MedAgent is an end-to-end NLP system built as a course project for DLNLP. It takes any medical lab report PDF and runs it through a five-module pipeline:

```
PDF Lab Report
      ↓
Module 01 — Document Ingestion     (PyMuPDF + Tesseract OCR)
      ↓
Module 02 — Medical NER            (BioBERT fine-tuned on BC5CDR, F1=87.01%)
      ↓
Module 03 — Lab Value Parser       (Regex + rule-based severity classification)
      ↓
Module 04 — Text Simplification    (Groq Llama-3.3-70B, 4-step prompt chain)
      ↓
Module 05 — MCP Agent              (Google Calendar + Gmail + Slack + Drug Interactions)
      ↓
Patient-ready report + real-world actions
```

---

## Key Results

| Module | Metric | Result |
|--------|--------|--------|
| BioBERT NER | F1 on BC5CDR test set | **87.01%** (target: 82%) |
| Lab Value Parser | Unit test accuracy | **100%** (11/11) |
| Text Simplification | FK grade improvement on MedQuAD | **+6.6 grades** |
| MCP Agent | Tools executed successfully | **4/4** |

---

## Features

### 🔬 Module 02 — Medical NER
- Fine-tuned `dmis-lab/biobert-base-cased-v1.2` on BC5CDR corpus
- Extracts **Chemical** entities (drugs, compounds) and **Disease** entities
- WordPiece subword tokenization with label alignment (`-100` for continuation subwords)
- Trained for 4 epochs on NVIDIA A100, ~10 minutes
- F1 = 87.01% on BC5CDR test set (5,865 sentences)

### 🧪 Module 03 — Lab Value Parser
- Parses 18 test types: Hemoglobin, HbA1c, eGFR, LDL, Glucose, Creatinine, etc.
- Three severity levels: **Normal**, **Borderline**, **Critical**
- Handles one-sided ranges (eGFR ≥ 60), upper-bound ranges (LDL < 100), two-sided ranges
- Dedicated HbA1c extractor for reliable parsing
- 100% accuracy on all unit tests and health check assertions

### 📝 Module 04 — Text Simplification
- 4-step prompt chain using Groq Llama-3.3-70B (temperature=0.3)
  1. Plain-language explanation per finding (PLAIN + IMPACT)
  2. 3-4 sentence overall health summary
  3. 6 specific doctor-visit questions
  4. 6 personalized action steps
- +6.6 Flesch-Kincaid grade improvement on MedQuAD samples
- Target reading level: Grade 8

### 🤖 Module 05 — MCP Agent
Real-world tool integrations triggered directly from NLP outputs:

| Tool | Integration | What it does |
|------|-------------|--------------|
| 📅 Google Calendar | Google Calendar API v3 | Creates daily medication reminders + follow-up appointment |
| 📧 Gmail | Gmail API | Sends patient-friendly lab summary email |
| 💬 Slack | Incoming Webhook | Posts morning health nudge with medication reminders |
| 💊 Drug Interactions | DrugBank + curated pairs | Checks interactions between detected medications |

---

## Datasets

| Dataset | Task | Size |
|---------|------|------|
| BC5CDR | NER training | 16,423 sentences |
| MedMentions | Rich NER | 4,392 documents |
| MedQuAD | QA + Simplification | 47,441 pairs |
| NCBI Disease | NER evaluation | 793 documents |
| DrugBank | Drug vocabulary | 19,842 drugs |

---

## Project Structure

```
medagent-nlp/
├── DLNLP.ipynb              # Full pipeline notebook (Modules 00-05)
├── report/
│   └── DLNLP.pdf            # ACL-style project report (7 pages)
├── app/
│   ├── backend/
│   │   ├── main.py          # FastAPI backend — full pipeline
│   │   └── requirements.txt
│   └── frontend/
│       └── index.html       # Web UI
├── data/
│   └── synthetic_lab_report.pdf
└── .gitignore
```

---

## Setup & Running

### Prerequisites
- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Optional: Google OAuth credentials for Calendar/Gmail
- Optional: Slack Incoming Webhook URL

### Backend

```bash
cd app/backend
pip install -r requirements.txt

export GROQ_API_KEY="gsk_your_key_here"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/xxx"   # optional
export PATIENT_EMAIL="patient@email.com"                           # optional
export GOOGLE_TOKEN_PATH="/path/to/token.json"                     # optional

uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd app/frontend
python3 -m http.server 3000
# Open http://localhost:3000/index.html
```

### API

```bash
# Health check
curl http://localhost:8000/health

# Analyze a PDF
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_lab_report.pdf"
```

### Run in Google Colab

```python
import subprocess, threading, time, os

os.environ['GROQ_API_KEY'] = 'your-key'

subprocess.run(['pip', 'install', '-q', 'fastapi', 'uvicorn',
                'python-multipart', 'pymupdf', 'groq'])

def run():
    subprocess.run(['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000'])

threading.Thread(target=run, daemon=True).start()
time.sleep(4)

# Tunnel with ngrok
from pyngrok import ngrok
tunnel = ngrok.connect(8000)
print(f'Public URL: {tunnel.public_url}')
```

---

## Google OAuth Setup (for real Calendar + Gmail)

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Enable **Google Calendar API** and **Gmail API**
3. Create OAuth 2.0 credentials (Web application type)
4. Add `http://localhost:8080` as authorized redirect URI
5. Download `credentials.json`
6. Run the OAuth flow to generate `token.json`
7. Set `GOOGLE_TOKEN_PATH=/path/to/token.json`

---

## Technical Details

### BioBERT Training

```
Model:     dmis-lab/biobert-base-cased-v1.2
Task:      Token classification (NER)
Dataset:   BC5CDR (Chemical + Disease)
Labels:    {O, B-Chemical, I-Chemical, B-Disease, I-Disease}
Epochs:    4
LR:        2e-5
Batch:     32 (train), 64 (eval)
Warmup:    0.1
GPU:       NVIDIA A100
Time:      ~10 minutes
F1:        87.01%
```

### Severity Classification Logic

```python
# One-sided lower bound (e.g. eGFR >= 60)
if ref_high >= 999:
    pct = (ref_low - value) / ref_low * 100
    return "Critical" if pct >= 25 else ("Borderline" if pct > 10 else "Normal")

# One-sided upper bound (e.g. LDL < 100)
if ref_low <= 0:
    pct = (value - ref_high) / ref_high * 100
    return "Critical" if pct > 30 else ("Borderline" if pct > 10 else "Normal")

# Two-sided range
pct = deviation / range_width * 100
return "Critical" if pct > 30 else ("Borderline" if pct > 10 else "Normal")
```

---

## Report

The full ACL-style project report is available at [`report/DLNLP.pdf`](report/DLNLP.pdf).

Covers: Abstract, Introduction, Related Work, Problem Statement, Technical Approach, Experimental Setup, Results, Conclusions, References (16 citations).

---

## Security Note

Never commit `token.json` or `credentials.json` to GitHub — these contain your Google OAuth tokens. They are excluded by `.gitignore`.

---

## Course

DLNLP — Deep Learning for Natural Language Processing
