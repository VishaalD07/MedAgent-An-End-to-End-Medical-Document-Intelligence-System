"""
MedAgent FastAPI Backend — Full Pipeline with Real Integrations
Module 01: PDF Ingestion (PyMuPDF)
Module 02: Medical NER (BioBERT)
Module 03: Lab Value Parser (regex + rule-based)
Module 04: Text Simplification (Groq Llama 3.3 70B)
Module 05: MCP Agent (Real Google Calendar + Gmail + Slack)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import tempfile, os, re, time, json, base64
from pathlib import Path
from datetime import datetime, timedelta

import fitz
from groq import Groq

app = FastAPI(title="MedAgent Full Pipeline", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Groq ───────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
GROQ_MODEL  = "llama-3.3-70b-versatile"
SYSTEM      = """You are a medical interpreter helping patients understand their lab results.
Use simple everyday language (8th grade level). Keep explanations to 1-2 sentences.
Do not diagnose. Always recommend consulting a doctor."""

def call_groq(prompt, max_tokens=600):
    for attempt in range(3):
        try:
            r = groq_client.chat.completions.create(
                model=GROQ_MODEL, temperature=0.3, max_tokens=max_tokens,
                messages=[{"role":"system","content":SYSTEM},
                          {"role":"user","content":prompt}])
            return r.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep((attempt+1)*10)
            else:
                return f"[Error: {e}]"
    return "[Error: max retries]"

# ── Google OAuth ───────────────────────────────────────────────────────────────
def get_google_creds():
    """Load Google credentials from token.json or env variable."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_path = os.environ.get("GOOGLE_TOKEN_PATH", "/content/token.json")
    token_json  = os.environ.get("GOOGLE_TOKEN_JSON", "")

    if token_json:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        tmp.write(token_json)
        tmp.close()
        token_path = tmp.name

    if not Path(token_path).exists():
        return None

    creds = Credentials.from_authorized_user_file(token_path)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_path, 'w') as f:
                f.write(creds.to_json())
        except Exception:
            pass
    return creds

# ── Module 01: PDF Ingestion ───────────────────────────────────────────────────
def module01_extract(pdf_path):
    doc   = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        raw   = page.get_text("text")
        clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
        pages.append({"page": i+1, "text": clean, "chars": len(clean)})
    doc.close()
    full_text = "\n\n".join(p["text"] for p in pages if p["text"])
    return {
        "full_text":   full_text,
        "page_count":  len(pages),
        "total_chars": len(full_text),
        "doc_type":    "text_pdf" if len(full_text) > 100 else "scanned_pdf",
    }

# ── Module 02: Medical NER ─────────────────────────────────────────────────────
_ner_pipeline = None
NER_MODEL_PATH = os.environ.get("NER_MODEL_PATH", "/content/biobert_ner/best_model")

def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            _ner_pipeline = hf_pipeline(
                "ner", model=NER_MODEL_PATH, tokenizer=NER_MODEL_PATH,
                aggregation_strategy="first",
                device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            print(f"BioBERT not available: {e}")
            _ner_pipeline = "fallback"
    return _ner_pipeline

def module02_ner(text):
    ner       = get_ner_pipeline()
    sentences = [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+", text)
                 if len(s.strip().split()) >= 3]
    entities  = []

    if ner == "fallback":
        drug_pat = re.compile(
            r"\b(Metformin|Lisinopril|Atorvastatin|Aspirin|Insulin|Warfarin|"
            r"Amlodipine|Losartan|Furosemide|Gabapentin|Omeprazole|Levothyroxine)\b", re.IGNORECASE)
        dis_pat  = re.compile(
            r"\b(diabetes|hypertension|anemia|nephropathy|carcinoma|"
            r"lymphoma|cancer|disease|syndrome|disorder)\b", re.IGNORECASE)
        for m in drug_pat.finditer(text):
            entities.append({"text": m.group(), "label": "Chemical", "confidence": 0.95})
        for m in dis_pat.finditer(text):
            entities.append({"text": m.group(), "label": "Disease", "confidence": 0.85})
        model_used = "regex_fallback"
    else:
        for sent in sentences[:50]:
            try:
                for p in ner(sent):
                    label = "Chemical" if "Chemical" in p["entity_group"] else "Disease"
                    entities.append({"text": p["word"], "label": label,
                                     "confidence": round(float(p["score"]), 4)})
            except Exception:
                continue
        model_used = "biobert_bc5cdr_finetuned"

    seen, unique = set(), []
    for e in entities:
        k = (e["text"].lower(), e["label"])
        if k not in seen:
            seen.add(k)
            unique.append(e)

    chemicals = [e for e in unique if e["label"] == "Chemical"]
    diseases  = [e for e in unique if e["label"] == "Disease"]
    return {"entities": unique, "chemicals": chemicals, "diseases": diseases,
            "model_used": model_used, "total": len(unique)}

# ── Module 03: Lab Value Parser ────────────────────────────────────────────────
REFS = {
    "hemoglobin":    (13.5, 17.5, "g/dL",          7.0,  20.0),
    "hgb":           (13.5, 17.5, "g/dL",          7.0,  20.0),
    "hematocrit":    (41.0, 53.0, "%",             20.0,  60.0),
    "wbc":           (4.5,  11.0, "10^3/uL",        2.0,  30.0),
    "platelet":      (150,  400,  "10^3/uL",       50.0, 1000.0),
    "glucose":       (70,   99,   "mg/dL",         40.0,  500.0),
    "creatinine":    (0.74, 1.35, "mg/dL",          0.3,  10.0),
    "egfr":          (60,   999,  "mL/min/1.73m2", 15.0, 9999.0),
    "sodium":        (136,  145,  "mEq/L",        120.0,  160.0),
    "potassium":     (3.5,  5.1,  "mEq/L",          2.5,   6.5),
    "ldl":           (0,    100,  "mg/dL",          0.0,  400.0),
    "hdl":           (40,   9999, "mg/dL",         10.0, 9999.0),
    "hba1c":         (0,    5.7,  "%",              0.0,  15.0),
    "tsh":           (0.4,  4.0,  "mIU/L",          0.1,  10.0),
    "alt":           (7,    56,   "U/L",            0.0,  500.0),
    "ast":           (10,   40,   "U/L",            0.0,  500.0),
    "triglycerides": (0,    150,  "mg/dL",          0.0, 1000.0),
    "cholesterol":   (0,    200,  "mg/dL",          0.0,  600.0),
}

def classify_severity(v, rl, rh, flag="", cl=None, ch=None):
    if flag.upper() in ("CRITICAL","PANIC","ALERT"): return "Critical"
    if rl is None and rh is None: return "Normal"
    if cl and cl > 0    and v <= cl: return "Critical"
    if ch and ch < 9999 and v >= ch: return "Critical"
    if rh and rh >= 999:
        if rl and v < rl:
            pct = (rl-v)/rl*100
            return "Critical" if pct>=25 else ("Borderline" if pct>10 else "Normal")
        return "Normal"
    if rl is not None and rl <= 0:
        if rh and v > rh:
            pct = (v-rh)/rh*100
            return "Critical" if pct>30 else ("Borderline" if pct>10 else "Normal")
        return "Normal"
    if rl is not None and rh is not None:
        w = rh-rl
        if w <= 0: return "Normal"
        if v < rl:
            pct = (rl-v)/w*100
            return "Critical" if pct>30 else ("Borderline" if pct>10 else "Normal")
        if v > rh:
            pct = (v-rh)/w*100
            return "Critical" if pct>30 else ("Borderline" if pct>10 else "Normal")
    return "Normal"

def module03_parse_labs(text):
    results, seen = [], set()
    np_ = re.compile(r"\d+(?:\.\d+)?")
    fp_ = re.compile(r"\b(HIGH|LOW|CRITICAL|PANIC|H\b|L\b)\b", re.IGNORECASE)
    tp_ = re.compile(r"(?i)\b(" + "|".join(re.escape(k) for k in sorted(REFS, key=len, reverse=True)) + r")\b")

    m = re.search(r"[Hh][Bb][Aa]1[Cc]\)?[:\s]+(\d+\.?\d*)\s*%", text)
    if m:
        v=float(m.group(1)); rl,rh,u,cl,ch=REFS["hba1c"]
        f="HIGH" if v>rh else ""
        results.append({"test_name":"HbA1c","value":v,"unit":u,"ref_range":f"{rl}-{rh}",
            "ref_low":rl,"ref_high":rh,"flag":f,"severity":classify_severity(v,rl,rh,f,cl,ch)})
        seen.add("hba1c")

    for m in tp_.finditer(text):
        k=m.group(0).lower().strip()
        if k in seen: continue
        win=text[m.end():m.end()+150]
        vm=np_.search(win)
        if not vm: continue
        try: v=float(vm.group())
        except: continue
        fm=fp_.search(win[:100])
        f=fm.group(1).upper() if fm else ""
        if f=="H": f="HIGH"
        if f=="L": f="LOW"
        rl,rh,u,cl,ch=REFS.get(k,(None,None,"",None,None))
        results.append({"test_name":k.title(),"value":v,"unit":u,
            "ref_range":f"{rl}-{rh}" if rl is not None else "",
            "ref_low":rl,"ref_high":rh,"flag":f,
            "severity":classify_severity(v,rl,rh,f,cl,ch)})
        seen.add(k)

    nc=sum(1 for r in results if r["severity"]=="Critical")
    nb=sum(1 for r in results if r["severity"]=="Borderline")
    nn=sum(1 for r in results if r["severity"]=="Normal")
    return {"lab_values":results,"summary":{"total":len(results),"critical":nc,"borderline":nb,"normal":nn}}

# ── Module 04: Text Simplification ────────────────────────────────────────────
def module04_simplify(lab_values, ner_result):
    abnormal = [lv for lv in lab_values if lv["severity"] in ("Critical","Borderline")]
    findings = []
    for lv in abnormal:
        resp   = call_groq(f"Test: {lv['test_name']}\nValue: {lv['value']} {lv['unit']}\n"
                           f"Normal: {lv['ref_range']}\nSeverity: {lv['severity']}\n\n"
                           "Respond:\nPLAIN: <one sentence>\nIMPACT: <one sentence>")
        plain  = next((l[6:].strip() for l in resp.split("\n") if l.startswith("PLAIN:")),  resp[:200])
        impact = next((l[7:].strip() for l in resp.split("\n") if l.startswith("IMPACT:")), "Discuss with your doctor.")
        findings.append({**lv, "plain_language": plain, "patient_impact": impact})

    ctx      = "\n".join(f"- {lv['test_name']}: {lv['value']} {lv['unit']} ({lv['severity']})" for lv in lab_values)
    meds     = [e["text"] for e in ner_result.get("chemicals",[])][:5]
    diseases = [e["text"] for e in ner_result.get("diseases",[])][:5]

    summary   = call_groq(f"Lab results:\n{ctx}\nMedications: {', '.join(meds) or 'none'}\n"
                          f"Conditions: {', '.join(diseases) or 'none'}\n\nWrite a 3-4 sentence plain-language health summary.", 400)
    q_raw     = call_groq(f"Lab results:\n{ctx}\n\nGenerate exactly 6 specific numbered questions for the doctor.", 400)
    a_raw     = call_groq(f"Lab results:\n{ctx}\nMedications: {', '.join(meds) or 'none'}\n\nCreate exactly 6 specific numbered action steps.", 400)
    questions = [re.match(r"^\d+\.?\s+(.+)",l.strip()).group(1) for l in q_raw.split("\n") if re.match(r"^\d+\.?\s+(.+)",l.strip())]
    actions   = [re.match(r"^\d+\.?\s+(.+)",l.strip()).group(1) for l in a_raw.split("\n") if re.match(r"^\d+\.?\s+(.+)",l.strip())]

    return {"findings":findings,"health_summary":summary,"doctor_questions":questions,
            "action_plan":actions,"model_used":GROQ_MODEL}

# ── Module 05: MCP Agent — REAL integrations ──────────────────────────────────
KNOWN_INTERACTIONS = {
    ("Atorvastatin","Lisinopril"):  "Safe combination. Monitor for muscle pain.",
    ("Metformin","Lisinopril"):     "Monitor kidney function — both affect renal clearance.",
    ("Metformin","Atorvastatin"):   "Generally safe. Rare increased myopathy risk.",
    ("Aspirin","Lisinopril"):       "High-dose aspirin may reduce Lisinopril effectiveness.",
    ("Warfarin","Atorvastatin"):    "Atorvastatin can increase Warfarin levels — monitor INR.",
    ("Amlodipine","Atorvastatin"):  "Amlodipine increases Atorvastatin exposure — max 20mg.",
    ("Metformin","Furosemide"):     "Furosemide may increase Metformin blood levels.",
    ("Lisinopril","Furosemide"):    "Risk of low blood pressure — monitor closely.",
}

def tool_google_calendar(medications, critical_findings, creds):
    """Create real Google Calendar events."""
    if creds is None:
        return _simulated_calendar(medications, critical_findings), "simulated"

    try:
        from googleapiclient.discovery import build
        service = build('calendar', 'v3', credentials=creds)
        today   = datetime.now()
        created = []

        # Medication reminders
        for med in medications:
            event = {
                'summary':     f'Take {med}',
                'description': f'Daily medication reminder from MedAgent',
                'start': {'dateTime': (today+timedelta(days=1)).strftime('%Y-%m-%dT08:00:00'),
                          'timeZone': 'America/Chicago'},
                'end':   {'dateTime': (today+timedelta(days=1)).strftime('%Y-%m-%dT08:15:00'),
                          'timeZone': 'America/Chicago'},
                'recurrence': ['RRULE:FREQ=DAILY'],
                'reminders': {'useDefault': False,
                              'overrides': [{'method':'popup','minutes':10}]},
            }
            result = service.events().insert(calendarId='primary', body=event).execute()
            created.append({"summary": result['summary'], "start": result['start']['dateTime'],
                            "link": result['htmlLink'], "status": "created"})

        # Follow-up appointment
        if critical_findings:
            critical_names = ', '.join(lv['test_name'] for lv in critical_findings[:3])
            event = {
                'summary':     'Doctor Follow-up — Lab Results Review',
                'description': f'Review critical results: {critical_names}\nScheduled by MedAgent',
                'start': {'dateTime': (today+timedelta(weeks=2)).strftime('%Y-%m-%dT10:00:00'),
                          'timeZone': 'America/Chicago'},
                'end':   {'dateTime': (today+timedelta(weeks=2)).strftime('%Y-%m-%dT11:00:00'),
                          'timeZone': 'America/Chicago'},
                'reminders': {'useDefault': False,
                              'overrides': [{'method':'email','minutes':1440},
                                            {'method':'popup','minutes':60}]},
            }
            result = service.events().insert(calendarId='primary', body=event).execute()
            created.append({"summary": result['summary'], "start": result['start']['dateTime'],
                            "link": result['htmlLink'], "status": "created"})

        return created, "success"
    except Exception as e:
        print(f"Calendar error: {e}")
        return _simulated_calendar(medications, critical_findings), "simulated"

def _simulated_calendar(medications, critical_findings):
    today = datetime.now()
    events = [{"summary": f"Take {med}", "start": (today+timedelta(days=1)).strftime("%Y-%m-%dT08:00:00"),
               "status": "simulated"} for med in medications]
    if critical_findings:
        events.append({"summary": "Doctor Follow-up — Lab Results Review",
                       "start": (today+timedelta(weeks=2)).strftime("%Y-%m-%dT10:00:00"),
                       "status": "simulated"})
    return events

def tool_gmail_send(patient_email, lab_values, simplification, creds):
    """Send real email via Gmail API."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    critical = [lv for lv in lab_values if lv["severity"] == "Critical"]
    critical_list = "\n".join(f"  • {lv['test_name']}: {lv['value']} {lv['unit']}" for lv in critical)

    # Generate email body via Groq
    email_body = call_groq(
        f"Write a patient-friendly email about these critical lab results:\n{critical_list}\n\n"
        f"Health summary: {simplification.get('health_summary','')}\n\n"
        "Format: Subject line, warm greeting, key findings, next steps, encouraging close. Under 250 words.",
        500
    )

    if creds is None:
        return email_body, "simulated"

    try:
        from googleapiclient.discovery import build
        service = build('gmail', 'v1', credentials=creds)

        msg            = MIMEMultipart('alternative')
        msg['subject'] = 'Your MedAgent Lab Results Summary'
        msg['to']      = patient_email
        msg['from']    = 'me'

        # Plain text version
        msg.attach(MIMEText(email_body, 'plain'))

        # HTML version
        html_body = email_body.replace('\n', '<br>')
        msg.attach(MIMEText(f"<html><body><p>{html_body}</p></body></html>", 'html'))

        raw     = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent    = service.users().messages().send(userId='me', body={'raw': raw}).execute()
        return email_body, f"sent (id: {sent['id']})"
    except Exception as e:
        print(f"Gmail error: {e}")
        return email_body, "generated (send failed)"

def tool_slack_nudge(medications, slack_webhook):
    """Post real Slack message via webhook."""
    meds_str = ", ".join(medications) if medications else "your medications"
    message  = call_groq(
        f"Write a friendly morning Slack health reminder for a patient taking: {meds_str}\n"
        "Under 80 words. Include 1-2 emojis. Remind them to take meds and one health tip.", 150
    )

    if not slack_webhook:
        return message, "simulated"

    try:
        import requests as req
        r = req.post(slack_webhook, json={"text": message}, timeout=5)
        status = "sent" if r.status_code == 200 else f"failed ({r.status_code})"
        return message, status
    except Exception as e:
        return message, f"failed ({e})"

def tool_drug_interactions(medications):
    """Check drug interactions."""
    interactions = []
    meds_sorted  = sorted(medications)
    checked      = set()
    for i, m1 in enumerate(meds_sorted):
        for m2 in meds_sorted[i+1:]:
            pair = tuple(sorted([m1, m2]))
            if pair in checked: continue
            checked.add(pair)
            note = KNOWN_INTERACTIONS.get(pair)
            if note:
                interactions.append(f"{pair[0]} + {pair[1]}: {note}")
    if not interactions:
        interactions.append("No major interactions detected among current medications.")
    return interactions

def module05_agent(lab_values, ner_result, simplification):
    """MCP Agent — dispatches all 4 tools with real integrations."""
    medications  = list({e["text"].title() for e in ner_result.get("chemicals",[]) if len(e["text"])>4})[:6]
    critical     = [lv for lv in lab_values if lv["severity"]=="Critical"]
    creds        = get_google_creds()
    slack_webhook= os.environ.get("SLACK_WEBHOOK_URL", "")
    patient_email= os.environ.get("PATIENT_EMAIL", "")
    tool_calls   = []

    # Tool 1: Google Calendar
    events, cal_status = tool_google_calendar(medications, critical, creds)
    tool_calls.append({"tool":"google_calendar","status":cal_status,
                       "output":f"{len(events)} events {'created in Google Calendar' if cal_status=='success' else 'simulated'}"})

    # Tool 2: Drug interactions
    interactions = tool_drug_interactions(medications)
    tool_calls.append({"tool":"drug_interaction_search","status":"success",
                       "output":f"{len(interactions)} interactions checked"})

    # Tool 3: Gmail
    email_body, email_status = tool_gmail_send(patient_email, lab_values, simplification, creds)
    tool_calls.append({"tool":"gmail_send","status":"success" if "sent" in email_status else "simulated",
                       "output":email_status})

    # Tool 4: Slack
    slack_msg, slack_status = tool_slack_nudge(medications, slack_webhook)
    tool_calls.append({"tool":"slack_nudge","status":"success" if slack_status=="sent" else "simulated",
                       "output":slack_status})

    return {
        "tool_calls":        tool_calls,
        "calendar_events":   events,
        "drug_interactions": interactions,
        "email_draft":       email_body,
        "slack_message":     slack_msg,
        "medications_found": medications,
    }

# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.get("/app", response_class=HTMLResponse)
def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Frontend not found</h1>")

# ── API endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status":"MedAgent Full Pipeline","version":"3.0.0","model":GROQ_MODEL}

@app.get("/health")
def health(): return {"status":"ok","model":GROQ_MODEL,
                      "google":get_google_creds() is not None,
                      "slack":bool(os.environ.get("SLACK_WEBHOOK_URL"))}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "PDF files only")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        ingestion      = module01_extract(tmp_path)
        text           = ingestion["full_text"]
        if len(text) < 50: raise HTTPException(400, "Could not extract text from PDF")
        ner_result     = module02_ner(text)
        lab_result     = module03_parse_labs(text)
        lab_values     = lab_result["lab_values"]
        simplification = module04_simplify(lab_values, ner_result)
        agent_result   = module05_agent(lab_values, ner_result, simplification)

        return JSONResponse({
            "success":    True,
            "pipeline": {
                "module01": {"doc_type":ingestion["doc_type"],"pages":ingestion["page_count"],"chars":ingestion["total_chars"]},
                "module02": {"entities":ner_result["total"],"chemicals":len(ner_result["chemicals"]),"diseases":len(ner_result["diseases"]),"model":ner_result["model_used"]},
                "module03": lab_result["summary"],
                "module04": {"findings":len(simplification["findings"]),"model":simplification["model_used"]},
                "module05": {"tools":len(agent_result["tool_calls"]),"events":len(agent_result["calendar_events"])},
            },
            "lab_values": lab_values,
            "ner_result": ner_result,
            "report":     simplification,
            "agent":      agent_result,
        })
    finally:
        os.unlink(tmp_path)
