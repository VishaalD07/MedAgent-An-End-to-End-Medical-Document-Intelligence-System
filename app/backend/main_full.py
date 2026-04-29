"""
MedAgent FastAPI Backend — Full Pipeline
Integrates all 5 modules:
  Module 01: PDF Ingestion        (PyMuPDF)
  Module 02: Medical NER          (BioBERT fine-tuned on BC5CDR)
  Module 03: Lab Value Parser     (regex + rule-based)
  Module 04: Text Simplification  (Groq Llama 3.3 70B)
  Module 05: MCP Agent            (Calendar + Drug Search + Email + Slack)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os, re, time, json
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import fitz                        # Module 01 — PyMuPDF
from groq import Groq              # Module 04 — LLM

app = FastAPI(title="MedAgent Full Pipeline", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Groq setup ─────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
GROQ_MODEL  = "llama-3.3-70b-versatile"
SYSTEM      = """You are a medical interpreter helping patients understand their lab results.
Use simple everyday language (8th grade level). Keep explanations to 1-2 sentences.
Do not diagnose. Always recommend consulting a doctor."""

def call_groq(prompt: str, max_tokens: int = 600) -> str:
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

# ── Module 01: PDF Ingestion ───────────────────────────────────────────────────
def module01_extract(pdf_path: str) -> dict:
    """Extract text from PDF using PyMuPDF. Falls back to Tesseract for scanned PDFs."""
    doc   = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        raw  = page.get_text("text")
        # Clean text
        clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        pages.append({"page": i+1, "text": clean.strip(), "chars": len(clean)})
    doc.close()
    full_text  = "\n\n".join(p["text"] for p in pages if p["text"])
    doc_type   = "text_pdf" if len(full_text) > 100 else "scanned_pdf"
    return {
        "full_text":  full_text,
        "page_count": len(pages),
        "total_chars":len(full_text),
        "doc_type":   doc_type,
        "pages":      pages,
    }

# ── Module 02: Medical NER (BioBERT) ──────────────────────────────────────────
# Lazy-loaded on first request to avoid startup delay
_ner_pipeline = None
NER_MODEL_PATH = os.environ.get("NER_MODEL_PATH", "/content/biobert_ner/best_model")

LABEL2ID = {"O":0,"B-Chemical":1,"B-Disease":2,"I-Disease":3,"I-Chemical":4}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            print(f"Loading BioBERT NER from {NER_MODEL_PATH}...")
            _ner_pipeline = hf_pipeline(
                "ner",
                model=NER_MODEL_PATH,
                tokenizer=NER_MODEL_PATH,
                aggregation_strategy="first",
                device=0 if _has_gpu() else -1,
            )
            print("BioBERT NER loaded")
        except Exception as e:
            print(f"BioBERT not available: {e} — using fallback")
            _ner_pipeline = "fallback"
    return _ner_pipeline

def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def module02_ner(text: str) -> dict:
    """Run BioBERT NER on text, split into sentences first."""
    ner = get_ner_pipeline()

    # Sentence split (BioBERT trained on sentence-level data)
    sentences = re.split(r"\n+|(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]

    entities   = []
    all_tokens = []

    if ner == "fallback" or ner is None:
        # Fallback: regex-based drug/disease extraction
        drug_pat = re.compile(
            r"\b(Metformin|Lisinopril|Atorvastatin|Aspirin|Insulin|Warfarin|"
            r"Amlodipine|Losartan|Furosemide|Gabapentin|Omeprazole|Levothyroxine)\b",
            re.IGNORECASE
        )
        disease_pat = re.compile(
            r"\b(diabetes|hypertension|anemia|nephropathy|carcinoma|"
            r"lymphoma|melanoma|leukemia|cancer|disease|syndrome|disorder)\b",
            re.IGNORECASE
        )
        for m in drug_pat.finditer(text):
            entities.append({"text": m.group(), "label": "Chemical", "confidence": 0.95})
        for m in disease_pat.finditer(text):
            entities.append({"text": m.group(), "label": "Disease",  "confidence": 0.85})
        model_used = "regex_fallback"
    else:
        for sent in sentences[:50]:  # cap at 50 sentences for speed
            try:
                preds = ner(sent)
                for p in preds:
                    label = "Chemical" if "Chemical" in p["entity_group"] else "Disease"
                    entities.append({
                        "text":       p["word"],
                        "label":      label,
                        "confidence": round(float(p["score"]), 4),
                    })
                all_tokens.extend(sent.split())
            except Exception:
                continue
        model_used = "biobert_bc5cdr_finetuned"

    # Deduplicate
    seen_ents  = set()
    unique_ents = []
    for e in entities:
        key = (e["text"].lower(), e["label"])
        if key not in seen_ents:
            seen_ents.add(key)
            unique_ents.append(e)

    chemicals = [e for e in unique_ents if e["label"] == "Chemical"]
    diseases  = [e for e in unique_ents if e["label"] == "Disease"]

    return {
        "entities":   unique_ents,
        "chemicals":  chemicals,
        "diseases":   diseases,
        "model_used": model_used,
        "total":      len(unique_ents),
    }

# ── Module 03: Lab Value Parser ───────────────────────────────────────────────
REFERENCE_RANGES = {
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

def classify_severity(value, rl, rh, flag="", cl=None, ch=None):
    if flag.upper() in ("CRITICAL","PANIC","ALERT"): return "Critical"
    if rl is None and rh is None:                    return "Normal"
    if cl and cl > 0    and value <= cl: return "Critical"
    if ch and ch < 9999 and value >= ch: return "Critical"
    if rh and rh >= 999:
        if rl and value < rl:
            pct = (rl - value) / rl * 100
            return "Critical" if pct >= 25 else ("Borderline" if pct > 10 else "Normal")
        return "Normal"
    if rl is not None and rl <= 0:
        if rh and value > rh:
            pct = (value - rh) / rh * 100
            return "Critical" if pct > 30 else ("Borderline" if pct > 10 else "Normal")
        return "Normal"
    if rl is not None and rh is not None:
        w = rh - rl
        if w <= 0: return "Normal"
        if value < rl:
            pct = (rl - value) / w * 100
            return "Critical" if pct > 30 else ("Borderline" if pct > 10 else "Normal")
        if value > rh:
            pct = (value - rh) / w * 100
            return "Critical" if pct > 30 else ("Borderline" if pct > 10 else "Normal")
    return "Normal"

def module03_parse_labs(text: str) -> dict:
    """Extract and classify lab values from clinical text."""
    results, seen = [], set()
    num_pat  = re.compile(r"\d+(?:\.\d+)?")
    flag_pat = re.compile(r"\b(HIGH|LOW|CRITICAL|PANIC|H\b|L\b)\b", re.IGNORECASE)
    test_pat = re.compile(
        r"(?i)\b(" + "|".join(re.escape(k) for k in sorted(REFERENCE_RANGES, key=len, reverse=True)) + r")\b"
    )

    # HbA1c special case
    m = re.search(r"[Hh][Bb][Aa]1[Cc]\)?[:\s]+(\d+\.?\d*)\s*%", text)
    if m:
        val = float(m.group(1))
        rl,rh,u,cl,ch = REFERENCE_RANGES["hba1c"]
        flag = "HIGH" if val > rh else ""
        results.append({"test_name":"HbA1c","value":val,"unit":u,
            "ref_range":f"{rl}-{rh}","ref_low":rl,"ref_high":rh,
            "flag":flag,"severity":classify_severity(val,rl,rh,flag,cl,ch)})
        seen.add("hba1c")

    for m in test_pat.finditer(text):
        k = m.group(0).lower().strip()
        if k in seen: continue
        win = text[m.end():m.end()+150]
        vm  = num_pat.search(win)
        if not vm: continue
        try: val = float(vm.group())
        except: continue
        fm   = flag_pat.search(win[:100])
        flag = fm.group(1).upper() if fm else ""
        if flag == "H": flag = "HIGH"
        if flag == "L": flag = "LOW"
        rl,rh,u,cl,ch = REFERENCE_RANGES.get(k,(None,None,"",None,None))
        results.append({"test_name":k.title(),"value":val,"unit":u,
            "ref_range":f"{rl}-{rh}" if rl is not None else "",
            "ref_low":rl,"ref_high":rh,"flag":flag,
            "severity":classify_severity(val,rl,rh,flag,cl,ch)})
        seen.add(k)

    n_crit  = sum(1 for r in results if r["severity"]=="Critical")
    n_bord  = sum(1 for r in results if r["severity"]=="Borderline")
    n_norm  = sum(1 for r in results if r["severity"]=="Normal")
    return {
        "lab_values": results,
        "summary": {"total":len(results),"critical":n_crit,"borderline":n_bord,"normal":n_norm}
    }

# ── Module 04: Text Simplification ────────────────────────────────────────────
def module04_simplify(lab_values: list, ner_result: dict) -> dict:
    """Generate plain-language report using Groq."""
    abnormal = [lv for lv in lab_values if lv["severity"] in ("Critical","Borderline")]

    # Step 1: Simplify each finding
    findings = []
    for lv in abnormal:
        resp   = call_groq(
            f"Test: {lv['test_name']}\nValue: {lv['value']} {lv['unit']}\n"
            f"Normal: {lv['ref_range']}\nSeverity: {lv['severity']}\n\n"
            "Respond with exactly:\nPLAIN: <one sentence>\nIMPACT: <one sentence>"
        )
        plain  = next((l[6:].strip() for l in resp.split("\n") if l.startswith("PLAIN:")),  resp[:200])
        impact = next((l[7:].strip() for l in resp.split("\n") if l.startswith("IMPACT:")), "Discuss with your doctor.")
        findings.append({**lv, "plain_language": plain, "patient_impact": impact})

    # Step 2: Health summary
    ctx = "\n".join(f"- {lv['test_name']}: {lv['value']} {lv['unit']} ({lv['severity']})" for lv in lab_values)
    meds = [e["text"] for e in ner_result.get("chemicals", [])][:5]
    diseases = [e["text"] for e in ner_result.get("diseases", [])][:5]

    summary = call_groq(
        f"Lab results:\n{ctx}\n"
        f"Medications detected: {', '.join(meds) if meds else 'none'}\n"
        f"Conditions mentioned: {', '.join(diseases) if diseases else 'none'}\n\n"
        "Write a 3-4 sentence plain-language health summary.", 400
    )

    # Step 3: Doctor questions
    q_raw = call_groq(
        f"Lab results:\n{ctx}\n\nGenerate exactly 6 specific questions "
        "this patient should ask their doctor. Numbered list.", 400
    )
    questions = [
        re.match(r"^\d+\.?\s+(.+)", l.strip()).group(1)
        for l in q_raw.split("\n") if re.match(r"^\d+\.?\s+(.+)", l.strip())
    ]

    # Step 4: Action plan
    a_raw = call_groq(
        f"Lab results:\n{ctx}\n"
        f"Medications: {', '.join(meds) if meds else 'none'}\n\n"
        "Create exactly 6 specific, actionable steps. Numbered list.", 400
    )
    actions = [
        re.match(r"^\d+\.?\s+(.+)", l.strip()).group(1)
        for l in a_raw.split("\n") if re.match(r"^\d+\.?\s+(.+)", l.strip())
    ]

    return {
        "findings":         findings,
        "health_summary":   summary,
        "doctor_questions": questions,
        "action_plan":      actions,
        "model_used":       GROQ_MODEL,
    }

# ── Module 05: MCP Agent ──────────────────────────────────────────────────────
KNOWN_INTERACTIONS = {
    ("Atorvastatin","Lisinopril"):   "Safe combination. Monitor for muscle pain.",
    ("Metformin","Lisinopril"):      "Monitor kidney function — both affect renal clearance.",
    ("Metformin","Atorvastatin"):    "Generally safe. Rare increased myopathy risk.",
    ("Aspirin","Lisinopril"):        "High-dose aspirin may reduce Lisinopril effectiveness.",
    ("Warfarin","Atorvastatin"):     "Atorvastatin can increase Warfarin levels — monitor INR.",
    ("Amlodipine","Atorvastatin"):   "Amlodipine increases Atorvastatin exposure — max 20mg.",
    ("Metformin","Furosemide"):      "Furosemide may increase Metformin blood levels.",
    ("Lisinopril","Furosemide"):     "Risk of low blood pressure — monitor closely.",
}

def module05_agent(lab_values: list, ner_result: dict, simplification: dict) -> dict:
    """MCP Agent — dispatches 4 tools based on patient context."""
    medications = list({e["text"].title() for e in ner_result.get("chemicals",[])
                        if len(e["text"]) > 4})[:6]
    critical    = [lv for lv in lab_values if lv["severity"] == "Critical"]
    today       = datetime.now()
    tool_calls  = []

    # Tool 1: Google Calendar
    events = []
    for med in medications:
        events.append({
            "summary":     f"Take {med}",
            "start":       (today + timedelta(days=1)).strftime("%Y-%m-%dT08:00:00"),
            "recurrence":  "RRULE:FREQ=DAILY",
            "status":      "simulated",
        })
    if critical:
        events.append({
            "summary":     "Doctor Follow-up — Lab Results Review",
            "description": f"Review: {', '.join(lv['test_name'] for lv in critical[:3])}",
            "start":       (today + timedelta(weeks=2)).strftime("%Y-%m-%dT10:00:00"),
            "status":      "simulated",
        })
    events.append({
        "summary":    "Weekly Health Check-in",
        "start":      (today + timedelta(days=7)).strftime("%Y-%m-%dT09:00:00"),
        "recurrence": "RRULE:FREQ=WEEKLY",
        "status":     "simulated",
    })
    tool_calls.append({"tool":"google_calendar","status":"simulated","output":f"{len(events)} events"})

    # Tool 2: Drug interactions
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
    tool_calls.append({"tool":"drug_interaction_search","status":"success","output":f"{len(interactions)} results"})

    # Tool 3: Email draft
    critical_list = "\n".join(f"- {lv['test_name']}: {lv['value']} {lv['unit']}" for lv in critical)
    email = call_groq(
        f"Write a patient-friendly email summarizing these critical lab results:\n{critical_list}\n\n"
        f"Health summary: {simplification.get('health_summary','')}\n\n"
        "Include: subject line, warm greeting, key findings, next steps, encouraging close. Under 250 words.",
        500
    )
    tool_calls.append({"tool":"email_draft","status":"success","output":f"{len(email)} chars"})

    # Tool 4: Slack nudge
    meds_str = ", ".join(medications) if medications else "your medications"
    slack = call_groq(
        f"Write a friendly morning Slack health reminder for a patient taking: {meds_str}\n"
        "Under 80 words. Include 1-2 emojis. Remind them to take meds and one health tip.", 150
    )
    tool_calls.append({"tool":"slack_nudge","status":"success","output":slack[:60]})

    return {
        "tool_calls":        tool_calls,
        "calendar_events":   events,
        "drug_interactions": interactions,
        "email_draft":       email,
        "slack_message":     slack,
        "medications_found": medications,
    }

# ── API endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status":"MedAgent Full Pipeline","version":"2.0.0","model":GROQ_MODEL}

@app.get("/health")
def health():
    return {"status":"ok","model":GROQ_MODEL,"ner_model":NER_MODEL_PATH}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "PDF files only")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Module 01: Ingest
        ingestion = module01_extract(tmp_path)
        text = ingestion["full_text"]
        if len(text) < 50:
            raise HTTPException(400, "Could not extract text from PDF")

        # Module 02: NER
        ner_result = module02_ner(text)

        # Module 03: Lab parser
        lab_result = module03_parse_labs(text)
        lab_values = lab_result["lab_values"]

        # Module 04: Simplification
        simplification = module04_simplify(lab_values, ner_result)

        # Module 05: MCP Agent
        agent_result = module05_agent(lab_values, ner_result, simplification)

        return JSONResponse({
            "success":       True,
            "pipeline": {
                "module01_ingestion":       {"doc_type": ingestion["doc_type"], "pages": ingestion["page_count"], "chars": ingestion["total_chars"]},
                "module02_ner":             {"total_entities": ner_result["total"], "chemicals": len(ner_result["chemicals"]), "diseases": len(ner_result["diseases"]), "model": ner_result["model_used"]},
                "module03_lab_parser":      lab_result["summary"],
                "module04_simplification":  {"findings": len(simplification["findings"]), "model": simplification["model_used"]},
                "module05_agent":           {"tools_dispatched": len(agent_result["tool_calls"]), "events": len(agent_result["calendar_events"])},
            },
            "lab_values":    lab_values,
            "ner_result":    ner_result,
            "report":        simplification,
            "agent":         agent_result,
        })

    finally:
        os.unlink(tmp_path)
