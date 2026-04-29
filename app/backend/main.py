"""
MedAgent FastAPI Backend — Groq Edition
Runs the full pipeline: PDF ingestion → lab parsing → Groq LLM simplification
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os, re, time
from groq import Groq
import fitz

app = FastAPI(title="MedAgent API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Groq setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client  = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a medical interpreter helping patients understand their lab results.
Use simple language (8th grade level). Keep explanations to 1-2 sentences.
Do not diagnose. Always recommend consulting a doctor."""

def call_groq(prompt, max_tokens=600):
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL, temperature=0.3, max_tokens=max_tokens,
                messages=[{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content":prompt}]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (attempt+1)*10
                print(f"Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"[Error: {e}]"
    return "[Error: max retries]"

def extract_text(pdf_path):
    doc  = fitz.open(pdf_path)
    text = "\n".join(p.get_text("text") for p in doc)
    doc.close()
    return text.strip()

REFERENCE_RANGES = {
    "hemoglobin":    (13.5, 17.5, "g/dL",         7.0,  20.0),
    "hgb":           (13.5, 17.5, "g/dL",         7.0,  20.0),
    "hematocrit":    (41.0, 53.0, "%",            20.0,  60.0),
    "wbc":           (4.5,  11.0, "10^3/uL",       2.0,  30.0),
    "platelet":      (150,  400,  "10^3/uL",      50.0, 1000.0),
    "glucose":       (70,   99,   "mg/dL",        40.0,  500.0),
    "creatinine":    (0.74, 1.35, "mg/dL",         0.3,  10.0),
    "egfr":          (60,   999,  "mL/min/1.73m2",15.0, 9999.0),
    "sodium":        (136,  145,  "mEq/L",       120.0,  160.0),
    "potassium":     (3.5,  5.1,  "mEq/L",         2.5,   6.5),
    "ldl":           (0,    100,  "mg/dL",          0.0,  400.0),
    "hdl":           (40,   9999, "mg/dL",         10.0, 9999.0),
    "hba1c":         (0,    5.7,  "%",              0.0,  15.0),
    "tsh":           (0.4,  4.0,  "mIU/L",          0.1,  10.0),
    "alt":           (7,    56,   "U/L",            0.0,  500.0),
    "ast":           (10,   40,   "U/L",            0.0,  500.0),
    "triglycerides": (0,    150,  "mg/dL",          0.0, 1000.0),
    "cholesterol":   (0,    200,  "mg/dL",          0.0,  600.0),
}

def classify_severity(value, ref_low, ref_high, flag="", crit_low=None, crit_high=None):
    if flag.upper() in ("CRITICAL","PANIC","ALERT"): return "Critical"
    if ref_low is None and ref_high is None:         return "Normal"
    if crit_low  and crit_low  > 0    and value <= crit_low:  return "Critical"
    if crit_high and crit_high < 9999 and value >= crit_high: return "Critical"
    if ref_high and ref_high >= 999:
        if ref_low and value < ref_low:
            pct = (ref_low - value) / ref_low * 100
            return "Critical" if pct >= 25 else ("Borderline" if pct > 10 else "Normal")
        return "Normal"
    if ref_low is not None and ref_low <= 0:
        if ref_high and value > ref_high:
            pct = (value - ref_high) / ref_high * 100
            return "Critical" if pct > 30 else ("Borderline" if pct > 10 else "Normal")
        return "Normal"
    if ref_low is not None and ref_high is not None:
        w = ref_high - ref_low
        if w <= 0: return "Normal"
        if value < ref_low:
            pct = (ref_low - value) / w * 100
            return "Critical" if pct > 30 else ("Borderline" if pct > 10 else "Normal")
        if value > ref_high:
            pct = (value - ref_high) / w * 100
            return "Critical" if pct > 30 else ("Borderline" if pct > 10 else "Normal")
    return "Normal"

def parse_lab_values(text):
    results, seen = [], set()
    num_pat  = re.compile(r"\d+(?:\.\d+)?")
    flag_pat = re.compile(r"\b(HIGH|LOW|CRITICAL|PANIC|H\b|L\b)\b", re.IGNORECASE)
    test_names = sorted(REFERENCE_RANGES.keys(), key=len, reverse=True)
    test_pat   = re.compile(r"(?i)\b(" + "|".join(re.escape(n) for n in test_names) + r")\b")

    m = re.search(r"[Hh][Bb][Aa]1[Cc]\)?[:\s]+(\d+\.?\d*)\s*%", text)
    if m:
        val = float(m.group(1))
        rl,rh,unit,cl,ch = REFERENCE_RANGES["hba1c"]
        flag = "HIGH" if val > rh else ""
        results.append({"test_name":"HbA1c","value":val,"unit":unit,
            "ref_range":f"{rl}-{rh}","ref_low":rl,"ref_high":rh,
            "flag":flag,"severity":classify_severity(val,rl,rh,flag,cl,ch)})
        seen.add("hba1c")

    for m in test_pat.finditer(text):
        key = m.group(0).lower().strip()
        if key in seen: continue
        window = text[m.end():m.end()+150]
        vm = num_pat.search(window)
        if not vm: continue
        try: value = float(vm.group())
        except: continue
        fm   = flag_pat.search(window[:100])
        flag = fm.group(1).upper() if fm else ""
        if flag == "H": flag = "HIGH"
        if flag == "L": flag = "LOW"
        rl,rh,unit,cl,ch = REFERENCE_RANGES.get(key,(None,None,"",None,None))
        results.append({"test_name":key.title(),"value":value,"unit":unit,
            "ref_range":f"{rl}-{rh}" if rl is not None else "",
            "ref_low":rl,"ref_high":rh,"flag":flag,
            "severity":classify_severity(value,rl,rh,flag,cl,ch)})
        seen.add(key)
    return results

def simplify_report(lab_values):
    abnormal = [lv for lv in lab_values if lv["severity"] in ("Critical","Borderline")]
    findings = []
    for lv in abnormal:
        prompt = (f"Test: {lv['test_name']}\nValue: {lv['value']} {lv['unit']}\n"
                  f"Normal: {lv['ref_range']}\nFlag: {lv['flag']} ({lv['severity']})\n\n"
                  "Respond:\nPLAIN: <one sentence>\nIMPACT: <one sentence>")
        resp   = call_groq(prompt)
        plain  = next((l[6:].strip() for l in resp.split("\n") if l.startswith("PLAIN:")),  resp[:200])
        impact = next((l[7:].strip() for l in resp.split("\n") if l.startswith("IMPACT:")), "Discuss with your doctor.")
        findings.append({**lv, "plain_language": plain, "patient_impact": impact})

    ctx = "\n".join(f"- {lv['test_name']}: {lv['value']} {lv['unit']} ({lv['severity']})" for lv in lab_values)
    summary   = call_groq(f"Lab results:\n{ctx}\n\nWrite a 3-4 sentence plain-language health summary.", 400)
    q_raw     = call_groq(f"Lab results:\n{ctx}\n\nGenerate 6 questions for the doctor. Numbered list.", 400)
    a_raw     = call_groq(f"Lab results:\n{ctx}\n\nCreate 6 specific action steps. Numbered list.", 400)
    questions = [re.match(r"^\d+\.?\s+(.+)", l.strip()).group(1) for l in q_raw.split("\n") if re.match(r"^\d+\.?\s+(.+)", l.strip())]
    actions   = [re.match(r"^\d+\.?\s+(.+)", l.strip()).group(1) for l in a_raw.split("\n") if re.match(r"^\d+\.?\s+(.+)", l.strip())]
    return {"findings":findings,"health_summary":summary,"doctor_questions":questions,"action_plan":actions}

@app.get("/")
def root(): return {"status":"MedAgent API running","model":GROQ_MODEL}

@app.get("/health")
def health(): return {"status":"ok","model":GROQ_MODEL}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "PDF files only")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
        if len(text) < 50: raise HTTPException(400, "Could not extract text from PDF")
        lab_values = parse_lab_values(text)
        report     = simplify_report(lab_values)
        return JSONResponse({"success":True,"lab_values":lab_values,"report":report,
                             "text_length":len(text),"model":GROQ_MODEL})
    finally:
        os.unlink(tmp_path)
