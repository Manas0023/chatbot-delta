from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, json, re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key)

# ─── In-memory conversation store ────────────────────────────────────────────
# Key: session_id (phone number or any unique string from frontend)
# Value: list of LangChain messages
conversation_store: dict = {}

MAX_HISTORY = 10  # keep last 10 exchanges to avoid token bloat

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful customer support assistant for Deltaa Material & Lab Solutions (deltaa.in).

About Deltaa:
- NABL-accredited testing and analysis laboratory in India.
- Tests metals, alloys, and materials for: Mechanical, Chemical, Metallographic, and Corrosion properties.
- Customers submit samples with a DC (Delivery Challan). They receive:
  - MRN (Material Receipt Number) — format: MRN/2026/XXXX  (e.g. MRN/2026/0003)
  - Job Number — format: J/2026/XXXX  (e.g. J/2026/3431)
- Contact: info@deltaa.in | Website: deltaa.in

--- TESTS WE OFFER (use this to answer "do you do X test?") ---

MECHANICAL TESTS:
- Tensile/UTS Testing: UTS, Yield Strength (YS), Elongation (%El), Reduction of Area (RA) — as per ASTM E8, IS 1608, ISO 6892-1, ASME IX
- Hardness Testing: Rockwell (HRC), Brinell (BHN), Vickers (HV) — as per ASTM E18, E10, E92, IS 1500, IS 1586, ISO 6506
- Impact/Charpy Testing — as per ASTM E23, ISO 148-1
- Bend Testing — as per ASTM E290, IS 1599, ASME IX
- Compression Test of Concrete Cubes — IS 516
- Hardenability (Jominy End-Quench Test) — ASTM A255
- Proof Load Test — ASTM A370
- Elevated Temperature Tensile Test — ASTM E21, ISO 6892-2
- Weld Tensile, Weld Bend, Weld Macro Tests — ASME IX

CHEMICAL TESTS:
- OES/Spectro Chemical Analysis (C, Mn, Si, P, S, Cr, Ni, Mo, Cu, V, Al, Ti, Nb, Co, N, etc.)
  Standards: ASTM E1086, ASTM E415, ASTM E1251, IS 8811, JIS G1253
- Carbon & Sulphur Analysis — ASTM E1019
- ICP Chemical Analysis — JIS G1258, ASTM E2594, ASTM E3061
- Portable PMI (Positive Material Identification) — ASTM E1476
- Chemical Test for Cast Iron — ASTM E1999, IS 15338
- Chemical Test for Copper Alloys — BS EN 15079
- Water & Wastewater Analysis (pH, TDS, heavy metals, BOD, COD, etc.) — IS 3025 / IS 10500, APHA

METALLOGRAPHY TESTS:
- Grain Size Measurement — ASTM E112
- Micro Examination / Microstructure Analysis — ASM Handbook Vol.9 & ASTM E407
- Macro Examination — ASTM E381, ASTM E340
- Inclusion Rating (cleanliness) — ASTM E45 Method A & D, DIN 50602
- IGC (Intergranular Corrosion) Practice A — ASTM A262 Pr.A
- Volume Fraction of Phases — ASTM E562
- Case Depth (Microscopic & Hardness Methods) — IS 6416
- Ferrite Content by Feritiscope — ASTM A800
- Cast Iron Microstructure — ISO 945-1, IS 7754, ASTM A247
- Intermetallic Phase Detection — ASTM A923 Method A

CORROSION TESTS:
- IGC Practice B & E — ASTM A262 Pr.B, Pr.E
- Pitting Corrosion Test — ASTM G48 Method A
- Crevice Corrosion Test — ASTM G48 Method B
- Salt Spray Test — ASTM B117
- Ferric Chloride Corrosion Test — ASTM A923 Method C
- IGC Method A (High Nickel Alloys) — ASTM G28 Method A

Common test name mappings (customer may use informal names):
- "UTS test" or "tensile test" → Mechanical (Tensile)
- "hardness test" → Mechanical (Hardness)
- "impact test" or "charpy test" → Mechanical (Impact)
- "chemical test" or "composition" or "spectro" or "OES" or "PMI" → Chemical
- "microstructure" or "grain size" or "inclusion" or "macro" → Metallography
- "IGC" or "corrosion test" or "salt spray" or "pitting" → Corrosion

--- TURNAROUND TIMES (based on actual completed jobs) ---
- Chemical only: ~2 working days
- Mechanical only: ~2 working days
- Metallography only: ~1 working day
- Chemical + Mechanical: ~4 working days
- Corrosion (any): ~5 working days
- All 4 departments: ~9 working days
- Overall average: ~3–4 working days
- Working days only (weekends and holidays not counted)

--- JOB STATUS MEANINGS ---
- Testing: Sample is currently being tested in the lab.
- PendingApproval: Testing complete, awaiting internal technical review.
- ResultsApproved / ResultsVerified: Results reviewed and approved internally.
- Verified: Report verified by technical team.
- Approved: Report fully approved, will be sent to customer by email.

--- STRICT RULES ---
1. NEVER share test results, numerical values, or pass/fail verdicts — status only.
2. If asked for results: "Test results are shared only via the official report. Please check your email or contact info@deltaa.in."
3. If no MRN/Job Number given, ask for it before looking up.
4. Be concise, friendly, professional. Use simple English.
5. If unsure about anything, direct to info@deltaa.in."""

# ─── Database Helpers ─────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "deltaa.db.json")

def load_jobs():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH) as f:
        return json.load(f).get("jobs", [])

def get_job_by_id(query: str):
    """Match MRN or Job Number — exact or partial."""
    q = query.upper().strip()
    jobs = load_jobs()
    # Exact match first
    for job in jobs:
        if q in (job.get("mrn", "").upper(), job.get("jobNo", "").upper()):
            return job
    # Partial match (e.g. user types "0003" or "3431")
    for job in jobs:
        if q in job.get("mrn", "").upper() or q in job.get("jobNo", "").upper():
            return job
    return None

def get_job_by_customer_name(name: str):
    """Fuzzy search by customer name (case-insensitive partial match)."""
    name_lower = name.lower().strip()
    jobs = load_jobs()
    matches = [j for j in jobs if name_lower in j.get("customerName", "").lower()]
    if len(matches) == 1:
        return matches[0], None  # single match
    elif len(matches) > 1:
        return None, matches    # multiple matches — need more info
    return None, None           # not found

def extract_job_id(message: str):
    """Extract MRN or Job Number from message — handles many formats."""
    msg_up = message.upper()

    # Full MRN: MRN/2026/0001 or MRN 2026 0001
    m = re.search(r'MRN[/\s\-]+(\d{4})[/\s\-]+(\d+)', msg_up)
    if m:
        return f"MRN/{m.group(1)}/{m.group(2).zfill(4)}"

    # Full Job No: J/2026/3431
    m = re.search(r'J[/\s\-]+(\d{4})[/\s\-]+(\d+)', msg_up)
    if m:
        return f"J/{m.group(1)}/{m.group(2)}"

    # Standalone 4+ digit number (treat as partial MRN/Job No)
    m = re.search(r'\b(\d{4,})\b', message)
    if m:
        return m.group(1)

    return None

def format_job_status(job: dict) -> str:
    """Build a context string for the LLM — status info only, no results."""
    # Extract departments from samples
    depts = []
    for s in job.get('samples', []):
        depts.extend(s.get('depts', []))
    depts = sorted(set(depts))
    depts_str = ', '.join(depts) if depts else 'Not specified'

    return (
        f"Job found in database. "
        f"MRN: {job.get('mrn')}, "
        f"Job No: {job.get('jobNo')}, "
        f"Customer: {job.get('customerName')}, "
        f"Date Received: {job.get('dateReceived')}, "
        f"Priority: {job.get('priority')}, "
        f"Departments being tested: {depts_str}, "
        f"Current Status: {job.get('status')}. "
        f"Tell the customer their current status in a friendly, clear way. "
        f"If they ask about time, use the department info above to give a specific estimate. "
        f"Do NOT mention or share any test results, values, or pass/fail verdict."
    )

# ─── Request / Response Models ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # unique per user/phone number

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Deltaa chatbot v2 is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message.strip()

    # Get or create conversation history for this session
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    history = conversation_store[session_id]

    # ── Step 1: Try to extract a job ID ──
    job_id = extract_job_id(user_message)
    db_context = None

    if job_id:
        job = get_job_by_id(job_id)
        if job:
            db_context = format_job_status(job)
        else:
            db_context = (
                f"Customer mentioned ID '{job_id}' but no matching job was found. "
                f"Politely inform them and ask them to double-check their MRN "
                f"(format: MRN/2026/XXXX) from the receipt they received."
            )

    # ── Step 2: If no ID, try customer name lookup ──
    if not job_id:
        # Only try name lookup if message has a meaningful company-like word (>= 4 chars)
        common_words = {
            "what", "when", "will", "this", "that", "with", "from", "have", "pass",
            "fail", "test", "does", "your", "about", "status", "sample", "help",
            "hello", "hey", "please", "thanks", "okay", "sure", "sent", "also",
            "did", "ready", "done", "result", "report", "need", "like", "also",
            "want", "know", "check", "tell", "update", "pending", "give"
        }
        words = [w.strip("?!.,") for w in user_message.split()
                 if len(w.strip("?!.,")) >= 4 and w.lower().strip("?!.,") not in common_words]
        for word in words:
            try:
                job, multi = get_job_by_customer_name(word)
                if job:
                    db_context = format_job_status(job)
                    break
                elif multi:
                    names = ", ".join(j.get("customerName", "") for j in multi[:3])
                    db_context = (
                        f"Multiple jobs found matching '{word}': {names}. "
                        f"Ask the customer to provide their specific MRN number to narrow it down."
                    )
                    break
            except Exception:
                continue

    # ── Step 3: Build messages for LLM ──
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add conversation history (trimmed to MAX_HISTORY exchanges)
    for msg in history[-(MAX_HISTORY * 2):]:
        messages.append(msg)

    # Add current user message (inject DB context if found)
    if db_context:
        augmented = f"[DB lookup result: {db_context}]\n\nCustomer message: {user_message}"
    else:
        augmented = user_message

    messages.append(HumanMessage(content=augmented))

    # ── Step 4: Call LLM ──
    response = llm.invoke(messages)
       # ── Step 4: Call LLM ──
    response = llm.invoke(messages)
    if isinstance(response.content, list):
        text = ""
        for part in response.content:
            if isinstance(part, dict):
                text += part.get("text", "")
            elif isinstance(part, str):
                text += part
    else:
        text = response.content

    # ── Step 5: Save to history ──
    history.append(HumanMessage(content=user_message))  # save original, not augmented
    history.append(AIMessage(content=text))
    conversation_store[session_id] = history

    return {"reply": text, "session_id": session_id}

@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session (optional utility)."""
    if session_id in conversation_store:
        del conversation_store[session_id]
    return {"message": f"Session {session_id} cleared."}
