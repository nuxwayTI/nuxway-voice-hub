from flask import Flask, request, Response, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather, Dial
import os
import logging
import re
import requests
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime
import pytz
import time

logging.basicConfig(level=logging.INFO)
flask_app = Flask(__name__)

# =========================
# CONFIG (ENV)
# =========================
BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")

TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "es-MX").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "Polly.Andres-Neural").strip()
TTS_USE_SSML = os.getenv("TTS_USE_SSML", "0").strip() == "1"

MEM_TTL_SECONDS = int(os.getenv("MEM_TTL_SECONDS", "1200"))

# =========================
# ZOHO FLOW (Bigin) - DIRECTO DESDE IVR
# (NO tocamos tu Zoho Flow: enviamos payload compatible con el Flow que ya te funciona con ElevenLabs)
# =========================
ZOHO_FLOW_WEBHOOK_URL = os.getenv("ZOHO_FLOW_WEBHOOK_URL", "").strip()
ZOHO_SOURCE = os.getenv("ZOHO_SOURCE", "twilio-ivr").strip()
ZOHO_CITY_DEFAULT = os.getenv("ZOHO_CITY_DEFAULT", "Cochabamba").strip()

# =========================
# PHONE NORMALIZATION (CRM)
# - externo: +591... o número largo -> últimos 8 dígitos (Bolivia)
# - interno: extensión corta (ej 22 / 4000) -> map o fallback (para que NO vaya null)
# =========================
STRIP_COUNTRY_CODE = os.getenv("STRIP_COUNTRY_CODE", "1") == "1"
LOCAL_NUMBER_LEN = int(os.getenv("LOCAL_NUMBER_LEN", "8"))  # Bolivia local usually 8

EXT_TO_MOBILE = {
    "22":   os.getenv("EXT_22_MOBILE", "").strip(),
    "4000": os.getenv("EXT_4000_MOBILE", "").strip(),
    "4001": os.getenv("EXT_4001_MOBILE", "").strip(),
    "4002": os.getenv("EXT_4002_MOBILE", "").strip(),
    "4003": os.getenv("EXT_4003_MOBILE", "").strip(),
    "4007": os.getenv("EXT_4007_MOBILE", "").strip(),
}

# fallback general para internas sin mapeo (solo para pruebas/evitar null)
INTERNAL_DEFAULT_MOBILE = os.getenv("INTERNAL_DEFAULT_MOBILE", "").strip()

# =========================
# TWILIO SIP DOMAIN
# =========================
SIP_DOMAIN = os.getenv("SIP_DOMAIN", "nuxway.sip.twilio.com").strip()

DEFAULT_EXTERNAL_FALLBACK_CALLER = f"sip:ivr@{SIP_DOMAIN}"
DEFAULT_INTERNAL_FALLBACK_CALLER = os.getenv("DEFAULT_INTERNAL_FALLBACK_CALLER", "5109").strip()

INTERNAL_EXT_MIN_LEN = 2
INTERNAL_EXT_MAX_LEN = 6
EXTERNAL_NUM_MIN_LEN = 7
EXTERNAL_NUM_MAX_LEN = 15

# =========================
# RUTEO (destino SIP por interno)
# =========================
DID_MAP = {
    "pablo": "5100",
    "gonzalo": "5101",
    "vladimir": "5102",
    "paola": "5103",
    "ximena": "5107",
    "cola": "6049"
}

# =========================
# TUNING PARA GSM / AUDIO
# =========================
INITIAL_PAUSE_SEC = 1.2
RETRY_PAUSE_SEC = 0.8
GATHER_TIMEOUT = 15
SPEECH_TIMEOUT = "auto"
MAX_NOINPUT_ATTEMPTS = 3

# IMPORTANTE: Máximo 2 preguntas al LLM antes de "agente o colgar"
MAX_LLM_TURNS = int(os.getenv("MAX_LLM_TURNS", "2"))

MIN_CONFIDENCE_REPROMPT = 0.35

# =========================
# OPENAI
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions").strip()
session = requests.Session()

# =========================
# MEMORIA / LIMITES
# =========================
conversaciones = defaultdict(list)
llm_turns = defaultdict(int)
last_seen = defaultdict(lambda: 0.0)

# Lead capture state (por CallSid)
lead_stage = defaultdict(lambda: "")   # "" | "ask_name" | "ask_company"
lead_name = defaultdict(lambda: "")
lead_company = defaultdict(lambda: "")

def touch_call(call_sid: str):
    last_seen[call_sid] = time.time()

def gc_memory():
    now = time.time()
    dead = [sid for sid, ts in last_seen.items() if now - ts > MEM_TTL_SECONDS]
    for sid in dead:
        conversaciones.pop(sid, None)
        llm_turns.pop(sid, None)
        last_seen.pop(sid, None)
        lead_stage.pop(sid, None)
        lead_name.pop(sid, None)
        lead_company.pop(sid, None)
    if dead:
        logging.info(f"[GC] cleaned {len(dead)} call sessions")

# =========================
# PROMPT SISTEMA (ENV override)
# =========================
DEFAULT_SYSTEM_PROMPT = """
Eres el asistente telefónico de Nuxway Technology S.R.L.
Atiendes llamadas en español, tono profesional y humano, estilo IVR moderno.

Reglas:
- Respuestas cortas: 1 a 2 frases.
- Máximo 1 pregunta por turno.
- No suenes robótico.
- Si el usuario pide hablar con una persona específica (por nombre), transfiere con esa persona.
- Si el usuario pide soporte o un ingeniero (genérico), transfiere a soporte.
- No inventes datos. Si no estás seguro, deriva a soporte.

Información (empresa):
- Nuxway es distribuidor oficial de Yeastar.
- Vendemos equipos Yeastar y configuramos servidores de comunicaciones unificadas:
  - Yeastar Serie P y Yeastar Serie S.
  - Soluciones Cloud y On-Premise.
  - Referencia: yeastar.com

Infraestructura de red:
- Routers, firewalls, switches y access points (AP).
- Cableado estructurado.
- Soporte y mantenimiento de redes de datos IP.

Innovación y desarrollo:
- Área de innovación y desarrollo que crea aplicaciones web.
- Integraciones para mejorar las comunicaciones empresariales, especialmente con servidores Yeastar.

Si el usuario pregunta “qué hacen” o “servicios”:
- Responde con 1 resumen corto y ofrece ampliar si desea.

Si no estás seguro:
- “Para darle una respuesta correcta, prefiero comunicarlo con soporte.”
""".strip()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()

# =========================
# IMPORTANT: prefix por montaje en /twilio
# =========================
TWILIO_PREFIX = "/twilio"

# =========================
# HELPERS
# =========================
def _maybe_ssml(text: str) -> str:
    if not TTS_USE_SSML:
        return text
    safe = (text or "").replace("&", "y")
    return f'<speak><prosody rate="92%" volume="+2dB">{safe}</prosody></speak>'

def say(vr, text: str):
    vr.say(_maybe_ssml(text), language=TTS_LANGUAGE, voice=TTS_VOICE)

def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-záéíóúñ0-9 @._-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def normalize_digits(value: str) -> str:
    value = (value or "").strip()
    return re.sub(r"\D", "", value)

def to_local_digits(digits: str) -> str:
    if not digits:
        return ""
    if STRIP_COUNTRY_CODE and LOCAL_NUMBER_LEN > 0 and len(digits) > LOCAL_NUMBER_LEN:
        return digits[-LOCAL_NUMBER_LEN:]
    return digits

def is_valid_e164(s: str) -> bool:
    return bool(re.fullmatch(r"\+\d{8,15}", (s or "").strip()))

def is_valid_digits_number(s: str, min_len=2, max_len=15) -> bool:
    s = (s or "").strip()
    return bool(re.fullmatch(rf"\d{{{min_len},{max_len}}}", s))

def build_sip_uri(user: str) -> str:
    user = (user or "").strip()
    return f"sip:{user}@{SIP_DOMAIN}"

def extract_e164(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\+\d{8,15}", text)
    return m.group(0) if m else ""

def extract_digits_from_sip_from(tw_from: str) -> str:
    if not tw_from:
        return ""
    m = re.search(r"^sip:(\d+)@", tw_from.strip(), re.IGNORECASE)
    return m.group(1) if m else ""

def get_original_caller(tw_from: str, sip_headers: dict) -> str:
    num = extract_e164(tw_from)
    if is_valid_e164(num):
        return num

    for k in ["P-Asserted-Identity", "Remote-Party-ID", "X-Original-Caller", "X-ANI"]:
        num = extract_e164(sip_headers.get(k, ""))
        if is_valid_e164(num):
            return num

    digits = extract_digits_from_sip_from(tw_from)
    if is_valid_digits_number(digits, EXTERNAL_NUM_MIN_LEN, EXTERNAL_NUM_MAX_LEN):
        return digits

    return ""

def is_internal_sip_from(tw_from: str) -> bool:
    digits = extract_digits_from_sip_from(tw_from)
    return is_valid_digits_number(digits, INTERNAL_EXT_MIN_LEN, INTERNAL_EXT_MAX_LEN)

def choose_caller_for_transfer(caller_real: str, tw_from: str) -> str:
    if caller_real and is_valid_e164(caller_real):
        return caller_real

    if caller_real and is_valid_digits_number(caller_real, EXTERNAL_NUM_MIN_LEN, EXTERNAL_NUM_MAX_LEN):
        return caller_real

    if is_internal_sip_from(tw_from):
        return DEFAULT_INTERNAL_FALLBACK_CALLER

    return DEFAULT_EXTERNAL_FALLBACK_CALLER

# =========================
# Phone for CRM:
# - externo: +591... o largo -> local 8 dígitos
# - interno: extensión corta -> map o fallback -> NUNCA null si puedes
# =========================
def get_phone_for_crm(caller_real: str, tw_from: str) -> str:
    # 1) caller_real E.164
    if caller_real and is_valid_e164(caller_real):
        digits = normalize_digits(caller_real)
        return to_local_digits(digits)

    # 2) caller_real dígitos
    if caller_real:
        digits = normalize_digits(caller_real)
        if len(digits) >= EXTERNAL_NUM_MIN_LEN:
            return to_local_digits(digits)

        # corto => interno
        ext = digits
        mapped = (EXT_TO_MOBILE.get(ext) or "").strip()
        if mapped:
            return to_local_digits(normalize_digits(mapped))
        if INTERNAL_DEFAULT_MOBILE:
            return to_local_digits(normalize_digits(INTERNAL_DEFAULT_MOBILE))
        return ext

    # 3) From SIP => extensión
    ext = extract_digits_from_sip_from(tw_from)
    if ext:
        mapped = (EXT_TO_MOBILE.get(ext) or "").strip()
        if mapped:
            return to_local_digits(normalize_digits(mapped))
        if INTERNAL_DEFAULT_MOBILE:
            return to_local_digits(normalize_digits(INTERNAL_DEFAULT_MOBILE))
        return ext

    return ""

# =========================
# Punto: split nombre/apellido para last_name requerido
# =========================
def split_first_last(full_name: str):
    full_name = (full_name or "").strip()
    parts = [p for p in full_name.split() if p]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], "N/A"
    return parts[0], " ".join(parts[1:])

# =========================
# SPEECH HINTS / PARSER
# =========================
SPEECH_HINTS = [
    "pablo", "gonzalo", "vladimir", "paola", "ximena",
    "soporte", "ayuda", "técnico", "tecnico", "mesa", "operador",
    "agente", "humano",
    "datos", "registrar", "cotización", "cotizacion", "ventas",
    "con pablo", "con gonzalo", "con vladimir", "con paola", "con ximena",
    "comunicame con pablo", "comunícame con pablo",
    "quiero hablar con pablo", "hablar con pablo",
]

FILLER_WORDS = {
    "con", "por", "favor", "porfa", "porfavor",
    "quiero", "hablar", "hable", "comunicarme", "comunicame", "comunícame",
    "me", "puede", "podria", "podría", "deseo", "para", "el", "la", "al", "a",
    "por", "fa"
}

def clean_for_name(text: str) -> str:
    words = (text or "").split()
    words = [w for w in words if w and w not in FILLER_WORDS]
    return " ".join(words).strip()

def extract_name_candidate(text: str) -> str:
    t = (text or "").strip()
    patterns = [
        r"(?:quiero\s+hablar|hablar|hable|comunicarme|comunicame|comunícame)\s+con\s+(.+)$",
        r"(?:con)\s+(.+)$",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return clean_for_name(m.group(1))
    return clean_for_name(t)

NAME_ALIASES = {
    "pablo": ["pablo", "pavlo", "pabloo", "palo", "pabloh"],
    "gonzalo": ["gonzalo", "gonza", "gonsalo", "consalo", "gonzal", "gonzaloz"],
    "vladimir": ["vladimir", "bladimir", "pladimir", "vlad", "vladimír", "vladmir", "vladmirr"],
    "paola": ["paola", "paula", "pa ola", "pau la", "pao la", "paolla"],
    "ximena": ["ximena", "xime", "xime na", "xi mena", "xim ena", "ximen a"],
}

def detect_name_from_text(text: str):
    words = text.split()
    candidates = words + [text]
    best_name = None
    best_score = 0.0
    for name, aliases in NAME_ALIASES.items():
        for alias in aliases:
            for c in candidates:
                score = similarity(c, alias)
                if score > best_score:
                    best_score = score
                    best_name = name
    return best_name, best_score

def saludo_por_hora():
    tz = pytz.timezone("America/La_Paz")
    h = datetime.now(tz).hour
    if 5 <= h < 12:
        return "Buenos días."
    if 12 <= h < 19:
        return "Buenas tardes."
    return "Buenas noches."

# =========================
# RESPUESTAS DETERMINÍSTICAS (anti-alucinación)
# =========================
def format_fecha_la_paz():
    tz = pytz.timezone("America/La_Paz")
    now = datetime.now(tz)
    dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    dia_sem = dias[now.weekday()]
    return f"{dia_sem} {now.day} de {meses[now.month-1]} de {now.year}"

def format_hora_la_paz():
    tz = pytz.timezone("America/La_Paz")
    now = datetime.now(tz)
    return now.strftime("%H:%M")

def detect_intent_fecha_hora(text_norm: str):
    t = text_norm or ""
    if any(k in t for k in ["que hora", "qué hora", "hora es", "la hora", "me dices la hora"]):
        return "hora"
    if any(k in t for k in ["que fecha", "qué fecha", "fecha es", "la fecha", "en que fecha", "en qué fecha"]):
        return "fecha"
    if any(k in t for k in ["que dia", "qué día", "que día", "día es", "hoy es", "que dia es hoy", "qué día es hoy"]):
        return "dia"
    return ""

# =========================
# INTENTS: DATOS / LEAD
# =========================
def detect_intent_lead(text_norm: str) -> bool:
    t = (text_norm or "").strip()
    if t in ["1", "uno"]:
        return True
    keys = [
        "datos", "dejar mis datos", "registrar", "registro",
        "cotizacion", "cotización", "ventas", "comercial",
        "quiero que me contacten", "contactenme", "contáctenme",
        "que me llamen", "me pueden llamar", "llamenme", "llámenme",
    ]
    return any(k in t for k in keys)

# =========================
# URLs / Gather
# =========================
def _abs_url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    if not path.startswith(TWILIO_PREFIX):
        path = TWILIO_PREFIX + path
    if BASE_URL:
        return f"{BASE_URL}{path}"
    return path

def gather_prompt(action_url: str, prompt_text: str, num_digits: int = 4):
    g = Gather(
        input="dtmf speech",
        num_digits=num_digits,
        language="es-MX",
        timeout=GATHER_TIMEOUT,
        speech_timeout=SPEECH_TIMEOUT,
        action=_abs_url(action_url),
        method="POST",
        barge_in=True,
        action_on_empty_result=True,
        speech_model="phone_call",
        enhanced=True,
        hints=",".join(SPEECH_HINTS),
    )
    say(g, prompt_text)
    return g

# ✅ NUEVO: Gather SOLO VOZ (para nombre/empresa) -> evita que DTMF dispare el action y rompa el flujo de "datos"
def gather_speech_prompt(action_url: str, prompt_text: str):
    g = Gather(
        input="speech",
        language="es-MX",
        timeout=GATHER_TIMEOUT,
        speech_timeout=SPEECH_TIMEOUT,
        action=_abs_url(action_url),
        method="POST",
        barge_in=True,
        action_on_empty_result=True,
        speech_model="phone_call",
        enhanced=True,
        hints=",".join(SPEECH_HINTS),
    )
    say(g, prompt_text)
    return g

def gather_post_answer(action_url: str, prompt_text: str):
    g = Gather(
        input="dtmf speech",
        num_digits=1,
        language="es-MX",
        timeout=8,
        speech_timeout="auto",
        action=_abs_url(action_url),
        method="POST",
        barge_in=True,
        action_on_empty_result=True,
        speech_model="phone_call",
        enhanced=True,
        hints="agente,humano,operador,soporte,cero,0,datos,uno,1",
    )
    say(g, prompt_text)
    return g

# =========================
# Transfer
# =========================
def transfer_to_user(vr, target_user: str, caller_real: str, tw_from: str):
    sip_target = build_sip_uri(target_user)
    say(vr, "Perfecto, le comunico.")

    caller_for_dial = choose_caller_for_transfer(caller_real, tw_from)
    d = Dial(caller_id=caller_for_dial)
    d.sip(sip_target)
    vr.append(d)

    logging.warning(
        f"TRANSFER -> {sip_target} | callerId={caller_for_dial} | srcFrom={tw_from} | caller_real={caller_real or '(none)'}"
    )
    return Response(str(vr), mimetype="text/xml")

# =========================
# OpenAI
# =========================
def llamar_openai(call_sid: str, user_text: str) -> str:
    if not OPENAI_API_KEY:
        logging.error("[OPENAI] OPENAI_API_KEY VACIA")
        return "En este momento no tengo acceso al asistente inteligente. Si desea, lo comunico con soporte."

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversaciones[call_sid]
    messages.append({"role": "user", "content": user_text})

    data = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": messages,
        "max_tokens": 110,
        "temperature": 0.0,
    }

    try:
        r = session.post(OPENAI_URL, json=data, headers=headers, timeout=15)
        logging.warning(f"[OPENAI] status={r.status_code} body={r.text[:200]}")
        if r.status_code != 200:
            return "Tengo un inconveniente técnico. Si desea, lo comunico con soporte."

        respuesta = r.json()["choices"][0]["message"]["content"].strip()
        conversaciones[call_sid].append({"role": "user", "content": user_text})
        conversaciones[call_sid].append({"role": "assistant", "content": respuesta})
        return respuesta

    except Exception as e:
        logging.exception(f"[OPENAI] EXCEPTION: {e}")
        return "Estoy teniendo un inconveniente técnico. Si desea, lo comunico con soporte."

# =========================
# Zoho Flow (payload COMPATIBLE con tu Flow actual)
# =========================
def send_to_zoho_flow(payload: dict) -> dict:
    if not ZOHO_FLOW_WEBHOOK_URL:
        logging.error("[ZOHO] ZOHO_FLOW_WEBHOOK_URL not configured")
        return {"ok": False, "error": "ZOHO_FLOW_WEBHOOK_URL not configured"}

    try:
        r = session.post(ZOHO_FLOW_WEBHOOK_URL, json=payload, timeout=20)
        ok = 200 <= r.status_code < 300
        logging.warning(f"[ZOHO] status={r.status_code} body={(r.text or '')[:300]}")
        return {"ok": ok, "status_code": r.status_code, "response_text": (r.text or "")[:800]}
    except Exception as e:
        logging.exception(f"[ZOHO] EXCEPTION: {e}")
        return {"ok": False, "error": str(e)}

def create_zoho_lead_from_call(call_sid: str, tw_from: str, caller_real: str, full_name: str, company: str, notes: str = "") -> dict:
    """
    IMPORTANTE:
    - NO cambiamos tu Zoho Flow.
    - Enviamos keys igual que el flujo que te funcionaba con ElevenLabs:
      first_name, last_name, phone, email, wa_id, company_name, reason, notes, source, etc.
    """
    first, last = split_first_last(full_name)

    phone_for_crm = get_phone_for_crm(caller_real, tw_from)
    phone_for_crm = str(phone_for_crm) if phone_for_crm else None

    payload = {
        # keys que tu Zoho Flow está usando en el action
        "first_name": first or None,
        "last_name": (last or "N/A"),  # requerido
        "phone": phone_for_crm,
        "email": None,
        "wa_id": None,

        # estilo ElevenLabs / tracking
        "source": ZOHO_SOURCE,
        "ts": int(time.time()),
        "created_at": int(time.time()),
        "call_sid": call_sid,
        "company_name": (company or "").strip() or None,
        "reason": "twilio-ivr",
        "notes": (notes or "").strip(),
        "human_requested": None,
        "callback_requested": True,
        "last_intent": "ivr_lead_capture",
        "city": ZOHO_CITY_DEFAULT,
    }

    return send_to_zoho_flow(payload)

# =========================
# DEBUG ENDPOINTS
# =========================
@flask_app.route("/debug-openai", methods=["GET"])
def debug_openai():
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY NO CARGADA", 200
    return "OPENAI_API_KEY OK", 200

@flask_app.route("/test-openai", methods=["GET"])
def test_openai():
    if not OPENAI_API_KEY:
        return jsonify({"ok": False, "error": "OPENAI_API_KEY NO CARGADA"}), 200

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": [
            {"role": "system", "content": "Responde solo con 'OK'."},
            {"role": "user", "content": "test"}
        ],
        "max_tokens": 5,
        "temperature": 0
    }

    try:
        r = session.post(OPENAI_URL, json=data, headers=headers, timeout=15)
        return jsonify({"ok": r.status_code == 200, "status": r.status_code, "body": r.text[:500]}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200

# =========================
# MAIN IVR
# =========================
@flask_app.route("/ivr-llm", methods=["GET", "POST"])
def ivr_llm():
    vr = VoiceResponse()

    try:
        gc_memory()

        if request.method == "GET":
            say(vr, "IVR OK.")
            return Response(str(vr), mimetype="text/xml")

        speech = request.values.get("SpeechResult")
        digits = request.values.get("Digits")
        confidence_raw = request.values.get("Confidence")
        call_sid = request.values.get("CallSid", "unknown")
        noinput = int(request.args.get("noinput", "1"))
        mode = (request.args.get("mode", "") or "").strip().lower()

        touch_call(call_sid)

        tw_from = request.values.get("From") or ""
        tw_to = request.values.get("To") or ""
        direction = request.values.get("Direction") or ""
        call_status = request.values.get("CallStatus") or ""

        sip_headers = {
            "X-Original-Caller": request.values.get("SipHeader_X-Original-Caller") or "",
            "X-ANI": request.values.get("SipHeader_X-ANI") or "",
            "P-Asserted-Identity": request.values.get("SipHeader_P-Asserted-Identity") or "",
            "Remote-Party-ID": request.values.get("SipHeader_Remote-Party-ID") or "",
        }

        caller_real = get_original_caller(tw_from, sip_headers)

        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except:
            confidence = None

        logging.warning(
            f"[TWILIO] CallSid={call_sid} From={tw_from} To={tw_to} caller_real={caller_real or '(none)'} "
            f"Direction={direction} Status={call_status} noinput={noinput} mode={mode}"
        )
        logging.info(f"[STT] SpeechResult='{speech}' Confidence={confidence}")
        logging.warning(f"[DTMF] digits recibido: {digits}")

        # =========================
        # POST-ANSWER MODE: decidir agente / datos / colgar
        # =========================
        if mode == "post":
            text_post = normalize(speech)

            if digits == "0" or any(k in text_post for k in ["agente", "humano", "operador", "soporte", "ayuda", "mesa", "tecnico", "técnico"]):
                return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

            if digits == "1" or detect_intent_lead(text_post):
                lead_stage[call_sid] = "ask_name"
                say(vr, "Perfecto. Voy a registrar sus datos.")
                vr.pause(length=0.2)
                # ✅ SOLO VOZ para nombre (evita que DTMF rompa el flujo)
                vr.append(gather_speech_prompt("/ivr-llm?noinput=1", "¿Cuál es su nombre y apellido?"))
                return Response(str(vr), mimetype="text/xml")

            say(vr, "De acuerdo. Gracias por llamar a Nuxway. Hasta luego.")
            vr.hangup()
            return Response(str(vr), mimetype="text/xml")

        # =========================
        # SILENCIO / NOINPUT
        # =========================
        if not speech and not digits:
            if noinput <= MAX_NOINPUT_ATTEMPTS:
                vr.pause(length=RETRY_PAUSE_SEC if noinput > 1 else INITIAL_PAUSE_SEC)

                if noinput == 1:
                    prompt = (
                        f"{saludo_por_hora()} Gracias por llamar a Nuxway Technology. "
                        "Diga el nombre de la persona con la que desea comunicarse. "
                        "Para soporte, marque cero o diga soporte. "
                        "Para dejar sus datos, diga datos o marque uno."
                    )
                else:
                    prompt = (
                        "Disculpe, no le escuché bien. "
                        "Para transferirle, dígame solo el nombre: Pablo, Gonzalo, Vladimir, Paola o Ximena. "
                        "O marque cero para soporte. "
                        "Para dejar sus datos, diga datos o marque uno."
                    )

                vr.append(gather_prompt(f"/ivr-llm?noinput={noinput+1}", prompt))
                return Response(str(vr), mimetype="text/xml")

            say(vr, "Parece que la llamada está con poco audio. Le comunico con soporte.")
            return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

        # =========================
        # Normalizamos
        # =========================
        text = normalize(speech)
        name_candidate = extract_name_candidate(text)
        logging.info(f"[PARSE] raw='{text}' candidate='{name_candidate}'")

        # Repregunta si STT viene flojo
        if speech and digits is None and confidence is not None and confidence < MIN_CONFIDENCE_REPROMPT:
            vr.pause(length=0.3)
            vr.append(gather_prompt("/ivr-llm?noinput=1", "Disculpe, no le entendí bien. Repita por favor."))
            return Response(str(vr), mimetype="text/xml")

        logging.info(f"[CALL {call_sid}] speech='{text}' candidate='{name_candidate}' digits='{digits}' llm_turns={llm_turns[call_sid]} lead_stage={lead_stage[call_sid]}")

        # =========================
        # LEAD CAPTURE (ETAPAS)
        # =========================
        if digits == "1" or detect_intent_lead(text):
            lead_stage[call_sid] = "ask_name"
            say(vr, "Perfecto. Voy a registrar sus datos en el CRM.")
            vr.pause(length=0.2)
            # ✅ SOLO VOZ para nombre
            vr.append(gather_speech_prompt("/ivr-llm?noinput=1", "¿Cuál es su nombre y apellido?"))
            return Response(str(vr), mimetype="text/xml")

        if lead_stage[call_sid] == "ask_name":
            possible_name = (speech or "").strip()
            if possible_name and len(possible_name) >= 2:
                lead_name[call_sid] = possible_name[:80]
                lead_stage[call_sid] = "ask_company"
                vr.pause(length=0.2)
                # ✅ SOLO VOZ para empresa
                vr.append(gather_speech_prompt("/ivr-llm?noinput=1", "Gracias. ¿Cuál es el nombre de su empresa?"))
                return Response(str(vr), mimetype="text/xml")

            vr.pause(length=0.2)
            # ✅ SOLO VOZ reintento nombre
            vr.append(gather_speech_prompt("/ivr-llm?noinput=1", "¿Me repite su nombre y apellido, por favor?"))
            return Response(str(vr), mimetype="text/xml")

        if lead_stage[call_sid] == "ask_company":
            possible_company = (speech or "").strip()
            if possible_company and len(possible_company) >= 2:
                lead_company[call_sid] = possible_company[:120]

                res = create_zoho_lead_from_call(
                    call_sid=call_sid,
                    tw_from=tw_from,
                    caller_real=caller_real,
                    full_name=lead_name[call_sid],
                    company=lead_company[call_sid],
                    notes=f"From={tw_from} To={tw_to} Direction={direction} Status={call_status}"
                )

                # Reset lead state
                lead_stage[call_sid] = ""
                lead_name[call_sid] = ""
                lead_company[call_sid] = ""

                if res.get("ok"):
                    say(vr, "Listo. Sus datos fueron enviados al CRM. Gracias, hasta luego.")
                    vr.hangup()
                    return Response(str(vr), mimetype="text/xml")

                say(vr, "Tuve un problema registrando sus datos. Le comunico con soporte.")
                return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

            vr.pause(length=0.2)
            # ✅ SOLO VOZ reintento empresa
            vr.append(gather_speech_prompt("/ivr-llm?noinput=1", "¿Cuál es el nombre de su empresa?"))
            return Response(str(vr), mimetype="text/xml")

        # =========================
        # RESPUESTAS DETERMINÍSTICAS (anti-alucinación)
        # =========================
        intent = detect_intent_fecha_hora(text)
        if intent:
            if intent == "hora":
                say(vr, f"Son las {format_hora_la_paz()} en Bolivia.")
            elif intent in ("fecha", "dia"):
                say(vr, f"Hoy es {format_fecha_la_paz()}.")

            vr.pause(length=0.2)
            vr.append(gather_post_answer(
                "/ivr-llm?mode=post",
                "Si desea un agente, marque cero o diga agente. Si desea dejar sus datos, diga datos o marque uno. O puede colgar."
            ))
            return Response(str(vr), mimetype="text/xml")

        # =========================
        # DTMF routing
        # =========================
        if digits == "4000":
            return transfer_to_user(vr, DID_MAP["pablo"], caller_real, tw_from)
        if digits == "4001":
            return transfer_to_user(vr, DID_MAP["gonzalo"], caller_real, tw_from)
        if digits == "4002":
            return transfer_to_user(vr, DID_MAP["vladimir"], caller_real, tw_from)
        if digits == "4003":
            return transfer_to_user(vr, DID_MAP["paola"], caller_real, tw_from)
        if digits == "4007":
            return transfer_to_user(vr, DID_MAP["ximena"], caller_real, tw_from)
        if digits == "0":
            return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

        # =========================
        # Voice: soporte/agente
        # =========================
        if any(k in text for k in ["soporte", "support", "ayuda", "mesa", "tecnico", "técnico", "cola", "agente", "humano", "operador"]):
            return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

        # =========================
        # Voice directo por nombre
        # =========================
        if speech and digits is None and name_candidate == "":
            vr.pause(length=0.2)
            vr.append(gather_prompt(
                "/ivr-llm?noinput=1",
                "Para transferirle, dígame solo el nombre: Pablo, Gonzalo, Vladimir, Paola o Ximena. "
                "O marque cero para soporte. "
                "Para dejar sus datos, diga datos o marque uno."
            ))
            return Response(str(vr), mimetype="text/xml")

        for name in ["pablo", "gonzalo", "vladimir", "paola", "ximena"]:
            if name in name_candidate:
                return transfer_to_user(vr, DID_MAP[name], caller_real, tw_from)

        # =========================
        # Fuzzy match
        # =========================
        bm, score = detect_name_from_text(name_candidate)
        if bm and score >= 0.72:
            logging.warning(f"[CALL {call_sid}] FUZZY_NAME -> '{name_candidate}' => '{bm}' score={score:.2f}")
            return transfer_to_user(vr, DID_MAP[bm], caller_real, tw_from)

        # =========================
        # OpenAI fallback (máximo 2 turnos)
        # =========================
        if llm_turns[call_sid] >= MAX_LLM_TURNS:
            say(vr, "Para continuar, le comunico con soporte.")
            return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

        llm_turns[call_sid] += 1
        respuesta = llamar_openai(call_sid, text)
        say(vr, respuesta)

        # Si ya llegó al límite, ofrecer post (agente/datos/colgar)
        if llm_turns[call_sid] >= MAX_LLM_TURNS:
            vr.pause(length=0.2)
            vr.append(gather_post_answer(
                "/ivr-llm?mode=post",
                "Si desea un agente, marque cero o diga agente. Si desea dejar sus datos, diga datos o marque uno. O puede colgar."
            ))
            return Response(str(vr), mimetype="text/xml")

        # Aún queda 1 pregunta más
        vr.pause(length=0.3)
        vr.append(gather_prompt(
            "/ivr-llm?noinput=1",
            "¿Qué otra consulta tiene? Si desea soporte, diga soporte o marque cero. Si desea dejar sus datos, diga datos o marque uno."
        ))
        return Response(str(vr), mimetype="text/xml")

    except Exception as e:
        logging.exception(f"ERROR en /ivr-llm: {e}")
        say(vr, "Hubo un problema técnico. Le comunico con soporte.")
        return transfer_to_user(vr, DID_MAP["cola"], "", "")

@flask_app.route("/", methods=["GET"])
def home():
    return "OK", 200



