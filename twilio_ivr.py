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
MAX_LLM_TURNS = 3
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

def touch_call(call_sid: str):
    last_seen[call_sid] = time.time()

def gc_memory():
    now = time.time()
    dead = [sid for sid, ts in last_seen.items() if now - ts > MEM_TTL_SECONDS]
    for sid in dead:
        conversaciones.pop(sid, None)
        llm_turns.pop(sid, None)
        last_seen.pop(sid, None)
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
    text = re.sub(r"[^a-záéíóúñ0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

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
# SPEECH HINTS / PARSER
# =========================
SPEECH_HINTS = [
    "pablo", "gonzalo", "vladimir", "paola", "ximena",
    "soporte", "ayuda", "técnico", "tecnico", "mesa", "operador",
    "con pablo", "con gonzalo", "con vladimir", "con paola", "con ximena",
    "comunicame con pablo", "comunícame con pablo",
    "comunicame con vladimir", "comunícame con vladimir",
    "quiero hablar con pablo", "quiero hablar con vladimir",
    "hablar con pablo", "hablar con vladimir",
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

def _abs_url(path: str) -> str:
    # Montado en /twilio. Aseguramos que Twilio vuelva a /twilio/...
    if not path.startswith("/"):
        path = "/" + path

    if not path.startswith(TWILIO_PREFIX):
        path = TWILIO_PREFIX + path  # /twilio/ivr-llm...

    if BASE_URL:
        return f"{BASE_URL}{path}"
    return path

def gather_prompt(action_url: str, prompt_text: str):
    g = Gather(
        input="dtmf speech",
        num_digits=4,
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

def llamar_openai(call_sid: str, user_text: str) -> str:
    if not OPENAI_API_KEY:
        logging.error("[OPENAI] OPENAI_API_KEY VACIA")
        return "En este momento no tengo acceso al asistente inteligente. Lo comunico con soporte."

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversaciones[call_sid]
    messages.append({"role": "user", "content": user_text})

    data = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": messages,
        "max_tokens": 140,
        "temperature": 0.25,
    }

    try:
        r = session.post(OPENAI_URL, json=data, headers=headers, timeout=15)
        logging.warning(f"[OPENAI] status={r.status_code} body={r.text[:200]}")
        if r.status_code != 200:
            return "Tengo un inconveniente técnico. Lo comunico con soporte."

        respuesta = r.json()["choices"][0]["message"]["content"].strip()
        conversaciones[call_sid].append({"role": "user", "content": user_text})
        conversaciones[call_sid].append({"role": "assistant", "content": respuesta})
        return respuesta

    except Exception as e:
        logging.exception(f"[OPENAI] EXCEPTION: {e}")
        return "Estoy teniendo un inconveniente técnico. Lo comunico con soporte."

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
            f"Direction={direction} Status={call_status} noinput={noinput}"
        )
        logging.info(f"[STT] SpeechResult='{speech}' Confidence={confidence}")
        logging.warning(f"[DTMF] digits recibido: {digits}")

        # SILENCIO
        if not speech and not digits:
            if noinput <= MAX_NOINPUT_ATTEMPTS:
                vr.pause(length=RETRY_PAUSE_SEC if noinput > 1 else INITIAL_PAUSE_SEC)

                if noinput == 1:
                    prompt = (
                        f"{saludo_por_hora()} Gracias por llamar a Nuxway Technology. "
                        "Diga el nombre de la persona con la que desea comunicarse. "
                        "Para soporte, marque cero o diga soporte."
                    )
                else:
                    prompt = (
                        "Disculpe, no le escuché bien. "
                        "Para transferirle, dígame solo el nombre: Pablo, Gonzalo, Vladimir, Paola o Ximena. "
                        "O marque cero para soporte."
                    )

                vr.append(gather_prompt(f"/ivr-llm?noinput={noinput+1}", prompt))
                return Response(str(vr), mimetype="text/xml")

            say(vr, "Parece que la llamada está con poco audio. Le comunico con soporte.")
            return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

        # Normalizamos speech + candidato de nombre
        text = normalize(speech)
        name_candidate = extract_name_candidate(text)
        logging.info(f"[PARSE] raw='{text}' candidate='{name_candidate}'")

        if speech and digits is None and confidence is not None and confidence < MIN_CONFIDENCE_REPROMPT:
            vr.pause(length=0.4)
            vr.append(gather_prompt("/ivr-llm?noinput=1", "Disculpe, no le entendí bien. Dígame el nombre otra vez, por favor."))
            return Response(str(vr), mimetype="text/xml")

        if speech and digits is None and name_candidate == "":
            vr.pause(length=0.3)
            vr.append(gather_prompt(
                "/ivr-llm?noinput=1",
                "Para transferirle, dígame solo el nombre: Pablo, Gonzalo, Vladimir, Paola o Ximena."
            ))
            return Response(str(vr), mimetype="text/xml")

        logging.info(f"[CALL {call_sid}] speech='{text}' candidate='{name_candidate}' digits='{digits}' llm_turns={llm_turns[call_sid]}")

        # DTMF routing
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

        # Voice: soporte
        if any(k in text for k in ["soporte", "support", "ayuda", "mesa", "tecnico", "técnico", "cola"]):
            return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

        # Voice directo por nombre
        for name in ["pablo", "gonzalo", "vladimir", "paola", "ximena"]:
            if name in name_candidate:
                return transfer_to_user(vr, DID_MAP[name], caller_real, tw_from)

        # Fuzzy match
        bm, score = detect_name_from_text(name_candidate)
        if bm and score >= 0.72:
            logging.warning(f"[CALL {call_sid}] FUZZY_NAME -> '{name_candidate}' => '{bm}' score={score:.2f}")
            return transfer_to_user(vr, DID_MAP[bm], caller_real, tw_from)

        # humano/operador -> soporte
        if any(k in text for k in ["ingeniero", "agente", "humano", "operador"]):
            return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

        # OpenAI fallback
        if llm_turns[call_sid] >= MAX_LLM_TURNS:
            say(vr, "Para continuar, le comunico con soporte.")
            return transfer_to_user(vr, DID_MAP["cola"], caller_real, tw_from)

        llm_turns[call_sid] += 1
        respuesta = llamar_openai(call_sid, text)
        say(vr, respuesta)

        vr.pause(length=0.4)
        vr.append(gather_prompt("/ivr-llm?noinput=1", "Diga el nombre de la persona. Para soporte, marque cero o diga soporte."))
        return Response(str(vr), mimetype="text/xml")

    except Exception as e:
        logging.exception(f"ERROR en /ivr-llm: {e}")
        say(vr, "Hubo un problema técnico. Le comunico con soporte.")
        return transfer_to_user(vr, DID_MAP["cola"], "", "")

@flask_app.route("/", methods=["GET"])
def home():
    return "OK"
