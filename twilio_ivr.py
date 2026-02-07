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
    return f"sip:{user}@{S

