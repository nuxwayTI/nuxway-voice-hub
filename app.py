import os

from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

from eleven_handoff import router as eleven_router
from twilio_ivr import flask_app  # Flask IVR

# OpenAI SDK (asegúrate de tenerlo en requirements.txt: openai>=1.0.0)
from openai import OpenAI

app = FastAPI(title="nuxway-voice-hub")

# ---- OpenAI client ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Log seguro al arrancar (Render logs)
print(
    "[boot] OPENAI_API_KEY present:",
    bool(OPENAI_API_KEY),
    "prefix:",
    (OPENAI_API_KEY[:6] + "..." if OPENAI_API_KEY else "None"),
)
print("[boot] OPENAI_MODEL:", OPENAI_MODEL)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---- Routers ----
# Rutas Eleven / Yeastar / Zoho
app.include_router(eleven_router)

# Montar IVR de Twilio (Flask) bajo /twilio
app.mount("/twilio", WSGIMiddleware(flask_app))


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "nuxway-voice-hub",
        "routes": {
            "twilio_ivr": "/twilio/ivr-llm",
            "handoff_call": "/tools/handoff_to_human",
            "zoho_lead": "/tools/zoho_lead",
            "health": "/health",
            "health_openai": "/health/openai",
        },
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "nuxway-voice-hub",
        "openai_key_present": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
    }


@app.get("/health/openai")
def health_openai():
    """
    Verifica integración real con OpenAI:
    - Confirma que existe OPENAI_API_KEY
    - Hace una llamada mínima al modelo
    """
    if not OPENAI_API_KEY or client is None:
        return {
            "ok": False,
            "error": "OPENAI_API_KEY is missing in environment variables",
        }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Responde solo con: ok"}],
            max_tokens=5,
        )
        answer = (resp.choices[0].message.content or "").strip()

        return {
            "ok": True,
            "model": OPENAI_MODEL,
            "answer": answer,
        }

    except Exception as e:
        # Render logs
        print("[health_openai] ERROR:", repr(e))
        return {"ok": False, "error": str(e)}

