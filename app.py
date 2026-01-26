from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

from eleven_handoff import router as eleven_router
from twilio_ivr import flask_app  # Flask IVR

app = FastAPI(title="nuxway-voice-hub")

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
        },
    }
