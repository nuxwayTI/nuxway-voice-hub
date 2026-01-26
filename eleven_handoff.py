import os
import time
import re
import logging
from typing import Optional, Any, Dict

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

# -----------------------------
# Logging (Render Logs)
# -----------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("eleven-handoff")

router = APIRouter()

# -----------------------------
# ENV VARS
# -----------------------------
ELEVEN_SHARED_SECRET = os.getenv("ELEVEN_SHARED_SECRET")  # optional
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

PBX_DOMAIN = os.getenv("PBX_DOMAIN")  # e.g. nuxwaytechnology.use.ycmcloud.com
YEASTAR_API_PATH = os.getenv("YEASTAR_API_PATH", "openapi/v1.0")
YEASTAR_USER_AGENT = os.getenv("YEASTAR_USER_AGENT", "OpenAPI")

YEASTAR_USERNAME = os.getenv("YEASTAR_USERNAME")
YEASTAR_PASSWORD = os.getenv("YEASTAR_PASSWORD")

# Control variables
CALLER = os.getenv("CALLER", "6200")                 # IVR 6200 or queue/ivr entry
CALLEE_PREFIX = os.getenv("CALLEE_PREFIX", "98")     # outbound prefix (your PBX)
DIAL_PERMISSION = os.getenv("DIAL_PERMISSION", "4002")  # permission extension
DEFAULT_AUTO_ANSWER = os.getenv("DEFAULT_AUTO_ANSWER", "no")  # keep 'no' for IVR/Queue flow

# If STRIP_COUNTRY_CODE=1, keep ONLY the last LOCAL_NUMBER_LEN digits.
# Example: 59161786583 -> last 8 digits -> 61786583
STRIP_COUNTRY_CODE = os.getenv("STRIP_COUNTRY_CODE", "1") == "1"
LOCAL_NUMBER_LEN = int(os.getenv("LOCAL_NUMBER_LEN", "8"))  # Bolivia local is usually 8

# -----------------------------
# NEW: Zoho Flow Webhook (Bigin)
# -----------------------------
ZOHO_FLOW_WEBHOOK_URL = os.getenv("ZOHO_FLOW_WEBHOOK_URL", "")
ZOHO_SOURCE = os.getenv("ZOHO_SOURCE", "elevenlabs-handoff")  # optional label


def _require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


# -----------------------------
# Helpers: phone normalization
# -----------------------------
def normalize_digits(value: str) -> str:
    """
    Accept:
      - +59161786583
      - 59161786583
      - 61786583
      - 61786583@c.us
      - 59161786583@s.whatsapp.net
    Returns only digits.
    """
    value = (value or "").strip()
    value = value.replace("@c.us", "").replace("@s.whatsapp.net", "")
    return re.sub(r"\D", "", value)


def to_local_digits(digits: str) -> str:
    """
    Global approach:
    If STRIP_COUNTRY_CODE is enabled and digits is longer than LOCAL_NUMBER_LEN,
    keep the last LOCAL_NUMBER_LEN digits.
    """
    if not digits:
        return ""
    if STRIP_COUNTRY_CODE and LOCAL_NUMBER_LEN > 0 and len(digits) > LOCAL_NUMBER_LEN:
        return digits[-LOCAL_NUMBER_LEN:]
    return digits


def build_callee(number: str) -> str:
    """
    Build final PBX dial string:
      - normalize to digits
      - convert to local digits (strip country code by taking last N)
      - add CALLEE_PREFIX (98) unless already present
    """
    digits = normalize_digits(number)
    prefix = normalize_digits(CALLEE_PREFIX)

    if not digits:
        return ""

    # If incoming number already contains PBX prefix, remove it temporarily.
    if prefix and digits.startswith(prefix) and len(digits) > len(prefix):
        digits_wo_prefix = digits[len(prefix):]
    else:
        digits_wo_prefix = digits

    local_digits = to_local_digits(digits_wo_prefix)
    if not local_digits:
        return ""

    if prefix:
        return f"{prefix}{local_digits}"
    return local_digits


# -----------------------------
# Yeastar Client
# -----------------------------
class YeastarClient:
    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @property
    def base_url(self) -> str:
        domain = _require_env("PBX_DOMAIN", PBX_DOMAIN)
        return f"https://{domain}"

    @property
    def api_base(self) -> str:
        return f"{self.base_url}/{YEASTAR_API_PATH}"

    async def get_token(self) -> str:
        url = f"{self.api_base}/get_token"
        headers = {"Content-Type": "application/json", "User-Agent": YEASTAR_USER_AGENT}
        payload = {
            "username": _require_env("YEASTAR_USERNAME", YEASTAR_USERNAME),
            "password": _require_env("YEASTAR_PASSWORD", YEASTAR_PASSWORD),
        }

        logger.info("Requesting Yeastar access token...")
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        if data.get("errcode") != 0:
            raise RuntimeError(f"Yeastar get_token failed: {data}")

        token = data["access_token"]
        expires_in = int(data.get("access_token_expire_time", 1800))

        self._token = token
        self._token_expiry = time.time() + max(0, expires_in - 30)

        logger.info(f"Yeastar token OK. Expires in ~{expires_in}s")
        return token

    async def access_token(self) -> str:
        if self._token and time.time() < self._token_expiry:
            return self._token
        return await self.get_token()

    async def dial(
        self,
        caller: str,
        callee: str,
        dial_permission: Optional[str],
        auto_answer: str,
    ) -> Dict[str, Any]:
        token = await self.access_token()
        url = f"{self.api_base}/call/dial"
        params = {"access_token": token}
        headers = {"Content-Type": "application/json", "User-Agent": YEASTAR_USER_AGENT}

        payload: Dict[str, Any] = {"caller": caller, "callee": callee}
        if dial_permission:
            payload["dial_permission"] = dial_permission
        if auto_answer:
            payload["auto_answer"] = auto_answer

        logger.info(f"Calling Yeastar /call/dial payload={payload}")
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, headers=headers, params=params, json=payload)
            r.raise_for_status()
            return r.json()


yeastar = YeastarClient()


# -----------------------------
# Payloads
# -----------------------------
class HandoffPayload(BaseModel):
    whatsapp_id: Optional[str] = Field(None, description="WhatsApp user id / phone number")
    confirmed: Optional[bool] = Field(None, description="True after user confirms")
    reason: Optional[str] = Field(None, description="Optional context.")
    # Optional overrides per request
    caller: Optional[str] = None
    dial_permission: Optional[str] = None
    auto_answer: Optional[str] = None


class ZohoLeadPayload(BaseModel):
    """
    Endpoint SOLO para registrar lead en Zoho Flow / Bigin.
    NO dispara llamada.
    """
    whatsapp_id: Optional[str] = Field(None, description="WhatsApp user id / phone number")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    name: Optional[str] = None          # full name (si no hay first/last)
    city: Optional[str] = None
    phone: Optional[str] = None         # puede venir con +código país o local
    email: Optional[str] = None
    company_name: Optional[str] = None
    reason: Optional[str] = None

    # banderas
    human_requested: Optional[bool] = None
    callback_requested: Optional[bool] = None

    # extra
    notes: Optional[str] = None
    last_intent: Optional[str] = None
    source: Optional[str] = None        # si quieres sobreescribir ZOHO_SOURCE


# -----------------------------
# Health
# -----------------------------
@router.get("/health")
async def health():
    return {
        "ok": True,
        "service": "nuxway-voice-hub",
        "dry_run": DRY_RUN,
        "caller": CALLER,
        "callee_prefix": CALLEE_PREFIX,
        "dial_permission": DIAL_PERMISSION,
        "strip_country_code": STRIP_COUNTRY_CODE,
        "local_number_len": LOCAL_NUMBER_LEN,
        "zoho_configured": bool(ZOHO_FLOW_WEBHOOK_URL),
        "zoho_source": ZOHO_SOURCE,
    }


@router.get("/yeastar/ping")
async def yeastar_ping(x_admin_key: Optional[str] = Header(default=None)):
    if ELEVEN_SHARED_SECRET and x_admin_key != ELEVEN_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    token = await yeastar.access_token()
    return {"ok": True, "message": "Yeastar token obtained", "token_prefix": token[:6]}


# -----------------------------
# Zoho Lead (NO CALL)
# -----------------------------
async def send_to_zoho_flow(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not ZOHO_FLOW_WEBHOOK_URL:
        return {"ok": False, "error": "ZOHO_FLOW_WEBHOOK_URL not configured"}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(ZOHO_FLOW_WEBHOOK_URL, json=payload)
        ok = 200 <= r.status_code < 300
        return {"ok": ok, "status_code": r.status_code, "response_text": r.text[:800]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/tools/zoho_lead")
async def zoho_lead(payload: ZohoLeadPayload, x_eleven_secret: Optional[str] = Header(default=None)):
    logger.info(">>> /tools/zoho_lead HIT")
    logger.info(f"Payload received: {payload.model_dump()}")

    if ELEVEN_SHARED_SECRET and x_eleven_secret != ELEVEN_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret")

    wa_digits = normalize_digits(payload.whatsapp_id or "")
    phone_digits = normalize_digits(payload.phone or "")

    phone_local = to_local_digits(phone_digits) if phone_digits else None
    wa_local = to_local_digits(wa_digits) if wa_digits else None

    zoho_payload = {
        "source": payload.source or ZOHO_SOURCE,
        "ts": int(time.time()),
        "created_at": int(time.time()),

        "wa_id": payload.whatsapp_id,
        "wa_id_digits": wa_digits or None,
        "wa_local": wa_local or None,

        "first_name": payload.first_name,
        "last_name": payload.last_name,
        "name": payload.name,

        "city": payload.city,
        "company_name": payload.company_name,

        "phone": payload.phone,
        "phone_digits": phone_digits or None,
        "phone_local": phone_local or None,

        "email": payload.email,

        "human_requested": bool(payload.human_requested) if payload.human_requested is not None else None,
        "callback_requested": bool(payload.callback_requested) if payload.callback_requested is not None else None,
        "last_intent": payload.last_intent,
        "reason": payload.reason,
        "notes": payload.notes,
    }

    res = await send_to_zoho_flow(zoho_payload)
    logger.info(f"Zoho result: {res}")

    if not res.get("ok"):
        return {"status": "failed", "detail": res}

    return {"status": "ok", "detail": res}


# -----------------------------
# Handoff -> Yeastar Call
# -----------------------------
@router.post("/tools/handoff_to_human")
async def handoff_to_human(payload: HandoffPayload, x_eleven_secret: Optional[str] = Header(default=None)):
    logger.info(">>> /tools/handoff_to_human HIT")
    logger.info(f"DRY_RUN={DRY_RUN}")
    logger.info(f"Payload received: {payload.model_dump()}")

    if ELEVEN_SHARED_SECRET and x_eleven_secret != ELEVEN_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret")

    # Tolerant: avoid 422 if ElevenLabs sends partial requests
    if payload.confirmed is not True:
        return {"status": "ignored", "reason": "confirmed is not true"}

    if not payload.whatsapp_id:
        return {"status": "ignored", "reason": "missing whatsapp_id"}

    caller = (payload.caller or CALLER).strip()
    dial_permission = (payload.dial_permission or DIAL_PERMISSION).strip() if (payload.dial_permission or DIAL_PERMISSION) else None
    auto_answer = (payload.auto_answer or DEFAULT_AUTO_ANSWER).strip()

    callee = build_callee(payload.whatsapp_id)

    if not caller:
        return {"status": "ignored", "reason": "CALLER not set"}

    if not callee:
        return {"status": "ignored", "reason": "invalid whatsapp_id after normalization"}

    logger.info(
        f"Prepared call -> caller={caller}, callee={callee}, dial_permission={dial_permission}, auto_answer={auto_answer} "
        f"(strip_country_code={STRIP_COUNTRY_CODE}, local_len={LOCAL_NUMBER_LEN})"
    )

    if DRY_RUN:
        return {
            "status": "dry_run_ok",
            "would_call": {
                "caller": caller,
                "callee": callee,
                "dial_permission": dial_permission,
                "auto_answer": auto_answer,
                "reason": payload.reason,
            },
        }

    res = await yeastar.dial(
        caller=caller,
        callee=callee,
        dial_permission=dial_permission,
        auto_answer=auto_answer,
    )

    if res.get("errcode") != 0:
        raise HTTPException(status_code=502, detail={"yeastar": res})

    return {"status": "ok", "call_id": res.get("call_id"), "yeastar": res}
