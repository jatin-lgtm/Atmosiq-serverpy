# server/app.py
import os
import time
import logging
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
from typing import Optional, Dict, Any, Tuple
from flask_cors import CORS

# Load .env
load_dotenv()

# Config / env
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENAQ_BASE = os.getenv("OPENAQ_BASE", "https://api.openaq.org/v2")
USER_TZ = os.getenv("USER_TZ", "Asia/Kolkata")
PORT = int(os.getenv("PORT", "5174"))

# Cache TTL (seconds) for OpenAQ results
OPENAQ_CACHE_TTL = int(os.getenv("OPENAQ_CACHE_TTL", str(60 * 5)))  # default 5 minutes

# Setup logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("atmosiq")

# Try to import new google generative SDK
USE_GOOGLE_CLIENT = False
try:
    from google.generativeai import GenerativeModel  # type: ignore
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=GEMINI_KEY)
    USE_GOOGLE_CLIENT = True
    logger.info("Using google.generativeai client for Gemini.")
except Exception:
    USE_GOOGLE_CLIENT = False
    logger.info("google.generativeai client not available; will try REST fallback if GEMINI_KEY present.")

# Single Flask app instance
app = Flask(__name__)
CORS(app)

# Simple in-memory cache for OpenAQ: {cache_key: (expiry_ts, data)}
_openaq_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _openaq_cache.get(key)
    if not entry:
        return None
    expiry, data = entry
    if time.time() > expiry:
        del _openaq_cache[key]
        return None
    return data

def _cache_set(key: str, data: Optional[Dict[str, Any]], ttl: int = OPENAQ_CACHE_TTL) -> None:
    _openaq_cache[key] = (time.time() + ttl, data)

def _to_local_iso(iso_str: Optional[str], tz_name: str = USER_TZ) -> Optional[str]:
    if not iso_str:
        return None
    try:
        dt = None
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        except Exception:
            try:
                dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            except Exception:
                dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ")
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        local_tz = pytz.timezone(tz_name)
        local_dt = dt.astimezone(local_tz)
        return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return iso_str

def call_gemini_chat(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.2, max_tokens: int = 512) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in environment.")

    if USE_GOOGLE_CLIENT:
        try:
            gemini_model = GenerativeModel(model)
            response = gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return getattr(response, "text", str(response))
        except Exception:
            logger.exception("Gemini SDK call failed")
            raise

    try:
        headers = {"Authorization": f"Bearer {GEMINI_KEY}", "Content-Type": "application/json"}
        url = os.getenv("GEMINI_REST_URL", "https://api.generative.googleapis.com/v1beta2/models/%s:predict" % model)
        data = {"prompt": prompt, "maxOutputTokens": max_tokens, "temperature": temperature}
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict) and "candidates" in j:
            return j.get("candidates", [{}])[0].get("content", "")
        return str(j)
    except Exception:
        logger.exception("Gemini REST call failed")
        raise

def fetch_openaq_by_coords(lat: float, lon: float, radius_m: int = 5000) -> Optional[Dict[str, Any]]:
    cache_key = f"coords:{lat:.5f},{lon:.5f}:{radius_m}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        url = f"{OPENAQ_BASE}/locations"
        params = {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": 100}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        j = r.json()
        best = None
        best_dt = None
        for loc in j.get("results", []) or []:
            for param in loc.get("parameters", []) or []:
                pname = (param.get("parameter") or "").lower()
                if pname in ("pm25", "pm2.5"):
                    measured_at = param.get("lastUpdated") or param.get("lastUpdatedAt") or (param.get("lastValue") and param.get("lastUpdated"))
                    loc_dt = _to_local_iso(measured_at)
                    parsed = None
                    try:
                        if measured_at:
                            parsed = datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
                    except Exception:
                        parsed = None
                    val = param.get("lastValue") if param.get("lastValue") is not None else param.get("value") or param.get("average")
                    if val is None:
                        continue
                    if not best or (parsed and (best_dt is None or parsed > best_dt)):
                        best = {"pm25": val, "unit": param.get("unit"), "measured_at": loc_dt, "source": loc.get("name") or loc.get("location")}
                        best_dt = parsed
        _cache_set(cache_key, best)
        return best
    except Exception:
        logger.exception("OpenAQ coords lookup failed")
        _cache_set(cache_key, None)
        return None

def fetch_openaq_by_city(city: str) -> Optional[Dict[str, Any]]:
    cache_key = f"city:{city.lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        url_latest = f"{OPENAQ_BASE}/latest"
        params = {"city": city}
        r = requests.get(url_latest, params=params, timeout=8)
        r.raise_for_status()
        j = r.json()
        best = None
        best_dt = None

        def try_update(meas_val, meas_unit, meas_time, source_name):
            nonlocal best, best_dt
            if meas_val is None:
                return
            parsed = None
            try:
                if meas_time:
                    parsed = datetime.fromisoformat(meas_time.replace("Z", "+00:00"))
            except Exception:
                parsed = None
            if not best or (parsed and (best_dt is None or parsed > best_dt)):
                best = {"pm25": meas_val, "unit": meas_unit, "measured_at": _to_local_iso(meas_time), "source": source_name}
                best_dt = parsed

        for res in j.get("results", []) or []:
            for m in (res.get("measurements") or []):
                param = (m.get("parameter") or "").lower()
                if param in ("pm25", "pm2.5"):
                    try_update(m.get("value"), m.get("unit"), m.get("lastUpdated") or m.get("lastUpdatedAt") or (m.get("date") or {}).get("utc"), res.get("location") or res.get("name"))

        if not best:
            url_loc = f"{OPENAQ_BASE}/locations"
            params2 = {"city": city, "limit": 100}
            r2 = requests.get(url_loc, params=params2, timeout=8)
            r2.raise_for_status()
            j2 = r2.json()
            for loc in j2.get("results", []) or []:
                for p in (loc.get("parameters") or []):
                    param_name = (p.get("parameter") or "").lower()
                    if param_name in ("pm25", "pm2.5"):
                        val = p.get("lastValue") if p.get("lastValue") is not None else p.get("average") or p.get("value")
                        measured = p.get("lastUpdated") or p.get("lastUpdatedAt") or None
                        try_update(val, p.get("unit"), measured, loc.get("name") or loc.get("location"))
                        break

        _cache_set(cache_key, best)
        return best
    except Exception:
        logger.exception("OpenAQ fetch failed for city=%s", city)
        _cache_set(cache_key, None)
        return None

def get_air_quality(location: Optional[str], lat: Optional[float], lon: Optional[float]) -> Optional[Dict[str, Any]]:
    if lat is not None and lon is not None:
        return fetch_openaq_by_coords(lat, lon)

    if not location:
        return None

    candidates = [location, "New " + location if not location.lower().startswith("new") else location, "Delhi NCR" if "delhi" in location.lower() else None]
    seen = set()
    for cand in [c for c in candidates if c]:
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        res = fetch_openaq_by_city(cand)
        if res:
            return res
    return None

def simple_local_recommendation(message: str, pm: Optional[Dict[str, Any]], time_text: Optional[str]) -> str:
    if pm and pm.get("pm25") is not None:
        try:
            pm25 = float(pm["pm25"])
        except Exception:
            pm25 = None
        if pm25 is not None:
            if pm25 <= 50:
                return "Air quality is good. It's safe to walk outdoors; follow normal precautions."
            if pm25 <= 100:
                return "Air quality is moderate. Sensitive people should take care; a light walk is acceptable for most."
            if pm25 <= 200:
                return "Air quality is unhealthy for sensitive groups — consider reducing strenuous outdoor activity or walk indoors."
            return "Air quality is poor. Avoid outdoor exercise and use a high-quality mask (N95) if you must go out."
    if time_text:
        return f"I don't have recent air-quality data for that location. At {time_text}, if you feel breathing discomfort, avoid long or intense walks."
    return "I couldn't fetch local air-quality data. If you feel unwell or see heavy pollution, avoid outdoor exercise and prefer indoor activity."

@app.route("/api/assistant", methods=["POST"])
def assistant():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    location = (body.get("location") or "").strip() or None
    lat = lon = None
    try:
        if body.get("lat") is not None:
            lat = float(body.get("lat"))
        if body.get("lon") is not None:
            lon = float(body.get("lon"))
    except Exception:
        lat = lon = None

    time_text = body.get("time")
    pm = get_air_quality(location, lat, lon)

    prompt_lines = [
        "You are AtmosIQ Safety Assistant. Provide a short safety recommendation (2-3 sentences).",
        "Be actionable and friendly. If air quality (PM2.5) is provided, tailor the advice to it."
    ]
    if pm:
        prompt_lines.append(f"Latest PM2.5: {pm['pm25']} {pm.get('unit','')}, measured at {pm.get('measured_at')}.")
    if time_text:
        prompt_lines.append(f"The user mentioned time: {time_text}.")
    prompt_lines.append(f"User asked: {message}")
    prompt = "\n".join(prompt_lines)

    try:
        reply_text = call_gemini_chat(prompt, model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    except Exception:
        logger.exception("Gemini call failed; returning deterministic fallback.")
        reply_text = simple_local_recommendation(message, pm, time_text)

    out = {"reply": reply_text, "pm25": pm, "model": os.getenv("GEMINI_MODEL", DEFAULT_MODEL)}
    return jsonify(out), 200

# health and root routes
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}), 200

@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "atmosiq-server", "status": "running"}), 200

if __name__ == "__main__":
    logger.info("Starting AtmosIQ server on port %s (model=%s)", PORT, os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    app.run(host="0.0.0.0", port=PORT, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
