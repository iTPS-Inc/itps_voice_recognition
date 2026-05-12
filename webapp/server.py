"""Real-time English -> Japanese lecture translator (Option B: gpt-realtime).

Architecture (browser-first):
    Browser microphone (WebRTC) <----> OpenAI Realtime API
                                          |
                                          +-- whisper-1 transcription (English)
                                          +-- gpt-realtime translation (Japanese text)

This server's only job is to mint short-lived ephemeral session tokens so
the browser can talk to OpenAI directly without ever seeing the real API
key.  Audio and event traffic do NOT flow through this server.

Run:
    pip install -r webapp/requirements.txt
    export OPENAI_API_KEY=sk-...
    uvicorn webapp.server:app --host 0.0.0.0 --port 8000

Open http://localhost:8000 (the browser requires localhost or HTTPS for
microphone access).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("translator")

STATIC_DIR = Path(__file__).parent / "static"
REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-realtime")
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "whisper-1")
SESSIONS_URL = "https://api.openai.com/v1/realtime/sessions"

TRANSLATION_INSTRUCTIONS = (
    "You are a professional simultaneous interpreter. The user is speaking "
    "English at a public lecture. For every utterance you receive, output a "
    "natural, fluent Japanese translation suitable for a live audience.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY the Japanese translation.\n"
    "- No commentary, no quotation marks, no romanization, no English.\n"
    "- Preserve technical terms appropriately (katakana or original English "
    "where commonly used in Japanese technical writing).\n"
    "- Match the speaker's register (formal lecture style by default).\n"
    "- If the input is too short or unintelligible, output an empty string."
)

app = FastAPI(title="Lecture Translator (Realtime)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/download")
async def download_desktop() -> FileResponse:
    """Serve the standalone, single-file desktop edition for download.

    The returned HTML is fully self-contained (inline CSS + JS) so the user
    can save it to the desktop and double-click to launch.  It uses the
    user's own OpenAI key (entered in the page, stored in localStorage) to
    talk directly to the Realtime API and needs no backend.
    """
    return FileResponse(
        STATIC_DIR / "lecture-translator.html",
        media_type="text/html; charset=utf-8",
        filename="lecture-translator.html",
        headers={
            "Content-Disposition": 'attachment; filename="lecture-translator.html"',
        },
    )


@app.get("/session")
async def session() -> JSONResponse:
    """Mint a short-lived ephemeral session token for the browser.

    The browser receives ``client_secret.value`` and uses it as a Bearer
    token for the WebRTC SDP exchange with OpenAI.  The token typically
    expires within a minute, so this endpoint is called once per session.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on the server")

    payload = {
        "model": REALTIME_MODEL,
        "modalities": ["text"],
        "instructions": TRANSLATION_INSTRUCTIONS,
        "input_audio_transcription": {"model": TRANSCRIBE_MODEL, "language": "en"},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 700,
            "create_response": True,
        },
        "temperature": 0.6,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                SESSIONS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        log.exception("session minting failed")
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc

    if resp.status_code >= 400:
        log.error("OpenAI returned %s: %s", resp.status_code, resp.text)
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JSONResponse(resp.json())
