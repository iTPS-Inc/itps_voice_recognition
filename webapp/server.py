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
    "You are a professional simultaneous interpreter for an English-language "
    "public lecture. Your sole task: translate every utterance you receive "
    "into natural, fluent Japanese while preserving the complete meaning of "
    "the source.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY the Japanese translation.\n"
    "- Translate the ENTIRE substantive input. Do not summarize, abbreviate, "
    "paraphrase loosely, or skip any sentence, clause, or idea that carries "
    "meaning. Every meaningful clause in the English input must appear in "
    "the Japanese output.\n"
    "- DO NOT reproduce disfluencies. Specifically:\n"
    "  * Drop filler words and hesitation markers such as 'um', 'uh', 'ah', "
    "'er', 'mm', 'you know', 'I mean', 'like', 'kind of' / 'sort of', and "
    "'so' / 'well' / 'right' / 'okay' when used purely as hesitation. Keep "
    "these words only when they carry real semantic weight.\n"
    "  * When the speaker mis-speaks and self-corrects (false starts, slips "
    "of the tongue, restatements), translate ONLY the corrected/final "
    "version. Do not reproduce the abandoned phrase.\n"
    "  * Collapse stutters and word repetitions ('the the', 'I-I think') "
    "into a single occurrence.\n"
    "- Finish your translation completely before stopping. Never end "
    "mid-sentence.\n"
    "- Match the speaker's register (formal lecture style by default).\n"
    "- Preserve technical terms appropriately (use katakana or the original "
    "English where commonly used in Japanese technical writing).\n"
    "- Numbers, percentages, dates, and proper nouns must be conveyed "
    "exactly.\n"
    "- No commentary, no quotation marks, no romanization, no English.\n"
    "- Always produce a translation whenever there is substantive speech in "
    "the input. Only return an empty response if the input contains nothing "
    "but disfluencies or silence."
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
            "silence_duration_ms": 900,
            # The browser drives response.create manually so each user
            # utterance gets exactly one response, even when the speaker
            # overlaps with the previous translation.  With auto-create
            # plus interrupt_response=False the server VAD races with the
            # in-flight response and silently drops some utterances,
            # which shifts the EN/JA timeline alignment.
            "create_response": False,
            "interrupt_response": False,
        },
        "temperature": 0.6,
        "max_response_output_tokens": 4096,
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
