"""Real-time English -> Japanese lecture translator.

Pipeline:
    browser mic -> WebSocket (audio chunks)
        -> Whisper (English transcription)
        -> GPT (Japanese translation)
        -> WebSocket -> browser (live two-pane display)

Run:
    pip install -r webapp/requirements.txt
    export OPENAI_API_KEY=sk-...
    uvicorn webapp.server:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in a browser (microphone access requires
either http://localhost or https://).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("translator")

STATIC_DIR = Path(__file__).parent / "static"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-1")
TRANSLATION_MODEL = os.environ.get("TRANSLATION_MODEL", "gpt-4o-mini")

TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter translating an English "
    "lecture into Japanese in real time. Translate the user's English text "
    "into natural, fluent Japanese suitable for a live audience. Preserve "
    "technical terms appropriately (use katakana or original English where "
    "common). Output ONLY the Japanese translation, with no commentary, "
    "no quotation marks, no romanization, and no English."
)

app = FastAPI(title="Lecture Translator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
client = AsyncOpenAI()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _suffix_for_mime(mime: str) -> str:
    mime = (mime or "").lower()
    if "ogg" in mime:
        return ".ogg"
    if "wav" in mime:
        return ".wav"
    if "mp4" in mime or "m4a" in mime or "aac" in mime:
        return ".m4a"
    return ".webm"


async def transcribe(audio_bytes: bytes, suffix: str) -> str:
    """Send a single audio chunk to Whisper and return the English text."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(audio_bytes)
        path = fh.name
    try:
        with open(path, "rb") as f:
            resp = await client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                language="en",
                response_format="text",
                temperature=0,
            )
        return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
    finally:
        Path(path).unlink(missing_ok=True)


async def translate(text: str) -> str:
    """Translate English text to Japanese via GPT."""
    resp = await client.chat.completions.create(
        model=TRANSLATION_MODEL,
        messages=[
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# Whisper tends to hallucinate these on silent / near-silent input.
_HALLUCINATION_PHRASES = {
    "thank you.",
    "thanks for watching.",
    "thanks for watching!",
    "thank you for watching.",
    "you",
    ".",
    "bye.",
    "okay.",
}


def _is_likely_hallucination(text: str) -> bool:
    return text.strip().lower() in _HALLUCINATION_PHRASES


async def _safe_send(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
    except Exception:  # noqa: BLE001 - client may have disconnected
        log.debug("send failed for payload type=%s", payload.get("type"))


async def _process_chunk(ws: WebSocket, chunk_id: int, audio: bytes, suffix: str) -> None:
    try:
        english = (await transcribe(audio, suffix=suffix)).strip()
    except Exception as exc:  # noqa: BLE001
        log.exception("transcribe failed for chunk %s", chunk_id)
        await _safe_send(ws, {"type": "error", "id": chunk_id, "message": f"transcribe: {exc}"})
        return

    if not english or _is_likely_hallucination(english):
        await _safe_send(ws, {"type": "english", "id": chunk_id, "text": ""})
        await _safe_send(ws, {"type": "japanese", "id": chunk_id, "text": ""})
        return

    await _safe_send(ws, {"type": "english", "id": chunk_id, "text": english})

    try:
        japanese = await translate(english)
    except Exception as exc:  # noqa: BLE001
        log.exception("translate failed for chunk %s", chunk_id)
        await _safe_send(ws, {"type": "error", "id": chunk_id, "message": f"translate: {exc}"})
        return

    await _safe_send(ws, {"type": "japanese", "id": chunk_id, "text": japanese})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Receive interleaved JSON meta + binary audio chunks from the browser.

    Protocol per chunk:
        1. text frame: {"type": "chunk", "id": <int>, "mime": "audio/webm"}
        2. binary frame: raw bytes of a self-contained audio file
    Server replies (out of order possible across chunks, in order within a chunk):
        {"type": "english",  "id": <int>, "text": "..."}
        {"type": "japanese", "id": <int>, "text": "..."}
        {"type": "error",    "id": <int>, "message": "..."}
    """
    await ws.accept()
    log.info("client connected")
    pending_meta: dict[str, Any] | None = None
    tasks: set[asyncio.Task[None]] = set()
    try:
        while True:
            msg = await ws.receive()
            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                break
            text = msg.get("text")
            data = msg.get("bytes")
            if text is not None:
                try:
                    pending_meta = json.loads(text)
                except json.JSONDecodeError:
                    log.warning("ignoring malformed json frame: %r", text[:120])
                    pending_meta = None
                continue
            if data is None:
                continue
            meta = pending_meta or {}
            pending_meta = None
            chunk_id = int(meta.get("id", 0))
            suffix = _suffix_for_mime(meta.get("mime", ""))
            task = asyncio.create_task(_process_chunk(ws, chunk_id, data, suffix))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass
    finally:
        log.info("client disconnected; %d pending task(s)", len(tasks))
        for t in tasks:
            t.cancel()
