// Real-time English -> Japanese lecture translator (browser side).
//
// Captures microphone audio, slices it into variable-length chunks (a
// minimum duration, then cut at the first detected silence / speech
// break), ships each chunk over a WebSocket, and live-updates a
// two-pane (EN / JA) transcript view.

(() => {
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const chunkSecInput = document.getElementById('chunkSec');
  const autoscrollInput = document.getElementById('autoscroll');
  const statusEl = document.getElementById('status');
  const enLog = document.getElementById('enLog');
  const jaLog = document.getElementById('jaLog');

  // Silence detection params.
  const SILENCE_RMS = 0.012;        // RMS amplitude below this counts as "quiet".
  const SILENCE_HOLD_MS = 600;      // continuous quiet duration that defines a "break".
  const VAD_POLL_MS = 80;           // RMS sampling cadence.
  const MAX_CHUNK_MULTIPLIER = 2;   // safety: hard-cut after min * this many seconds.
  const MAX_CHUNK_CEILING_S = 300;  // absolute upper bound (5 min).

  /** @type {WebSocket | null} */ let ws = null;
  /** @type {MediaStream | null} */ let mediaStream = null;
  /** @type {MediaRecorder | null} */ let recorder = null;
  /** @type {AudioContext | null} */ let audioCtx = null;
  /** @type {AnalyserNode | null} */ let analyser = null;
  /** @type {Float32Array | null} */ let vadBuf = null;
  /** @type {number | null} */ let safetyTimer = null;
  /** @type {number | null} */ let vadTimer = null;
  let chunkId = 0;
  let running = false;
  /** @type {Map<number, {en: HTMLElement, ja: HTMLElement}>} */
  const segments = new Map();

  const setStatus = (text, klass) => {
    statusEl.textContent = text;
    statusEl.className = 'status ' + (klass || '');
  };

  const pickMime = () => {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    for (const m of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
    }
    return '';
  };

  const minChunkSec = () => {
    const raw = Number(chunkSecInput.value);
    if (!Number.isFinite(raw)) return 30;
    return Math.max(5, Math.min(180, Math.round(raw)));
  };

  const appendSegment = (id) => {
    const en = document.createElement('div');
    en.className = 'segment pending';
    en.dataset.id = String(id);
    en.textContent = '…';
    enLog.appendChild(en);

    const ja = document.createElement('div');
    ja.className = 'segment pending';
    ja.dataset.id = String(id);
    ja.textContent = '…';
    jaLog.appendChild(ja);

    segments.set(id, { en, ja });
    scrollLogs();
  };

  const scrollLogs = () => {
    if (!autoscrollInput.checked) return;
    enLog.scrollTop = enLog.scrollHeight;
    jaLog.scrollTop = jaLog.scrollHeight;
  };

  const handleMessage = (raw) => {
    let data;
    try { data = JSON.parse(raw); } catch { return; }
    const seg = segments.get(data.id);
    if (!seg) return;
    if (data.type === 'english') {
      if (data.text) {
        seg.en.textContent = data.text;
        seg.en.classList.remove('pending', 'empty');
      } else {
        seg.en.textContent = '(silence)';
        seg.en.classList.remove('pending');
        seg.en.classList.add('empty');
      }
    } else if (data.type === 'japanese') {
      if (data.text) {
        seg.ja.textContent = data.text;
        seg.ja.classList.remove('pending', 'empty');
      } else {
        seg.ja.textContent = '(無音)';
        seg.ja.classList.remove('pending');
        seg.ja.classList.add('empty');
      }
    } else if (data.type === 'error') {
      seg.en.classList.add('error');
      seg.ja.classList.add('error');
      const msg = 'Error: ' + (data.message || 'unknown');
      if (seg.en.classList.contains('pending')) seg.en.textContent = msg;
      if (seg.ja.classList.contains('pending')) seg.ja.textContent = msg;
    }
    scrollLogs();
  };

  const openWs = () => new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const sock = new WebSocket(`${proto}://${location.host}/ws`);
    sock.binaryType = 'arraybuffer';
    sock.addEventListener('open', () => resolve(sock), { once: true });
    sock.addEventListener('error', (e) => reject(e), { once: true });
    sock.addEventListener('message', (ev) => handleMessage(ev.data));
    sock.addEventListener('close', () => {
      if (running) setStatus('disconnected', 'error');
    });
  });

  const setupVad = (stream) => {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0;
    src.connect(analyser);
    vadBuf = new Float32Array(analyser.fftSize);
  };

  const teardownVad = () => {
    if (vadTimer != null) { clearInterval(vadTimer); vadTimer = null; }
    analyser = null;
    vadBuf = null;
    if (audioCtx) {
      try { audioCtx.close(); } catch { /* ignore */ }
      audioCtx = null;
    }
  };

  const readRms = () => {
    if (!analyser || !vadBuf) return 0;
    analyser.getFloatTimeDomainData(vadBuf);
    let sumSq = 0;
    for (let i = 0; i < vadBuf.length; i++) sumSq += vadBuf[i] * vadBuf[i];
    return Math.sqrt(sumSq / vadBuf.length);
  };

  const startRecorderCycle = (mime) => {
    if (!mediaStream || !running) return;

    const id = ++chunkId;
    const rec = new MediaRecorder(mediaStream, mime ? { mimeType: mime } : undefined);
    /** @type {Blob[]} */ const parts = [];

    rec.addEventListener('dataavailable', (e) => {
      if (e.data && e.data.size) parts.push(e.data);
    });

    rec.addEventListener('stop', async () => {
      try {
        const blob = new Blob(parts, { type: rec.mimeType || mime || 'audio/webm' });
        if (blob.size && ws && ws.readyState === WebSocket.OPEN) {
          appendSegment(id);
          ws.send(JSON.stringify({ type: 'chunk', id, mime: blob.type }));
          ws.send(await blob.arrayBuffer());
        }
      } catch (err) {
        console.error('chunk send failed', err);
      }
      if (running) startRecorderCycle(mime);
    });

    rec.start();
    recorder = rec;

    const minSec = minChunkSec();
    const minMs = minSec * 1000;
    const maxSec = Math.min(MAX_CHUNK_CEILING_S, minSec * MAX_CHUNK_MULTIPLIER);
    const cycleStart = performance.now();
    let silenceStart = null;

    const stopThisCycle = () => {
      if (vadTimer != null) { clearInterval(vadTimer); vadTimer = null; }
      if (safetyTimer != null) { clearTimeout(safetyTimer); safetyTimer = null; }
      if (rec.state !== 'inactive') {
        try { rec.stop(); } catch { /* ignore */ }
      }
    };

    safetyTimer = window.setTimeout(stopThisCycle, maxSec * 1000);
    setStatus(`recording (waiting min ${minSec}s)`, 'listening');

    vadTimer = window.setInterval(() => {
      if (!running || recorder !== rec) return;
      const elapsed = performance.now() - cycleStart;
      if (elapsed < minMs) return;

      if (silenceStart == null) {
        setStatus('recording (listening for break)', 'listening');
      }
      const rms = readRms();
      if (rms < SILENCE_RMS) {
        if (silenceStart == null) silenceStart = performance.now();
        if (performance.now() - silenceStart >= SILENCE_HOLD_MS) {
          stopThisCycle();
        }
      } else {
        silenceStart = null;
      }
    }, VAD_POLL_MS);
  };

  const start = async () => {
    if (running) return;
    startBtn.disabled = true;
    setStatus('connecting…', 'connecting');
    try {
      ws = await openWs();
    } catch (err) {
      console.error(err);
      setStatus('ws error', 'error');
      startBtn.disabled = false;
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      console.error(err);
      setStatus('mic denied', 'error');
      ws.close();
      ws = null;
      startBtn.disabled = false;
      return;
    }

    try {
      setupVad(mediaStream);
    } catch (err) {
      console.error(err);
      setStatus('audio init failed', 'error');
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
      ws.close();
      ws = null;
      startBtn.disabled = false;
      return;
    }

    running = true;
    stopBtn.disabled = false;
    setStatus('listening', 'listening');
    const mime = pickMime();
    startRecorderCycle(mime);
  };

  const stop = () => {
    running = false;
    if (vadTimer != null) { clearInterval(vadTimer); vadTimer = null; }
    if (safetyTimer != null) { clearTimeout(safetyTimer); safetyTimer = null; }
    if (recorder && recorder.state !== 'inactive') {
      try { recorder.stop(); } catch { /* ignore */ }
    }
    recorder = null;
    teardownVad();
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    ws = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus('stopped', '');
  };

  startBtn.addEventListener('click', () => { start().catch((e) => {
    console.error(e);
    setStatus('error', 'error');
    stop();
  }); });
  stopBtn.addEventListener('click', stop);
  window.addEventListener('beforeunload', stop);

  if (!navigator.mediaDevices || !window.MediaRecorder) {
    setStatus('browser unsupported', 'error');
    startBtn.disabled = true;
  }
})();
