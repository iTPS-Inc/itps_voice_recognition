// Real-time English -> Japanese lecture translator (browser side).
//
// Captures microphone audio, slices it into fixed-length chunks via a
// MediaRecorder restart cycle, ships each chunk to the server over a
// WebSocket, and live-updates a two-pane (EN / JA) transcript view.

(() => {
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const chunkSecInput = document.getElementById('chunkSec');
  const autoscrollInput = document.getElementById('autoscroll');
  const statusEl = document.getElementById('status');
  const enLog = document.getElementById('enLog');
  const jaLog = document.getElementById('jaLog');

  /** @type {WebSocket | null} */ let ws = null;
  /** @type {MediaStream | null} */ let mediaStream = null;
  /** @type {MediaRecorder | null} */ let recorder = null;
  /** @type {number | null} */ let cycleTimer = null;
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

    const chunkSec = Math.max(2, Math.min(15, Number(chunkSecInput.value) || 5));
    cycleTimer = window.setTimeout(() => {
      if (rec.state !== 'inactive') rec.stop();
    }, chunkSec * 1000);
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

    running = true;
    stopBtn.disabled = false;
    setStatus('listening', 'listening');
    const mime = pickMime();
    startRecorderCycle(mime);
  };

  const stop = () => {
    running = false;
    if (cycleTimer != null) { clearTimeout(cycleTimer); cycleTimer = null; }
    if (recorder && recorder.state !== 'inactive') {
      try { recorder.stop(); } catch { /* ignore */ }
    }
    recorder = null;
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
