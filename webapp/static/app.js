// Real-time English -> Japanese lecture translator (Option B, browser side).
//
// Architecture:
//   1. Fetch an ephemeral session token from our /session endpoint.
//   2. Open a WebRTC peer connection directly to the OpenAI Realtime API,
//      attaching the microphone track and a data channel.
//   3. Stream incoming events through the data channel and render the
//      English transcript and Japanese translation in two live panes.

(() => {
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const clearBtn = document.getElementById('clearBtn');
  const statusEl = document.getElementById('status');
  const statusText = statusEl.querySelector('.status-text');
  const enLog = document.getElementById('enLog');
  const jaLog = document.getElementById('jaLog');
  const enMeta = document.getElementById('enMeta');
  const jaMeta = document.getElementById('jaMeta');
  const errorBanner = document.getElementById('errorBanner');
  const meterBars = Array.from(document.querySelectorAll('.meter span'));

  /** @type {RTCPeerConnection | null} */ let pc = null;
  /** @type {RTCDataChannel | null} */ let dc = null;
  /** @type {MediaStream | null} */ let mediaStream = null;
  /** @type {AudioContext | null} */ let audioCtx = null;
  /** @type {AnalyserNode | null} */ let analyser = null;
  /** @type {Float32Array | null} */ let meterBuf = null;
  /** @type {number | null} */ let meterRaf = null;
  /** @type {HTMLAudioElement | null} */ let remoteAudioEl = null;
  let running = false;

  // ----- segment state -----
  /** @typedef {{enEl: HTMLElement, jaEl: HTMLElement, time: string}} Row */
  /** @type {Map<string, Row>} */ const rowsByItem = new Map();
  /** @type {string[]} */ const pendingItemQueue = [];
  /** @type {Map<string, string>} */ const responseToItem = new Map();

  let enChars = 0;
  let jaChars = 0;

  // ---------- ui helpers ----------

  const setStatus = (text, klass) => {
    statusText.textContent = text;
    statusEl.className = 'status ' + (klass || 'idle');
  };

  const setError = (msg) => {
    if (!msg) { errorBanner.hidden = true; errorBanner.textContent = ''; return; }
    errorBanner.hidden = false;
    errorBanner.textContent = msg;
  };

  const stampNow = () => {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
  };

  const updateMeta = () => {
    enMeta.textContent = enChars ? `${enChars.toLocaleString()} chars` : '—';
    jaMeta.textContent = jaChars ? `${jaChars.toLocaleString()} chars` : '—';
  };

  const newSegment = (kind, time) => {
    const el = document.createElement('div');
    el.className = 'segment pending';
    el.dataset.time = time;
    el.textContent = '';
    (kind === 'en' ? enLog : jaLog).appendChild(el);
    autoscroll();
    return el;
  };

  const autoscroll = () => {
    enLog.scrollTop = enLog.scrollHeight;
    jaLog.scrollTop = jaLog.scrollHeight;
  };

  const getOrCreateRow = (itemId) => {
    let row = rowsByItem.get(itemId);
    if (row) return row;
    const time = stampNow();
    row = {
      enEl: newSegment('en', time),
      jaEl: newSegment('ja', time),
      time,
    };
    rowsByItem.set(itemId, row);
    pendingItemQueue.push(itemId);
    return row;
  };

  // ---------- event handling ----------

  const handleEvent = (ev) => {
    switch (ev.type) {
      case 'session.created':
      case 'session.updated':
        // No-op; session is configured server-side via /session.
        break;

      case 'input_audio_buffer.speech_started':
        setStatus('speaking', 'speaking');
        break;

      case 'input_audio_buffer.speech_stopped':
        setStatus('listening', 'listening');
        break;

      case 'conversation.item.created': {
        const item = ev.item;
        if (item && item.type === 'message' && item.role === 'user') {
          getOrCreateRow(item.id);
        }
        break;
      }

      case 'conversation.item.input_audio_transcription.delta': {
        const row = getOrCreateRow(ev.item_id);
        const delta = ev.delta || '';
        row.enEl.textContent += delta;
        enChars += delta.length;
        updateMeta();
        autoscroll();
        break;
      }

      case 'conversation.item.input_audio_transcription.completed': {
        const row = getOrCreateRow(ev.item_id);
        if (ev.transcript) {
          if (row.enEl.textContent !== ev.transcript) {
            enChars += Math.max(0, ev.transcript.length - row.enEl.textContent.length);
            row.enEl.textContent = ev.transcript;
          }
        }
        row.enEl.classList.remove('pending');
        if (!row.enEl.textContent.trim()) {
          row.enEl.classList.add('empty');
          row.enEl.textContent = '(silence)';
        }
        updateMeta();
        autoscroll();
        break;
      }

      case 'conversation.item.input_audio_transcription.failed': {
        const row = getOrCreateRow(ev.item_id);
        row.enEl.classList.remove('pending');
        row.enEl.classList.add('error');
        row.enEl.textContent = 'Transcription failed';
        break;
      }

      case 'response.created': {
        const responseId = ev.response && ev.response.id;
        const itemId = pendingItemQueue.shift();
        if (responseId && itemId) responseToItem.set(responseId, itemId);
        break;
      }

      case 'response.text.delta':
      case 'response.output_text.delta': {
        const itemId = responseToItem.get(ev.response_id);
        const row = itemId ? rowsByItem.get(itemId) : null;
        if (!row) break;
        const delta = ev.delta || '';
        row.jaEl.textContent += delta;
        jaChars += delta.length;
        updateMeta();
        autoscroll();
        break;
      }

      case 'response.text.done':
      case 'response.output_text.done': {
        const itemId = responseToItem.get(ev.response_id);
        const row = itemId ? rowsByItem.get(itemId) : null;
        if (!row) break;
        if (ev.text && row.jaEl.textContent !== ev.text) {
          row.jaEl.textContent = ev.text;
        }
        row.jaEl.classList.remove('pending');
        if (!row.jaEl.textContent.trim()) {
          row.jaEl.classList.add('empty');
          row.jaEl.textContent = '(無音)';
        }
        autoscroll();
        break;
      }

      case 'response.done': {
        const responseId = ev.response && ev.response.id;
        if (!responseId) break;
        const itemId = responseToItem.get(responseId);
        const row = itemId ? rowsByItem.get(itemId) : null;
        if (row && row.jaEl.classList.contains('pending')) {
          row.jaEl.classList.remove('pending');
          if (!row.jaEl.textContent.trim()) {
            row.jaEl.classList.add('empty');
            row.jaEl.textContent = '(無音)';
          }
        }
        responseToItem.delete(responseId);
        break;
      }

      case 'error': {
        const msg = (ev.error && (ev.error.message || ev.error.code)) || 'Unknown error';
        console.error('Realtime error:', ev);
        setError('Realtime error: ' + msg);
        break;
      }

      default:
        // Many other events are informational; ignore.
        break;
    }
  };

  // ---------- mic level meter ----------

  const startMeter = (stream) => {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.5;
    src.connect(analyser);
    meterBuf = new Float32Array(analyser.fftSize);
    const tick = () => {
      if (!analyser || !meterBuf) return;
      analyser.getFloatTimeDomainData(meterBuf);
      let sumSq = 0;
      for (let i = 0; i < meterBuf.length; i++) sumSq += meterBuf[i] * meterBuf[i];
      const rms = Math.sqrt(sumSq / meterBuf.length);
      const level = Math.min(1, rms * 12); // 0..1
      meterBars.forEach((bar, i) => {
        const threshold = (i + 1) / meterBars.length;
        const active = level >= threshold * 0.6;
        const h = active ? 4 + (i + 1) * 2 : 4;
        bar.style.height = h + 'px';
        bar.style.background = active
          ? (i < 5 ? 'rgba(90,169,255,0.9)' : i < 7 ? 'rgba(255,168,115,0.9)' : 'rgba(248,113,113,0.95)')
          : 'var(--bg-3)';
      });
      meterRaf = requestAnimationFrame(tick);
    };
    tick();
  };

  const stopMeter = () => {
    if (meterRaf != null) cancelAnimationFrame(meterRaf);
    meterRaf = null;
    analyser = null;
    meterBuf = null;
    if (audioCtx) {
      try { audioCtx.close(); } catch { /* ignore */ }
      audioCtx = null;
    }
    meterBars.forEach((bar) => {
      bar.style.height = '4px';
      bar.style.background = 'var(--bg-3)';
    });
  };

  // ---------- lifecycle ----------

  const start = async () => {
    if (running) return;
    setError('');
    startBtn.disabled = true;
    setStatus('connecting…', 'connecting');

    // 1. ephemeral session token
    let session;
    try {
      const r = await fetch('/session');
      if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + (await r.text()));
      session = await r.json();
    } catch (err) {
      console.error(err);
      setStatus('session error', 'error');
      setError('Failed to fetch session: ' + err.message);
      startBtn.disabled = false;
      return;
    }
    const ephemeralKey = session.client_secret && session.client_secret.value;
    const model = session.model || 'gpt-realtime';
    if (!ephemeralKey) {
      setStatus('session error', 'error');
      setError('No client_secret in session response');
      startBtn.disabled = false;
      return;
    }

    // 2. microphone
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
      setError('Microphone access denied');
      startBtn.disabled = false;
      return;
    }
    startMeter(mediaStream);

    // 3. peer connection
    pc = new RTCPeerConnection();
    pc.oniceconnectionstatechange = () => {
      if (!pc) return;
      const s = pc.iceConnectionState;
      if (s === 'failed' || s === 'disconnected') {
        setStatus('disconnected', 'error');
      }
    };

    // The Realtime API will emit a single audio track on the answer.
    // Even though we asked for text-only, attaching it keeps SDP happy.
    remoteAudioEl = new Audio();
    remoteAudioEl.autoplay = true;
    remoteAudioEl.muted = true;
    pc.ontrack = (ev) => { remoteAudioEl.srcObject = ev.streams[0]; };

    mediaStream.getTracks().forEach((t) => pc.addTrack(t, mediaStream));

    // 4. data channel for events (must be created before the offer)
    dc = pc.createDataChannel('oai-events');
    dc.addEventListener('open', () => {
      // Re-assert session config in case the server-side defaults need
      // overriding (model already configured via /session).
      try {
        dc.send(JSON.stringify({
          type: 'session.update',
          session: {
            modalities: ['text'],
            input_audio_transcription: { model: 'whisper-1', language: 'en' },
            turn_detection: {
              type: 'server_vad',
              threshold: 0.5,
              prefix_padding_ms: 300,
              silence_duration_ms: 700,
              create_response: true,
            },
          },
        }));
      } catch (err) {
        console.warn('session.update send failed', err);
      }
      setStatus('listening', 'listening');
    });
    dc.addEventListener('message', (e) => {
      try { handleEvent(JSON.parse(e.data)); }
      catch (err) { console.warn('bad event json', err, e.data); }
    });
    dc.addEventListener('close', () => {
      if (running) setStatus('disconnected', 'error');
    });

    // 5. SDP offer/answer with OpenAI
    let offer;
    try {
      offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
    } catch (err) {
      console.error(err);
      setStatus('webrtc error', 'error');
      setError('createOffer failed: ' + err.message);
      stop();
      return;
    }

    let answerSdp;
    try {
      const url = `https://api.openai.com/v1/realtime?model=${encodeURIComponent(model)}`;
      const r = await fetch(url, {
        method: 'POST',
        body: offer.sdp,
        headers: {
          Authorization: `Bearer ${ephemeralKey}`,
          'Content-Type': 'application/sdp',
          'OpenAI-Beta': 'realtime=v1',
        },
      });
      if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + (await r.text()));
      answerSdp = await r.text();
    } catch (err) {
      console.error(err);
      setStatus('sdp error', 'error');
      setError('SDP exchange failed: ' + err.message);
      stop();
      return;
    }

    try {
      await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
    } catch (err) {
      console.error(err);
      setStatus('webrtc error', 'error');
      setError('setRemoteDescription failed: ' + err.message);
      stop();
      return;
    }

    running = true;
    stopBtn.disabled = false;
  };

  const stop = () => {
    running = false;
    setStatus('stopped', 'idle');
    try { if (dc && dc.readyState === 'open') dc.close(); } catch { /* ignore */ }
    dc = null;
    if (pc) {
      try { pc.getSenders().forEach((s) => { try { s.track && s.track.stop(); } catch { /* ignore */ } }); } catch { /* ignore */ }
      try { pc.close(); } catch { /* ignore */ }
    }
    pc = null;
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (remoteAudioEl) { remoteAudioEl.srcObject = null; remoteAudioEl = null; }
    stopMeter();
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };

  const clear = () => {
    enLog.innerHTML = '';
    jaLog.innerHTML = '';
    rowsByItem.clear();
    responseToItem.clear();
    pendingItemQueue.length = 0;
    enChars = 0;
    jaChars = 0;
    updateMeta();
    setError('');
  };

  // ---------- wire up ----------

  startBtn.addEventListener('click', () => start().catch((e) => {
    console.error(e);
    setStatus('error', 'error');
    setError(String(e.message || e));
    stop();
  }));
  stopBtn.addEventListener('click', stop);
  clearBtn.addEventListener('click', clear);
  window.addEventListener('beforeunload', stop);

  if (!navigator.mediaDevices || !window.RTCPeerConnection) {
    setStatus('browser unsupported', 'error');
    setError('This browser does not support WebRTC + getUserMedia.');
    startBtn.disabled = true;
  } else {
    setStatus('idle', 'idle');
  }
})();
