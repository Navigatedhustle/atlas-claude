/* Atlas front end. Vanilla JS, no network beyond this box. */
(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const BLUE = '#004BDF', BLUE_SOFT = 'rgba(0,75,223,0.42)', META = '#8C8C8C', GRID = '#D1DDF1';

  const S = {
    persona: 'enterprise', cfg: null, ws: null, state: 'idle', chart: null, lastResult: null,
    audioCtx: null, audioQueue: new Map(), nextSeq: 0, playing: null, answerSeqs: 0, doneEvent: null,
    tAsk: null, tRelease: null, firstAudioAt: null, source: 'typed', asrMs: null,
    mic: null, micNode: null, asrWs: null, listening: false, finalText: '', partialText: '',
    lastInteraction: Date.now(), attract: false, attractIdx: 0, goldenIdx: 0,
    people: 0, bigPerson: false, voice: true, lastNet: null, history: [],
  };

  // ------------------------------------------------------------------ layout scaling
  function fit() {
    const frame = $('#frame');
    const w = document.body.classList.contains('v916') ? 1080 : 1920;
    const h = document.body.classList.contains('v916') ? 1920 : 1080;
    const s = Math.min(window.innerWidth / w, window.innerHeight / h);
    frame.style.transform = `scale(${s})`;
    frame.style.left = `${(window.innerWidth - w * s) / 2}px`;
    frame.style.top = `${(window.innerHeight - h * s) / 2}px`;
  }
  window.addEventListener('resize', fit);

  // ------------------------------------------------------------------ formatting (mirrors atlas/analytics/common.py)
  function money(x, digits) {
    if (x === null || x === undefined || isNaN(x)) return 'n/a';
    const s = x < 0 ? '-' : ''; const a = Math.abs(x);
    if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(digits ?? 2)}B`;
    if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(digits ?? 1)}M`;
    if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(digits ?? 0)}K`;
    return `${s}$${a.toFixed(0)}`;
  }
  const fmt = {
    money: (v) => money(v), pct: (v) => `${(v * 100).toFixed(1)}%`, num: (v) => Math.round(v).toLocaleString('en-US'),
    x: (v) => `${v.toFixed(2)}x`, score: (v) => v.toFixed(0), dec: (v) => v.toFixed(2), text: (v) => String(v),
  };
  const axisFmt = {
    money: (v) => money(v, Math.abs(v) >= 1e9 ? 1 : 0), pct: (v) => `${Math.round(v * 100)}%`,
    num: (v) => Math.round(v).toLocaleString('en-US'), x: (v) => `${v.toFixed(1)}x`, score: (v) => v.toFixed(0),
  };
  function bytes(n) {
    if (n < 1024) return `${n} bytes`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // ------------------------------------------------------------------ state machine
  function setState(st) {
    S.state = st;
    document.body.classList.remove('state-idle', 'state-listening', 'state-thinking', 'state-speaking');
    document.body.classList.add(`state-${st}`);
    $('#ptt-text').textContent = { idle: 'HOLD TO ASK', listening: 'LISTENING…', thinking: 'THINKING…', speaking: 'SPEAKING · PRESS TO STOP' }[st];
    if (window.AtlasOrb) window.AtlasOrb.setState(st);
  }
  // Short label on the orb for each pipeline stage, shown the moment it really happens.
  function orbStage(text) {
    const el = $('#orb-stage'); if (!el) return;
    el.textContent = text; el.classList.add('on');
    clearTimeout(S.stageTimer); S.stageTimer = setTimeout(() => el.classList.remove('on'), 1600);
  }
  const ink = () => getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#000';
  const gridC = () => getComputedStyle(document.body).getPropertyValue('--line').trim() || GRID;

  // ------------------------------------------------------------------ main websocket
  function connect() {
    const ws = new WebSocket(`ws://${location.host}/ws`);
    S.ws = ws;
    ws.onmessage = (e) => onEvent(JSON.parse(e.data));
    ws.onclose = () => setTimeout(connect, 1000);
  }

  function ask(text, source = 'typed') {
    if (!text || !text.trim()) return;
    stopAudio();
    S.source = source;
    S.tAsk = performance.now();
    if (source !== 'voice') S.tRelease = S.tAsk;
    S.firstAudioAt = null; S.nextSeq = 0; S.audioQueue.clear(); S.answerSeqs = 0; S.doneEvent = null;
    ensureAudio();
    setState('thinking');
    $('#answer').innerHTML = '';
    $('#grounded').textContent = ''; $('#grounded').className = 'grounded';
    $('#s1-items').innerHTML = '<div class="meta">Deciding…</div>'; $('#s1-ms').textContent = '';
    $('#route-info').textContent = '';
    S.askId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const msg = JSON.stringify({ type: 'ask', id: S.askId, text, persona: S.persona, voice: S.voice, source, lang: S.lang || 'en' });
    if (S.ws && S.ws.readyState === 1) S.ws.send(msg);
    else { const t = setInterval(() => { if (S.ws && S.ws.readyState === 1) { clearInterval(t); S.ws.send(msg); } }, 200); }
    addHistory(text);
    // Watchdog: never leave the booth stuck in "thinking".
    clearTimeout(S.watchdog);
    S.watchdog = setTimeout(() => {
      if (S.state !== 'thinking') return;
      setState('idle');
      if (!$('#answer').textContent.trim()) $('#answer').innerHTML = '<span class="meta">That took too long. Please ask again.</span>';
    }, S.persona === 'developer' ? 60000 : 20000);
  }

  function flow(k, text, state = 'done') {
    const el = document.querySelector(`#flow [data-k="${k}"]`); if (!el) return;
    el.className = state; if (text) el.textContent = text;
  }
  function flowReset() {
    const labels = { heard: S.source === 'voice' ? 'HEARD' : 'TYPED', pick: 'TOOL PICKED', math: 'COMPUTED', proof: 'VERIFIED', spoke: S.voice ? 'SPOKE' : 'ANSWERED' };
    Object.entries(labels).forEach(([k, t]) => flow(k, t, ''));
    $('#flow').classList.remove('hidden');
  }
  function onEvent(ev) {
    // Ignore anything from an answer that was interrupted or superseded.
    if (ev.id !== undefined && ev.id !== null && ev.id !== S.askId) return;
    switch (ev.type) {
      case 'question':
        $('#question').textContent = ev.text;
        if (S.persona === 'developer') { $('#dev-question').textContent = ev.text; devReset(); }
        flowReset();
        flow('heard', S.source === 'voice' && S.asrMs ? `HEARD · ${Math.round(S.asrMs)} MS` : (S.source === 'voice' ? 'HEARD' : 'TYPED'));
        orbStage(S.source === 'voice' ? 'HEARD' : 'QUESTION RECEIVED');
        flow('pick', null, 'active');
        break;
      case 'route':
        $('#route-info').textContent = `${ev.by === 'llm' ? 'Nemotron chose' : 'Router chose'} ${ev.tool}`;
        // Document questions open the second brain so the cited passages visibly light up.
        if (ev.tool === 'search_knowledge' && S.cfg.features.brain && S.persona !== 'developer' && !document.body.classList.contains('view-brain')) setView(true);
        flow('pick', `${ev.by === 'llm' ? 'LLM' : 'ROUTER'} PICKED ${ev.tool.toUpperCase()} · ${Math.round(ev.ms || 0)} MS`);
        orbStage(`TOOL PICKED · ${ev.tool.replace(/_/g, ' ').toUpperCase()}`);
        flow('math', null, 'active');
        break;
      case 'result':
        renderResult(ev.result);
        flow('math', `COMPUTED · ${Math.max(1, Math.round(ev.ms || 0))} MS`);
        orbStage(`COMPUTED · ${Math.max(1, Math.round(ev.ms || 0))} MS`);
        flow('proof', null, 'active');
        break;
      case 'sentence': {
        if (ev.seq === 0) { flow('proof', 'VERIFYING…', 'active'); flow('spoke', null, S.voice ? 'active' : ''); orbStage('VERIFIED · SPEAKING'); }
        const span = document.createElement('span');
        span.className = 's'; span.dataset.seq = ev.seq; span.textContent = ev.text + ' ';
        $('#answer').appendChild(span);
        S.answerSeqs = Math.max(S.answerSeqs, ev.seq + 1);
        if (!S.voice && !S.firstAudioAt) { S.firstAudioAt = performance.now(); }
        break;
      }
      case 'audio': enqueueAudio(ev); break;
      case 'audio_error': S.audioQueue.set(ev.seq, { skip: true }); pump(); break;
      case 'decisions': renderDecisions(ev); break;
      case 'code': renderCode(ev); break;
      case 'run': renderRun(ev); break;
      case 'endpoint': S.endpointLive = ev; renderEndpoint(); break;
      case 'provenance': if (window.AtlasBrain && S.cfg.features.brain) window.AtlasBrain.provenance(ev); break;
      case 'done': {
        S.doneEvent = ev;
        const spans = $$('#answer .s');
        const last = spans[spans.length - 1];
        if (last && /[,;:]\s*$/.test(last.textContent)) last.textContent = last.textContent.replace(/[,;:]\s*$/, '. ');
        renderGrounding(ev.grounding);
        if (ev.grounding) flow('proof', `${ev.grounding.verified}/${ev.grounding.numbers} NUMBERS VERIFIED`);
        if (!S.voice) flow('spoke', `ANSWERED · ${((performance.now() - S.tAsk) / 1000).toFixed(2)} S`);
        renderLatency(ev.timings);
        if (!S.voice || !ev.audio || ev.sentences === 0) finishAnswer();
        if (S.voice && !ev.audio && ev.sentences > 0) flow('spoke', 'TEXT ONLY · VOICE OFFLINE', '');
        break;
      }
      case 'stopped': break;
    }
  }

  // ------------------------------------------------------------------ audio out (sentence queue, barge-in)
  function ensureAudio() {
    if (!S.audioCtx) {
      S.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      // Every TTS chunk plays through this analyser so the orb moves with the real voice.
      S.outAn = S.audioCtx.createAnalyser(); S.outAn.connect(S.audioCtx.destination);
      if (window.AtlasOrb) window.AtlasOrb.attachOutput(S.outAn);
    }
    if (S.audioCtx.state === 'suspended') S.audioCtx.resume();
  }
  async function enqueueAudio(ev) {
    ensureAudio();
    const bin = Uint8Array.from(atob(ev.wav), (c) => c.charCodeAt(0));
    try {
      const buf = await S.audioCtx.decodeAudioData(bin.buffer);
      S.audioQueue.set(ev.seq, { buf, tts_ms: ev.tts_ms });
      pump();
    } catch (e) { console.warn('decode', e); }
  }
  function pump() {
    if (S.playing) return;
    const item = S.audioQueue.get(S.nextSeq);
    if (!item) {
      if (S.doneEvent && S.nextSeq >= S.answerSeqs) finishAnswer();
      return;
    }
    S.audioQueue.delete(S.nextSeq);
    if (item.skip) { S.nextSeq += 1; pump(); return; }
    const src = S.audioCtx.createBufferSource();
    src.buffer = item.buf; src.connect(S.outAn);
    const seq = S.nextSeq;
    $$('#answer .s').forEach((s) => s.classList.toggle('speaking', +s.dataset.seq === seq));
    src.onended = () => {
      S.playing = null; S.nextSeq += 1; pump();
      // Spoke a status line but the tools are still running: the orb goes back to thinking.
      if (!S.playing && !S.doneEvent && S.state === 'speaking') setState('thinking');
    };
    S.playing = src;
    src.start();
    setState('speaking');
    if (!S.firstAudioAt) {
      S.firstAudioAt = performance.now();
      const ttfa = S.firstAudioAt - (S.tRelease || S.tAsk);
      S.lastTTFA = ttfa;
      flow('spoke', `SPOKE · ${(ttfa / 1000).toFixed(2)} S`);
      renderLatency(null, { ttfa, tts: item.tts_ms });
      fetch('/api/latency', { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ ttfa_ms: ttfa, source: S.source, asr_ms: S.asrMs, tts_ms: item.tts_ms, server: S.doneEvent && S.doneEvent.timings }) });
    }
  }
  function stopAudio() {
    if (S.playing) { try { S.playing.onended = null; S.playing.stop(); } catch (e) { /* ignore */ } S.playing = null; }
    S.audioQueue.clear();
    $$('#answer .s').forEach((s) => s.classList.remove('speaking'));
  }
  function finishAnswer() {
    $$('#answer .s').forEach((s) => s.classList.remove('speaking'));
    if (S.state !== 'listening') setState('idle');
    S.lastInteraction = Date.now();
    if (S.attract) setTimeout(() => { if (S.attract) attractNext(); }, 9000);
  }
  function bargeIn() {
    S.askId = null;
    if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify({ type: 'stop' }));
    stopAudio();
    S.doneEvent = null;
    setState('idle');
  }

  // ------------------------------------------------------------------ rendering
  function countUp(el, value, kind) {
    if (kind === 'text' || typeof value !== 'number') {
      el.textContent = value; el.classList.add('text'); el.classList.toggle('long', String(value).length > 18); return;
    }
    el.classList.remove('text');
    const t0 = performance.now(), dur = 900;
    const f = fmt[kind] || fmt.num;
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      el.textContent = f(kind === 'num' ? Math.round(value * e) : value * e);
      if (k < 1) requestAnimationFrame(step); else el.textContent = f(value);
    };
    requestAnimationFrame(step);
  }

  function renderResult(r) {
    S.lastResult = r;
    $('#how-idle').classList.add('hidden'); $('#how-live').classList.remove('hidden');
    $('#headline').classList.remove('idle');
    $('#headline-label').textContent = r.headline.label;
    countUp($('#headline'), r.headline.value, r.headline.fmt);
    $('#headline-sub').textContent = r.headline.sub || '';
    // percentiles
    const p = r.percentiles;
    if (p && p.P10 !== undefined) {
      const f = fmt[p.fmt] || fmt.num;
      $('#pctl').classList.remove('hidden');
      const lo = p.P10 - (p.P90 - p.P10) * 0.6, hi = p.P90 + (p.P90 - p.P10) * 0.6;
      const pos = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
      $('.pctl-fill').style.left = pos(p.P10); $('.pctl-fill').style.width = `${((p.P90 - p.P10) / (hi - lo)) * 100}%`;
      $('.pctl-mid').style.left = pos(p.P50);
      $('#p10').textContent = `P10 ${f(p.P10)}`; $('#p50').textContent = `P50 ${f(p.P50)}`; $('#p90').textContent = `P90 ${f(p.P90)}`;
      $('#how-range').classList.remove('hidden');
      $('#how-range-v').textContent = `${p.label}: P10 ${f(p.P10)} · P50 ${f(p.P50)} · P90 ${f(p.P90)}`;
    } else { $('#pctl').classList.add('hidden'); $('#how-range').classList.add('hidden'); }
    // how I got this
    $('#how-tool').textContent = r.tool;
    $('#how-summary').textContent = r.method.summary;
    $('#how-formula').textContent = r.method.formula;
    $('#how-inputs').innerHTML = r.inputs.slice(0, r.percentiles ? 4 : 6).map((i) => `<tr><td>${esc(i.label)}</td><td>${esc(i.text)}</td></tr>`).join('');
    $('#how-steps').innerHTML = (r.method.steps || []).slice(0, r.percentiles ? 1 : 2).map((s) => `<li>${esc(s)}</li>`).join('');
    $('#how-sources').textContent = r.sources.join(' · ');
    renderChart(r.chart, r.table);
  }

  const bandPlugin = {
    id: 'band',
    afterDatasetsDraw(chart, args, opts) {
      const band = opts && opts.band; const ref = opts && opts.ref;
      const { ctx } = chart;
      if (band) {
        const meta = chart.getDatasetMeta(chart.data.datasets.length - 1);
        const bar = meta.data[meta.data.length - 1]; if (!bar) return;
        const y = chart.scales.y;
        const x = bar.x, y1 = y.getPixelForValue(band.low), y2 = y.getPixelForValue(band.high);
        ctx.save(); ctx.strokeStyle = ink(); ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2);
        ctx.moveTo(x - 14, y1); ctx.lineTo(x + 14, y1); ctx.moveTo(x - 14, y2); ctx.lineTo(x + 14, y2); ctx.stroke();
        ctx.fillStyle = ink(); ctx.font = 'bold 13px Arial'; ctx.textAlign = 'left';
        if (Math.abs(y1 - y2) < 22) ctx.fillText('P10–P90', x + 18, (y1 + y2) / 2 + 5);
        else { ctx.fillText('P90', x + 18, y2 + 5); ctx.fillText('P10', x + 18, y1 + 5); }
        ctx.restore();
      }
      if (ref) {
        const y = chart.scales.y.getPixelForValue(ref.value);
        ctx.save(); ctx.strokeStyle = META; ctx.setLineDash([6, 5]); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(chart.chartArea.left, y); ctx.lineTo(chart.chartArea.right, y); ctx.stroke();
        ctx.fillStyle = META; ctx.font = 'bold 13px Arial'; ctx.fillText(ref.label, chart.chartArea.left + 8, y - 6);
        ctx.restore();
      }
    },
  };
  const valueLabels = {
    id: 'valueLabels',
    afterDatasetsDraw(chart, args, opts) {
      if (!opts || !opts.fmt) return;
      const { ctx } = chart; const horiz = chart.options.indexAxis === 'y';
      chart.data.datasets.forEach((ds, di) => {
        if (ds._muted) return;
        const meta = chart.getDatasetMeta(di);
        meta.data.forEach((el, i) => {
          const v = ds.data[i]; if (v === null || v === undefined) return;
          ctx.save(); ctx.fillStyle = ink(); ctx.font = 'bold 14px Arial';
          const t = (fmt[opts.fmt] || fmt.num)(v);
          if (horiz) { ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(t, el.x + 8, el.y); }
          else {
            let top = Math.min(el.y, el.base);
            const band = chart.options.plugins.band && chart.options.plugins.band.band;
            if (band && di === chart.data.datasets.length - 1 && i === meta.data.length - 1) top = Math.min(top, chart.scales.y.getPixelForValue(band.high)) - 16;
            ctx.textAlign = 'center'; ctx.fillText(t, el.x, top - 8);
          }
          ctx.restore();
        });
      });
    },
  };

  function renderCitations(r) {
    $('#chart-empty').classList.add('hidden');
    $('#chart-title').textContent = 'CITED PASSAGES · SYNTHETIC DOCUMENTS';
    $('#table-wrap').classList.remove('hidden'); $('.chart-block').classList.add('has-table');
    $('#table-wrap').innerHTML = r.citations.map((c) => `<div class="cite"><div class="cite-h"><b>${esc(c.title)}</b><span>passage ${c.n + 1} · match ${c.score.toFixed(2)}</span></div><div class="cite-t">${esc(c.text)}</div></div>`).join('');
  }
  function renderChart(c, table) {
    if (S.lastResult && S.lastResult.citations && S.lastResult.tool === 'search_knowledge') { if (S.chart) { S.chart.destroy(); S.chart = null; } renderCitations(S.lastResult); return; }
    if (c.type === 'none') {
      if (S.chart) { S.chart.destroy(); S.chart = null; }
      $('#chart-title').textContent = 'TRY ASKING';
      $('#chart-empty').classList.remove('hidden');
      $('#table-wrap').classList.add('hidden');
      $('.chart-block').classList.remove('has-table');
      return;
    }
    $('#chart-empty').classList.add('hidden');
    $('#chart-title').textContent = c.title;
    if (S.chart) S.chart.destroy();
    const horiz = c.type === 'hbar';
    const n = c.labels.length;
    const datasets = c.series.map((s, si) => {
      const last = si === c.series.length - 1;
      let color = s.muted || (!last && c.series.length > 1) ? '#9FB4DF' : BLUE;
      let bg = color;
      if (c.highlight_last && last) bg = c.labels.map((_, i) => (i === n - 1 ? BLUE : BLUE_SOFT));
      return { label: s.name, data: s.data, backgroundColor: bg, borderWidth: 0, _muted: !!s.muted || (c.series.length > 1 && !last && horiz),
        barPercentage: 0.78, categoryPercentage: 0.8 };
    });
    // For hbar with two series (amount at risk vs deal amount): show total as muted background
    if (horiz && datasets.length > 1) { datasets[1].backgroundColor = '#C9D6EE'; datasets[0].backgroundColor = BLUE; datasets[1]._muted = true; datasets[0]._muted = false; }
    const vals = c.series.flatMap((s) => s.data).concat(c.band ? [c.band.high] : []).concat(c.ref ? [c.ref.value] : []);
    const max = Math.max(...vals);
    const ax = axisFmt[c.fmt] || axisFmt.num;
    S.chart = new Chart($('#chart'), {
      type: 'bar',
      data: { labels: c.labels, datasets },
      plugins: [bandPlugin, valueLabels],
      options: {
        indexAxis: horiz ? 'y' : 'x', responsive: true, maintainAspectRatio: false, animation: { duration: 900, easing: 'easeOutCubic' },
        layout: { padding: { top: 26, right: horiz ? 90 : 46, left: 6, bottom: 0 } },
        plugins: {
          legend: { display: c.series.length > 1, position: 'top', align: 'end', labels: { font: { family: 'Arial', size: 14, weight: 'bold' }, color: ink(), boxWidth: 14 } },
          tooltip: { enabled: false }, band: { band: c.band, ref: c.ref }, valueLabels: { fmt: c.fmt },
        },
        scales: {
          [horiz ? 'x' : 'y']: { beginAtZero: true, suggestedMax: max * (horiz ? 1.12 : 1.08),
            grid: { color: gridC() }, border: { display: false },
            ticks: { font: { family: 'Arial', size: 13 }, color: ink(), callback: (v) => ax(v), maxTicksLimit: 6 } },
          [horiz ? 'y' : 'x']: { grid: { display: false }, border: { color: ink() },
            ticks: { font: { family: 'Arial', size: 14, weight: 'bold' }, color: ink(), autoSkip: false } },
        },
      },
    });
    const tw = $('#table-wrap');
    $('.chart-block').classList.toggle('has-table', !!(table && table.rows && table.rows.length));
    if (table && table.rows && table.rows.length) {
      tw.classList.remove('hidden');
      tw.innerHTML = `<table class="grid"><tr>${table.columns.map((h) => `<th>${esc(h)}</th>`).join('')}</tr>${table.rows.slice(0, 5).map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join('')}</tr>`).join('')}</table>`;
    } else { tw.classList.add('hidden'); tw.innerHTML = ''; }
  }

  function renderDecisions(d) {
    const box = $('#s1-items');
    if (!d.items || !d.items.length) { box.innerHTML = '<div class="meta">No typed decision for this question.</div>'; $('#s1-ms').textContent = ''; return; }
    $('#s1-ms').textContent = `${d.engine === 'kev-4b' ? '' : 'LLM fallback · '}${Math.round(d.ms)} ms`;
    const rows = [];
    d.items.slice(0, 5).forEach((it) => {
      if (it.type === 'noul') {
        const q = it.label === it.question ? it.question : `Is ${it.label} at risk?`;
        rows.push({ q, p: it.p_yes });
      } else {
        rows.push({ q: it.question, header: true });
        it.options.slice(0, 3).forEach((o) => rows.push({ q: o.name, p: o.p }));
      }
    });
    box.innerHTML = rows.map((r) => r.header ? `<div class="s1-item"><div class="s1-q">${esc(r.q)}</div></div>` :
      `<div class="s1-item"><div class="s1-q"><span>${esc(r.q)}</span><b>${r.p.toFixed(2)}</b></div><div class="s1-bar"><div data-w="${(r.p * 100).toFixed(1)}"></div></div></div>`).join('');
    requestAnimationFrame(() => $$('#s1-items .s1-bar > div').forEach((b) => { b.style.width = `${b.dataset.w}%`; }));
  }

  function renderGrounding(g) {
    const el = $('#grounded');
    if (!g) return;
    if (g.mode === 'template') { el.textContent = `TEMPLATE ANSWER · ${g.verified}/${g.numbers} NUMBERS FROM TOOLS`; el.className = 'grounded warn'; }
    else if (g.failed_sentence) { el.textContent = `✓ ${g.verified}/${g.numbers} NUMBERS VERIFIED · 1 UNGROUNDED SENTENCE BLOCKED`; el.className = 'grounded warn'; }
    else { el.textContent = `✓ ${g.verified}/${g.numbers} NUMBERS VERIFIED AGAINST TOOL OUTPUT`; el.className = 'grounded ok'; }
  }

  const LAT = { asr: 0, route: 0, tool: 0, first: 0, tts: 0, ttfa: 0 };
  function renderLatency(t, client) {
    if (t) {
      LAT.route = t.llm_route_ms || t.route_ms || 0; LAT.tool = t.tool_ms || 0; LAT.first = t.first_sentence_ms || 0;
      LAT.asr = S.source === 'voice' ? (S.asrMs || 0) : 0;
    }
    if (client) { LAT.ttfa = client.ttfa; LAT.tts = client.tts || 0; }
    const rows = [['Speech → text', LAT.asr], ['LLM picks tool', LAT.route], ['Python analytics', LAT.tool],
      ['First grounded sentence', LAT.first], ['Voice synthesis', LAT.tts]];
    const max = Math.max(1500, ...rows.map((r) => r[1]));
    $('#lat-bars').innerHTML = rows.map(([k, v]) => `<div class="lat-row"><span>${k}</span><div class="lat-bar"><div style="width:${(v / max) * 100}%"></div></div><span class="v">${v ? Math.round(v) + ' ms' : '–'}</span></div>`).join('');
    $('#lat-total').textContent = LAT.ttfa ? `TIME TO FIRST AUDIO ${(LAT.ttfa / 1000).toFixed(2)} s` : '';
  }

  function addHistory(q) {
    S.history = [q, ...S.history.filter((x) => x !== q)].slice(0, 6);
    $('#history').innerHTML = S.history.map((h) => `<li>${esc(h)}</li>`).join('');
    $$('#history li').forEach((li, i) => li.addEventListener('click', () => { interact(); ask(S.history[i], 'typed'); }));
  }

  // ------------------------------------------------------------------ developer mode
  function devReset() {
    $('#code-wrap').innerHTML = '<div class="meta code-empty">Writing code…</div>';
    $('#code-status').textContent = ''; $('#code-status').className = 'code-status';
    $('#code-actions').classList.add('hidden');
    $('#out-empty').classList.remove('hidden'); $('#out-table').innerHTML = ''; $('#out-stdout').classList.add('hidden');
    $('#out-chart-wrap').classList.add('hidden'); $('#out-chart-wrap').innerHTML = ''; $('#out-meta').textContent = '';
    S.dev = null;
  }
  function badge(text, kind) { const b = $('#code-status'); b.textContent = text; b.className = `code-status ${kind || ''}`; }
  function renderCode(ev) {
    if (!(ev.part && ev.part > 0)) $('#code-title').textContent = ev.title || 'CODE';
    if (ev.status === 'writing') {
      $('#code-wrap').innerHTML = `<pre class="code-raw">${esc(ev.code)}<span class="caret">▍</span></pre>`;
      badge('WRITING…', 'writing'); return;
    }
    if (ev.part && ev.part > 0) $('#code-wrap').insertAdjacentHTML('beforeend', `<div class="code-part-title">${esc(ev.title)}</div>${ev.html}`);
    else $('#code-wrap').innerHTML = ev.html || `<pre class="code-raw">${esc(ev.code)}</pre>`;
    const hl = $('#code-wrap .hll'); if (hl) hl.scrollIntoView({ block: 'center' });
    if (ev.status === 'source') badge('READ-ONLY SOURCE', 'source');
    else if (ev.status === 'example') badge('EXAMPLE · NOT EXECUTED', 'pending');
    else badge(ev.attempt > 1 ? 'FIX WRITTEN · NOT EXECUTED YET' : 'NOT EXECUTED', 'pending');
    if (['sql', 'python'].includes(ev.lang) && !ev.read_only) {
      S.dev = { lang: ev.lang, code: ev.code, tests: /PYTEST/.test(ev.title || '') };
      $('#code-actions').classList.remove('hidden'); $('#sandbox-note').textContent = 'Read-only DB copy · 10 s · 2 GB · no network';
    }
  }
  function renderRun(ev) {
    const pre = ev.attempt === 2 ? 'FIXED AND RE-RUN · ' : (ev.attempt === 'manual' ? 'RE-RUN · ' : '');
    if (ev.ok) badge(`${pre}EXECUTED IN SANDBOX · ${Math.round(ev.ms)} MS · EXIT 0`, 'ok');
    else if (ev.timed_out) badge(`${pre}TIMED OUT IN SANDBOX · KILLED AT 10 S`, 'bad');
    else badge(`${pre}RAN IN SANDBOX · FAILED · EXIT ${ev.exit_code}`, 'bad');
    $('#out-empty').classList.add('hidden');
    $('#out-meta').textContent = `run ${ev.run_id} · ${Math.round(ev.ms)} ms${ev.files && ev.files.length ? ' · wrote ' + ev.files.map((f) => f.name).join(', ') : ''}`;
    const cw = $('#out-chart-wrap');
    if (ev.images && ev.images.length) { cw.innerHTML = `<img alt="chart produced by the code" src="data:image/png;base64,${ev.images[0].b64}">`; cw.classList.remove('hidden'); }
    else { cw.classList.add('hidden'); cw.innerHTML = ''; }
    const t = ev.table;
    if (t && t.rows && t.rows.length) {
      // Display formatting only; the raw values are what the code produced (see stdout / files).
      const cell = (v) => {
        if (v === null || v === '' || typeof v === 'boolean') return v; const x = Number(v); if (isNaN(x)) return v;
        if (Number.isInteger(x)) return Math.abs(x) >= 10000 ? x.toLocaleString('en-US') : String(x);
        return Math.abs(x) >= 1000 ? x.toLocaleString('en-US', { maximumFractionDigits: 0 }) : String(+x.toFixed(4));
      };
      const max = ev.images && ev.images.length ? 6 : 14;
      $('#out-table').innerHTML = `<table class="grid"><tr>${t.columns.map((h) => `<th>${esc(h)}</th>`).join('')}</tr>${t.rows.slice(0, max).map((r) => `<tr>${r.map((v) => `<td>${esc(cell(v))}</td>`).join('')}</tr>`).join('')}</table>` +
        (t.rows.length > max ? `<div class="meta">${t.row_count || t.rows.length} rows · first ${max} shown</div>` : '');
    } else if (t && t.rows && ev.ok) { $('#out-table').innerHTML = '<div class="out-zero">0 rows returned</div>'; }
    else $('#out-table').innerHTML = '';
    let out = ev.stdout || '';
    if (t && t.rows && t.rows.length) out = out.split('\n').filter((l) => !/^\s*\[\s*\{/.test(l)).join('\n'); // JSON already shown as the table
    const txt = ev.ok ? out : `${ev.stdout || ''}${ev.stderr || ''}`;
    const so = $('#out-stdout');
    if (txt.trim() && !(t && t.rows && t.rows.length && ev.lang === 'sql')) { so.textContent = txt.trim().slice(-1800); so.classList.remove('hidden'); so.classList.toggle('err', !ev.ok); }
    else so.classList.add('hidden');
  }
  async function runSandbox() {
    if (!S.dev) return;
    badge('RUNNING IN SANDBOX…', 'writing');
    const r = await (await fetch('/api/dev/run', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(S.dev) })).json();
    renderRun({ ...r, attempt: 'manual' });
  }
  async function pollEndpoint() {
    if (S.persona !== 'developer') return;
    try { S.endpoint = await (await fetch('/api/dev/endpoint')).json(); renderEndpoint(); } catch (e) { /* restarting */ }
  }
  function renderEndpoint() {
    const e = S.endpoint; if (!e) return;
    const live = S.endpointLive;
    const tps = live ? `${live.tok_s.toFixed(1)} tok/s <span class="meta">measured just now</span>` :
      (e.last_tok_s ? `${e.last_tok_s.toFixed(1)} tok/s <span class="meta">last generation, ${Math.round(e.last_age_s)} s ago</span>` : '<span class="meta">no generation yet</span>');
    $('#endpoint').innerHTML = `<table class="kv ep">
      <tr><td>Base URL</td><td>${esc(e.base_url)}</td></tr>
      <tr><td>Model name</td><td>${esc(e.model)}</td></tr>
      <tr><td>Speed</td><td>${tps}</td></tr>
      <tr><td>Context</td><td>${e.n_ctx ? e.n_ctx.toLocaleString('en-US') + ' tokens per slot' : '–'}</td></tr>
      <tr><td>Load</td><td>${e.ok ? `${e.busy} of ${e.slots} slots busy` : 'model offline'}</td></tr></table>
      <pre class="curl">curl ${esc(e.base_url)}/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model": "${esc(e.model)}", "messages": [{"role": "user", "content": "Hello"}]}'</pre>
      <div class="meta ep-note">Point your editor or coding agent at this box. Standard chat-completions API. ${e.bind === '0.0.0.0' ? 'LAN access is on.' : 'Loopback only; start with ATLAS_HOST=0.0.0.0 for LAN access.'}</div>`;
  }

  // ------------------------------------------------------------------ voice in (push to talk)
  async function initMic() {
    if (S.mic) return true;
    try {
      let stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      const devs = await navigator.mediaDevices.enumerateDevices();
      const poly = devs.find((d) => d.kind === 'audioinput' && /poly/i.test(d.label));
      if (poly && !stream.getAudioTracks()[0].label.match(/poly/i)) {
        stream.getTracks().forEach((t) => t.stop());
        stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: poly.deviceId }, echoCancellation: true, noiseSuppression: true } });
      }
      ensureAudio();
      await S.audioCtx.audioWorklet.addModule('/static/pcm-worklet.js');
      const src = S.audioCtx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(S.audioCtx, 'pcm-worklet');
      src.connect(node);
      const micAn = S.audioCtx.createAnalyser(); src.connect(micAn);
      if (window.AtlasOrb) window.AtlasOrb.attachMic(micAn);
      node.port.onmessage = (e) => {
        const ws = S.asrWs;
        if (!ws || ws.readyState !== 1) return;
        if (S.listening || e.data.final) ws.send(e.data.pcm);
        if (e.data.final) ws.send(JSON.stringify({ type: 'commit' }));
      };
      S.mic = stream; S.micNode = node;
      return true;
    } catch (e) {
      $('#transcript').innerHTML = `<span class="meta">Microphone unavailable (${esc(e.name || e)}). Type a question instead.</span>`;
      return false;
    }
  }

  async function startListening() {
    if (S.listening) return;
    interact();
    if (S.state === 'speaking' || S.state === 'thinking') bargeIn();
    if (!(await initMic())) return;
    S.listening = true; S.finalText = ''; S.partialText = '';
    setState('listening');
    $('#transcript').innerHTML = '<span class="partial">Listening…</span>';
    const ws = new WebSocket(`ws://${location.host}/ws/asr`);
    ws.binaryType = 'arraybuffer';
    S.asrWs = ws;
    ws.onerror = () => { if (S.state === 'listening' || S.state === 'thinking') { S.listening = false; setState('idle'); } };
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.type === 'error') {
        S.listening = false; ws.close(); setState('idle');
        $('#transcript').innerHTML = '<span class="meta">Voice input is unavailable right now. Type your question instead.</span>';
        return;
      }
      if (d.type === 'partial') { S.partialText += d.text; $('#transcript').innerHTML = `<span>${esc(S.partialText)}</span>`; }
      else if (d.type === 'final') {
        S.asrMs = d.asr_ms; S.tAskAfterRelease = true; setTimeout(() => { S.tAskAfterRelease = false; }, 4500);
        const text = normalizeAsr(d.text || S.partialText);
        $('#transcript').innerHTML = `<span>${esc(text)}</span>`;
        ws.close();
        if (/\b(send me that|send that to me|executive brief|make (a|the|me a) brief)\b/i.test(text)) { setState('idle'); makeBrief(); }
        else if (text.trim()) ask(text, 'voice'); else setState('idle');
      }
    };
  }
  function stopListening() {
    if (!S.listening) return;
    S.listening = false;
    S.tRelease = performance.now();
    setState('thinking');
    const ws = S.asrWs;
    // Flush the worklet's partial frame; its reply carries the last samples and triggers the commit.
    const flush = () => S.micNode.port.postMessage('flush');
    setTimeout(() => { if (S.state === 'thinking' && S.asrWs === ws && ws.readyState !== 3 && !S.tAskAfterRelease) {
      const text = normalizeAsr(S.partialText || '');
      ws.close(); if (text.trim()) ask(text, 'voice'); else setState('idle'); } }, 4000);
    if (ws.readyState === 1) flush(); else ws.onopen = flush;
  }
  function normalizeAsr(t) {
    return t.replace(/\b(queue|cue|q)\s+(for|four)\b/gi, 'Q4').replace(/\bf\s?y\s?(twenty\s?seven|27)\b/gi, 'FY27')
      .replace(/\bcommits? stage\b/gi, 'Commit stage');
  }

  // ------------------------------------------------------------------ camera + person detection
  async function initCamera() {
    try {
      let stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      const devs = await navigator.mediaDevices.enumerateDevices();
      const poly = devs.find((d) => d.kind === 'videoinput' && /poly/i.test(d.label));
      if (poly && !stream.getVideoTracks()[0].label.match(/poly/i)) {
        stream.getTracks().forEach((t) => t.stop());
        stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: poly.deviceId }, width: 1280, height: 720 } });
      }
      $('#cam').srcObject = stream;
      $('#cam-empty').classList.add('hidden');
      $('#cam-label').textContent = stream.getVideoTracks()[0].label.replace(/\(.*\)/, '').trim().slice(0, 30);
      visionLoop();
    } catch (e) {
      $('#cam-empty').classList.remove('hidden');
      $('#people-count').textContent = '–';
      $('#vision-status').textContent = 'no camera';
    }
  }
  async function visionLoop() {
    const v = $('#cam'), cv = document.createElement('canvas'), ov = $('#cam-overlay');
    cv.width = 640; cv.height = 360;
    for (;;) {
      await new Promise((r) => setTimeout(r, 350));
      if (!v.videoWidth) continue;
      cv.getContext('2d').drawImage(v, 0, 0, 640, 360);
      const blob = await new Promise((r) => cv.toBlob(r, 'image/jpeg', 0.7));
      try {
        const res = await (await fetch('/api/vision', { method: 'POST', body: blob })).json();
        const people = res.people || [];
        $('#people-count').textContent = res.ok === false ? '–' : people.length;
        $('#vision-status').textContent = res.ok === false ? 'detector offline' : `${Math.round(res.ms || 0)} ms`;
        ov.width = ov.clientWidth; ov.height = ov.clientHeight;
        const ctx = ov.getContext('2d'); ctx.clearRect(0, 0, ov.width, ov.height);
        ctx.strokeStyle = BLUE; ctx.lineWidth = 3; ctx.font = 'bold 12px Arial'; ctx.fillStyle = BLUE;
        let big = false;
        people.forEach((p) => {
          const [x1, y1, x2, y2] = p.box; // normalised 0..1
          ctx.strokeRect(x1 * ov.width, y1 * ov.height, (x2 - x1) * ov.width, (y2 - y1) * ov.height);
          ctx.fillRect(x1 * ov.width, y1 * ov.height - 16, 74, 16);
          ctx.fillStyle = '#fff'; ctx.fillText(`PERSON ${Math.round(p.score * 100)}%`, x1 * ov.width + 3, y1 * ov.height - 4); ctx.fillStyle = BLUE;
          if ((x2 - x1) * (y2 - y1) > 0.12) big = true;
        });
        if (big && !S.bigPerson && S.attract) greet();
        S.bigPerson = big; S.people = people.length;
      } catch (e) { /* server restarting */ }
    }
  }

  // ------------------------------------------------------------------ attract mode
  function interact() {
    S.lastInteraction = Date.now();
    if (S.attract) { S.attract = false; $('#attract-banner').classList.add('hidden'); }
  }
  function attractNext() {
    const qs = S.cfg.golden[S.persona];
    const q = qs[S.attractIdx % qs.length]; S.attractIdx += 1;
    typeQuestion(q, () => ask(q, 'attract'));
  }
  function typeQuestion(q, done) {
    let i = 0; const el = $('#transcript');
    const tick = () => { el.innerHTML = `<span>${esc(q.slice(0, i))}</span>`; i += 2; if (i <= q.length + 1) setTimeout(tick, 28); else done(); };
    tick();
  }
  function greet() {
    bargeIn();
    S.attract = false; $('#attract-banner').classList.add('hidden');
    const org = S.cfg.personas[S.persona].org;
    const text = `Hi, I'm ${S.cfg.name}. Hold the blue button and ask me anything about ${org}.`;
    $('#question').textContent = 'Welcome';
    $('#answer').innerHTML = `<span class="s">${esc(text)}</span>`;
    fetch('/api/say', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text }) })
      .then((r) => r.ok ? r.arrayBuffer() : null).then(async (ab) => {
        if (!ab) return; ensureAudio();
        const buf = await S.audioCtx.decodeAudioData(ab); const src = S.audioCtx.createBufferSource();
        src.buffer = buf; src.connect(S.outAn); src.start(); S.playing = src; setState('speaking');
        src.onended = () => { S.playing = null; setState('idle'); };
      });
    S.lastInteraction = Date.now();
  }
  setInterval(() => {
    if (!$('#attract-toggle').checked || S.attract || S.state !== 'idle' || !S.cfg) return;
    if (Date.now() - S.lastInteraction > 60000) {
      S.attract = true; $('#attract-banner').classList.remove('hidden'); attractNext();
    }
  }, 1000);

  // ------------------------------------------------------------------ stats + model strip
  async function pollStats() {
    try {
      const s = await (await fetch('/api/stats')).json();
      const pill = $('#offline-pill');
      if (s.net.offline) { pill.className = 'pill pill-offline'; $('#offline-text').textContent = 'OFFLINE'; }
      else {
        pill.className = 'pill pill-online';
        const kind = (i) => (/^wl/.test(i) ? 'WI-FI' : /^(en|eth)/.test(i) ? 'ETHERNET' : /^ww/.test(i) ? 'CELLULAR' : i.toUpperCase());
        $('#offline-text').textContent = `ONLINE · ${[...new Set(s.net.default_route_ifaces.map(kind))].join(' + ')}`;
        pill.title = `Default route via ${s.net.default_route_ifaces.join(', ')} (live from this machine's routing table)`;
      }
      if (S.lastNet !== null && S.lastNet !== s.net.offline) {
        pill.classList.add('flash'); setTimeout(() => pill.classList.remove('flash'), 2600);
        const nf = $('#net-flash');
        nf.className = s.net.offline ? 'off' : 'on';
        $('#nf-big').textContent = s.net.offline ? 'NETWORK DISCONNECTED' : 'NETWORK CONNECTED';
        $('#nf-sub').textContent = s.net.offline ? `No route off this box. ${S.cfg.name} keeps answering, fully on device.` : 'Atlas still runs locally. Nothing it does needs the network.';
        void nf.offsetWidth; nf.classList.remove('hidden');
        clearTimeout(S.nfTimer); S.nfTimer = setTimeout(() => nf.classList.add('hidden'), 4300);
      }
      S.lastNet = s.net.offline;
      const hhmm = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const offWin = s.net.offline && s.net.tx_since_offline !== undefined;
      const sent = offWin ? s.net.tx_since_offline : s.net.tx_bytes;
      $('#cloud-bytes').textContent = bytes(sent);
      $('#cloud-since').textContent = offWin ? `since offline ${hhmm(s.net.offline_since)}` : `since ${hhmm(s.net.since)}`;
      $('#cloud-stat').classList.toggle('nonzero', sent > 0);
      $('#mem-stat').textContent = `${s.mem.atlas_gb.toFixed(1)} GB`;
      $('#gpu-stat').textContent = s.gpu.util === null || s.gpu.util === undefined ? '–' : `${Math.round(s.gpu.util)}%`;
      $('#tok-stat').textContent = s.tokens.toLocaleString('en-US');
      $('#models').innerHTML = s.models.map((m) => `<div class="model ${m.ok ? 'ok' : ''}" title="${esc(m.detail)}"><span class="dot"></span><span class="role">${m.role.toUpperCase()}</span><span>${esc(m.name)}</span>${m.ms ? `<span class="ms">${Math.round(m.ms)} ms</span>` : ''}</div>`).join('');
    } catch (e) { /* server restarting */ }
  }

  // ------------------------------------------------------------------ persona, drawer, brief, record
  function setPersona(p) {
    S.persona = p;
    document.body.classList.toggle('persona-university', p === 'university');
    document.body.classList.toggle('persona-enterprise', p === 'enterprise');
    $$('.persona-toggle button[data-persona]').forEach((b) => b.classList.toggle('on', b.dataset.persona === p));
    const per = S.cfg.personas[p];
    $('#org-name').textContent = per.org; $('#org-role').textContent = p === 'developer' ? 'Developer mode · local coding assistant' : `${per.role} briefing`;
    document.body.classList.toggle('persona-developer', p === 'developer');
    if (p === 'developer' && document.body.classList.contains('view-brain')) setView(false);
    $('#brain-btn').classList.toggle('hidden', p === 'developer' || !S.cfg.features.brain);
    if (window.AtlasBrain && S.cfg.features.brain) window.AtlasBrain.setPersona(p);
    if (p === 'developer') { devReset(); $('#code-wrap').innerHTML = '<div class="meta code-empty">Ask for SQL, a Python script, a chart, an endpoint or tests. Atlas writes it, runs it in a sandbox, and only reports what really ran.</div>'; $('#dev-question').textContent = 'Ask Atlas to write and run code against the Meridian data.'; pollEndpoint(); }
    $('#question').textContent = `Ask ${S.cfg.name} about ${per.org}.`;
    $('#headline').textContent = 'Ask me anything.'; $('#headline').className = 'headline idle';
    $('#headline-label').textContent = 'READY · FULLY OFFLINE';
    $('#headline-sub').textContent = p === 'enterprise' ? 'Forecasts, deal risk, site decisions, hiring: answered on this box in about a second.'
      : 'Research funding, sponsor risk, new sites, enrollment: answered on this box in about a second.';
    $('#pctl').classList.add('hidden');
    $('#flow').classList.add('hidden');
    $('#answer').innerHTML = ''; $('#grounded').textContent = '';
    $('#how-idle').classList.remove('hidden'); $('#how-live').classList.add('hidden');
    if (S.chart) { S.chart.destroy(); S.chart = null; }
    $('#chart-title').textContent = 'TRY ASKING';
    $('#table-wrap').classList.add('hidden');
    $('#chart-empty').classList.remove('hidden');
    $('#s1-items').innerHTML = '<div class="meta">Typed judgments with calibrated probabilities appear here.</div>'; $('#s1-ms').textContent = '';
    const picks = p === 'enterprise' ? [0, 1, 2, 3] : [0, 1, 2, 3];
    $('#suggest').innerHTML = picks.map((i) => S.cfg.golden[p][i]).map((q) => `<button data-q="${esc(q)}"><div>${esc(q)}</div><span>TAP OR SAY IT</span></button>`).join('');
    $$('#suggest button').forEach((b) => b.addEventListener('click', () => { interact(); $('#transcript').innerHTML = `<span>${esc(b.dataset.q)}</span>`; ask(b.dataset.q, 'golden'); }));
    renderGolden();
  }
  function renderGolden(tab) {
    const g = S.cfg.golden; let n = 0;
    tab = tab || S.goldenTab || (g[S.persona] ? S.persona : 'enterprise'); S.goldenTab = tab;
    $$('.golden-tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
    const titles = { enterprise: 'ENTERPRISE · MERIDIAN DYNAMICS', university: 'UNIVERSITY · ALTA VISTA UNIVERSITY', developer: 'DEVELOPER · LOCAL CODING ASSISTANT' };
    const grp = (p, title) => `<div class="label grp">${title}</div>` + (g[p] || []).map((q) => `<button data-p="${p}" data-q="${esc(q)}"><b>${++n}</b>${esc(q)}</button>`).join('');
    $('#golden-list').innerHTML = grp(tab, titles[tab]);
    $$('#golden-list button').forEach((b) => b.addEventListener('click', () => {
      interact(); $('#golden-drawer').classList.add('hidden');
      if (b.dataset.p !== S.persona) setPersona(b.dataset.p);
      $('#transcript').innerHTML = `<span>${esc(b.dataset.q)}</span>`;
      ask(b.dataset.q, 'golden');
    }));
  }
  async function makeBrief() {
    interact();
    const btn = $('#brief-btn'); btn.textContent = 'GENERATING…';
    try {
      const r = await fetch('/api/brief', { method: 'POST' });
      const d = await r.json();
      if (!r.ok) { btn.textContent = 'ASK A QUESTION FIRST'; setTimeout(() => { btn.textContent = 'EXECUTIVE BRIEF'; }, 1800); return; }
      $('#brief-img').src = `${d.png}?t=${Date.now()}`; $('#brief-png').href = d.png; $('#brief-pdf').href = d.pdf;
      $('#brief-modal').classList.remove('hidden');
    } finally { if (btn.textContent === 'GENERATING…') btn.textContent = 'EXECUTIVE BRIEF'; }
  }
  async function feedBrain() {
    const btn = $('#feed-btn'), st = $('#feed-status');
    btn.disabled = true; const t0 = performance.now();
    const tick = setInterval(() => { st.innerHTML = `Writing a new synthetic document with the local LLM… <b>${((performance.now() - t0) / 1000).toFixed(1)} s</b>`; }, 100);
    let d;
    try { d = await (await fetch(`/api/brain/feed?persona=${S.persona}`, { method: 'POST' })).json(); } catch (e) { d = { ok: false, error: 'server unavailable' }; }
    clearInterval(tick); btn.disabled = false;
    if (!d.ok) { st.innerHTML = `<span class="bad">${esc(d.error || 'failed')}</span>`; return; }
    if (window.AtlasBrain) window.AtlasBrain.grow(d.new_nodes);
    const steps = d.steps.map((x) => `<tr><td>${esc(x.step)}</td><td>${x.ms >= 1000 ? (x.ms / 1000).toFixed(1) + ' s' : x.ms.toFixed(x.ms < 10 ? 1 : 0) + ' ms'}</td></tr><tr class="sd"><td colspan="2">${esc(x.detail)}</td></tr>`).join('');
    st.innerHTML = `Ingested “${esc(d.doc.title)}” in <b>${Math.round(d.ingest_ms)} ms</b>. Details on the brain.`;
    const card = $('#brain-grow'); card.classList.remove('hidden');
    card.innerHTML = `<button class="btn-ghost bg-close" aria-label="close">CLOSE</button><div class="feed-title">FED THE BRAIN · “${esc(d.doc.title)}” · SYNTHETIC</div>
      <div class="feed-big">Ingest ${Math.round(d.ingest_ms)} ms <span>chunk + embed + index + graph, measured</span></div>
      <table class="kv feed-steps">${steps}</table>
      <div class="feed-proof"><div><b>Before:</b> “${esc(d.question)}” matched ${esc(d.before.title)} (${d.before.score.toFixed(2)})</div>
      <div><b>After:</b> it matches ${esc(d.after.title)} (${d.after.score.toFixed(2)}) ${d.proven ? '<span class="ok">✓ only answerable after ingestion</span>' : '<span class="bad">not proven</span>'}</div></div>
      <button id="feed-ask" class="btn-secondary">ASK IT: ${esc(d.question)}</button>`;
    card.querySelector('.bg-close').addEventListener('click', () => card.classList.add('hidden'));
    $('#feed-ask').addEventListener('click', () => { interact(); card.classList.add('hidden'); $('#transcript').innerHTML = `<span>${esc(d.question)}</span>`; ask(d.question, 'golden'); });
  }
  function setView(brainOn) {
    document.body.classList.toggle('view-brain', brainOn);
    $('#brain-btn').classList.toggle('on', brainOn);
    $('#brain-btn').textContent = brainOn ? 'ANSWER VIEW' : 'SECOND BRAIN';
    if (window.AtlasBrain) window.AtlasBrain.show(brainOn);
    if (!brainOn && S.chart) setTimeout(() => S.chart.resize(), 50);
  }
  function setRecord(on, layout) {
    document.body.classList.toggle('record', on);
    document.body.classList.toggle('v916', on && layout === '9x16');
    $('#record-bar').classList.toggle('hidden', !on);
    $$('#record-bar [data-layout]').forEach((b) => b.classList.toggle('on', b.dataset.layout === (layout || '16x9')));
    fit();
    if (S.chart) { S.chart.resize(1, 1); setTimeout(() => S.chart.resize(), 60); }
  }

  // ------------------------------------------------------------------ wiring
  async function init() {
    fit();
    S.cfg = await (await fetch('/api/config')).json();
    $('#wordmark').textContent = S.cfg.name.toUpperCase();
    $('#says-label').textContent = `${S.cfg.name.toUpperCase()} SAYS`;
    document.title = `${S.cfg.name} · Executive AI on HP ZGX Nano`;
    setPersona('enterprise');
    connect();
    renderLatency(null);
    pollStats(); setInterval(pollStats, 1000);
    initCamera();

    if (window.AtlasOrb && S.cfg.features.orb && !/[?&]orb=0/.test(location.search)) {
      S.orbRenderer = window.AtlasOrb.init($('#orb'), $('#orb-glow'));
      const hit = $('#orb-hit');
      hit.addEventListener('pointerdown', (e) => { e.preventDefault(); hit.setPointerCapture(e.pointerId); if (S.state === 'speaking') { bargeIn(); return; } startListening(); });
      hit.addEventListener('pointerup', () => stopListening());
      hit.addEventListener('pointercancel', () => stopListening());
      hit.addEventListener('contextmenu', (e) => e.preventDefault());
    } else { document.body.classList.add('no-orb'); }
    if (window.AtlasBrain && S.cfg.features.brain) {
      window.AtlasBrain.init(); window.AtlasBrain.setPersona(S.persona);
      $('#brain-btn').addEventListener('click', () => { interact(); setView(!document.body.classList.contains('view-brain')); });
      if (S.cfg.features.feed && S.cfg.features.docs) $('#feed-btn').addEventListener('click', () => { interact(); feedBrain(); });
      else $('#feed-btn').classList.add('hidden');
      $('#brain-reset').addEventListener('click', async () => { await fetch(`/api/brain/feed/reset?persona=${S.persona}`, { method: 'POST' }); $('#feed-status').textContent = 'Fed documents removed; index rebuilt from the base corpus.'; window.AtlasBrain.reload(); });
    } else $('#brain-btn').classList.add('hidden');
    const setStage = (on) => {
      document.body.classList.toggle('stage', on); $('#stage-btn').classList.toggle('on', on);
      if (window.AtlasOrb) window.AtlasOrb.setDark(on);
      try { localStorage.setItem('atlas-stage', on ? '1' : '0'); } catch (e) { /* private mode */ }
      if (S.lastResult && S.lastResult.chart) renderChart(S.lastResult.chart, S.lastResult.table);
    };
    $('#stage-btn').addEventListener('click', () => setStage(!document.body.classList.contains('stage')));
    try { if (localStorage.getItem('atlas-stage') === '1' || /stage=1/.test(location.search)) setStage(true); } catch (e) { /* ignore */ }
    S.setStage = setStage;
    const ptt = $('#ptt');
    ptt.addEventListener('pointerdown', (e) => { e.preventDefault(); if (S.state === 'speaking') { bargeIn(); return; } startListening(); });
    ptt.addEventListener('pointerup', () => stopListening());
    ptt.addEventListener('contextmenu', (e) => e.preventDefault());
    ptt.style.touchAction = 'none';
    ptt.addEventListener('pointerleave', () => stopListening());
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'Space' && !e.repeat) { e.preventDefault(); if (S.state === 'speaking') { bargeIn(); return; } startListening(); }
      if (e.key === 'r' || e.key === 'R') setRecord(!document.body.classList.contains('record'), '16x9');
      if (e.key === 'g' || e.key === 'G') $('#golden-drawer').classList.toggle('hidden');
      if (e.key === 'd' || e.key === 'D') S.setStage(!document.body.classList.contains('stage'));
      if ((e.key === 'b' || e.key === 'B') && S.cfg.features.brain && S.persona !== 'developer') setView(!document.body.classList.contains('view-brain'));
      if (e.key === 'Escape') { $('#golden-drawer').classList.add('hidden'); $('#brief-modal').classList.add('hidden'); }
      interact();
    });
    document.addEventListener('keyup', (e) => { if (e.code === 'Space' && e.target.tagName !== 'INPUT') { e.preventDefault(); stopListening(); } });
    document.addEventListener('pointerdown', () => { ensureAudio(); });
    $('#ask-form').addEventListener('submit', (e) => { e.preventDefault(); interact(); const v = $('#ask-input').value; $('#ask-input').value = ''; $('#transcript').innerHTML = `<span>${esc(v)}</span>`; ask(v, 'typed'); });
    $$('.persona-toggle button[data-persona]').forEach((b) => b.addEventListener('click', () => { interact(); setPersona(b.dataset.persona); }));
    $$('.lang-toggle button').forEach((b) => b.addEventListener('click', () => {
      interact(); S.lang = b.dataset.lang;
      $$('.lang-toggle button').forEach((x) => x.classList.toggle('on', x === b));
    }));
    $('#golden-btn').addEventListener('click', () => { interact(); S.goldenTab = null; renderGolden(); $('#golden-drawer').classList.toggle('hidden'); });
    $('#golden-close').addEventListener('click', () => $('#golden-drawer').classList.add('hidden'));
    $$('.golden-tabs button').forEach((b) => b.addEventListener('click', () => renderGolden(b.dataset.tab)));
    $('#run-btn').addEventListener('click', () => { interact(); runSandbox(); });
    setInterval(pollEndpoint, 2000);
    if (!S.cfg.features.developer) $$('[data-persona="developer"], .golden-tabs [data-tab="developer"]').forEach((b) => b.remove());
    $('#voice-toggle').addEventListener('change', (e) => { S.voice = e.target.checked; });
    $('#net-reset').addEventListener('click', () => fetch('/api/net/reset', { method: 'POST' }));
    $('#brief-btn').addEventListener('click', makeBrief);
    $('#brief-close').addEventListener('click', () => $('#brief-modal').classList.add('hidden'));
    $('#record-btn').addEventListener('click', () => setRecord(true, '16x9'));
    $$('#record-bar [data-layout]').forEach((b) => b.addEventListener('click', () => setRecord(true, b.dataset.layout)));
    $('#record-exit').addEventListener('click', () => setRecord(false));
    // Test hooks for automated checks
    window.atlas = { ask, setPersona, setRecord, setView, feedBrain, state: S, pollStats, greet, setStage: (on) => S.setStage(on),
      orbFps: () => (window.AtlasOrb ? window.AtlasOrb.fps() : null), orb: () => window.AtlasOrb };
  }
  init();
})();
