/* Atlas second brain: the knowledge graph Atlas is connected to, drawn on a canvas, with real provenance.
   When an answer arrives, the exact nodes its tool drew on light up and a pulse travels from the orb into
   the brain. Every count shown comes from /api/brain/* (computed live from the data). */
(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const COLORS = {
    region: '#004BDF', quarter: '#002F8C', center: '#0060FF', industry: '#2C6BE8', segment: '#4D86FF',
    rep: '#1F5FD6', account: '#5C8FF0', deal: '#9DB9F2',
    college: '#004BDF', sponsor: '#002F8C', fy: '#2C6BE8', site: '#0060FF', cohort: '#4D86FF',
    pi: '#1F5FD6', award: '#86A9F0', proposal: '#B3C8F5',
    document: '#0E8A3B', passage: '#5BBF7E',
  };
  const B = { persona: null, g: null, canvas: null, ctx: null, on: false, raf: 0, last: 0, lit: new Map(), litAt: 0,
    pulses: [], pending: null, hover: null, t0: performance.now(), pos: null, knowsTimer: 0, growing: [] };

  function money(x) {
    const a = Math.abs(x);
    if (a >= 1e9) return `$${(x / 1e9).toFixed(2)}B`; if (a >= 1e6) return `$${(x / 1e6).toFixed(1)}M`;
    if (a >= 1e3) return `$${(x / 1e3).toFixed(0)}K`; return `$${Math.round(x)}`;
  }
  const fmtVal = (n) => (n.fmt === 'num' ? `${Math.round(n.value).toLocaleString('en-US')}` : money(n.value));
  const isDark = () => document.body.classList.contains('stage');

  async function load(persona) {
    B.persona = persona;
    try {
      const r = await fetch(`/api/brain/graph?persona=${persona}`);
      if (!r.ok) { B.g = null; return; }
      B.g = await r.json();
    } catch (e) { B.g = null; return; }
    B.lit.clear(); B.litE = []; B.layerKey = null;
    B.pos = new Float32Array(B.g.nodes.length * 2);
    legend();
  }

  function legend() {
    if (!B.g) return;
    const c = {}; B.g.nodes.forEach((n) => { c[n.type] = (c[n.type] || 0) + 1; });
    $('#brain-legend').innerHTML = B.g.types.filter((t) => c[t.type]).map((t) =>
      `<span><i style="background:${COLORS[t.type]}"></i>${esc(t.label)} <b>${c[t.type].toLocaleString('en-US')}</b></span>`).join('');
    $('#brain-meta').textContent = `${B.g.nodes.length.toLocaleString('en-US')} nodes · ${B.g.edges.length.toLocaleString('en-US')} edges · built from the live data`;
  }

  function resize() {
    const c = B.canvas; const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
    if (w && h && (c.width !== w || c.height !== h)) { c.width = w; c.height = h; }
    B.dpr = dpr;
  }

  // The whole graph is drawn once into a cached layer (it only changes with size, theme, provenance or new nodes);
  // each frame blits that layer with a slow drift and draws only the lit / growing nodes on top. This keeps the
  // brain cheap enough to run beside the orb and the models on the same box.
  function ensureLayer(W, H, dpr, dark, anyLit) {
    const key = `${W}x${H}|${dark}|${anyLit}|${B.litVersion || 0}|${B.g.nodes.length}|${B.g.edges.length}|${B.g.built_at}`;
    if (B.layerKey === key) return;
    B.layerKey = key;
    const N = B.g.nodes, m = 26 * dpr;
    B.base = new Float32Array(N.length * 2);
    for (let i = 0; i < N.length; i++) { B.base[i * 2] = m + N[i].x * (W - 2 * m); B.base[i * 2 + 1] = m + N[i].y * (H - 2 * m); }
    const L = B.layer || (B.layer = document.createElement('canvas'));
    L.width = W; L.height = H;
    const c = L.getContext('2d'), P = B.base;
    c.clearRect(0, 0, W, H);
    c.lineWidth = 0.6 * dpr;
    c.strokeStyle = dark ? 'rgba(110,155,255,0.10)' : (anyLit ? 'rgba(0,75,223,0.05)' : 'rgba(0,75,223,0.09)');
    c.beginPath();
    for (const [a, b] of B.g.edges) { c.moveTo(P[a * 2], P[a * 2 + 1]); c.lineTo(P[b * 2], P[b * 2 + 1]); }
    c.stroke();
    for (let i = 0; i < N.length; i++) {
      c.globalAlpha = anyLit ? 0.28 : 0.85;
      c.fillStyle = COLORS[N[i].type] || '#5C8FF0';
      c.beginPath(); c.arc(P[i * 2], P[i * 2 + 1], Math.max(N[i].r * dpr * 0.9, 0.8 * dpr), 0, 6.283); c.fill();
    }
    c.font = `bold ${11 * dpr}px Arial`; c.textAlign = 'center'; c.fillStyle = dark ? '#DCE6FF' : '#0B1E4A';
    c.globalAlpha = anyLit ? 0.35 : 0.95;
    for (let i = 0; i < N.length; i++) {
      const nd = N[i];
      if (!['region', 'center', 'industry', 'segment', 'college', 'sponsor', 'site'].includes(nd.type) || B.lit.has(i)) continue;
      c.fillText(nd.label.length > 28 ? nd.label.slice(0, 27) + '…' : nd.label, P[i * 2], P[i * 2 + 1] - nd.r * dpr - 5 * dpr);
    }
    c.globalAlpha = 1;
  }

  function haloSprite(dark) {
    const key = dark ? 'd' : 'l';
    if (B.sprite && B.spriteKey === key) return B.sprite;
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const x = c.getContext('2d'), gr = x.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, dark ? 'rgba(120,170,255,0.55)' : 'rgba(0,75,223,0.32)'); gr.addColorStop(1, 'rgba(0,75,223,0)');
    x.fillStyle = gr; x.fillRect(0, 0, 64, 64);
    B.sprite = c; B.spriteKey = key; return c;
  }

  function ensureLitLayer(W, H, dpr, dark) {
    const key = `${B.layerKey}|lit`;
    if (B.litKey === key) return;
    B.litKey = key;
    const L = B.litLayer || (B.litLayer = document.createElement('canvas'));
    L.width = W; L.height = H;
    const c = L.getContext('2d'), P = B.base, N = B.g.nodes;
    c.clearRect(0, 0, W, H);
    for (const [w, al] of [[4.5, 0.10], [1.3, 0.75]]) {
      c.lineWidth = w * dpr; c.strokeStyle = dark ? `rgba(140,180,255,${al})` : `rgba(0,75,223,${al})`;
      c.beginPath();
      for (let i = 0; i < (B.litE || []).length; i += 2) { const a = B.litE[i], b = B.litE[i + 1]; c.moveTo(P[a * 2], P[a * 2 + 1]); c.lineTo(P[b * 2], P[b * 2 + 1]); }
      c.stroke();
    }
    c.fillStyle = dark ? '#8FB3FF' : '#004BDF';
    for (const i of B.lit.keys()) { c.beginPath(); c.arc(P[i * 2], P[i * 2 + 1], Math.max(N[i].r * dpr * 1.35, 0.8 * dpr), 0, 6.283); c.fill(); }
  }

  function place(t) {
    const W = B.canvas.width, H = B.canvas.height, dpr = B.dpr;
    ensureLayer(W, H, dpr, isDark(), B.lit.size > 0);
    const drift = window.AtlasOrb && window.AtlasOrb.reduced ? 0 : 1;
    B.ox = drift * Math.sin(t * 0.13) * 4 * dpr; B.oy = drift * Math.cos(t * 0.11) * 4 * dpr;
    for (let i = 0; i < B.base.length; i += 2) { B.pos[i] = B.base[i] + B.ox; B.pos[i + 1] = B.base[i + 1] + B.oy; }
  }

  function frame(now) {
    B.raf = requestAnimationFrame(frame);
    if (now - B.last < 1000 / 30 - 2) return; // the brain drifts slowly: 30 fps is plenty
    B.last = now;
    if (!B.g) return;
    resize();
    const t = (now - B.t0) / 1000, ctx = B.ctx, W = B.canvas.width, H = B.canvas.height, dpr = B.dpr;
    if (B.pos.length !== B.g.nodes.length * 2) B.pos = new Float32Array(B.g.nodes.length * 2);
    place(t);
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(B.layer, B.ox, B.oy);
    const dark = isDark(), N = B.g.nodes, P = B.pos;
    // lit state: one cached layer (glowing edges + bright nodes), faded in, plus halo sprites for the largest nodes
    if (B.lit.size) {
      const k = Math.min(1, Math.max(0, (now - B.litAt) / 600));
      if (k > 0) {
        ensureLitLayer(W, H, dpr, dark);
        ctx.globalAlpha = k; ctx.drawImage(B.litLayer, B.ox, B.oy);
        const sprite = haloSprite(dark);
        for (const i of B.haloSet || []) {
          const h = N[i].r * dpr * 1.35 * (3 + 0.6 * Math.sin(t * 3 + i));
          ctx.drawImage(sprite, P[i * 2] - h, P[i * 2 + 1] - h, 2 * h, 2 * h);
        }
        ctx.globalAlpha = 1;
      }
    }
    const draw = (i, r, alpha, color) => { ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.beginPath(); ctx.arc(P[i * 2], P[i * 2 + 1], Math.max(r, 0.8 * dpr), 0, 6.283); ctx.fill(); };
    for (const g of B.growing) {
      const k = Math.min(1, Math.max(0, (now - g.t) / 900)); if (k <= 0) continue;
      draw(g.i, N[g.i].r * dpr * 0.9 * (0.2 + 0.8 * k + 0.6 * Math.sin(Math.PI * k)), 1, COLORS[N[g.i].type] || '#0E8A3B');
    }
    // labels for lit hubs, documents, sites and the hovered node
    ctx.font = `bold ${11 * dpr}px Arial`; ctx.textAlign = 'center'; ctx.fillStyle = dark ? '#DCE6FF' : '#0B1E4A'; ctx.globalAlpha = 0.95;
    const label = (i) => { const nd = N[i]; ctx.fillText(nd.label.length > 28 ? nd.label.slice(0, 27) + '…' : nd.label, P[i * 2], P[i * 2 + 1] - nd.r * dpr - 5 * dpr); };
    for (const [i] of B.lit) {
      if (now < B.litAt) continue;
      if (['region', 'center', 'industry', 'segment', 'college', 'sponsor', 'site', 'document'].includes(N[i].type)) label(i);
    }
    if (B.hover !== null && B.hover !== undefined && N[B.hover]) label(B.hover);
    ctx.globalAlpha = 1;
    pulses(now);
  }

  // ---- pulse: dots travel from the orb into the brain, then the nodes light up
  function frameRect(el) {
    const fr = document.getElementById('frame').getBoundingClientRect(); const s = fr.width / document.getElementById('frame').offsetWidth;
    const r = el.getBoundingClientRect();
    return { x: (r.left - fr.left) / s, y: (r.top - fr.top) / s, w: r.width / s, h: r.height / s };
  }
  function launch(targets) {
    const pl = document.getElementById('pulse-layer'); if (!pl) return;
    pl.width = 1920; pl.height = 1080;
    const o = frameRect(document.getElementById('orb')); const c = frameRect(B.canvas);
    const ox = o.x + o.w / 2, oy = o.y + o.h / 2;
    const now = performance.now();
    targets.forEach((i, k) => {
      const tx = c.x + B.pos[i * 2] / B.dpr, ty = c.y + B.pos[i * 2 + 1] / B.dpr;
      B.pulses.push({ ox, oy, tx, ty, t: now + (k % 12) * 35, d: 850 + (k % 7) * 40 });
    });
  }
  function pulses(now) {
    const pl = document.getElementById('pulse-layer'); if (!pl || !B.pulses.length) { if (pl && B.hadPulse) { pl.getContext('2d').clearRect(0, 0, pl.width, pl.height); B.hadPulse = false; } return; }
    B.hadPulse = true;
    const ctx = pl.getContext('2d'); ctx.clearRect(0, 0, pl.width, pl.height);
    const dark = isDark();
    B.pulses = B.pulses.filter((p) => now < p.t + p.d + 50);
    for (const p of B.pulses) {
      const k = Math.min(1, Math.max(0, (now - p.t) / p.d)); if (k <= 0) continue;
      const e = 1 - Math.pow(1 - k, 3);
      const mx = (p.ox + p.tx) / 2, my = Math.min(p.oy, p.ty) - 120;
      const bx = (1 - e) * (1 - e) * p.ox + 2 * (1 - e) * e * mx + e * e * p.tx;
      const by = (1 - e) * (1 - e) * p.oy + 2 * (1 - e) * e * my + e * e * p.ty;
      const g = ctx.createRadialGradient(bx, by, 0, bx, by, 14);
      g.addColorStop(0, dark ? 'rgba(170,200,255,0.95)' : 'rgba(0,75,223,0.9)'); g.addColorStop(1, 'rgba(0,75,223,0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(bx, by, 14, 0, 6.283); ctx.fill();
    }
  }

  function applyProvenance(pr) {
    if (!B.g || !pr || !pr.nodes) return;
    const idx = new Map(B.g.nodes.map((n, i) => [n.id, i]));
    const ids = pr.nodes.map((id) => idx.get(id)).filter((i) => i !== undefined);
    B.lit.clear();
    const now = performance.now();
    // Hubs and the biggest rows get a visible pulse; every node used still lights up.
    const sorted = ids.slice().sort((a, b) => B.g.nodes[b].r - B.g.nodes[a].r);
    const targets = sorted.slice(0, 36);
    const visible = B.on && !document.hidden;
    if (visible) { resize(); B.pos = new Float32Array(B.g.nodes.length * 2); place((now - B.t0) / 1000); launch(targets); }
    ids.forEach((i, k) => B.lit.set(i, now + (visible ? 850 + (k % 40) * 8 : 0)));
    B.litAt = now + (visible ? 850 : 0);
    B.haloSet = new Set(sorted.slice(0, 120));
    const L = new Set(ids); B.litE = [];
    for (const [x, y] of B.g.edges) if (L.has(x) && L.has(y)) B.litE.push(x, y);
    B.litVersion = (B.litVersion || 0) + 1;
    $('#brain-title').textContent = `SECOND BRAIN · ${ids.length.toLocaleString('en-US')} NODES USED`;
    const what = pr.tool === 'search_knowledge' ? `cited ${ids.filter((i) => B.g.nodes[i].type === 'passage').length} passages` : `drew on ${pr.rows.toLocaleString('en-US')} rows`;
    $('#brain-meta').innerHTML = `<b>${esc(pr.tool || '')}</b> ${what} · ${ids.length.toLocaleString('en-US')} of ${B.g.nodes.length.toLocaleString('en-US')} nodes lit`;
  }

  // ---- click a node: show its real record
  async function click(ev) {
    if (!B.g) return;
    const r = B.canvas.getBoundingClientRect(); const sx = B.canvas.width / r.width;
    const x = (ev.clientX - r.left) * sx, y = (ev.clientY - r.top) * sx;
    let best = -1, bd = 1e9;
    for (let i = 0; i < B.g.nodes.length; i++) {
      const dx = B.pos[i * 2] - x, dy = B.pos[i * 2 + 1] - y, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    const card = $('#brain-card');
    if (best < 0 || Math.sqrt(bd) > 14 * B.dpr) { card.classList.add('hidden'); B.hover = null; return; }
    const nd = B.g.nodes[best]; B.hover = best;
    card.classList.remove('hidden');
    card.innerHTML = `<div class="label-sm">${esc(nd.type)} · ${esc(nd.id)}</div><div class="bc-title">${esc(nd.label)}</div><div class="meta">loading record…</div>`;
    try {
      const rec = await (await fetch(`/api/brain/node/${encodeURIComponent(nd.id)}`)).json();
      const rows = rec.rows || [];
      const body = rec.text ? `<div class="bc-text">${esc(rec.text)}</div>` :
        rows.slice(0, 4).map((row) => `<table class="kv">${Object.entries(row).slice(0, 12).map(([k, v]) =>
          `<tr><td>${esc(k)}</td><td>${esc(typeof v === 'number' && Math.abs(v) >= 1000 ? v.toLocaleString('en-US') : v)}</td></tr>`).join('')}</table>`).join('<hr>');
      card.innerHTML = `<div class="label-sm">${esc(nd.type)} · ${esc(nd.id)}</div><div class="bc-title">${esc(nd.label)}</div>
        <div class="meta">Real record from ${esc(rec.table || 'the index')}${rec.source ? ' · ' + esc(rec.source) : ''}</div>${body}`;
    } catch (e) { card.innerHTML += '<div class="meta">record unavailable</div>'; }
  }

  // ---- HOW ATLAS KNOWS cards (all live)
  async function knows() {
    if (!B.persona) return;
    let k; try { k = await (await fetch(`/api/brain/knows?persona=${B.persona}`)).json(); } catch (e) { return; }
    const org = B.persona === 'enterprise' ? 'MERIDIAN' : 'ALTA VISTA';
    $('#knows-title').textContent = `HOW ATLAS KNOWS ${org}`;
    const n = (x) => Number(x).toLocaleString('en-US');
    const bytes = (b) => (b < 1024 ? `${b} bytes` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`);
    const top = (o, m) => Object.entries(o).slice(0, m).map(([a, b]) => `${esc(a)} ${n(b)}`).join(' · ');
    const d = k.docs || {};
    const net = k.net || {}; const sent = net.offline && net.tx_since_offline !== undefined ? net.tx_since_offline : net.tx_bytes;
    const ago = (ts) => { const s = Math.max(0, Math.round(k.now - ts)); return s < 90 ? `${s} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`; };
    const tm = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    $('#knows').innerHTML = `
      <div class="kcard"><div class="label-sm">DATA CONNECTED</div><div class="kbig">${n(k.data.rows)} <span>rows</span></div>
        <div class="ksub">${k.data.tables} tables · ${esc(k.data.from)} → ${esc(k.data.to)}</div><div class="ksmall">${top(k.data.by_table, 4)}</div></div>
      <div class="kcard"><div class="label-sm">KNOWLEDGE GRAPH</div><div class="kbig">${n(k.graph.nodes)} <span>nodes</span> · ${n(k.graph.edges)} <span>edges</span></div>
        <div class="ksmall">${top(k.graph.nodes_by_type, 9)}</div></div>
      <div class="kcard"><div class="label-sm">DOCUMENTS INDEXED</div>${d.docs !== undefined ? `<div class="kbig">${n(d.docs)} <span>docs</span> · ${n(d.passages)} <span>passages</span></div>
        <div class="ksub">${n(d.vectors)} vectors · ${esc(d.dim)}-dim · ${esc(d.model)}</div><div class="ksmall">all labelled synthetic · grounded in cited passages</div>` : `<div class="ksub">${esc(d.error || 'not indexed yet')}</div>`}</div>
      <div class="kcard"><div class="label-sm">DECISION MODEL</div><div class="ksub">${esc(k.decision.base)}${k.decision.ok ? '' : ' · offline'}</div>
        <div class="ksmall">${esc(k.decision.calibration)}</div><div class="kft">${esc(k.decision.finetune_text)}</div></div>
      <div class="kcard"><div class="label-sm">WHERE IT LIVES</div><div class="ksub">On this HP ZGX Nano · database ${bytes(k.data.db_bytes)}</div>
        <div class="ksmall">Sent to cloud: <b>${bytes(sent || 0)}</b> ${net.offline ? '(offline)' : '(all machine traffic, live counter)'}</div></div>
      <div class="kcard"><div class="label-sm">LAST REFRESHED</div><div class="ksub">Graph ${tm(k.graph.built_at)} (${ago(k.graph.built_at)})${d.updated_at ? ` · index ${tm(d.updated_at)} (${ago(d.updated_at)})` : ''}</div></div>`;
  }

  function start() { if (!B.raf) { B.last = 0; B.raf = requestAnimationFrame(frame); } }
  function stop() { if (B.raf) cancelAnimationFrame(B.raf); B.raf = 0; }

  window.AtlasBrain = {
    init() {
      B.canvas = $('#brain'); B.ctx = B.canvas.getContext('2d');
      B.canvas.addEventListener('click', click);
      document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else if (B.on) start(); });
    },
    async setPersona(p) {
      if (p !== 'enterprise' && p !== 'university') { B.persona = null; return; }
      if (p !== B.persona || !B.g) await load(p);
      if (B.on) knows();
    },
    async show(on) {
      B.on = on;
      if (on) {
        start(); knows(); clearInterval(B.knowsTimer); B.knowsTimer = setInterval(knows, 4000);
        if (B.pending) { const p = B.pending; B.pending = null; setTimeout(() => applyProvenance(p), 350); }
      } else { stop(); clearInterval(B.knowsTimer); }
    },
    provenance(pr) { if (B.on) applyProvenance(pr); else { B.pending = pr; applyProvenance(pr); } },
    async reload() { const lit = B.lit; await load(B.persona); knows(); return lit; },
    async grow(newIds) {
      await load(B.persona);
      const idx = new Map(B.g.nodes.map((n, i) => [n.id, i])); const now = performance.now();
      B.growing = newIds.map((id, k) => ({ i: idx.get(id), t: now + k * 120 })).filter((g) => g.i !== undefined);
      resize(); B.pos = new Float32Array(B.g.nodes.length * 2); place((now - B.t0) / 1000);
      launch(B.growing.map((g) => g.i));
      knows();
      setTimeout(() => { B.growing = []; }, 900 + newIds.length * 120 + 200);
    },
    knows,
    state: B,
  };
})();
