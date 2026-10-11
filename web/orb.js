/* Atlas voice orb. A particle sphere driven by real audio: an AnalyserNode on the TTS output while speaking,
   the live mic level while listening. No canned animation: with no signal the orb only breathes.
   Raw WebGL (gl.POINTS); falls back to Canvas 2D. Capped at 60 fps, paused when the tab is hidden,
   honours prefers-reduced-motion. */
(() => {
  'use strict';
  const NS = 2600, NR = 900, N = NS + NR, BANDS = 8;
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const O = {
    canvas: null, gl: null, ctx2d: null, prog: null, buf: null, data: null, dark: false,
    state: 'idle', w: { idle: 1, listening: 0, thinking: 0, speaking: 0 }, stateAt: 0,
    out: null, mic: null, outBins: null, micBins: null, td: null,
    level: 0, micLevel: 0, bands: new Float32Array(BANDS), t0: performance.now(), last: 0, raf: 0,
    frames: [], dirs: null, seeds: null, bandOf: null, pulse: 0, enabled: true, stateLat: [], stateShown: true,
  };

  // Fibonacci sphere: evenly spread unit directions.
  function makeParticles() {
    O.dirs = new Float32Array(N * 3); O.seeds = new Float32Array(N); O.bandOf = new Uint8Array(N);
    const ga = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < NS; i++) {
      const y = 1 - (i / (NS - 1)) * 2, r = Math.sqrt(1 - y * y), th = ga * i;
      O.dirs[i * 3] = Math.cos(th) * r; O.dirs[i * 3 + 1] = y; O.dirs[i * 3 + 2] = Math.sin(th) * r;
      O.seeds[i] = Math.random();
      // Map latitude to a frequency band so low frequencies move the equator, highs the poles.
      O.bandOf[i] = Math.min(BANDS - 1, Math.floor((1 - Math.abs(y)) * BANDS * 0.999));
    }
    for (let i = NS; i < N; i++) { O.seeds[i] = Math.random(); O.bandOf[i] = Math.floor(Math.random() * BANDS); }
    O.data = new Float32Array(N * 4); // x, y, size, tint(0..1) packed with alpha in the sign-free fraction
  }

  const VS = `
    attribute vec4 a; uniform float uDpr; uniform vec2 uScale; varying float vT; varying float vA;
    void main() {
      gl_Position = vec4(a.x * uScale.x, a.y * uScale.y, 0.0, 1.0);
      gl_PointSize = a.z * uDpr;
      vT = fract(a.w); vA = floor(a.w) / 100.0;
    }`;
  const FS = `
    precision mediump float; varying float vT; varying float vA; uniform vec3 uC0; uniform vec3 uC1;
    void main() {
      vec2 d = gl_PointCoord - 0.5; float r = length(d);
      float a = smoothstep(0.5, 0.15, r) * vA;
      gl_FragColor = vec4(mix(uC0, uC1, vT), a);
    }`;

  function initGL() {
    const gl = O.canvas.getContext('webgl', { premultipliedAlpha: false, antialias: true, alpha: true });
    if (!gl) return false;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) return false;
    gl.useProgram(p);
    O.buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, O.buf);
    const loc = gl.getAttribLocation(p, 'a'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
    gl.enable(gl.BLEND);
    O.gl = gl; O.prog = p;
    O.u = { dpr: gl.getUniformLocation(p, 'uDpr'), scale: gl.getUniformLocation(p, 'uScale'),
      c0: gl.getUniformLocation(p, 'uC0'), c1: gl.getUniformLocation(p, 'uC1') };
    return true;
  }

  function resize() {
    const c = O.canvas; if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    O.dpr = dpr;
  }

  // ---- audio analysis (real signal only)
  function analyse() {
    let lvl = 0, mic = 0;
    if (O.out && O.state === 'speaking') {
      O.out.getFloatTimeDomainData(O.td);
      let s = 0; for (let i = 0; i < O.td.length; i++) s += O.td[i] * O.td[i];
      lvl = Math.min(1, Math.sqrt(s / O.td.length) * 4.5);
      O.out.getByteFrequencyData(O.outBins);
      // log-spaced bands over ~90 Hz .. 6 kHz
      const n = O.outBins.length, ny = O.out.context.sampleRate / 2;
      for (let b = 0; b < BANDS; b++) {
        const f0 = 90 * Math.pow(6000 / 90, b / BANDS), f1 = 90 * Math.pow(6000 / 90, (b + 1) / BANDS);
        const i0 = Math.max(1, Math.floor(f0 / ny * n)), i1 = Math.max(i0 + 1, Math.ceil(f1 / ny * n));
        let m = 0; for (let i = i0; i < i1 && i < n; i++) m = Math.max(m, O.outBins[i]);
        const v = Math.max(0, (m - 70) / 150);
        O.bands[b] += (v - O.bands[b]) * (v > O.bands[b] ? 0.55 : 0.18);
      }
    } else {
      for (let b = 0; b < BANDS; b++) O.bands[b] *= 0.85;
    }
    if (O.mic && O.state === 'listening') {
      O.mic.getFloatTimeDomainData(O.td);
      let s = 0; for (let i = 0; i < O.td.length; i++) s += O.td[i] * O.td[i];
      mic = Math.min(1, Math.sqrt(s / O.td.length) * 9);
    }
    O.level += (lvl - O.level) * (lvl > O.level ? 0.6 : 0.15);
    O.micLevel += (mic - O.micLevel) * (mic > O.micLevel ? 0.6 : 0.12);
  }

  // ---- particle positions (shared by WebGL and 2D)
  function compute(t) {
    const w = O.w, k = reduced ? 0.3 : 1;
    const rot = reduced ? 0.4 : t * 0.12 + w.thinking * t * 0.5;
    const breath = 1 + 0.035 * Math.sin(t * 1.15) * (w.idle + 0.4 * w.listening);
    const lean = 1 + 0.10 * w.listening * (0.4 + O.micLevel);
    const pull = 1 - 0.30 * w.thinking;
    const lvl = O.level * w.speaking;
    for (let i = NS; i < N; i++) ring(i, t, k);
    for (let i = 0; i < NS; i++) {
      let x = O.dirs[i * 3], y = O.dirs[i * 3 + 1], z = O.dirs[i * 3 + 2];
      const sd = O.seeds[i];
      // thinking: each latitude orbits at its own speed, so particles visibly swirl inward
      const ang = rot + w.thinking * (reduced ? 0 : t * (0.8 + 1.6 * sd) * (y > 0 ? 1 : -1));
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const rx = x * ca + z * sa, rz = -x * sa + z * ca; x = rx; z = rz;
      const tilt = 0.35, ct = Math.cos(tilt), st = Math.sin(tilt);
      const ry = y * ct - z * st, rz2 = y * st + z * ct; y = ry; z = rz2;
      // speaking: frequency bands push the surface out, modulated by a travelling wave
      const band = O.bands[O.bandOf[i]] * w.speaking;
      const wave = Math.sin(x * 5.0 + t * 6.0 + sd * 2.0) * Math.cos(y * 4.0 - t * 4.0);
      let r = 0.62 * breath * pull * lean;
      r *= 1 + k * (0.30 * band * (0.55 + 0.45 * wave) + 0.14 * lvl);
      // listening: ripples run across the side facing the person, scaled by the live mic level
      r *= 1 + k * w.listening * O.micLevel * 0.14 * Math.sin(z * 14.0 - t * 9.0);
      // idle shimmer, tiny
      r *= 1 + 0.008 * Math.sin(t * 2.0 + sd * 40.0);
      const px = x * r, py = y * r, pz = z * r;
      const persp = 1 / (1 - pz * 0.35);
      const depth = (z + 1) / 2;
      const size = (2.0 + 3.2 * depth + 5.0 * band + 2.0 * w.listening * O.micLevel) * persp * O.sizeK;
      const alpha = Math.min(99, Math.round((0.22 + 0.78 * depth) * (O.dark ? 80 : 95)));
      const tint = Math.min(0.999, 0.15 + 0.55 * (1 - depth) + 0.5 * band);
      O.data[i * 4] = px * persp; O.data[i * 4 + 1] = py * persp; O.data[i * 4 + 2] = size;
      O.data[i * 4 + 3] = alpha + tint;
    }
  }

  // Halo: a tilted ring of particles orbiting the sphere; it spins up while tools run and swells with the voice.
  function ring(i, t, k) {
    const w = O.w, sd = O.seeds[i];
    const speed = reduced ? 0 : (0.10 + 0.9 * w.thinking + 0.25 * w.speaking);
    O.ringPhase = O.ringPhase || new Float32Array(N);
    O.ringPhase[i] += speed * (0.6 + 0.8 * sd) * 0.016;
    const a = sd * 6.2832 * 7 + O.ringPhase[i];
    const band = O.bands[O.bandOf[i]] * w.speaking;
    const rr = (0.80 + 0.07 * Math.sin(sd * 91)) * (1 - 0.18 * w.thinking) * (1 + k * (0.12 * band + 0.06 * w.listening * O.micLevel));
    let x = Math.cos(a) * rr, z = Math.sin(a) * rr, y = (Math.sin(sd * 53) * 0.025);
    const tilt = 1.18, ct = Math.cos(tilt), st = Math.sin(tilt);
    const y2 = y * ct - z * st, z2 = y * st + z * ct;
    const tw = -0.32, cw = Math.cos(tw), sw = Math.sin(tw);
    const x3 = x * cw - y2 * sw, y3 = x * sw + y2 * cw;
    const persp = 1 / (1 - z2 * 0.35), depth = (z2 / rr + 1) / 2;
    const vis = 0.25 + 0.55 * w.thinking + 0.35 * w.speaking + 0.2 * w.listening;
    O.data[i * 4] = x3 * persp; O.data[i * 4 + 1] = y3 * persp;
    O.data[i * 4 + 2] = (1.2 + 1.8 * depth + 3 * band) * persp * O.sizeK;
    O.data[i * 4 + 3] = Math.min(99, Math.round((0.15 + 0.6 * depth) * vis * 95)) + 0.55;
  }

  function frame(now) {
    O.raf = requestAnimationFrame(frame);
    if (now - O.last < 1000 / 60 - 2) return; // cap at 60 fps on high refresh displays
    O.last = now;
    O.frames.push(now); while (O.frames.length && now - O.frames[0] > 1000) O.frames.shift();
    const t = (now - O.t0) / 1000;
    // ease state weights; the target state starts moving on the very next frame (<17 ms)
    for (const s of Object.keys(O.w)) { const tgt = s === O.state ? 1 : 0; O.w[s] += (tgt - O.w[s]) * 0.16; }
    // Measured: time from a state change to the first frame drawn with the new state's weights moving.
    if (O.stateAt && !O.stateShown) { O.stateShown = true; O.stateLat.push(performance.now() - O.stateAt); if (O.stateLat.length > 50) O.stateLat.shift(); }
    analyse();
    compute(t);
    draw();
    if (O.glow) {
      const g = 1 + 0.25 * O.level * O.w.speaking + 0.18 * O.micLevel * O.w.listening - 0.12 * O.w.thinking;
      O.glow.style.transform = `scale(${g.toFixed(3)})`;
      O.glow.style.opacity = (0.55 + 0.45 * Math.max(O.level, O.micLevel, O.w.thinking * 0.6)).toFixed(3);
    }
  }

  function draw() {
    const c = O.canvas, aspect = c.width / c.height;
    const sx = aspect > 1 ? 1 / aspect : 1, sy = aspect > 1 ? 1 : aspect;
    if (O.gl) {
      const gl = O.gl;
      gl.viewport(0, 0, c.width, c.height);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      if (O.dark) gl.blendFunc(gl.SRC_ALPHA, gl.ONE); else gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniform1f(O.u.dpr, O.dpr); gl.uniform2f(O.u.scale, sx, sy);
      // HP blue -> light tint (thinking cools toward deeper blue)
      if (O.dark) { gl.uniform3f(O.u.c0, 0.20, 0.47, 1.0); gl.uniform3f(O.u.c1, 0.72, 0.84, 1.0); }
      else { gl.uniform3f(O.u.c0, 0.0, 0.294, 0.875); gl.uniform3f(O.u.c1, 0.42, 0.62, 0.98); }
      gl.bufferData(gl.ARRAY_BUFFER, O.data, gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.POINTS, 0, N);
    } else if (O.ctx2d) {
      const ctx = O.ctx2d; ctx.clearRect(0, 0, c.width, c.height);
      const cx = c.width / 2, cy = c.height / 2, R = Math.min(c.width, c.height) / 2;
      for (let i = 0; i < N; i += 2) {
        const a = Math.floor(O.data[i * 4 + 3]) / 100, tt = O.data[i * 4 + 3] % 1;
        ctx.fillStyle = `rgba(${Math.round(0 + 107 * tt)},${Math.round(75 + 83 * tt)},${Math.round(223 + 27 * tt)},${a})`;
        ctx.beginPath(); ctx.arc(cx + O.data[i * 4] * R, cy - O.data[i * 4 + 1] * R, O.data[i * 4 + 2] * O.dpr * 0.5, 0, 6.283); ctx.fill();
      }
    }
  }

  function start() { if (!O.raf && O.enabled && !document.hidden) { O.last = 0; O.raf = requestAnimationFrame(frame); } }
  function stop() { if (O.raf) cancelAnimationFrame(O.raf); O.raf = 0; O.frames = []; }

  window.AtlasOrb = {
    init(canvas, glow) {
      O.canvas = canvas; O.glow = glow;
      makeParticles();
      let ok = false; try { ok = initGL(); } catch (e) { console.warn('orb webgl', e); }
      if (!ok) { O.gl = null; O.ctx2d = canvas.getContext('2d'); }
      O.sizeK = 1;
      resize(); new ResizeObserver(() => { resize(); O.sizeK = Math.max(0.6, Math.min(1.6, canvas.clientHeight / 560)); }).observe(canvas);
      document.addEventListener('visibilitychange', () => (document.hidden ? stop() : start()));
      start();
      return O.gl ? 'webgl' : '2d';
    },
    setState(st) { if (st !== O.state) { O.state = st; O.stateAt = performance.now(); O.stateShown = false; } },
    stateLatencies() { return O.stateLat.slice(); },
    setDark(on) { O.dark = !!on; },
    setEnabled(on) { O.enabled = !!on; if (on) start(); else stop(); },
    attachOutput(an) { O.out = an; an.fftSize = 1024; an.smoothingTimeConstant = 0.5; O.outBins = new Uint8Array(an.frequencyBinCount); O.td = O.td || new Float32Array(1024); },
    attachMic(an) { O.mic = an; an.fftSize = 1024; O.td = O.td || new Float32Array(1024); },
    fps() { return O.frames.length; },
    levels() { return { level: O.level, mic: O.micLevel, bands: Array.from(O.bands), state: O.state, weights: { ...O.w } }; },
    renderer() { return O.gl ? 'webgl' : (O.ctx2d ? '2d' : 'none'); },
    reduced,
  };
})();
