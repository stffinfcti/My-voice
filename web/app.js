/* My Voice: app logic. Records 16 kHz mono WAV in the browser and talks to the api/ endpoints. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const mq = window.matchMedia ? matchMedia("(prefers-reduced-motion: reduce)") : null;
  let reduced = !!(mq && mq.matches);
  if (mq && mq.addEventListener) mq.addEventListener("change", e => { reduced = e.matches; Wave.kick(); Ring.kick(); });
  const MAX_SEC = 15;
  const TABS = ["teach", "training", "talk"];
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
    del(k) { try { localStorage.removeItem(k); } catch (e) {} },
  };
  const S = { phrases: [], test: [], min: 10, i: 0, recs: [], voice: null, voiceN: null, testK: -1, demo: null, busy: false, tab: "teach",
              train: { state: "idle", p: 0, n: 0, seconds: null, err: "", eta: null }, last: null };
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const ease = t => 1 - Math.pow(1 - t, 3);
  const nRec = () => S.recs.filter(Boolean).length;

  function announce(msg) { const el = $("srStatus"); el.textContent = ""; setTimeout(() => { el.textContent = msg; }, 60); }
  function toast(msg, ms = 3200) {
    const el = $("toast"); el.textContent = msg; el.classList.add("on");
    clearTimeout(toast.h); toast.h = setTimeout(() => el.classList.remove("on"), ms);
  }

  /* ---------------- audio: getUserMedia -> Web Audio -> 16 kHz mono 16-bit WAV ---------------- */
  const Mic = {
    ctx: null, stream: null, src: null, proc: null, an: null, mute: null, chunks: [], on: false, tbuf: null,
    // Must run synchronously inside the tap handler (iOS only unlocks audio during a user gesture).
    unlock() {
      if (!this.ctx) { const AC = window.AudioContext || window.webkitAudioContext; if (!AC) throw new Error("nosupport"); this.ctx = new AC(); }
      if (this.ctx.state !== "running") { try { this.ctx.resume(); } catch (e) {} }
    },
    async start() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("nosupport");
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      // Some iOS versions misreport audio if the context rate differs from the mic; rebuild it then.
      const tr = this.stream.getAudioTracks()[0], set = tr && tr.getSettings ? tr.getSettings() : {};
      if (set.sampleRate && Math.abs(set.sampleRate - this.ctx.sampleRate) > 1) {
        try {
          const AC = window.AudioContext || window.webkitAudioContext, fresh = new AC({ sampleRate: set.sampleRate });
          await Promise.race([fresh.resume(), sleep(800)]);
          if (fresh.state === "running") { this.ctx.close(); this.ctx = fresh; } else fresh.close();
        } catch (e) {}
      }
      if (this.ctx.state !== "running") await Promise.race([this.ctx.resume(), sleep(1500)]);
      const ctx = this.ctx;
      this.src = ctx.createMediaStreamSource(this.stream);
      this.an = ctx.createAnalyser(); this.an.fftSize = 1024; this.an.smoothingTimeConstant = 0.6;
      this.src.connect(this.an);
      this.proc = ctx.createScriptProcessor(4096, 1, 1); this.chunks = [];
      this.proc.onaudioprocess = e => { if (this.on) this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
      this.mute = ctx.createGain(); this.mute.gain.value = 0;
      this.src.connect(this.proc); this.proc.connect(this.mute); this.mute.connect(ctx.destination);
      this.tbuf = new Float32Array(this.an.fftSize);
      this.on = true;
    },
    read(buf) {                           // live time-domain samples for the waveform
      if (!this.on || !this.an) return null;
      this.an.getFloatTimeDomainData(buf); return buf;
    },
    level() {                             // 0..1 loudness for the mic rings
      if (!this.on || !this.an) return 0;
      this.an.getFloatTimeDomainData(this.tbuf);
      let s = 0; for (let i = 0; i < this.tbuf.length; i += 2) s += this.tbuf[i] * this.tbuf[i];
      return Math.min(1, Math.sqrt(s / (this.tbuf.length / 2)) * 5.5);
    },
    async stop() {
      this.on = false;
      try { this.src.disconnect(); this.an.disconnect(); this.proc.disconnect(); this.mute.disconnect(); } catch (e) {}
      if (this.stream) this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
      const sr = this.ctx.sampleRate;
      let n = 0; for (const c of this.chunks) n += c.length;
      const all = new Float32Array(n); let o = 0; for (const c of this.chunks) { all.set(c, o); o += c.length; }
      this.chunks = [];
      try { this.ctx.suspend(); } catch (e) {}
      // Send the mic's own sample rate untouched; the server resamples to 16 kHz with a proper filter.
      return { blob: encodeWav(all, sr), seconds: n / sr, env: envelope(all, 72) };
    },
  };
  function resample(x, from, to) {
    if (from === to) return x;
    const r = from / to, n = Math.floor(x.length / r), y = new Float32Array(n);
    for (let i = 0; i < n; i++) {            // box filter + decimate (good enough for speech at 16 kHz)
      const a = Math.floor(i * r), b = Math.min(x.length, Math.floor((i + 1) * r));
      let s = 0; for (let j = a; j < b; j++) s += x[j]; y[i] = s / Math.max(1, b - a);
    }
    return y;
  }
  function encodeWav(x, sr) {
    const b = new ArrayBuffer(44 + x.length * 2), v = new DataView(b);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, "data"); v.setUint32(40, x.length * 2, true);
    for (let i = 0; i < x.length; i++) { const s = Math.max(-1, Math.min(1, x[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
    return new Blob([b], { type: "audio/wav" });
  }
  function envelope(x, bins) {             // loudness shape of a take, 0..1, for the Talk waveforms
    const out = new Array(bins).fill(0); if (!x.length) return out;
    const step = x.length / bins; let mx = 1e-6;
    for (let b = 0; b < bins; b++) {
      const a = Math.floor(b * step), e = Math.min(x.length, Math.floor((b + 1) * step)); let s = 0;
      for (let i = a; i < e; i++) s += x[i] * x[i];
      out[b] = Math.sqrt(s / Math.max(1, e - a)); mx = Math.max(mx, out[b]);
    }
    return out.map(v => Math.sqrt(v / mx));
  }

  /* ---------------- live level -> mic rings ---------------- */
  const Meter = { wrap: null, src: null, v: 0, raf: 0 };
  function meterStart(wrap, src) { Meter.wrap = wrap; Meter.src = src; if (!Meter.raf) Meter.raf = requestAnimationFrame(meterTick); }
  function meterStop() { Meter.src = null; }
  function meterTick() {
    const raw = Meter.src ? Meter.src() : 0;
    Meter.v += (raw - Meter.v) * (raw > Meter.v ? 0.4 : 0.1);       // quick attack, slow release
    const lv = reduced ? 0 : Math.min(1, Math.pow(Math.max(0, Meter.v), 0.75));
    if (Meter.wrap) Meter.wrap.style.setProperty("--lvl", lv.toFixed(3));
    if (Meter.src || Meter.v > 0.003) Meter.raf = requestAnimationFrame(meterTick);
    else { Meter.raf = 0; Meter.v = 0; if (Meter.wrap) Meter.wrap.style.setProperty("--lvl", "0"); }
  }

  /* ---------------- Teach waveform: dense orange + gray lines on canvas ---------------- */
  const Wave = {
    cv: $("wave"), ctx: null, W: 0, H: 0, dpr: 1, L: [], raf: 0, on: false, live: null, buf: new Float32Array(1024),
    size() {
      const r = this.cv.getBoundingClientRect(); if (!r.width || !r.height) return false;
      this.dpr = Math.min(3, window.devicePixelRatio || 1); this.W = r.width; this.H = r.height;
      this.cv.width = Math.round(r.width * this.dpr); this.cv.height = Math.round(r.height * this.dpr);
      this.ctx = this.cv.getContext("2d");
      let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      const n = Math.floor(this.W / 1.6);
      this.L = Array.from({ length: n }, () => { const q = rnd(); return { c: q < .56 ? 0 : q < .74 ? 1 : 2, hi: rnd() > .45 ? 1 : 0, up: .65 + .7 * rnd(), dn: .65 + .7 * rnd(), ph: rnd() * 6.283, v: 0 }; });
      return true;
    },
    show(on) { this.on = on; if (on) this.kick(); else { cancelAnimationFrame(this.raf); this.raf = 0; } },
    kick() { if (this.on && !this.raf) this.raf = requestAnimationFrame(t => this.frame(t)); },
    frame(now) {
      this.raf = 0;
      if (!this.ctx && !this.size()) { if (this.on) this.raf = requestAnimationFrame(t => this.frame(t)); return; }
      const { ctx, W, H, L, dpr } = this, n = L.length, cy = H / 2, max = H / 2 - 1, t = now / 1000;
      const data = this.live ? this.live(this.buf) : null;
      let lvl = 0;
      if (data) { let s = 0; for (let i = 0; i < data.length; i += 4) s += data[i] * data[i]; lvl = Math.min(1, Math.sqrt(s / (data.length / 4)) * 5); }
      const still = reduced && !data;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const cols = ["#F07A2E", "#F4A064", "#A9A9AE"];
      const paths = [[], [], [], [], [], []];
      for (let i = 0; i < n; i++) {
        const l = L[i], x = (i + .5) * W / n, d = (x / W - .5) / .31, env = .045 + .955 * Math.exp(-d * d * 1.45);
        let tgt;
        if (data) tgt = Math.min(1, Math.abs(data[(i * 7) % data.length]) * 3.4 + lvl * (.25 + .75 * Math.random()));
        else {
          const tt = still ? 1.3 : t;
          tgt = .06 + .1 * (.5 + .5 * Math.sin(tt * 1.5 + i * .16 + l.ph * .25)) * (.5 + .5 * Math.sin(tt * .7 - i * .041)) + (still ? 0 : .025 * Math.random());
        }
        l.v += (tgt - l.v) * (still ? 1 : tgt > l.v ? .55 : .13);
        const h = l.v * env * max, a = Math.max(.4, Math.min(max, h * l.up)), b = Math.max(.4, Math.min(max, h * l.dn));
        paths[l.c * 2 + l.hi].push(x, cy - a, cy + b);
      }
      ctx.lineWidth = 1;
      for (let k = 0; k < 6; k++) {
        const p = paths[k]; if (!p.length) continue;
        ctx.beginPath();
        for (let j = 0; j < p.length; j += 3) { ctx.moveTo(p[j], p[j + 1]); ctx.lineTo(p[j], p[j + 2]); }
        ctx.strokeStyle = cols[k >> 1]; ctx.globalAlpha = (k & 1) ? .95 : .6; ctx.stroke();
      }
      ctx.globalAlpha = .5; ctx.fillStyle = cols[0]; ctx.fillRect(0, cy - .4, W, .8); ctx.globalAlpha = 1;
      if (this.on && !still) this.raf = requestAnimationFrame(t2 => this.frame(t2));
    },
  };

  /* ---------------- Training ring: overlapping ribbons of fine lines and dots ---------------- */
  const Ring = {
    cv: $("ring"), ctx: null, Z: 0, dpr: 1, raf: 0, on: false, T: 3, rot: 0, last: 0, amp: .8, spd: .45, vis: .85, mode: "idle",
    defs: [
      { c: "#F07A2E", a: .95, off: .02, w: .21, h: [[3, .07, .35, 0], [5, .04, -.5, 1.3], [7, .022, .8, 2.1]], wn: 2, ws: .4, wp: 0 },
      { c: "#A3A3A8", a: .8, off: -.05, w: .17, h: [[4, .06, -.3, .7], [6, .035, .45, 2.6], [2, .03, .25, 1]], wn: 3, ws: -.3, wp: 1.2 },
      { c: "#F39A5B", a: .85, off: .08, w: .15, h: [[5, .05, .4, 2.2], [3, .05, -.25, .4], [8, .02, .6, 1.7]], wn: 2, ws: .5, wp: 2.5 },
      { c: "#BDBDC2", a: .8, off: .12, w: .12, h: [[6, .045, -.4, 1.1], [4, .04, .3, 3.0], [2, .035, -.2, .2]], wn: 3, ws: .35, wp: .6 },
      { c: "#E8682A", a: .9, off: -.1, w: .14, h: [[4, .05, .5, 2.8], [7, .03, -.35, .9], [3, .04, .2, 1.9]], wn: 2, ws: -.45, wp: 3.3 },
      { c: "#9A9AA0", a: .7, off: .0, w: .1, h: [[5, .055, -.45, .3], [3, .045, .3, 2.3], [9, .015, .7, 1.4]], wn: 4, ws: .25, wp: 4.1 },
    ],
    dust: null,
    size() {
      const r = this.cv.getBoundingClientRect(); if (!r.width) return false;
      this.dpr = Math.min(3, window.devicePixelRatio || 1); this.Z = r.width;
      this.cv.width = Math.round(r.width * this.dpr); this.cv.height = Math.round(r.width * this.dpr);
      this.ctx = this.cv.getContext("2d");
      if (!this.dust) {
        let seed = 5; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
        this.dust = Array.from({ length: 560 }, () => {
          const u = rnd(), side = rnd() < .5 ? -1 : 1;
          return { th: rnd() * 6.283, rr: 1 + side * (.12 + .42 * u * u), sz: .7 + rnd() * .9, c: rnd() < .34 ? 1 : 0, sp: .5 + rnd(), ph: rnd() * 6.283 };
        });
      }
      return true;
    },
    setMode(m) { this.mode = m; this.kick(); },
    show(on) { this.on = on; if (on) { this.last = 0; this.kick(); } else { cancelAnimationFrame(this.raf); this.raf = 0; } },
    kick() { if (this.on && !this.raf) this.raf = requestAnimationFrame(t => this.frame(t)); },
    frame(now) {
      this.raf = 0;
      if (!this.ctx && !this.size()) { if (this.on) this.raf = requestAnimationFrame(t => this.frame(t)); return; }
      const dt = this.last ? Math.min(.05, (now - this.last) / 1000) : 0; this.last = now;
      const tg = { idle: [.8, .45, .82], run: [1, 1, 1], done: [.32, .4, 1] }[this.mode] || [1, 1, 1];
      if (reduced) { this.amp = tg[0]; this.vis = tg[2]; }
      else { const k = 1 - Math.exp(-dt * 1.6); this.amp += (tg[0] - this.amp) * k; this.spd += (tg[1] - this.spd) * k; this.vis += (tg[2] - this.vis) * k; }
      if (!reduced) { this.T += dt * this.spd; this.rot += dt * .07 * this.spd; }
      this.draw();
      if (this.on && !reduced) this.raf = requestAnimationFrame(t => this.frame(t));
    },
    draw() {
      const { ctx, Z, dpr, T, rot, amp, vis } = this, cx = Z / 2, cy = Z / 2, R = Z * .345, M = 300, TAU = Math.PI * 2;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, Z, Z);
      // dust
      const dp = [[], []];
      for (const d of this.dust) {
        const th = d.th + rot * .8 + Math.sin(T * .3 * d.sp + d.ph) * .03;
        const r = R * (1 + (d.rr - 1) * (.55 + .45 * amp)) * (1 + .025 * Math.sin(T * .7 * d.sp + d.ph));
        dp[d.c].push(cx + r * Math.cos(th), cy + r * Math.sin(th), d.sz);
      }
      [["#A3A3A8", .5], ["#F07A2E", .55]].forEach(([c, a], k) => {
        const p = dp[k]; ctx.beginPath();
        for (let j = 0; j < p.length; j += 3) ctx.rect(p[j], p[j + 1], p[j + 2], p[j + 2]);
        ctx.fillStyle = c; ctx.globalAlpha = a * vis; ctx.fill();
      });
      // ribbons
      const e1 = new Float32Array(M * 2), e2 = new Float32Array(M * 2);
      for (const f of this.defs) {
        ctx.beginPath();
        for (let i = 0; i < M; i++) {
          const th = i / M * TAU;
          let rc = R * (1 + f.off * (.4 + .6 * amp));
          for (const h of f.h) rc += R * h[1] * amp * Math.sin(h[0] * th + h[3] + h[2] * T);
          const w = R * f.w * (.3 + .7 * amp) * Math.sin(f.wn * th + f.wp + f.ws * T);
          const a1 = th + rot, a2 = a1 + .034, r1 = rc + w / 2, r2 = rc - w / 2;
          const x1 = cx + r1 * Math.cos(a1), y1 = cy + r1 * Math.sin(a1), x2 = cx + r2 * Math.cos(a2), y2 = cy + r2 * Math.sin(a2);
          ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
          e1[i * 2] = x1; e1[i * 2 + 1] = y1; e2[i * 2] = x2; e2[i * 2 + 1] = y2;
        }
        ctx.strokeStyle = f.c; ctx.lineWidth = .55; ctx.globalAlpha = f.a * .5 * vis; ctx.stroke();
        ctx.beginPath();
        for (const e of [e1, e2]) { ctx.moveTo(e[0], e[1]); for (let i = 1; i < M; i++) ctx.lineTo(e[i * 2], e[i * 2 + 1]); ctx.closePath(); }
        ctx.lineWidth = .8; ctx.globalAlpha = f.a * .8 * vis; ctx.stroke();
        ctx.beginPath();
        for (let i = 0; i < M; i += 3) { const j = i * 2, k = (i * 37) % 11 / 11 - .5; ctx.rect(e1[j] + (e1[j] - cx) * k * .06, e1[j + 1] + (e1[j + 1] - cy) * k * .06, 1.1, 1.1); }
        ctx.fillStyle = f.c; ctx.globalAlpha = f.a * .7 * vis; ctx.fill();
      }
      ctx.globalAlpha = 1;
    },
  };

  /* ---------------- shared record-button controller ---------------- */
  function makeRec(wrap, { ariaIdle, onStart, onTick, onFail, onBlob }) {
    const btn = wrap.querySelector(".mic");
    let timer = 0, tick = 0, t0 = 0;
    const R = {
      on: false,
      ui(on) {
        R.on = on; wrap.classList.toggle("on", on);
        btn.setAttribute("aria-label", on ? "Stop recording" : ariaIdle);
        if (on) meterStart(wrap, () => Mic.level()); else meterStop();
      },
      async toggle() {
        if (S.busy || S.demo) return;
        if (!R.on) {
          try { Mic.unlock(); } catch (e) { onFail("This browser can't record audio"); return; }
          S.busy = true;
          try { await Mic.start(); }
          catch (e) {
            S.busy = false;
            const denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
            onFail(denied ? "microphone is blocked" : "couldn't start the microphone");
            toast(denied ? "Allow microphone access for this site in your browser settings, then try again." :
                  (e && e.message === "nosupport") ? "This browser can't record audio. Try Safari or Chrome." : "Couldn't start the microphone. Try again.", 5200);
            return;
          }
          S.busy = false; R.ui(true); t0 = performance.now(); onStart(); onTick(0);
          announce("Recording. Tap the button again to stop.");
          tick = setInterval(() => onTick((performance.now() - t0) / 1000), 250);
          timer = setTimeout(() => { if (R.on) R.toggle(); }, MAX_SEC * 1000);
          return;
        }
        clearTimeout(timer); clearInterval(tick); S.busy = true; R.ui(false);
        let out; try { out = await Mic.stop(); } finally { S.busy = false; }
        await onBlob(out);
      },
    };
    btn.addEventListener("click", () => R.toggle());
    return R;
  }
  const clock = s => `0:${String(Math.floor(Math.min(MAX_SEC, s))).padStart(2, "0")}`;

  /* ---------------- tabs ---------------- */
  function setTab(name, focus) {
    const prev = S.tab; S.tab = name;
    for (const n of TABS) {
      const on = n === name, tab = $("tab-" + n);
      tab.setAttribute("aria-selected", on); tab.tabIndex = on ? 0 : -1;
      $("view-" + n).hidden = !on;
    }
    if (prev !== name && !reduced) {
      const v = $("view-" + name);
      v.classList.remove("in-r", "in-l"); void v.offsetWidth;
      v.classList.add(TABS.indexOf(name) > TABS.indexOf(prev) ? "in-r" : "in-l");
    }
    Wave.show(name === "teach");
    Ring.show(name === "training");
    if (focus) $("tab-" + name).focus();
  }
  for (const n of TABS) {
    $("view-" + n).addEventListener("animationend", e => { if (e.target === e.currentTarget) e.currentTarget.classList.remove("in-r", "in-l"); });
    $("tab-" + n).addEventListener("click", () => { if (!S.busy && !rec(1)) setTab(n); });
  }
  $("tabbar").addEventListener("keydown", e => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault(); if (rec(1)) return;
    const k = TABS.indexOf(S.tab) + (e.key === "ArrowRight" ? 1 : -1);
    setTab(TABS[(k + TABS.length) % TABS.length], true);
  });
  function rec(warn) { const on = teachR.on || talkR.on; if (on && warn) toast("Stop recording first"); return on; }
  window.addEventListener("resize", () => { Wave.ctx = null; Ring.ctx = null; Wave.kick(); Ring.kick(); });

  /* ---------------- TEACH ---------------- */
  let shownPhrase = null;
  function showPhrase(text) {
    const el = $("phrase");
    if (reduced) { el.textContent = text; return; }
    el.innerHTML = text.split(" ").map((w, k) => `<span class="w" style="--i:${k}">${esc(w)}</span>`).join(" ");
  }
  function teachIdle() {
    $("teachHint").textContent = "tap the mic and read it aloud";
    $("teachUnder").textContent = S.recs[S.i] ? "Recorded. Tap to redo." : "Tap to record.";
  }
  function renderTeach() {
    const n = S.phrases.length, done = nRec();
    $("count").textContent = n ? `${S.i + 1} of ${n}` : "";
    const text = S.phrases[S.i] || "";
    if (text !== shownPhrase) { shownPhrase = text; showPhrase(text); }
    $("backBtn").disabled = S.i === 0;
    $("playBtn").hidden = !S.recs[S.i];
    $("goTrain").hidden = done < S.min || S.train.state === "running";
    if (!teachR.on) teachIdle();
    renderTrain();
  }
  function nextMissing(from) {
    const n = S.phrases.length;
    for (let k = 1; k <= n; k++) { const j = (from + k) % n; if (!S.recs[j]) return j; }
    return null;
  }
  const teachR = makeRec($("teachWrap"), {
    ariaIdle: "Record this phrase",
    onStart() { $("teachHint").textContent = "listening"; Wave.live = b => Mic.read(b); Wave.kick(); },
    onTick(s) { $("teachUnder").textContent = `Tap to stop · ${clock(s)}`; },
    onFail(msg) { $("teachHint").textContent = msg; announce(msg); },
    async onBlob({ blob, seconds }) {
      Wave.live = null; Wave.kick();
      if (seconds < 0.5) { $("teachHint").textContent = "too short, try again"; $("teachUnder").textContent = "Tap to record."; announce("Too short. Try again."); return; }
      const at = S.i; S.recs[at] = blob; renderTeach();
      $("teachHint").textContent = "saved"; $("teachUnder").textContent = `${nRec()} of ${S.phrases.length} recorded`;
      announce(`Saved. ${nRec()} of ${S.phrases.length} recorded.`);
      await sleep(reduced ? 300 : 750);
      if (S.i !== at || teachR.on) return;
      const nx = nextMissing(at);
      if (nx !== null) { S.i = nx; renderTeach(); }
      else { renderTeach(); $("teachHint").textContent = "all phrases recorded"; announce("All phrases recorded. You can train your voice."); }
    },
  });
  $("backBtn").addEventListener("click", () => { if (rec(1)) return; if (S.i > 0) { S.i--; renderTeach(); } });
  $("skipBtn").addEventListener("click", () => { if (rec(1)) return; S.i = (S.i + 1) % Math.max(1, S.phrases.length); renderTeach(); });
  let player = null;
  $("playBtn").addEventListener("click", () => {
    const b = S.recs[S.i]; if (!b || rec(1)) return;
    try { if (player) { player.pause(); URL.revokeObjectURL(player.src); } player = new Audio(URL.createObjectURL(b)); player.play().catch(() => toast("Couldn't play that recording")); }
    catch (e) { toast("Couldn't play that recording"); }
  });
  $("goTrain").addEventListener("click", () => { if (rec(1)) return; setTab("training"); train(); });

  /* ---------------- TRAINING ---------------- */
  let wake = null;
  async function keepAwake() { try { if (navigator.wakeLock) wake = await navigator.wakeLock.request("screen"); } catch (e) {} }
  function letSleep() { try { if (wake) wake.release(); } catch (e) {} wake = null; }
  document.addEventListener("visibilitychange", () => { if (!document.hidden && wake === null && S.train.state === "running") keepAwake(); });

  // smooth percent: the shown value eases toward the latest polled value
  const TP = { shown: 0, target: 0, raf: 0, last: 0, wait: null };
  function drawTP() {
    const p = Math.min(100, Math.floor(TP.shown * 100 + 1e-4));
    if (S.train.state === "running" || S.train.state === "done" || S.train.state === "error") $("pctNum").textContent = p;
    $("tabProg").style.strokeDashoffset = (100 - TP.shown * 100).toFixed(1);
  }
  function trainTo(p) {
    TP.target = Math.max(TP.target, Math.min(1, p));
    if (reduced) { TP.shown = TP.target; drawTP(); if (TP.wait) { TP.wait(); TP.wait = null; } return; }
    if (!TP.raf) { TP.last = performance.now(); TP.raf = requestAnimationFrame(tpTick); }
  }
  function tpTick(now) {
    const dt = Math.min(0.1, (now - TP.last) / 1000); TP.last = now;
    TP.shown += (TP.target - TP.shown) * (1 - Math.exp(-dt * 2.6));
    if (Math.abs(TP.target - TP.shown) < 0.0008) TP.shown = TP.target;
    drawTP();
    if (TP.shown !== TP.target) TP.raf = requestAnimationFrame(tpTick);
    else { TP.raf = 0; if (TP.wait) { TP.wait(); TP.wait = null; } }
  }
  const settled = () => TP.shown === TP.target ? Promise.resolve() : new Promise(r => { TP.wait = r; });
  function resetTP() { cancelAnimationFrame(TP.raf); TP.raf = 0; TP.shown = TP.target = 0; drawTP(); }

  function fmtLeft(sec) {
    if (sec == null || !isFinite(sec)) return "Working out the time left";
    if (sec < 50) return "Less than a minute left";
    return `About ${Math.round(sec / 60)} min left`;
  }
  function fmtDur(s) {
    if (s == null || !isFinite(s)) return "";
    s = Math.round(s); const m = Math.floor(s / 60), r = s % 60;
    return m ? `${m} min ${r} s` : `${r} s`;
  }
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  let trKey = "";
  function renderTrain() {
    const T = S.train, done = nRec(), left = S.min - done;
    let st = T.state;
    const ready = st === "idle" && S.voice && done < S.min;      // a voice exists from before; nothing new to train on
    const pct = $("pct"), tb = $("trainBtn");
    let title, sub, eta = "", cap = "", count = null, btn = null, after = false;
    if (st === "running") {
      title = "Learning your voice..."; sub = `Analyzing ${plural(T.n, "recording")}.`;
      eta = T.p < 0.02 ? "Uploading your recordings" : fmtLeft(T.eta);
    } else if (st === "done" || ready) {
      title = "Your voice is ready";
      const n = st === "done" ? T.n : S.voiceN;
      sub = st === "done" ? `Learned from ${plural(n, "recording")}${T.seconds ? " in " + fmtDur(T.seconds) : ""}.` : (n ? `Trained on ${plural(n, "phrase")}. Record more in Teach to retrain.` : "Record more in Teach to retrain.");
      after = true;
      if (ready) { count = n || "✓"; cap = n ? "phrases learned" : ""; }
    } else if (st === "error") {
      title = "Training stopped"; sub = T.err || "Something went wrong."; btn = done >= S.min ? "Try again" : `Record ${left} more`;
    } else {
      count = done; cap = done === 1 ? "recording" : "recordings";
      if (done >= S.min) { title = "Ready to learn your voice"; sub = `${plural(done, "recording")}. Training takes a few minutes.`; btn = "Train my voice"; }
      else { title = "Teach it your voice"; sub = `Record at least ${S.min} phrases in Teach.`; btn = `Record ${left} more`; }
      if (S.voice) eta = "Training again replaces your current voice.";
    }
    pct.classList.toggle("count-mode", count !== null);
    if (count !== null) $("pctNum").textContent = count;
    else drawTP();
    $("ringCap").textContent = cap;
    $("trTitle").textContent = title; $("trSub").textContent = sub; $("trEta").textContent = eta;
    tb.hidden = !btn; if (btn) { tb.textContent = btn; tb.disabled = done < S.min; }
    $("tryBtn").hidden = !after; $("saveBtn").hidden = !after;
    $("saveBtn").href = S.voice && S.voice !== "demo" ? `api/voice/${encodeURIComponent(S.voice)}` : "#";
    Ring.setMode(st === "running" ? "run" : (st === "done" || ready) ? "done" : "idle");
    $("tabbar").classList.toggle("training", st === "running");
    const key = st + (ready ? "r" : "") + (btn || "");
    if (key !== trKey) {
      trKey = key;
      if (!reduced) [...document.querySelectorAll(".tr-text > *")].forEach((el, k) => { el.classList.remove("rise"); void el.offsetWidth; el.style.animationDelay = k * 60 + "ms"; el.classList.add("rise"); });
    }
  }
  let etaSmooth = null, lastAnn = -1;
  function setTrain(p, eta) {
    S.train.p = p; trainTo(p);
    if (eta != null) etaSmooth = etaSmooth == null ? eta : etaSmooth * 0.7 + eta * 0.3;
    S.train.eta = eta == null ? null : etaSmooth;
    renderTrain();
    const dec = Math.floor(p * 10);
    if (dec !== lastAnn) { lastAnn = dec; $("trLive").textContent = `Training ${Math.round(p * 100)} percent.`; }
  }
  async function train() {
    if (S.train.state === "running") return;
    const fd = new FormData(), idx = [];
    S.recs.forEach((b, k) => { if (b) { fd.append("files", b, `p${k}.wav`); idx.push(k); } });
    if (idx.length < S.min) { toast(`Record at least ${S.min} phrases first`); return; }
    fd.append("indices", idx.join(","));
    etaSmooth = null; lastAnn = -1; resetTP();
    S.train = { state: "running", p: 0, n: idx.length, seconds: null, err: "", eta: null };
    renderTeach(); setTrain(0, null);
    keepAwake();
    try {
      const r = await fetch("api/train", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await errText(r, "Training couldn't start."));
      const { job } = await r.json(); const t0 = Date.now(); let fails = 0;
      for (;;) {
        await sleep(1000);
        let s;
        try { const q = await fetch(`api/train/${job}`, { cache: "no-store" }); if (!q.ok) throw new Error(await errText(q, "Lost the training job.")); s = await q.json(); fails = 0; }
        catch (e) { if (++fails > 10) throw e; continue; }
        if (s.state === "error") throw new Error(s.error || "Training failed.");
        const el = (Date.now() - t0) / 1000, p = Math.max(0.01, s.progress || 0);
        if (s.state === "done") {
          trainTo(1); await settled();
          S.train.state = "done"; S.train.n = s.n || idx.length; S.train.seconds = s.seconds;
          setVoice(s.voice, s.n || idx.length);
          renderTeach(); $("trTitle").focus({ preventScroll: true });
          announce("Your voice is ready."); break;
        }
        setTrain(p, p > 0.04 ? el * (1 - p) / p : null);
      }
    } catch (e) {
      S.train.state = "error"; S.train.err = (e && e.message) || "Training failed.";
      renderTeach(); announce("Training stopped. " + S.train.err);
    } finally { letSleep(); }
  }
  async function errText(r, fallback) {
    try { const j = await r.json(); return (j && (typeof j.detail === "string" ? j.detail : null)) || fallback; } catch (e) { return fallback; }
  }
  $("trainBtn").addEventListener("click", () => { if (!rec(1)) train(); });
  $("tryBtn").addEventListener("click", () => { setTab("talk"); setTimeout(() => $("talkMic").focus({ preventScroll: true }), 60); });
  $("saveBtn").addEventListener("click", e => { if (!S.voice || S.demo) { e.preventDefault(); toast(S.demo ? "This is example data" : "No voice file yet"); } });

  /* ---------------- voice id ---------------- */
  function setVoice(id, n) {
    S.voice = id || null; S.voiceN = n || null;
    if (id) { store.set("myvoice.voice", id); if (n) store.set("myvoice.n", String(n)); }
    else { store.del("myvoice.voice"); store.del("myvoice.n"); }
    renderVoice(); renderTrain();
  }
  function renderVoice() {
    $("voiceLine").textContent = S.voice ? `Using your voice${S.voiceN ? ` · ${S.voiceN} phrases` : ""}` : "Using the shared model";
  }
  async function checkVoice() {        // the server forgets voices on restart; check the stored id still exists
    if (!S.voice) return;
    const ac = new AbortController();
    try {
      const r = await fetch(`api/voice/${encodeURIComponent(S.voice)}`, { signal: ac.signal, cache: "no-store" });
      ac.abort();
      if (r.status === 404) { setVoice(null); toast("Your saved voice isn't on this server any more. Load your voice file in Talk.", 5200); }
    } catch (e) {}
  }
  $("loadBtn").addEventListener("click", () => { if (!rec(1)) $("voiceFile").click(); });
  $("voiceFile").addEventListener("change", async e => {
    const f = e.target.files && e.target.files[0]; e.target.value = "";
    if (!f) return;
    const fd = new FormData(); fd.append("file", f, f.name || "my-voice.safetensors");
    try {
      const r = await fetch("api/voice", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await errText(r, "That isn't a My Voice file."));
      const j = await r.json(); setVoice(j.voice, j.n); toast("Your voice is loaded");
    } catch (err) { toast(err.message || "Couldn't load that file"); }
  });

  /* ---------------- TALK ---------------- */
  const inp = $("expected");
  inp.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); inp.blur(); } });
  let fillTok = 0;
  function typeInto(text) {
    const tok = ++fillTok;
    if (reduced) { inp.value = text; return; }
    inp.value = ""; let i = 0;
    const step = () => { if (tok !== fillTok) return; inp.value = text.slice(0, ++i); if (i < text.length) setTimeout(step, 16 + Math.random() * 22); };
    step();
  }
  $("testBtn").addEventListener("click", () => {
    if (!S.test.length) return;
    S.testK = (S.testK + 1) % S.test.length; typeInto(S.test[S.testK]);
    $("testBtn").setAttribute("aria-label", `Test phrase ${S.testK + 1} of ${S.test.length}`);
    announce(`Test phrase: ${S.test[S.testK]}`);
  });

  const talkR = makeRec($("talkWrap"), {
    ariaIdle: "Start talking",
    onStart() {},
    onTick(s) { $("talkUnder").textContent = `listening · ${clock(s)}`; },
    onFail(msg) { $("talkUnder").textContent = msg; announce(msg); },
    async onBlob({ blob, seconds, env }) {
      const st = $("talkUnder"), wrap = $("talkWrap");
      if (seconds < 0.5) { st.textContent = "Too short. Try again."; announce("Too short. Try again."); return; }
      S.busy = true; $("talkMic").disabled = true; wrap.classList.add("busy");
      st.textContent = "Listening back…"; announce("Listening back.");
      try {
        const fd = new FormData();
        fd.append("audio", blob, "talk.wav"); fd.append("voice", S.voice || ""); fd.append("expected", inp.value.trim());
        const r = await fetch("api/transcribe", { method: "POST", body: fd });
        if (!r.ok) throw new Error(await errText(r, "Couldn't understand that recording."));
        renderResult(await r.json(), env);
        st.textContent = "Tap to talk";
      } catch (e) { st.textContent = "Tap and try again"; toast((e && e.message) || "Couldn't reach the server"); }
      finally { S.busy = false; $("talkMic").disabled = false; wrap.classList.remove("busy"); }
    },
  });

  // card waveforms (SVG, drawn from the loudness shape of the take)
  const W_ = 340;
  function sampleEnv(env, u) {
    if (!env || !env.length) return 0;
    const f = u * (env.length - 1), i = Math.floor(f), t = f - i;
    return env[i] * (1 - t) + (env[Math.min(env.length - 1, i + 1)] || 0) * t;
  }
  function regWaveSVG(env) {
    const cy = 28; let d = "";
    for (let x = 0; x <= W_; x += 1.25) {
      const u = x / W_, e = sampleEnv(env, u) * Math.min(1, Math.min(u, 1 - u) * 9);
      const y = cy + e * 22 * Math.sin(x * .62 + Math.sin(x * .11) * 2.4) * (.55 + .45 * Math.sin(x * 1.9 + 1));
      d += (x ? "L" : "M") + x.toFixed(2) + " " + y.toFixed(2);
    }
    return `<path d="${d}" fill="none" stroke="#A3A3A8" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>`;
  }
  function mineWaveSVG(env) {
    const cy = 32, top = [], bot = [];
    for (let x = 0; x <= W_; x += 2) {
      const u = x / W_, e = Math.pow(sampleEnv(env, u), 1.1) * Math.min(1, Math.min(u, 1 - u) * 7);
      const lump = Math.pow(Math.abs(Math.sin(x * Math.PI / 41 + .4)), .7);
      const h = .6 + e * 27 * (.3 + .7 * lump);
      top.push(`${x} ${(cy - h).toFixed(2)}`); bot.unshift(`${x} ${(cy + h * .82).toFixed(2)}`);
    }
    const d = "M" + top.join("L") + "L" + bot.join("L") + "Z";
    return `<defs><linearGradient id="mg" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#F07A2E" stop-opacity=".15"/><stop offset=".22" stop-color="#F07A2E" stop-opacity=".75"/>
      <stop offset=".5" stop-color="#EE6F22"/><stop offset=".78" stop-color="#F07A2E" stop-opacity=".75"/><stop offset="1" stop-color="#F07A2E" stop-opacity=".15"/></linearGradient>
      <filter id="mgl" x="-10%" y="-60%" width="120%" height="220%"><feGaussianBlur stdDeviation="5"/></filter></defs>
      <line x1="0" y1="${cy}" x2="${W_}" y2="${cy}" stroke="#F07A2E" stroke-opacity=".5" stroke-width="1" vector-effect="non-scaling-stroke"/>
      <g class="reveal" id="mineReveal"><path d="${d}" fill="#F5934F" opacity=".55" filter="url(#mgl)"/><path d="${d}" fill="url(#mg)"/></g>`;
  }
  function flatMine() {
    return `<line x1="0" y1="32" x2="${W_}" y2="32" stroke="#F07A2E" stroke-opacity=".45" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
  }

  function renderTalkEmpty() {
    $("regText").innerHTML = `<span class="lab">Regular:</span> <span class="ph">what regular tools hear</span>`;
    $("mineText").innerHTML = `<span class="lab">My Voice:</span> <span class="ph">tap the mic and talk</span>`;
    $("regErr").textContent = ""; $("mineErr").textContent = "";
    $("regWave").innerHTML = regWaveSVG(null); $("mineWave").innerHTML = flatMine();
    for (const id of ["copyBtn", "speakBtn", "shareBtn"]) $(id).disabled = true;
  }
  const words = (marks, raw) => marks ? marks.filter(m => m.s !== "missing").map(m => ({ w: m.w, x: m.s === "wrong" })) :
    String(raw || "").trim().split(/\s+/).filter(Boolean).map(w => ({ w, x: false }));
  const wordsHTML = (ws, upto) => {
    let left = upto == null ? Infinity : upto, h = "";
    for (let k = 0; k < ws.length && left > 0; k++) {
      const w = ws[k].w.slice(0, left); left -= ws[k].w.length + 1;
      h += (k ? " " : "") + (ws[k].x ? `<span class="x">${esc(w)}</span>` : esc(w));
    }
    return h;
  };
  let resTok = 0;
  function countTo(el, from, to, ms, suffix) {
    if (reduced || from === to) { el.textContent = to + suffix; return; }
    const t0 = performance.now(), tok = resTok;
    const f = now => { if (tok !== resTok) return; const p = Math.min(1, (now - t0) / ms); el.textContent = Math.round(from + (to - from) * ease(p)) + suffix; if (p < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  }
  function renderResult(res, env) {
    const tok = ++resTok, later = (ms, fn) => setTimeout(() => { if (tok === resTok) fn(); }, ms);
    const scored = res.expected != null && Array.isArray(res.regular_marks);
    const rw = words(scored ? res.regular_marks : null, res.regular), mw = words(scored ? res.myvoice_marks : null, res.myvoice);
    S.last = { regular: res.regular || "", myvoice: res.myvoice || "" };
    for (const id of ["copyBtn", "speakBtn", "shareBtn"]) $(id).disabled = !S.last.myvoice;
    const regText = $("regText"), mineText = $("mineText"), regErr = $("regErr"), mineErr = $("mineErr");
    regText.innerHTML = `<span class="lab">Regular:</span> ${rw.length ? wordsHTML(rw) : "<span class=\"ph\">(nothing heard)</span>"}`;
    regErr.textContent = scored ? `${res.regular_wer}% word errors` : "Type what you'll say to score both.";
    $("regWave").innerHTML = regWaveSVG(env);
    $("mineWave").innerHTML = mineWaveSVG(env);
    $("resultSr").textContent = `Regular voice tools heard: ${res.regular || "nothing"}. My Voice heard: ${res.myvoice || "nothing"}.` +
      (scored ? ` Word errors: regular ${res.regular_wer} percent, My Voice ${res.myvoice_wer} percent.` : "");
    const lab = `<span class="lab">My Voice:</span> `, total = mw.reduce((s, w) => s + w.w.length + 1, 0);
    const finish = () => { regText.querySelectorAll(".x").forEach(e => e.classList.add("on")); mineText.querySelectorAll(".x").forEach(e => e.classList.add("on")); };
    if (reduced) {
      mineText.innerHTML = lab + (mw.length ? wordsHTML(mw) : "<span class=\"ph\">(nothing heard)</span>");
      mineErr.textContent = scored ? `${res.myvoice_wer}% word errors` : ""; finish(); return;
    }
    const g = $("mineReveal"); g.style.clipPath = "inset(0 100% 0 0)"; g.style.transform = "scaleY(.35)"; g.style.transformBox = "fill-box"; g.style.transformOrigin = "50% 50%";
    mineText.innerHTML = lab + `<span class="caret"></span>`; mineErr.textContent = "";
    regText.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 420, easing: "cubic-bezier(.22,.8,.24,1)" });
    const xs = [...regText.querySelectorAll(".x")];
    xs.forEach((e, k) => later(420 + k * 170, () => e.classList.add("on")));
    const t1 = 420 + xs.length * 170 + 380;
    later(t1, () => {
      g.style.transition = "clip-path 1.5s cubic-bezier(.22,.8,.24,1), transform 1.5s cubic-bezier(.22,.8,.24,1)";
      g.style.clipPath = "inset(0 0 0 0)"; g.style.transform = "none";
      if (scored) countTo(mineErr, res.regular_wer, res.myvoice_wer, 1200, "% word errors");
      if (!mw.length) { mineText.innerHTML = lab + "<span class=\"ph\">(nothing heard)</span>"; return; }
      let i = 0;
      const step = () => {
        if (tok !== resTok) return;
        i = Math.min(total, i + 1);
        mineText.innerHTML = lab + wordsHTML(mw, i) + (i < total ? `<span class="caret"></span>` : "");
        if (i < total) setTimeout(step, 26 + Math.random() * 26);
        else later(150, finish);
      };
      step();
    });
  }
  function cap1(s) { s = String(s || "").trim(); return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
  async function copyText(t) {
    try { await navigator.clipboard.writeText(t); return true; } catch (e) {
      try { const ta = document.createElement("textarea"); ta.value = t; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select(); const ok = document.execCommand("copy"); ta.remove(); return ok; } catch (e2) { return false; }
    }
  }
  $("copyBtn").addEventListener("click", async () => { if (!S.last) return; toast(await copyText(cap1(S.last.myvoice)) ? "Copied" : "Couldn't copy"); });
  $("speakBtn").addEventListener("click", () => {
    if (!S.last) return;
    const ss = window.speechSynthesis; if (!ss || !window.SpeechSynthesisUtterance) { toast("This browser can't read text aloud"); return; }
    if (ss.speaking) { ss.cancel(); return; }
    const u = new SpeechSynthesisUtterance(cap1(S.last.myvoice)); u.rate = 1; ss.speak(u);
  });
  $("shareBtn").addEventListener("click", async () => {
    if (!S.last) return;
    const text = cap1(S.last.myvoice);
    if (navigator.share) { try { await navigator.share({ text }); return; } catch (e) { if (e && e.name === "AbortError") return; } }
    toast(await copyText(text) ? "Copied to share" : "Couldn't share");
  });

  /* ---------------- demo mode (for screenshots) ---------------- */
  function fakeLevel() {
    const t0 = performance.now();
    return () => {
      const t = (performance.now() - t0) / 1000;
      const syl = Math.pow(Math.max(0, Math.sin(t * 2 * Math.PI * 2.2)), 1.3) * (0.6 + 0.4 * Math.sin(t * 2 * Math.PI * 0.31 + 1.2));
      return Math.min(1, syl * 0.75 + 0.1);
    };
  }
  function fakeVoice() {
    const lv = fakeLevel();
    return buf => { const a = lv() * .62; for (let i = 0; i < buf.length; i++) buf[i] = (Math.random() * 2 - 1) * a * (.35 + .65 * Math.abs(Math.sin(i * .021))); return buf; };
  }
  function demoEnv() {
    const syl = [[.12, .55], [.2, .8], [.3, .6], [.4, 1], [.5, .85], [.58, .95], [.68, .7], [.78, .75], [.88, .45]];
    return Array.from({ length: 72 }, (_, k) => { const u = k / 71; let s = 0; for (const [c, a] of syl) s += a * Math.exp(-Math.pow((u - c) / .045, 2)); return Math.min(1, s); });
  }
  async function runDemo(which) {
    $("example").hidden = false;
    if (!S.phrases.length) S.phrases = Array.from({ length: 30 }, () => "Turn right at the red light");
    S.recs = new Array(S.phrases.length).fill(null);
    const fake = new Blob([new Uint8Array(44)], { type: "audio/wav" });
    for (let k = 0; k < 12; k++) S.recs[k] = fake;
    S.phrases[12] = "Turn right at the red light"; S.i = 12;
    renderTeach(); renderVoice();
    if (which === "teach") {
      const wrap = $("teachWrap");
      wrap.classList.add("on"); $("teachMic").setAttribute("aria-label", "Stop recording");
      $("teachHint").textContent = "listening"; $("teachUnder").textContent = "Tap to stop · 0:04";
      meterStart(wrap, fakeLevel()); Wave.live = fakeVoice(); Wave.kick();
      teachR.on = false;
    }
    if (which === "training") {
      S.recs[12] = fake;
      S.train = { state: "running", p: 0, n: 13, seconds: null, err: "", eta: 130 }; etaSmooth = 130;
      setTab("training"); renderTeach(); setTrain(0.65, 130);
    }
    if (which === "done") {
      S.recs[12] = fake; S.voice = null;
      S.train = { state: "done", p: 1, n: 13, seconds: 288, err: "", eta: null };
      TP.shown = TP.target = 1; setTab("training"); renderTeach();
    }
    if (which === "talk") {
      S.voice = "demo"; S.voiceN = 30; renderVoice(); setTab("talk");
      inp.value = "Turn right at the red light";
      renderResult({
        expected: "Turn right at the red light", regular: "turn white at the wed light", myvoice: "turn right at the red light",
        regular_wer: 50, myvoice_wer: 0,
        regular_marks: [{ w: "turn", s: "ok" }, { w: "white", s: "wrong" }, { w: "right", s: "missing" }, { w: "at", s: "ok" }, { w: "the", s: "ok" },
                        { w: "wed", s: "wrong" }, { w: "red", s: "missing" }, { w: "light", s: "ok" }],
        myvoice_marks: ["turn", "right", "at", "the", "red", "light"].map(w => ({ w, s: "ok" })),
      }, demoEnv());
    }
  }

  /* ---------------- boot ---------------- */
  async function boot() {
    const demo = new URLSearchParams(location.search).get("demo");
    S.demo = demo;
    let info = null;
    try { const r = await fetch("api/info", { cache: "no-store" }); if (!r.ok) throw 0; info = await r.json(); }
    catch (e) { toast("Can't reach the My Voice server. Check your connection and reload.", 6000); }
    info = info || { phrases: [], test: [], min_to_train: 10 };
    S.phrases = info.phrases || []; S.test = info.test || []; S.min = info.min_to_train || 10;
    S.recs = new Array(S.phrases.length).fill(null);
    S.voice = store.get("myvoice.voice"); const n = parseInt(store.get("myvoice.n"), 10); S.voiceN = isFinite(n) ? n : null;
    renderTeach(); renderVoice(); renderTalkEmpty(); setTab("teach");
    if (demo) await runDemo(demo);
    else checkVoice();
    document.documentElement.dataset.ready = "1";
  }
  boot();
})();
