/* The dossier's background: one retinal vessel network laid under the whole
   document rather than a picture in the hero.

   - Entrance: a focusing reticle closes onto the optic disc, the disc ignites
     with a flash and a shockwave, and the network grows out of it.
   - The disc then lives: a slow heartbeat swells its glow, sends a ripple out
     and launches light along the arcades on every beat.
   - Colour is a radial gradient from each vessel's source: the fundus' amber
     at the disc, through rose and magenta to violet and cyan at the tips.
   - Further down, vessel systems emerge from the page edges at intervals —
     slanted trunks, fans from a disc just beyond the edge, and arcades that
     loop in and back out — so the network follows the reader without a
     rail running down the sides.
   - Two parallax depths, drawn additively, with a starfield and haze behind.
     A slow perfusion wave rolls down the grown vessels.
   - Pointer: an ophthalmoscope beam — nearby vessels brighten and throb, and
     light sparks off along the vessel nearest the cursor.
   - It stays background: dimmed through the reading column, capped alpha,
     adaptive quality, and a single static frame under reduced motion.

   Canvas 2D, no dependencies. Fully grown branches are cached as Path2D in
   layer coordinates; only branches intersecting the viewport are stroked. */
(function () {
  "use strict";
  const cv = document.getElementById("nerves");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches ||
                  !!(navigator.connection && navigator.connection.saveData);
  const FINE = matchMedia("(pointer: fine)").matches;

  let W = 0, VH = 0, DOC = 0, dpr = 1;
  let layers = [];
  let seed = 1;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) & 0x7fffffff) / 0x7fffffff;
  const now = () => performance.now() / 1000;
  const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
  const smooth = (a, b, v) => { const x = clamp01((v - a) / (b - a)); return x * x * (3 - 2 * x); };
  const easeOut = x => 1 - Math.pow(1 - clamp01(x), 3);

  // the entrance is timed from here; arriving through the page transition,
  // it waits for the curtain to withdraw
  const T0 = now();
  const CURTAIN = document.documentElement.classList.contains("dr-entering");
  const IGN = CURTAIN ? 1.75 : 0.9;              // seconds from T0 to the disc igniting
  const BEAT0 = 1.7, PERIOD = 3.4;               // heartbeat, measured from ignition

  // Pinned stretches of the page (the dossier's card rail) publish themselves as
  // window.drPins = [{start, len}] in document pixels. Inside one the page is
  // not moving, so neither is the background: the network reads the scroll
  // with every pinned stretch taken out, and resumes where it left off.
  const pins = () => Array.isArray(window.drPins) ? window.drPins : [];
  function scrolled() {
    let y = scrollY;
    for (const p of pins()) if (y > p.start) y -= Math.min(y - p.start, p.len);
    return y;
  }
  function docHeight() {
    let d = document.documentElement.scrollHeight;
    for (const p of pins()) d -= p.len;
    return d;
  }

  const started = new Map();             // root id → start time; survives regeneration, so growth carries on through it
  let quality = 2;                       // 2 full, 1 lighter, 0 minimal
  let ema = 8;

  // palettes: [distance from source 0..1, hue (unwrapped), lightness]
  const PAL = {
    fundus: [[0, 36, 68], [0.2, 18, 64], [0.4, -22, 63], [0.6, -62, 65], [0.8, -110, 67], [1, -160, 70]],
    violet: [[0, 272, 64], [0.5, 238, 66], [1, 192, 70]],
    orchid: [[0, 318, 62], [0.5, 282, 64], [1, 232, 68]],
    teal:   [[0, 182, 60], [0.5, 214, 64], [1, 262, 68]],
    ember:  [[0, 14, 62], [0.5, -34, 62], [1, -84, 66]]
  };
  function colourAt(pal, f) {
    f = clamp01(f);
    for (let i = 1; i < pal.length; i++) if (f <= pal[i][0]) {
      const a = pal[i - 1], b = pal[i], k = (f - a[0]) / (b[0] - a[0] || 1);
      return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
    }
    const z = pal[pal.length - 1]; return [z[1], z[2]];
  }
  // Soft round glows are drawn from small pre-rendered sprites (one per hue
  // band) rather than a fresh radial gradient per light per frame — the
  // single largest cost in the field.
  const sprites = new Map();
  function sprite(h, sat, l) {
    const band = ((((h % 360) + 360) % 360) / 10) | 0, key = band + ":" + sat + ":" + l;
    let c = sprites.get(key);
    if (!c) {
      c = document.createElement("canvas"); c.width = c.height = 64;
      const g = c.getContext("2d"), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32), hh = band * 10 + 5;
      gr.addColorStop(0, `hsla(${hh},${sat}%,${l}%,1)`); gr.addColorStop(1, `hsla(${hh},${sat}%,${l}%,0)`);
      g.fillStyle = gr; g.fillRect(0, 0, 64, 64); sprites.set(key, c);
    }
    return c;
  }
  function blob(h, sat, l, x, y, r, a) {
    if (a <= 0.003) return;
    ctx.globalAlpha = Math.min(1, a); ctx.drawImage(sprite(h, sat, l), x - r, y - r, 2 * r, 2 * r); ctx.globalAlpha = 1;
  }
  function hsl(h, s, l, a) { return `hsla(${(((h % 360) + 360) % 360).toFixed(0)},${s}%,${l.toFixed(0)}%,${a.toFixed(3)})`; }

  // ── generation ──────────────────────────────────────────────────────────
  function Layer(p, alpha) {
    return {p: REDUCED ? 1 : p, alpha, branches: [], roots: [], stars: [], dust: [], nebulae: [], glows: [], pulses: [], H: 0, disc: null, beatN: -1};
  }

  function branch(L, pts, cum, tot, w, depth, parent, src) {
    const xs = pts.filter((_, i) => !(i & 1)), ys = pts.filter((_, i) => i & 1), n = pts.length, o = src.o;
    const b = {pts, cum, len: tot, w, depth, parent, kids: [], src,
      f0: Math.hypot(pts[0] - o.x, pts[1] - o.y) / src.R, f1: Math.hypot(pts[n - 2] - o.x, pts[n - 1] - o.y) / src.R,
      x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys) - 4, y1: Math.max(...ys) + 4,
      start: Infinity, dur: Math.max(0.32, tot / (230 + w * 60)), path: null, grad: null};
    b.mid = colourAt(src.pal, (b.f0 + b.f1) / 2);
    L.branches.push(b);
    if (parent) parent.kids.push(b);
    return b;
  }

  // curl: a steady turn per step, which is what makes an arcade loop back
  function grow(L, x, y, ang, len, wide, depth, parent, src, curl) {
    if (depth <= 0 || len < 9) return null;
    const steps = 9, pts = [x, y], cum = [0];
    let cx = x, cy = y, a = ang, tot = 0;
    for (let i = 0; i < steps; i++) {
      a += (rnd() - 0.5) * 0.34 + (curl || 0);
      const s = len / steps;
      cx += Math.cos(a) * s; cy += Math.sin(a) * s; tot += s;
      pts.push(cx, cy); cum.push(tot);
    }
    const b = branch(L, pts, cum, tot, Math.max(0.55, wide), depth, parent, src);
    // keep the reading column quiet: a lineage that wanders into it thins out
    const centre = cx > W * 0.32 && cx < W * 0.68;
    const n = centre ? (rnd() > 0.72 ? 1 : 0) : depth > 3 ? 2 : (rnd() > 0.35 ? 2 : 1);
    for (let i = 0; i < n; i++) {
      const spread = (i === 0 ? -1 : 1) * (0.26 + rnd() * 0.42);
      grow(L, cx, cy, a + spread, len * (0.64 + rnd() * 0.16), wide * 0.68, depth - 1, b, src, curl ? curl * 0.4 : 0);
    }
    // occasional lesion-coloured points along the vessel, in the model's own key
    if (rnd() < 0.18) {
      const k = (rnd() * (pts.length / 2 - 1) | 0) * 2;
      const key = ["#FF5A4D", "#FF8A2B", "#F5D02E", "#FF63DC"][(rnd() * 4) | 0];
      L.dust.push({x: pts[k] + (rnd() - .5) * 26, y: pts[k + 1] + (rnd() - .5) * 26, r: 0.9 + rnd() * 1.8, c: key, ph: rnd() * 6.28, sp: .4 + rnd() * .8});
    }
    return b;
  }

  // a vessel system emerging from one edge: a slanted trunk, a fan from a disc
  // just beyond the edge, or an arcade that loops in and back out
  function system(L, y, side, far) {
    const left = side < 0, inward = left ? 0 : Math.PI, ex = left ? -24 : W + 24;
    const names = far ? ["violet", "orchid", "violet", "teal"] : ["violet", "orchid", "teal", "ember", "violet"];
    const pal = PAL[names[(rnd() * names.length) | 0]];
    const src = {pal, R: W * (far ? 0.28 : 0.34), o: {x: ex, y}};
    const roll = rnd(), roots = [];
    if (roll < (far ? 0.6 : 0.42)) {
      roots.push(grow(L, ex, y, inward + (rnd() - 0.5) * 1.3, W * (0.17 + rnd() * 0.1), far ? 1.4 : 2.4, 5, null, src));
    } else if (far || roll < 0.74) {
      const k = 3 + (rnd() < 0.5 ? 1 : 0);
      for (let i = 0; i < k; i++) {
        const a = inward - 0.85 + 1.7 * (i + 0.5) / k + (rnd() - 0.5) * 0.25;
        roots.push(grow(L, ex, y, a, W * (0.12 + rnd() * 0.08), far ? 1.1 : 1.9, far ? 4 : 5, null, src));
      }
    } else {
      const down = rnd() < 0.5, a0 = down ? -1.25 : 1.25, c = down ? 0.35 : -0.35;
      roots.push(grow(L, ex, y, left ? a0 : Math.PI - a0, W * (0.28 + rnd() * 0.08), 2.1, 5, null, src, left ? c : -c));
    }
    const rs = roots.filter(Boolean);
    rs.forEach(r => L.roots.push(r));
    if (rs.length) L.glows.push({x: left ? 0 : W, y, root: rs[0], pal, ph: rnd() * 6.28});
  }

  function build() {
    const oldH = layers.length ? layers[0].H : 0;
    layers = [Layer(0.52, 0.42), Layer(0.86, 1)];
    layers.forEach((L, li) => {
      seed = 20260923 + li * 7919;
      L.H = Math.max(VH, (DOC - VH) * L.p + VH);
      const near = li === 1;
      if (near) {
        // the hero's optic disc: arcades sweep in toward the macula, a few
        // shorter vessels leave toward the edge, one trunk carries on down the page
        const disc = {x: W * (W < 760 ? 0.86 : 0.9), y: VH * 0.42};
        L.disc = disc;
        const src = {pal: PAL.fundus, R: Math.max(W, VH) * 0.62, o: disc, hero: true};
        const base = Math.min(W, VH) * 0.3;
        [[-2.62, 1.1, 6], [2.62, 1.1, 6], [-2.05, 0.86, 6], [2.05, 0.86, 6], [3.14, 0.72, 6],
         [1.52, 0.95, 6], [-1.45, 0.62, 5], [-0.35, 0.42, 4], [0.4, 0.46, 4]].forEach(([a, k, dp], i) => {
          const r = grow(L, disc.x, disc.y, a, base * k, k < 0.5 ? 1.8 : 2.6, dp, null, src);
          if (r) { r.hero = true; r.delay = i * 0.05; L.roots.push(r); }
        });
        let y = VH * 1.05, side = rnd() < 0.5 ? -1 : 1, run = 0;
        while (y < L.H - 150) {
          system(L, y, side, false);
          const flip = run >= 1 || rnd() < 0.55;
          run = flip ? 0 : run + 1; if (flip) side = -side;
          y += 400 + rnd() * 380;
        }
      } else {
        let y = VH * 0.2, side = rnd() < 0.5 ? -1 : 1;
        while (y < L.H) {
          system(L, y, side, true);
          if (rnd() < 0.7) side = -side;
          y += 360 + rnd() * 320;
        }
        for (let k = 0; k < Math.ceil(L.H / 900); k++) {
          L.nebulae.push({x: (rnd() < .5 ? rnd() * .3 : .7 + rnd() * .3) * W, y: rnd() * L.H, r: 260 + rnd() * 380,
            h: [265, 305, 188, 225][(rnd() * 4) | 0], ph: rnd() * 6.28});
        }
      }
      // stars, sorted by y for cheap culling
      const n = Math.round(W * L.H / (near ? 11000 : 6500));
      for (let k = 0; k < n; k++) {
        const edge = rnd();
        const x = edge < .38 ? rnd() * W * .3 : edge < .76 ? W * .7 + rnd() * W * .3 : rnd() * W;
        L.stars.push({x, y: rnd() * L.H, r: .35 + rnd() * (near ? 1.05 : .8), ph: rnd() * 6.28, sp: .6 + rnd() * 1.8,
          c: rnd() < .12 ? "#7FF3EC" : rnd() < .08 ? "#FF9BEA" : "#DCE4F2"});
      }
      L.stars.sort((a, b) => a.y - b.y);
    });
    // what had started keeps its timing; anything above the fold that never started is simply grown
    layers.forEach((L, li) => L.roots.forEach((r, ri) => {
      const id = li + ":" + ri;
      if (REDUCED) schedule(r, -1e9, true);
      else if (started.has(id)) schedule(r, started.get(id), false);
    }));
    if (oldH && !REDUCED) {
      const top = scrolled();
      layers.forEach((L, li) => L.roots.forEach((r, ri) => {
        if (r.start === Infinity && r.y0 < top * L.p) { started.set(li + ":" + ri, -1e9); schedule(r, -1e9, true); }
      }));
    }
  }

  // growth: the disc's tree starts at ignition; a system starts when its
  // region reaches the viewport; each child sprouts as its parent nears the end
  function schedule(b, t, instant) {
    b.start = instant ? -1e9 : t;
    b.kids.forEach(k => schedule(k, instant ? -1e9 : t + b.dur * 0.82, instant));
  }
  function trigger(t) {
    const e = t - T0 - IGN;
    if (e < 0.05) return;
    layers.forEach((L, li) => {
      const top = scrolled() * L.p, bottom = top + VH * 0.92;
      L.roots.forEach((r, ri) => {
        if (r.start !== Infinity) return;
        if (r.hero || (e > 0.6 && r.y0 < bottom && r.y1 > top - VH * 0.2)) {
          started.set(li + ":" + ri, t + (r.delay || 0));
          schedule(r, t + (r.delay || 0), false);
        }
      });
    });
  }

  function pathOf(b, upTo) {
    const p = new Path2D(), pts = b.pts;
    p.moveTo(pts[0], pts[1]);
    if (upTo >= b.len) { for (let i = 2; i < pts.length; i += 2) p.lineTo(pts[i], pts[i + 1]); return p; }
    for (let i = 1; i < b.cum.length; i++) {
      if (b.cum[i] <= upTo) { p.lineTo(pts[i * 2], pts[i * 2 + 1]); continue; }
      const f = (upTo - b.cum[i - 1]) / (b.cum[i] - b.cum[i - 1]);
      p.lineTo(pts[i * 2 - 2] + (pts[i * 2] - pts[i * 2 - 2]) * f, pts[i * 2 - 1] + (pts[i * 2 + 1] - pts[i * 2 - 1]) * f);
      break;
    }
    return p;
  }
  function at(b, s) {                   // point at arc length s along a branch
    const c = b.cum, pts = b.pts;
    if (s <= 0) return [pts[0], pts[1]];
    for (let i = 1; i < c.length; i++) if (c[i] >= s) {
      const f = (s - c[i - 1]) / (c[i] - c[i - 1]);
      return [pts[i * 2 - 2] + (pts[i * 2] - pts[i * 2 - 2]) * f, pts[i * 2 - 1] + (pts[i * 2 + 1] - pts[i * 2 - 1]) * f];
    }
    return [pts[pts.length - 2], pts[pts.length - 1]];
  }
  // colour runs along each vessel with its distance from the source
  function gradOf(b) {
    if (b.grad) return b.grad;
    const p = b.pts, n = p.length, g = ctx.createLinearGradient(p[0], p[1], p[n - 2], p[n - 1]);
    for (const k of [0, 0.5, 1]) { const [h, l] = colourAt(b.src.pal, b.f0 + (b.f1 - b.f0) * k); g.addColorStop(k, hsl(h, 88, l, 1)); }
    return (b.grad = g);
  }

  // ── the disc ────────────────────────────────────────────────────────────
  function envelope(x) { return x < 0 ? 0 : (1 - Math.exp(-x / 0.05)) * Math.exp(-x / 0.3); }
  function beat(e) {                      // lub-dub, 0..1
    if (REDUCED || e < BEAT0) return 0;
    const p = (e - BEAT0) % PERIOD;
    return Math.min(1, envelope(p) * 1.6 + envelope(p - 0.28) * 0.95);
  }
  const beatIndex = e => REDUCED || e < BEAT0 ? -1 : Math.floor((e - BEAT0) / PERIOD);

  function discUnder(d, e, bt, bot) {
    if (d.y - 520 > bot) return;
    const k = (REDUCED ? 1 : smooth(-0.25, 0.7, e)) * (0.9 + 0.22 * bt);
    if (k <= 0) return;
    const R = Math.max(W, VH) * 0.5, g = ctx.createRadialGradient(d.x, d.y, 0, d.x, d.y, R);
    g.addColorStop(0, `rgba(150,70,40,${(0.3 * k).toFixed(3)})`);
    g.addColorStop(0.35, `rgba(90,35,90,${(0.1 * k).toFixed(3)})`);
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.fillRect(d.x - R, d.y - R, R * 2, R * 2);
  }

  function discOver(d, t, e, bt, bot) {
    if (d.y - 520 > bot) return;
    const ig = REDUCED ? 1 : smooth(-0.25, 0.7, e);
    const fl = REDUCED ? 0 : Math.exp(-Math.pow(e / 0.22, 2));
    // the focus: a reticle closing onto the disc before it lights, then idling round it
    const fIn = REDUCED ? 1 : easeOut((e + 0.95) / 1.05);
    if (fIn > 0) {
      const rr = (210 - 170 * fIn) * (1 + 0.035 * bt), rot = REDUCED ? 0.4 : t * 0.14;
      const ra = smooth(0, 0.15, fIn) * (0.15 + 0.4 * smooth(0, 0.25, fIn) * (e < 0 ? 1 : Math.exp(-e / 0.8)));
      ctx.strokeStyle = `rgba(62,214,206,${ra.toFixed(3)})`; ctx.lineWidth = 1;
      for (let k = 0; k < 4; k++) {
        const a = rot + k * Math.PI / 2;
        ctx.beginPath(); ctx.arc(d.x, d.y, rr, a + 0.3, a + Math.PI / 2 - 0.3); ctx.stroke();
        const b = -rot * 0.6 + k * Math.PI / 2 + Math.PI / 4;
        ctx.beginPath(); ctx.moveTo(d.x + Math.cos(b) * (rr + 5), d.y + Math.sin(b) * (rr + 5));
        ctx.lineTo(d.x + Math.cos(b) * (rr + 13), d.y + Math.sin(b) * (rr + 13)); ctx.stroke();
      }
      ctx.strokeStyle = `rgba(62,214,206,${(ra * 0.4).toFixed(3)})`;
      ctx.setLineDash([2, 7]); ctx.lineDashOffset = -t * 6;
      ctx.beginPath(); ctx.arc(d.x, d.y, rr * 1.6, 0, 6.3); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (ig <= 0 && fl < 0.01) return;
    // the shockwave when it lights
    if (!REDUCED && e > 0 && e < 2.4) {
      const p = e / 2.4, r = easeOut(p) * Math.max(W, VH) * 0.8, a = 0.32 * (1 - p) * (1 - p);
      ctx.beginPath(); ctx.arc(d.x, d.y, r, 0, 6.3);
      ctx.strokeStyle = hsl(32, 100, 72, a); ctx.lineWidth = 1.4; ctx.stroke();
      ctx.strokeStyle = hsl(300, 90, 62, a * 0.25); ctx.lineWidth = 12; ctx.stroke();
    }
    // a ripple on every beat
    if (!REDUCED && e > BEAT0) {
      const p = ((e - BEAT0) % PERIOD) / 1.9;
      if (p < 1) {
        ctx.beginPath(); ctx.arc(d.x, d.y, 22 + 170 * easeOut(p), 0, 6.3);
        ctx.strokeStyle = hsl(28 - 70 * p, 95, 70, 0.22 * (1 - p) * (1 - p)); ctx.lineWidth = 1.2; ctx.stroke();
      }
    }
    // the bloom
    const cr = 70 * (1 + 0.14 * bt + 0.9 * fl), g = ctx.createRadialGradient(d.x, d.y, 0, d.x, d.y, cr);
    g.addColorStop(0, `rgba(255,214,150,${Math.min(1, (0.42 + 0.22 * bt) * ig + 0.55 * fl).toFixed(3)})`);
    g.addColorStop(0.4, `rgba(230,140,80,${Math.min(1, (0.14 + 0.08 * bt) * ig + 0.2 * fl).toFixed(3)})`);
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(d.x, d.y, cr, 0, 6.3); ctx.fill();
    // a streak of light at the moment of ignition
    if (fl > 0.02) {
      const sw = 380 * (0.6 + fl), s = ctx.createLinearGradient(d.x - sw, 0, d.x + sw, 0);
      s.addColorStop(0, "rgba(255,220,170,0)"); s.addColorStop(0.5, `rgba(255,220,170,${(0.5 * fl).toFixed(3)})`); s.addColorStop(1, "rgba(255,220,170,0)");
      ctx.fillStyle = s; ctx.fillRect(d.x - sw, d.y - 1.2, sw * 2, 2.4);
    }
    // the disc itself: its rim, and the bright cup
    ctx.strokeStyle = `rgba(255,228,190,${(0.38 * ig * (0.8 + 0.4 * bt)).toFixed(3)})`; ctx.lineWidth = 1.1;
    ctx.beginPath(); ctx.ellipse(d.x, d.y, 13 * (1 + 0.06 * bt), 16 * (1 + 0.06 * bt), 0.12, 0, 6.3); ctx.stroke();
    ctx.fillStyle = `rgba(255,246,228,${Math.min(1, 0.75 * ig + fl).toFixed(3)})`;
    ctx.beginPath(); ctx.arc(d.x, d.y, 2.6 + 1.2 * bt, 0, 6.3); ctx.fill();
  }

  // ── drawing ─────────────────────────────────────────────────────────────
  let mx = -1e4, my = -1e4, lastMove = -1e9, lastSpark = 0, pxs = 0, pys = 0;
  const LR = 200;
  const lens = document.createElement("canvas"), lctx = lens.getContext("2d");
  lens.width = lens.height = LR * 2;

  function frame(t, dt) {
    const t0 = performance.now();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, VH);
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    const sy = scrolled(), e = REDUCED ? 99 : t - T0 - IGN;
    const heroFade = VH ? Math.max(0, 1 - sy / (VH * 0.9)) : 1;   // the hero may be a little brighter (VH is 0 in a hidden tab)
    const appear = REDUCED ? 1 : smooth(-IGN + 0.1, 1.2, e); // stars and haze arrive with the focus
    const bt = beat(e);

    // parallax: the layers lean away from the pointer (or drift, without one)
    if (!REDUCED) {
      const has = FINE && mx > -1e3;
      const tx = has ? mx / W - 0.5 : Math.sin(t * 0.07) * 0.3, ty = has ? my / VH - 0.5 : Math.cos(t * 0.05) * 0.3;
      const k = Math.min(1, dt * 2.5); pxs += (tx - pxs) * k; pys += (ty - pys) * k;
    }
    const moving = performance.now() < scrollingUntil;                // mid-scroll: keep the frame light
    const pointerOn = FINE && !REDUCED && quality > 0 && !moving && t - lastMove < 2.5;
    const lensK = pointerOn ? clamp01(1 - (t - lastMove - 1.5)) * (0.68 + 0.32 * Math.sin(t * 6.2)) : 0;

    layers.forEach((L, li) => {
      const off = sy * L.p, top = off - 80, bot = off + VH + 80;
      const sx = -pxs * (li ? 14 : 6), sh = -pys * (li ? 10 : 4);
      const cx = mx - sx, cy = my + off - sh;               // pointer in layer coordinates
      const la = L.alpha;
      ctx.save(); ctx.translate(sx, sh - off);

      // nebula haze
      if (quality > 0) L.nebulae.forEach(n => {
        const y = n.y + Math.sin(t * 0.05 + n.ph) * 30, x = n.x + Math.cos(t * 0.04 + n.ph) * 40;
        if (y + n.r < top || y - n.r > bot) return;
        blob(n.h, 80, 50, x, y, n.r, 0.055 * appear);
      });

      // light where a system comes in from the edge
      L.glows.forEach(gl => {
        if (gl.y + 200 < top || gl.y - 200 > bot || gl.root.start === Infinity) return;
        const k = REDUCED ? 1 : clamp01((t - gl.root.start) / 1.2);
        if (k <= 0) return;
        blob(colourAt(gl.pal, 0)[0], 85, 58, gl.x, gl.y, 190, (li ? 0.12 : 0.07) * k * (0.75 + 0.25 * Math.sin(t * 0.9 + gl.ph)));
      });

      if (L.disc) discUnder(L.disc, e, bt, bot);

      // stars
      const S = L.stars;
      let lo = 0, hi = S.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (S[m].y < top) lo = m + 1; else hi = m; }
      const step = quality === 0 ? 3 : quality === 1 ? 2 : 1;
      for (let i = lo; i < S.length && S[i].y < bot; i += step) {
        const s = S[i], tw = REDUCED ? .7 : .45 + .55 * Math.sin(t * s.sp + s.ph);
        ctx.globalAlpha = Math.max(0, tw) * (li ? .75 : .55) * appear;
        ctx.fillStyle = s.c; ctx.fillRect(s.x - s.r, s.y - s.r, 2 * s.r, 2 * s.r);   // squares: at this size, indistinguishable
      }
      ctx.globalAlpha = 1;

      // vessels
      const visible = [], growing = [];
      for (const b of L.branches) {
        if (b.y1 < top || b.y0 > bot || b.start === Infinity) continue;
        const prog = REDUCED ? 1 : Math.min(1, (t - b.start) / b.dur);
        if (prog <= 0) continue;
        const heroBoost = b.src.hero ? 0.55 + 0.45 * heroFade : 1;
        // perfusion: a slow band of brightness rolls down the network, like a pulse wave
        const wave = REDUCED ? 1 : 0.74 + 0.26 * Math.sin(t * 0.8 - b.y0 / 230);
        const a = Math.min(0.6, 0.12 + b.depth * 0.07) * la * heroBoost * wave;
        const path = prog >= 1 ? (b.path || (b.path = pathOf(b, b.len))) : pathOf(b, b.len * prog);
        ctx.strokeStyle = gradOf(b);
        if (!moving && (quality > 0 || li === 1)) { ctx.globalAlpha = a * 0.16; ctx.lineWidth = b.w * 5.5; ctx.stroke(path); }
        ctx.globalAlpha = a; ctx.lineWidth = b.w; ctx.stroke(path);
        if (prog >= 1) visible.push(b); else growing.push([b, prog, a]);
      }
      ctx.globalAlpha = 1;

      // a growing vessel carries a spark at its tip
      growing.forEach(([b, prog, a]) => {
        const [x, y] = at(b, b.len * prog), [h] = colourAt(b.src.pal, b.f0 + (b.f1 - b.f0) * prog);
        blob(h, 100, 82, x, y, 6 + b.w * 3, a * 1.6);
      });

      // lesion-key dust
      L.dust.forEach(d => {
        if (d.y < top || d.y > bot) return;
        ctx.globalAlpha = (REDUCED ? .5 : .35 + .35 * Math.sin(t * d.sp + d.ph)) * la * appear;
        ctx.fillStyle = d.c; ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 6.3); ctx.fill();
      });
      ctx.globalAlpha = 1;

      // vessels near the pointer
      const near = li === 1 && pointerOn
        ? visible.filter(b => b.x1 > cx - LR && b.x0 < cx + LR && b.y1 > cy - LR && b.y0 < cy + LR) : [];

      // light travelling along the vessels
      if (!REDUCED && visible.length) {
        const want = moving ? (li ? 5 : 1) : li ? (quality === 2 ? 14 : quality === 1 ? 8 : 4) : (quality === 2 ? 5 : 2);
        let ambient = 0; for (const p of L.pulses) if (!p.kind) ambient++;
        while (ambient < want) {
          let b = visible[(rnd() * visible.length) | 0];
          for (let k = 0; k < 4; k++) { const c = visible[(rnd() * visible.length) | 0]; if (c.w > b.w) b = c; }
          L.pulses.push({b, s: rnd() * b.len * .3, v: 90 + rnd() * 140, kind: 0}); ambient++;
        }
        // every beat of the disc sends light out along the arcades
        if (L.disc) {
          const n = beatIndex(e);
          if (n !== L.beatN) {
            L.beatN = n;
            if (n >= 0 && quality > 0) L.roots.forEach(r => { if (r.hero && t - r.start > r.dur) L.pulses.push({b: r, s: 0, v: 210 + rnd() * 90, kind: 1}); });
          }
        }
        // and light sparks off along the vessel nearest the pointer
        if (near.length && t - lastMove < 0.4 && t - lastSpark > 0.11) {
          let best = null, bs = 0, bd = 110 * 110;
          for (const b of near) for (let i = 0; i < b.pts.length; i += 2) {
            const dx = b.pts[i] - cx, dy = b.pts[i + 1] - cy, dd = dx * dx + dy * dy;
            if (dd < bd) { bd = dd; best = b; bs = b.cum[i >> 1]; }
          }
          let sparks = 0; for (const p of L.pulses) if (p.kind === 2) sparks++;
          if (best && sparks < 14) { L.pulses.push({b: best, s: bs, v: 150 + rnd() * 100, kind: 2}); lastSpark = t; }
        }
        L.pulses = L.pulses.filter(p => {
          p.s += p.v * dt;
          if (p.s > p.b.len) {
            const kids = p.b.kids.filter(k => t - k.start > k.dur);
            if (!kids.length) return false;
            p.b = kids[(rnd() * kids.length) | 0]; p.s = 0;
          }
          if (p.b.y1 < top || p.b.y0 > bot) return false;
          const [h] = colourAt(p.b.src.pal, p.b.f0 + (p.b.f1 - p.b.f0) * (p.s / p.b.len));
          const big = p.kind ? 1.35 : 1;
          for (let k = 4; k >= 1; k--) {
            const [x, y] = at(p.b, p.s - k * 8), f = 1 - k / 5, rr = (0.7 + 1.5 * f) * big;
            ctx.fillStyle = hsl(h, 95, 78, Math.min(1, 0.5 * f * f * la * big));
            ctx.fillRect(x - rr, y - rr, 2 * rr, 2 * rr);
          }
          const [hx, hy] = at(p.b, p.s);
          blob(h, 100, 84, hx, hy, 9 * big, 0.55 * la * big);
          return true;
        });
      }

      if (L.disc) discOver(L.disc, t, e, bt, bot);

      // the ophthalmoscope beam: nearby vessels brighten and throb, faded at the rim
      if (li === 1 && lensK > 0.01) {
        if (near.length) {
          lctx.setTransform(1, 0, 0, 1, 0, 0);
          lctx.globalCompositeOperation = "source-over";
          lctx.clearRect(0, 0, LR * 2, LR * 2);
          lctx.globalCompositeOperation = "lighter";
          lctx.setTransform(1, 0, 0, 1, LR - cx, LR - cy);
          lctx.lineCap = "round"; lctx.lineJoin = "round";
          near.forEach(b => {
            const p = b.path || pathOf(b, b.len), [h] = b.mid;
            lctx.strokeStyle = hsl(h, 100, 70, .3); lctx.lineWidth = b.w * 5; lctx.stroke(p);
            lctx.strokeStyle = hsl(h, 100, 80, .95); lctx.lineWidth = b.w * 1.7; lctx.stroke(p);
          });
          lctx.setTransform(1, 0, 0, 1, 0, 0);
          lctx.globalCompositeOperation = "destination-in";
          const g = lctx.createRadialGradient(LR, LR, 0, LR, LR, LR);
          g.addColorStop(0, `rgba(0,0,0,${(.85 * lensK).toFixed(3)})`); g.addColorStop(1, "rgba(0,0,0,0)");
          lctx.fillStyle = g; lctx.fillRect(0, 0, LR * 2, LR * 2);
          ctx.drawImage(lens, cx - LR, cy - LR);
        }
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, LR);
        g.addColorStop(0, `rgba(62,214,206,${(0.04 * lensK).toFixed(3)})`); g.addColorStop(1, "rgba(62,214,206,0)");
        ctx.fillStyle = g; ctx.fillRect(cx - LR, cy - LR, LR * 2, LR * 2);
      }
      ctx.restore();
    });

    // keep it background: quieter through the reading column, more so below the hero
    // (on a phone the whole width is the reading column, so only the edges stay bright)
    ctx.globalCompositeOperation = "destination-out";
    const narrow = W < 760;
    const m = narrow ? 0.46 + 0.28 * (1 - heroFade) : 0.28 + 0.34 * (1 - heroFade);
    const [c0, c1, c2, c3] = narrow ? [0.02, 0.1, 0.9, 0.98] : [0, 0.16, 0.62, 0.86];
    const g = ctx.createLinearGradient(0, 0, W, 0);
    g.addColorStop(c0, "rgba(0,0,0,0)"); g.addColorStop(c1, `rgba(0,0,0,${m})`);
    g.addColorStop(c2, `rgba(0,0,0,${m})`); g.addColorStop(c3, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, VH);
    ctx.globalCompositeOperation = "source-over";

    // adapt to the machine: shed layers of detail if a frame costs too much
    ema = ema * 0.92 + (performance.now() - t0) * 0.08;
    if (ema > 9 && quality > 0) { quality--; ema = 6; }
    else if (ema < 3.5 && quality < 2) { quality++; ema = 6; }
  }

  // ── lifecycle ───────────────────────────────────────────────────────────
  function size() {
    dpr = 1;                                   // a background of soft light needs no retina pixels
    const w = innerWidth, h = innerHeight, d = docHeight();
    const rebuild = w !== W || Math.abs(d - DOC) > 160;
    W = w; VH = h; DOC = d;
    cv.width = Math.round(W * dpr); cv.height = Math.round(VH * dpr);
    if (rebuild) build();
  }
  let last = null, scrollingUntil = 0;
  addEventListener("scroll", () => { scrollingUntil = performance.now() + 180; }, {passive: true});
  function loop(ts) {
    if (last !== null && ts - last * 1000 < 31) { requestAnimationFrame(loop); return; }   // 30 fps is ample here
    const t = ts / 1000, dt = last === null ? 0.033 : Math.min(0.066, t - last);
    last = t;
    trigger(t);
    frame(t, dt);
    requestAnimationFrame(loop);
  }
  size();
  if (REDUCED) {
    const paint = () => frame(now(), 0);
    paint();
    addEventListener("scroll", () => requestAnimationFrame(paint), {passive: true});
    addEventListener("resize", () => { size(); paint(); });
  } else {
    // a first frame straight away, so the field is there even if frames are throttled
    trigger(now()); frame(now(), 0);
    requestAnimationFrame(loop);
    let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(size, 150); });
    if (FINE) addEventListener("pointermove", e => { mx = e.clientX; my = e.clientY; lastMove = now(); }, {passive: true});
  }
  // the document grows when a run's results open, so the network must too
  if ("ResizeObserver" in window) new ResizeObserver(() => {
    const d = docHeight();
    if (Math.abs(d - DOC) > 160) { size(); if (REDUCED) frame(now(), 0); }
  }).observe(document.body);
  // the canvas comes into focus as the reticle closes (CSS: blur → sharp)
  setTimeout(() => cv.classList.add("lit"), REDUCED ? 0 : CURTAIN ? 700 : 30);
  setTimeout(() => cv.classList.add("settled"), REDUCED ? 0 : CURTAIN ? 3400 : 2600);
})();
