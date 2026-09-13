"""The Build page's "theater": the whole construction as one full-screen player.

The player shows the *finished* tree's drawing and reveals it step by step:
nodes the learner has not reached are hidden, nodes waiting in the queue are
drawn as grey "pending" boxes, the node being processed is outlined, and the
camera zooms to whatever is visible.  Because the layout is the final one from
the start, nodes never jump around as the tree grows — and only one drawing has
to be sent to the browser however many steps there are.

Each frame also carries a one-line caption saying what is being computed and
the formula for it, rendered under the tree like subtitles.  Only the formula
that *this* phase computes is shown; the full working stays on the page below.
Playback, pausing, stepping, scrubbing and full-screen all happen in the browser.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

from ..core import latex as tex
from ..core.criteria import get_impurity
from ..core.playback import (
    PHASE_CHOICE,
    PHASE_IMPURITY,
    PHASE_SCORE,
    PHASE_STOP,
    PHASE_WORKING,
    Cursor,
    advance,
    max_phase,
)
from ..core.trace import ACTION_SPLIT, STOP_REASONS, BuildStep, TrainingTrace
from .tree_dot import RenderOptions, TreeDotRenderer, dot_to_svg, has_graphviz_binary

MAX_SCOREBOARD = 6
MAX_ARGMAX = 4
THEATER_OPTIONS = RenderOptions(orientation="TB", show_node_ids=True,
                                show_distribution_bar=True, font_size=12)

PHASE_NAMES = {PHASE_IMPURITY: "impurity", PHASE_SCORE: "score", PHASE_WORKING: "working",
               PHASE_CHOICE: "choice"}


@dataclass
class Frame:
    """One tick of the player."""

    step: int            # 1-based, for display
    node_id: int
    depth: int
    phase: str           # impurity | score | working | choice | stop
    state: int           # index into the reveal-state list
    title: str
    what: str
    here: str            # why this node exists: the path from the root
    lines: List[str] = field(default_factory=list)   # LaTeX, one formula each
    final: bool = False
    banner: str = ""


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #
def _num(value: float) -> str:
    return f"{value:.3f}"


def _score_lhs(candidate, score_name: str) -> str:
    key = "gain_ratio" if score_name == "Gain Ratio" and "gain_ratio" in candidate.computations else "gain"
    computation = candidate.computations.get(key)
    if computation is not None and "=" in computation.symbolic:
        return computation.lhs
    return tex.text(candidate.feature_name)


def _argmax_latex(ranked, chosen, score_name: str, top: int = MAX_ARGMAX) -> str:
    """``A* = argmax {Outlook: 0.32, Humidity: 0.26, …} ⇒ Outlook`` — short enough to read."""
    entries = [r"%s{:}\ %s" % (tex.text(c.feature_name), _num(c.score)) for c in ranked[:top]]
    if len(ranked) > top:
        entries.append(r"\ldots")
    return (r"A^{*} = \arg\max_{A}\ %s(S, A) = \arg\max\left\{%s\right\} \Rightarrow %s"
            % (tex.escape(score_name), r",\ ".join(entries), tex.text(chosen.feature_name)))


def _here(step: BuildStep) -> str:
    if not step.path_conditions:
        return f"Root — all {step.n_samples:g} training samples start here."
    counts = f" ({step.counts_text()})" if step.class_counts else ""
    return f"Here because {' and '.join(step.path_conditions)} — {step.n_samples:g} samples{counts}."


def _frame_text(step: BuildStep, phase: int, trace: TrainingTrace) -> Tuple[str, str, List[str]]:
    """``(title, what, lines)`` for one phase of one step."""
    impurity = get_impurity(trace.impurity_name)
    score = trace.score_name

    if phase == PHASE_IMPURITY:
        return (
            f"Node #{step.node_id} · how impure is it?",
            f"{impurity.display} of the samples in this node — the baseline every split will be "
            f"measured against.",
            [step.impurity_computation.full_latex()],
        )

    if step.action != ACTION_SPLIT:  # PHASE_STOP
        reason = STOP_REASONS.get(step.stop_reason or "", step.stop_reason or "no split was possible")
        # A pure node has no separate rule to show: its impurity of 0 is the evidence.
        evidence = step.stop_computation if step.stop_computation is not None else step.impurity_computation
        return (
            f"Node #{step.node_id} becomes a leaf → predicts {step.prediction}",
            f"It stops because {reason}.",
            [evidence.full_latex()],
        )

    ranked = [c for c in step.ranked_candidates if c.is_valid]
    rejected = sum(1 for c in step.candidates if not c.is_valid)
    chosen = step.chosen

    if phase == PHASE_SCORE:
        shown = ranked[:MAX_SCOREBOARD]
        what = f"{len(ranked)} usable candidate(s), best first"
        if len(ranked) > len(shown):
            what += f" (top {len(shown)} shown)"
        if rejected:
            what += f"; {rejected} rejected"
        return (
            f"Score every candidate split by {score}",
            what + ".",
            [f"{_score_lhs(c, score)} = {_num(c.score)}" for c in shown],
        )

    if phase == PHASE_WORKING:
        comps = chosen.computations
        keys = ["gain"] + (["split_info", "gain_ratio"] if score == "Gain Ratio" else [])
        what = "Parent impurity minus the size-weighted impurity of the branches."
        if score == "Gain Ratio":
            what += " Then divide by the split information."
        return (
            f"The working for {chosen.describe()}",
            what,
            [comps[k].full_latex() for k in keys if k in comps],
        )

    # PHASE_CHOICE
    labels = ", ".join(str(v).strip() for v in chosen.branch_labels.values())
    return (
        f"Split node #{step.node_id} on {chosen.feature_name}",
        f"{chosen.feature_name} has the highest {score} ({_num(chosen.score)}) → "
        f"{chosen.n_branches} branches: {labels}. Each child joins the queue.",
        [_argmax_latex(ranked, chosen, score)],
    )


# --------------------------------------------------------------------------- #
# Frames and reveal states
# --------------------------------------------------------------------------- #
def reveal_states(trace: TrainingTrace) -> List[Dict[str, List[int]]]:
    """Which nodes are decided / queued after each step.

    Index 0 is "before anything happened" (only the root, still pending); index
    ``k + 1`` is the state right after step ``k`` finished.
    """
    nodes = trace.tree.nodes_by_id
    processed: List[int] = []
    states: List[Dict[str, List[int]]] = []

    def snapshot() -> Dict[str, List[int]]:
        done = set(processed)
        pending = [child.node_id for pid in processed
                   for child in nodes[pid].children.values() if child.node_id not in done]
        if not processed:
            pending = [trace.tree.root.node_id]
        return {"processed": list(processed), "pending": pending}

    states.append(snapshot())
    for step in trace.steps:
        processed.append(step.node_id)
        states.append(snapshot())
    return states


def build_frames(trace: TrainingTrace) -> List[Frame]:
    """All frames in playback order."""
    frames: List[Frame] = []
    if len(trace) == 0:
        return frames

    cursor = Cursor(0, 0)
    while True:
        step = trace.steps[cursor.step]
        deciding = cursor.phase >= max_phase(step)
        title, what, lines = _frame_text(step, cursor.phase, trace)
        phase_name = "stop" if (step.action != ACTION_SPLIT and cursor.phase == PHASE_STOP) \
            else PHASE_NAMES[cursor.phase]
        frames.append(Frame(
            step=cursor.step + 1, node_id=step.node_id, depth=step.depth, phase=phase_name,
            state=cursor.step + 1 if deciding else cursor.step,
            title=title, what=what, here=_here(step), lines=lines,
        ))
        cursor, finished = advance(cursor, trace)
        if finished:
            break

    frames[-1].final = True
    if trace.tree is not None:
        stats = trace.tree.stats()
        frames[-1].banner = (f"Tree complete — {stats['nodes']} nodes, {stats['leaves']} leaves, "
                             f"depth {stats['depth']}")
    return frames


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _svg_body(svg: bytes) -> str:
    """Drop the XML prologue so the SVG can be dropped straight into the DOM."""
    text = svg.decode("utf-8", errors="replace")
    start = text.find("<svg")
    return text[start:] if start >= 0 else text


def theater_payload(trace: TrainingTrace, key: str, prefer_svg: bool = True) -> dict:
    dot = TreeDotRenderer(THEATER_OPTIONS).render(trace.tree)
    svg = dot_to_svg(dot) if prefer_svg and has_graphviz_binary() else None
    return {
        "key": key,
        "algorithm": trace.algorithm,
        "n_steps": len(trace),
        "tree": {"svg": _svg_body(svg)} if svg else {"dot": dot},
        "nodes": {str(n.node_id): {"samples": f"{n.n_samples:g}"} for n in trace.tree.root.iter_nodes()},
        "states": reveal_states(trace),
        "frames": [asdict(f) for f in build_frames(trace)],
    }


def theater_html(trace: TrainingTrace, key: str, prefer_svg: bool = True) -> str:
    payload = theater_payload(trace, key, prefer_svg=prefer_svg)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
  :root { --tex:17px; --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --muted:#5f5f5c; --line:#e4e2dd;
          --accent:#2a78d6; --accent-ink:#ffffff; --bar:#1d1d1b; --bar-ink:#f4f4f2; --warn:#c0392b;
          --pending-fill:#EFEFEF; --pending-line:#B8B8B4; --pending-ink:#8a8a86; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--ink);
              font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
  #app { display:flex; flex-direction:column; height:100vh; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:var(--bg); }
  #top { display:flex; align-items:center; gap:12px; padding:8px 14px; border-bottom:1px solid var(--line); background:var(--panel); font-size:13px; color:var(--muted); }
  #top b { color:var(--ink); font-weight:600; }
  #top .spacer { flex:1; }
  #stage { flex:1; min-height:120px; position:relative; overflow:hidden; }
  #stage svg { display:block; width:100%; height:100%; }
  #stage .placeholder { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:14px; }
  #banner { position:absolute; top:12px; left:50%; transform:translateX(-50%); background:#1f7a4d; color:#fff; padding:6px 14px; border-radius:999px; font-size:13px; font-weight:600; display:none; }
  #caption { border-top:1px solid var(--line); background:var(--panel); padding:10px 16px 8px; max-height:42%; overflow:auto; }
  #here { font-size:12px; color:var(--muted); margin:0 0 4px; }
  #title { font-size:17px; font-weight:650; margin:0 0 2px; }
  #what { font-size:13px; color:var(--muted); margin:0 0 6px; }
  #lines { display:flex; flex-direction:column; gap:4px; }
  .tex { font-size:var(--tex); overflow-x:auto; overflow-y:hidden; white-space:nowrap; padding:2px 0 2px 12px; }
  .tex .katex-display { margin:0; }
  .tex .katex-display > .katex { text-align:left; }
  .tex.raw { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12px; color:var(--muted); white-space:normal; }
  #controls { display:flex; align-items:center; gap:8px; padding:8px 12px; background:var(--bar); color:var(--bar-ink); }
  button { background:transparent; color:var(--bar-ink); border:1px solid rgba(255,255,255,.18); border-radius:6px; padding:5px 9px; font-size:14px; cursor:pointer; line-height:1; }
  button:hover { background:rgba(255,255,255,.1); }
  button.primary { background:var(--accent); border-color:var(--accent); color:var(--accent-ink); min-width:88px; font-weight:600; }
  #seek { flex:1; accent-color:var(--accent); }
  #count { font-variant-numeric:tabular-nums; font-size:12px; opacity:.85; min-width:64px; text-align:center; }
  select { background:#2c2c2a; color:var(--bar-ink); border:1px solid rgba(255,255,255,.18); border-radius:6px; padding:4px 6px; font-size:12px; }
  #err { position:absolute; bottom:8px; left:12px; font-size:12px; color:var(--warn); }
  /* reveal states applied to the graphviz groups */
  .node.hidden, .edge.hidden { display:none; }
  .node.pending > *:not(title):not(.outline):not(.pendtext) { display:none; }
  .node.pending > .outline { fill:var(--pending-fill) !important; stroke:var(--pending-line) !important; stroke-dasharray:6 4; stroke-width:1.4; }
  .pendtext { display:none; fill:var(--pending-ink); font-family:Helvetica,Arial,sans-serif; font-size:12px; }
  .node.pending > .pendtext { display:block; }
  .node.current > .outline { stroke:var(--warn) !important; stroke-width:3.2; }
  .node.fresh > .outline { animation: pop .6s ease-out; }
  @keyframes pop { from { stroke-width:6; } to { stroke-width:1.2; } }
  :fullscreen { --tex:24px; }
  :fullscreen #app { border:none; border-radius:0; }
  :fullscreen #title { font-size:24px; }
  :fullscreen #what, :fullscreen #here { font-size:16px; }
  :fullscreen #top { font-size:15px; }
  :fullscreen #caption { padding:14px 24px 12px; }
  @media (max-width:640px) { #title{font-size:15px} .tex{font-size:14px} button{padding:5px 7px} #count{display:none} }
</style></head>
<body>
<div id="app" tabindex="0">
  <div id="top"><span id="stepinfo"></span><span class="spacer"></span><span id="phaseinfo"></span></div>
  <div id="stage"><div class="placeholder">Preparing the tree…</div><div id="banner"></div><div id="err"></div></div>
  <div id="caption">
    <p id="here"></p>
    <p id="title"></p>
    <p id="what"></p>
    <div id="lines"></div>
  </div>
  <div id="controls">
    <button id="first" title="First (Home)">⏮</button>
    <button id="prev" title="Back one phase (←)">◀</button>
    <button id="play" class="primary" title="Play / pause (space)">▶ Play</button>
    <button id="next" title="Forward one phase (→)">▶|</button>
    <button id="last" title="Last (End)">⏭</button>
    <button id="replay" title="Replay from the start (R)">🔁</button>
    <input id="seek" type="range" min="0" max="0" value="0" step="1">
    <span id="count"></span>
    <select id="speed" title="Seconds per phase">
      <option value="1000">1 s</option><option value="1500">1.5 s</option>
      <option value="2500" selected>2.5 s</option><option value="3500">3.5 s</option>
      <option value="5000">5 s</option><option value="8000">8 s</option>
    </select>
    <button id="full" title="Full screen (F)">⛶</button>
  </div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById('data').textContent);
  const frames = data.frames, states = data.states;
  const $ = (id) => document.getElementById(id);
  const stage = $('stage'), lines = $('lines'), seek = $('seek'), playBtn = $('play');
  const storeKey = 'dt_theater_' + data.key;
  const MAX_UPSCALE = 1.6, PAD = 28, ZOOM_MS = 450;

  let idx = 0, playing = false, timer = null, speed = 2500;
  let svg = null, nodes = new Map(), edges = [], lastState = -1, lastCurrent = null;
  let view = null, zoomAnim = null;

  // ------------------------------------------------------------ persistence
  // Streamlit re-mounts this iframe whenever the page reruns; remembering where
  // we were means a click elsewhere on the page does not restart the video.
  try {
    const saved = JSON.parse(sessionStorage.getItem(storeKey) || 'null');
    if (saved && typeof saved.idx === 'number') { idx = Math.min(saved.idx, frames.length - 1); playing = !!saved.playing; speed = saved.speed || speed; }
    else { playing = true; }
  } catch (e) { playing = true; }
  function persist() { try { sessionStorage.setItem(storeKey, JSON.stringify({ idx, playing, speed })); } catch (e) {} }

  // -------------------------------------------------------------- the tree
  function loadSvg() {
    if (data.tree.svg) {
      const doc = new DOMParser().parseFromString(data.tree.svg, 'image/svg+xml');
      return Promise.resolve(document.importNode(doc.documentElement, true));
    }
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/@viz-js/viz@3.11.0/lib/viz-standalone.js';
      s.onload = () => Viz.instance().then(v => resolve(v.renderSVGElement(data.tree.dot)), reject);
      s.onerror = () => reject(new Error('viz.js failed to load — install graphviz, or allow access to cdn.jsdelivr.net'));
      document.head.appendChild(s);
    });
  }
  function indexSvg(el) {
    const SHAPES = new Set(['path', 'polygon', 'ellipse', 'rect']);
    for (const g of el.querySelectorAll('g.node')) {
      const title = g.querySelector('title'); if (!title) continue;
      const id = parseInt(title.textContent.replace(/^n/, ''), 10);
      const outline = [...g.children].find(c => SHAPES.has(c.tagName.toLowerCase()));
      if (outline) outline.classList.add('outline');
      nodes.set(id, { g, outline, pend: null });
    }
    for (const g of el.querySelectorAll('g.edge')) {
      const title = g.querySelector('title'); if (!title) continue;
      const m = title.textContent.match(/n(\d+)\s*(?:->|&#45;&gt;|→)\s*n(\d+)/);
      if (m) edges.push({ g, from: parseInt(m[1], 10), to: parseInt(m[2], 10) });
    }
    const bg = el.querySelector('g.graph > polygon'); if (bg) bg.setAttribute('fill', 'transparent');
    const vb = (el.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number);
    el.dataset.fullView = JSON.stringify(vb);
    el.removeAttribute('width'); el.removeAttribute('height');
    el.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  }
  function pendingText(entry, id) {
    // Two muted lines centred in the box, standing in for the real label until
    // the node is processed.
    if (entry.pend) return;
    const b = entry.outline.getBBox();
    const NS = 'http://www.w3.org/2000/svg';
    const grp = document.createElementNS(NS, 'g'); grp.setAttribute('class', 'pendtext');
    const mk = (y, txt, bold) => { const t = document.createElementNS(NS, 'text'); t.setAttribute('x', b.x + b.width / 2); t.setAttribute('y', y); t.setAttribute('text-anchor', 'middle'); if (bold) t.setAttribute('font-weight', 'bold'); t.textContent = txt; return t; };
    const cy = b.y + b.height / 2;
    grp.appendChild(mk(cy - 2, 'node #' + id, true));
    grp.appendChild(mk(cy + 14, 'samples = ' + ((data.nodes[id] || {}).samples || '?'), false));
    entry.g.appendChild(grp); entry.pend = grp;
  }
  function applyState(stateIdx, currentId) {
    const st = states[stateIdx];
    const processed = new Set(st.processed), pending = new Set(st.pending);
    const visible = new Set([...processed, ...pending]);
    for (const [id, entry] of nodes) {
      const cls = entry.g.classList;
      cls.remove('hidden', 'pending', 'current', 'fresh');
      if (pending.has(id)) { pendingText(entry, id); cls.add('pending'); }
      else if (!processed.has(id)) cls.add('hidden');
      if (id === currentId) cls.add('current');
    }
    for (const e of edges) e.g.classList.toggle('hidden', !(visible.has(e.from) && visible.has(e.to)));
    // A node that just got decided pops once so the eye lands on it.
    if (stateIdx !== lastState && lastState >= 0 && currentId !== null && nodes.has(currentId)) nodes.get(currentId).g.classList.add('fresh');
    lastState = stateIdx; lastCurrent = currentId;
    zoomToVisible(visible);
  }
  function unionBox(ids) {
    const ctm = svg.getScreenCTM(); if (!ctm) return null;
    const inv = ctm.inverse(); let box = null;
    const pt = svg.createSVGPoint();
    const add = (el) => {
      const r = el.getBoundingClientRect(); if (!r.width && !r.height) return;
      for (const [x, y] of [[r.left, r.top], [r.right, r.bottom]]) {
        pt.x = x; pt.y = y; const p = pt.matrixTransform(inv);
        if (!box) box = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
        else { box.x0 = Math.min(box.x0, p.x); box.y0 = Math.min(box.y0, p.y); box.x1 = Math.max(box.x1, p.x); box.y1 = Math.max(box.y1, p.y); }
      }
    };
    for (const id of ids) { const n = nodes.get(id); if (n) add(n.g); }
    for (const e of edges) if (!e.g.classList.contains('hidden')) add(e.g);
    return box;
  }
  function zoomToVisible(visible) {
    if (!svg) return;
    const full = JSON.parse(svg.dataset.fullView);
    let target = full.slice();
    const box = unionBox(visible);
    if (box) {
      // Fit the visible part, but never blow a tiny subtree up past MAX_UPSCALE.
      const sw = stage.clientWidth, sh = stage.clientHeight;
      let w = box.x1 - box.x0 + 2 * PAD, h = box.y1 - box.y0 + 2 * PAD;
      const minW = sw / MAX_UPSCALE, minH = sh / MAX_UPSCALE;
      if (w < minW) w = minW; if (h < minH) h = minH;
      const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
      target = [cx - w / 2, cy - h / 2, w, h];
    }
    animateView(target);
  }
  function setView(v) { view = v; svg.setAttribute('viewBox', v.map(n => n.toFixed(2)).join(' ')); }
  function animateView(target) {
    if (zoomAnim) cancelAnimationFrame(zoomAnim);
    if (!view) { setView(target); return; }
    const from = view.slice(), t0 = performance.now();
    const ease = (t) => 1 - Math.pow(1 - t, 3);
    const stepFn = (now) => {
      const t = Math.min(1, (now - t0) / ZOOM_MS), k = ease(t);
      setView(from.map((f, i) => f + (target[i] - f) * k));
      if (t < 1) zoomAnim = requestAnimationFrame(stepFn); else zoomAnim = null;
    };
    zoomAnim = requestAnimationFrame(stepFn);
  }

  // ---------------------------------------------------------------- formulas
  function renderLines(texs) {
    lines.innerHTML = '';
    for (const tex of texs) {
      const div = document.createElement('div');
      div.className = 'tex';
      if (window.katex) {
        try { katex.render(tex, div, { displayMode: true, throwOnError: false, strict: 'ignore' }); }
        catch (e) { div.classList.add('raw'); div.textContent = tex; }
      } else { div.classList.add('raw'); div.textContent = tex; }
      lines.appendChild(div);
    }
    // Shrink a formula that overflows the strip rather than making the reader scroll.
    requestAnimationFrame(() => {
      const base = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--tex')) || 17;
      for (const div of lines.children) {
        const inner = div.firstElementChild; if (!inner) continue;
        const ratio = div.clientWidth / inner.scrollWidth;
        if (ratio < 1) div.style.fontSize = Math.max(0.6, ratio * 0.98) * base + 'px';
      }
    });
  }
  window.addEventListener('load', () => { if (window.katex) renderLines(frames[idx].lines); });

  // ------------------------------------------------------------------ frames
  const PHASE_LABEL = { impurity: 'Measuring impurity', score: 'Scoring candidates', working: 'The working',
                        choice: 'Choosing the split', stop: 'Stopping — a leaf' };
  function render() {
    const f = frames[idx];
    $('stepinfo').innerHTML = `<b>${data.algorithm}</b> · step <b>${f.step}</b> / ${data.n_steps} · node <b>#${f.node_id}</b> (depth ${f.depth})`;
    $('phaseinfo').textContent = PHASE_LABEL[f.phase] || f.phase;
    $('here').textContent = f.here;
    $('title').textContent = f.title;
    $('what').textContent = f.what;
    renderLines(f.lines);
    seek.value = idx; $('count').textContent = `${idx + 1} / ${frames.length}`;
    const banner = $('banner'); banner.style.display = f.final && f.banner ? 'block' : 'none'; banner.textContent = f.banner || '';
    playBtn.textContent = playing ? '⏸ Pause' : (idx >= frames.length - 1 ? '🔁 Replay' : '▶ Play');
    if (svg) applyState(f.state, f.node_id);
    persist();
  }
  function goto(i) { idx = Math.max(0, Math.min(frames.length - 1, i)); render(); }
  function tick() { if (idx >= frames.length - 1) { pause(); return; } goto(idx + 1); }
  function play() {
    if (idx >= frames.length - 1) idx = 0;
    playing = true; clearInterval(timer); timer = setInterval(tick, speed); render();
  }
  function pause() { playing = false; clearInterval(timer); timer = null; render(); }
  function toggle() { playing ? pause() : play(); }
  function step(delta) { pause(); goto(idx + delta); }

  // ---------------------------------------------------------------- controls
  $('first').onclick = () => { pause(); goto(0); };
  $('last').onclick = () => { pause(); goto(frames.length - 1); };
  $('prev').onclick = () => step(-1);
  $('next').onclick = () => step(1);
  playBtn.onclick = toggle;
  $('replay').onclick = () => { idx = 0; play(); };
  seek.max = frames.length - 1;
  seek.oninput = () => { const target = parseInt(seek.value, 10); pause(); goto(target); };  // read before pause() re-renders
  const speedSel = $('speed'); speedSel.value = String(speed);
  if (speedSel.value !== String(speed)) speedSel.value = '2500';
  speedSel.onchange = () => { speed = parseInt(speedSel.value, 10); if (playing) play(); else persist(); };
  $('full').onclick = () => {
    const root = document.documentElement;
    if (document.fullscreenElement) document.exitFullscreen();
    else if (root.requestFullscreen) root.requestFullscreen().catch(() => { $('err').textContent = 'Full screen was blocked by the browser.'; });
  };
  function refit() { if (svg) { view = null; applyState(frames[idx].state, frames[idx].node_id); } renderLines(frames[idx].lines); }
  document.addEventListener('fullscreenchange', () => setTimeout(refit, 60));
  window.addEventListener('resize', refit);
  $('app').addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); toggle(); }
    else if (e.key === 'ArrowRight') step(1);
    else if (e.key === 'ArrowLeft') step(-1);
    else if (e.key === 'Home') { pause(); goto(0); }
    else if (e.key === 'End') { pause(); goto(frames.length - 1); }
    else if (e.key === 'f' || e.key === 'F') $('full').click();
    else if (e.key === 'r' || e.key === 'R') $('replay').click();
  });
  stage.addEventListener('click', () => $('app').focus());

  render();  // captions first, so something is on screen while the drawing loads
  loadSvg().then((el) => {
    stage.querySelectorAll('.placeholder').forEach(n => n.remove());
    stage.appendChild(el); svg = el; indexSvg(el);
    render();
    if (playing) play();
  }).catch((e) => { $('err').textContent = 'Could not draw the tree: ' + (e && e.message ? e.message : e); if (playing) play(); });
})();
</script>
</body></html>
"""
