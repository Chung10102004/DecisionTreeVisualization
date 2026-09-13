"""The Build page's "theater": the whole construction as one single-screen player.

Everything a step needs is on screen at once, like a slide in an explainer video:
the tree grows on the left, and a board on the right fills in block by block —
the node's impurity, then the gain of the winning column worked out with the
node's real numbers, then a one-line-per-column list of the others ("the same
formula, so only the results"), then the decision.  Nothing scrolls; text shrinks
to fit instead.

The drawing is the *finished* tree revealed progressively: unreached nodes are
hidden, queued nodes are grey "pending" boxes, the node being processed is
outlined, and the camera fits whatever is visible.  Because the layout is the
final one from the start, nodes never jump as the tree grows, and only one
drawing is sent to the browser however many steps there are.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core import latex as tex
from ..core.criteria import Computation, get_impurity
from ..core.playback import (
    PHASE_CHOICE,
    PHASE_IMPURITY,
    PHASE_STOP,
    PHASE_WORKING,
    Cursor,
    advance,
    max_phase,
)
from ..core.trace import ACTION_SPLIT, STOP_REASONS, BuildStep, TrainingTrace
from .tree_dot import RenderOptions, TreeDotRenderer, dot_to_svg, has_graphviz_binary

MAX_BRANCH_LINES = 4     # branch entropies written out for the winner
MAX_OTHER_LINES = 5      # "similarly for the other columns" entries
MAX_ARGMAX = 4
THEATER_OPTIONS = RenderOptions(orientation="TB", show_node_ids=True,
                                show_distribution_bar=True, font_size=12)

PHASE_NAMES = {PHASE_IMPURITY: "impurity", PHASE_WORKING: "working", PHASE_CHOICE: "choice"}


@dataclass
class Section:
    """One block on the board; ``beat`` says at which phase it appears."""

    beat: int
    kind: str            # impurity | working | others | choice | stop
    title: str
    lines: List[str] = field(default_factory=list)   # LaTeX, one formula each
    note: str = ""


@dataclass
class Frame:
    """One tick of the player: a step, and how much of its board is revealed."""

    step: int            # 1-based, for display
    node_id: int
    depth: int
    phase: str           # impurity | working | choice | stop
    beat: int            # sections with beat <= this are visible
    state: int           # index into the reveal-state list
    title: str
    here: str            # why this node exists: the path from the root
    sections: List[Section] = field(default_factory=list)
    final: bool = False
    banner: str = ""

    @property
    def lines(self) -> List[str]:
        """Every formula visible at this tick, in reading order."""
        return [line for section in self.sections if section.beat <= self.beat for line in section.lines]


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


def _score_symbol(trace: TrainingTrace) -> str:
    """``IG``, ``GainRatio``, ``\\Delta Gini`` … — the short form used in the formulas."""
    if trace.score_name == "Gain Ratio":
        return "GainRatio"
    return get_impurity(trace.impurity_name).gain_symbol


def _argmax_latex(ranked, chosen, symbol: str, top: int = MAX_ARGMAX) -> List[str]:
    """Two lines: ``A* = argmax_A IG(S,A)`` then ``= argmax{Outlook: 0.32, …} ⇒ Outlook``."""
    entries = [r"%s{:}\ %s" % (tex.text(c.feature_name), _num(c.score)) for c in ranked[:top]]
    if len(ranked) > top:
        entries.append(r"\ldots")
    return [
        r"A^{*} = \arg\max_{A}\ %s(S, A)" % symbol,
        r"A^{*} = \arg\max\left\{%s\right\} \Rightarrow %s"
        % (r",\ ".join(entries), tex.text(chosen.feature_name)),
    ]


def _here(step: BuildStep) -> str:
    if not step.path_conditions:
        return f"Root — all {step.n_samples:g} training samples start here."
    counts = f" ({step.counts_text()})" if step.class_counts else ""
    return f"Here because {' and '.join(step.path_conditions)} — {step.n_samples:g} samples{counts}."


def _pretty_branch(label: str) -> str:
    return label.replace("<=", "≤").replace(">=", "≥").strip()


def _numbers_only(computation: Computation) -> str:
    """``lhs = numbers = result`` — the symbolic middle is shown once, separately."""
    return f"{computation.lhs} = {computation.substituted} = {tex.num(computation.result)}"


def _branch_lines(chosen, symbol: str) -> List[str]:
    """The winner's branch impurities, each with its own numbers substituted."""
    weighted = chosen.computations.get("weighted_impurity")
    if weighted is None:
        return []
    lines: List[str] = []
    keys = list(chosen.partitions)
    for key in keys[:MAX_BRANCH_LINES]:
        child = weighted.children.get(str(key))
        if child is None:
            continue
        label = _pretty_branch(f"{chosen.feature_name} {chosen.branch_labels.get(key, key)}")
        lines.append(r"%s(S_{%s}) = %s = %s"
                     % (symbol, tex.text(label), child.substituted, tex.num(child.result)))
    if len(keys) > MAX_BRANCH_LINES:
        lines.append(r"\ldots\ \text{%d more branches, computed the same way}"
                     % (len(keys) - MAX_BRANCH_LINES))
    return lines


def _working_lines(chosen, score_name: str) -> List[str]:
    """Generic formula once, then the winner's numbers — and C4.5's ratio if used."""
    comps = chosen.computations
    lines: List[str] = []
    gain = comps.get("gain")
    if gain is not None:
        lines.append(gain.symbolic)          # IG(S, A) = H(S) - Σ |S_v|/|S| H(S_v)
        lines.append(_numbers_only(gain))    # IG(S, Outlook) = 0.971 - [...] = 0.322
    if score_name == "Gain Ratio":
        for key in ("split_info", "gain_ratio"):
            if key in comps:
                lines.append(_numbers_only(comps[key]))
    return lines


def _other_lines(ranked, chosen, score_name: str) -> Tuple[List[str], str]:
    others = [c for c in ranked if c is not chosen]
    lines = [f"{_score_lhs(c, score_name)} = {_num(c.score)}" for c in others[:MAX_OTHER_LINES]]
    note = ""
    if len(others) > MAX_OTHER_LINES:
        note = f"and {len(others) - MAX_OTHER_LINES} more column(s), all lower."
    return lines, note


def build_sections(step: BuildStep, trace: TrainingTrace) -> Tuple[str, List[Section]]:
    """``(title, sections)`` for one step — the whole board, beats included."""
    impurity = get_impurity(trace.impurity_name)
    score = trace.score_name
    symbol = impurity.symbol

    sections: List[Section] = [
        Section(
            beat=PHASE_IMPURITY, kind="impurity",
            title=f"① How impure is node #{step.node_id}?",
            lines=[step.impurity_computation.full_latex()],
            note=f"{impurity.display} of the {step.n_samples:g} samples here"
                 + (f" ({step.counts_text()})." if step.class_counts else "."),
        )
    ]

    if step.action != ACTION_SPLIT:
        reason = STOP_REASONS.get(step.stop_reason or "", step.stop_reason or "no split was possible")
        if step.stop_computation is not None:
            lines = [step.stop_computation.full_latex()]
        else:
            lines = [r"%s(S_{%d}) = 0 \Rightarrow \text{pure} \Rightarrow \text{leaf: } %s"
                     % (symbol, step.node_id, tex.text(step.prediction))]
        sections.append(Section(
            beat=PHASE_STOP, kind="stop",
            title=f"② Stop — leaf predicts {step.prediction}",
            lines=lines,
            note=f"No split is made because {reason}.",
        ))
        return f"Node #{step.node_id} becomes a leaf", sections

    ranked = [c for c in step.ranked_candidates if c.is_valid]
    rejected = sum(1 for c in step.candidates if not c.is_valid)
    chosen = step.chosen

    sections.append(Section(
        beat=PHASE_WORKING, kind="working",
        title=f"② {score} of the best column, {chosen.feature_name}",
        lines=_branch_lines(chosen, symbol) + _working_lines(chosen, score),
        note=f"Impurity of each branch, then the parent's impurity minus their size-weighted sum."
             + (" Divide by the split information for the ratio." if score == "Gain Ratio" else ""),
    ))

    other_lines, other_note = _other_lines(ranked, chosen, score)
    if other_lines:
        note = "Same formula, so only the results are listed."
        if other_note:
            note += " " + other_note
        if rejected:
            note += f" {rejected} column(s) rejected by the size rules."
        sections.append(Section(
            beat=PHASE_WORKING, kind="others",
            title="Similarly for the other columns",
            lines=other_lines, note=note,
        ))

    sections.append(_choice_section(step, trace, ranked, chosen))
    return f"Node #{step.node_id} — which column splits it best?", sections


def _choice_section(step: BuildStep, trace: TrainingTrace, ranked, chosen) -> Section:
    """The decision — including C4.5's average-gain filter when it changed the answer."""
    score = trace.score_name
    symbol = _score_symbol(trace)
    labels = ", ".join(_pretty_branch(str(v)) for v in chosen.branch_labels.values())
    tail = f" → {chosen.n_branches} branches: {labels}. Each child joins the queue."

    if ranked and ranked[0] is chosen:
        return Section(
            beat=PHASE_CHOICE, kind="choice",
            title=f"③ Decision — split on {chosen.feature_name}",
            lines=_argmax_latex(ranked, chosen, symbol),
            note=f"Highest {score} ({_num(chosen.score)})" + tail,
        )

    # The best-scoring column was not chosen: Quinlan's rule set it aside because
    # its raw gain is below the average gain of all candidates.
    mean_gain = sum(c.gain for c in ranked) / len(ranked)
    shortlist = [c for c in ranked if c.gain >= mean_gain - 1e-12] or ranked
    skipped = [c for c in ranked if c not in shortlist]
    mean_line = (r"\bar{g} = \frac{%s}{%d} = %s"
                 % (" + ".join(_num(c.gain) for c in ranked), len(ranked), _num(mean_gain)))
    filter_line = (r"\text{shortlist} = \{A : IG(S, A) \geq \bar{g}\} = \{%s\}"
                   % r",\ ".join(tex.text(c.feature_name) for c in shortlist))
    better = [c for c in skipped if c.score > chosen.score]
    who = " and ".join(f"{c.feature_name} ({symbol} {_num(c.score)}, but IG {_num(c.gain)} < {_num(mean_gain)})"
                       for c in better[:2])
    return Section(
        beat=PHASE_CHOICE, kind="choice",
        title=f"③ Decision — split on {chosen.feature_name}",
        lines=[mean_line, filter_line] + _argmax_latex(shortlist, chosen, symbol),
        note=(f"{who} scores higher but is set aside: C4.5 only ranks columns whose gain is at "
              f"least the average, so a tiny SplitInfo cannot inflate a weak split. Among the rest, "
              f"{chosen.feature_name} has the highest {score} ({_num(chosen.score)})" + tail),
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
    """All frames in playback order — one per (step, phase)."""
    frames: List[Frame] = []
    if len(trace) == 0:
        return frames

    boards = {step.step_id: build_sections(step, trace) for step in trace.steps}
    cursor = Cursor(0, 0)
    while True:
        step = trace.steps[cursor.step]
        title, sections = boards[step.step_id]
        deciding = cursor.phase >= max_phase(step)
        phase_name = "stop" if (step.action != ACTION_SPLIT and cursor.phase == PHASE_STOP) \
            else PHASE_NAMES[cursor.phase]
        frames.append(Frame(
            step=cursor.step + 1, node_id=step.node_id, depth=step.depth, phase=phase_name,
            beat=cursor.phase, state=cursor.step + 1 if deciding else cursor.step,
            title=title, here=_here(step), sections=sections,
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
    frames = build_frames(trace)
    # Boards are shared by every frame of a step; ship each once.
    boards: Dict[str, List[dict]] = {}
    slim: List[dict] = []
    for frame in frames:
        key_ = str(frame.step)
        if key_ not in boards:
            boards[key_] = [asdict(section) for section in frame.sections]
        record = asdict(frame)
        record.pop("sections")
        slim.append(record)
    return {
        "key": key,
        "algorithm": trace.algorithm,
        "n_steps": len(trace),
        "tree": {"svg": _svg_body(svg)} if svg else {"dot": dot},
        "nodes": {str(n.node_id): {"samples": f"{n.n_samples:g}"} for n in trace.tree.root.iter_nodes()},
        "states": reveal_states(trace),
        "boards": boards,
        "frames": slim,
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
  :root { --scale:1; --tex:18px; --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --muted:#5f5f5c; --line:#e4e2dd;
          --accent:#2a78d6; --accent-ink:#ffffff; --bar:#1d1d1b; --bar-ink:#f4f4f2; --warn:#c0392b; --good:#1f7a4d;
          --pending-fill:#EFEFEF; --pending-line:#B8B8B4; --pending-ink:#8a8a86; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--ink);
              font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
  #app { display:flex; flex-direction:column; height:100vh; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:var(--bg); }
  #top { display:flex; align-items:center; gap:12px; padding:7px 14px; border-bottom:1px solid var(--line); background:var(--panel); font-size:13px; color:var(--muted); }
  #top b { color:var(--ink); font-weight:600; }
  #top .spacer { flex:1; }
  #progress { height:3px; background:var(--line); }
  #progress > div { height:100%; background:var(--accent); width:0; transition:width .3s; }
  #main { flex:1; min-height:0; display:flex; }
  #stage { flex:0 0 44%; min-width:260px; position:relative; overflow:hidden; border-right:1px solid var(--line); }
  #stage svg { display:block; width:100%; height:100%; }
  #stage .placeholder { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:14px; }
  #banner { position:absolute; top:12px; left:50%; transform:translateX(-50%); background:var(--good); color:#fff; padding:6px 14px; border-radius:999px; font-size:13px; font-weight:600; display:none; white-space:nowrap; }
  #board { flex:1; min-width:0; overflow:hidden; padding:calc(14px*var(--scale)) calc(20px*var(--scale)); background:var(--panel); display:flex; flex-direction:column; }
  #here { font-size:calc(12.5px*var(--scale)); color:var(--muted); margin:0 0 2px; }
  #headline { font-size:calc(20px*var(--scale)); font-weight:700; margin:0 0 calc(10px*var(--scale)); }
  .sec { margin:0 0 calc(12px*var(--scale)); padding-left:calc(12px*var(--scale)); border-left:3px solid var(--line); }
  .sec.new { border-left-color:var(--accent); animation: rise .45s ease-out both; }
  .sec.choice.new, .sec.stop.new { border-left-color:var(--good); }
  .sec h3 { margin:0 0 calc(2px*var(--scale)); font-size:calc(14.5px*var(--scale)); font-weight:650; }
  .sec .note { margin:0 0 calc(4px*var(--scale)); font-size:calc(12.5px*var(--scale)); color:var(--muted); }
  .sec.others .note { margin-bottom:0; }
  .lines { display:flex; flex-direction:column; gap:calc(2px*var(--scale)); }
  .sec.others .lines { flex-direction:row; flex-wrap:wrap; column-gap:calc(22px*var(--scale)); row-gap:0; }
  .tex { font-size:calc(var(--tex)*var(--scale)); white-space:nowrap; overflow:hidden; padding:1px 0; }
  .tex .katex-display { margin:0; }
  .tex .katex-display > .katex { text-align:left; }
  .tex.raw { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12px; color:var(--muted); white-space:normal; }
  .sec.new .tex { animation: rise .4s ease-out both; animation-delay: calc(var(--i) * 220ms + 120ms); }
  @keyframes rise { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
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
  :fullscreen #headline { font-size:calc(26px*var(--scale)); }
  :fullscreen .sec h3 { font-size:calc(18px*var(--scale)); }
  :fullscreen .sec .note, :fullscreen #here { font-size:calc(15px*var(--scale)); }
  :fullscreen #top { font-size:15px; }
  @media (max-width:760px) { #main{flex-direction:column} #stage{flex:0 0 40%; border-right:none; border-bottom:1px solid var(--line)} button{padding:5px 7px} #count{display:none} }
</style></head>
<body>
<div id="app" tabindex="0">
  <div id="top"><span id="stepinfo"></span><span class="spacer"></span><span id="phaseinfo"></span></div>
  <div id="progress"><div></div></div>
  <div id="main">
    <div id="stage"><div class="placeholder">Preparing the tree…</div><div id="banner"></div><div id="err"></div></div>
    <div id="board">
      <p id="here"></p>
      <p id="headline"></p>
      <div id="sections"></div>
    </div>
  </div>
  <div id="controls">
    <button id="first" title="First (Home)">⏮</button>
    <button id="prev" title="Back one beat (←)">◀</button>
    <button id="play" class="primary" title="Play / pause (space)">▶ Play</button>
    <button id="next" title="Forward one beat (→)">▶|</button>
    <button id="last" title="Last (End)">⏭</button>
    <button id="replay" title="Replay from the start (R)">🔁</button>
    <input id="seek" type="range" min="0" max="0" value="0" step="1">
    <span id="count"></span>
    <select id="speed" title="Seconds per beat">
      <option value="1500">1.5 s</option><option value="2500">2.5 s</option>
      <option value="3500" selected>3.5 s</option><option value="5000">5 s</option>
      <option value="8000">8 s</option><option value="12000">12 s</option>
    </select>
    <button id="full" title="Full screen (F)">⛶</button>
  </div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById('data').textContent);
  const frames = data.frames, states = data.states, boards = data.boards;
  const $ = (id) => document.getElementById(id);
  const stage = $('stage'), board = $('board'), sectionsEl = $('sections'), seek = $('seek'), playBtn = $('play');
  const storeKey = 'dt_theater_' + data.key;
  const MAX_UPSCALE = 1.6, PAD = 28, ZOOM_MS = 450;

  let idx = 0, playing = false, timer = null, speed = 3500;
  let svg = null, nodes = new Map(), edges = [], lastState = -1;
  let view = null, zoomAnim = null, lastKey = null;

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
    if (stateIdx !== lastState && lastState >= 0 && currentId !== null && nodes.has(currentId)) nodes.get(currentId).g.classList.add('fresh');
    lastState = stateIdx;
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

  // ------------------------------------------------------------------ board
  function texLine(tex, i) {
    const div = document.createElement('div');
    div.className = 'tex'; div.style.setProperty('--i', i);
    if (window.katex) {
      try { katex.render(tex, div, { displayMode: true, throwOnError: false, strict: 'ignore' }); }
      catch (e) { div.classList.add('raw'); div.textContent = tex; }
    } else { div.classList.add('raw'); div.textContent = tex; }
    return div;
  }
  function renderBoard(f, animateBeat) {
    $('here').textContent = f.here;
    $('headline').textContent = f.title;
    sectionsEl.innerHTML = '';
    for (const sec of boards[String(f.step)]) {
      if (sec.beat > f.beat) continue;
      const el = document.createElement('section');
      el.className = 'sec ' + sec.kind + (animateBeat && sec.beat === f.beat ? ' new' : '');
      const h = document.createElement('h3'); h.textContent = sec.title; el.appendChild(h);
      if (sec.note && sec.kind !== 'others') { const n = document.createElement('p'); n.className = 'note'; n.textContent = sec.note; el.appendChild(n); }
      const wrap = document.createElement('div'); wrap.className = 'lines';
      sec.lines.forEach((tex, i) => wrap.appendChild(texLine(tex, i)));
      el.appendChild(wrap);
      if (sec.note && sec.kind === 'others') { const n = document.createElement('p'); n.className = 'note'; n.textContent = sec.note; el.appendChild(n); }
      sectionsEl.appendChild(el);
    }
    fitBoard();
  }
  // Nothing on the board may scroll: shrink every formula that is too wide, then
  // shrink the whole board until it fits the height it has.
  let fitReq = null;
  function fitBoard() {
    // Reset and measure inside the same animation frame: two renders in a row
    // (pause + seek) would otherwise measure text the first pass had already
    // shrunk, and relax it again.
    if (fitReq) cancelAnimationFrame(fitReq);
    fitReq = requestAnimationFrame(() => {
      fitReq = null;
      document.documentElement.style.setProperty('--scale', '1');
      for (const div of sectionsEl.querySelectorAll('.tex')) div.style.fontSize = '';
      let scale = 1;
      for (let i = 0; i < 8 && board.scrollHeight > board.clientHeight + 1 && scale > 0.6; i++) {
        scale = Math.round((scale - 0.06) * 100) / 100;
        document.documentElement.style.setProperty('--scale', String(scale));
      }
      const base = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--tex')) || 18) * scale;
      for (const div of sectionsEl.querySelectorAll('.tex')) {
        // overflow:hidden keeps scrollWidth honest about the formula's real width
        const ratio = div.clientWidth / div.scrollWidth;
        if (ratio < 1) div.style.fontSize = Math.max(0.55, ratio * 0.97) * base + 'px';
      }
    });
  }

  // ------------------------------------------------------------------ frames
  const PHASE_LABEL = { impurity: '① Measuring impurity', working: '② The gain, worked out', choice: '③ Choosing the split', stop: '② Stopping — a leaf' };
  function render() {
    const f = frames[idx];
    const key = f.step + ':' + f.beat;
    $('stepinfo').innerHTML = `<b>${data.algorithm}</b> · step <b>${f.step}</b> / ${data.n_steps} · node <b>#${f.node_id}</b> (depth ${f.depth})`;
    $('phaseinfo').textContent = PHASE_LABEL[f.phase] || f.phase;
    $('progress').firstElementChild.style.width = ((idx + 1) / frames.length * 100) + '%';
    renderBoard(f, key !== lastKey); lastKey = key;
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
  seek.oninput = () => { const target = parseInt(seek.value, 10); pause(); goto(target); };
  const speedSel = $('speed'); speedSel.value = String(speed);
  if (speedSel.value !== String(speed)) speedSel.value = '3500';
  speedSel.onchange = () => { speed = parseInt(speedSel.value, 10); if (playing) play(); else persist(); };
  $('full').onclick = () => {
    const root = document.documentElement;
    if (document.fullscreenElement) document.exitFullscreen();
    else if (root.requestFullscreen) root.requestFullscreen().catch(() => { $('err').textContent = 'Full screen was blocked by the browser.'; });
  };
  function refit() { if (svg) { view = null; applyState(frames[idx].state, frames[idx].node_id); } lastKey = null; renderBoard(frames[idx], false); }
  document.addEventListener('fullscreenchange', () => setTimeout(refit, 60));
  window.addEventListener('resize', refit);
  window.addEventListener('load', () => { if (window.katex) { lastKey = null; renderBoard(frames[idx], false); } });
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

  render();  // the board first, so something is on screen while the drawing loads
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
