# 🌳 Decision Tree Explainer

A Streamlit app for learning how decision trees are actually built. ID3, C4.5 and
CART are implemented from scratch in `src/core/` — no scikit-learn — so every
entropy, gain ratio and Gini value on screen comes with the formula it was
computed from.

## Quick start

```bash
git clone https://github.com/Chung10102004/DecisionTreeVisualization.git
cd DecisionTreeVisualization
./run.sh
```

`run.sh` creates `.venv/` and installs the dependencies the first time you run
it, then starts Streamlit. The app opens at <http://localhost:8501>.

Anything you pass to the script goes through to Streamlit, so a different port is
just:

```bash
./run.sh --server.port 8600
```

### Manual setup

If you would rather not use the script, or you are on Windows:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Requirements

- Python 3.10 or newer (developed on 3.12)
- The packages in `requirements.txt`: streamlit, numpy, pandas, matplotlib,
  plotly, pytest

Graphviz is **optional but recommended**. Without it, trees are drawn in the
browser (the Build page's player then loads viz.js from cdn.jsdelivr.net);
with the system package (`sudo apt install graphviz`) they render locally, and
the Tree page gains SVG/PNG export. The player typesets its formulas with KaTeX
from the same CDN, so the Build page needs internet access either way.

## Using the app

The sidebar walks through seven pages in order:

| Page | What it does |
| --- | --- |
| 1 · Data | Pick a bundled dataset or upload your own CSV, and set column types |
| 2 · Build | Plays the construction like a video: the tree grows on a full-screen stage while one strip below shows the formula each moment computes — play, pause, step, scrub, replay, full screen |
| 3 · Tree | The finished tree, plus DOT/SVG/PNG export |
| 4 · Predict | Send a row down the tree and see the path it takes |
| 5 · Evaluate | Accuracy, confusion matrix, and the effect of pruning |
| 6 · Compare | ID3 vs C4.5 vs CART on the same data, side by side |
| 7 · Theory | The formulas behind each criterion |

Nine dataset presets are built in (over eight CSVs in `data/`), from the 14-row PlayTennis table used in most ID3
lectures up to Penguins, Titanic and Auto MPG. Start with **PlayTennis** — it is
small enough to check every number by hand — then try **PlayTennis + Day** to
watch ID3 get fooled by an ID column while C4.5 rejects it.

## Development

Run the test suite:

```bash
.venv/bin/pytest
```

Or the end-to-end smoke test, which trains every algorithm on every bundled
dataset and renders every figure without touching Streamlit:

```bash
.venv/bin/python scripts/smoke_e2e.py
```

## Layout

```
app.py            Streamlit entry point; wires the pages together
src/core/         ID3, C4.5, CART, splitting criteria, pruning, metrics
src/datasets/     dataset catalogue, loading, preprocessing
src/viz/          DOT generation, matplotlib figures, LaTeX formatting
src/ui/           the Streamlit layer (pages, state, components)
data/             the bundled CSVs
tests/            pytest suite
scripts/          end-to-end smoke test
```
