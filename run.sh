#!/usr/bin/env bash
# Create the virtual environment on first run, then start the app.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating .venv …"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

exec .venv/bin/streamlit run app.py "$@"
