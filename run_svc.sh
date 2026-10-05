#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python ensure_models.py
exec python app.py
