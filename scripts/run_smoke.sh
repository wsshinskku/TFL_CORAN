#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[dev]'
python -m tfl_coran.cli run --config configs/smoke.yaml --output runs/smoke
python -m pytest -q
