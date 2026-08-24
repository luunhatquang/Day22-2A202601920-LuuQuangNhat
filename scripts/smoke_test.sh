#!/usr/bin/env bash
set -euo pipefail

PREF_LAB="pref-lab"
if [ -f ".venv/bin/pref-lab" ]; then
    PREF_LAB=".venv/bin/pref-lab"
fi

$PREF_LAB validate data/sample_preferences.jsonl
$PREF_LAB evaluate --config configs/local.yaml
cat outputs/metrics.json
