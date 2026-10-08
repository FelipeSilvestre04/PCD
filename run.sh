#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python3}" -u campaign.py --seconds 120 --repetitions 5 --output "${1:-results_new}"
