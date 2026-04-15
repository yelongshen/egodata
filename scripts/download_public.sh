#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-downloads}"

python -m motion_dataset_downloaders.cli download-public --root "$ROOT_DIR"