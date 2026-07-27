#!/usr/bin/env bash
set -Eeuo pipefail

python /download_models.py
exec /start.sh
