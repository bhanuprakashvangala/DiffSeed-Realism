#!/usr/bin/env bash
# Download the seed image corpora used by the full pipeline.
#
#   bash data/download.sh            soybean corpus (primary)
#   bash data/download.sh maize      also the maize corpus (cross-species check)
#
# Needs the Kaggle CLI (pip install kaggle) with an API token in
# ~/.kaggle/kaggle.json. The soybean corpus is also available from Mendeley Data
# (https://doi.org/10.17632/v6vzvfszj6.6) for manual download.
set -euo pipefail
DEST="${DATA:-$(cd "$(dirname "$0")" && pwd)}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found: pip install kaggle, then add ~/.kaggle/kaggle.json" >&2
  exit 3
fi

if [ ! -d "$DEST/soybean_seeds" ]; then
  kaggle datasets download -d warcoder/soyabean-seeds --unzip -p "$DEST/soybean_seeds"
fi
echo "soybean corpus: $DEST/soybean_seeds"

if [ "${1:-}" = "maize" ] && [ ! -d "$DEST/maize" ]; then
  kaggle datasets download -d yungprof123/maize-seed-dataset --unzip -p "$DEST/maize"
  echo "maize corpus: $DEST/maize"
fi
