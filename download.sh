#!/usr/bin/env bash
set -e

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

BASE_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K"

for FILE in stories260K.bin tok512.bin; do
    DST="$MODELS_DIR/$FILE"
    if [ -f "$DST" ]; then
        echo "$FILE: already exists, skipping"
    else
        curl -L "$BASE_URL/$FILE" -o "$DST"
    fi
done
