#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Point the bundled SlideSpeech .jsonl files at your local copy of the audio.
#
#   bash scripts/prepare_slidespeech.sh /path/to/slidespeech/audio
#
# The argument is the directory holding one sub-directory per domain, each with an audios/
# folder, e.g. <root>/agriculture/audios/agriculture_0001-00000.wav. Every "source" field in
# data/slidespeech/**/*.jsonl is rewritten from the <SLIDESPEECH_ROOT> placeholder to that
# path. The script is idempotent and can be re-run to point at a different copy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/data/slidespeech"
PLACEHOLDER="<SLIDESPEECH_ROOT>"

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/prepare_slidespeech.sh /path/to/slidespeech/audio" >&2
    exit 1
fi

ROOT="${1%/}"
[ -d "$ROOT" ] || { echo "ERROR: '$ROOT' is not a directory" >&2; exit 1; }

mapfile -t FILES < <(find "$DATA_DIR" -name '*.jsonl' | sort)
[ "${#FILES[@]}" -gt 0 ] || { echo "ERROR: no .jsonl files under $DATA_DIR" >&2; exit 1; }

for f in "${FILES[@]}"; do
    if grep -q "$PLACEHOLDER" "$f"; then
        sed -i "s|$PLACEHOLDER|$ROOT|g" "$f"
        echo "  set root:   ${f#"$REPO_ROOT"/}"
    else
        # Already prepared: swap whatever root is in there for the new one.
        current=$(head -1 "$f" | sed -n 's|.*"source": "\(.*\)/[^/]*/audios/[^"]*".*|\1|p')
        if [ -n "$current" ] && [ "$current" != "$ROOT" ]; then
            sed -i "s|$current/|$ROOT/|g" "$f"
            echo "  re-pointed: ${f#"$REPO_ROOT"/}"
        else
            echo "  unchanged:  ${f#"$REPO_ROOT"/}"
        fi
    fi
done

echo
echo "Checking that the referenced audio exists..."
missing=0
for f in "${FILES[@]}"; do
    first=$(head -1 "$f" | sed -n 's|.*"source": "\([^"]*\)".*|\1|p')
    [ -f "$first" ] || { echo "  MISSING: $first" >&2; missing=1; }
done
if [ "$missing" -eq 0 ]; then
    echo "  OK: the first utterance of every split was found."
else
    echo >&2
    echo "  Some audio was not found. Expected layout:" >&2
    echo "    $ROOT/<domain>/audios/<utterance-id>.wav" >&2
    exit 1
fi
