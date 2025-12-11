#!/bin/sh
# Export Marimo notebooks in ${MARIMO_FOLDER} to HTML under _marimushka
# This replicates the previous Makefile logic for maintainability and reuse.

set -e

MARIMO_FOLDER=${MARIMO_FOLDER:-book/marimo}
MARIMUSHKA_OUTPUT=${MARIMUSHKA_OUTPUT:-_marimushka}
UV_BIN=${UV_BIN:-./bin/uv}

BLUE="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

printf "%b[INFO] Exporting notebooks from %s...%b\n" "$BLUE" "$MARIMO_FOLDER" "$RESET"

if [ ! -d "$MARIMO_FOLDER" ]; then
  printf "%b[WARN] Directory '%s' does not exist. Skipping marimushka.%b\n" "$YELLOW" "$MARIMO_FOLDER" "$RESET"
  exit 0
fi

# Ensure output directory exists
mkdir -p "$MARIMUSHKA_OUTPUT"

# Discover .py files (top-level only) using globbing; handle no-match case
set -- "$MARIMO_FOLDER"/*.py
if [ "$1" = "$MARIMO_FOLDER/*.py" ]; then
  printf "%b[WARN] No Python files found in '%s'.%b\n" "$YELLOW" "$MARIMO_FOLDER" "$RESET"
  # Create a minimal index.html indicating no notebooks
  printf '<html><head><title>Marimo Notebooks</title></head><body><h1>Marimo Notebooks</h1><p>No notebooks found.</p></body></html>' > "$MARIMUSHKA_OUTPUT/index.html"
  exit 0
fi

#CURRENT_DIR=$(pwd)
#OUTPUT_DIR="$CURRENT_DIR/$MARIMUSHKA_OUTPUT"
#
#
## Explicitly loop over .py files in $MARIMO_FOLDER and export each with marimo (no sandbox)
#notes=""
#for nb in "$@"; do
#  base=$(basename "$nb")
#  name="${base%.py}"
#  out="$OUTPUT_DIR/${name}.html"
#  printf "%b[INFO] Exporting %s -> %s...%b\n" "$BLUE" "$nb" "$out" "$RESET"
#  if "$UV_BIN" run marimo export html --no-sandbox "$nb" -f -o "$out"; then
#    notes="$notes<li><a href=\"${name}.html\">${name}</a></li>\n"
#  else
#    printf "%b[WARN] Failed to export %s; continuing...%b\n" "$YELLOW" "$nb" "$RESET"
#  fi
#done
#
## Generate a simple index.html linking to exported notebooks
#if [ -n "$notes" ]; then
#  {
#    printf '<html><head><meta charset="utf-8"><title>Marimo Notebooks</title></head><body>'
#    printf '<h1>Marimo Notebooks</h1><ul>'
#    printf '%b' "$notes"
#    printf '</ul></body></html>'
#  } > "$OUTPUT_DIR/index.html"
#else
#  printf '<html><head><title>Marimo Notebooks</title></head><body><h1>Marimo Notebooks</h1><p>No notebooks exported.</p></body></html>' > "$OUTPUT_DIR/index.html"
#fi

uv run python ./.github/templates/scripts/apply_jinja.py

# Bypass this code using Marimushka
#uv pip install marimushka
#uv pip install -e .
#uv run --help
#uv run marimushka export -n $MARIMO_FOLDER -o $OUTPUT_DIR --no-sandbox
#$UVX_BIN marimushka export -n $MARIMO_FOLDER -o $OUTPUT_DIR --no-sandbox --bin-path "./bin/uv"

# Ensure GitHub Pages does not process with Jekyll
: > "$OUTPUT_DIR/.nojekyll"
