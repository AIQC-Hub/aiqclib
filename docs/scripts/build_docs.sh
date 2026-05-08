#!/bin/bash

set -e  # Exit on error
set -x  # Echo commands (for debugging)

# Paths
DOCS_DIR="$(dirname "$0")/.."
SOURCE_DIR="$DOCS_DIR/source"
BUILD_DIR="$DOCS_DIR/build"
API_DIR="$SOURCE_DIR/api"
MODULE_DIR="$DOCS_DIR/../src/aiqclib"

# Step 1: Clean old build and API .rst files
rm -rf "$BUILD_DIR"
rm -rf "$API_DIR"
mkdir -p "$API_DIR"

# Step 2: Re-generate API RSTs
uv run sphinx-apidoc -e -d 5 -o "$API_DIR" "$MODULE_DIR"

# Step 3: Clean "Module contents" from API RST files
uv run python "$DOCS_DIR/scripts/clean_api_rst.py"

# Step 4: Build HTML
uv run sphinx-build -b html "$SOURCE_DIR" "$BUILD_DIR"
