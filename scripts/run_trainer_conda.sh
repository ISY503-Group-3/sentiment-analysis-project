#!/usr/bin/env bash
# Helper to run the background trainer inside a conda environment.
# Edit `TF_ENV` to match your conda env name (example: tf).

set -euo pipefail

TF_ENV="tf"
CSV_PATH="data/input_demo.csv"
OUT_DIR="artifacts"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please run trainer in a Python environment that has TensorFlow installed."
  exit 2
fi

echo "Running trainer in conda env: $TF_ENV"
echo "CSV: $CSV_PATH  Out: $OUT_DIR"

# Use conda run to execute trainer inside that environment. Adjust `--no-capture-output` if using older conda.
conda run -n "$TF_ENV" --no-capture-output python Models/trainer.py --csv "$CSV_PATH" --out-dir "$OUT_DIR" --epochs 3 --final-epochs 10

echo "Trainer finished. Check $OUT_DIR for artifacts."
