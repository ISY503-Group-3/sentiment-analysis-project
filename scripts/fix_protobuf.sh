#!/usr/bin/env bash
# fix_protobuf.sh
# Run this inside your conda env that you want TensorFlow installed in (e.g., `conda activate tf`).
# It downgrades protobuf to a compatible version and reinstalls TF wheels.

set -euo pipefail

echo "Ensure you're running this inside the conda env where TF should be installed (e.g., conda activate tf)."
read -p "Press Enter to continue or Ctrl+C to abort..."

echo "Uninstalling protobuf (if present)..."
python -m pip uninstall -y protobuf || true

echo "Installing protobuf==3.20.3..."
python -m pip install protobuf==3.20.3

echo "Reinstalling TensorFlow wheels (force-reinstall)..."
python -m pip install --upgrade --force-reinstall tensorflow-macos tensorflow-metal || true

echo "Verifying installations..."
python - <<'PY'
import sys
try:
    import google.protobuf
    print('protobuf version:', getattr(google.protobuf,'__version__','unknown'))
except Exception as e:
    print('protobuf import FAILED:', e)

try:
    import tensorflow as tf
    print('tensorflow version:', tf.__version__)
except Exception as e:
    print('tensorflow import FAILED:', e)
    sys.exit(1)

print('Sanity checks done; if no errors above, try running ./start.sh again.')
PY

echo "Done. If issues persist, consider creating a fresh conda env as documented in the README."