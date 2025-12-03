#!/usr/bin/env bash
# start.sh - Run Streamlit with environment guards to reduce TensorFlow native mutex/thread issues on macOS

export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_BLOCKTIME=1

echo "Starting Streamlit with TF env guards..."
streamlit run run.py
