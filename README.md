**Overview**
- **Purpose**: This repository contains a Streamlit port of the `Model_Architect.ipynb` notebook for sentiment analysis, plus utilities to run heavy TensorFlow training outside the web process and to persist evaluation artifacts.
- **Main files**: `run.py` (Streamlit app), `Models/model_architect_full.py` (converted notebook helpers), `Models/trainer.py` (CLI trainer), `start.sh` (Streamlit launcher with environment guards), `artifacts/` (saved evaluation PNGs and JSON summaries).

**Quick Start (recommended for macOS / zsh with Apple Silicon)**
- **Make start script executable**: run:

```bash
chmod +x start.sh
```

- **Start Streamlit with recommended environment guards**: this avoids noisy TF native logs and reduces thread contention.

```bash
./start.sh
```

- **Alternative — run Streamlit directly** (export guards first):

```bash
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_BLOCKTIME=1
streamlit run run.py
```

**Run training (isolated, recommended)**
- **Why**: On macOS/arm64 the in-process TensorFlow import may crash due to native protobuf or binary incompatibilities. Use `Models/trainer.py` in a compatible conda env to keep Streamlit stable.
- **Create conda env (example using Miniforge/conda-forge)**:

```bash
conda create -n sa-train python=3.10 -y
conda activate sa-train
pip install -r requirements.txt   # or install packages below
```

- **Install macOS TF (Apple Silicon)**:

```bash
pip install tensorflow-macos tensorflow-metal
# If you hit protobuf errors, pin protobuf to a compatible version
pip install "protobuf==3.20.3"
```

- **Run the trainer CLI** (from project root):

```bash
conda run -n sa-train --no-capture-output python Models/trainer.py --data data/input_demo.csv --out-dir artifacts
```

**Artifacts and evaluation visuals**
- **Location**: evaluation images and summary JSON are saved to `artifacts/` (e.g. `artifacts/evaluation_roc_curve.png`, `artifacts/evaluation_confusion_matrix.png`, `artifacts/evaluation_confidence_hist.png`, `artifacts/evaluation_summary.json`).
- **Streamlit behavior**: The app attempts to render evaluation figures inline when possible, but to keep the UI stable the app prefers saved PNGs. Use the bottom-page expander "Saved evaluation artifacts (from `artifacts/`)" to view saved images. There is also a button to generate placeholder evaluation figures from code (no TF required).

**Developer notes & utilities**
- **Lazy TensorFlow import**: `Models/model_architect_full.ensure_tf()` defers importing TensorFlow until necessary.
- **Trainer CLI**: `Models/trainer.py` runs tournaments and final training outside Streamlit and writes artifacts to disk. Useful for running training in a separate conda env.
- **Placeholder figures**: If you don't have training artifacts, the UI's bottom expander provides a button to generate TF-free placeholder figures using `Models.model_architect_full.generate_placeholder_evaluation_figures()`.

**Troubleshooting**
- **Streamlit crashes on import / libprotobuf FATAL**: This commonly happens when your Python version or installed TensorFlow wheel is incompatible (especially on macOS/arm64 + Python 3.13). Fix options:
	- Create a conda env with Python 3.10 or 3.11 and install `tensorflow-macos` + `tensorflow-metal` there.
	- Pin `protobuf==3.20.3` if you see protobuf native version conflicts.
	- Run training in `Models/trainer.py` inside the compatible env instead of importing TF inside Streamlit.
- **If `scikit-learn` import fails**: install the correct package with `pip install scikit-learn` (the `sklearn` meta-package on PyPI is deprecated).

**Project structure (key files)**
- **`run.py`**: Streamlit app entrypoint (Light quick mode and Full notebook mode).
- **`Models/model_architect_full.py`**: Notebook conversion — data loading, preprocessing, model builders, tournament runner, evaluation and figure-saving helpers.
- **`Models/trainer.py`**: CLI trainer for isolated runs (accepts tokenizer/dataset paths and writes artifacts).
- **`start.sh`**: Helper that exports environment variables before launching Streamlit.
- **`scripts/`**: helper scripts (placeholder generators and conda-run examples).

**Common commands**
- Run Streamlit (recommended via start script):

```bash
./start.sh
```

- Run trainer in a conda env:

```bash
conda run -n sa-train --no-capture-output python Models/trainer.py --data data/input_demo.csv --out-dir artifacts
```

**License & acknowledgements**
- This repository is a course project scaffold; check with your team for licensing preferences before publishing.

If you'd like, I can also:
- add a `requirements.txt` with pinned packages, or
- add a small button to the UI that saves generated placeholder figures into `artifacts/` automatically.


**Github Link 
- https://github.com/ISY503-Group-3/sentiment-analysis-project
