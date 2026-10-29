# Environment for the TOAST thesis pipeline. Source it from the repo root before running anything:
#   source env.sh
# The SLURM scripts (scripts/run_pipeline_row_by_row.sh, scripts/metrics/run_output_level_metrics.sh,
# experiments/*/run_*.sh) source it themselves. Every value can be overridden by exporting it first;
# the defaults are the paths used on the cluster.

# --- paths ---
export PROJECT_DIR="${PROJECT_DIR:-/cluster/home/mwylie/toast-extensions}"           # this repo's checkout
export BASE_DIR="${BASE_DIR:-/cluster/customapps/biomed/vogtlab/users/mwylie/toast}"  # project storage
export TOAST_DATA_ROOT="${TOAST_DATA_ROOT:-$BASE_DIR}"  # holds the <dataset>_clean DatasetDicts (scripts/data/)
export WRITABLE_CACHE="${WRITABLE_CACHE:-/cluster/home/mwylie/hf_writable}"
# Virtual environment the SLURM scripts activate (source $GPU_ENV/bin/activate). Set it here or in your shell:
# export GPU_ENV=/path/to/venv

# --- Hugging Face caches ---
# Checkpoints are read from the shared HF_HOME. Datasets, transformers and remote-code modules are cached
# in WRITABLE_CACHE, because the jobs cannot write to HF_HOME. Jobs run fully offline, so every model and
# dataset must already be downloaded (scripts/data/download_encoders.py, download_medical_datasets.py).
export HF_HOME="${HF_HOME:-$BASE_DIR/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WRITABLE_CACHE/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$WRITABLE_CACHE/transformers}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$WRITABLE_CACHE/modules}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- quiet noisy libraries (progress bars / load reports) ---
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export DATASETS_VERBOSITY="${DATASETS_VERBOSITY:-error}"

# --- imports ---
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_DIR/src"

mkdir -p "$WRITABLE_CACHE" 2>/dev/null || true
