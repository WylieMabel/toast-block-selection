#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --job-name=s1_head_similarity
#SBATCH --output=logs/s1_head_similarity_%j.out.txt

# Head redundancy by linear CKA and angular distance on imagenet-1k (500 samples), swept over
# thresholds 0.05..0.95, for vitsmall, deitsmall, dinobase (facebook/dinov2-base) and vitlarge.
# Run from the project root:
#   sbatch experiments/A1_attention_heads/run_head_similarity.sh
# Subset of models / recompute existing scores:
#   MODELS="deitsmall vitlarge" FORCE=1 sbatch experiments/A1_attention_heads/run_head_similarity.sh
#
# Writes experiments/A1_attention_heads/results/head_similarity/head_keep_imagenet1k.csv
# (+ per-model .npy matrices) and configs/experiments_heads_<model>_imagenet1k.csv.

# 1. Paths, Hugging Face caches and PYTHONPATH (see env.sh; run from the repo root)
source env.sh

# 2. Activate venv
source $GPU_ENV/bin/activate
pip install -e .

# 3. Run
cd $PROJECT_DIR

MODELS=${MODELS:-"vitsmall deitsmall dinobase vitlarge"}
FORCE=${FORCE:-0}
force_flag=""
[ "$FORCE" = "1" ] && force_flag="--force"

python experiments/A1_attention_heads/head_similarity_thresholds.py \
    --models $MODELS \
    --dataset imagenet-1k \
    --num-samples 500 \
    $force_flag
