#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --time=6:00:00
#SBATCH --output=logs/block_similarity_%j.out.txt

# One layer_priority.py run (block x block CKA / angular distance / head similarity) for
# MODEL x DATASET, on 500 train-split images (shuffle seed 1234, layer_priority's defaults).
# Submitted per combo by experiments/4_1-4_2_choosing_blocks/block_similarity/launch_block_similarity.sh. Overwrites (--force) the
# matrices in experiments/4_1-4_2_choosing_blocks/block_similarity/outputs/ that experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py reads.

set -e
source env.sh

cd "$PROJECT_DIR"
source $GPU_ENV/bin/activate

echo "Model: $MODEL | Dataset: $DATASET | Samples: ${NUM_SAMPLES:-500} | Start: $(date)"
python scripts/metrics/layer_priority.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num-samples "${NUM_SAMPLES:-500}" \
    --split train \
    --output-dir experiments/4_1-4_2_choosing_blocks/block_similarity/outputs \
    --force
echo "End: $(date)"
