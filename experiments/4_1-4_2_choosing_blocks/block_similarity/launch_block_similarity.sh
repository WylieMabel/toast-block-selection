#!/bin/bash
# Recomputes the block-similarity matrices (CKA, angular distance) for every group-1 and
# group-2 model x dataset, one job each, on train-split calibration images. Run from the
# project root:
#   bash experiments/4_1-4_2_choosing_blocks/block_similarity/launch_block_similarity.sh
# Then sync experiments/4_1-4_2_choosing_blocks/block_similarity/outputs/ and plot, e.g.
#   python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py --stem owkin_phikon-v2_tcga_ut --title "Phikon-v2, TCGA-UT"

mkdir -p logs

GROUP1_MODELS=("facebook/dinov2-base" "microsoft/rad-dino" "microsoft/rad-dino-maira-2")
GROUP1_DATASETS=("chestxray14_multilabel" "chestxray_pneumonia" "imagenet-1k")
GROUP2_MODELS=("google/vit-large-patch16-224" "owkin/phikon-v2")
GROUP2_DATASETS=("patchcamelyon" "tcga_ut" "imagenet-1k")

submit() {
    sbatch --job-name="blocksim_$(basename "$1")_$2" \
        --export=ALL,MODEL="$1",DATASET="$2" experiments/4_1-4_2_choosing_blocks/block_similarity/run_block_similarity.sh
}

for m in "${GROUP1_MODELS[@]}"; do for d in "${GROUP1_DATASETS[@]}"; do submit "$m" "$d"; done; done
for m in "${GROUP2_MODELS[@]}"; do for d in "${GROUP2_DATASETS[@]}"; do submit "$m" "$d"; done; done
