# Which Blocks to Approximate? A Study of General-Purpose and Medical Vision Transformers

Code and results for the Master's thesis of **Mabel Wylie** (ETH Zurich, Department of Computer Science, 2026).
Advisor: Dr. Irene Cannistraci · Supervisor: Prof. Dr. Julia E. Vogt.

This repository extends **TOAST** (*Transformer Optimization using Adaptive and Simple Transformations*,
Cannistraci et al., [TMLR 2026](https://openreview.net/forum?id=fSwMCsBtTG)). TOAST replaces a span of
transformer blocks with a linear translator fitted in closed form, so no retraining is needed. This thesis asks
**which** blocks to approximate.

> We test whether similarity metrics can predict the performance drop of an approximation, across seven vision
> transformers and six datasets covering natural images, chest X-rays and histopathology. On natural images, the
> number of replaced blocks matters more to performance than their positions. Among spans of equal length, metrics
> that compare the final representations of the original and approximated models rank approximations best on
> natural images. Summing single-block drops ranks two-block approximations well without running them.
> Domain-specific medical models are far more robust than general-purpose models on the same data, and on them
> local metrics, including TOAST's own criterion, rank best. Finally, a public dataset from the same domain
> suffices both to fit the linear translators and to choose which single blocks to approximate.

## Repository layout

```
env.sh                     paths, Hugging Face caches, PYTHONPATH (sourced by the SLURM scripts)
src/toast/                 TOAST library (encoding, translators, skipped models, linear heads)
scripts/
  run_pipeline_row_by_row.sh   SLURM runner: encode -> fit translators -> train heads -> summarise, per config row
  calculate_accuracies.py      per-seed results -> mean/std per configuration
  metrics/                     similarity metrics for every span and their evaluation against the true drops
  data/                        dataset and checkpoint download scripts
experiments/               one folder per paper section, each with configs/, results/, accuracies/, analysis/, figures/
  4_1-4_2_choosing_blocks/     Sec. 4.1-4.2: ranking single spans (natural/ and medical/), block similarity
  4_3_combining/               Sec. 4.3: two-block approximations
  4_4_without_target_data/     Sec. 4.4: translators fitted on a source dataset
  A1_attention_heads/          App. A.1: head pruning
  A2_sublayers/                App. A.2: block vs. MLP vs. attention approximation (result files)
  tables/check_tables.ipynb    rebuilds every table in the thesis from the result CSVs
```

Inside each section, `configs/` holds the experiment sweeps that were run, and `results/` holds one row per seed.
`accuracies/` holds the mean and SD over seeds. `analysis/` and `figures/` hold everything derived from them.
All result CSVs needed for the paper are committed, so every table and figure can be rebuilt without a GPU.

## Installation

```bash
pip install -e ".[analysis]"     # library + matplotlib/scipy/jupyter for the analysis
```

## Paper → code

Run every command from the repo root unless noted otherwise.

| Paper item | Produced by | Reads |
|---|---|---|
| All tables (1–4, A.1–A.3, B.1–B.2, C.1–C.8) | `experiments/tables/check_tables.ipynb` | section `accuracies/`, `results/`, `analysis/`, `A2_sublayers/` |
| Table 1, C.1–C.3 (metric quality) | `experiments/4_1-4_2_choosing_blocks/metric_table.ipynb` | `natural/` and `medical/` `analysis/output_level_eval_*.csv` |
| Figure 1 (CKA, TCGA-UT) | `python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_cka_tcga_pair.py` | `block_similarity/outputs/*.npy` |
| Figures C.1, C.2 (CKA grids) | `python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py --grid` | `block_similarity/outputs/*.npy` |
| Per-section metric analyses | `natural/analysis/s2_metric_analysis.ipynb`, `medical/analysis/s4_metric_analysis.ipynb` | `analysis/grouped_metrics_*.csv` |
| Table 3, C.4–C.6 (two blocks) | `experiments/4_3_combining/analysis/s3_multiskip_analysis.ipynb` | `4_3_combining/accuracies/combined/`, `analysis/cka_angular_1_2_layer_spans.csv` |
| Table 4, C.8, Figures C.3–C.6 | `python experiments/4_4_without_target_data/analysis/plot_transfer.py` (+ `s5_source_choice.py`, `s5_random_angdist.py`) | `4_4_without_target_data/accuracies/`, `analysis/collected/` |
| Figure A.1 (head pruning) | `experiments/A1_attention_heads/analysis/head_similarity.ipynb` (run from its folder) | `A1_attention_heads/analysis/*.csv` |
| Tables A.2, A.3 (sublayers) | `experiments/tables/check_tables.ipynb` | `experiments/A2_sublayers/*.csv` |

## Re-running the experiments

The experiments ran on a SLURM cluster with RTX 4090 GPUs. Every sweep is a config CSV with one row per
approximation. The launchers in each section submit one job per config:

```bash
bash experiments/4_1-4_2_choosing_blocks/natural/launch_s2.sh        # Sec. 4.1, natural images
bash experiments/4_1-4_2_choosing_blocks/medical/launch_s4.sh        # Sec. 4.1-4.2, medical
bash experiments/4_3_combining/launch_s3.sh                          # Sec. 4.3
python experiments/4_4_without_target_data/make_configs.py && bash experiments/4_4_without_target_data/launch_s5.sh   # Sec. 4.4
bash experiments/A1_attention_heads/launch_s1.sh                     # App. A.1
```

Each job runs `scripts/run_pipeline_row_by_row.sh <config>`. For each row it runs
`src/toast/utils/encode_vision_full.py` (extract representations, fit the translator on 500 calibration images)
and then `src/toast/utils/train_skipped_full.py` (train a linear head per seed, score on the test split). Finally
`scripts/calculate_accuracies.py` summarises the seeds. A config under `experiments/<section>/configs/` writes into
that section's `results/` and `accuracies/`.

**Similarity metrics.** Metrics for every span are computed by `scripts/metrics/output_level_eval.py`, one job per
model/dataset combination, using `scripts/metrics/layer_priority.py`. They are evaluated against the true drops by
`scripts/metrics/build_group_stats.py`:

```bash
python experiments/4_1-4_2_choosing_blocks/medical/combine_s4.py     # merge the split s4 jobs first
bash experiments/4_1-4_2_choosing_blocks/natural/launch_s2_metrics.sh
bash experiments/4_1-4_2_choosing_blocks/medical/launch_s4_metrics.sh
python scripts/metrics/build_group_stats.py --group s2
python scripts/metrics/build_group_stats.py --group s4
bash experiments/4_1-4_2_choosing_blocks/block_similarity/launch_block_similarity.sh   # CKA / angular-distance matrices
```

The other scripts in `scripts/metrics/` (`join_and_evaluate.py`, `combined_stats_table.py`,
`safe_prediction_auc.py`, `add_safety_labels.py`, `cross_dataset_true_accuracy_correlations.py`,
`correct_metric_polarity.py`, `rebuild_cifar100_imagenet1k_plots.py`) are kept from the exploratory phase. They
refer to earlier runs whose outputs are not included, and are not needed for the paper.

### Environment

All paths, Hugging Face cache settings and `PYTHONPATH` are set in [`env.sh`](env.sh). The SLURM scripts source it
from the repo root, and you can run `source env.sh` before running anything by hand. Its defaults are the cluster
paths. Export a variable before sourcing to override it:

| Variable | Used for |
|---|---|
| `PROJECT_DIR` | repo checkout the jobs `cd` into |
| `TOAST_DATA_ROOT` | folder holding the `<dataset>_clean` DatasetDicts (defaults to `BASE_DIR`) |
| `HF_HOME` | shared Hugging Face cache with the model checkpoints |
| `WRITABLE_CACHE` | writable cache for datasets, transformers and remote-code modules |
| `GPU_ENV` | virtual environment activated by the SLURM scripts (not set by `env.sh`) |

The jobs run offline (`HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), so download everything first.

Prepare datasets with `scripts/data/download_medical_datasets.py` and checkpoints with
`scripts/data/download_encoders.py`. Embeddings, translators and heads are written to `data/` (gitignored).

## Models and datasets

| Model | Hugging Face key | Blocks |
|---|---|---|
| DeiT-S | `facebook/deit-small-patch16-224` | 12 |
| ViT-S | `WinKawaks/vit-small-patch16-224` | 12 |
| ViT-L | `google/vit-large-patch16-224` | 24 |
| DINOv2-B | `facebook/dinov2-base` | 12 |
| RAD-DINO | `microsoft/rad-dino` | 12 |
| RAD-DINO-MAIRA-2 | `microsoft/rad-dino-maira-2` | 12 |
| Phikon-v2 | `owkin/phikon-v2` | 24 |

Datasets: CIFAR-100, ImageNet-1k, ChestX-ray14 (14-label), Chest X-ray Pneumonia, PatchCamelyon and TCGA-UT. For
ImageNet-1k, the official validation set is split 80/20 into train and test, because the test labels are not
public.

## Citation

If you use this code, please also cite TOAST:

```bibtex
@article{cannistraci2026toast,
  title   = {{TOAST}: Transformer Optimization using Adaptive and Simple Transformations},
  author  = {Irene Cannistraci and Simone Antonelli and Emanuele Palumbo and Thomas M. Sutter and Emanuele Rodol{\`a} and Bastian Rieck and Julia E. Vogt},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=fSwMCsBtTG}
}
```

## License

MIT, see [LICENSE](LICENSE). The TOAST library code is © Irene Cannistraci.
