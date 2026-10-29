"""Thesis figure: linear CKA and angular distance between every pair of block outputs of one
model on one dataset, side by side.

Reads the matrices layer_priority.py writes to experiments/4_1-4_2_choosing_blocks/block_similarity/outputs/{stem}_cka.npy and
{stem}_angdist.npy. Both are [L+1, L+1] over hidden-state indices: 0 is the embedding output
(the input to block 1), k is the output of block k. Cell (i, j) is the pair a span from
boundary i to boundary j has to bridge.

Run from the project root:
    python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py --all     # all 10 chest + histopathology figures
    python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py --all --cka-only   # the same, linear CKA only
    python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py --grid    # CKA grids: one chest, one histopathology figure
    python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py
    python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py --stem owkin_phikon-v2_tcga_ut --title "Phikon-v2, TCGA-UT"

Writes experiments/4_1-4_2_choosing_blocks/block_similarity/figures/block_similarity_{stem}.pdf (vector, for LaTeX) and .png.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = Path("experiments/4_1-4_2_choosing_blocks/block_similarity/outputs")
OUT_DIR = Path("experiments/4_1-4_2_choosing_blocks/block_similarity/figures")

# Computer Modern (bundled with matplotlib), so the figure matches the thesis body text.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no unicode minus
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

TEXT_WIDTH_IN = 6.3  # A4 thesis text width; scale with \includegraphics[width=\textwidth]


def tick_positions(num_blocks):
    """Every 4th block for 24-block models / every 2nd for 12, plus the last."""
    step = 4 if num_blocks > 16 else 2
    ticks = list(range(0, num_blocks, step))
    if ticks[-1] != num_blocks - 1:
        ticks.append(num_blocks - 1)
    return ticks


def panel(ax, fig, mat, title, cmap, vmax, cbar_label=None):
    # Block-output axes: matrix index k is hidden_states[k], the
    # output of block k-1 (0-indexed). Index 0, the embedding output, is dropped, so cell (i, j)
    # is exactly the pair the span (skip_from=i, skip_to=j) has to bridge.
    mat = mat[1:, 1:]
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
    ticks = tick_positions(mat.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Block output $j$")
    ax.set_ylabel("Block output $i$")
    ax.set_title(title)
    ax.tick_params(length=2.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.6)
    return im


# The thesis figures: chest (3 models x 2 datasets) and histopathology (2 x 2). Stems follow
# layer_priority.py: lower-cased HF id with "/" -> "_", then the dataset key.
MODELS = {
    "facebook/dinov2-base": "DINOv2-B",
    "microsoft/rad-dino": "RAD-DINO",
    "microsoft/rad-dino-maira-2": "RAD-DINO-MAIRA-2",
    "google/vit-large-patch16-224": "ViT-L",
    "owkin/phikon-v2": "Phikon-v2",
}
DATASETS = {
    "chestxray14_multilabel": "ChestX-ray14",
    "chestxray_pneumonia": "Pneumonia",
    "patchcamelyon": "PatchCamelyon",
    "tcga_ut": "TCGA-UT",
}
# Domain-specific models first, the general-purpose one last in each group.
GROUPS = {
    "chest": (["microsoft/rad-dino", "microsoft/rad-dino-maira-2", "facebook/dinov2-base"],
              ["chestxray14_multilabel", "chestxray_pneumonia"]),
    "histo": (["owkin/phikon-v2", "google/vit-large-patch16-224"], ["patchcamelyon", "tcga_ut"]),
}


def stem_of(hf_id, dataset):
    return f"{hf_id.lower().replace('/', '_')}_{dataset}"


def plot(stem, title, out_dir):
    cka = np.load(INPUT_DIR / f"{stem}_cka.npy")
    ang = np.load(INPUT_DIR / f"{stem}_angdist.npy")

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9), constrained_layout=True)
    # Both colour scales put "similar" at the bright end: high CKA, low angular distance.
    panel(axes[0], fig, cka, "Linear CKA", "magma", 1.0)
    panel(axes[1], fig, ang, "Angular distance", "magma_r", float(np.ceil(ang.max() * 10) / 10))
    if title:
        fig.suptitle(title, fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"block_similarity_{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


def plot_cka_only(stem, title, out_dir):
    """Linear CKA on its own, sized for half the text width (two side by side, or one subfigure)."""
    cka = np.load(INPUT_DIR / f"{stem}_cka.npy")
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN / 2, 2.6), constrained_layout=True)
    panel(ax, fig, cka, title or "Linear CKA", "magma", 1.0, cbar_label="Linear CKA")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"block_cka_{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


def plot_cka_grid(out_dir):
    """The chest X-ray and histopathology CKA matrices as one figure per group: a row per
    model (general-purpose model last), a column per dataset, a colour bar underneath.
    Panels are the same size in both figures."""
    for group, (models, datasets) in GROUPS.items():
        # 1.8 in per row, plus room for the colour bar. No heading: the LaTeX caption names
        # the group.
        fig = plt.figure(figsize=(3.5, 1.8 * len(models) + 0.7), constrained_layout=True)
        axes = fig.subplots(len(models), len(datasets), squeeze=False)
        for r, hf_id in enumerate(models):
            for c, dataset in enumerate(datasets):
                ax = axes[r][c]
                mat = np.load(INPUT_DIR / f"{stem_of(hf_id, dataset)}_cka.npy")[1:, 1:]
                im = ax.imshow(mat, cmap="magma", vmin=0, vmax=1, origin="lower",
                               interpolation="nearest")
                ticks = tick_positions(mat.shape[0])
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.tick_params(length=2, labelsize=6)
                ax.set_title(f"{MODELS[hf_id]}, {DATASETS[dataset]}", fontsize=7)
                if r == len(models) - 1:
                    ax.set_xlabel("Block output $j$", fontsize=7.5)
                if c == 0:
                    ax.set_ylabel("Block output $i$", fontsize=7.5)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location="bottom",
                            fraction=0.12 / len(models), pad=0.02, shrink=0.7)
        cbar.set_label("Linear CKA")
        cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6)
        cbar.outline.set_linewidth(0.6)

        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            path = out_dir / f"block_cka_{group}_grid.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"saved {path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--all", action="store_true",
                        help="plot every chest + histopathology combination (10 figures)")
    parser.add_argument("--grid", action="store_true",
                        help="linear CKA grids, one figure per group (block_cka_{chest,histo}_grid.pdf)")
    parser.add_argument("--stem", default="google_vit-large-patch16-224_tcga_ut")
    parser.add_argument("--title", default="ViT-L, TCGA-UT", help="shown above the figure; '' for none")
    parser.add_argument("--cka-only", action="store_true",
                        help="linear CKA alone (block_cka_<stem>.pdf) instead of CKA + angular distance")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    draw = plot_cka_only if args.cka_only else plot

    if args.grid:
        plot_cka_grid(args.out_dir)
        return
    if not args.all:
        draw(args.stem, args.title, args.out_dir)
        return

    missing = []
    for models, datasets in GROUPS.values():
        for hf_id in models:
            for dataset in datasets:
                stem = stem_of(hf_id, dataset)
                if not (INPUT_DIR / f"{stem}_cka.npy").exists():
                    missing.append(stem)
                    continue
                draw(stem, f"{MODELS[hf_id]}, {DATASETS[dataset]}", args.out_dir)
    if missing:
        print(f"\n{len(missing)} combination(s) not in {INPUT_DIR} yet "
              f"(run experiments/4_1-4_2_choosing_blocks/block_similarity/launch_block_similarity.sh and sync):")
        for stem in missing:
            print(f"  {stem}")


if __name__ == "__main__":
    main()
