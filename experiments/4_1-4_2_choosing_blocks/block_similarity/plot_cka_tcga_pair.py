"""One-off: linear CKA between block outputs for Phikon-v2 and ViT-L on TCGA-UT, side by side with
one shared colour bar. Sized for \\includegraphics[width=0.85\\linewidth] on a 6.3 in text width.
Same style and matrix convention as experiments/4_1-4_2_choosing_blocks/block_similarity/plot_block_similarity.py. Run from the repo root:
    python experiments/4_1-4_2_choosing_blocks/block_similarity/plot_cka_tcga_pair.py
Writes experiments/4_1-4_2_choosing_blocks/block_similarity/figures/block_cka_tcga_ut_pair.{pdf,png}.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "experiments/4_1-4_2_choosing_blocks/block_similarity")
from plot_block_similarity import INPUT_DIR, tick_positions  # noqa: E402  (also sets the rcParams)

PANELS = [("owkin_phikon-v2_tcga_ut", "Phikon-v2"), ("google_vit-large-patch16-224_tcga_ut", "ViT-L")]
WIDTH_IN = 0.85 * 6.3  # drawn at its natural size, so fonts match the other figures
OUT = Path("experiments/4_1-4_2_choosing_blocks/block_similarity/figures")

fig, axes = plt.subplots(1, len(PANELS), figsize=(WIDTH_IN, 2.35), constrained_layout=True)
for c, (ax, (stem, name)) in enumerate(zip(axes, PANELS)):
    mat = np.load(INPUT_DIR / f"{stem}_cka.npy")[1:, 1:]
    im = ax.imshow(mat, cmap="magma", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ticks = tick_positions(mat.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(length=2.5)
    ax.set_title(name)
    ax.set_xlabel("Block output $j$")
    if c == 0:
        ax.set_ylabel("Block output $i$")
cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.035, pad=0.02)
cbar.set_label("Linear CKA")
cbar.ax.tick_params(labelsize=8, length=2.5, width=0.6)
cbar.outline.set_linewidth(0.6)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"block_cka_tcga_ut_pair.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {OUT / f'block_cka_tcga_ut_pair.{ext}'}")
