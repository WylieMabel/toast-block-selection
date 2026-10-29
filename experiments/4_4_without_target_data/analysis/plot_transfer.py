"""Plots the s5 translator-transfer results: per-skip accuracy / AUROC / AUPRC with the translator
fitted on the evaluation dataset itself (own-fit), on the other dataset of the same group
(in-domain donor), and on CIFAR-100 (out-of-domain donor). Run from the project root after
collect_accuracies.py:
    python experiments/4_4_without_target_data/analysis/plot_transfer.py                 # AUROC
    python experiments/4_4_without_target_data/analysis/plot_transfer.py --metric all
    python experiments/4_4_without_target_data/analysis/plot_transfer.py --head frozen

--head picks how the skipped model is scored: "retrained" trains a new head on the skipped model,
"frozen" reuses the head trained on the unskipped model unchanged (frozen_head_* columns), as when
condensing a deployed model without retraining. The default draws both.

Same layout as the earlier translator-reuse grid figures,
with the own-fit and in-domain lines both taken from s5, so they come from the same runs:
the own-fit curve on dataset X is the fitting rows of the combo whose donor is X.

The CIFAR-100 line comes from the s5 group*_crossfit_medical_from_cifar100 runs, falling back to
the older translator_reuse_cifar100.csv (accuracy only) until those exist.

Metrics are plotted x 100, as in the thesis tables. Figures are drawn at the printed width
(TEXT_WIDTH_IN), so the font sizes below are the sizes on the page.

Writes to experiments/4_4_without_target_data/figures/ (each as .pdf and .png):
    group{1,2}_translator_transfer_{metric}         retrained head, one panel per model x dataset
    group{1,2}_translator_transfer_{metric}_frozen  frozen head, same layout
    transfer_gap_summary.csv                             mean / worst (transfer - own-fit) per panel,
                                                         donor, metric and head
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

S5_INPUT = Path("experiments/4_4_without_target_data/analysis/collected/transfer_learning")
CIFAR_REUSE = Path("experiments/4_4_without_target_data/analysis/collected/transfer_learning/translator_reuse_cifar100.csv")
# No-skip baselines: s4 first (same pipeline as s5), the older final_accuracies runs as fallback.
BASELINE_SOURCES = ["experiments/4_1-4_2_choosing_blocks/medical/accuracies/*.csv",
                    "experiments/4_4_without_target_data/analysis/collected/group*_*.csv"]
OUT = Path("experiments/4_4_without_target_data/figures")

METRICS = {"accuracy": "Accuracy", "auroc": "AUROC", "auprc": "AUPRC"}
# Column prefix per head: the frozen head's scores are stored as frozen_head_<metric>_mean / _std.
HEADS = {"retrained": "", "frozen": "frozen_head_"}

# Domain-specific models first, the general-purpose one last in each group.
GROUPS = {
    1: (["chestxray14_multilabel", "chestxray_pneumonia"],
        ["microsoft/rad-dino", "microsoft/rad-dino-maira-2", "facebook/dinov2-base"]),
    2: (["patchcamelyon", "tcga_ut"],
        ["owkin/phikon-v2", "google/vit-large-patch16-224"]),
}
SCALE = 100  # plot AUROC etc. x 100, matching the tables
TEXT_WIDTH_IN = 6.3  # A4 thesis text width
FONT = dict(title=8, label=8, tick=7, legend=8)

# Categorical slots 1-3 in fixed order, plus a marker per series so identity is not colour alone
# (print / colour-blind readers). The baseline is neutral grey, not a series colour.
SERIES = {
    "own":   dict(color="#2a78d6", marker="o", label="Target (translator fitted on this dataset)"),
    "donor": dict(color="#1baf7a", marker="s", label="In-domain (fitted on the other dataset of the same modality)"),
    "cifar": dict(color="#eb6834", marker="^", label="CIFAR-100 (fitted on CIFAR-100)"),
}
MODEL_NAMES = {"microsoft/rad-dino": "RAD-DINO", "microsoft/rad-dino-maira-2": "RAD-DINO-MAIRA-2",
               "facebook/dinov2-base": "DINOv2-B", "google/vit-large-patch16-224": "ViT-L",
               "owkin/phikon-v2": "Phikon-v2"}
DATASET_NAMES = {"chestxray14_multilabel": "ChestX-ray14", "chestxray_pneumonia": "Pneumonia",
                 "patchcamelyon": "PatchCamelyon", "tcga_ut": "TCGA-UT"}
BASELINE_COLOR = "#6b6b68"


def skip_start(approx_layer):
    """H_s for a single-layer skip "[(i, i+1)]"; None for the baseline or longer / multiple skips."""
    spans = re.findall(r"\((\d+),\s*(\d+)\)", str(approx_layer))
    if len(spans) != 1:
        return None
    a, b = map(int, spans[0])
    return a if b - a == 1 else None


def with_hs(df):
    df = df.copy()
    df["H_s"] = df["approx_layer"].map(skip_start)
    return df.dropna(subset=["H_s"]).astype({"H_s": int}).sort_values("H_s")


def load_s5(input_dir):
    frames = [pd.read_csv(p) for p in sorted(input_dir.glob("group*_crossfit_*_from_*.csv"))]
    if not frames:
        raise SystemExit(f"no s5 results in {input_dir} - run analysis/collect_accuracies.py first")
    df = with_hs(pd.concat(frames, ignore_index=True))
    # A combo file holds the donor's fitting rows and the transfer rows; across the two combos of
    # a group every (model, dataset) gets both. Duplicates would mean a shard was merged twice.
    key = ["model", "dataset", "fit_dataset", "H_s"]
    dupes = df.duplicated(key, keep=False)
    if dupes.any():
        print(f"warning: {dupes.sum()} duplicate rows, keeping the last of each:\n"
              f"{df.loc[dupes, key].drop_duplicates().to_string(index=False)}")
        df = df.drop_duplicates(key, keep="last")
    return df


def load_baselines(metric):
    """(model, dataset) -> (no-skip value, source file) for the metric, s4 preferred."""
    col = f"{metric}_mean"
    rows = []
    for priority, pattern in enumerate(BASELINE_SOURCES):
        for path in glob.glob(pattern):
            df = pd.read_csv(path)
            if not {"model", "dataset", "approx_layer", col} <= set(df.columns):
                continue
            base = df[(df["approx_layer"] == "[]") & df[col].notna()]
            if "fit_dataset" in base:
                base = base[base["fit_dataset"].fillna(base["dataset"]) == base["dataset"]]
            for _, r in base.iterrows():
                rows.append((priority, r["model"], r["dataset"], r[col], path))
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["priority", "model", "dataset", "value", "source"])
    df = df.sort_values("priority").drop_duplicates(["model", "dataset"])
    return {(r.model, r.dataset): (r.value, r.source) for r in df.itertuples()}


def plot_series(ax, df, kind, col):
    mean, std = f"{col}_mean", f"{col}_std"
    if df.empty or mean not in df or df[mean].isna().all():
        return
    style = SERIES[kind]
    y = SCALE * df[mean]
    ax.plot(df["H_s"], y, color=style["color"], marker=style["marker"],
            markersize=3, linewidth=1.2, label=style["label"])
    sd = SCALE * df[std].fillna(0) if std in df else 0
    ax.fill_between(df["H_s"], y - sd, y + sd, color=style["color"], alpha=0.15, linewidth=0)


def tick_positions(blocks):
    """Every 2nd approximated block for 12-block models, every 4th for 24. The last block is
    not forced in: on 24-block models it lands two blocks after a tick and the labels collide."""
    return blocks[::4 if len(blocks) > 12 else 2]


def plot_group(group, metric, head, s5, cifar, baselines, out_dir):
    datasets, models = GROUPS[group]
    present = [m for m in models if (s5["model"] == m).any()]
    if not present:
        print(f"group {group}: no s5 rows yet, skipped")
        return []
    # The no-skip baseline is the same for both heads (the frozen head is the baseline's own head).
    metric_col = HEADS[head] + metric
    col = f"{metric_col}_mean"
    cifar_has_metric = col in cifar and cifar[col].notna().any()
    fig, axes = plt.subplots(len(present), len(datasets), squeeze=False,
                             figsize=(TEXT_WIDTH_IN, 2.4 * len(present) + 0.6))
    summary = []
    for r, model in enumerate(present):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            donor = next(d for d in datasets if d != dataset)
            here = s5[(s5["model"] == model) & (s5["dataset"] == dataset)]
            own = here[here["fit_dataset"] == dataset]
            transfer = here[here["fit_dataset"] == donor]
            ood = cifar[(cifar["model"] == model) & (cifar["dataset"] == dataset)
                        & (cifar["fit_dataset"] == "cifar100")]
            plot_series(ax, own, "own", metric_col)
            plot_series(ax, transfer, "donor", metric_col)
            plot_series(ax, ood, "cifar", metric_col)

            # The no-skip value goes in the title: the curves cross the baseline everywhere, so a
            # label on the line itself always collides with one of them.
            title = (f"{MODEL_NAMES.get(model, model)} on {DATASET_NAMES.get(dataset, dataset)},\n"
                     f"in-domain source: {DATASET_NAMES.get(donor, donor)}")
            base = baselines.get((model, dataset))
            if base is not None:
                ax.axhline(SCALE * base[0], color=BASELINE_COLOR, linestyle="--", linewidth=1)
                title += f"; original: {SCALE * base[0]:.1f}"
            ax.set_title(title, fontsize=FONT["title"])
            ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            # Ticks name the approximated block s -> s+1, the span notation used in the text.
            blocks = sorted(set(own["H_s"]) | set(transfer["H_s"]) | set(ood["H_s"]))
            ticks = tick_positions(blocks)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"${s}{{\\to}}{s + 1}$" for s in ticks])
            ax.tick_params(labelsize=FONT["tick"])
            if r == len(present) - 1:
                ax.set_xlabel(r"Approximated block $s \to e$", fontsize=FONT["label"])
            if c == 0:
                ax.set_ylabel(METRICS[metric] + (" (frozen head)" if head == "frozen" else ""),
                              fontsize=FONT["label"])

            for name, other in (("in-domain", transfer), ("cifar100", ood)):
                if col not in other or col not in own:
                    continue
                gap = (other.set_index("H_s")[col] - own.set_index("H_s")[col]).dropna()
                if not gap.empty:
                    summary.append(dict(group=group, metric=metric, head=head, model=model, dataset=dataset,
                                        donor=name, n_skips=len(gap), mean_gap=gap.mean(),
                                        worst_gap=gap.min(), worst_H_s=gap.idxmin()))
            if own.empty or transfer.empty:
                missing = [n for n, d in (("own-fit", own), ("transfer", transfer)) if d.empty]
                print(f"group {group}: {model} on {dataset}: no {' / '.join(missing)} rows yet")

    series = SERIES if cifar_has_metric else {k: v for k, v in SERIES.items() if k != "cifar"}
    handles = [plt.Line2D([], [], color=s["color"], marker=s["marker"], label=s["label"])
               for s in series.values()]
    handles.append(plt.Line2D([], [], color=BASELINE_COLOR, linestyle="--", label="Original model"))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=FONT["legend"], bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    suffix = "_frozen" if head == "frozen" else ""
    for ext in ("pdf", "png"):
        path = out_dir / f"group{group}_translator_transfer_{metric}{suffix}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=S5_INPUT, help="folder with the merged s5 CSVs")
    parser.add_argument("--out", type=Path, default=OUT)
    # AUROC is the thesis metric; accuracy / AUPRC only on request.
    parser.add_argument("--metric", choices=[*METRICS, "all"], default="auroc")
    parser.add_argument("--head", choices=[*HEADS, "both"], default="both")
    args = parser.parse_args()

    s5 = load_s5(args.input)
    # The CIFAR-100 donor runs inside s5 now (group*_crossfit_medical_from_cifar100); the older
    # translator_reuse_cifar100.csv (accuracy only, binary chestxray14) is the fallback.
    cifar = s5[s5["fit_dataset"] == "cifar100"]
    if cifar.empty:
        print(f"no CIFAR-100 donor rows in s5, using {CIFAR_REUSE.name}")
        cifar = with_hs(pd.read_csv(CIFAR_REUSE))
    plotted = {(m, d) for datasets, models in GROUPS.values() for m in models for d in datasets}
    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    heads = list(HEADS) if args.head == "both" else [args.head]
    for metric in (METRICS if args.metric == "all" else [args.metric]):
        baselines = load_baselines(metric)
        for (model, dataset), (_, source) in sorted(baselines.items()):
            if "/medical/" not in source and (model, dataset) in plotted:
                print(f"{metric}: no-skip baseline for {model} on {dataset} from the older runs ({source})")
        for head in heads:
            col = f"{HEADS[head]}{metric}_mean"
            if col not in s5 or s5[col].isna().all():
                print(f"{metric}, {head} head: not in the s5 results, skipped")
                continue
            if col not in cifar or cifar[col].isna().all():
                print(f"{metric}, {head} head: not in the CIFAR-100 donor results, CIFAR-100 line left out")
            for group in GROUPS:
                summary += plot_group(group, metric, head, s5, cifar, baselines, args.out)

    if summary:
        path = args.out / "transfer_gap_summary.csv"
        pd.DataFrame(summary).round(4).to_csv(path, index=False)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
