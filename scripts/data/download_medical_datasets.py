"""
Download real (non-MedMNIST) medical imaging datasets from the HF Hub and save each as a
clean on-disk DatasetDict, same convention as download_medmnist.py.

Roles per dataset, matching the rad-dino / rad-dino-maira-2 / phikon-v2 domain-overlap
analysis:
  - chestxray14    : full-resolution NIH ChestX-ray14 -- the literal "NIH-CXR" source dataset
                     rad-dino / rad-dino-maira-2 / MAIRA-2 were pretrained on (MedMNIST's
                     chestmnist is a 28px-derived toy version of this same source).
                     Collapsed to binary (any pathology vs "No Finding").
  - chestxray14_multilabel : same images, original 14-pathology multi-hot labels.
  - chestxray_pneu : full-resolution Kermany pediatric pneumonia CXR (Guangzhou Women and
                     Children's Medical Center) -- same source cohort as MedMNIST's
                     pneumoniamnist, just full-res, and genuinely unseen by rad-dino-maira-2
                     (its 6 training sources are all adult-skewed, US/Spain/Brazil).
  - patchcamelyon  : real H&E lymph-node histology (Camelyon16), never in phikon-v2's PANCAN-XL
                     pretraining set (TCGA/CPTAC/GTEx) -- same modality, unseen cohort.
  - bach           : real H&E breast histology (ICIAR BACH challenge), also outside PANCAN-XL --
                     a second, much smaller unseen-cohort histology set.
  - tcga_ut        : patches from TCGA whole-slide images (31 cancer types) -- unlike bach and
                     patchcamelyon, this one IS in-domain for phikon-v2 (TCGA is one of the
                     three PANCAN-XL pretraining sources, alongside CPTAC and GTEx), so it
                     fills the "seen by phikon-v2" role neither of those two can.

All of the above are plain parquet on the Hub (no trust_remote_code / loading script) except
tcga_ut, which is WebDataset-format tar shards -- still loadable via plain load_dataset()
with no trust_remote_code, just a different HF-side storage format. Contrast with
randall-lab/medmnist and albertvillanova/medmnist-v2, which require a now-unsupported loading
script.

Usage:
    python download_medical_datasets.py                  # everything in DATASETS below
    python download_medical_datasets.py chestxray14       # just one
"""

import os

# Set cache BEFORE importing datasets. load_dataset() otherwise caches to
# ~/.cache/huggingface on the small home-directory quota and dies with "Disk quota exceeded"
# partway through a multi-GB download (as happened here -- every one of these datasets is
# several hundred MB to a few GB). download_encoders.py splits this across HF_HOME and a
# separate /cluster/home/mwylie/hf_writable, but that path isn't writable from this host
# (login/client node) -- BASE_CACHE_DIR below already holds this project's existing
# downloads (mnist, cifar10/100, rad-dino, dinov2-base, ...), so everything just goes there.
BASE_CACHE_DIR = os.environ.get("HF_HOME", "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache")
os.environ["HF_HOME"] = BASE_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_CACHE_DIR, "transformers")
os.environ["HF_MODULES_CACHE"] = os.path.join(BASE_CACHE_DIR, "modules")
os.makedirs(BASE_CACHE_DIR, exist_ok=True)

from collections import Counter

import pyarrow as pa
from datasets import DatasetDict, Features, Sequence, Value, load_dataset

BASE_DOWNLOAD_DIR = os.environ.get("TOAST_DATA_ROOT", "/cluster/customapps/biomed/vogtlab/users/mwylie/toast")

# NIH ChestX-ray14's "No Finding" is the healthy label; everything else is some pathology.
# Collapsed to binary the same way download_medmnist.py's chestmnist handling is, so this
# slots into the same DATASET2NUM_CLASSES=2 pipeline without a multi-label training path.
def _chestxray14_label(example):
    labels = example['label']
    return 0 if list(labels) == ['No Finding'] else 1


# The original 14-pathology multi-label task (Wang et al. 2017), saved as a multi-hot vector in
# this fixed order; "No Finding" is all zeros. Kept as a separate dataset from chestxray14 above
# rather than replacing it, so the binary results already on disk stay reproducible.
CXR14_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]


def _chestxray14_multilabel(example):
    # Spaces normalised to underscores in case the mirror spells it "Pleural Thickening".
    present = {str(l).strip().replace(' ', '_') for l in example['label']}
    unknown = present - set(CXR14_PATHOLOGIES) - {"No_Finding"}
    # Fail loudly rather than silently dropping a misspelt pathology to all-zeros.
    if unknown:
        raise ValueError(f"unexpected ChestX-ray14 label(s) {unknown}; update CXR14_PATHOLOGIES")
    return [int(p in present) for p in CXR14_PATHOLOGIES]


def _patchcamelyon_label(example):
    return int(example['label'])


def _chestxray_pneu_label(example):
    return int(example['label'])


# For datasets whose raw label is a string (or otherwise undocumented) class name rather than
# a ready-to-use int -- BACH and tcga_ut so far. The mapping is built from the data itself, but
# that MUST be done over the union of all splits, not per split: building it separately per
# split is what silently remapped BACH's raw label 4 (present only in 'test', never in
# 'train') to index 0, colliding with whatever raw label 0 means in 'train'. Building it once
# over every split's raw values first is what surfaced that BACH's 'test' is 100% one raw
# label with zero overlap with 'train' -- i.e. that mirror's official test split isn't usable
# (see the loud warning below).
#
# label_col: pass explicitly when the dataset has other non-image columns besides the label
# (tcga_ut ships '__key__'/'__url__' webdataset metadata alongside 'label'); leave None to
# auto-detect the single non-image column, as BACH needs since its column name isn't fixed.
def _string_label_fn_factory(raw_splits, label_col=None):
    first = next(iter(raw_splits.values()))
    if label_col is None:
        label_col = next((c for c in first.column_names if c != 'image'), None)
        if label_col is None:
            raise ValueError(
                f"expected exactly one non-'image' column to use as the label, found "
                f"columns {first.column_names}. Pass label_col explicitly instead."
            )
    classes = sorted(set().union(*(set(ds[label_col]) for ds in raw_splits.values())))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"  using '{label_col}' as label column, {len(classes)} classes: {class_to_idx}")
    for split_name, ds in raw_splits.items():
        split_classes = set(ds[label_col])
        if len(split_classes) < 2:
            print(f"  WARNING: split '{split_name}' contains only raw label(s) "
                  f"{split_classes} -- not usable as a classification split on its own.")
    return label_col, lambda example: class_to_idx[example[label_col]]


# name -> (HF repo id, split name -> canonical split name, label extraction)
# label extraction is either a fixed callable(example) -> int, or None to mean
# "derive it from the loaded dataset" (BACH, where the column name isn't fixed up front).
DATASETS = {
    'chestxray14': {
        'repo': 'BahaaEldin0/NIH-Chest-Xray-14',
        'split_map': {'train': 'train', 'valid': 'val', 'test': 'test'},
        'label_fn': _chestxray14_label,
    },
    'chestxray14_multilabel': {
        'repo': 'BahaaEldin0/NIH-Chest-Xray-14',
        'split_map': {'train': 'train', 'valid': 'val', 'test': 'test'},
        'label_fn': _chestxray14_multilabel,
        'label_feature': Sequence(Value('int64'), length=len(CXR14_PATHOLOGIES)),
    },
    'patchcamelyon': {
        'repo': 'pavan316/PatchCamelyon',
        'split_map': {'train': 'train', 'valid': 'val', 'test': 'test'},
        'label_fn': _patchcamelyon_label,
    },
    'chestxray_pneumonia': {
        'repo': 'hf-vision/chest-xray-pneumonia',
        # Kermany's official val split is only 16 images; kept as its own split rather than
        # merged into train/test since that's the split the source dataset actually ships.
        'split_map': {'train': 'train', 'validation': 'val', 'test': 'test'},
        'label_fn': _chestxray_pneu_label,
    },
    'bach': {
        'repo': '1aurent/BACH',
        'split_map': {'train': 'train', 'test': 'test'},
        'label_fn': None,
    },
    'tcga_ut': {
        'repo': 'dakomura/tcga-ut',
        'hf_config': 'internal',  # patient-disjoint 70:15:15; 'external' is facility-held-out
        'image_col': 'jpg',
        # Confirmed via actual row inspection: label lives nested at json['label'] (e.g.
        # {'json': {'label': 'Skin_Cutaneous_Melanoma'}, '__key__': ..., '__url__': ...}), not
        # a flat 'label' column -- flattened to a top-level 'label' string column below before
        # the generic union-of-classes logic (which needs a flat, hashable column) can run.
        'label_path': ('json', 'label'),
        'split_map': {'train': 'train', 'valid': 'val', 'test': 'test'},
        'label_fn': None,
        'label_col': 'label',  # explicit: '__key__'/'__url__' webdataset metadata columns
                                # would otherwise break the single-non-image-column auto-detect
    },
}


def _relabel(dataset, label_fn, label_feature):
    """Replace every non-image column with a single 'label' column, without touching images.

    The old version used dataset.map(...) followed by cast_column(...). Both rewrite the
    WHOLE table, images included, into a new HF cache file (cast_column on a non-image
    column goes through Dataset.cast, which is itself a map). For chestxray14 that's
    ~45GB per pass on top of save_to_disk's own copy, and it ran the disk out of space
    partway through. Here only the small label columns are read, the new 'label' column is
    built directly with the target Arrow type (an explicit type also avoids the
    patchcamelyon problem where a reused 'label' kept its original bool type), and it is
    attached with remove_columns/add_column. Both of those only change the table metadata,
    so the images are written exactly once, by save_to_disk. Images are never decoded either:
    select_columns leaves the image column out before rows are read.
    """
    other_cols = [c for c in dataset.column_names if c != 'image']
    labels = [label_fn(row) for row in dataset.select_columns(other_cols)]
    arrow_type = Features({'label': label_feature}).arrow_schema.field('label').type
    return (
        dataset.remove_columns(other_cols)
        .add_column('label', pa.array(labels, type=arrow_type))
    )


def prepare_dataset(name: str, download_root: str = None):
    if download_root is None:
        download_root = BASE_DOWNLOAD_DIR

    print(f"\n{'='*50}")
    print(f"Processing {name.upper()}")
    print(f"{'='*50}")

    spec = DATASETS[name]
    output_dir = os.path.join(download_root, f"{name}_clean")
    os.makedirs(output_dir, exist_ok=True)

    try:
        image_col = spec.get('image_col', 'image')
        raw_splits = {}
        for hf_split, canonical_split in spec['split_map'].items():
            print(f"  Loading {hf_split} -> {canonical_split}...")
            ds = load_dataset(spec['repo'], name=spec.get('hf_config'), split=hf_split)
            if image_col != 'image':
                ds = ds.rename_column(image_col, 'image')
            label_path = spec.get('label_path')
            if label_path is not None:
                outer_col, inner_key = label_path
                ds = ds.map(
                    lambda value: {'label': value[inner_key]},
                    input_columns=[outer_col],
                    remove_columns=[outer_col],
                    desc=f"  flattening nested label for {canonical_split}",
                )
            raw_splits[canonical_split] = ds

        label_fn = spec['label_fn']
        if label_fn is None:
            _, label_fn = _string_label_fn_factory(raw_splits, label_col=spec.get('label_col'))

        label_feature = spec.get('label_feature', Value('int64'))
        processed_splits = {}
        for canonical_split, dataset in raw_splits.items():
            print(f"  labeling {canonical_split}...")
            processed_splits[canonical_split] = _relabel(dataset, label_fn, label_feature)

        processed = DatasetDict(processed_splits)
        processed.save_to_disk(output_dir)
        print(f"✓ Saved to {output_dir}")

        # Majority-class rate: the accuracy a constant predictor gets for free. A probe
        # landing on this number has learned nothing, same diagnostic as download_medmnist.py.
        for split, data in processed.items():
            print(f"  {split}: {len(data)} samples")
            if 'label_feature' in spec:
                # Multi-hot: no majority class, report per-label positive rate instead.
                labels = data['label']
                pos = [sum(row[j] for row in labels) / len(data) for j in range(len(labels[0]))]
                print("    positive rate per label: "
                      + ", ".join(f"{p}:{r:.4f}" for p, r in zip(CXR14_PATHOLOGIES, pos)))
                print(f"    no finding (all zeros): "
                      f"{sum(1 for row in labels if not any(row)) / len(data):.4f}")
                continue
            counts = Counter(data['label'])
            majority = max(counts.values()) / len(data)
            print(f"    {len(counts)} classes, distribution: "
                  + ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)))
            print(f"    majority-class rate: {majority:.4f} "
                  f"(a probe scoring this has learned nothing)")

    except Exception as e:
        print(f"✗ Error processing {name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == "__main__":
    import sys

    names_to_download = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    failed = []
    for name in names_to_download:
        if name.lower() in DATASETS:
            if not prepare_dataset(name.lower()):
                failed.append(name)
        else:
            print(f"Unknown dataset: {name}. Available: {', '.join(DATASETS.keys())}")
            failed.append(name)

    # Used to print success unconditionally, even right after a traceback.
    if failed:
        print(f"\n✗ Failed: {', '.join(failed)}")
        sys.exit(1)
    print(f"\n✓ All datasets prepared!")
