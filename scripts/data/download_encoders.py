#!/usr/bin/env python3
"""
Download vision encoder models to HuggingFace cache.
Models are accessed via the MODEL_NAME2HF_NAME dictionary in src/toast/utils/dictionaries.py

Usage:
    python download_encoders.py                    # Download all models
    python download_encoders.py rad-dino          # Download specific model
    python download_encoders.py --verbose          # Print model structure details
"""

import os
import sys

# Set cache BEFORE importing transformers.
# IMPORTANT: these must match the env vars the sbatch job scripts export
# (see e.g. scripts/metrics/run_output_level_metrics.sh, scripts/run_pipeline_row_by_row.sh),
# which set TRANSFORMERS_CACHE to $WRITABLE_CACHE/transformers -- a DIFFERENT
# directory than the old BASE_CACHE_DIR/hub this script used to write to. Since the
# job scripts run fully offline (HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1), a
# model downloaded to the wrong cache dir here is invisible to them and shows up as
# LocalEntryNotFoundError on the cluster (this is why WinKawaks/vit-small-patch16-224
# failed even though this script reported it as downloaded successfully).
BASE_CACHE_DIR = os.environ.get("HF_HOME", "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache")
WRITABLE_CACHE = os.environ.get("WRITABLE_CACHE", "/cluster/home/mwylie/hf_writable")
os.environ["HF_HOME"] = BASE_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(WRITABLE_CACHE, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(WRITABLE_CACHE, "datasets")
os.environ["HF_MODULES_CACHE"] = os.path.join(WRITABLE_CACHE, "modules")

try:
    from transformers import AutoModel, AutoModelForImageClassification, AutoImageProcessor, AutoConfig
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Install with: pip install transformers")
    exit(1)

# Model name to HuggingFace identifier mapping
ENCODERS = {
    "vit-small-patch16-224": "WinKawaks/vit-small-patch16-224",
    "vit-tiny-patch16-224": "WinKawaks/vit-tiny-patch16-224",
    "vit-base-patch16-224": "google/vit-base-patch16-224",
    "deit-base-patch16-224": "facebook/deit-base-patch16-224",
    "dinov2-small": "facebook/dinov2-small",
    "rad-dino": "microsoft/rad-dino",
    "rad-dino-maira-2": "microsoft/rad-dino-maira-2",
    "phikon-v2": "owkin/phikon-v2",
}

# Models that also need AutoModelForImageClassification warmed (used by
# train_skipped_full.py's _get_orig_head for the orig_head_accuracy metric).
# Downloading via AutoModel already pulls the full checkpoint (weights are shared
# across wrapper classes), but this makes sure the classifier-head code path is
# actually exercised once while we have internet access, rather than discovering a
# problem with it for the first time on an offline compute node.
HAS_IMAGENET_HEAD = {"vit-small-patch16-224", "vit-tiny-patch16-224", "vit-base-patch16-224",
                      "deit-base-patch16-224"}


def download_encoder(model_name: str, hf_id: str):
    """Download and cache an encoder model."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"HuggingFace ID: {hf_id}")
    print(f"{'='*60}")

    try:
        # Load config
        print("  Loading config...")
        config = AutoConfig.from_pretrained(hf_id, output_hidden_states=True, return_dict=True)

        # Load image processor (if available for vision models)
        print("  Loading image processor...")
        try:
            processor = AutoImageProcessor.from_pretrained(hf_id)
        except Exception as e:
            print(f"  Note: No image processor found for {hf_id}")

        # Load model (caches to HF_HOME via environment variable)
        print("  Loading model...")
        model = AutoModel.from_pretrained(hf_id, config=config)

        if model_name in HAS_IMAGENET_HEAD:
            print("  Loading AutoModelForImageClassification (for orig_head_accuracy)...")
            AutoModelForImageClassification.from_pretrained(hf_id)

        # Print model info
        print(f"✓ Successfully downloaded {model_name}")
        print(f"  Model class: {model.__class__.__name__}")
        print(f"  Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}")
        print(f"  Num hidden layers: {config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'N/A'}")

        # Optionally print model structure for debugging
        if "--verbose" in sys.argv:
            print(f"\n  Model structure (first few layers):")
            count = 0
            for name, module in model.named_modules():
                if count > 20:
                    break
                if "layer" in name.lower() or "block" in name.lower() or "encoder" in name.lower():
                    print(f"    {name}")
                    count += 1

        return True

    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Determine which models to download
    skip_models = {"--verbose"}
    specified_models = [arg for arg in sys.argv[1:] if arg not in skip_models]

    if specified_models:
        models_to_download = [
            (name, hf_id) for name, hf_id in ENCODERS.items()
            if name in specified_models
        ]

        unknown = [m for m in specified_models if m not in ENCODERS]
        if unknown:
            print(f"Unknown models: {unknown}")
            print(f"Available models: {', '.join(ENCODERS.keys())}")
    else:
        models_to_download = list(ENCODERS.items())

    print(f"\nCache directory: {BASE_CACHE_DIR}")
    print(f"Models to download: {len(models_to_download)}")

    # Download models
    successful = 0
    failed = 0

    for model_name, hf_id in models_to_download:
        if download_encoder(model_name, hf_id):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Download complete: {successful} successful, {failed} failed")
    print(f"{'='*60}")
