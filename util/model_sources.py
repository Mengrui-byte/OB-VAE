"""Freeze canonical model sources together with legacy import adapters."""

import hashlib
import shutil
from pathlib import Path


COMPATIBILITY_PATHS = (
    "model/ae.py",
    "model/conv.py",
    "model/decoder.py",
    "model/encoder.py",
    "model/geometry.py",
    "model/peaks.py",
    "model/quantity.py",
    "check/backbone/architecture.py",
    "check/backbone/objective.py",
    "check/variants.py",
    "check/keypoint_fusion.py",
    "check/latent_keypoints.py",
    "check/ordered_price.py",
    "check/pair_refine.py",
    "check/main_peak_objective.py",
    "check/campaign_lib/models.py",
    "check/campaign_lib/operators.py",
    "check/combination/architecture.py",
    "check/peak_ablation/architecture.py",
    "check/refinement/architecture.py",
    "check/state_lib/architecture.py",
    "check/peak_information/architecture.py",
    "check/full_mlp/architecture.py",
)


def model_source_hashes(project_root=None):
    """Hash the actual implementation, rather than just its import adapter."""
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "model").rglob("*.py"))
    }


def freeze_model_sources(destination, project_root=None):
    """Include all model dependencies before computing a new run's source hash.

    Use only while preparing a new run, never to update an existing experiment.
    Returns hashes of the copied Python files relative to the snapshot root.
    """
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    destination = Path(destination)
    paths = [
        *sorted((root / "model").rglob("*.py")),
        *(root / name for name in COMPATIBILITY_PATHS if (root / name).is_file()),
        root / "util/model_sources.py",
    ]
    hashes = {}
    for path in paths:
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes
