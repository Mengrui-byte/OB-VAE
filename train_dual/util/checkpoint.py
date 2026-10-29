"""Epoch-boundary checkpoints, random states and complete source snapshots."""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch

from util.model_sources import freeze_model_sources, model_source_hashes


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def source_hashes():
    paths = [*sorted((PROJECT_ROOT / "train_dual").rglob("*.py")),
             PROJECT_ROOT / "util/dataset.py", PROJECT_ROOT / "util/scaling.py",
             PROJECT_ROOT / "util/model_sources.py"]
    return {**model_source_hashes(PROJECT_ROOT), **{
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths}}


def freeze_sources(destination):
    destination = Path(destination)
    hashes = freeze_model_sources(destination, PROJECT_ROOT)
    for name, digest in source_hashes().items():
        path = PROJECT_ROOT / name
        target = destination / name
        if not name.startswith("model/"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        hashes[name] = digest
    (destination / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return source_hashes()


def atomic_save(payload, path):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def random_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random_state(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def load_for_inference(path, device="cpu"):
    """Load a trusted local best.pt/last.pt and its invertible normalizer.

    Architecture code must match; run from the experiment's source_snapshot when
    the working model has subsequently changed. Returns (model, normalizer).
    """
    from model.dual import OrderBookVAE
    from train_dual.util.data import DualNormalizer

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported dual checkpoint format")
    if "direct_readout_config" in payload:
        from model.autoencoder.direct_readout_vae import DirectReadoutOrderBookVAE
        if "regional_phase_config" not in payload or any(key in payload for key in [
                "quantity_context_config", "fine_phase_config", "paired_logit_config", "logit_readout_config"]):
            raise ValueError("Direct quantity readout requires its explicit regional source family")
        model = DirectReadoutOrderBookVAE(**payload["direct_readout_config"],
            **payload["regional_phase_config"], **payload["unified_vae_config"])
    elif "quantity_context_config" in payload:
        from model.autoencoder.quantity_context_vae import QuantityContextOrderBookVAE
        model = QuantityContextOrderBookVAE(**payload["quantity_context_config"], **payload["unified_vae_config"])
    elif "fine_phase_config" in payload:
        from model.autoencoder.fine_phase_vae import FinePhaseOrderBookVAE
        model = FinePhaseOrderBookVAE(**payload["fine_phase_config"], **payload["unified_vae_config"])
    elif "paired_logit_config" in payload:
        from model.autoencoder.paired_logit_vae import PairedLogitOrderBookVAE
        model = PairedLogitOrderBookVAE(**payload["paired_logit_config"], **payload["unified_vae_config"])
    elif "logit_readout_config" in payload:
        from model.autoencoder.logit_readout_vae import LogitReadoutOrderBookVAE
        model = LogitReadoutOrderBookVAE(**payload["logit_readout_config"], **payload["unified_vae_config"])
    elif "regional_phase_config" in payload:
        from model.autoencoder.regional_phase_vae import RegionalPhaseOrderBookVAE
        model = RegionalPhaseOrderBookVAE(**payload["regional_phase_config"], **payload["unified_vae_config"])
    elif "local_phase_config" in payload:
        from model.autoencoder.local_phase_vae import LocalPhaseOrderBookVAE
        model = LocalPhaseOrderBookVAE(**payload["local_phase_config"], **payload["unified_vae_config"])
    elif "shape_mass_config" in payload:
        from model.autoencoder.shape_mass_vae import ShapeMassOrderBookVAE
        model = ShapeMassOrderBookVAE(**payload["shape_mass_config"], **payload["unified_vae_config"])
    elif "sparse_address_config" in payload:
        from model.autoencoder.sparse_address_vae import SparseAddressOrderBookVAE
        model = SparseAddressOrderBookVAE(**payload["sparse_address_config"], **payload["unified_vae_config"])
    elif "local_hint_config" in payload:
        from model.autoencoder.local_hint_vae import LocalHintOrderBookVAE
        model = LocalHintOrderBookVAE(**payload["local_hint_config"], **payload["unified_vae_config"])
    elif "region_grid_config" in payload:
        from model.autoencoder.region_grid_vae import RegionGridOrderBookVAE
        model = RegionGridOrderBookVAE(**payload["region_grid_config"], **payload["unified_vae_config"])
    elif "time_layout_config" in payload:
        from model.autoencoder.time_layout_vae import TimeLayoutOrderBookVAE
        model = TimeLayoutOrderBookVAE(**payload["time_layout_config"], **payload["unified_vae_config"])
    elif "query_decoder_config" in payload:
        from model.autoencoder.query_book_vae import QueryOrderBookVAE
        model = QueryOrderBookVAE(**payload["query_decoder_config"], **payload["unified_vae_config"])
    elif "peak_aux_config" in payload:
        from model.autoencoder.peak_book_vae import PeakAuxOrderBookVAE
        model = PeakAuxOrderBookVAE(**payload["unified_vae_config"])
    elif "phase_decoder_config" in payload:
        from model.autoencoder.phase_book_vae import PhaseDetailOrderBookVAE
        model = PhaseDetailOrderBookVAE(**payload["phase_decoder_config"], **payload["unified_vae_config"])
    elif "gradient_route_config" in payload:
        from model.experimental.gradient_route import GradientRouteOrderBookVAE
        model = GradientRouteOrderBookVAE(**payload["gradient_route_config"], **payload["unified_vae_config"])
    elif "unified_vae_config" in payload:
        from model.autoencoder.unified_book_vae import UnifiedOrderBookVAE
        model = UnifiedOrderBookVAE(**payload["unified_vae_config"])
    elif "global_local_merge_config" in payload:
        from model.experimental.global_into_local import GlobalIntoLocalOrderBookVAE
        model = GlobalIntoLocalOrderBookVAE(**payload["global_local_merge_config"], **payload["model_config"])
    elif "window_matrix_config" in payload:
        from model.experimental.full_window_matrix import FullWindowMatrixOrderBookVAE
        model = FullWindowMatrixOrderBookVAE(**payload["window_matrix_config"], **payload["model_config"])
    elif "frame_matrix_config" in payload:
        from model.experimental.frame_matrix_state import FrameMatrixOrderBookVAE
        model = FrameMatrixOrderBookVAE(**payload["frame_matrix_config"], **payload["model_config"])
    elif "tensor_cluster_config" in payload:
        from model.experimental.tensor_cluster_state import TensorClusterOrderBookVAE
        model = TensorClusterOrderBookVAE(**payload["tensor_cluster_config"], **payload["model_config"])
    elif "local_paired_config" in payload:
        from model.experimental.local_paired_state import LocalPairedOrderBookVAE
        model = LocalPairedOrderBookVAE(**payload["local_paired_config"], **payload["model_config"])
    elif "paired_matrix_config" in payload:
        from model.experimental.paired_matrix_state import PairedMatrixOrderBookVAE
        model = PairedMatrixOrderBookVAE(**payload["paired_matrix_config"], **payload["model_config"])
    elif "rq_config" in payload:
        from model.experimental.rq_window_state import RQWindowStateVAE
        model = RQWindowStateVAE(**payload["rq_config"], **payload["learned_state_config"],
                                 **payload["residual_pca_config"], **payload["pca_state_config"], **payload["model_config"])
    elif "learned_state_config" in payload:
        from model.experimental.learned_window_state import LearnedWindowStateVAE
        model = LearnedWindowStateVAE(**payload["learned_state_config"], **payload["residual_pca_config"],
                                     **payload["pca_state_config"], **payload["model_config"])
    elif "residual_pca_config" in payload:
        from model.experimental.pca_residual_state import ResidualPCAWindowVAE
        model = ResidualPCAWindowVAE(**payload["residual_pca_config"], **payload["pca_state_config"], **payload["model_config"])
    elif "pca_state_config" in payload:
        from model.experimental.pca_state import PCAWindowVAE
        model = PCAWindowVAE(**payload["pca_state_config"], **payload["model_config"])
    elif "window_state_config" in payload:
        from model.experimental.window_state import WindowStateVAE
        model = WindowStateVAE(**payload["window_state_config"], **payload["model_config"])
    elif "fusion_variant" in payload:
        from model.experimental.dual_fusion import FusedOrderBookVAE
        model = FusedOrderBookVAE(fusion_variant=payload["fusion_variant"],
                                  fusion_seed=payload.get("refinement_seed",42),
                                  **payload["model_config"])
    elif "research_variant" in payload:
        from model.experimental.dual_long import ResearchOrderBookVAE
        model = ResearchOrderBookVAE(variant=payload["research_variant"],
                                    refinement_seed=payload.get("refinement_seed",42),
                                    **payload["model_config"])
    else:
        model = OrderBookVAE(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    if "precision_policy_config" in payload:
        from model.layers.sensitive_precision import apply_sensitive_precision_
        apply_sensitive_precision_(model, **payload["precision_policy_config"])
    if "branch_precision_config" in payload:
        from model.layers.branch_precision import apply_branch_precision_
        apply_branch_precision_(model, **payload["branch_precision_config"])
    model.to(device).eval()
    normalization = payload["config"]["normalization"]
    normalizer = DualNormalizer(tick_size=normalization["tick_size"],
                                price_scale=normalization["price_scale"])
    return model, normalizer


def load_initial_weights(model, path, model_config):
    """Warm-start a new experiment without restoring optimizer or RNG state."""
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or payload.get("model_config") != model_config:
        raise ValueError("Initial checkpoint model configuration differs from this run")
    model.load_state_dict(payload["model"], strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
