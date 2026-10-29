"""Independent price-geometry and quantity-pattern encoders for reconstruction."""

import torch

from model.dual.encoder import OrderBookEncoder


class ModalityOrderBookEncoder(OrderBookEncoder):
    def __init__(self, *, modality, **config):
        if modality not in ("price", "quantity"):
            raise ValueError(f"Unknown modality: {modality}")
        super().__init__(**config)
        self.modality = modality
        gate = torch.zeros(16)
        gate[[0, 3]] = 1  # Normalized price and log gap.
        if modality == "quantity":
            gate = 1 - gate  # Raw/log quantity and twelve high-pass features.
        self.register_buffer("feature_gate", gate[None, :, None], persistent=False)

    def build_features(self, price, volume, mask, tick_norm):
        features = super().build_features(price, volume, mask, tick_norm)
        return features * self.feature_gate.to(features.dtype)
