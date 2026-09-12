"""
models.py — Shared model architectures for South Sudan Flood Prediction

Defines the GRU sequence model used for the month-ahead outlook, imported by
both train.py (training) and app.py (inference) so the architecture can never
drift between training and deployment.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

# The GRU is tiny; pinning to 1 thread avoids OpenMP thread-pool init crashes
# observed when torch is first touched from a non-main worker thread (e.g.
# a request-handler thread) alongside numpy/xgboost/lightgbm's own
# BLAS/OpenMP pools.
torch.set_num_threads(1)

# Dynamic (time-varying, fed through the GRU one month at a time) features.
# Deliberately excludes the target month's own contemporaneous climate values
# — the outlook model only ever sees months strictly before the one it is
# forecasting, unlike the tabular nowcast models which use the target month's
# own (live-updating) climate inputs.
SEQ_FEATURES = [
    "rainfall_mm", "soil_moisture_mm", "max_temperature_celsius",
    "min_temperature_celsius", "vapor_pressure_deficit_kPa", "ndvi", "flood",
]

# Static (per-county, non-time-varying) features, concatenated once after the
# GRU's final hidden state.
STATIC_FEATURES_SEQ = ["wetland_fraction", "elevation_m", "slope_deg"]

SEQ_LEN = 12  # months of history used to forecast the next month


class FloodGRU(nn.Module):
    """GRU over `SEQ_LEN` months of climate history + static terrain features
    -> single-logit month-ahead flood probability."""

    def __init__(self, n_seq_feat=len(SEQ_FEATURES), n_static_feat=len(STATIC_FEATURES_SEQ), hidden=32):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(input_size=n_seq_feat, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static_feat, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x_seq, x_static):
        _, h = self.gru(x_seq)
        h = h.squeeze(0)
        return self.head(torch.cat([h, x_static], dim=1)).squeeze(-1)


def load_outlook_bundle(bundle: dict) -> "FloodGRU":
    """Reconstruct a trained FloodGRU from the dict saved by train.py."""
    model = FloodGRU(
        n_seq_feat=len(bundle["seq_features"]),
        n_static_feat=len(bundle["static_features"]),
        hidden=bundle["hidden"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def predict_outlook(bundle: dict, model: "FloodGRU", seq_window, static_row) -> float:
    """Predict next-month flood probability given `SEQ_LEN` months of raw
    climate history (list/array of dicts or rows with SEQ_FEATURES) and one
    row of STATIC_FEATURES_SEQ values. Applies the training-time
    normalisation stored in `bundle`.
    """
    import numpy as np

    x_seq = np.array([[row[f] for f in bundle["seq_features"]] for row in seq_window], dtype=np.float32)
    x_seq = (x_seq - bundle["seq_mean"]) / bundle["seq_std"]
    x_static = np.array([static_row[f] for f in bundle["static_features"]], dtype=np.float32)
    x_static = (x_static - bundle["static_mean"]) / bundle["static_std"]

    with torch.no_grad():
        logit = model(
            torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0),
            torch.tensor(x_static, dtype=torch.float32).unsqueeze(0),
        )
        prob = torch.sigmoid(logit).item()
    return float(prob)


def predict_outlook_multistep(bundle: dict, model: "FloodGRU", seq_window, static_row, n_months: int) -> list:
    """Recursive multi-step outlook: forecast month t+1, then roll the window
    forward using that forecast as the newest "flood" signal (climate fields
    held at the county's historical medians, since no future climate data
    exists) and forecast t+2, and so on.

    This compounds uncertainty the further out it goes — each step's error
    feeds the next — so it's reported as an indicative trend, not a series of
    independent point forecasts. Returns a list of `n_months` probabilities.
    """
    window = [dict(row) for row in seq_window]
    probs = []
    for _ in range(n_months):
        p = predict_outlook(bundle, model, window, static_row)
        probs.append(p)
        next_month = dict(window[-1])  # carry forward last known climate values (seasonal proxy)
        next_month["flood"] = p
        window = window[1:] + [next_month]
    return probs
