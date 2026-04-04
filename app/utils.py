import torch
import numpy as np
import joblib
import json

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ttvae_model import TTVAE

device = torch.device("cpu")

# ---------------------------------------------------------------------
# Load pseudotime reference bounds (computed ONCE from training latents)
# ---------------------------------------------------------------------
with open("models/pseudotime_bounds.json", "r") as f:
    _pt_bounds = json.load(f)

_Z1_MIN = _pt_bounds["z1_min"]
_Z1_MAX = _pt_bounds["z1_max"]

# ---------------------------------------------------------------------
# Build preprocessing (no pickling, runtime safe)
# ---------------------------------------------------------------------
def build_preprocessor(continuous_cols, binary_cols, categorical_cols):

    cont_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", ohe)
    ])

    return ColumnTransformer([
        ("cont", cont_pipe, continuous_cols),
        ("bin", bin_pipe, binary_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

# ---------------------------------------------------------------------
# Load trained TTVAE using training feature space
# ---------------------------------------------------------------------
def load_ttvae():
    with open("models/feature_names.json", "r") as f:
        feature_names = json.load(f)

    D_in = len(feature_names)

    model = TTVAE(D_in=D_in).to(device)
    state = torch.load("models/ttvae_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, feature_names

# ---------------------------------------------------------------------
# Auxiliary artifacts
# ---------------------------------------------------------------------
def load_ood_threshold():
    with open("models/ood_threshold.json", "r") as f:
        return json.load(f)["ood_threshold"]

def load_cluster_model():
    return joblib.load("models/kmeans_model.joblib")

# ---------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------
def compute_latent(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()

def compute_pseudotime(latents):
    """
    Compute pseudotime relative to the training population.
    Works for single-patient and multi-patient inference.
    """
    z1 = latents[:, 0]
    return (z1 - _Z1_MIN) / (_Z1_MAX - _Z1_MIN + 1e-10)

def check_ood(model, X, threshold):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        rec, _, _ = model(X_t)
    err = ((rec.cpu().numpy() - X) ** 2).mean(axis=1)
    return err > threshold, err

def assign_cluster(kmeans, latents):
    return kmeans.predict(latents)
