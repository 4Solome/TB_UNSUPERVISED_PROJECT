import torch
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ttvae_model import TTVAE

device = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------
# Load pseudotime reference bounds (computed once from training latents)
# ---------------------------------------------------------------------
def _load_pseudotime_bounds():
    bounds_path = MODELS_DIR / "pseudotime_bounds.json"
    if bounds_path.exists():
        with open(bounds_path, "r") as f:
            bounds = json.load(f)
        z1_min = float(bounds["z1_min"])
        z1_max = float(bounds["z1_max"])
    else:
        z1_min = 0.0
        z1_max = 1.0
    return z1_min, z1_max

_Z1_MIN, _Z1_MAX = _load_pseudotime_bounds()

# ---------------------------------------------------------------------
# Build preprocessing (runtime-safe, no pickled preprocessor)
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
# Load trained TTVAE using saved training feature space
# ---------------------------------------------------------------------
def load_ttvae():
    feature_names_path = MODELS_DIR / "feature_names.json"
    weights_path = MODELS_DIR / "ttvae_best.pth"

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    d_in = len(feature_names)

    model = TTVAE(D_in=d_in).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, feature_names

# ---------------------------------------------------------------------
# Auxiliary artifacts
# ---------------------------------------------------------------------
def load_ood_threshold():
    threshold_path = MODELS_DIR / "ood_threshold.json"
    with open(threshold_path, "r") as f:
        return float(json.load(f)["ood_threshold"])

def load_cluster_model():
    return joblib.load(MODELS_DIR / "kmeans_model.joblib")

def load_pseudotime_bounds():
    return {"z1_min": _Z1_MIN, "z1_max": _Z1_MAX}

# ---------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------
def compute_latent(model, X):
    X_np = np.asarray(X, dtype=np.float32)
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()

def compute_pseudotime(latents):
    """
    Compute pseudotime relative to training latent bounds.
    Works for single-patient and cohort inference.
    """
    latents = np.asarray(latents, dtype=np.float32)
    z1 = latents[:, 0]
    pt = (z1 - _Z1_MIN) / (_Z1_MAX - _Z1_MIN + 1e-10)
    return np.clip(pt, 0.0, 1.0)

def check_ood(model, X, threshold):
    X_np = np.asarray(X, dtype=np.float32)
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        rec, _, _ = model(X_t)
    rec_np = rec.cpu().numpy()
    err = ((rec_np - X_np) ** 2).mean(axis=1)
    return err > threshold, err

def assign_cluster(kmeans, latents):
    latents = np.asarray(latents, dtype=np.float32)
    return kmeans.predict(latents)

def decode_latent(model, z):
    z_np = np.asarray(z, dtype=np.float32)
    z_t = torch.tensor(z_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        decoded = model.decode(z_t)
    return decoded.cpu().numpy()
