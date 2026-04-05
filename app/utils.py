import os
import json
import torch
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ttvae_model import TTVAE

# ============================================================
# PATH HANDLING (CORRECT FOR YOUR REPO STRUCTURE)
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

device = torch.device("cpu")

# ============================================================
# LOAD PSEUDOTIME REFERENCE BOUNDS
# ============================================================
PT_BOUNDS_PATH = os.path.join(MODELS_DIR, "pseudotime_bounds.json")

with open(PT_BOUNDS_PATH, "r") as f:
    _pt_bounds = json.load(f)

_Z1_MIN = _pt_bounds["z1_min"]
_Z1_MAX = _pt_bounds["z1_max"]

# ============================================================
# BUILD PREPROCESSOR (NO PICKLING)
# ============================================================
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

# ============================================================
# LOAD TRAINED TTVAE
# ============================================================
def load_ttvae():

    feature_names_path = os.path.join(MODELS_DIR, "feature_names.json")
    weights_path = os.path.join(MODELS_DIR, "ttvae_best.pth")

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    D_in = len(feature_names)

    model = TTVAE(D_in=D_in).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, feature_names

# ============================================================
# LOAD OTHER ARTIFACTS
# ============================================================
def load_cluster_model():
    return joblib.load(os.path.join(MODELS_DIR, "kmeans_model.joblib"))

def load_ood_threshold():
    with open(os.path.join(MODELS_DIR, "ood_threshold.json"), "r") as f:
        return json.load(f)["ood_threshold"]

# ============================================================
# INFERENCE UTILITIES
# ============================================================
def compute_latent(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()

def compute_pseudotime(latents):
    """
    Reference-normalized pseudotime.
    Works for single patient and cohort analysis.
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
