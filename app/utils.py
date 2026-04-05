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

# ============================================================
# Runtime-safe preprocessing (NO pickling, NO version mismatch)
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
# Load trained TTVAE model (CORRECT SIGNATURE)
# ============================================================
def load_ttvae(input_dim):
    model = TTVAE(D_in=input_dim).to(device)
    state = torch.load("models/ttvae_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ============================================================
# Metadata & auxiliary models
# ============================================================
def load_feature_names():
    with open("models/feature_names.json", "r") as f:
        return json.load(f)

def load_cluster_model():
    return joblib.load("models/kmeans_model.joblib")

# ============================================================
# Inference utilities
# ============================================================
def compute_latent(model, X):
    """
    Encode tabular features into latent space (mu).
    """
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()

def compute_pseudotime(latents):
    """
    Cohort-based pseudotime (true use).
    Normalized z1 across the uploaded cohort.
    """
    z1 = latents[:, 0]
    pt = (z1 - z1.min()) / (z1.max() - z1.min() + 1e-10)
    return np.clip(pt, 0.0, 1.0)

def assign_cluster(kmeans, latents):
    """
    Assign latent-space clusters.
    """
    return kmeans.predict(latents)
