import torch
import numpy as np
import joblib
import json
from sklearn.cluster import KMeans
from ttvae_model import TTVAE

device = torch.device("cpu")

# Load preprocessor

def load_preprocessor():
    return joblib.load("models/preprocessor.joblib")


# Load TTVAE model

def load_ttvae(input_dim):
    model = TTVAE(input_dim=input_dim).to(device)
    state = torch.load("models/ttvae_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# Load feature name
def load_feature_names():
    with open("models/feature_names.json", "r") as f:
        return json.load(f)


# Load OOD threshold

def load_ood_threshold():
    with open("models/ood_threshold.json", "r") as f:
        return json.load(f)["ood_threshold"]


# Load trained cluster model

def load_cluster_model():
    return joblib.load("models/kmeans_model.joblib")


# Compute latent

def compute_latent(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()


# Compute pseudotime (z1 normalized)

def compute_pseudotime(latents):
    z1 = latents[:, 0]
    return (z1 - z1.min()) / (z1.max() - z1.min() + 1e-10)


# OOD detection

def check_ood(model, X, threshold):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        rec, _, _ = model(X_t)
    err = ((rec.cpu().numpy() - X)**2).mean(axis=1)
    return err > threshold, err


# Predict phenotype cluster

def assign_cluster(kmeans, latents):
    return kmeans.predict(latents)
