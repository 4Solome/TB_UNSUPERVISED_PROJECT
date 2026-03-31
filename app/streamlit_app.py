import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_preprocessor, load_feature_names, load_ttvae,
    load_ood_threshold, load_cluster_model,
    compute_latent, compute_pseudotime, check_ood, assign_cluster
)

st.set_page_config(page_title="TTVAE TB Risk Sequencer", layout="wide")

st.title("TTVAE Tuberculosis Risk Sequencer")
st.markdown("Upload a TB patient CSV to obtain risk pseudotime, phenotype cluster, and explanations.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)
    st.write("Uploaded Data Preview:", df.head())

    pre = load_preprocessor()
    feature_names = load_feature_names()
    threshold = load_ood_threshold()
    kmeans = load_cluster_model()

    # Preprocess
    X = pre.transform(df)
    model = load_ttvae(input_dim=X.shape[1])

    # Latents
    latents = compute_latent(model, X)

    # Pseudotime
    pseudotime = compute_pseudotime(latents)

    # Cluster prediction
    clusters = assign_cluster(kmeans, latents)

    # OOD detection
    ood_flags, errors = check_ood(model, X, threshold)

    st.subheader("Results")

    results = pd.DataFrame({
        "Pseudotime": pseudotime,
        "Cluster": clusters,
        "OOD_Flag": ood_flags,
        "Reconstruction_Error": errors
    })

    st.write(results)

    st.download_button("Download Results CSV", results.to_csv(index=False), "ttvae_results.csv")

    st.subheader("Pseudotime Visualization")
    plt.figure(figsize=(7,5))
    plt.scatter(latents[:,0], latents[:,1], c=pseudotime, cmap="plasma", alpha=0.7)
    plt.colorbar(label="Pseudotime")
    plt.xlabel("z1")
    plt.ylabel("z2")
    st.pyplot(plt)
