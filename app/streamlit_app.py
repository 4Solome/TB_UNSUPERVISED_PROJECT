import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import (
    build_preprocessor,
    load_ttvae,
    load_feature_names,
    load_cluster_model,
    compute_latent,
    compute_pseudotime,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="TTVAE Tuberculosis Risk Sequencer", layout="centered")

st.title("TTVAE Tuberculosis Risk Sequencer")
st.caption(
    "Upload a TB patient CSV to obtain pseudotime risk scores, "
    "phenotype assignments, and reliability indicators."
)

# ============================================================
# CONFIGURATION
# ============================================================
continuous_cols = ["age_census", "cough_d", "fever_d", "wloss_d", "sputum_d"]
binary_cols = [
    "sex_census", "cough", "fever", "weight_loss", "night_sweats",
    "chest_pain", "blood_sputum", "sputum",
    "smoke_now", "smoke_past", "hiv_res", "hist_rx",
    "xray_normal", "smear_pos", "culture", "cult_pos", "bact"
]
categorical_cols = ["region", "married", "edu", "occupation"]

# ============================================================
# LOAD MODELS
# ============================================================
feature_names = load_feature_names()
model = load_ttvae(input_dim=len(feature_names))
kmeans = load_cluster_model()

# ============================================================
# CSV UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df_raw = pd.read_csv(uploaded_file)

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------
    st.subheader("Uploaded Data Preview")
    st.dataframe(df_raw.head())

    # --------------------------------------------------------
    # Handle missing columns (PDF behavior)
    # --------------------------------------------------------
    expected_cols = set(continuous_cols + binary_cols + categorical_cols)
    present_cols = set(df_raw.columns)

    missing_cols = sorted(list(expected_cols - present_cols))
    if missing_cols:
        st.warning(
            "The following expected columns were missing and ignored: "
            + ", ".join(missing_cols)
        )

    # Keep only known columns
    df = df_raw[[c for c in df_raw.columns if c in expected_cols]].copy()

    # --------------------------------------------------------
    # Runtime preprocessing
    # --------------------------------------------------------
    pre = build_preprocessor(continuous_cols, binary_cols, categorical_cols)

    # Dummy fit to initialize transformers
    dummy = {c: 0 for c in continuous_cols + binary_cols}
    dummy.update({c: "Unknown" for c in categorical_cols})
    pre.fit(pd.DataFrame([dummy]))

    X = pre.transform(df)
    X = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    # --------------------------------------------------------
    # Latent inference
    # --------------------------------------------------------
    latents = compute_latent(model, X)

    # Pseudotime & clustering
    pseudotime = compute_pseudotime(latents)
    cluster = kmeans.predict(latents)

    # Reconstruction error (OOD proxy)
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        rec, _, _ = model(X_t)
    recon_error = ((rec.numpy() - X) ** 2).mean(axis=1)

    # --------------------------------------------------------
    # Patient‑level results table
    # --------------------------------------------------------
    results = pd.DataFrame({
        "pseudotime": pseudotime.round(4),
        "cluster": cluster,
        "OOD_flag": recon_error > np.percentile(recon_error, 95),
        "reconstruction_error": recon_error.round(4)
    })

    st.subheader("Patient‑Level Results")
    st.dataframe(results)

    st.download_button(
        "Download Results CSV",
        results.to_csv(index=False),
        file_name="ttvae_results.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------
    # Latent space visualization
    # --------------------------------------------------------
    st.subheader("Latent Space & Pseudotime")

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        latents[:, 0], latents[:, 1],
        c=pseudotime, cmap="plasma", s=40
    )
    ax.set_xlabel("Latent Dimension 1 (z1)")
    ax.set_ylabel("Latent Dimension 2 (z2)")
    plt.colorbar(sc, ax=ax, label="Pseudotime")

    st.pyplot(fig)

else:
    st.info("Upload a cohort CSV file to begin analysis.")
# ============================================================
# SYNTHETIC DATA GENERATION (DECODED)
# ============================================================
st.divider()
st.header("Synthetic Patient Generation")

num_samples = st.slider("Number of synthetic patients", 10, 100, 50)

if st.button("Generate Synthetic Patients"):

    model, feature_names = load_ttvae()
    device = next(model.parameters()).device

    example_z = compute_latent(model, np.zeros((1, len(feature_names))))
    latent_dim = example_z.shape[1]

    z = torch.randn(num_samples, latent_dim).to(device)

    with torch.no_grad():
        synthetic = model.decode(z).cpu().numpy()

    syn = pd.DataFrame(synthetic, columns=feature_names)

    # ===========================
    # ✅ DECODE SYNTHETIC DATA
    # ===========================

    decoded = pd.DataFrame()

    # ---- Age (inverse scaling: assume 0–100)
    decoded["age_census"] = (syn["cont__age_census"] * 100).round().astype(int)

    # ---- Binary variables
    bin_cols = [c for c in syn.columns if c.startswith("bin__")]
    for col in bin_cols:
        decoded[col.replace("bin__", "")] = (syn[col] >= 0.5).astype(int)

    # ---- Region (one-hot)
    region_cols = [c for c in syn.columns if c.startswith("cat__region")]
    decoded["region"] = (
        syn[region_cols].idxmax(axis=1).str.replace("cat__region_", "")
    )

    st.success(f"Generated {num_samples} decoded synthetic patients")

    st.dataframe(decoded.head(10))

    st.download_button(
        "Download Decoded Synthetic Dataset",
        decoded.to_csv(index=False),
        file_name="synthetic_tb_patients_decoded.csv"
    )

st.divider()
st.caption(
    "Synthetic data are generated in model feature space and decoded for clinical "
    "interpretability. This system does not replace medical diagnosis."
)
