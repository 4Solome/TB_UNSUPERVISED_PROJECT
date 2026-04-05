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
    assign_cluster
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TTVAE Tuberculosis Risk Sequencer",
    layout="centered"
)

st.title("TTVAE Tuberculosis Risk Sequencer")
st.caption(
    "Upload a TB patient CSV to obtain pseudotime risk scores, "
    "phenotype assignments, and reliability indicators."
)

# ============================================================
# PHENOTYPE LABELS
# ============================================================
CLUSTER_NAMES = {
    0: "Low‑Symptom TB Risk",
    1: "Active Symptomatic TB",
    2: "Minimal‑Information Profile",
    3: "Transitional TB Risk",
    4: "Laboratory‑Confirmed TB"
}

# ============================================================
# TRAINING FEATURE GROUPS
# ============================================================
continuous_cols = [
    "age_census", "cough_d", "fever_d", "wloss_d",
    "sputum_d", "tbhist_y", "tbtreat_w"
]

binary_cols = [
    "sex_census", "setting", "smoke_now", "smoke_past", "hiv_res",
    "cough", "fever", "weight_loss", "night_sweats",
    "chest_pain", "blood_sputum", "sputum", "hist_rx",
    "current_rx", "xray_normal", "smear_pos",
    "culture", "cult_pos", "bact"
]

categorical_cols = [
    "region", "married", "edu", "occupation",
    "xrayres", "central_cxr_res",
    "zn", "genexpert", "final_result"
]

EXPECTED_COLS = continuous_cols + binary_cols + categorical_cols

# ============================================================
# LOAD MODELS
# ============================================================
feature_names = load_feature_names()
model = load_ttvae(input_dim=len(feature_names))
kmeans = load_cluster_model()

# ============================================================
# CSV UPLOAD (OPTIONAL)
# ============================================================
st.header("Cohort Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# ============================================================
# COHORT ANALYSIS (ONLY IF CSV PROVIDED)
# ============================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ------------------ Missing column handling ------------------
    present_cols = set(df.columns)
    missing_cols = [c for c in EXPECTED_COLS if c not in present_cols]

    if missing_cols:
        st.warning(
            "The following expected columns were missing and filled with defaults:\n\n"
            + ", ".join(missing_cols)
        )

    for c in continuous_cols:
        if c not in df.columns:
            df[c] = 0.0
    for c in binary_cols:
        if c not in df.columns:
            df[c] = 0
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "Unknown"

    df = df[EXPECTED_COLS]

    # ------------------ Preprocessing ------------------
    pre = build_preprocessor(
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols
    )

    dummy = {c: 0 for c in continuous_cols + binary_cols}
    dummy.update({c: "Unknown" for c in categorical_cols})
    pre.fit(pd.DataFrame([dummy]))

    X = pre.transform(df)
    X = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    # ------------------ Latent / pseudotime / cluster ------------------
    latents = compute_latent(model, X)
    pseudotime = compute_pseudotime(latents)

    cluster_ids = assign_cluster(kmeans, latents)
    phenotypes = [CLUSTER_NAMES.get(c, f"Cluster {c}") for c in cluster_ids]

    # ------------------ Reconstruction error ------------------
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        rec, _, _ = model(X_t)

    rec_error = ((rec.numpy() - X) ** 2).mean(axis=1)
    ood_flag = rec_error > np.percentile(rec_error, 95)

    # ------------------ Results table ------------------
    results = pd.DataFrame({
        "pseudotime": np.round(pseudotime, 4),
        "cluster_id": cluster_ids,
        "phenotype": phenotypes,
        "OOD_flag": ood_flag,
        "reconstruction_error": np.round(rec_error, 4)
    })

    st.subheader("Patient‑Level Results")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download Results CSV",
        results.to_csv(index=False),
        file_name="ttvae_patient_results.csv",
        mime="text/csv"
    )

    # ------------------ Latent plot ------------------
    st.subheader("Latent Space & Pseudotime")

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        latents[:, 0],
        latents[:, 1],
        c=pseudotime,
        cmap="plasma",
        s=40
    )
    ax.set_xlabel("Latent Dimension 1 (z1)")
    ax.set_ylabel("Latent Dimension 2 (z2)")
    plt.colorbar(sc, ax=ax, label="Pseudotime")
    st.pyplot(fig)

# ============================================================
# SYNTHETIC DATA GENERATION (ALWAYS VISIBLE ✅)
# ============================================================
st.divider()
st.header("Synthetic Patient Generation")

num_samples = st.slider(
    "Number of synthetic patients",
    min_value=10,
    max_value=200,
    value=50
)

if st.button("Generate Synthetic Patients"):

    latent_dim = 32  # from training
    z = torch.randn(num_samples, latent_dim)

    with torch.no_grad():
        synthetic = model.decode(z).numpy()

    syn = pd.DataFrame(synthetic, columns=feature_names)

    decoded = pd.DataFrame()
    decoded["age_census"] = (syn["cont__age_census"] * 100).round().astype(int)

    bin_cols = [c for c in syn.columns if c.startswith("bin__")]
    for col in bin_cols:
        decoded[col.replace("bin__", "")] = (syn[col] >= 0.5).astype(int)

    region_cols = [c for c in syn.columns if c.startswith("cat__region")]
    decoded["region"] = syn[region_cols].idxmax(axis=1).str.replace("cat__region_", "")

    st.success(f"Generated {num_samples} synthetic patients")
    st.dataframe(decoded.head(10), use_container_width=True)

    st.download_button(
        "Download Decoded Synthetic Dataset",
        decoded.to_csv(index=False),
        file_name="synthetic_tb_patients.csv",
        mime="text/csv"
    )

st.caption(
    "Synthetic data are generated in model feature space and decoded for "
    "approximate clinical interpretability only."
)
