import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
st.set_page_config(page_title="TB Risk Profiling System", layout="centered")

st.title("TB Risk Profiling System (TTVAE‑Based)")
st.caption(
    "Cohort-based latent tuberculosis phenotyping and progression sequencing "
    "using a Transformer-based Tabular Variational Autoencoder."
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
# COHORT CSV UPLOAD
# ============================================================
st.header("Cohort Analysis (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Check columns
    required = continuous_cols + binary_cols + categorical_cols
    missing = set(required) - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        st.stop()

    # Build & fit runtime-safe preprocessor
    pre = build_preprocessor(continuous_cols, binary_cols, categorical_cols)

    dummy = {c: 0 for c in continuous_cols + binary_cols}
    dummy.update({c: "Unknown" for c in categorical_cols})
    pre.fit(pd.DataFrame([dummy]))

    # Transform data
    X = pre.transform(df)
    X = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    # Latent inference
    latents = compute_latent(model, X)

    # Clustering & pseudotime
    df["cluster_id"] = assign_cluster(kmeans, latents)
    df["phenotype"] = df["cluster_id"].map(CLUSTER_NAMES)
    df["pseudotime"] = compute_pseudotime(latents)

    # ========================================================
    # RESULTS
    # ========================================================
    st.subheader("Cohort Results")
    st.dataframe(
        df.sort_values("pseudotime", ascending=False),
        use_container_width=True
    )

    st.subheader("Pseudotime Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["pseudotime"], bins=20)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Number of Patients")
    st.pyplot(fig)

    st.caption(
        "Pseudotime reflects relative progression along a latent TB risk axis "
        "within the uploaded cohort."
    )

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
