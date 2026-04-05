import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

from utils import (
    build_preprocessor,
    load_ttvae,
    load_cluster_model,
    load_ood_threshold,
    compute_latent,
    compute_pseudotime,
    check_ood,
    assign_cluster
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="TB Risk Profiling System", layout="centered")

st.title("TB Risk Profiling System (TTVAE‑Based)")
st.caption(
    "Latent tuberculosis phenotype discovery and progression sequencing "
    "using a Transformer‑based Tabular Variational Autoencoder."
)

# ============================================================
# PHENOTYPES
# ============================================================
cluster_info = {
    0: ("Low‑Symptom TB Risk", "Low symptom burden"),
    1: ("Active Symptomatic TB", "High clinical symptom burden"),
    2: ("Minimal‑Information Profile", "Sparse diagnostic information"),
    3: ("Transitional TB Risk", "Intermediate phenotype"),
    4: ("Laboratory‑Confirmed TB", "Strong bacteriological evidence")
}

# ============================================================
# TRAINING FEATURE GROUPS
# ============================================================
continuous_cols = ["age_census", "cough_d", "fever_d", "wloss_d", "sputum_d"]
binary_cols = [
    "sex_census","cough","fever","weight_loss","night_sweats",
    "chest_pain","blood_sputum","sputum",
    "smoke_now","smoke_past","hiv_res","hist_rx",
    "xray_normal","smear_pos","culture","cult_pos","bact"
]
categorical_cols = ["region","married","edu","occupation"]

# ============================================================
# LOAD MODELS (SAFE)
# ============================================================
@st.cache_resource
def load_models():
    model, feature_names = load_ttvae()
    kmeans = load_cluster_model()
    threshold = load_ood_threshold()
    return model, kmeans, threshold, feature_names

model, kmeans, threshold, feature_names = load_models()

# ============================================================
# INPUT MODE
# ============================================================
mode = st.radio("Select mode:", ["Single Patient", "Cohort (CSV Upload)"])

# ============================================================
# SINGLE PATIENT MODE
# ============================================================
if mode == "Single Patient":

    age = st.slider("Age", 0, 100, 35)
    cough_d = st.number_input("Cough days", 0, 365, 0)
    fever_d = st.number_input("Fever days", 0, 365, 0)
    wloss_d = st.number_input("Weight loss days", 0, 2000, 0)
    sputum_d = st.number_input("Sputum days", 0, 365, 0)

    input_df = pd.DataFrame([{
        "age_census": age,
        "cough_d": cough_d,
        "fever_d": fever_d,
        "wloss_d": wloss_d,
        "sputum_d": sputum_d,
        "sex_census": 1,
        "cough": int(cough_d > 0),
        "fever": int(fever_d > 0),
        "weight_loss": int(wloss_d > 0),
        "night_sweats": 0,
        "chest_pain": 0,
        "blood_sputum": 0,
        "sputum": int(sputum_d > 0),
        "smoke_now": 0,
        "smoke_past": 0,
        "hiv_res": 0,
        "hist_rx": 0,
        "xray_normal": 1,
        "smear_pos": 0,
        "culture": 0,
        "cult_pos": 0,
        "bact": 0,
        "region": "Unknown",
        "married": "Unknown",
        "edu": "Unknown",
        "occupation": "Unknown"
    }])

    pre = build_preprocessor(continuous_cols, binary_cols, categorical_cols)
    dummy = {c: 0 for c in continuous_cols + binary_cols}
    dummy.update({c: "Unknown" for c in categorical_cols})
    pre.fit(pd.DataFrame([dummy]))

    X = pre.transform(input_df)
    X = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    latents = compute_latent(model, X)
    pseudotime = float(np.clip(compute_pseudotime(latents)[0], 0, 1))
    cluster = int(assign_cluster(kmeans, latents)[0])
    recon = float(np.squeeze(check_ood(model, X, threshold)[1]))

    st.metric("Pseudotime (reference-based)", f"{pseudotime:.2f}")
    st.warning("Single-patient pseudotime is approximate")
    st.write("Phenotype:", cluster_info[cluster][0])
    st.write("Reconstruction error:", recon)

# ============================================================
# COHORT MODE
# ============================================================
else:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        pre = build_preprocessor(continuous_cols, binary_cols, categorical_cols)
        dummy = {c: 0 for c in continuous_cols + binary_cols}
        dummy.update({c: "Unknown" for c in categorical_cols})
        pre.fit(pd.DataFrame([dummy]))

        X = pre.transform(df)
        X = pd.DataFrame(X, columns=pre.get_feature_names_out())
        X = X.reindex(columns=feature_names, fill_value=0).values

        latents = compute_latent(model, X)
        z1 = latents[:, 0]
        pseudotime = np.clip((z1 - z1.min()) / (z1.max() - z1.min() + 1e-10), 0, 1)
        clusters = assign_cluster(kmeans, latents)

        df["pseudotime"] = pseudotime
        df["phenotype"] = [cluster_info[c][0] for c in clusters]

        st.dataframe(df.sort_values("pseudotime", ascending=False))

        fig, ax = plt.subplots()
        ax.scatter(pseudotime, range(len(pseudotime)))
        ax.set_xlabel("Pseudotime")
        st.pyplot(fig)
    
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
