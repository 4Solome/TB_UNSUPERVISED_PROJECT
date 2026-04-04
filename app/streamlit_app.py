import streamlit as st
import pandas as pd
import numpy as np
import torch

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
st.set_page_config(
    page_title="TB Risk Profiling System",
    layout="centered"
)

st.title("TB Risk Profiling System (TTVAE‑Based)")
st.caption(
    "Latent tuberculosis risk sequencing and phenotype profiling using "
    "Transformer‑based unsupervised representation learning."
)

# ============================================================
# PHENOTYPE DEFINITIONS
# ============================================================
cluster_info = {
    0: {
        "name": "Low-Symptom TB Risk",
        "description": "Low symptom burden but still within the learned TB-risk latent space."
    },
    1: {
        "name": "Active Symptomatic TB",
        "description": "High clinical symptom burden resembling active TB-like presentation."
    },
    2: {
        "name": "Minimal-Information Profile",
        "description": "Sparse or weak feature activation due to limited diagnostic information."
    },
    3: {
        "name": "Transitional TB Risk",
        "description": "Intermediate profile between low-risk and confirmed TB phenotypes."
    },
    4: {
        "name": "Laboratory-Confirmed TB",
        "description": "Strong laboratory and bacteriological feature dominance."
    }
}

# ============================================================
# INPUT PANEL
# ============================================================
st.header("Patient Data Entry")

age = st.slider("Age", 0, 100, 35)
sex = st.selectbox("Sex", ["Male", "Female"])

st.subheader("Symptoms")
cough = st.checkbox("Cough")
fever = st.checkbox("Fever")
weight_loss = st.checkbox("Weight loss")
chest_pain = st.checkbox("Chest pain")
night_sweats = st.checkbox("Night sweats")
blood_sputum = st.checkbox("Blood in sputum")

st.subheader("Behavioral / Clinical")
smoking = st.selectbox("Smoking history", ["Never", "Past", "Current"])
hiv = st.selectbox("HIV status", ["Negative", "Positive", "Unknown"])
tb_history = st.selectbox("Previous TB treatment", ["No", "Yes"])

st.subheader("Radiology")
xray = st.selectbox("Chest X‑ray", ["Normal", "Abnormal"])

st.subheader("Laboratory (optional)")
smear = st.selectbox("Smear", ["Not done", "Negative", "Positive"])
genexpert = st.selectbox("GeneXpert", ["Not done", "Negative", "Positive"])
culture = st.selectbox("Culture", ["Not done", "Negative", "Positive"])

# ============================================================
# ANALYSIS
# ============================================================
if st.button("Analyze Patient"):

    input_df = pd.DataFrame([{
        "age_census": age,
        "sex_census": 1 if sex == "Male" else 2,
        "cough": int(cough),
        "fever": int(fever),
        "weight_loss": int(weight_loss),
        "chest_pain": int(chest_pain),
        "night_sweats": int(night_sweats),
        "blood_sputum": int(blood_sputum),
        "smoke_now": 1 if smoking == "Current" else 0,
        "smoke_past": 1 if smoking == "Past" else 0,
        "hiv_res": 1 if hiv == "Positive" else 0,
        "hist_rx": 1 if tb_history == "Yes" else 0,
        "xray_normal": 1 if xray == "Normal" else 0,
        "smear_pos": 1 if smear == "Positive" else 0,
        "bact": 1 if genexpert == "Positive" else 0
    }])

    model, feature_names = load_ttvae()
    kmeans = load_cluster_model()
    ood_threshold = load_ood_threshold()

    pre = build_preprocessor(
        continuous_cols=["age_census"],
        binary_cols=[c for c in input_df.columns if c != "age_census"],
        categorical_cols=[]
    )

    X = pre.fit_transform(input_df)
    X = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    latent = compute_latent(model, X)
    pseudotime = compute_pseudotime(latent)[0]
    cluster = assign_cluster(kmeans, latent)[0]
    ood_flag, recon_error = check_ood(model, X, ood_threshold)

    # ========================================================
    # RESULTS
    # ========================================================
    st.header("Results")

    st.metric("TB Risk Score (Pseudotime)", f"{pseudotime:.2f}")

    if pseudotime < 0.3:
        st.success("Risk Category: Low Risk")
    elif pseudotime < 0.7:
        st.warning("Risk Category: Moderate Risk")
    else:
        st.error("Risk Category: High Risk")

    st.progress(pseudotime)

    phenotype = cluster_info[cluster]["name"]
    desc = cluster_info[cluster]["description"]

    st.subheader("Latent Phenotype")
    st.write(f"**{phenotype}**")
    st.caption(desc)

    st.subheader("Reliability Check")
    if ood_flag:
        st.warning(
            "⚠️ Input lies outside the training distribution. Interpretation may be unreliable."
        )
    else:
        st.success("✅ Input lies within known training patterns.")

    st.subheader("Model Confidence")
    st.write(f"Reconstruction Error: `{recon_error:.4f}`")

# ============================================================
# ✅ SYNTHETIC DATA GENERATION (FIXED)
# ============================================================
st.divider()
st.header("Synthetic Patient Generation")

st.caption(
    "Synthetic TB patient profiles can be generated by sampling the learned latent space "
    "of the trained generative model."
)

num_samples = st.slider("Number of synthetic patients", 10, 100, 50)

if st.button("Generate Synthetic Patients"):

    model, feature_names = load_ttvae()
    device = next(model.parameters()).device

    z = torch.randn(num_samples, model.latent_dim).to(device)

    model.eval()
    with torch.no_grad():
        synthetic = model.decode(z).cpu().numpy()

    syn_df = pd.DataFrame(synthetic, columns=feature_names)

    st.success(f"Generated {num_samples} synthetic patient profiles")
    st.dataframe(syn_df.head(10))

    st.download_button(
        label="Download Synthetic Dataset (CSV)",
        data=syn_df.to_csv(index=False),
        file_name="synthetic_tb_patients.csv"
    )

st.divider()
st.caption(
    "This system provides latent risk stratification and phenotype profiling using unsupervised learning. "
    "It does not replace clinical diagnosis."
)
