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
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="TB Risk Profiling System",
    layout="centered"
)

st.title("TB Risk Profiling System (TTVAE‑Based)")
st.caption(
    "An interactive prototype for latent tuberculosis risk sequencing, "
    "phenotype identification, uncertainty detection, and synthetic patient generation "
    "using Transformer‑based unsupervised representation learning."
)

# ============================================================
# LATENT PHENOTYPE DEFINITIONS (FINAL)
# ============================================================
cluster_info = {
    0: {
        "name": "Low-Symptom TB Risk",
        "description": "Low symptom burden but still within the learned TB‑risk latent space."
    },
    1: {
        "name": "Active Symptomatic TB",
        "description": "High clinical symptom burden resembling active TB‑like presentation."
    },
    2: {
        "name": "Minimal-Information Profile",
        "description": "Sparse or weak feature activation due to limited diagnostic information."
    },
    3: {
        "name": "Transitional TB Risk",
        "description": "Intermediate phenotype between lower-risk and laboratory-confirmed profiles."
    },
    4: {
        "name": "Laboratory-Confirmed TB",
        "description": "Strong bacteriological and laboratory feature dominance."
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

st.subheader("Behavioral / Clinical History")
smoking = st.selectbox("Smoking history", ["Never", "Past", "Current"])
hiv = st.selectbox("HIV status", ["Negative", "Positive", "Unknown"])
tb_history = st.selectbox("Previous TB treatment", ["No", "Yes"])

st.subheader("Radiology")
xray = st.selectbox("Chest X‑ray result", ["Normal", "Abnormal"])

st.subheader("Laboratory (Optional)")
smear = st.selectbox("Smear microscopy", ["Not done", "Negative", "Positive"])
genexpert = st.selectbox("GeneXpert", ["Not done", "Negative", "Positive"])
culture = st.selectbox("Culture", ["Not done", "Negative", "Positive"])

# ============================================================
# ANALYZE PATIENT
# ============================================================
if st.button("Analyze Patient"):

    # -------------------------------
    # Build input dataframe (1 row)
    # -------------------------------
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

    # -------------------------------
    # Load trained components
    # -------------------------------
    model, feature_names = load_ttvae()
    kmeans = load_cluster_model()
    ood_threshold = load_ood_threshold()

    # -------------------------------
    # Preprocessing
    # -------------------------------
    pre = build_preprocessor(
        continuous_cols=["age_census"],
        binary_cols=[c for c in input_df.columns if c != "age_census"],
        categorical_cols=[]
    )

    X = pre.fit_transform(input_df)
    X = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    # -------------------------------
    # Inference
    # -------------------------------
    latent = compute_latent(model, X)

    pseudotime = float(compute_pseudotime(latent)[0])
    cluster = int(assign_cluster(kmeans, latent)[0])
    ood_flag, recon_error = check_ood(model, X, ood_threshold)

    # ✅ CRITICAL FIX
    recon_error = float(np.squeeze(recon_error))

    # ========================================================
    # RESULTS PANEL
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
    st.caption(
        "Continuous latent risk position approximating tuberculosis progression "
        "derived from cross‑sectional observations."
    )

    st.subheader("Latent Phenotype")
    st.write(f"**{cluster_info[cluster]['name']}**")
    st.caption(cluster_info[cluster]['description'])

    st.subheader("Reliability Assessment")
    if ood_flag:
        st.warning(
            "⚠️ This patient profile lies outside the model’s training distribution. "
            "Interpretation may be unreliable."
        )
    else:
        st.success("✅ This patient profile lies within known training patterns.")

    st.subheader("Model Confidence")
    st.write(f"Reconstruction Error: `{recon_error:.4f}`")
    st.caption(
        "Lower error indicates better representation within the learned latent space."
    )

# ============================================================
# SYNTHETIC DATA GENERATION (OBJECTIVE 6) — FIXED
# ============================================================
st.divider()
st.header("Synthetic Patient Generation")

st.caption(
    "The trained generative model samples the learned latent space to "
    "generate realistic synthetic tuberculosis patient profiles."
)

num_samples = st.slider("Number of synthetic patients", 10, 100, 50)

if st.button("Generate Synthetic Patients"):

    model, feature_names = load_ttvae()
    device = next(model.parameters()).device

    # ✅ Robust latent dimension inference
    example_z = compute_latent(
        model,
        np.zeros((1, len(feature_names)))
    )
    latent_dim = example_z.shape[1]

    z = torch.randn(num_samples, latent_dim).to(device)

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
    "This prototype provides latent risk stratification and phenotype profiling "
    "based on unsupervised generative modeling. It does not replace clinical diagnosis."
)
