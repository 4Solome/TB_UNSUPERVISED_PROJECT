import streamlit as st
import pandas as pd
import numpy as np
import torch
import json

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
    "Latent tuberculosis risk sequencing using a Transformer‑based "
    "Tabular Variational Autoencoder."
)

# ============================================================
# PHENOTYPE LABELS
# ============================================================
cluster_info = {
    0: ("Low‑Symptom TB Risk",
        "Low symptom burden but within the learned TB‑risk space."),
    1: ("Active Symptomatic TB",
        "High clinical symptom burden consistent with active TB."),
    2: ("Minimal‑Information Profile",
        "Sparse diagnostic information and weak signals."),
    3: ("Transitional TB Risk",
        "Mixed features between early and confirmed TB profiles."),
    4: ("Laboratory‑Confirmed TB",
        "Strong bacteriological and laboratory evidence.")
}

# ============================================================
# COLUMN GROUPS (EXACTLY AS TRAINING)
# ============================================================
continuous_cols = [
    "age_census", "cough_d", "fever_d", "wloss_d", "sputum_d"
]

binary_cols = [
    "sex_census", "cough", "fever", "weight_loss", "night_sweats",
    "chest_pain", "blood_sputum", "sputum",
    "smoke_now", "smoke_past", "hiv_res", "hist_rx",
    "xray_normal", "smear_pos", "culture", "cult_pos", "bact"
]

categorical_cols = [
    "region", "married", "edu", "occupation"
]

# ============================================================
# INPUT PANEL
# ============================================================
st.header("Patient Data Entry")

# Demographics
age = st.slider("Age (years)", 0, 100, 35)
sex = st.selectbox("Sex", ["Male", "Female"])
region = st.selectbox("Region", ["Central", "East", "North", "West"])
married = st.selectbox(
    "Marital status",
    ["Single", "Married", "Separated", "Divorced",
     "Widowed", "Don't know", "Unknown"]
)
education = st.selectbox(
    "Education level",
    ["None", "Primary", "Senior 1–4", "Senior 5–6",
     "Tertiary", "Don't know", "Unknown"]
)
occupation = st.selectbox(
    "Occupation",
    ["Business", "Civil servant", "Healthcare worker", "Student",
     "Unemployed", "Farmer", "House wife/husband",
     "Skilled labor", "Other"]
)

# Symptoms + durations
st.subheader("Symptoms")

cough = st.checkbox("Cough")
cough_d = st.number_input("Cough duration (days)", 0, 365, 0) if cough else 0

fever = st.checkbox("Fever")
fever_d = st.number_input("Fever duration (days)", 0, 365, 0) if fever else 0

weight_loss = st.checkbox("Weight loss")
wloss_d = st.number_input("Weight‑loss duration (days)", 0, 2000, 0) if weight_loss else 0

sputum = st.checkbox("Sputum production")
sputum_d = st.number_input("Sputum duration (days)", 0, 365, 0) if sputum else 0

night_sweats = st.checkbox("Night sweats")
chest_pain = st.checkbox("Chest pain")
blood_sputum = st.checkbox("Blood‑stained sputum")

# Behaviour & labs
st.subheader("Clinical History")

smoke_now = st.selectbox("Currently smoking?", ["No", "Yes"])
smoke_past = st.selectbox("Smoked in the past?", ["No", "Yes"])
hiv = st.selectbox("HIV status", ["Negative", "Positive", "Unknown"])
hist_rx = st.selectbox("Previously treated for TB?", ["No", "Yes"])

xray = st.selectbox("Chest X‑ray", ["Normal", "Abnormal"])
smear = st.selectbox("Smear microscopy", ["Negative", "Positive"])
culture = st.selectbox("Culture", ["Negative", "Positive"])
genexpert = st.selectbox("GeneXpert", ["Negative", "Positive"])

# ============================================================
# ANALYSIS
# ============================================================
if st.button("Analyze Patient"):

    # -------------------------------
    # Build raw input row
    # -------------------------------
    input_df = pd.DataFrame([{
        # Continuous
        "age_census": age,
        "cough_d": cough_d,
        "fever_d": fever_d,
        "wloss_d": wloss_d,
        "sputum_d": sputum_d,

        # Binary
        "sex_census": 1 if sex == "Male" else 2,
        "cough": int(cough),
        "fever": int(fever),
        "weight_loss": int(weight_loss),
        "night_sweats": int(night_sweats),
        "chest_pain": int(chest_pain),
        "blood_sputum": int(blood_sputum),
        "sputum": int(sputum),
        "smoke_now": 1 if smoke_now == "Yes" else 0,
        "smoke_past": 1 if smoke_past == "Yes" else 0,
        "hiv_res": 1 if hiv == "Positive" else 0,
        "hist_rx": 1 if hist_rx == "Yes" else 0,
        "xray_normal": 1 if xray == "Normal" else 0,
        "smear_pos": 1 if smear == "Positive" else 0,
        "culture": 1 if culture == "Positive" else 0,
        "cult_pos": 1 if culture == "Positive" else 0,
        "bact": 1 if genexpert == "Positive" else 0,

        # Categorical (AS STRINGS)
        "region": region,
        "married": married,
        "edu": education,
        "occupation": occupation
    }])

    # -------------------------------
    # Build preprocessor (no pickle)
    # -------------------------------
    preprocessor = build_preprocessor(
        continuous_cols,
        binary_cols,
        categorical_cols
    )

    # Fit ONCE on dummy data to initialise encoders
    dummy = {}
    for c in continuous_cols:
        dummy[c] = 0.0
    for c in binary_cols:
        dummy[c] = 0
    for c in categorical_cols:
        dummy[c] = "Unknown"

    dummy_df = pd.DataFrame([dummy])
    preprocessor.fit(dummy_df)

    # Transform real input
    X = preprocessor.transform(input_df)
    X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())

    # Align with training feature space
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)

    X = X.reindex(columns=feature_names, fill_value=0).values

    # -------------------------------
    # Load models & infer
    # -------------------------------
    model, _ = load_ttvae()
    kmeans = load_cluster_model()
    threshold = load_ood_threshold()

    latents = compute_latent(model, X)
    pseudotime = float(compute_pseudotime(latents)[0])
    cluster = int(assign_cluster(kmeans, latents)[0])
    ood, recon = check_ood(model, X, threshold)
    recon = float(np.squeeze(recon))

    # -------------------------------
    # RESULTS
    # -------------------------------
    st.header("Results")

    st.metric("TB Risk Score (Pseudotime)", f"{pseudotime:.2f}")

    if pseudotime < 0.3:
        st.success("Risk Category: Low Risk")
    elif pseudotime < 0.7:
        st.warning("Risk Category: Moderate Risk")
    else:
        st.error("Risk Category: High Risk")

    st.progress(pseudotime)

    name, desc = cluster_info[cluster]
    st.subheader("Latent Phenotype")
    st.write(f"**{name}**")
    st.caption(desc)

    st.subheader("Model Confidence")
    st.write(f"Reconstruction Error: `{recon:.4f}`")

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
