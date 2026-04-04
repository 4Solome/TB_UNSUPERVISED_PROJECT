import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os

from utils import (
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
    "Latent tuberculosis risk sequencing using a Transformer-based "
    "Tabular Variational Autoencoder (TTVAE)."
)

# ============================================================
# CLUSTER LABELS
# ============================================================
cluster_info = {
    0: {
        "name": "Low‑Symptom TB Risk",
        "description": "Low symptom burden but still within the learned TB‑risk space."
    },
    1: {
        "name": "Active Symptomatic TB",
        "description": "High clinical symptom burden consistent with active TB-like disease."
    },
    2: {
        "name": "Minimal‑Information Profile",
        "description": "Sparse diagnostic information with weak overall feature activation."
    },
    3: {
        "name": "Transitional TB Risk",
        "description": "Mixed signals between lower‑risk and laboratory‑confirmed profiles."
    },
    4: {
        "name": "Laboratory‑Confirmed TB",
        "description": "Strong bacteriological and laboratory evidence of tuberculosis."
    }
}

# ============================================================
# DATA DICTIONARY ENCODINGS
# ============================================================
EDU_MAP = {
    "None": 1,
    "Primary": 2,
    "Senior 1–4": 3,
    "Senior 5–6": 4,
    "Tertiary": 5,
    "Don't know": 6,
    "Unknown": 7
}

MARITAL_MAP = {
    "Single": 1,
    "Married": 2,
    "Separated": 3,
    "Divorced": 4,
    "Widowed": 5,
    "Don't know": 6,
    "Unknown": 7
}

OCCUPATION_MAP = {
    "Business": 1,
    "Civil servant": 2,
    "Healthcare worker": 3,
    "Student": 4,
    "Unemployed": 5,
    "Farmer": 6,
    "House wife/husband": 7,
    "Skilled labor": 8,
    "Other": 9
}

REGIONS = ["Central", "East", "North", "West"]

# ============================================================
# INPUT PANEL
# ============================================================
st.header("Patient Data Entry")

age = st.slider("Age (years)", 0, 100, 35)
sex = st.selectbox("Sex", ["Male", "Female"])
region = st.selectbox("Region", REGIONS)
married = st.selectbox("Marital status", list(MARITAL_MAP.keys()))
education = st.selectbox("Education level", list(EDU_MAP.keys()))
occupation = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()))

st.subheader("Symptoms")

cough = st.checkbox("Cough")
cough_d = st.number_input("Duration of cough (days)", 0, 365, 0) if cough else 0

fever = st.checkbox("Fever")
fever_d = st.number_input("Duration of fever (days)", 0, 365, 0) if fever else 0

weight_loss = st.checkbox("Weight loss")
wloss_d = st.number_input("Duration of weight loss (days)", 0, 2000, 0) if weight_loss else 0

sputum = st.checkbox("Sputum production")
sputum_d = st.number_input("Duration of sputum (days)", 0, 365, 0) if sputum else 0

night_sweats = st.checkbox("Night sweats")
chest_pain = st.checkbox("Chest pain")
blood_sputum = st.checkbox("Blood-stained sputum")

st.subheader("Behavioral & Clinical History")

smoke_now = st.selectbox("Currently smoking?", ["No", "Yes"])
smoke_past = st.selectbox("Smoked in the past?", ["No", "Yes"])
hiv = st.selectbox("HIV status", ["Negative", "Positive", "Unknown"])
hist_rx = st.selectbox("Previously treated for TB?", ["No", "Yes"])

st.subheader("Radiology & Lab")

xray = st.selectbox("Chest X‑ray result", ["Normal", "Abnormal"])
smear = st.selectbox("Smear microscopy", ["Negative", "Positive"])
culture = st.selectbox("Culture", ["Negative", "Positive"])
genexpert = st.selectbox("GeneXpert", ["Negative", "Positive"])

# ============================================================
# ANALYZE
# ============================================================
if st.button("Analyze Patient"):

    # -------------------------------
    # Build input row
    # -------------------------------
    input_df = pd.DataFrame([{
        "age_census": age,
        "cough_d": cough_d,
        "fever_d": fever_d,
        "wloss_d": wloss_d,
        "sputum_d": sputum_d,

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
        "bact": 1 if genexpert == "Positive" else 0,

        "region": region,
        "married": MARITAL_MAP[married],
        "edu": EDU_MAP[education],
        "occupation": OCCUPATION_MAP[occupation]
    }])

    # -------------------------------
    # Load models
    # -------------------------------
    model, feature_names = load_ttvae()
    kmeans = load_cluster_model()
    ood_threshold = load_ood_threshold()

    # -------------------------------
    # Load preprocessor SAFELY
    # -------------------------------
    if not os.path.exists("preprocessor.pkl"):
        st.error(
            "❌ The trained preprocessor file (`preprocessor.pkl`) was not found.\n\n"
            "Please ensure it is saved during training and placed in the app directory."
        )
        st.stop()

    preprocessor = joblib.load("preprocessor.pkl")

    X = preprocessor.transform(input_df)
    X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    # -------------------------------
    # Inference
    # -------------------------------
    latent = compute_latent(model, X)
    pseudotime = float(compute_pseudotime(latent)[0])
    cluster = int(assign_cluster(kmeans, latent)[0])
    ood, recon_error = check_ood(model, X, ood_threshold)
    recon_error = float(np.squeeze(recon_error))

    # -------------------------------
    # Results
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

    st.subheader("Latent Phenotype")
    st.write(f"**{cluster_info[cluster]['name']}**")
    st.caption(cluster_info[cluster]['description'])

    st.subheader("Model Confidence")
    st.write(f"Reconstruction Error: `{recon_error:.4f}`")

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
