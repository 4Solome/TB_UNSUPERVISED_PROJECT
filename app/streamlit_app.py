import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import (
    build_preprocessor,
    load_ttvae,
    load_cluster_model,
    load_ood_threshold,
    compute_latent,
    compute_pseudotime,
    check_ood,
)

# ============================================================
# PATHS (MATCH YOUR REPO STRUCTURE)
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

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
# PHENOTYPE LABELS (FIXED)
# ============================================================
cluster_info = {
    0: ("Low‑Symptom TB Risk",
        "Low symptom burden within the learned TB‑risk space."),
    1: ("Active Symptomatic TB",
        "High clinical symptom burden consistent with active TB."),
    2: ("Minimal‑Information Profile",
        "Sparse diagnostic information with weak latent signals."),
    3: ("Transitional TB Risk",
        "Intermediate phenotype between early and confirmed TB."),
    4: ("Laboratory‑Confirmed TB",
        "Strong bacteriological and laboratory evidence of TB.")
}

# ============================================================
# TRAINING FEATURE GROUPS (MUST MATCH TRAINING)
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
# LOAD TRAINED ARTIFACTS (ONCE)
# ============================================================
@st.cache_resource
def load_artifacts():
    model, feature_names = load_ttvae()
    kmeans = load_cluster_model()
    threshold = load_ood_threshold()
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    return model, kmeans, threshold, feature_names, preprocessor

model, kmeans, threshold, feature_names, trained_preprocessor = load_artifacts()

# ============================================================
# INPUT MODE SELECTION
# ============================================================
st.header("Input Mode")
mode = st.radio(
    "Select analysis mode:",
    ["Single Patient", "Cohort (CSV Upload)"]
)

# ============================================================
# MODE 1 — SINGLE PATIENT (REFERENCE‑BASED)
# ============================================================
if mode == "Single Patient":

    st.header("Single Patient Entry")

    # --- Demographics
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

    # --- Symptoms
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

    # --- Lab
    xray = st.selectbox("Chest X‑ray", ["Normal", "Abnormal"])
    smear = st.selectbox("Smear microscopy", ["Negative", "Positive"])
    culture = st.selectbox("Culture", ["Negative", "Positive"])
    genexpert = st.selectbox("GeneXpert", ["Negative", "Positive"])

    if st.button("Analyze Single Patient"):

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
            "smoke_now": 0,
            "smoke_past": 0,
            "hiv_res": 0,
            "hist_rx": 0,
            "xray_normal": 1 if xray == "Normal" else 0,
            "smear_pos": 1 if smear == "Positive" else 0,
            "culture": 1 if culture == "Positive" else 0,
            "cult_pos": 1 if culture == "Positive" else 0,
            "bact": 1 if genexpert == "Positive" else 0,

            "region": region,
            "married": married,
            "edu": education,
            "occupation": occupation
        }])

        X = trained_preprocessor.transform(input_df)
        X = pd.DataFrame(X, columns=trained_preprocessor.get_feature_names_out())
        X = X.reindex(columns=feature_names, fill_value=0).values

        latents = compute_latent(model, X)
        pseudotime = float(np.clip(compute_pseudotime(latents)[0], 0.0, 1.0))
        cluster = int(kmeans.predict(latents)[0])
        recon = float(np.squeeze(check_ood(model, X, threshold)[1]))

        st.subheader("Results")
        st.warning(
            "Single‑patient pseudotime is reference‑based and may appear low. "
            "Cohort analysis provides stable progression ordering."
        )

        st.metric("Pseudotime (reference‑based)", f"{pseudotime:.2f}")

        name, desc = cluster_info[cluster]
        st.subheader("Phenotype (approximate)")
        st.write(f"**{name}**")
        st.caption(desc)

        st.write(f"Reconstruction Error: `{recon:.2f}`")

# ============================================================
# MODE 2 — CSV COHORT (TRUE PSEUDOTIME + CLUSTERING)
# ============================================================
else:
    st.header("Cohort Analysis (CSV Upload)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to compute cohort‑based pseudotime and phenotypes.")
    else:
        df = pd.read_csv(uploaded_file)

        st.subheader("CSV Preview")
        st.dataframe(df.head())

        required_cols = continuous_cols + binary_cols + categorical_cols
        missing = set(required_cols) - set(df.columns)

        if missing:
            st.error(f"Missing required columns: {sorted(missing)}")
            st.stop()

        # ✅ USE TRAINING PREPROCESSOR (THIS FIXES PHENOTYPES)
        X = trained_preprocessor.transform(df)
        X = pd.DataFrame(X, columns=trained_preprocessor.get_feature_names_out())
        X = X.reindex(columns=feature_names, fill_value=0).values

        latents = compute_latent(model, X)

        # ✅ TRUE COHORT PSEUDOTIME
        z1 = latents[:, 0]
        pseudotime = (z1 - z1.min()) / (z1.max() - z1.min() + 1e-10)
        pseudotime = np.clip(pseudotime, 0.0, 1.0)

        clusters = kmeans.predict(latents)

        df["pseudotime"] = pseudotime
        df["phenotype"] = [cluster_info[c][0] for c in clusters]

        st.subheader("Cohort Results")
        st.dataframe(df.sort_values("pseudotime", ascending=False), use_container_width=True)

        st.subheader("Pseudotime Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["pseudotime"], bins=20)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Number of patients")
        st.pyplot(fig)

        st.caption(
            "Pseudotime here reflects true relative progression across the uploaded cohort."
        )
    
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
