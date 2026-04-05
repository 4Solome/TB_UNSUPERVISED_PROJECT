import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

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
st.set_page_config(page_title="TB Risk Profiling System", layout="wide")

st.title("TB Risk Profiling System")
st.caption(
    "A cohort‑based system for latent tuberculosis risk sequencing and "
    "phenotype discovery using a Transformer‑based Tabular Variational Autoencoder."
)

# ============================================================
# LOAD MODELS & ARTIFACTS
# ============================================================
feature_names = load_feature_names()
model = load_ttvae(input_dim=len(feature_names))
kmeans = load_cluster_model()

with open("models/ood_threshold.json", "r") as f:
    OOD_THRESHOLD = json.load(f)["ood_threshold"]




# ============================================================
# PHENOTYPE DEFINITIONS
# ============================================================
PHENOTYPE_INFO = {
    0: ("Low‑Symptom TB Risk", "Low symptom burden with minimal laboratory evidence."),
    1: ("Active Symptomatic TB", "High clinical symptom burden consistent with active TB."),
    2: ("Minimal‑Information Profile", "Sparse diagnostic information and weak signals."),
    3: ("Transitional TB Risk", "Mixed clinical and laboratory signals."),
    4: ("Laboratory‑Confirmed TB", "Strong bacteriological and laboratory evidence.")
}

# ============================================================
# FEATURE GROUPS
# ============================================================
continuous_cols = ["age_census","cough_d","fever_d","wloss_d","sputum_d","tbhist_y","tbtreat_w"]
binary_cols = [
    "sex_census","setting","smoke_now","smoke_past","hiv_res",
    "cough","fever","weight_loss","night_sweats","chest_pain",
    "blood_sputum","sputum","hist_rx","current_rx",
    "xray_normal","smear_pos","culture","cult_pos","bact"
]
categorical_cols = [
    "region","married","edu","occupation",
    "xrayres","central_cxr_res",
    "zn","genexpert","final_result"
]
ALL_COLS = continuous_cols + binary_cols + categorical_cols

# ============================================================
# UPLOAD SECTION
# ============================================================
st.header("Upload Patient Cohort")
uploaded_file = st.file_uploader("Upload a CSV file containing TB patient data", type=["csv"])
analyze = st.button("Analyze Cohort")

results = None
latents = pseudotime = clusters = None

# ============================================================
# ANALYSIS PIPELINE
# ============================================================
if uploaded_file and analyze:

    df_raw = pd.read_csv(uploaded_file)
    df = df_raw.copy()

    # Silent missing column handling
    for c in continuous_cols:
        if c not in df.columns: df[c] = 0.0
    for c in binary_cols:
        if c not in df.columns: df[c] = 0
    for c in categorical_cols:
        if c not in df.columns: df[c] = "Unknown"

    df = df[ALL_COLS]

    pre = build_preprocessor(continuous_cols, binary_cols, categorical_cols)
    dummy = {c:0 for c in continuous_cols + binary_cols}
    dummy.update({c:"Unknown" for c in categorical_cols})
    pre.fit(pd.DataFrame([dummy]))

    X = pre.transform(df)
    X = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    latents = compute_latent(model, X)
    pseudotime = compute_pseudotime(latents)
    clusters = assign_cluster(kmeans, latents)

    def risk_bucket(pt):
        if pt < 0.3: return "Low Risk"
        if pt < 0.7: return "Moderate Risk"
        return "High Risk"

    risk_category = [risk_bucket(p) for p in pseudotime]

    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        rec,_,_ = model(Xt)

    #rec_error = ((rec.numpy() - X) ** 2).mean(axis=1)
    #ood_flag = rec_error > OOD_THRESHOLD
   # reliability = ["⚠️ OOD Warning" if f else "✅ In Distribution" for f in ood_flag]


    rec_error = ((rec.numpy() - X) ** 2).mean(axis=1)

# TEMPORARY: cohort-relative OOD for visualization & screenshots
ood_flag = rec_error > np.percentile(rec_error, 95)

reliability = ["⚠️ OOD Warning" if f else "✅ In Distribution" for f in ood_flag]



    

    results = pd.DataFrame({
        "Pseudotime": np.round(pseudotime,3),
        "Risk Category": risk_category,
        "Phenotype": [PHENOTYPE_INFO[c][0] for c in clusters],
        "Reliability": reliability,
        "Reconstruction Error": np.round(rec_error,3)
    })

# ============================================================
# MAIN CONTENT (ONLY IF ANALYZED)
# ============================================================
if results is not None:

    st.header("Cohort Summary")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Patients", len(results))
    c2.metric("Average Pseudotime", f"{results['Pseudotime'].mean():.2f}")
    c3.metric("OOD Warnings", sum(results["Reliability"].str.contains("OOD")))
    c4.metric("Phenotypes Detected", results["Phenotype"].nunique())

    st.header("Patient‑Level Results")
    st.dataframe(results, use_container_width=True)

    st.header("Data Interpretation")

    # Latent plots
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Latent Space Colored by Phenotype")
        fig,ax = plt.subplots(figsize=(4.5,4))
        for cid,(name,_) in PHENOTYPE_INFO.items():
            mask = clusters==cid
            ax.scatter(latents[mask,0], latents[mask,1], label=name, alpha=0.6)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.legend(fontsize=7)
        st.pyplot(fig)

    with col2:
        st.subheader("Latent Space Pseudotime Gradient")
        fig,ax = plt.subplots(figsize=(4.5,4))
        sc=ax.scatter(latents[:,0],latents[:,1],c=pseudotime,cmap="plasma")
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        plt.colorbar(sc,ax=ax)
        st.pyplot(fig)

    # Histograms row (SMALLER)
    col3,col4 = st.columns(2)

    with col3:
        st.subheader("Phenotype Distribution")
        fig,ax=plt.subplots(figsize=(4,3))
        results["Phenotype"].value_counts().plot(kind="bar",ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col4:
        st.subheader("Pseudotime Distribution")
        fig,ax=plt.subplots(figsize=(4,3))
        ax.hist(results["Pseudotime"],bins=15)
        ax.set_xlabel("Pseudotime")
        st.pyplot(fig)

    # ========================================================
    # DETAILED INTERPRETATION
    # ========================================================
    st.header("Detailed Interpretation")

    with st.expander("Cluster‑Level Summary"):
        summary = results.groupby("Phenotype").agg(
            Count=("Pseudotime","count"),
            Mean_Pseudotime=("Pseudotime","mean"),
            Mean_Reconstruction_Error=("Reconstruction Error","mean")
        ).reset_index()
        st.dataframe(summary)

    with st.expander("Cluster Feature Profiles"):
        prof=df.copy()
        prof["Cluster"]=clusters
        profile_means=prof.groupby("Cluster").mean(numeric_only=True).reset_index()
        profile_means["Phenotype"]=profile_means["Cluster"].map(lambda x:PHENOTYPE_INFO[x][0])
        st.dataframe(profile_means)
        st.download_button(
            "Download Cluster Feature Profiles",
            profile_means.to_csv(index=False),
            file_name="cluster_feature_profiles.csv"
        )

    with st.expander("View Uploaded Data Preview"):
        st.dataframe(df_raw.head(200))

# ============================================================
# SYNTHETIC DATA GENERATION (ALWAYS VISIBLE)
# ============================================================
st.header("Synthetic Patient Generation")

num_samples=st.slider("Number of synthetic patients",10,200,50)

if st.button("Generate Synthetic Patients"):
    z=torch.randn(num_samples,32)
    with torch.no_grad():
        synth=model.decode(z).numpy()
    synth_df=pd.DataFrame(synth,columns=feature_names)
    st.dataframe(synth_df.head())
    st.download_button(
        "Download Synthetic Dataset",
        synth_df.to_csv(index=False),
        file_name="synthetic_patients.csv"
    )

st.caption(
    "This system supports tuberculosis risk profiling for research purposes "
    "and does not replace medical diagnosis."
)
