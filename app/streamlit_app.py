import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    build_preprocessor,
    load_ttvae,
    load_cluster_model,
    load_ood_threshold,
    compute_latent,
    compute_pseudotime,
    check_ood,
    assign_cluster,
    decode_latent
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TB Risk Profiling System",
    layout="centered"
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model, feature_names = load_ttvae()
    kmeans = load_cluster_model()
    threshold = load_ood_threshold()
    return model, feature_names, kmeans, threshold

model, feature_names, kmeans, ood_threshold = load_artifacts()

# ============================================================
# APP TEXT
# ============================================================
st.title("TB Risk Profiling System (TTVAE-Based)")
st.caption(
    "Latent tuberculosis phenotype discovery, pseudotime-based risk sequencing, "
    "and synthetic profile generation using a Transformer-based Tabular Variational Autoencoder."
)

# ============================================================
# PHENOTYPE LABELS
# ============================================================
cluster_info = {
    0: {
        "name": "Low-Symptom TB Risk",
        "description": "Individuals with relatively low symptom burden but still located within the learned tuberculosis risk space."
    },
    1: {
        "name": "Active Symptomatic TB",
        "description": "Patients with stronger clinical symptoms and patterns consistent with active symptomatic tuberculosis."
    },
    2: {
        "name": "Minimal-Information Profile",
        "description": "Cases with sparse or limited diagnostic information and weak overall feature activation."
    },
    3: {
        "name": "Transitional TB Risk",
        "description": "Individuals showing mixed clinical and laboratory signals between lower-risk and confirmed profiles."
    },
    4: {
        "name": "Laboratory-Confirmed TB",
        "description": "Patients with stronger bacteriological and laboratory evidence consistent with confirmed tuberculosis."
    }
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
    "cough", "fever", "weight_loss", "night_sweats", "chest_pain",
    "blood_sputum", "sputum", "hist_rx", "current_rx",
    "xray_normal", "smear_pos", "culture", "cult_pos", "bact"
]

categorical_cols = [
    "region", "married", "edu", "occupation",
    "xrayres", "central_cxr_res", "zn", "genexpert", "final_result"
]

all_expected_cols = continuous_cols + binary_cols + categorical_cols

# ============================================================
# HELPERS
# ============================================================
def risk_label(score: float) -> str:
    if score < 0.3:
        return "Low Risk"
    elif score < 0.7:
        return "Moderate Risk"
    return "High Risk"

def render_risk_bar(score: float):
    st.progress(float(score))
    if score < 0.3:
        st.caption("Low ----|---- Moderate ----|---- High\n↑ Patient")
    elif score < 0.7:
        st.caption("Low ----|---- Moderate ----|---- High\n           ↑ Patient")
    else:
        st.caption("Low ----|---- Moderate ----|---- High\n                        ↑ Patient")

def parse_categories_from_feature_names(feature_names, categorical_cols):
    categories = {}
    for col in categorical_cols:
        vals = []
        prefixes = [f"cat__{col}_", f"categorical__{col}_"]
        for f in feature_names:
            for p in prefixes:
                if f.startswith(p):
                    vals.append(f.replace(p, ""))
        categories[col] = sorted(list(set(vals))) if vals else ["Unknown"]
    return categories

category_options = parse_categories_from_feature_names(feature_names, categorical_cols)

def build_reference_frame():
    anchor_ranges = {
        "age_census": [0, 100],
        "cough_d": [0, 365],
        "fever_d": [0, 365],
        "wloss_d": [0, 2000],
        "sputum_d": [0, 365],
        "tbhist_y": [1900, 2025],
        "tbtreat_w": [0, 365]
    }

    max_len = max(
        2,
        max((len(category_options.get(c, ["Unknown"])) for c in categorical_cols), default=2)
    )

    rows = []
    for i in range(max_len):
        row = {}
        for c in continuous_cols:
            vals = anchor_ranges.get(c, [0, 1])
            row[c] = vals[i % len(vals)]
        for c in binary_cols:
            row[c] = i % 2
        for c in categorical_cols:
            opts = category_options.get(c, ["Unknown"])
            row[c] = opts[i % len(opts)] if opts else "Unknown"
        rows.append(row)

    return pd.DataFrame(rows)

reference_fit_df = build_reference_frame()

def prepare_dataframe(df):
    df = df.copy()

    for c in all_expected_cols:
        if c not in df.columns:
            if c in continuous_cols:
                df[c] = 0.0
            elif c in binary_cols:
                df[c] = 0
            else:
                default_cat = category_options.get(c, ["Unknown"])[0]
                df[c] = default_cat

    df = df[all_expected_cols].copy()

    for c in continuous_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in binary_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1)

    for c in categorical_cols:
        df[c] = df[c].astype(str)

    return df

def transform_with_runtime_reference(df_raw):
    df_raw = prepare_dataframe(df_raw)

    pre = build_preprocessor(
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols
    )

    fit_df = pd.concat([reference_fit_df, df_raw], ignore_index=True)
    pre.fit(fit_df)

    X = pre.transform(df_raw)
    X = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0.0)

    return X.values

def single_patient_pseudotime(latents, kmeans_model):
    z1 = float(latents[0, 0])
    centroid_z1 = kmeans_model.cluster_centers_[:, 0]
    score = (z1 - centroid_z1.min()) / (centroid_z1.max() - centroid_z1.min() + 1e-10)
    return float(np.clip(score, 0.0, 1.0))

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "Analyze Single Patient",
    "Analyze Cohort (CSV Upload)",
    "Generate Synthetic Data"
])

# ============================================================
# TAB 1: SINGLE PATIENT
# ============================================================
with tab1:
    st.subheader("Single Patient Entry")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 0, 100, 35)
        sex = st.selectbox("Sex (0/1 style)", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        setting = st.selectbox("Setting", [0, 1], help="Use dataset-compatible coding")
        hiv_res = st.selectbox("HIV status", [0, 1], format_func=lambda x: "Negative / Unknown" if x == 0 else "Positive")
    with col2:
        smoke_now = st.selectbox("Current smoking", [0, 1])
        smoke_past = st.selectbox("Past smoking", [0, 1])
        hist_rx = st.selectbox("Previous TB treatment", [0, 1])
        current_rx = st.selectbox("Currently on treatment", [0, 1])

    st.markdown("**Symptoms**")
    c1, c2, c3 = st.columns(3)
    with c1:
        cough = st.selectbox("Cough", [0, 1])
        cough_d = st.number_input("Cough duration (days)", 0, 365, 0)
        fever = st.selectbox("Fever", [0, 1])
        fever_d = st.number_input("Fever duration (days)", 0, 365, 0)
    with c2:
        weight_loss = st.selectbox("Weight loss", [0, 1])
        wloss_d = st.number_input("Weight-loss duration (days)", 0, 2000, 0)
        sputum = st.selectbox("Sputum", [0, 1])
        sputum_d = st.number_input("Sputum duration (days)", 0, 365, 0)
    with c3:
        night_sweats = st.selectbox("Night sweats", [0, 1])
        chest_pain = st.selectbox("Chest pain", [0, 1])
        blood_sputum = st.selectbox("Blood in sputum", [0, 1])

    st.markdown("**Radiology and Laboratory**")
    c4, c5, c6 = st.columns(3)
    with c4:
        xray_normal = st.selectbox("X-ray normal", [0, 1])
        xrayres = st.selectbox("X-ray result code", category_options.get("xrayres", ["Unknown"]))
        central_cxr_res = st.selectbox("Central CXR result code", category_options.get("central_cxr_res", ["Unknown"]))
    with c5:
        smear_pos = st.selectbox("Smear positive", [0, 1])
        culture = st.selectbox("Culture positive", [0, 1])
        cult_pos = culture
    with c6:
        bact = st.selectbox("Bacteriological confirmation", [0, 1])
        zn = st.selectbox("ZN result", category_options.get("zn", ["Unknown"]))
        genexpert = st.selectbox("GeneXpert result", category_options.get("genexpert", ["Unknown"]))

    st.markdown("**Sociodemographic Categories**")
    c7, c8, c9 = st.columns(3)
    with c7:
        region = st.selectbox("Region", category_options.get("region", ["Unknown"]))
        married = st.selectbox("Marital status code", category_options.get("married", ["Unknown"]))
    with c8:
        edu = st.selectbox("Education code", category_options.get("edu", ["Unknown"]))
        occupation = st.selectbox("Occupation code", category_options.get("occupation", ["Unknown"]))
    with c9:
        final_result = st.selectbox("Final result code", category_options.get("final_result", ["Unknown"]))
        tbhist_y = st.number_input("TB history year", 1900, 2025, 2000)
        tbtreat_w = st.number_input("TB treatment weeks", 0, 365, 0)

    if st.button("Analyze Patient", type="primary"):
        patient_df = pd.DataFrame([{
            "age_census": age,
            "cough_d": cough_d,
            "fever_d": fever_d,
            "wloss_d": wloss_d,
            "sputum_d": sputum_d,
            "tbhist_y": tbhist_y,
            "tbtreat_w": tbtreat_w,
            "sex_census": sex,
            "setting": setting,
            "smoke_now": smoke_now,
            "smoke_past": smoke_past,
            "hiv_res": hiv_res,
            "cough": cough,
            "fever": fever,
            "weight_loss": weight_loss,
            "night_sweats": night_sweats,
            "chest_pain": chest_pain,
            "blood_sputum": blood_sputum,
            "sputum": sputum,
            "hist_rx": hist_rx,
            "current_rx": current_rx,
            "xray_normal": xray_normal,
            "smear_pos": smear_pos,
            "culture": culture,
            "cult_pos": cult_pos,
            "bact": bact,
            "region": str(region),
            "married": str(married),
            "edu": str(edu),
            "occupation": str(occupation),
            "xrayres": str(xrayres),
            "central_cxr_res": str(central_cxr_res),
            "zn": str(zn),
            "genexpert": str(genexpert),
            "final_result": str(final_result)
        }])

        X = transform_with_runtime_reference(patient_df)
        latents = compute_latent(model, X)
        cluster = int(assign_cluster(kmeans, latents)[0])

        pseudotime = single_patient_pseudotime(latents, kmeans)
        ood_flags, recon_errors = check_ood(model, X, ood_threshold)
        ood_flag = bool(ood_flags[0])
        recon_error = float(recon_errors[0])

        st.divider()
        st.header("Results")

        st.metric("TB Risk Score (Pseudotime)", f"{pseudotime:.2f}")

        risk_cat = risk_label(pseudotime)
        if risk_cat == "Low Risk":
            st.success("Risk Category: Low Risk")
        elif risk_cat == "Moderate Risk":
            st.warning("Risk Category: Moderate Risk")
        else:
            st.error("Risk Category: High Risk")

        render_risk_bar(pseudotime)

        st.subheader("Latent Phenotype")
        st.write(f"**{cluster_info[cluster]['name']}**")
        st.caption(cluster_info[cluster]["description"])

        st.subheader("Reliability Assessment")
        if ood_flag:
            st.warning("⚠️ Input lies outside training distribution.")
        else:
            st.success("✅ Input lies within training distribution.")

        st.subheader("Model Confidence")
        st.write(f"Reconstruction Error: `{recon_error:.4f}`")

        single_result = pd.DataFrame({
            "TB_Risk_Score": [pseudotime],
            "Risk_Category": [risk_cat],
            "Cluster": [cluster],
            "Phenotype": [cluster_info[cluster]["name"]],
            "OOD_Flag": [ood_flag],
            "Reconstruction_Error": [recon_error],
            "z1": [latents[0, 0]],
            "z2": [latents[0, 1]]
        })

        st.download_button(
            "Download Patient Result CSV",
            single_result.to_csv(index=False),
            file_name="single_patient_tb_result.csv",
            mime="text/csv"
        )

# ============================================================
# TAB 2: COHORT UPLOAD
# ============================================================
with tab2:
    st.subheader("Cohort Analysis (CSV Upload)")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="cohort_uploader")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("**Uploaded Data Preview**")
        st.dataframe(df.head(), use_container_width=True)

        X = transform_with_runtime_reference(df)
        latents = compute_latent(model, X)
        clusters = assign_cluster(kmeans, latents)
        pseudotime = compute_pseudotime(latents)
        ood_flags, recon_errors = check_ood(model, X, ood_threshold)

        result_df = df.copy()
        result_df["Cluster"] = clusters
        result_df["Phenotype"] = [cluster_info[int(c)]["name"] for c in clusters]
        result_df["Pseudotime"] = pseudotime
        result_df["OOD_Flag"] = ood_flags
        result_df["Reconstruction_Error"] = recon_errors
        result_df["z1"] = latents[:, 0]
        result_df["z2"] = latents[:, 1]

        st.divider()
        st.header("Cohort Results")
        st.dataframe(result_df, use_container_width=True)

        cluster_summary = (
            result_df
            .groupby(["Cluster", "Phenotype"])
            .agg(
                Count=("Cluster", "count"),
                Mean_Pseudotime=("Pseudotime", "mean"),
                Mean_Reconstruction_Error=("Reconstruction_Error", "mean")
            )
            .reset_index()
            .sort_values("Mean_Pseudotime")
        )

        st.subheader("Cluster-Level Summary")
        st.dataframe(cluster_summary, use_container_width=True)

        # Cluster scatter
        st.subheader("Latent Space Visualization by Cluster")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            x=latents[:, 0],
            y=latents[:, 1],
            hue=clusters,
            palette="tab10",
            alpha=0.7,
            s=25,
            ax=ax1
        )
        ax1.set_xlabel("Latent Dimension 1 (z1)")
        ax1.set_ylabel("Latent Dimension 2 (z2)")
        ax1.set_title("Latent Space Colored by Cluster")
        ax1.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig1)

        # Pseudotime scatter
        st.subheader("Latent Space Pseudotime Gradient")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sc = ax2.scatter(
            latents[:, 0],
            latents[:, 1],
            c=pseudotime,
            cmap="plasma",
            alpha=0.7,
            s=25
        )
        fig2.colorbar(sc, ax=ax2, label="Pseudotime")
        ax2.set_xlabel("Latent Dimension 1 (z1)")
        ax2.set_ylabel("Latent Dimension 2 (z2)")
        ax2.set_title("Latent Space Colored by Pseudotime")
        st.pyplot(fig2)

        st.download_button(
            "Download Cohort Results CSV",
            result_df.to_csv(index=False),
            file_name="cohort_tb_results.csv",
            mime="text/csv"
        )

# ============================================================
# TAB 3: SYNTHETIC DATA
# ============================================================
with tab3:
    st.subheader("Synthetic Patient Generation")
    st.caption(
        "Generate synthetic tuberculosis patient profiles from the learned latent space."
    )

    num_samples = st.slider("Number of synthetic patients", 10, 200, 50)

    if st.button("Generate Synthetic Patients"):
        latent_dim = model.mu.out_features
        z = np.random.normal(0, 1, size=(num_samples, latent_dim))

        synthetic = decode_latent(model, z)
        syn_df = pd.DataFrame(synthetic, columns=feature_names)

        latent_syn = compute_latent(model, synthetic)
        pseudo_syn = compute_pseudotime(latent_syn)
        cluster_syn = assign_cluster(kmeans, latent_syn)

        synth_summary = pd.DataFrame({
            "Metric": [
                "Number of Profiles",
                "Mean Pseudotime",
                "Min Pseudotime",
                "Max Pseudotime"
            ],
            "Value": [
                num_samples,
                float(np.mean(pseudo_syn)),
                float(np.min(pseudo_syn)),
                float(np.max(pseudo_syn))
            ]
        })

        cluster_counts = (
            pd.Series(cluster_syn)
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Count")
        )
        cluster_counts["Phenotype"] = cluster_counts["Cluster"].map(
            lambda x: cluster_info[int(x)]["name"]
        )

        st.success(f"Generated {num_samples} synthetic patient profiles.")
        st.markdown("**Synthetic Data Summary**")
        st.dataframe(synth_summary, use_container_width=True)

        st.markdown("**Synthetic Phenotype Distribution**")
        st.dataframe(cluster_counts, use_container_width=True)

        st.markdown("**Synthetic Data Preview**")
        st.dataframe(syn_df.head(10), use_container_width=True)

        st.download_button(
            "Download Synthetic Dataset (CSV)",
            syn_df.to_csv(index=False),
            file_name="synthetic_tb_patients.csv",
            mime="text/csv"
        )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "This system supports latent tuberculosis risk profiling, phenotype discovery, cohort exploration, "
    "and synthetic data generation using unsupervised representation learning. It does not replace clinical diagnosis."
)
