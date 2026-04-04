import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    build_preprocessor,
    load_ttvae,
    load_ood_threshold,
    load_cluster_model,
    compute_latent,
    compute_pseudotime,
    check_ood,
    assign_cluster
)

st.set_page_config(page_title="TTVAE TB Risk Sequencer", layout="wide")

st.title("TTVAE Tuberculosis Risk Sequencer")
st.markdown(
    "Upload a TB patient CSV to obtain pseudotime risk scores, "
    "phenotype assignments, and reliability indicators."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# ---------------------------------------------------------------------
# Feature groups (must match training definitions)
# ---------------------------------------------------------------------

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
    "xrayres", "central_cxr_res",
    "zn", "genexpert", "final_result"
]

# ---------------------------------------------------------------------
# App logic
# ---------------------------------------------------------------------

if uploaded:

    df = pd.read_csv(uploaded)
    st.write("Uploaded Data Preview:", df.head())

    # ---------------------------------------------------------------
    # Align schema with available columns
    # ---------------------------------------------------------------
    available_cols = set(df.columns)

    cont = [c for c in continuous_cols if c in available_cols]
    bin_ = [c for c in binary_cols if c in available_cols]
    cat  = [c for c in categorical_cols if c in available_cols]

    missing = set(continuous_cols + binary_cols + categorical_cols) - available_cols
    if missing:
        st.warning(
            "The following expected columns were missing and ignored:\n"
            + ", ".join(sorted(missing))
        )

    # ---------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------
    pre = build_preprocessor(cont, bin_, cat)
    X = pre.fit_transform(df)

    # ---------------------------------------------------------------
    # Load model + ensure fixed training input space
    # ---------------------------------------------------------------
    model, feature_names = load_ttvae()

    X_df = pd.DataFrame(X, columns=pre.get_feature_names_out())
    X_df = X_df.reindex(columns=feature_names, fill_value=0)
    X = X_df.values

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    threshold = load_ood_threshold()
    kmeans = load_cluster_model()

    latents = compute_latent(model, X)
    pseudotime = compute_pseudotime(latents)
    clusters = assign_cluster(kmeans, latents)
    ood_flags, errors = check_ood(model, X, threshold)

    # ---------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------
    st.subheader("Patient-Level Results")

    results = pd.DataFrame({
        "Pseudotime": pseudotime,
        "Cluster": clusters,
        "OOD_Flag": ood_flags,
        "Reconstruction_Error": errors
    })

    st.dataframe(results)

    st.download_button(
        "Download Results CSV",
        results.to_csv(index=False),
        "ttvae_results.csv"
    )

    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------
    st.subheader("Latent Space & Pseudotime")

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        latents[:, 0],
        latents[:, 1],
        c=pseudotime,
        cmap="plasma",
        alpha=0.7
    )
    fig.colorbar(sc, ax=ax, label="Pseudotime")
    ax.set_xlabel("Latent Dimension 1 (z1)")
    ax.set_ylabel("Latent Dimension 2 (z2)")
    ax.set_title("Latent Space Pseudotime Gradient")

    st.pyplot(fig)
