import json
import joblib
import os

# --------------------------------------------------------
# LOAD TRAINED ARTIFACTS (ONCE)
# --------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.joblib")
    kmeans = joblib.load("kmeans_model.joblib")

    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)

    with open("ood_threshold.json", "r") as f:
        ood_threshold = json.load(f)["threshold"]

    model = load_ttvae()[0]

    return preprocessor, kmeans, feature_names, ood_threshold, model


# --------------------------------------------------------
# ANALYZE PATIENT
# --------------------------------------------------------
if st.button("Analyze Patient"):

    preprocessor, kmeans, feature_names, ood_threshold, model = load_artifacts()

    # ---- build input_df exactly as before ----
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
        "cult_pos": 1 if culture == "Positive" else 0,
        "bact": 1 if genexpert == "Positive" else 0,

        "region": region,
        "married": MARITAL_MAP[married],
        "edu": EDU_MAP[education],
        "occupation": OCCUPATION_MAP[occupation]
    }])

    # ----------------------------------------------------
    # ✅ CORRECT PREPROCESSING (NO FITTING)
    # ----------------------------------------------------
    X = preprocessor.transform(input_df)
    X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
    X = X.reindex(columns=feature_names, fill_value=0).values

    # ----------------------------------------------------
    # MODEL INFERENCE
    # ----------------------------------------------------
    latent = compute_latent(model, X)
    pseudotime = float(compute_pseudotime(latent)[0])
    cluster = int(assign_cluster(kmeans, latent)[0])
    ood, recon_error = check_ood(model, X, ood_threshold)

    recon_error = float(np.squeeze(recon_error))

    # ----------------------------------------------------
    # RESULTS
    # ----------------------------------------------------
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
    st.caption(cluster_info[cluster]["description"])

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
