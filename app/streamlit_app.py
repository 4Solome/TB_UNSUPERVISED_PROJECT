from utils import build_preprocessor

# --------------------------------------------------------
# DEFINE COLUMN GROUPS (EXACTLY AS TRAINING)
# --------------------------------------------------------
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

# --------------------------------------------------------
# BUILD PREPROCESSOR (NO PICKLING)
# --------------------------------------------------------
preprocessor = build_preprocessor(
    continuous_cols=continuous_cols,
    binary_cols=binary_cols,
    categorical_cols=categorical_cols
)

# --------------------------------------------------------
# FIT ON FIRST USE (SAFE, REQUIRED)
# --------------------------------------------------------
# We fit once on a zero-row template to initialize encoders
template = pd.DataFrame(
    [{c: 0 for c in continuous_cols + binary_cols}] |
    [{c: "Unknown" for c in categorical_cols}]
)

preprocessor.fit(template)

# --------------------------------------------------------
# TRANSFORM INPUT
# --------------------------------------------------------
X = preprocessor.transform(input_df)

X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
X = X.reindex(columns=feature_names, fill_value=0).values

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
