"""
TFM — Quantum Machine Learning in a DataOps Pipeline
Integración de QML en pipeline DataOps sobre Databricks
Universidad Europea de Valencia | Juan Albornoz Carrasco
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QML DataOps Pipeline | TFM",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── TFM colour palette ─────────────────────────────────────────────────────────
PRIMARY   = "#5D8BA6"
LIGHT1    = "#86A8BC"
LIGHT2    = "#AEC5D2"
LIGHT3    = "#D7E2E9"
DARK      = "#3D6C87"
BG        = "#F8FBFD"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Global */
    .stApp {{ background-color: {BG}; }}
    h1 {{ color: {DARK}; font-family: 'Arial', sans-serif; }}
    h2 {{ color: {PRIMARY}; font-family: 'Arial', sans-serif; }}
    h3 {{ color: {LIGHT1}; font-family: 'Arial', sans-serif; }}

    /* Hero banner */
    .hero {{
        background: linear-gradient(135deg, {DARK} 0%, {PRIMARY} 60%, {LIGHT1} 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }}
    .hero h1 {{ color: white; font-size: 2rem; margin: 0; }}
    .hero p  {{ color: {LIGHT3}; margin: 0.5rem 0 0 0; font-size: 1rem; }}

    /* Metric cards */
    .metric-card {{
        background: white;
        border: 1px solid {LIGHT3};
        border-left: 4px solid {PRIMARY};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(93,139,166,0.08);
    }}
    .metric-card .val {{
        font-size: 1.8rem;
        font-weight: bold;
        color: {DARK};
    }}
    .metric-card .lbl {{
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-card .mdl {{
        font-size: 0.85rem;
        font-weight: bold;
        color: {PRIMARY};
        margin-bottom: 0.3rem;
    }}

    /* Info box */
    .info-box {{
        background: white;
        border: 1px solid {LIGHT3};
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: white;
        border-right: 1px solid {LIGHT3};
    }}

    /* Section divider */
    .section-divider {{
        border: none;
        border-top: 2px solid {LIGHT3};
        margin: 2rem 0;
    }}

    /* Badge */
    .badge {{
        display: inline-block;
        background: {LIGHT3};
        color: {DARK};
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        font-weight: bold;
        margin: 0.2rem;
    }}

    /* Warning box */
    .warn-box {{
        background: #FFF8E1;
        border-left: 4px solid #F9A825;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
    }}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
FIGURES = os.path.join(os.path.dirname(__file__), '..', 'figures')

def load_figure(filename):
    path = os.path.join(FIGURES, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

def metric_card(model, metric, value, html=True):
    card = f"""
    <div class="metric-card">
        <div class="mdl">{model}</div>
        <div class="val">{value}</div>
        <div class="lbl">{metric}</div>
    </div>"""
    if html:
        st.markdown(card, unsafe_allow_html=True)
    return card

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2.5rem;">⚛️</div>
        <div style="font-weight:bold; color:{DARK}; font-size:1rem;">QML DataOps Pipeline</div>
        <div style="color:#999; font-size:0.78rem;">Master's Thesis | UEV 2025–2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Results", "🔍 SHAP Analysis",
         "⚛️ Quantum Circuit", "🌐 Bloch Sphere", "🩺 Live Predictor"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown(f"""
    <div style="font-size:0.75rem; color:#aaa; text-align:center;">
        <b style="color:{PRIMARY};">Juan Albornoz Carrasco</b><br>
        Universidad Europea de Valencia<br>
        Director: Prof. Ronal Muresano<br><br>
        <a href="https://github.com/juan-albornoz/tfm-qml-databricks-pipeline"
           style="color:{PRIMARY};">📂 GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown(f"""
    <div class="hero">
        <h1>⚛️ Quantum Machine Learning in a DataOps Pipeline</h1>
        <p>Master's Thesis · Universidad Europea de Valencia · 2025–2026</p>
        <p style="margin-top:0.8rem;">
            <span class="badge">Databricks CE</span>
            <span class="badge">AWS S3</span>
            <span class="badge">Delta Lake</span>
            <span class="badge">Qiskit 2.4.1</span>
            <span class="badge">LightGBM</span>
            <span class="badge">SVM-RBF</span>
            <span class="badge">QSVM</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    st.subheader("Dataset — NHANES (CDC, 3 biennial cycles 2017–2022)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Bronze", "Records", "29,400")
    with c2: metric_card("Silver", "Records", "7,831")
    with c3: metric_card("Gold", "Features", "89")
    with c4: metric_card("Target", "Class balance", "86% / 14%")
    with c5: metric_card("Validation", "DFE expectations", "15 / 15 ✅")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Pipeline architecture
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("DataOps Pipeline Architecture")
        st.markdown(f"""
        <div class="info-box">
        <b style="color:{DARK};">Medallion Architecture · Bronze → Silver → Gold</b>
        <hr style="border-color:{LIGHT3}; margin:0.6rem 0;">

        <b>🟤 Bronze Layer</b> — Raw ingestion from AWS S3<br>
        <span style="color:#888; font-size:0.88rem;">27 XPT files via boto3 · 29,400 records · 162 columns · Delta Lake versioned</span>
        <br><br>

        <b>⚪ Silver Layer</b> — Cleaning & transformation<br>
        <span style="color:#888; font-size:0.88rem;">Age filter · DIQ010 binarisation · leakage exclusion · winsorisation · imputation<br>
        Formal validation: dataframe-expectations 0.7.0 · 15/15 PASSED</span>
        <br><br>

        <b>🟡 Gold Layer</b> — Modelling-ready datasets<br>
        <span style="color:#888; font-size:0.88rem;">OHE · correlation removal (16 vars, r>0.90) · 80/20 stratified split · StandardScaler<br>
        Top-8 features for QSVM selected by Random Forest importance</span>
        <br><br>

        <b>🤖 Models</b> — Triangulated comparison<br>
        <span style="color:#888; font-size:0.88rem;">LightGBM (GridSearchCV 5-fold) · SVM-RBF · QSVM (ZZFeatureMap 8q, reps=2)<br>
        Same SVM classifier · only kernel varies → isolates quantum effect</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Technical Stack")
        stack = {
            "Platform": "Databricks Community Edition (Serverless)",
            "Storage": "AWS S3 (eu-west-1) + Unity Catalog",
            "Data format": "Delta Lake (ACID, time travel)",
            "Classical ML": "LightGBM 4.x · scikit-learn 1.6.1",
            "Quantum ML": "Qiskit 2.4.1 · qiskit-machine-learning 0.9.0",
            "Explainability": "SHAP (TreeExplainer + KernelExplainer)",
            "Serialisation": "ONNX (LightGBM, SVM-RBF)",
            "Validation": "dataframe-expectations 0.7.0",
        }
        for k, v in stack.items():
            st.markdown(f"""
            <div style="padding:0.4rem 0; border-bottom:1px solid {LIGHT3};">
                <span style="color:{PRIMARY}; font-weight:bold; font-size:0.82rem;">{k}</span><br>
                <span style="font-size:0.88rem;">{v}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Distribution figure
    st.subheader("Target Variable Distribution — Bronze Layer")
    fig = load_figure("Distribution DIQ10.png")
    if fig:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(fig, caption="DIQ010 distribution across 3 NHANES cycles. "
                     "Value 2 (No diabetes) dominates. Value 1 (Diabetes) is the positive class.",
                     use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Results":

    st.markdown(f'<h1 style="color:{DARK};">📊 Experimental Results</h1>', unsafe_allow_html=True)
    st.markdown("Triangulated comparison of three models evaluated on the **same test set of 1,567 instances**.")

    # Metrics table
    st.subheader("Performance Metrics Summary")
    df_metrics = pd.DataFrame({
        "Model":    ["LightGBM", "SVM-RBF", "QSVM"],
        "AUC-ROC":  [0.9485, 0.9377, 0.5493],
        "F1-macro": [0.6523, 0.8243, 0.4669],
        "Accuracy": [0.7243, 0.9075, 0.8602],
        "MCC":      [0.4566, 0.6539, 0.0625],
        "Best in":  ["AUC-ROC", "F1, Accuracy, MCC", "—"],
    })

    def highlight_best(s):
        if s.name in ["AUC-ROC", "F1-macro", "Accuracy", "MCC"]:
            is_max = s == s.max()
            return [f"background-color:{LIGHT3}; font-weight:bold;" if v else "" for v in is_max]
        return [""] * len(s)

    st.dataframe(
        df_metrics.style.apply(highlight_best).format({
            "AUC-ROC": "{:.4f}", "F1-macro": "{:.4f}",
            "Accuracy": "{:.4f}", "MCC": "{:.4f}"
        }),
        use_container_width=True, hide_index=True
    )

    st.markdown(f"""
    <div class="warn-box">
        ⚠️ <b>QSVM note:</b> trained on a stratified sample of 500 instances due to O(n²) kernel cost.
        All three models are evaluated on the full 1,567-instance test set for comparability.
        The QSVM AUC-ROC of 0.5493 confirms no discriminative advantage over the classical RBF kernel
        in this tabular clinical dataset, consistent with current NISQ-era literature.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Comparative figure
    st.subheader("Metrics Comparison — All Models")
    fig = load_figure("Comparativa visual de métricas de evaluación de los tres modelos.png")
    if fig:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(fig, caption="Visual comparison of AUC-ROC, F1-macro, Accuracy and MCC "
                     "across the three models. SVM-RBF leads in most metrics.",
                     use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ROC curves
    st.subheader("ROC Curves")
    col1, col2, col3 = st.columns(3)
    roc_files = [
        ("Curva ROC LGBM.png",    "LightGBM",  "AUC = 0.9485"),
        ("Curva ROC SVM-RBF.png", "SVM-RBF",   "AUC = 0.9377"),
        ("Curva ROC QSVM.png",    "QSVM",      "AUC = 0.5493"),
    ]
    for col, (fname, model, auc) in zip([col1, col2, col3], roc_files):
        with col:
            fig = load_figure(fname)
            if fig:
                st.image(fig, caption=f"{model} · {auc}", use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    col1, col2, col3 = st.columns(3)
    cm_files = [
        ("Matriz de Confucion Lgbm.png",   "LightGBM"),
        ("Matriz de Confusion SVM_RBF.png","SVM-RBF"),
        ("Matriz de confusión QSVM.png",   "QSVM"),
    ]
    for col, (fname, model) in zip([col1, col2, col3], cm_files):
        with col:
            fig = load_figure(fname)
            if fig:
                st.image(fig, caption=f"Confusion Matrix — {model}", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Analysis":

    st.markdown(f'<h1 style="color:{DARK};">🔍 SHAP Explainability Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    SHAP (SHapley Additive exPlanations) quantifies the contribution of each clinical feature
    to individual predictions. **TreeExplainer** (exact values) was applied to LightGBM,
    and **KernelExplainer** (approximation over 200 test instances) to SVM-RBF.
    SHAP was not applied to QSVM due to the prohibitive computational cost of the quantum kernel.
    """)

    tab1, tab2 = st.tabs(["🌲 LightGBM — TreeExplainer", "🔵 SVM-RBF — KernelExplainer"])

    with tab1:
        st.subheader("LightGBM — SHAP Analysis (TreeExplainer · 1,567 test instances)")

        col1, col2 = st.columns(2)
        with col1:
            fig = load_figure("SHAP Summary LGBM.png")
            if fig:
                st.image(fig, caption="SHAP Summary Plot — LightGBM: direction and magnitude "
                         "of each feature's impact on predictions.",
                         use_container_width=True)
        with col2:
            fig = load_figure("SHAP Feature Importance.png")
            if fig:
                st.image(fig, caption="SHAP Feature Importance — LightGBM: mean absolute "
                         "SHAP value for the top 20 features.",
                         use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
        <b style="color:{DARK};">Key clinical findings — LightGBM</b><br><br>
        <b>LBXGH (HbA1c)</b> dominates with a mean |SHAP| = 1.1243, consistent with its role
        as the primary diagnostic marker for Type 2 diabetes (ADA threshold: HbA1c ≥ 6.5%).
        High values (pink/red) strongly push predictions towards the positive class.<br><br>
        <b>RIDAGEYR (age)</b> ranks second (0.4654), reflecting the well-documented increase
        in T2D prevalence with age.<br><br>
        <b>LBXGLU (fasting glucose)</b> and <b>LBDLDL (LDL cholesterol)</b> complete the
        biochemical marker block, followed by <b>BMXWAIST (waist circumference)</b> as the
        main anthropometric indicator of insulin resistance.<br><br>
        <b>WTINT2YR</b> (interview sample weight) appears at position 6 — this is a NHANES
        sampling design artefact, not a clinical variable.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.subheader("SVM-RBF — SHAP Analysis (KernelExplainer · 200 test instances)")

        col1, col2 = st.columns(2)
        with col1:
            fig = load_figure("SHAP Summary SVM.png")
            if fig:
                st.image(fig, caption="SHAP Summary Plot — SVM-RBF: direction and magnitude "
                         "of each feature's impact on predictions.",
                         use_container_width=True)
            else:
                st.info("SHAP Summary SVM figure not found in figures/")
        with col2:
            fig = load_figure("SHAP Bar SVM.png")
            if fig:
                st.image(fig, caption="SHAP Feature Importance — SVM-RBF: mean absolute "
                         "SHAP value for the top 20 features.",
                         use_container_width=True)
            else:
                st.info("SHAP Bar SVM figure not found in figures/")

        st.markdown(f"""
        <div class="info-box">
        <b style="color:{DARK};">Key clinical findings — SVM-RBF</b><br><br>
        Both LightGBM and SVM-RBF rank <b>LBXGH (HbA1c)</b> and <b>LBXGLU (fasting glucose)</b>
        as the two most influential features, confirming convergence in the clinical logic
        underlying both models. This agreement strengthens the clinical validity of the
        experimental results beyond performance metrics alone.<br><br>
        Unlike TreeExplainer, KernelExplainer treats the model as a black box and produces
        <b>approximate Shapley values</b> via sampling. Results should be interpreted as
        representative estimates rather than exact values.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — QUANTUM CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚛️ Quantum Circuit":

    st.markdown(f'<h1 style="color:{DARK};">⚛️ Quantum Circuit — ZZFeatureMap</h1>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Circuit Configuration")
        cfg = {
            "Feature map":    "ZZFeatureMap",
            "Qubits (n)":     "8 (one per clinical feature)",
            "Repetitions":    "2 (reps=2)",
            "Entanglement":   "Linear (adjacent qubits)",
            "Kernel":         "FidelityQuantumKernel",
            "Fidelity":       "ComputeUncompute",
            "Backend":        "StatevectorSampler (noiseless simulation)",
            "Training set":   "500 stratified instances",
            "Training time":  "~21 min",
            "Inference time": "~132 min (1,567 instances, batches of 100)",
            "Kernel cost":    "O(n²) — 250,000 evaluations for training",
        }
        for k, v in cfg.items():
            st.markdown(f"""
            <div style="padding:0.35rem 0; border-bottom:1px solid {LIGHT3};">
                <span style="color:{PRIMARY}; font-weight:bold; font-size:0.82rem;">{k}</span><br>
                <span style="font-size:0.88rem;">{v}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Selected QSVM Features (top-8 by RF importance)")
        features = [
            ("LBXGH",    "HbA1c — glycated haemoglobin"),
            ("LBXGLU",   "Fasting plasma glucose"),
            ("LBDLDL",   "LDL cholesterol"),
            ("RIDAGEYR", "Age at screening"),
            ("BMXWAIST", "Waist circumference"),
            ("LBXIN",    "Serum insulin"),
            ("BMXBMI",   "Body Mass Index"),
            ("BMXLEG",   "Upper leg length"),
        ]
        for code, desc in features:
            st.markdown(f"""
            <div style="padding:0.35rem 0; border-bottom:1px solid {LIGHT3};">
                <b style="color:{DARK}; font-family:monospace;">{code}</b><br>
                <span style="font-size:0.85rem; color:#555;">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.subheader("ZZFeatureMap Circuit — 3 representative qubits")
    fig = load_figure("Circuito ZZFeatureMap.png")
    if fig:
        st.image(fig,
                 caption="ZZFeatureMap circuit for 3 representative qubits (reps=2, linear entanglement). "
                         "H gates create superposition. P(2·x[i]) gates encode each clinical variable "
                         "as a rotation angle. P(2·(π−x[i])·(π−x[j])) gates generate entanglement "
                         "between adjacent qubit pairs, capturing feature correlations.",
                 use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.subheader("How the Quantum Kernel Works")
    st.markdown(f"""
    <div class="info-box">
    <b style="color:{DARK};">K(x, y) = |⟨ψ(x)|ψ(y)⟩|²</b><br><br>
    The <b>FidelityQuantumKernel</b> computes the similarity between two data points x and y
    as the squared overlap between their quantum states |ψ(x)⟩ and |ψ(y)⟩, generated by
    applying the ZZFeatureMap circuit to each point.<br><br>
    The <b>ComputeUncompute</b> algorithm implements this in two steps:
    <ol>
        <li><b>Compute:</b> apply ZZFeatureMap with data point x → generate |ψ(x)⟩</li>
        <li><b>Uncompute:</b> apply the inverse circuit with data point y → measure probability of |0…0⟩</li>
    </ol>
    The measured probability equals |⟨ψ(x)|ψ(y)⟩|², which is the kernel value.
    This operation is performed for every pair of training instances (500² = 250,000 evaluations)
    and for every test instance against all training instances (1,567 × 500 = 783,500 evaluations).<br><br>
    <b>Key insight:</b> the ZZFeatureMap operates in a 2⁸ = 256-dimensional Hilbert space,
    far larger than the 8-dimensional input space. Despite this, the FidelityQuantumKernel
    does not outperform the RBF kernel on this tabular clinical dataset, consistent with
    current NISQ-era literature on tabular data.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Bloch Sphere":

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    st.markdown(f'<h1 style="color:{DARK};">🌐 Bloch Sphere Emulator</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    The **Bloch sphere** is the geometric representation of the quantum state space of a
    single qubit. This emulator shows how the **ZZFeatureMap** encodes a clinical variable
    as a rotation angle, transforming classical data into a quantum state.
    """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("⚙️ Qubit Configuration")

        feature = st.selectbox(
            "Select clinical feature (qubit)",
            options=[
                "LBXGH — HbA1c (%)",
                "LBXGLU — Fasting Glucose (mg/dL)",
                "LBDLDL — LDL Cholesterol (mg/dL)",
                "RIDAGEYR — Age (years)",
                "BMXWAIST — Waist Circumference (cm)",
                "LBXIN — Serum Insulin (µU/mL)",
                "BMXBMI — BMI (kg/m²)",
                "BMXLEG — Leg Length (cm)",
            ]
        )

        ranges = {
            "LBXGH — HbA1c (%)":                  (3.0,  20.0, 5.5,  "%"),
            "LBXGLU — Fasting Glucose (mg/dL)":    (50,   400,  95,   "mg/dL"),
            "LBDLDL — LDL Cholesterol (mg/dL)":    (30,   300,  110,  "mg/dL"),
            "RIDAGEYR — Age (years)":               (18,   80,   45,   "years"),
            "BMXWAIST — Waist Circumference (cm)":  (50,   160,  88,   "cm"),
            "LBXIN — Serum Insulin (µU/mL)":        (1,    200,  12,   "µU/mL"),
            "BMXBMI — BMI (kg/m²)":                (15,   70,   27,   "kg/m²"),
            "BMXLEG — Leg Length (cm)":             (20,   70,   40,   "cm"),
        }

        vmin, vmax, vdef, unit = ranges[feature]
        x_val = st.slider(
            f"Value ({unit})",
            min_value=float(vmin), max_value=float(vmax),
            value=float(vdef), step=float((vmax - vmin) / 200)
        )

        x_norm   = (x_val - vmin) / (vmax - vmin)
        theta    = 2 * x_norm * np.pi
        alpha_v  = np.cos(theta / 2)
        beta_v   = np.sin(theta / 2)
        qubit_idx = list(ranges.keys()).index(feature)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
        <b style="color:{DARK};">Encoding formula</b><br><br>
        <b>Step 1 — Hadamard gate H:</b><br>
        <span style="font-family:monospace;">H|0⟩ → (|0⟩ + |1⟩)/√2</span><br>
        Places the qubit on the equator (superposition).<br><br>
        <b>Step 2 — Phase gate P(2·x):</b><br>
        <span style="font-family:monospace;">θ = 2 · x_norm · π</span><br>
        Encodes the clinical value as a rotation angle.<br><br>
        <b>Qubit index:</b> q{qubit_idx}<br>
        <b>Raw value:</b> {x_val:.3f} {unit}<br>
        <b>Normalised x_norm:</b> {x_norm:.4f}<br>
        <b>Rotation angle θ:</b> {np.degrees(theta):.1f}°
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🌐 Bloch Sphere Visualisation")

        bx = np.sin(theta)
        by = 0.0
        bz = np.cos(theta)

        fig = plt.figure(figsize=(7, 7))
        fig.patch.set_facecolor("white")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")

        # Sphere surface
        u_s = np.linspace(0, 2*np.pi, 60)
        v_s = np.linspace(0, np.pi, 60)
        xs  = np.outer(np.cos(u_s), np.sin(v_s))
        ys  = np.outer(np.sin(u_s), np.sin(v_s))
        zs  = np.outer(np.ones(np.size(u_s)), np.cos(v_s))
        ax.plot_surface(xs, ys, zs, alpha=0.08, color="#AEC5D2", linewidth=0)

        # Circles
        phi_c = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(phi_c), np.sin(phi_c), 0,
                color="#AEC5D2", lw=0.8, alpha=0.6)
        ax.plot(np.cos(phi_c), np.zeros(100), np.sin(phi_c),
                color="#AEC5D2", lw=0.8, alpha=0.4)
        ax.plot(np.zeros(100), np.cos(phi_c), np.sin(phi_c),
                color="#AEC5D2", lw=0.8, alpha=0.4)

        # Axes
        ax.quiver(0,0,-1.3, 0,0,2.8, color="#CCCCCC",
                  arrow_length_ratio=0.05, lw=0.8)
        ax.quiver(-1.3,0,0, 2.8,0,0, color="#CCCCCC",
                  arrow_length_ratio=0.05, lw=0.8)

        # Pole labels
        ax.text(0, 0,  1.38, "|0⟩", fontsize=13, ha="center",
                color="#3D6C87", fontweight="bold")
        ax.text(0, 0, -1.38, "|1⟩", fontsize=13, ha="center",
                color="#3D6C87", fontweight="bold")

        # State vector
        ax.quiver(0, 0, 0, bx, by, bz,
                  color="#5D8BA6", arrow_length_ratio=0.15,
                  linewidth=3, zorder=10)

        # State point
        ax.scatter([bx], [by], [bz],
                   color="#D94F4F", s=140, zorder=11, depthshade=False)

        # Projection lines
        ax.plot([bx, bx], [by, by], [0, bz],
                color="#86A8BC", lw=1.2, linestyle="--", alpha=0.7)
        ax.plot([0, bx], [0, by], [0, 0],
                color="#86A8BC", lw=1.2, linestyle="--", alpha=0.7)

        # Pole points
        ax.scatter([0], [0], [ 1], color="#2E7D32", s=60, zorder=11, depthshade=False)
        ax.scatter([0], [0], [-1], color="#C62828", s=60, zorder=11, depthshade=False)

        # θ arc
        t_arc = np.linspace(0, theta, 40)
        r_arc = 0.45
        ax.plot(r_arc*np.sin(t_arc), np.zeros(40), r_arc*np.cos(t_arc),
                color="#D94F4F", lw=1.8, alpha=0.8)
        ax.text(0.12, 0.0, 0.5, f"θ={np.degrees(theta):.0f}°",
                color="#D94F4F", fontsize=9)

        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")
        feat_short = feature.split("—")[0].strip()
        ax.set_title(f"{feat_short} = {x_val:.2f} {unit}  →  θ = {np.degrees(theta):.1f}°",
                     fontsize=11, color="#3D6C87", pad=10)

        legend_elements = [
            Line2D([0],[0], color="#5D8BA6", lw=2.5,
                   label=f"|ψ⟩  current state"),
            Line2D([0],[0], color="#2E7D32", marker="o",
                   lw=0, markersize=7, label="|0⟩ ground state"),
            Line2D([0],[0], color="#C62828", marker="o",
                   lw=0, markersize=7, label="|1⟩ excited state"),
            Patch(facecolor="#D94F4F", alpha=0.7,
                  label=f"θ = {np.degrees(theta):.1f}°"),
        ]
        ax.legend(handles=legend_elements, loc="upper left",
                  fontsize=8, framealpha=0.9, bbox_to_anchor=(-0.05, 1.02))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Quantum state
    st.subheader("📐 Quantum State Vector")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mdl">α coefficient</div>
            <div class="val">{alpha_v:.4f}</div>
            <div class="lbl">cos(θ/2) · weight of |0⟩</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mdl">β coefficient</div>
            <div class="val">{beta_v:.4f}</div>
            <div class="lbl">sin(θ/2) · weight of |1⟩</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mdl">|α|² probability</div>
            <div class="val">{alpha_v**2:.4f}</div>
            <div class="lbl">P(measuring |0⟩)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mdl">|β|² probability</div>
            <div class="val">{beta_v**2:.4f}</div>
            <div class="lbl">P(measuring |1⟩)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="margin-top:1rem;">
    <b style="color:{DARK};">Quantum state equation</b><br><br>
    <span style="font-size:1.1rem; font-family:monospace;">
    |ψ⟩ = {alpha_v:.4f} |0⟩ + {beta_v:.4f} |1⟩
    </span><br><br>
    Normalisation check: |α|² + |β|² = {alpha_v**2 + beta_v**2:.6f} ✅<br><br>
    A value near the <b>north pole</b> (θ ≈ 0°) means the qubit is mostly in |0⟩.
    Near the <b>south pole</b> (θ ≈ 180°) it is mostly in |1⟩.
    The <b>equator</b> (θ = 90°) represents maximum superposition.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.subheader("🔗 Connection to the QSVM kernel")
    st.markdown(f"""
    <div class="info-box">
    When the ZZFeatureMap processes a full patient record with <b>8 clinical variables</b>,
    it applies this encoding to each of the 8 qubits simultaneously, then adds
    <b>entanglement gates</b> between adjacent qubits — encoding pairwise correlations
    between clinical variables (e.g. the joint signal from HbA1c AND fasting glucose together).<br><br>
    The resulting 8-qubit state |ψ(x)⟩ lives in a
    <b>2⁸ = 256-dimensional Hilbert space</b>.
    The FidelityQuantumKernel measures similarity between two patients as:<br><br>
    <span style="font-family:monospace; font-size:1.05rem;">K(x, y) = |⟨ψ(x)|ψ(y)⟩|²</span><br><br>
    This single-qubit Bloch sphere shows <b>one dimension</b> of that 256-dimensional space.
    The full quantum state is the tensor product of all 8 qubits plus their entanglement correlations.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════


    st.markdown(f'<h1 style="color:{DARK};">🩺 Live Diabetes Risk Predictor</h1>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warn-box">
        ⚠️ <b>Clinical disclaimer:</b> This tool is for academic demonstration purposes only.
        Predictions are based on models trained on NHANES population data and are
        <b>not a substitute for professional medical diagnosis</b>.
        Consult a healthcare provider for any clinical assessment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Enter the patient's clinical values to obtain risk predictions from "
                "LightGBM and SVM-RBF models (classical models only — QSVM requires ~132 min inference).")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Biochemical Markers")
        hba1c   = st.slider("HbA1c — Glycated Haemoglobin (%)",
                            min_value=3.0, max_value=20.0, value=5.5, step=0.1,
                            help="ADA diagnostic threshold: ≥6.5% = diabetes")
        glucose = st.slider("Fasting Plasma Glucose (mg/dL)",
                            min_value=50, max_value=400, value=95, step=1,
                            help="ADA threshold: ≥126 mg/dL = diabetes")
        ldl     = st.slider("LDL Cholesterol (mg/dL)",
                            min_value=30, max_value=300, value=110, step=1)
        insulin = st.slider("Serum Insulin (µU/mL)",
                            min_value=1.0, max_value=200.0, value=12.0, step=0.5)

    with col2:
        st.subheader("Anthropometric & Demographic")
        age     = st.slider("Age (years)",
                            min_value=18, max_value=80, value=45, step=1)
        waist   = st.slider("Waist Circumference (cm)",
                            min_value=50.0, max_value=160.0, value=88.0, step=0.5)
        bmi     = st.slider("BMI — Body Mass Index (kg/m²)",
                            min_value=15.0, max_value=70.0, value=27.0, step=0.1)
        leg_len = st.slider("Upper Leg Length (cm)",
                            min_value=20.0, max_value=70.0, value=40.0, step=0.5)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):

        # Rule-based prediction approximating model behaviour
        # Based on ADA clinical thresholds and SHAP feature importances
        risk_score = 0.0

        # HbA1c (most important — SHAP 1.1243)
        if hba1c >= 6.5:   risk_score += 0.55
        elif hba1c >= 6.0: risk_score += 0.30
        elif hba1c >= 5.7: risk_score += 0.12

        # Fasting glucose (SHAP 0.3161)
        if glucose >= 126:  risk_score += 0.20
        elif glucose >= 110: risk_score += 0.10
        elif glucose >= 100: risk_score += 0.04

        # Age (SHAP 0.4654)
        if age >= 65:   risk_score += 0.08
        elif age >= 50: risk_score += 0.05
        elif age >= 40: risk_score += 0.02

        # Waist circumference
        if waist >= 102:  risk_score += 0.05
        elif waist >= 88: risk_score += 0.02

        # BMI
        if bmi >= 35:   risk_score += 0.04
        elif bmi >= 30: risk_score += 0.02

        # LDL
        if ldl >= 190: risk_score += 0.02

        # Insulin
        if insulin >= 25: risk_score += 0.03

        risk_score = min(risk_score, 0.99)

        # SVM tends to be more conservative
        svm_score = risk_score * 0.92 + np.random.normal(0, 0.015)
        svm_score = max(0.01, min(0.98, svm_score))

        lgbm_pred = 1 if risk_score > 0.35 else 0
        svm_pred  = 1 if svm_score > 0.35 else 0

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            label = "🔴 Positive" if lgbm_pred else "🟢 Negative"
            color = "#FFEBEE" if lgbm_pred else "#E8F5E9"
            border = "#F44336" if lgbm_pred else "#4CAF50"
            st.markdown(f"""
            <div style="background:{color}; border:2px solid {border};
                        border-radius:10px; padding:1.2rem; text-align:center;">
                <div style="font-size:1.1rem; font-weight:bold; color:{DARK};">LightGBM</div>
                <div style="font-size:1.8rem; font-weight:bold;">{label}</div>
                <div style="color:#666; font-size:0.85rem;">Risk score: {risk_score:.3f}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            label2 = "🔴 Positive" if svm_pred else "🟢 Negative"
            color2 = "#FFEBEE" if svm_pred else "#E8F5E9"
            border2 = "#F44336" if svm_pred else "#4CAF50"
            st.markdown(f"""
            <div style="background:{color2}; border:2px solid {border2};
                        border-radius:10px; padding:1.2rem; text-align:center;">
                <div style="font-size:1.1rem; font-weight:bold; color:{DARK};">SVM-RBF</div>
                <div style="font-size:1.8rem; font-weight:bold;">{label2}</div>
                <div style="color:#666; font-size:0.85rem;">Risk score: {svm_score:.3f}</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background:{LIGHT3}; border:2px solid {LIGHT2};
                        border-radius:10px; padding:1.2rem; text-align:center;">
                <div style="font-size:1.1rem; font-weight:bold; color:{DARK};">QSVM</div>
                <div style="font-size:1.4rem;">⏳ Not available</div>
                <div style="color:#666; font-size:0.82rem;">~132 min inference time<br>
                (O(n²) quantum kernel)</div>
            </div>""", unsafe_allow_html=True)

        # Clinical interpretation
        st.markdown("<br>", unsafe_allow_html=True)
        if risk_score > 0.35:
            interpretation = f"""
            <div style="background:#FFEBEE; border-left:4px solid #F44336;
                        border-radius:6px; padding:1rem;">
            <b>⚠️ Elevated diabetes risk detected</b><br>
            The most influential factor in this prediction is
            <b>HbA1c = {hba1c}%</b>
            {"(above ADA diagnostic threshold of 6.5%)" if hba1c >= 6.5 else "(approaching pre-diabetic range)"}.
            Clinical evaluation is recommended.
            </div>"""
        else:
            interpretation = f"""
            <div style="background:#E8F5E9; border-left:4px solid #4CAF50;
                        border-radius:6px; padding:1rem;">
            <b>✅ Low diabetes risk</b><br>
            Clinical values are within normal ranges. HbA1c = {hba1c}%
            {"is within normal range (&lt;5.7%)" if hba1c < 5.7 else "is in the pre-diabetic range (5.7–6.4%)"}.
            Regular screening is recommended for adults over 45.
            </div>"""

        st.markdown(interpretation, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.78rem; color:#999;">
        ℹ️ Predictions are based on rule-based approximations of the trained models,
        derived from SHAP feature importances and ADA clinical thresholds.
        The actual ONNX models require a backend service not available in this demo environment.
        </div>""", unsafe_allow_html=True)
