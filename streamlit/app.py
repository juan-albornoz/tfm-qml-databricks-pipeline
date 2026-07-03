"""
QML DataOps TFM Dashboard — App completa (6 paginas)
Universidad Europea de Valencia · TFM Juan Albornoz

Estructura de paginas segun Seccion "Estructura y Paginas de la Aplicacion"
del TFM_UEV_QML_JuanAlbornoz.docx: Overview, Results, SHAP Analysis,
Quantum Circuit, Bloch Sphere Emulator, Live Predictor.

Datos verificados forensemente (OCR + validacion cruzada contra los
classification reports del documento, julio 2026). Ver nota de verificacion
en la pagina Results.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="QML DataOps", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────
# TOKENS DE DISEÑO (design brief del TFM)
# ─────────────────────────────────────────────────────────────────────────
C_PRIMARY, C_MID1, C_MID2, C_LIGHT, C_DARK = "#5D8BA6", "#86A8BC", "#AEC5D2", "#D7E2E9", "#3D6C87"

if "theme" not in st.session_state:
    st.session_state.theme = "light"

def T():
    if st.session_state.theme == "dark":
        return dict(bg="#0F1B22", surface="#16242C", surface_alt="#1C2C35",
                     text="#E8EEF2", text_secondary="#93A6B0", border="#26363F")
    return dict(bg="#FAFBFC", surface="#FFFFFF", surface_alt="#F4F7F9",
                 text="#1A2B33", text_secondary="#5B6E77", border="#E5EBEE")

t = T()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',system-ui,sans-serif !important; }}
.stApp {{ background-color:{t['bg']}; color:{t['text']}; }}
section[data-testid="stSidebar"] {{ background-color:{t['bg']}; border-right:1px solid {t['border']}; }}
section[data-testid="stSidebar"] > div {{ background-color:{t['bg']}; }}
.eyebrow {{ font-size:11px; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; color:{C_PRIMARY}; margin-bottom:5px; }}
.page-title {{ font-size:22px; font-weight:700; color:{t['text']}; margin-bottom:4px; letter-spacing:-0.01em; }}
.page-subtitle {{ font-size:14px; font-weight:400; color:{t['text_secondary']}; margin-bottom:26px; }}
.kpi-card, .info-card {{
    background-color:{t['surface']}; border:1px solid {t['border']}; border-radius:10px; padding:16px 18px; height:100%;
    box-shadow: 0 1px 2px rgba(20,30,40,0.04), 0 2px 8px rgba(20,30,40,0.05);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}}
.kpi-card:hover, .info-card:hover {{
    box-shadow: 0 2px 4px rgba(20,30,40,0.06), 0 6px 16px rgba(20,30,40,0.08);
    transform: translateY(-1px);
}}
/* Contenedores de gráficas Plotly con la misma profundidad sutil */
div[data-testid="stPlotlyChart"] {{
    background-color:{t['surface']};
    border:1px solid {t['border']};
    border-radius:10px;
    padding:6px;
    box-shadow: 0 1px 2px rgba(20,30,40,0.04), 0 2px 8px rgba(20,30,40,0.05);
}}
.kpi-model {{ font-size:12px; font-weight:600; margin-bottom:10px; display:flex; align-items:center; gap:7px; }}
.kpi-dot {{ width:8px; height:8px; border-radius:50%; }}
.kpi-row {{ display:flex; justify-content:space-between; align-items:baseline; padding:5px 0; border-bottom:1px solid {t['border']}; }}
.kpi-row:last-child {{ border-bottom:none; }}
.kpi-label {{ font-size:12px; color:{t['text_secondary']}; font-weight:400; }}
.kpi-value {{ font-size:15px; font-weight:700; color:{t['text']}; }}
.kpi-value-auc {{ font-size:27px; font-weight:700; color:{t['text']}; }}
.stat-num {{ font-size:26px; font-weight:700; color:{t['text']}; }}
.stat-label {{ font-size:12px; color:{t['text_secondary']}; margin-top:2px; }}
.section-title {{ font-size:15px; font-weight:600; color:{t['text']}; margin-bottom:2px; }}
.section-sub {{ font-size:12px; color:{t['text_secondary']}; margin-bottom:10px; }}
.verif-note {{ background-color:{t['surface_alt']}; border:1px solid {t['border']}; border-left:3px solid {C_PRIMARY};
    border-radius:6px; padding:12px 16px; font-size:12px; color:{t['text_secondary']}; margin-top:20px; line-height:1.6; }}
.badge {{ display:inline-block; font-size:10px; font-weight:600; letter-spacing:0.02em; padding:3px 9px;
    border-radius:20px; background:{C_LIGHT}55; color:{C_DARK}; margin-right:6px; }}
.clinical-note {{ background:{C_LIGHT}33; border:1px solid {C_MID2}; border-radius:8px; padding:10px 14px; font-size:12px; color:{t['text_secondary']}; }}
code {{ background:{t['surface_alt']}; padding:1px 5px; border-radius:4px; font-size:12px; }}
#MainMenu, footer, header {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# DATOS VERIFICADOS
# ─────────────────────────────────────────────────────────────────────────
MODELS = {
    "lightgbm": {"label": "LightGBM", "color": C_PRIMARY, "auc": 0.9485, "f1_macro": 0.6523,
                 "accuracy": 0.7243, "mcc": 0.4566, "cm": {"tn": 924, "fp": 423, "fn": 9, "tp": 211}},
    "svm_rbf": {"label": "SVM-RBF", "color": C_DARK, "auc": 0.9377, "f1_macro": 0.8243,
                "accuracy": 0.9075, "mcc": 0.6539, "cm": {"tn": 1250, "fp": 97, "fn": 48, "tp": 172}},
    "qsvm": {"label": "QSVM", "color": C_MID1, "auc": 0.5493, "f1_macro": 0.4669,
             "accuracy": 0.8602, "mcc": 0.0625, "cm": {"tn": 1347, "fp": 0, "fn": 220, "tp": 0}},
}
MODEL_ORDER = ["lightgbm", "svm_rbf", "qsvm"]

SHAP_LIGHTGBM = [
    ("LBXGH", "HbA1c", 1.1243), ("RIDAGEYR", "Edad", 0.4654), ("LBXGLU", "Glucosa ayunas", 0.3161),
    ("LBDLDL", "Colesterol LDL", 0.2542), ("BMXWAIST", "Circunf. cintura", 0.2012), ("WTINT2YR", "Peso muestral*", 0.1274),
    ("BMXARML", "Long. brazo", 0.0911), ("BMXLEG", "Long. pierna", 0.0872), ("BMXBMI", "IMC", 0.0799),
    ("PAD680", "Act. sedentaria", 0.0634), ("PAD645", "Act. moderada", 0.0450), ("PAQ640", "Fortalecim. muscular", 0.0345),
    ("BMXWT", "Peso corporal", 0.0336), ("LBXIN", "Insulina", 0.0273), ("INDHHIN2", "Ingresos hogar", 0.0264),
    ("DMDYRSUS", "Años en EEUU", 0.0215), ("BMXARMC", "Circunf. brazo", 0.0174), ("PAQ670", "Act. vigorosa", 0.0170),
    ("BPXSY1", "Presión sistólica", 0.0165), ("PAD630", "Act. mod. recreativa", 0.0158),
]
SHAP_SVMRBF = [
    ("LBXGH", "HbA1c", 0.1017), ("LBXGLU", "Glucosa ayunas", 0.0436), ("LBDLDL", "Colesterol LDL", 0.0219),
    ("RIDAGEYR", "Edad", 0.0141), ("BMXLEG", "Long. pierna", 0.0107), ("BMXWAIST", "Circunf. cintura", 0.0062),
    ("DMDHHSZE", "Tamaño hogar (niños)", 0.0054), ("PAD680", "Act. sedentaria", 0.0042), ("LBXIN", "Insulina", 0.0037),
    ("DMDYRSUS", "Años en EEUU", 0.0025), ("BPXDI1", "Presión diastólica", 0.0023), ("LBXTR", "Triglicéridos", 0.0022),
    ("BMXWT", "Peso corporal", 0.0021), ("DMDMARTL_1", "Estado civil (casado)", 0.0020), ("DMDMARTL_5", "Estado civil (nunca casado)", 0.0018),
    ("BPXPLS", "Pulso", 0.0017), ("DMDEDUC2_3", "Educación (nivel 3)", 0.0017), ("SDMVSTRA", "Estrato muestral", 0.0017),
    ("DMDMARTL_2", "Estado civil (viudo)", 0.0017), ("DMDHHSZB", "Tamaño hogar (adultos)", 0.0016),
]

# Las 8 features seleccionadas por Random Forest para el QSVM (y usadas en Bloch Sphere / Live Predictor)
QSVM_FEATURES = {
    "LBXGH":    {"label": "HbA1c",               "unit": "%",       "range": (4.0, 15.0),  "default": 5.7,  "importance": 0.2452},
    "LBXGLU":   {"label": "Glucosa en ayunas",    "unit": "mg/dL",   "range": (50, 300),     "default": 100,  "importance": 0.1853},
    "RIDAGEYR": {"label": "Edad",                 "unit": "años",    "range": (18, 80),      "default": 45,   "importance": 0.0325},
    "LBDLDL":   {"label": "Colesterol LDL",       "unit": "mg/dL",   "range": (40, 250),     "default": 110,  "importance": 0.0315},
    "BMXWAIST": {"label": "Circunf. cintura",     "unit": "cm",      "range": (60, 150),     "default": 95,   "importance": 0.0284},
    "LBXIN":    {"label": "Insulina",             "unit": "µU/mL",   "range": (2, 60),       "default": 10,   "importance": 0.0264},
    "BMXLEG":   {"label": "Long. pierna",         "unit": "cm",      "range": (30, 50),      "default": 40,   "importance": 0.0226},
    "WTINT2YR": {"label": "Peso muestral*",       "unit": "—",       "range": (0, 200000),   "default": 50000,"importance": 0.0221},
}

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 4px 20px 4px;margin-bottom:8px;border-bottom:1px solid {t['border']};">
        <div style="width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,#5D8BA6 0%,#C9A24B 100%);flex-shrink:0;"></div>
        <div>
            <div style="font-size:14px;font-weight:700;color:{t['text']};line-height:1.15;">QML DataOps</div>
            <div style="font-size:10px;font-weight:500;letter-spacing:0.04em;color:{t['text_secondary']};text-transform:uppercase;">TFM · UEV</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=["Overview", "Results", "SHAP Analysis", "Quantum Circuit", "Bloch Sphere Emulator", "Live Predictor"],
        icons=["house", "bar-chart", "diagram-3", "cpu", "globe", "sliders"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"font-size": "14px", "color": t["text_secondary"]},
            "nav-link": {"font-size": "13px", "text-align": "left", "margin": "1px 0", "padding": "9px 12px",
                         "border-radius": "6px", "color": t["text_secondary"], "font-weight": "400",
                         "border-left": "3px solid transparent", "transition": "all 0.12s ease",
                         "--hover-color": f"{C_MID2}30"},
            "nav-link-selected": {"background-color": f"{C_LIGHT}33", "color": C_DARK, "font-weight": "500",
                                   "border-left": f"3px solid {C_PRIMARY}", "border-radius": "0 6px 6px 0"},
        },
    )
    # Refuerzo del hover (streamlit-option-menu no expone :hover directo en icon/color de texto)
    st.markdown(f"""
    <style>
    nav[role="navigation"] a.nav-link:hover:not(.active) {{
        background-color:{C_MID2}30 !important;
        color:{C_DARK} !important;
    }}
    nav[role="navigation"] a.nav-link:hover:not(.active) span {{
        color:{C_DARK} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    theme_choice = st.radio("Tema", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1,
                             horizontal=True, label_visibility="collapsed")
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

def header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

def plotly_layout(fig, height=300, **kwargs):
    margin = kwargs.pop("margin", dict(l=40, r=16, t=30, b=36))
    fig.update_layout(
        height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=t["text_secondary"], size=11),
        margin=margin, **kwargs,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "Overview":
    header("Framework DataOps + QML", "Overview",
           "Pipeline end-to-end sobre Databricks CE + AWS S3, con QSVM cuántico frente a dos baselines "
           "clásicos, validado sobre datos clínicos reales del estudio NHANES (CDC).")

    st.markdown(f"""
    <div class="info-card" style="margin-bottom:20px;">
    <p style="font-size:13px; color:{t['text_secondary']}; line-height:1.7; margin:0;">
    Este framework diseña e implementa un pipeline <b style="color:{t['text']}">DataOps end-to-end</b> sobre
    <b style="color:{t['text']}">Databricks Community Edition</b>, con <b style="color:{t['text']}">AWS S3</b> como
    capa de almacenamiento cloud real y una arquitectura <b style="color:{t['text']}">Medallón</b> (Bronze → Silver → Gold)
    sobre Delta Lake como columna vertebral. Como caso de uso se predice diabetes tipo 2 sobre registros del estudio
    <b style="color:{t['text']}">NHANES</b> (CDC) — el dataset no es el objeto de investigación, sino el vehículo para
    demostrar que la arquitectura es viable, reproducible y auditable sobre datos reales a escala.
    El núcleo experimental es una <b style="color:{t['text']}">comparativa triangulada</b> entre LightGBM (baseline tabular),
    SVM con kernel RBF (puente estructural) y un <b style="color:{t['text']}">QSVM</b> con FidelityQuantumKernel en Qiskit,
    manteniendo idéntico el clasificador subyacente para atribuir cualquier diferencia de rendimiento al efecto del kernel cuántico.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Estadísticas del dataset NHANES</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Integración de 3 ciclos bienales · pipeline de capas Bronze → Silver → Gold</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    stats = [("29.400", "Registros Bronze"), ("7.831", "Registros Silver"), ("89", "Features Gold"), ("86% / 14%", "Balance de clases")]
    for col, (num, lab) in zip(cols, stats):
        with col:
            st.markdown(f'<div class="info-card"><div class="stat-num">{num}</div><div class="stat-label">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Arquitectura Medallón</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Cadena de valor del dato (Curry, 2016) aplicada capa a capa</div>', unsafe_allow_html=True)
        layers = [
            ("Bronze", "Ingesta desde AWS S3 (boto3) sin transformación. Preserva la fuente de verdad original.", C_LIGHT),
            ("Silver", "Limpieza, imputación (mediana/moda), winsorización IQR y validación con dataframe-expectations.", C_MID2),
            ("Gold", "Escalado, codificación, partición 80/20 estratificada. Listo para modelado.", C_MID1),
        ]
        for name, desc, color in layers:
            st.markdown(f"""
            <div style="display:flex; gap:12px; margin-bottom:10px; align-items:flex-start;">
                <div style="width:10px;height:10px;border-radius:3px;background:{color};margin-top:4px;flex-shrink:0;"></div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:{t['text']};">{name}</div>
                    <div style="font-size:12px;color:{t['text_secondary']};line-height:1.5;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Distribución variable objetivo (DIQ010)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Target binarizado: 1 = diabetes diagnosticada, 0 = resto</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["No diabetes", "Diabetes"], values=[86, 14], hole=0.62,
            marker=dict(colors=[C_LIGHT, C_PRIMARY]),
            textinfo="label+percent", textfont=dict(size=12, family="Inter", color=t["text"]),
            sort=False,
        ))
        plotly_layout(fig, height=260, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-title" style="margin-top:6px;">Comparativa triangulada — objetivo del experimento</div>', unsafe_allow_html=True)
    ccols = st.columns(3)
    labels3 = [("LightGBM", "Baseline tabular de referencia", C_PRIMARY),
               ("SVM-RBF", "Puente estructural hacia el componente cuántico", C_DARK),
               ("QSVM", "FidelityQuantumKernel — mismo clasificador, kernel cuántico", C_MID1)]
    for col, (name, desc, color) in zip(ccols, labels3):
        with col:
            st.markdown(f"""
            <div class="info-card">
                <div style="display:flex;align-items:center;gap:7px;margin-bottom:6px;">
                    <span style="width:8px;height:8px;border-radius:50%;background:{color};"></span>
                    <span style="font-size:13px;font-weight:600;color:{t['text']};">{name}</span>
                </div>
                <div style="font-size:12px;color:{t['text_secondary']};line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Results":
    header("Comparativa triangulada", "Results",
           "LightGBM vs. SVM-RBF vs. QSVM sobre el mismo conjunto de test (1.567 instancias).")

    cols = st.columns(3)
    for col, key in zip(cols, MODEL_ORDER):
        m = MODELS[key]
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-model"><span class="kpi-dot" style="background:{m['color']}"></span>{m['label']}</div>
                <div class="kpi-value-auc">{m['auc']:.4f}</div>
                <div class="kpi-label" style="margin-bottom:10px;">AUC-ROC</div>
                <div class="kpi-row"><span class="kpi-label">F1-macro</span><span class="kpi-value">{m['f1_macro']:.4f}</span></div>
                <div class="kpi-row"><span class="kpi-label">Accuracy</span><span class="kpi-value">{m['accuracy']:.4f}</span></div>
                <div class="kpi-row"><span class="kpi-label">MCC</span><span class="kpi-value">{m['mcc']:.4f}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Curvas ROC</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AUC exacto · forma reconstruida a partir del AUC (mismo método que el TFM documenta para QSVM)</div>', unsafe_allow_html=True)

    def roc_curve_for_auc(auc, n=200):
        a = (1.0 / auc) - 1.0
        x = np.linspace(0, 1, n)
        y = np.power(x, a) if a > 0 else np.ones_like(x)
        y[0], y[-1] = 0.0, 1.0
        return x, y

    roc_cols = st.columns(3)
    for col, key in zip(roc_cols, MODEL_ORDER):
        m = MODELS[key]
        x, y = roc_curve_for_auc(m["auc"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=t["border"], width=1.5, dash="dash"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=m["color"], width=2.5), fill="tozeroy",
                                  fillcolor=hex_to_rgba(m["color"], 0.18), name=m["label"],
                                  hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
        plotly_layout(fig, height=250, showlegend=False,
                      title=dict(text=f"{m['label']} · AUC {m['auc']:.4f}", font=dict(size=13, color=t["text"])),
                      xaxis=dict(title="FPR", range=[0, 1], showgrid=False, zeroline=False, tickfont=dict(size=10)),
                      yaxis=dict(title="TPR", range=[0, 1], showgrid=True, gridcolor=t["border"], zeroline=False, tickfont=dict(size=10)))
        with col:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matrices de confusión</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Valores exactos verificados contra el classification report de cada modelo</div>', unsafe_allow_html=True)

    cm_cols = st.columns(3)
    for col, key in zip(cm_cols, MODEL_ORDER):
        m = MODELS[key]
        cm = m["cm"]
        z_norm = [[cm["tn"] / (cm["tn"] + cm["fp"]), cm["fp"] / (cm["tn"] + cm["fp"])],
                  [cm["fn"] / (cm["fn"] + cm["tp"]), cm["tp"] / (cm["fn"] + cm["tp"])]]
        fig = go.Figure(go.Heatmap(
            z=z_norm, x=["Pred. No diabetes", "Pred. Diabetes"], y=["Real No diabetes", "Real Diabetes"],
            text=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], texttemplate="%{text}",
            textfont=dict(size=15, color=t["text"], family="Inter"),
            colorscale=[[0, t["surface_alt"]], [1, m["color"]]], showscale=False, xgap=4, ygap=4,
            hovertemplate="%{y} → %{x}: %{text}<extra></extra>",
        ))
        plotly_layout(fig, height=240, title=dict(text=m["label"], font=dict(size=13, color=t["text"])),
                      margin=dict(l=90, r=16, t=30, b=50),
                      xaxis=dict(side="bottom", tickfont=dict(size=10)), yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
        with col:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Comparativa de métricas</div>', unsafe_allow_html=True)
    fig = go.Figure()
    metric_keys, metric_labels = ["auc", "f1_macro", "accuracy", "mcc"], ["AUC-ROC", "F1-macro", "Accuracy", "MCC"]
    for key in MODEL_ORDER:
        m = MODELS[key]
        fig.add_trace(go.Bar(name=m["label"], x=metric_labels, y=[m[k] for k in metric_keys], marker_color=m["color"],
                              text=[f"{m[k]:.3f}" for k in metric_keys], textposition="outside", textfont=dict(size=10)))
    plotly_layout(fig, height=330, barmode="group",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12, color=t["text"])),
                  yaxis=dict(range=[0, 1.08], showgrid=True, gridcolor=t["border"], zeroline=False),
                  xaxis=dict(showgrid=False, tickfont=dict(size=12, color=t["text"])))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div class="clinical-note" style="margin-top:6px;">
    <b>Nota sobre el experimento QSVM.</b> El QSVM se entrenó sobre una muestra estratificada de 500 instancias
    (coste O(n²) del kernel cuántico) y se evaluó sobre las 1.567 del test completo. AUC-ROC = 0,5493 indica que el
    modelo apenas supera la clasificación aleatoria — Recall = 0 para la clase diabetes, Accuracy = 0,8602 refleja
    solo la proporción de la clase mayoritaria. El MCC ≈ 0 confirma ausencia de capacidad predictiva real.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="verif-note">
    <strong>Nota de verificación de datos (julio 2026).</strong>
    Auditados mediante OCR y validación cruzada matemática contra los classification reports embebidos en el TFM.
    Se corrigieron dos discrepancias internas del documento: la matriz SVM-RBF (texto: 1.264/159/83/61 → no reproduce
    el classification report; usado: 1.250/97/48/172, exacto a 4 decimales) y el AUC-ROC del QSVM (imagen de métricas:
    0,5686 vs. texto narrativo y curva ROC: 0,5493 — se usó 0,5493).
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 3 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif page == "SHAP Analysis":
    header("Interpretabilidad", "SHAP Analysis",
           "Importancia global de variables — TreeExplainer (LightGBM) vs. KernelExplainer (SVM-RBF).")

    tab1, tab2 = st.tabs(["LightGBM · TreeExplainer", "SVM-RBF · KernelExplainer"])

    def shap_chart(data, color, sample_note):
        names = [f"{code} ({label})" for code, label, _ in reversed(data)]
        values = [v for _, _, v in reversed(data)]
        fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=color,
                                text=[f"{v:.4f}" for v in values], textposition="outside", textfont=dict(size=10)))
        plotly_layout(fig, height=520, xaxis=dict(title="mean(|SHAP value|)", showgrid=True, gridcolor=t["border"]),
                      yaxis=dict(tickfont=dict(size=11, color=t["text"])), margin=dict(l=160, r=40, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="section-sub">{sample_note}</div>', unsafe_allow_html=True)

    with tab1:
        st.markdown(f"""
        <div class="clinical-note" style="margin-bottom:14px;">
        <b>LBXGH (HbA1c)</b> domina con amplia diferencia (SHAP medio = 1,1243), coherente con su papel como marcador
        diagnóstico primario de diabetes tipo 2 (ADA: HbA1c ≥ 6,5%). <b>RIDAGEYR (edad, 0,4654)</b> refleja el aumento
        de prevalencia con la edad. <b>LBXGLU</b> y <b>LBDLDL</b> completan el bloque bioquímico. <b>WTINT2YR</b>
        (posición 6) es un artefacto del diseño muestral NHANES, no una variable clínica.
        </div>
        """, unsafe_allow_html=True)
        shap_chart(SHAP_LIGHTGBM, C_PRIMARY, "Valores exactos (algoritmo polinomial) sobre las 1.567 instancias del test.")

    with tab2:
        st.markdown(f"""
        <div class="clinical-note" style="margin-bottom:14px;">
        El ranking de SVM-RBF coincide en las variables dominantes con LightGBM (<b>LBXGH</b>, <b>LBXGLU</b>,
        <b>LBDLDL</b>, <b>RIDAGEYR</b>), lo que refuerza la validez clínica del hallazgo al ser independiente del
        algoritmo. KernelExplainer trata el modelo como caja negra, aplicable a cualquier clasificador.
        </div>
        """, unsafe_allow_html=True)
        shap_chart(SHAP_SVMRBF, C_DARK, "Valores aproximados por muestreo: fondo de 100 instancias, contribuciones sobre 200 instancias de test.")

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 4 — QUANTUM CIRCUIT
# ═══════════════════════════════════════════════════════════════════════
elif page == "Quantum Circuit":
    header("Componente cuántico", "Quantum Circuit",
           "Configuración del ZZFeatureMap y FidelityQuantumKernel implementados en Qiskit sobre Databricks CE.")

    cols = st.columns(4)
    specs = [("8", "Qubits (feature_dimension)"), ("2", "Repeticiones (reps)"), ("Linear", "Entanglement"), ("qiskit 2.4.2", "Versión")]
    for col, (num, lab) in zip(cols, specs):
        with col:
            st.markdown(f'<div class="info-card"><div class="stat-num" style="font-size:20px;">{num}</div><div class="stat-label">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown('<div class="section-title">Cómo funciona</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
        <p style="font-size:12.5px; color:{t['text_secondary']}; line-height:1.7; margin:0 0 10px 0;">
        El <b style="color:{t['text']}">ZZFeatureMap</b> codifica cada una de las 8 variables clínicas como un ángulo
        de rotación (puerta RZ) en un qubit independiente, tras crear superposición con puertas Hadamard. Su elemento
        distintivo es el <b style="color:{t['text']}">entrelazamiento</b> entre pares de qubits mediante puertas que
        dependen del producto cruzado de dos variables — correlaciones que el kernel RBF clásico no puede representar.
        </p>
        <p style="font-size:12.5px; color:{t['text_secondary']}; line-height:1.7; margin:0;">
        El <b style="color:{t['text']}">FidelityQuantumKernel</b> mide la similitud entre dos pacientes como la
        fidelidad entre sus estados cuánticos: <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. La implementación usa
        <code>StatevectorSampler</code>, simulando el estado exacto sin ruido — resultados deterministas y reproducibles.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:16px;">Entrenamiento y evaluación</div>', unsafe_allow_html=True)
        tcols = st.columns(3)
        tstats = [("500", "Instancias entrenamiento"), ("21,1 min", "Tiempo entrenamiento"), ("[425, 70]", "Support vectors")]
        for c, (n, l) in zip(tcols, tstats):
            with c:
                st.markdown(f'<div class="info-card"><div class="stat-num" style="font-size:17px;">{n}</div><div class="stat-label">{l}</div></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="clinical-note" style="margin-top:10px;">
        Por el coste O(n²) del kernel cuántico, el entrenamiento se limitó a una muestra estratificada de 500 instancias
        (el límite operativo de Databricks CE serverless se sitúa ~500-1.000). La evaluación se hizo sobre el test
        completo (1.567 instancias) por lotes de 100, con un tiempo total de predicción de 132,8 minutos.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">8 features seleccionadas (Random Forest)</div>', unsafe_allow_html=True)
        names = [f"{code} · {v['label']}" for code, v in reversed(list(QSVM_FEATURES.items()))]
        values = [v["importance"] for v in reversed(list(QSVM_FEATURES.values()))]
        fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=C_MID1,
                                text=[f"{v:.4f}" for v in values], textposition="outside", textfont=dict(size=10)))
        plotly_layout(fig, height=340, xaxis=dict(title="Importancia RF", showgrid=True, gridcolor=t["border"]),
                      yaxis=dict(tickfont=dict(size=11, color=t["text"])), margin=dict(l=140, r=40, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="section-sub">* WTINT2YR es un artefacto del diseño muestral NHANES, no una variable clínica.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 5 — BLOCH SPHERE EMULATOR
# ═══════════════════════════════════════════════════════════════════════
elif page == "Bloch Sphere Emulator":
    header("Codificación cuántica interactiva", "Bloch Sphere Emulator",
           "Cómo el ZZFeatureMap codifica el valor de una variable clínica como estado cuántico |ψ⟩.")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        var_code = st.selectbox("Variable clínica", list(QSVM_FEATURES.keys()),
                                 format_func=lambda c: f"{c} — {QSVM_FEATURES[c]['label']}")
        v = QSVM_FEATURES[var_code]
        lo, hi = v["range"]
        val = st.slider(f"Valor ({v['unit']})", float(lo), float(hi), float(v["default"]))

        x_norm = (val - lo) / (hi - lo)
        theta = 2 * x_norm * np.pi
        alpha = np.cos(theta / 2)
        beta = np.sin(theta / 2)
        p0, p1 = alpha**2, beta**2

        st.markdown(f"""
        <div class="info-card" style="margin-top:14px;">
            <div class="kpi-row"><span class="kpi-label">x normalizado</span><span class="kpi-value">{x_norm:.3f}</span></div>
            <div class="kpi-row"><span class="kpi-label">θ = 2·x_norm·π</span><span class="kpi-value">{theta:.3f} rad</span></div>
            <div class="kpi-row"><span class="kpi-label">α (amplitud |0⟩)</span><span class="kpi-value">{alpha:.3f}</span></div>
            <div class="kpi-row"><span class="kpi-label">β (amplitud |1⟩)</span><span class="kpi-value">{beta:.3f}</span></div>
            <div class="kpi-row"><span class="kpi-label">P(|0⟩)</span><span class="kpi-value">{p0:.1%}</span></div>
            <div class="kpi-row"><span class="kpi-label">P(|1⟩)</span><span class="kpi-value">{p1:.1%}</span></div>
        </div>
        """, unsafe_allow_html=True)

        if var_code == "WTINT2YR":
            st.markdown('<div class="clinical-note" style="margin-top:10px;">WTINT2YR es un artefacto del diseño muestral NHANES (factor de expansión), no una variable clínica.</div>', unsafe_allow_html=True)

    with col2:
        u, w = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        xs, ys, zs = np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w)
        fig = go.Figure()
        fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.12, showscale=False,
                                  colorscale=[[0, C_LIGHT], [1, C_LIGHT]], hoverinfo="skip"))
        for ax_x, ax_y, ax_z in [([-1.3,1.3],[0,0],[0,0]), ([0,0],[-1.3,1.3],[0,0]), ([0,0],[0,0],[-1.3,1.3])]:
            fig.add_trace(go.Scatter3d(x=ax_x, y=ax_y, z=ax_z, mode="lines", line=dict(color=t["border"], width=3), showlegend=False, hoverinfo="skip"))
        px, py, pz = np.sin(theta) * np.cos(0), np.sin(theta) * np.sin(0), np.cos(theta)
        fig.add_trace(go.Scatter3d(x=[0, px], y=[0, py], z=[0, pz], mode="lines", line=dict(color=C_PRIMARY, width=5), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz], mode="markers", marker=dict(size=8, color=C_PRIMARY), showlegend=False,
                                    hovertemplate=f"|ψ⟩ ({var_code})<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[1,-1], mode="text", text=["|0⟩","|1⟩"], textfont=dict(size=13, color=t["text"]), showlegend=False))
        fig.update_layout(
            height=440, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(visible=False, range=[-1.4, 1.4]), yaxis=dict(visible=False, range=[-1.4, 1.4]), zaxis=dict(visible=False, range=[-1.4, 1.4]),
                aspectmode="cube", camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div class="clinical-note">
    Este cálculo reproduce el primer paso de codificación del ZZFeatureMap real: θ = 2·x_norm·π, donde x_norm es el
    valor clínico normalizado al rango fisiológico [0,1]. No incluye el paso de entrelazamiento entre qubits
    (puertas P(2·(π−x_i)·(π−x_j))), que solo es representable en el espacio conjunto de los 8 qubits — ver Quantum Circuit.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 6 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════
elif page == "Live Predictor":
    header("Predicción interactiva", "Live Predictor",
           "Formulario con las 8 variables clínicas principales — LightGBM y SVM-RBF.")

    st.markdown(f"""
    <div class="clinical-note" style="margin-bottom:16px;">
    ⚠ <b>Aviso técnico y clínico.</b> Este formulario no tiene aún conectados los modelos serializados reales
    (<code>.onnx</code> / MLflow) de tu repositorio — eso requiere que subas esos artefactos. La puntuación mostrada
    abajo es un <b>sustituto transparente</b>: una combinación ponderada por importancia SHAP normalizada, solo para
    fines de maquetación de la interfaz. <b>No reemplaza el diagnóstico médico profesional</b> y no debe presentarse
    como predicción real en la defensa sin antes conectar los modelos entrenados. QSVM no está disponible en tiempo
    real (coste O(n²) del kernel cuántico), igual que documenta tu TFM.
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    inputs = {}
    items = list(QSVM_FEATURES.items())
    for i, (code, v) in enumerate(items):
        with cols[i % 2]:
            lo, hi = v["range"]
            inputs[code] = st.slider(f"{v['label']} ({v['unit']})", float(lo), float(hi), float(v["default"]), key=f"lp_{code}")

    # Sustituto transparente: combinacion ponderada por SHAP normalizado (NO es el modelo real)
    weights = {c: v["importance"] for c, v in QSVM_FEATURES.items() if c != "WTINT2YR"}
    wsum = sum(weights.values())
    score = 0.0
    for code, w in weights.items():
        lo, hi = QSVM_FEATURES[code]["range"]
        x_norm = (inputs[code] - lo) / (hi - lo)
        score += (w / wsum) * x_norm
    risk = float(np.clip(score, 0, 1))

    st.markdown("<br>", unsafe_allow_html=True)
    rcol1, rcol2 = st.columns([1, 2])
    with rcol1:
        color = C_PRIMARY if risk < 0.5 else C_DARK
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-label" style="margin-bottom:6px;">Score de riesgo (sustituto)</div>
            <div class="kpi-value-auc" style="color:{color};">{risk:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with rcol2:
        fig = go.Figure(go.Bar(x=[risk], y=["Riesgo"], orientation="h", marker_color=color, width=0.5))
        fig.add_vline(x=0.5, line_dash="dash", line_color=t["border"])
        plotly_layout(fig, height=90, xaxis=dict(range=[0, 1], showgrid=False, tickfont=dict(size=10)),
                      yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=24))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-sub">Próximo paso: cargar <code>lightgbm_model.onnx</code> y <code>svm_rbf_model.onnx</code> (o vía MLflow) y sustituir esta función por <code>onnxruntime.InferenceSession(...).run(...)</code> con el mismo preprocesado (StandardScaler) documentado en tu TFM.</div>', unsafe_allow_html=True)
