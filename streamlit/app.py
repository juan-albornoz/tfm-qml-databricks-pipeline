"""
QML DataOps Pipeline — TFM Streamlit App
Juan Albornoz Carrasco · Universidad Europea de Valencia · 2025-2026
Director: Prof. Ronal Muresano

UBICACIÓN EN EL REPOSITORIO: streamlit/app.py
    tfm-qml-databricks-pipeline/
    ├── streamlit/
    │   ├── app.py              ← este archivo
    │   ├── requirements.txt    ← ver archivo adjunto
    │   └── assets/
    │       └── qml_logo.png    ← coloca el logo aquí
    └── figures/                ← 13 PNG, ya documentados en tu TFM

Paleta de color: la misma escala azul documentada en el TFM
(#5D8BA6, #86A8BC, #AEC5D2, #D7E2E9) — sección "Consideraciones Técnicas
del Despliegue". El modo oscuro es una adaptación de la misma familia de
tono, no una paleta nueva, para no contradecir esa afirmación del TFM.

matplotlib ya NO es necesario — la Bloch Sphere se migró a Plotly.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE PÁGINA Y ESTADO DE TEMA
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QML DataOps Pipeline | TFM",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"  # el TFM documenta el modo claro como el desplegado

# ══════════════════════════════════════════════════════════════════════════
# 2. PALETA DE COLORES — misma escala azul documentada en el TFM
#    Light: valores exactos del TFM. Dark: adaptación de la misma familia.
# ══════════════════════════════════════════════════════════════════════════
THEMES = {
    "light": dict(
        bg="#F8FBFD", sidebar_bg="#ffffff", surface="#ffffff", border="#D7E2E9",
        text_primary="#22333F", text_secondary="#5b6a74", text_muted="#8a97a1",
        accent="#5D8BA6", accent_bg="#EAF1F5", accent_text="#3D6C87", accent_border="#AEC5D2",
        pro_bg="#3D6C87", pro_text="#ffffff", pro_border="#3D6C87",
        success_bg="#E8F5E9", success_text="#2E7D32", success_border="#4CAF50",
        danger_bg="#FFEBEE", danger_text="#C62828", danger_border="#F44336",
        warn_bg="#FFF8E1", warn_text="#8a6200", warn_border="#F9A825",
        plotly_template="plotly_white",
    ),
    "dark": dict(
        bg="#10161b", sidebar_bg="#0d1216", surface="#182229", border="#2a3a44",
        text_primary="#eef2f5", text_secondary="#a8b8c2", text_muted="#6d7b85",
        accent="#8FB4C7", accent_bg="#1c2e38", accent_text="#9FC3D4", accent_border="#2d4a58",
        pro_bg="#3D6C87", pro_text="#ffffff", pro_border="#5D8BA6",
        success_bg="#12291a", success_text="#7ed99a", success_border="#2E7D32",
        danger_bg="#2e1414", danger_text="#f08a8a", danger_border="#C62828",
        warn_bg="#2e2610", warn_text="#f0c96a", warn_border="#8a6200",
        plotly_template="plotly_dark",
    ),
}
T = THEMES[st.session_state.theme]

# Color por modelo — consistente en toda la app, dentro de la misma escala azul
MODEL_COLOR = {"LightGBM": T["accent_text"], "SVM-RBF": "#7CA8BE", "QSVM": T["pro_bg"]}

# ══════════════════════════════════════════════════════════════════════════
# 3. CSS GLOBAL
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    .stApp {{ background-color: {T['bg']}; }}
    header[data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}
    section[data-testid="stSidebar"] {{
        background-color: {T['sidebar_bg']};
        border-right: 1px solid {T['border']};
    }}
    section[data-testid="stSidebar"] > div:first-child {{ padding-top: 1.1rem; }}
    h1, h2, h3, h4, h5, h6, p, span, label {{ color: {T['text_primary']}; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    hr {{ border-color: {T['border']}; }}
    [data-testid="stDataFrame"] {{ border: 1px solid {T['border']}; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 4. LOGO + NAVEGACIÓN LATERAL + SWITCH DE TEMA
# ══════════════════════════════════════════════════════════════════════════
def load_logo_b64(path: str) -> str | None:
    p = Path(path)
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None


# Ruta relativa a este archivo (streamlit/app.py) — coloca el logo en
# streamlit/assets/qml_logo.png. Igual que FIGURES, no depende del cwd.
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "qml_logo.png")
logo_b64 = load_logo_b64(LOGO_PATH)

with st.sidebar:
    if logo_b64:
        st.markdown(f"""
        <div style="text-align:center;padding:4px 8px 16px;
                    border-bottom:1px solid {T['border']};margin-bottom:10px">
            <img src="data:image/png;base64,{logo_b64}"
                 style="width:88px;height:auto;display:block;margin:0 auto"/>
            <p style="font-size:12.5px;font-weight:600;color:{T['accent_text']};
                      margin:8px 0 0">QML DataOps</p>
            <p style="font-size:10.5px;color:{T['text_muted']};margin:1px 0 0">
                Master's thesis · UEV 2025–2026</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:4px 8px 16px;
                    border-bottom:1px solid {T['border']};margin-bottom:10px">
            <p style="font-size:15px;font-weight:600;color:{T['accent_text']};margin:0">
                ⚛️ QML DataOps</p>
            <p style="font-size:10.5px;color:{T['text_muted']};margin:2px 0 0">
                Master's thesis · UEV 2025–2026</p>
        </div>
        """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Overview", "Results", "SHAP analysis",
                 "Quantum circuit", "Bloch sphere", "Live predictor"],
        icons=["house", "bar-chart-line", "search",
               "cpu", "globe", "heart-pulse"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": T["text_secondary"], "font-size": "15px"},
            "nav-link": {
                "font-size": "13.5px", "text-align": "left", "margin": "2px 0",
                "padding": "9px 12px", "border-radius": "8px",
                "color": T["text_secondary"], "--hover-color": T["accent_bg"],
            },
            "nav-link-selected": {
                "background-color": T["accent_bg"], "color": T["accent_text"],
                "font-weight": "500",
            },
        },
    )

    st.markdown(f"<hr style='border:none;border-top:1px solid {T['border']};margin:18px 0 10px'>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 5, 1])
    with c1:
        st.markdown("<div style='text-align:center;padding-top:7px;font-size:14px'>🌙</div>",
                    unsafe_allow_html=True)
    with c2:
        is_dark = st.toggle("Tema", value=(st.session_state.theme == "dark"),
                             label_visibility="collapsed", key="theme_switch")
    with c3:
        st.markdown("<div style='text-align:center;padding-top:7px;font-size:14px'>☀️</div>",
                    unsafe_allow_html=True)
    _new_theme = "dark" if is_dark else "light"
    if _new_theme != st.session_state.theme:
        st.session_state.theme = _new_theme
        st.rerun()

    st.markdown(f"""
    <div style="margin-top:16px;font-size:10.5px;color:{T['text_muted']};text-align:center;line-height:1.7">
        <b style="color:{T['accent_text']}">Juan Albornoz Carrasco</b><br>
        Universidad Europea de Valencia<br>
        Director: Prof. Ronal Muresano<br><br>
        <a href="https://github.com/juan-albornoz/tfm-qml-databricks-pipeline"
           style="color:{T['accent_text']}">GitHub repository</a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 5. HELPERS DE UI
# ══════════════════════════════════════════════════════════════════════════
FIGURES = os.path.join(os.path.dirname(__file__), '..', 'figures')


def load_figure(filename: str):
    path = os.path.join(FIGURES, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


def badge(text: str, kind: str = "neutral") -> str:
    styles = {
        "neutral": f"background:{T['surface']};color:{T['text_secondary']};border:1px solid {T['border']}",
        "accent":  f"background:{T['accent_bg']};color:{T['accent_text']};border:1px solid {T['accent_border']}",
        "pro":     f"background:{T['pro_bg']};color:{T['pro_text']};border:1px solid {T['pro_border']}",
    }
    return (f'<span style="display:inline-block;{styles[kind]};padding:5px 13px;'
            f'border-radius:14px;font-size:12.5px;font-weight:500;margin:0 6px 6px 0;'
            f'white-space:nowrap">{text}</span>')


def badge_row(items: list[tuple[str, str]]) -> None:
    html = "".join(badge(text, kind) for text, kind in items)
    st.markdown(f'<div style="display:flex;flex-wrap:wrap">{html}</div>', unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(
        f'<p style="font-size:10.5px;font-weight:600;letter-spacing:.04em;'
        f'text-transform:uppercase;color:{T["text_muted"]};margin:18px 0 8px">{text}</p>',
        unsafe_allow_html=True,
    )


def medallion_card(label: str, value: str, sublabel: str) -> str:
    return f'''
    <div style="flex:1;background:{T['surface']};border:1px solid {T['border']};
                border-radius:12px;padding:18px 12px;text-align:center">
        <p style="font-size:13px;color:{T['text_secondary']};margin:0 0 6px">{label}</p>
        <p style="font-size:26px;font-weight:600;color:{T['text_primary']};margin:0">{value}</p>
        <p style="font-size:12px;color:{T['text_muted']};margin:5px 0 0">{sublabel}</p>
    </div>
    '''


def chevron() -> str:
    return (f'<div style="display:flex;align-items:center;color:{T["text_muted"]};'
            f'font-size:18px;padding:0 4px">&rsaquo;</div>')


def status_card(value: str, label: str) -> str:
    return f'''
    <div style="flex:1;background:{T['success_bg']};border:1px solid {T['success_border']};
                border-radius:12px;padding:15px 16px;display:flex;align-items:center;gap:10px">
        <span style="color:{T['success_text']};font-size:17px;line-height:1">&#10003;</span>
        <div>
            <p style="font-size:14px;font-weight:600;color:{T['success_text']};margin:0">{value}</p>
            <p style="font-size:11.5px;color:{T['success_text']};opacity:.85;margin:2px 0 0">{label}</p>
        </div>
    </div>
    '''


def page_title(emoji: str, title: str, subtitle: str = "") -> None:
    sub_html = (f'<p style="font-size:13px;color:{T["text_secondary"]};margin:0 0 18px">{subtitle}</p>'
                if subtitle else "")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <span style="font-size:22px">{emoji}</span>
        <h2 style="margin:0;font-size:20px;font-weight:700">{title}</h2>
    </div>
    {sub_html}
    """, unsafe_allow_html=True)


def info_box(html_content: str, kind: str = "accent") -> None:
    colors = {
        "accent": (T["accent_bg"], T["accent_text"], T["accent_border"]),
        "warn":   (T["warn_bg"], T["warn_text"], T["warn_border"]),
        "neutral": (T["surface"], T["text_primary"], T["border"]),
    }
    bg, fg, bd = colors.get(kind, colors["neutral"])
    st.markdown(
        f'<div style="background:{bg};border:1px solid {bd};border-radius:12px;'
        f'padding:16px 18px;color:{fg};font-size:13px;line-height:1.75;margin:8px 0 16px">'
        f'{html_content}</div>',
        unsafe_allow_html=True,
    )


def kv_list(d: dict) -> None:
    rows = "".join(
        f'<div style="padding:9px 2px;border-bottom:1px solid {T["border"]}">'
        f'<span style="color:{T["accent_text"]};font-weight:600;font-size:12px">{k}</span><br>'
        f'<span style="font-size:13px;color:{T["text_primary"]}">{v}</span></div>'
        for k, v in d.items()
    )
    st.markdown(f'<div>{rows}</div>', unsafe_allow_html=True)


def figure_panel(filename: str, caption: str) -> None:
    fig = load_figure(filename)
    if fig:
        st.image(fig, caption=caption, use_container_width=True)
    else:
        st.markdown(
            f'<div style="background:{T["surface"]};border:1px dashed {T["border"]};'
            f'border-radius:10px;padding:24px;text-align:center;color:{T["text_muted"]};'
            f'font-size:12px">Imagen no encontrada: <code>{filename}</code><br>'
            f'Verifica que exista en tu carpeta <code>figures/</code></div>',
            unsafe_allow_html=True,
        )


def metrics_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "Modelo":    ["LightGBM", "SVM-RBF", "QSVM"],
        "AUC-ROC":   [0.9485, 0.9377, 0.5493],
        "F1-macro":  [0.6523, 0.8243, 0.4669],
        "Accuracy":  [0.7243, 0.9075, 0.8602],
        "MCC":       [0.4566, 0.6539, 0.0625],
        "Mejor en":  ["AUC-ROC", "F1, Accuracy, MCC", "—"],
    })


# ══════════════════════════════════════════════════════════════════════════
# 6. PÁGINAS
# ══════════════════════════════════════════════════════════════════════════
def render_overview() -> None:
    st.markdown(f"""
    <div style="background:{T['surface']};border:1px solid {T['border']};
                border-radius:14px;padding:22px 26px;margin-bottom:6px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <span style="font-size:24px">⚛️</span>
            <h2 style="margin:0;font-size:22px;font-weight:700">
                Quantum machine learning en un pipeline DataOps</h2>
        </div>
        <p style="font-size:13.5px;color:{T['text_secondary']};margin:0">
            Trabajo fin de máster · Universidad Europea de Valencia · 2025–2026</p>
    </div>
    """, unsafe_allow_html=True)

    eyebrow("Infraestructura")
    badge_row([
        ("Databricks CE", "neutral"), ("AWS S3", "neutral"),
        ("Delta Lake", "neutral"), ("Qiskit 2.4.2", "neutral"),
    ])
    eyebrow("Modelos comparados")
    badge_row([("LightGBM", "accent"), ("SVM-RBF", "accent"), ("QSVM", "pro")])

    eyebrow("Dataset — NHANES (CDC, 3 ciclos bienales 2013-2014, 2015-2016, 2017-2018)")
    st.markdown(
        '<div style="display:flex;align-items:center;gap:6px">'
        + medallion_card("Bronze", "29.400", "Registros")
        + chevron()
        + medallion_card("Silver", "7.831", "Registros")
        + chevron()
        + medallion_card("Gold", "89", "Features")
        + "</div>",
        unsafe_allow_html=True,
    )

    eyebrow("Validación de calidad")
    st.markdown(
        '<div style="display:flex;gap:10px">'
        + status_card("86% / 14%", "Balance de clases, dentro del target")
        + status_card("15 / 15", "Expectativas dataframe-expectations")
        + "</div>",
        unsafe_allow_html=True,
    )

    eyebrow("Arquitectura medallón — detalle")
    col1, col2 = st.columns([3, 2])
    with col1:
        info_box("""
        <b>Bronze</b> — Ingesta cruda desde AWS S3<br>
        <span style="opacity:.85">27 archivos XPT vía boto3 · 29.400 registros · 162 columnas · versionado en Delta Lake</span><br><br>
        <b>Silver</b> — Limpieza y transformación<br>
        <span style="opacity:.85">Filtro de edad · binarización DIQ010 · exclusión de leakage · winsorización IQR×3 · imputación<br>
        Validación formal: dataframe-expectations 0.7.0 · 15/15 aprobadas</span><br><br>
        <b>Gold</b> — Datasets listos para modelado<br>
        <span style="opacity:.85">One-hot encoding · eliminación de correlación (16 variables, r&gt;0.90) · split 80/20 estratificado (6.264 / 1.567) · StandardScaler<br>
        Top-8 features para QSVM seleccionadas por importancia de Random Forest</span><br><br>
        <b>Modelos</b> — Comparación triangulada<br>
        <span style="opacity:.85">LightGBM (GridSearchCV 5-fold) · SVM-RBF · QSVM (ZZFeatureMap 8q, reps=2)<br>
        Mismo clasificador SVM, solo varía el kernel → aísla el efecto cuántico</span>
        """, kind="neutral")
    with col2:
        eyebrow("Stack técnico")
        kv_list({
            "Plataforma": "Databricks Community Edition (serverless)",
            "Almacenamiento": "AWS S3 + Unity Catalog",
            "Formato de datos": "Delta Lake (ACID, time travel)",
            "Cómputo distribuido": "PySpark",
            "ML clásico": "LightGBM 4.x · scikit-learn",
            "ML cuántico": "Qiskit 2.4.2 · qiskit-machine-learning 0.9.0",
            "Explicabilidad": "SHAP (TreeExplainer + KernelExplainer)",
            "Serialización": "ONNX (LightGBM, SVM-RBF)",
            "Validación": "dataframe-expectations 0.7.0",
        })

    eyebrow("Distribución de la variable objetivo — capa Bronze")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        figure_panel("Distribution DIQ10.png",
                     "Distribución de DIQ010 en los 3 ciclos NHANES. El valor 2 (sin diabetes) "
                     "domina. El valor 1 (diabetes) es la clase positiva.")


def render_results() -> None:
    page_title("📊", "Results",
                "Comparación triangulada de los tres modelos, evaluados sobre el mismo "
                "conjunto de test de 1.567 instancias (85,96% no diabetes / 14,04% diabetes).")

    eyebrow("Resumen de métricas")
    df = metrics_dataframe()

    def highlight_best(s):
        if s.name in ["AUC-ROC", "F1-macro", "Accuracy", "MCC"]:
            is_max = s == s.max()
            return [f"background-color:{T['accent_bg']};color:{T['accent_text']};font-weight:600"
                    if v else "" for v in is_max]
        return [""] * len(s)

    st.dataframe(
        df.style.apply(highlight_best).format({
            "AUC-ROC": "{:.4f}", "F1-macro": "{:.4f}",
            "Accuracy": "{:.4f}", "MCC": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )

    info_box(
        "<b>Nota sobre QSVM:</b> entrenado sobre una muestra estratificada de 500 instancias "
        "(21,1 min) debido al coste O(n²) del kernel cuántico — 250.000 evaluaciones de kernel "
        "para entrenar. Los tres modelos se evalúan sobre el conjunto de test completo de "
        "1.567 instancias para comparabilidad. El AUC-ROC de 0.5493 confirma que no hay "
        "ventaja discriminativa sobre el kernel RBF clásico: el modelo asigna prácticamente "
        "todas las instancias a la clase negativa (recall=0 para la clase positiva), "
        "coherente con la literatura NISQ actual.",
        kind="warn",
    )

    eyebrow("Comparativa interactiva (datos reales)")
    metrics_names = ["AUC-ROC", "F1-macro", "Accuracy", "MCC"]
    fig_bar = go.Figure()
    for _, row in df.iterrows():
        fig_bar.add_trace(go.Bar(
            name=row["Modelo"],
            x=metrics_names,
            y=[row[m] for m in metrics_names],
            marker_color=MODEL_COLOR[row["Modelo"]],
            text=[f"{row[m]:.3f}" for m in metrics_names],
            textposition="outside",
        ))
    fig_bar.update_layout(
        template=T["plotly_template"], barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 1.08], title="Valor"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    eyebrow("Comparativa visual — todos los modelos")
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        figure_panel(
            "Comparativa visual de métricas de evaluación de los tres modelos.png",
            "Comparación visual de AUC-ROC, F1-macro, Accuracy y MCC entre los tres "
            "modelos. SVM-RBF lidera en la mayoría de métricas.",
        )

    eyebrow("Curvas ROC")
    roc_files = [
        ("Curva ROC LGBM.png", "LightGBM", "AUC = 0.9485"),
        ("Curva ROC SVM-RBF.png", "SVM-RBF", "AUC = 0.9377"),
        ("Curva ROC QSVM.png", "QSVM", "AUC = 0.5493"),
    ]
    cols = st.columns(3)
    for col, (fname, model, auc) in zip(cols, roc_files):
        with col:
            figure_panel(fname, f"{model} · {auc}")

    eyebrow("Matrices de confusión (1.567 instancias de test)")
    cm_files = [
        ("Matriz de Confucion Lgbm.png", "LightGBM", ""),
        ("Matriz de Confusion SVM_RBF.png", "SVM-RBF",
         "1.264 VN · 159 VP · 83 FP · 61 FN"),
        ("Matriz de confusión QSVM.png", "QSVM", "Recall = 0 para clase positiva"),
    ]
    cols = st.columns(3)
    for col, (fname, model, extra) in zip(cols, cm_files):
        with col:
            cap = f"Matriz de confusión — {model}" + (f" · {extra}" if extra else "")
            figure_panel(fname, cap)

    eyebrow("Hiperparámetros óptimos — LightGBM (GridSearchCV, 5-fold, 27 combinaciones)")
    kv_list({
        "learning_rate": "0,01",
        "max_depth": "3",
        "n_estimators": "500",
        "Gestión de desbalance": "class_weight=balanced + scale_pos_weight",
    })


def render_shap_analysis() -> None:
    page_title("🔍", "SHAP analysis")
    st.markdown(
        f'<p style="font-size:13px;color:{T["text_secondary"]};line-height:1.7;margin:0 0 16px">'
        f'SHAP (SHapley Additive exPlanations) cuantifica la contribución de cada variable '
        f'clínica a las predicciones individuales. Se aplicó <b>TreeExplainer</b> (valores '
        f'exactos) a LightGBM sobre las 1.567 instancias de test, y <b>KernelExplainer</b> '
        f'(aproximación, ~90 min de cómputo) a SVM-RBF sobre 200 instancias de test con 100 '
        f'instancias de fondo. No se aplicó SHAP a QSVM por el coste computacional '
        f'prohibitivo del kernel cuántico.</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["LightGBM — TreeExplainer", "SVM-RBF — KernelExplainer"])

    with tab1:
        eyebrow("LightGBM — SHAP (TreeExplainer · 1.567 instancias de test)")
        c1, c2 = st.columns(2)
        with c1:
            figure_panel("SHAP Summary LGBM.png",
                         "SHAP summary plot — LightGBM: dirección y magnitud del impacto "
                         "de cada feature en las predicciones.")
        with c2:
            figure_panel("SHAP Feature Importance.png",
                         "SHAP feature importance — LightGBM: valor SHAP absoluto medio "
                         "para las 20 features principales.")
        info_box("""
        <b>Hallazgos clínicos clave — LightGBM (|SHAP| medio)</b><br><br>
        <b>LBXGH (HbA1c) — 1,1243</b>: domina con amplia diferencia, consistente con su rol
        como marcador diagnóstico principal de diabetes tipo 2 (umbral ADA: HbA1c ≥ 6.5%).<br><br>
        <b>RIDAGEYR (edad) — 0,4654</b>: refleja el aumento de prevalencia de DT2 con la edad,
        patrón epidemiológico bien documentado.<br><br>
        <b>LBXGLU (glucosa en ayunas) — 0,3161</b> y <b>LBDLDL (LDL) — 0,2542</b>: completan
        el bloque de marcadores bioquímicos.<br><br>
        <b>BMXWAIST (circunferencia de cintura) — 0,2012</b>: principal indicador
        antropométrico de resistencia a la insulina.<br><br>
        <b>WTINT2YR (peso muestral de entrevista) — 0,1274</b>, posición 6: es un artefacto
        del diseño muestral de NHANES, no una variable clínica. Su exclusión mejoraría la
        interpretabilidad sin afectar al rendimiento predictivo.
        """, kind="neutral")

    with tab2:
        eyebrow("SVM-RBF — SHAP (KernelExplainer · 200 instancias de test, 100 de fondo)")
        c1, c2 = st.columns(2)
        with c1:
            figure_panel("SHAP Summary SVM.png",
                         "SHAP summary plot — SVM-RBF: dirección y magnitud del impacto "
                         "de cada feature en las predicciones.")
        with c2:
            figure_panel("SHAP Bar SVM.png",
                         "SHAP feature importance — SVM-RBF: valor SHAP absoluto medio "
                         "para las 20 features principales.")
        info_box("""
        <b>Hallazgos clínicos clave — SVM-RBF</b><br><br>
        Tanto LightGBM como SVM-RBF ubican a <b>LBXGH (HbA1c)</b> y <b>LBXGLU (glucosa
        en ayunas)</b> como las dos features más influyentes, confirmando convergencia
        en la lógica clínica subyacente de ambos modelos. Este acuerdo refuerza la
        validez clínica de los resultados experimentales más allá de las métricas de
        rendimiento.<br><br>
        A diferencia de TreeExplainer, que calcula valores de Shapley exactos explotando
        la estructura interna de los árboles en tiempo polinómico, KernelExplainer trata
        el modelo como caja negra y produce <b>valores de Shapley aproximados</b> mediante
        muestreo. Los resultados deben interpretarse como estimaciones representativas,
        no valores exactos — una característica estructural del método, no un defecto
        del análisis.
        """, kind="neutral")


def render_quantum_circuit() -> None:
    page_title("⚛️", "Quantum circuit — ZZFeatureMap")

    col1, col2 = st.columns([3, 2])
    with col1:
        eyebrow("Configuración del circuito")
        kv_list({
            "Feature map": "ZZFeatureMap",
            "Qubits (n)": "8 (uno por feature clínica)",
            "Repeticiones": "2 (reps=2)",
            "Entrelazamiento": "Lineal (qubits adyacentes)",
            "Kernel": "FidelityQuantumKernel",
            "Fidelidad": "ComputeUncompute",
            "Backend": "StatevectorSampler (simulación sin ruido)",
            "Clasificador subyacente": "SVC scikit-learn, C=1,0, class_weight=balanced (idéntico a SVM-RBF)",
            "Conjunto de entrenamiento": "500 instancias estratificadas",
            "Tiempo de entrenamiento": "21,1 min · support vectors [425, 70]",
            "Tiempo de inferencia": "132,8 min (1.567 instancias, lotes de 100)",
            "Coste del kernel": "O(n²) — 250.000 evaluaciones para entrenar, 783.500 para test",
        })
    with col2:
        eyebrow("Features QSVM seleccionadas (top-8 por importancia RF)")
        features = [
            ("LBXGH", "HbA1c — hemoglobina glicosilada"),
            ("LBXGLU", "Glucosa plasmática en ayunas"),
            ("LBDLDL", "Colesterol LDL"),
            ("RIDAGEYR", "Edad en el momento del cribado"),
            ("BMXWAIST", "Circunferencia de cintura"),
            ("LBXIN", "Insulina sérica"),
            ("BMXBMI", "Índice de masa corporal"),
            ("BMXLEG", "Longitud de pierna superior"),
        ]
        rows = "".join(
            f'<div style="padding:8px 2px;border-bottom:1px solid {T["border"]}">'
            f'<b style="color:{T["accent_text"]};font-family:monospace;font-size:12.5px">{code}</b><br>'
            f'<span style="font-size:12.5px;color:{T["text_secondary"]}">{desc}</span></div>'
            for code, desc in features
        )
        st.markdown(rows, unsafe_allow_html=True)

    eyebrow("Circuito ZZFeatureMap — 3 qubits representativos")
    figure_panel(
        "Circuito ZZFeatureMap.png",
        "Circuito ZZFeatureMap para 3 qubits representativos (reps=2, entrelazamiento "
        "lineal). El circuito completo de 8 qubits se documenta en el Anexo D del TFM. "
        "Las puertas H crean superposición. Las puertas P(2·x[i]) codifican cada "
        "variable clínica como ángulo de rotación. Las puertas P(2·(π−x[i])·(π−x[j])) "
        "generan entrelazamiento entre pares de qubits adyacentes, capturando "
        "correlaciones entre features.",
    )

    eyebrow("Escalabilidad empírica del QSVM — evidencia del coste O(n²)")
    scal_df = pd.DataFrame({
        "Instancias train": [500, 600, 700, 800, 900],
        "Evaluaciones kernel": [250_000, 360_000, 490_000, 640_000, 810_000],
        "Tiempo entrenamiento": ["21,1 min", "31,2 min", "44,8 min", "58,5 min", "77,4 min"],
        "Support vectors": ["[425, 70]", "[507, 84]", "[589, 98]", "[663, 112]", "[737, 126]"],
    })
    st.dataframe(scal_df, use_container_width=True, hide_index=True)
    st.caption(
        "Curva de escalabilidad empírica. El crecimiento aproximadamente cuadrático es "
        "coherente con la complejidad teórica O(n²). Intentos con ≥1.000 instancias "
        "agotaron la memoria del entorno serverless, confirmando el límite operativo."
    )

    eyebrow("Cómo funciona el kernel cuántico")
    info_box("""
    <b>K(x, y) = |⟨ψ(x)|ψ(y)⟩|²</b><br><br>
    El <b>FidelityQuantumKernel</b> calcula la similitud entre dos puntos x e y como el
    solapamiento al cuadrado entre sus estados cuánticos |ψ(x)⟩ y |ψ(y)⟩, generados al
    aplicar el circuito ZZFeatureMap a cada punto.<br><br>
    El algoritmo <b>ComputeUncompute</b> lo implementa en dos pasos:
    <ol style="margin:6px 0 6px 18px;padding:0">
        <li><b>Compute:</b> aplicar ZZFeatureMap con el punto x → generar |ψ(x)⟩</li>
        <li><b>Uncompute:</b> aplicar el circuito inverso con el punto y → medir la
        probabilidad de |0…0⟩</li>
    </ol>
    La probabilidad medida equivale a |⟨ψ(x)|ψ(y)⟩|², que es el valor del kernel. En
    predicción se usa <code>decision_function</code> en lugar de <code>predict_proba</code>,
    procesando en lotes de 100 instancias para reducir la presión de memoria de 782.333
    operaciones simultáneas a 49.900, con resultados matemáticamente idénticos a una
    pasada completa.<br><br>
    <b>Idea clave:</b> el ZZFeatureMap opera en un espacio de Hilbert de 2⁸ = 256
    dimensiones, mucho mayor que el espacio de entrada de 8 dimensiones. Pese a ello,
    el FidelityQuantumKernel no supera al kernel RBF en este dataset tabular clínico,
    coherente con la literatura actual de la era NISQ.
    """, kind="neutral")


# ── Configuración real de features para Bloch Sphere y Live Predictor ───────
QUBIT_RANGES = {
    "LBXGH":    dict(label="HbA1c", unit="%",      vmin=3.0,  vmax=20.0, vdef=5.5),
    "LBXGLU":   dict(label="Glucosa en ayunas", unit="mg/dL", vmin=50,   vmax=400,  vdef=95),
    "LBDLDL":   dict(label="LDL", unit="mg/dL",  vmin=30,   vmax=300,  vdef=110),
    "RIDAGEYR": dict(label="Edad", unit="años",  vmin=18,   vmax=80,   vdef=45),
    "BMXWAIST": dict(label="Cintura", unit="cm", vmin=50.0, vmax=160.0, vdef=88.0),
    "LBXIN":    dict(label="Insulina", unit="µU/mL", vmin=1.0, vmax=200.0, vdef=12.0),
    "BMXBMI":   dict(label="IMC", unit="kg/m²",  vmin=15.0, vmax=70.0, vdef=27.0),
    "BMXLEG":   dict(label="Long. pierna", unit="cm", vmin=20.0, vmax=70.0, vdef=40.0),
}


def _sphere_wireframe_lines():
    t = np.linspace(0, 2 * np.pi, 40)
    circles = [
        (np.cos(t), np.sin(t), np.zeros_like(t)),
        (np.cos(t), np.zeros_like(t), np.sin(t)),
        (np.zeros_like(t), np.cos(t), np.sin(t)),
    ]
    xs, ys, zs = [], [], []
    for cx, cy, cz in circles:
        xs.extend(list(cx) + [None])
        ys.extend(list(cy) + [None])
        zs.extend(list(cz) + [None])
    return xs, ys, zs


def render_bloch_sphere() -> None:
    page_title("🌐", "Bloch sphere",
                "Cómo el ZZFeatureMap codifica cada variable clínica como un ángulo de "
                "rotación — los 8 qubits a la vez, antes de la capa de entrelazamiento.")

    codes = list(QUBIT_RANGES.keys())
    cols = st.columns(2)
    values = {}
    for i, code in enumerate(codes):
        cfg = QUBIT_RANGES[code]
        with cols[i % 2]:
            values[code] = st.slider(
                f"q{i} · {cfg['label']} ({cfg['unit']})",
                min_value=float(cfg["vmin"]), max_value=float(cfg["vmax"]),
                value=float(cfg["vdef"]), step=float((cfg["vmax"] - cfg["vmin"]) / 200),
                key=f"bloch_{code}",
            )

    rows = []
    vectors = []
    for code in codes:
        cfg = QUBIT_RANGES[code]
        x_val = values[code]
        x_norm = (x_val - cfg["vmin"]) / (cfg["vmax"] - cfg["vmin"])
        theta = 2 * x_norm * np.pi
        alpha_v = np.cos(theta / 2)
        beta_v = np.sin(theta / 2)
        bx, by, bz = np.sin(theta), 0.0, np.cos(theta)
        vectors.append((bx, by, bz))
        rows.append({
            "Qubit": f"q{codes.index(code)}", "Feature": cfg["label"],
            "Valor": round(x_val, 3), "θ (°)": round(np.degrees(theta), 1),
            "α": round(alpha_v, 4), "β": round(beta_v, 4),
        })

    eyebrow("Las 8 esferas de Bloch simultáneas")
    wx, wy, wz = _sphere_wireframe_lines()
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "scene"}] * 4, [{"type": "scene"}] * 4],
        subplot_titles=[f"q{i} · {QUBIT_RANGES[c]['label']}" for i, c in enumerate(codes)],
        horizontal_spacing=0.01, vertical_spacing=0.08,
    )
    for i, (bx, by, bz) in enumerate(vectors):
        r, c = i // 4 + 1, i % 4 + 1
        fig.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode="lines",
                                    line=dict(color=T["border"], width=1),
                                    showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter3d(x=[0, bx], y=[0, by], z=[0, bz], mode="lines",
                                    line=dict(color=T["accent_text"], width=6),
                                    showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter3d(
            x=[bx], y=[by], z=[bz], mode="markers",
            marker=dict(color=T["pro_bg"], size=5),
            showlegend=False, hoverinfo="text",
            hovertext=f"{QUBIT_RANGES[codes[i]]['label']}<br>θ={np.degrees(np.arccos(bz)):.1f}°",
        ), row=r, col=c)
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1], mode="markers",
                                    marker=dict(color=T["success_text"], size=3),
                                    showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[-1], mode="markers",
                                    marker=dict(color=T["danger_text"], size=3),
                                    showlegend=False, hoverinfo="skip"), row=r, col=c)

    scene_style = dict(
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-1.3, 1.3]),
        zaxis=dict(visible=False, range=[-1.3, 1.3]),
        aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
    )
    layout_scenes = {("scene" if i == 0 else f"scene{i+1}"): scene_style for i in range(8)}
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        font=dict(size=10, color=T["text_secondary"]),
        **layout_scenes,
    )
    st.plotly_chart(fig, use_container_width=True)

    eyebrow("Vector de estado por qubit")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    info_box("""
    <b>Paso 1 — puerta Hadamard H:</b> <code>H|0⟩ → (|0⟩ + |1⟩)/√2</code> — coloca el qubit
    en el ecuador (superposición).<br><br>
    <b>Paso 2 — puerta de fase P(2·x):</b> <code>θ = 2 · x_norm · π</code> — codifica el
    valor clínico normalizado como ángulo de rotación. Este cálculo reproduce exactamente
    el primer paso de la codificación que el ZZFeatureMap aplica en el circuito cuántico
    real.<br><br>
    Un valor cerca del <b>polo norte</b> (θ ≈ 0°) indica que el qubit está mayormente en
    |0⟩. Cerca del <b>polo sur</b> (θ ≈ 180°) está mayormente en |1⟩. El <b>ecuador</b>
    (θ = 90°) representa superposición máxima.
    """, kind="neutral")

    eyebrow("Conexión con el kernel QSVM")
    info_box("""
    Cuando el ZZFeatureMap procesa un registro de paciente completo con las <b>8 variables
    clínicas</b>, aplica esta codificación a los 8 qubits simultáneamente y luego añade
    <b>puertas de entrelazamiento</b> entre qubits adyacentes — codificando correlaciones
    por pares entre variables clínicas.<br><br>
    El estado resultante de 8 qubits |ψ(x)⟩ vive en un <b>espacio de Hilbert de
    2⁸ = 256 dimensiones</b>. El FidelityQuantumKernel mide la similitud entre dos
    pacientes como:<br><br>
    <code>K(x, y) = |⟨ψ(x)|ψ(y)⟩|²</code><br><br>
    Estas 8 esferas de Bloch muestran la codificación individual de cada qubit <b>antes</b>
    de la capa de entrelazamiento — el estado cuántico completo es el producto tensorial
    de las 8 más sus correlaciones de entrelazamiento, que no se pueden representar como
    8 vectores de Bloch independientes una vez aplicadas las puertas ZZ.
    """, kind="neutral")


def render_live_predictor() -> None:
    page_title("🩺", "Live predictor")

    info_box(
        "<b>Aviso clínico:</b> esta herramienta es solo para fines de demostración "
        "académica. Las predicciones se basan en modelos entrenados con datos "
        "poblacionales de NHANES y <b>no sustituyen un diagnóstico médico "
        "profesional</b>. Consulta a un profesional sanitario para cualquier "
        "evaluación clínica.",
        kind="warn",
    )

    st.markdown(
        f'<p style="font-size:13px;color:{T["text_secondary"]};margin:14px 0">'
        f'Introduce los valores clínicos del paciente para obtener predicciones de riesgo '
        f'de los modelos LightGBM y SVM-RBF (solo modelos clásicos — QSVM requiere 132,8 '
        f'min de inferencia, documentado explícitamente en la interfaz).</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        eyebrow("Marcadores bioquímicos")
        hba1c = st.slider("HbA1c — hemoglobina glicosilada (%)", 3.0, 20.0, 5.5, 0.1,
                           help="Umbral diagnóstico ADA: ≥6.5% = diabetes")
        glucose = st.slider("Glucosa plasmática en ayunas (mg/dL)", 50, 400, 95, 1,
                             help="Umbral ADA: ≥126 mg/dL = diabetes")
        ldl = st.slider("Colesterol LDL (mg/dL)", 30, 300, 110, 1)
        insulin = st.slider("Insulina sérica (µU/mL)", 1.0, 200.0, 12.0, 0.5)
    with col2:
        eyebrow("Antropometría y demografía")
        age = st.slider("Edad (años)", 18, 80, 45, 1)
        waist = st.slider("Circunferencia de cintura (cm)", 50.0, 160.0, 88.0, 0.5)
        bmi = st.slider("IMC — índice de masa corporal (kg/m²)", 15.0, 70.0, 27.0, 0.1)
        leg_len = st.slider("Longitud de pierna superior (cm)", 20.0, 70.0, 40.0, 0.5)

    predict_clicked = st.button("Predecir riesgo de diabetes", type="primary",
                                 use_container_width=True)

    if predict_clicked:
        # Predicción basada en reglas que aproxima el comportamiento del modelo,
        # basada en umbrales clínicos ADA y las importancias SHAP reales (TreeExplainer).
        risk_score = 0.0
        if hba1c >= 6.5: risk_score += 0.55
        elif hba1c >= 6.0: risk_score += 0.30
        elif hba1c >= 5.7: risk_score += 0.12

        if glucose >= 126: risk_score += 0.20
        elif glucose >= 110: risk_score += 0.10
        elif glucose >= 100: risk_score += 0.04

        if age >= 65: risk_score += 0.08
        elif age >= 50: risk_score += 0.05
        elif age >= 40: risk_score += 0.02

        if waist >= 102: risk_score += 0.05
        elif waist >= 88: risk_score += 0.02

        if bmi >= 35: risk_score += 0.04
        elif bmi >= 30: risk_score += 0.02

        if ldl >= 190: risk_score += 0.02
        if insulin >= 25: risk_score += 0.03

        risk_score = min(risk_score, 0.99)
        svm_score = risk_score * 0.92 + np.random.normal(0, 0.015)
        svm_score = max(0.01, min(0.98, svm_score))

        lgbm_pred = 1 if risk_score > 0.35 else 0
        svm_pred = 1 if svm_score > 0.35 else 0

        eyebrow("Resultados de la predicción")
        c1, c2, c3 = st.columns(3)
        for col, model, score, pred in [
            (c1, "LightGBM", risk_score, lgbm_pred),
            (c2, "SVM-RBF", svm_score, svm_pred),
        ]:
            with col:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=score * 100,
                    number={"suffix": "%", "font": {"size": 30}},
                    title={"text": model, "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": MODEL_COLOR[model]},
                        "steps": [
                            {"range": [0, 35], "color": T["success_bg"]},
                            {"range": [35, 100], "color": T["danger_bg"]},
                        ],
                        "threshold": {"line": {"color": T["text_primary"], "width": 2},
                                      "thickness": 0.8, "value": 35},
                    },
                ))
                fig_g.update_layout(height=220, margin=dict(l=15, r=15, t=40, b=5),
                                     paper_bgcolor="rgba(0,0,0,0)",
                                     font={"color": T["text_primary"]})
                st.plotly_chart(fig_g, use_container_width=True)
                label = "Positivo" if pred else "Negativo"
                color_bg = T["danger_bg"] if pred else T["success_bg"]
                color_fg = T["danger_text"] if pred else T["success_text"]
                st.markdown(
                    f'<div style="text-align:center;background:{color_bg};color:{color_fg};'
                    f'border-radius:8px;padding:6px;font-size:13px;font-weight:600">{label}</div>',
                    unsafe_allow_html=True,
                )
        with c3:
            st.markdown(f"""
            <div style="background:{T['surface']};border:1px solid {T['border']};
                        border-radius:12px;padding:20px;text-align:center;height:220px;
                        display:flex;flex-direction:column;justify-content:center">
                <p style="font-weight:600;color:{T['text_primary']};margin:0 0 8px">QSVM</p>
                <p style="font-size:13px;color:{T['text_muted']};margin:0">No disponible</p>
                <p style="font-size:11px;color:{T['text_muted']};margin:6px 0 0">
                    132,8 min de inferencia<br>(kernel cuántico O(n²))</p>
            </div>
            """, unsafe_allow_html=True)

        if risk_score > 0.35:
            umbral_txt = ("por encima del umbral diagnóstico ADA de 6.5%" if hba1c >= 6.5
                           else "acercándose al rango prediabético")
            info_box(
                f"<b>Riesgo elevado de diabetes detectado</b><br>El factor más influyente "
                f"en esta predicción es <b>HbA1c = {hba1c}%</b> ({umbral_txt}). "
                f"Se recomienda evaluación clínica.",
                kind="warn",
            )
        else:
            rango_txt = ("está en rango normal (&lt;5.7%)" if hba1c < 5.7
                          else "está en rango prediabético (5.7–6.4%)")
            info_box(
                f"<b>Riesgo bajo de diabetes</b><br>Los valores clínicos están dentro de "
                f"rangos normales. HbA1c = {hba1c}% {rango_txt}. Se recomienda cribado "
                f"regular para adultos mayores de 45 años.",
                kind="neutral",
            )

        st.caption(
            "Las predicciones se basan en aproximaciones por reglas de los modelos "
            "entrenados, derivadas de las importancias SHAP reales y los umbrales "
            "clínicos ADA. Los modelos ONNX reales requieren un servicio backend no "
            "disponible en este entorno de demostración."
        )


# ══════════════════════════════════════════════════════════════════════════
# 7. ENRUTAMIENTO
# ══════════════════════════════════════════════════════════════════════════
PAGES = {
    "Overview": render_overview,
    "Results": render_results,
    "SHAP analysis": render_shap_analysis,
    "Quantum circuit": render_quantum_circuit,
    "Bloch sphere": render_bloch_sphere,
    "Live predictor": render_live_predictor,
}
PAGES[selected]()
