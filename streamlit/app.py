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

import base64
import io
import json
import textwrap
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

ASSETS_DIR = Path(__file__).parent / "assets"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
MODELS_DIR = Path(__file__).parent / "models"

@st.cache_data
def _b64_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

@st.cache_data
def _b64_image_autocrop(path: str, pad: int = 16) -> str:
    """Recorta el margen blanco sobrante alrededor del contenido real (p. ej. capturas de
    diagramas exportadas con aire de más) y devuelve el PNG resultante en base64. Mantiene un
    margen uniforme pequeño para que no quede pegado al borde de la tarjeta contenedora."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    non_white = np.any(arr < 250, axis=2)
    rows, cols = np.any(non_white, axis=1), np.any(non_white, axis=0)
    if not rows.any():
        return _b64_image(path)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    box = (max(cmin - pad, 0), max(rmin - pad, 0),
           min(cmax + pad, img.width), min(rmax + pad, img.height))
    buf = io.BytesIO()
    img.crop(box).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# initial_sidebar_state="auto": expandida en escritorio, COLAPSADA en móvil. Con "expanded" se quedaba
# abierta también en el teléfono, comiéndose 270 de los ~390 px de pantalla.
st.set_page_config(page_title="QML DataOps", page_icon="◆", layout="wide", initial_sidebar_state="auto")

# ─────────────────────────────────────────────────────────────────────────
# TOKENS DE DISEÑO (design brief del TFM)
# ─────────────────────────────────────────────────────────────────────────
C_PRIMARY, C_MID1, C_MID2, C_LIGHT, C_DARK = "#5D8BA6", "#86A8BC", "#AEC5D2", "#D7E2E9", "#3D6C87"

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "sidebar_narrow" not in st.session_state:
    st.session_state.sidebar_narrow = False

def T():
    if st.session_state.theme == "dark":
        return dict(bg="#0F1B22", surface="#16242C", surface_alt="#1C2C35",
                     text="#E8EEF2", text_secondary="#93A6B0", border="#26363F",
                     sidebar_bg="#141F26", sidebar_active="#1F2E37")
    return dict(bg="#FAFBFC", surface="#FFFFFF", surface_alt="#F4F7F9",
                 text="#1A2B33", text_secondary="#5B6E77", border="#E5EBEE",
                 sidebar_bg="#F3F5F7", sidebar_active="#E8ECF3")

t = T()
narrow = st.session_state.sidebar_narrow
SIDEBAR_WIDTH = "84px" if narrow else "270px"
# Color del carril vacío de los sliders: claro en tema claro, hundido en tema oscuro (si dejáramos
# C_LIGHT fijo, en oscuro el carril quedaría un surco brillante sobre fondo oscuro).
SLIDER_GROOVE = C_LIGHT if st.session_state.theme == "light" else t["surface_alt"]

# ── Tratamiento de las figuras PNG de fondo blanco (beeswarm SHAP, circuito) según tema ──
# Esas imágenes tienen fondo blanco intrínseco (figuras científicas del TFM). En tema claro se funden
# con la tarjeta. En oscuro, un bloque blanco a ancho completo sobre fondo casi negro "flota" y rompe
# la elegancia del dark mode. En vez de regenerar las imágenes (fuera de alcance), las presentamos
# como "lámina enmarcada": un anillo fino en color de marca (C_MID2) las ata a la paleta, una sombra
# profunda las asienta, y un filtro casi imperceptible baja el fogonazo del blanco puro. En claro se
# mantiene el aspecto plano y sobrio de siempre.
_is_dark = st.session_state.theme == "dark"
FIG_IMG_FILTER = "brightness(0.965) saturate(1.03)" if _is_dark else "none"
FIG_CARD_SHADOW = (f"0 0 0 1px {C_MID2}33, 0 12px 34px rgba(0,0,0,0.42)" if _is_dark
                   else "0 1px 2px rgba(20,30,40,0.04), 0 2px 8px rgba(20,30,40,0.05)")
FIG_CARD_SHADOW_HOVER = (f"0 0 0 1px {C_PRIMARY}66, 0 16px 42px rgba(0,0,0,0.50)" if _is_dark
                         else "0 2px 4px rgba(20,30,40,0.06), 0 8px 20px rgba(20,30,40,0.09)")
# Padding-"passe-partout" algo mayor en oscuro: la lámina blanca respira dentro del marco.
FIG_CARD_PAD = "14px" if _is_dark else "10px"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',system-ui,sans-serif !important; }}
.stApp {{ background-color:{t['bg']}; color:{t['text']}; }}
/* Reduce a la mitad el espacio vacío por defecto entre el borde superior y el contenido,
   igual en las 6 páginas ya que todas comparten esta misma estructura de bloque principal. */
div[data-testid="stMainBlockContainer"], section.main > div.block-container {{
    padding-top:3rem !important;
    /* Con layout="wide" el contenido se estiraba sin límite: en monitores anchos las líneas quedaban
       larguísimas. Un ancho máximo lo mantiene bien proporcionado a zoom 100% (en pantallas de
       1366-1920 px no cambia nada; solo actúa por encima de ~1500 px de área de contenido). */
    max-width:1500px !important;
    margin-left:auto !important; margin-right:auto !important;
}}
section[data-testid="stSidebar"] {{
    background-color:{t['sidebar_bg']}; border-right:1px solid {t['border']};
    box-shadow: 3px 0 16px rgba(20,30,40,0.07), 1px 0 3px rgba(20,30,40,0.05);
    width:{SIDEBAR_WIDTH} !important; min-width:{SIDEBAR_WIDTH} !important; max-width:{SIDEBAR_WIDTH} !important;
    /* Colapsar/descolapsar desliza el ancho en vez de saltar de golpe entre 270 y 84px. */
    transition: width 0.32s cubic-bezier(0.4,0,0.2,1), min-width 0.32s cubic-bezier(0.4,0,0.2,1),
                max-width 0.32s cubic-bezier(0.4,0,0.2,1) !important;
}}
section[data-testid="stSidebar"] > div {{ background-color:{t['sidebar_bg']}; }}
/* El resize handle NATIVO de Streamlit (arrastrar el borde derecho de la sidebar) deja el ancho en
   cualquier valor arbitrario, algo que nuestro modo "narrow" (booleano con solo dos anchos fijos,
   84/270px) no contempla — arrastrándolo, la sidebar queda visualmente angosta pero
   sidebar_narrow sigue en False, así que el texto de los ítems se sigue dibujando a tamaño completo
   junto al icono, apretado. Ese handle además se solapa físicamente con nuestro botón circular de
   colapso (mismo borde) y le bloquea el clic. Lo desactivamos del todo: el ancho lo controla
   exclusivamente nuestro propio toggle. Sin testid propio, pero el estilo inline con
   cursor:col-resize es estable entre versiones de Streamlit (a diferencia de sus clases con hash).
*/
section[data-testid="stSidebar"] div[style*="cursor: col-resize"] {{ pointer-events:none !important; }}
/* Ocultamos el botón nativo de colapso de la cabecera: usamos nuestro propio toggle «/» */
[data-testid="stSidebarCollapseButton"] {{ display:none !important; }}
/* Colapsamos el espaciador nativo del logo/cabecera (reservado por Streamlit cuando no se usa
   st.logo()) para que nuestro propio logo quede pegado arriba, sin hueco encima. */
[data-testid="stSidebarHeader"], [data-testid="stLogoSpacer"] {{
    height:0 !important; min-height:0 !important; padding:0 !important; margin:0 !important;
}}
/* El botón nativo para volver a expandir (cuando la sidebar está totalmente oculta) se
   mantiene visible y estilizado, por si el usuario la colapsa del todo por otra vía. */
[data-testid="collapsedControl"] button {{
    background-color:{t['surface']} !important;
    border:1px solid {t['border']} !important;
    border-radius:50% !important;
    box-shadow: 0 2px 6px rgba(20,30,40,0.10), 0 1px 3px rgba(20,30,40,0.08) !important;
    width:34px !important; height:34px !important;
}}
[data-testid="collapsedControl"] button:hover {{
    background-color:{t['surface_alt']} !important;
}}
/* Botones de la sidebar centrados en su columna */
section[data-testid="stSidebar"] div[data-testid="stButton"] {{ display:flex; justify-content:center; }}
/* Toggle de colapso estilo Notion/Linear: círculo pequeño anclado al borde derecho de la
   sidebar, semi-superpuesto (mitad dentro, mitad fuera), a la altura del logo. */
.st-key-toggle_sidebar {{ height:0 !important; overflow:visible !important; }}
.st-key-toggle_sidebar div[data-testid="stButton"] {{ display:contents !important; }}
.st-key-toggle_sidebar button {{
    position:fixed !important; top:{16 + (40 if narrow else 64) // 2 - 12}px !important;
    left:{SIDEBAR_WIDTH} !important; transform:translateX(-50%) !important;
    width:24px !important; height:24px !important; min-height:24px !important; padding:0 !important;
    border-radius:50% !important; border:1px solid {t['border']} !important;
    background-color:{t['sidebar_bg']} !important; color:{t['text_secondary']} !important;
    box-shadow: 0 1px 4px rgba(20,30,40,0.15), 0 1px 2px rgba(20,30,40,0.10) !important;
    z-index:1000 !important; margin:0 !important;
    /* Acompaña al borde de la sidebar en el mismo tiempo/curva que su transición de ancho, así el
       botón viaja pegado al borde en vez de saltar de golpe a su nueva posición. */
    transition: left 0.32s cubic-bezier(0.4,0,0.2,1), top 0.32s cubic-bezier(0.4,0,0.2,1),
                color 0.15s ease, border-color 0.15s ease !important;
}}
.st-key-toggle_sidebar button:hover {{
    color:{C_PRIMARY} !important; border-color:{C_PRIMARY} !important;
}}
.st-key-toggle_sidebar button p {{
    font-size:17px !important; line-height:1 !important; font-weight:800 !important;
}}
/* Cápsula-interruptor de tema: fija al fondo del viewport (ancho = ancho actual de la sidebar),
   así queda siempre visible sin depender del scroll interno, colapsada o no. */
.st-key-theme_toggle {{
    position:fixed !important; bottom:64px; left:0; width:{SIDEBAR_WIDTH};
    display:flex !important; justify-content:center; z-index:999;
    transition: width 0.32s cubic-bezier(0.4,0,0.2,1);
}}
.st-key-theme_toggle button {{
    width:30px !important; height:15px !important; min-height:15px !important; padding:0 !important;
    border-radius:999px !important; border:none !important;
    background-color:{"#FFFFFF" if st.session_state.theme == "light" else "#0B1319"} !important;
    box-shadow: 0 0 0 1px {C_MID2}55, 0 0 7px 1.5px {C_MID2}99, 0 0 15px 4px {C_MID2}55 !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}}
.st-key-theme_toggle button:hover {{
    box-shadow: 0 0 0 1px {C_MID2}88, 0 0 10px 2px {C_MID2}CC, 0 0 20px 5px {C_MID2}77 !important;
    transform: scale(1.05);
}}
.st-key-theme_toggle button p {{ font-size:0 !important; }}
/* Footer fijo al fondo de la sidebar (por debajo de la cápsula de tema) */
.sidebar-footer {{
    position:fixed; bottom:0; left:0; width:{SIDEBAR_WIDTH};
    padding:8px 6px 10px; text-align:center; box-sizing:border-box;
    border-top:1px solid {t['border']}; background-color:{t['sidebar_bg']};
    color:{t['text_secondary']}; overflow:hidden; z-index:997; line-height:1.35;
    transition: width 0.32s cubic-bezier(0.4,0,0.2,1);
}}
.sidebar-footer .footer-name {{ font-size:14.5px; color:{t['text']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.sidebar-footer .footer-uni {{ font-size:14.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
/* El option_menu vive en un iframe con fondo propio: igualarlo al de la sidebar (sin caja/sombra
   propia — ya tiene el mismo fondo, así que se funde visualmente con el resto de la sidebar) */
section[data-testid="stSidebar"] iframe {{ background-color:{t['sidebar_bg']} !important; }}
section[data-testid="stSidebar"] div[data-testid="stIFrame"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(iframe) {{
    background-color:{t['sidebar_bg']} !important;
}}
/* El menú (option_menu) remonta el iframe entero al colapsar/descolapsar (key distinta para narrow
   vs. ancho — necesario para que reaplique sus estilos, ver comentario junto a esa key), así que el
   texto no puede "encogerse" con una transición: aparece/desaparece de golpe con el nuevo iframe. Un
   fundido suave en el propio iframe disimula ese salto, en vez de eliminarlo — reutiliza el mismo
   keyframe pageFadeIn que ya usan las páginas al navegar. */
@media (prefers-reduced-motion: no-preference) {{
    section[data-testid="stSidebar"] iframe {{ animation: pageFadeIn 0.3s ease-out; }}
}}
.page-title {{ font-size:38px; font-weight:600; color:{t['text']}; margin-bottom:14px; letter-spacing:-0.02em; line-height:1.2; }}
.page-subtitle {{ font-size:18px; font-weight:400; color:{t['text_secondary']}; margin-bottom:32px; line-height:1.6; }}
.kpi-card, .info-card {{
    background-color:{t['surface']}; border:1px solid {t['border']}; border-radius:12px; padding:18px 20px; height:100%;
    box-shadow: 0 1px 2px rgba(20,30,40,0.04), 0 2px 8px rgba(20,30,40,0.05);
    transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
}}
.kpi-card:hover, .info-card:hover {{
    box-shadow: 0 2px 4px rgba(20,30,40,0.06), 0 8px 20px rgba(20,30,40,0.09);
    transform: translateY(-2px);
}}
/* Tarjeta "lead" (párrafo introductorio) con acento lateral en el color primario */
.lead-card {{ border-left:3px solid {C_PRIMARY}; }}
/* Fila de tarjetas comparativas: grid en vez de st.columns para que las 3 tengan SIEMPRE la
   misma altura (el estirado es nativo del grid), sin importar cuánto texto envuelva ni el zoom. */
.compare-grid {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px; align-items:stretch; }}
.compare-grid .info-card {{ height:100%; box-sizing:border-box; }}
/* Esfera de Bloch: el alto lo fija la figura Plotly en píxeles (height=BLOCH_H en Python), SIN
   autosize. Así el tamaño es idéntico en cada rerun y en CUALQUIER navegador (Firefox incluido):
   no depende de medir el contenedor (lo que con autosize hacía que "volviera a quedar pequeña" al
   cambiar de variable o mover el slider, con distinto comportamiento entre Firefox y Chrome). Aquí
   solo centramos el recuadro; no forzamos alturas por CSS para no reintroducir el bucle de medición. */
.st-key-bloch_row div[data-testid="stPlotlyChart"] {{ display:flex; align-items:center; justify-content:center; }}
.st-key-bloch_row .info-card {{ height:auto; box-sizing:border-box; }}
/* Reserva SIEMPRE el hueco de la barra de scroll vertical. Sin esto, cuando la barra aparece/desaparece
   al re-renderizar (Firefox la reserva de verdad), el ancho del contenido salta ~15 px y la esfera de
   Bloch —limitada por el ancho de su columna— se redimensionaba al cambiar de variable o mover el slider.
   El ÚNICO contenedor con scroll en Streamlit es section[data-testid="stMain"] (overflow:auto); aplicarlo
   ahí basta y evita reservar el hueco por duplicado. En móvil (barras superpuestas) es un no-op. */
section[data-testid="stMain"] {{ scrollbar-gutter: stable; }}
/* Circuito Cuántico: la tarjeta de KPIs "Entrenamiento y evaluación" (columna derecha) crece para
   igualar la altura de la tarjeta de la gráfica (columna izquierda) — mismo nivel, sin hueco vacío
   bajo ella. Es el ÚNICO elemento de esa columna (la nota se renderiza aparte, a ancho completo,
   debajo de la fila), así se estira limpio. El contenedor exterior sí crece con flex, pero height:100%
   no se propaga solo a través de los divs intermedios de Streamlit (stMarkdown, su wrapper,
   stMarkdownContainer — todos con height:auto por defecto): hay que fijarlo en cada nivel de la
   cadena para que llegue hasta la tarjeta. */
.st-key-qc_stats_row div[data-testid="stColumn"]:last-of-type div[data-testid="stElementContainer"]:last-child {{
    flex:1 1 auto !important;
}}
.st-key-qc_stats_row div[data-testid="stColumn"]:last-of-type div[data-testid="stElementContainer"]:last-child,
.st-key-qc_stats_row div[data-testid="stColumn"]:last-of-type div[data-testid="stElementContainer"]:last-child div[data-testid="stMarkdown"],
.st-key-qc_stats_row div[data-testid="stColumn"]:last-of-type div[data-testid="stElementContainer"]:last-child div[data-testid="stMarkdown"] > div,
.st-key-qc_stats_row div[data-testid="stColumn"]:last-of-type div[data-testid="stElementContainer"]:last-child div[data-testid="stMarkdownContainer"] {{
    height:100%;
}}
/* Dentro de la tarjeta estirada, las 5 filas de KPI se reparten proporcionalmente en todo el alto
   (space-between) en vez de quedar apiladas arriba con hueco vacío debajo. El margin-top:-8px corrige
   un espaciado que Streamlit añade distinto según el tipo de widget (st.plotly_chart vs st.markdown):
   medido en el navegador, el título de esta columna termina exactamente a la misma altura que el de
   la gráfica, pero esta tarjeta arrancaba 8px más abajo que la suya — con esto ambas tarjetas quedan
   alineadas arriba Y abajo. */
.st-key-qc_stats_row .info-card {{
    height:100%; box-sizing:border-box; display:flex; flex-direction:column; justify-content:space-between;
    margin-top:-8px;
}}
/* Predictor en Vivo: "Score de riesgo" (izquierda) y velocímetro (derecha) fusionados en UNA sola
   tarjeta. Ahora es el propio contenedor de la fila el que hace de tarjeta (borde, fondo, sombra,
   padding y hover); los dos hijos —la kpi-card y la gráfica Plotly— pierden su recuadro individual
   y se funden dentro, separados por una línea vertical fina. Con vertical_alignment="center" en las
   columnas, ambas mitades quedan centradas entre sí sin necesidad de fijar alturas en px. */
.st-key-predictor_gauge_row {{
    background-color:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;
    padding:18px 26px; box-shadow: 0 1px 2px rgba(20,30,40,0.04), 0 2px 8px rgba(20,30,40,0.05);
    transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
}}
.st-key-predictor_gauge_row:hover {{
    box-shadow: 0 2px 4px rgba(20,30,40,0.06), 0 8px 20px rgba(20,30,40,0.09);
    transform: translateY(-2px);
}}
.st-key-predictor_gauge_row .kpi-card, .st-key-predictor_gauge_row .kpi-card:hover {{
    background:transparent; border:none; box-shadow:none; transform:none; padding:6px 4px; height:auto;
    box-sizing:border-box; display:flex; flex-direction:column; justify-content:center;
}}
.st-key-predictor_gauge_row div[data-testid="stPlotlyChart"],
.st-key-predictor_gauge_row div[data-testid="stPlotlyChart"]:hover {{
    background:transparent; border:none; box-shadow:none; padding:0; transform:none;
}}
/* Separador vertical entre las dos mitades de la tarjeta unificada */
.st-key-predictor_gauge_row div[data-testid="stColumn"]:first-of-type {{
    border-right:1px solid {t['border']};
}}
/* Ítem de la arquitectura Medallón como mini-tarjeta con acento lateral por capa */
.medallion-item {{
    display:flex; gap:12px; align-items:flex-start; padding:13px 15px; margin-bottom:10px;
    background:{t['surface']}; border:1px solid {t['border']}; border-left:3px solid {C_PRIMARY};
    border-radius:10px; box-shadow:0 1px 2px rgba(20,30,40,0.04);
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}}
.medallion-item:hover {{ box-shadow: 0 2px 4px rgba(20,30,40,0.06), 0 8px 20px rgba(20,30,40,0.09); transform:translateY(-2px); }}
/* Contenedores de gráficas Plotly con la misma profundidad sutil Y el mismo hover de elevación que
   las tarjetas .kpi-card / .info-card, para que TODAS las tarjetas reaccionen igual al pasar el ratón. */
div[data-testid="stPlotlyChart"] {{
    background-color:{t['surface']};
    border:1px solid {t['border']};
    border-radius:12px;
    padding:8px;
    box-shadow: 0 1px 2px rgba(20,30,40,0.04), 0 2px 8px rgba(20,30,40,0.05);
    transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
}}
div[data-testid="stPlotlyChart"]:hover {{
    box-shadow: 0 2px 4px rgba(20,30,40,0.06), 0 8px 20px rgba(20,30,40,0.09);
    transform: translateY(-2px);
}}
/* Tarjeta contenedora de una figura/imagen (SHAP summary, circuito cuántico): mismo aspecto y mismo
   hover de elevación que el resto. Fondo blanco fijo porque esas imágenes lo son. En tema oscuro se
   convierte en "lámina enmarcada" (anillo de marca + sombra profunda + brillo atenuado) para que el
   bloque blanco no flote sobre el fondo oscuro — ver FIG_CARD_SHADOW / FIG_IMG_FILTER. */
.fig-card {{
    background:#FFFFFF; border:1px solid {t['border']}; border-radius:12px; padding:{FIG_CARD_PAD};
    box-shadow: {FIG_CARD_SHADOW};
    transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
}}
.fig-card:hover {{
    box-shadow: {FIG_CARD_SHADOW_HOVER};
    transform: translateY(-2px);
}}
/* Atenúa el fogonazo del blanco puro en oscuro (imperceptible en claro: filter:none). El radio
   redondea la imagen igual que el inline style, por si algún navegador lo ignorara. */
.fig-card img {{ filter: {FIG_IMG_FILTER}; }}
.kpi-model {{ font-size:13px; font-weight:600; margin-bottom:12px; display:flex; align-items:center; gap:7px; }}
.kpi-dot {{ width:8px; height:8px; border-radius:50%; }}
.kpi-row {{ display:flex; justify-content:space-between; align-items:baseline; padding:7px 0; border-bottom:1px solid {t['border']}; }}
.kpi-row:last-child {{ border-bottom:none; }}
.kpi-label {{ font-size:13px; color:{t['text_secondary']}; font-weight:400; }}
.kpi-value {{ font-size:17px; font-weight:700; color:{t['text']}; }}
.kpi-value-auc {{ font-size:clamp(20px, 2.4vw, 34px); font-weight:700; color:{t['text']}; }}
.stat-num {{ font-size:clamp(16px, 2vw, 30px); font-weight:700; color:{t['text']}; white-space:nowrap; line-height:1.1; }}
.stat-label {{ font-size:13px; color:{t['text_secondary']}; margin-top:3px; }}
/* Tarjeta de estadística: altura consistente entre las 4 columnas, contenido alineado abajo */
.stat-card {{ display:flex; flex-direction:column; justify-content:flex-end; min-height:112px; }}
.section-title {{ font-size:18px; font-weight:600; color:{t['text']}; margin-bottom:4px; display:flex; align-items:center; gap:9px; }}
.section-title::before {{ content:""; width:3px; height:16px; border-radius:2px; background:{C_PRIMARY}; flex-shrink:0; }}
.section-sub {{ font-size:14px; color:{t['text_secondary']}; margin-bottom:14px; padding-left:12px; }}
.badge {{ display:inline-block; font-size:12px; font-weight:600; letter-spacing:0.02em; padding:4px 10px;
    border-radius:20px; background:{C_LIGHT}55; color:{C_DARK}; margin-right:6px; }}
.clinical-note {{ background:{C_LIGHT}33; border:1px solid {C_MID2}; border-left:3px solid {C_PRIMARY}; border-radius:8px; padding:12px 16px 12px 18px; font-size:14px; color:{t['text_secondary']}; line-height:1.6;
    box-shadow: 0 1px 2px rgba(20,30,40,0.04), 0 2px 8px rgba(20,30,40,0.05);
    transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease; }}
.clinical-note:hover {{ box-shadow: 0 2px 4px rgba(20,30,40,0.06), 0 8px 20px rgba(20,30,40,0.09); transform: translateY(-2px); }}
/* Matriz de confusión (cuadrícula HTML: celdas legibles, etiquetas horizontales) */
.cm-title {{ font-size:14px; font-weight:600; color:{t['text']}; margin-bottom:14px; display:flex; align-items:center; gap:8px; }}
.cm-grid {{ display:grid; grid-template-columns:64px 1fr 1fr; gap:6px; align-items:stretch; }}
.cm-collabel {{ font-size:11px; font-weight:500; color:{t['text_secondary']}; text-align:center; align-self:end; padding-bottom:5px; line-height:1.3; }}
.cm-rowlabel {{ font-size:11px; font-weight:500; color:{t['text_secondary']}; text-align:right; align-self:center; padding-right:9px; line-height:1.3; }}
.cm-cell {{ border-radius:9px; aspect-ratio:1 / 0.72; display:flex; flex-direction:column; align-items:center; justify-content:center; transition:transform 0.15s ease; }}
.cm-cell:hover {{ transform:scale(1.03); }}
.cm-num {{ font-size:23px; font-weight:700; line-height:1; }}
.cm-tag {{ font-size:10px; font-weight:600; letter-spacing:0.04em; margin-top:4px; }}
code {{ background:{t['surface_alt']}; padding:2px 6px; border-radius:4px; font-size:13px;
    font-family:'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace; color:{t['text']}; }}
/* Tabs nativos de Streamlit (st.tabs): usar la paleta del proyecto en vez del rojo por defecto */
button[data-baseweb="tab"] {{ color:{t['text_secondary']} !important; }}
button[data-baseweb="tab"]:hover {{ color:{C_PRIMARY} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{C_PRIMARY} !important; }}
[data-baseweb="tab-highlight"] {{ background-color:{C_PRIMARY} !important; }}
[data-baseweb="tab-border"] {{ background-color:{t['border']} !important; }}
/* Scrollbar fino y en paleta (WebKit + Firefox): sustituye la barra gruesa del SO por un acabado sobrio */
*::-webkit-scrollbar {{ width:10px; height:10px; }}
*::-webkit-scrollbar-track {{ background:transparent; }}
*::-webkit-scrollbar-thumb {{ background:{t['border']}; border-radius:8px; border:2px solid {t['bg']}; }}
*::-webkit-scrollbar-thumb:hover {{ background:{C_MID2}; }}
* {{ scrollbar-width:thin; scrollbar-color:{t['border']} transparent; }}
/* Selección de texto en color de marca (en vez del azul del navegador) */
::selection {{ background:{C_PRIMARY}33; color:{t['text']}; }}
/* Foco de teclado visible y consistente (accesibilidad) sin desplazar el layout */
button:focus-visible, [role="slider"]:focus-visible, [data-baseweb="tab"]:focus-visible,
[data-baseweb="select"]:focus-visible, [data-baseweb="select"] div:focus-visible,
a.nav-link:focus-visible, input:focus-visible, [role="combobox"]:focus-visible {{
    outline:2px solid {C_PRIMARY} !important; outline-offset:2px !important; border-radius:8px;
}}
/* Etiquetas de los widgets (slider, selectbox): Streamlit las pinta con SU tema base (claro), que no
   sigue nuestro tema custom — en modo oscuro quedaban casi ilegibles. Las atamos a nuestros tokens. */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{ color:{t['text_secondary']} !important; }}
/* ─── Sliders: estilo «regla» (ticks + carril hundido + pulgar con muesca) ───
   Sin ámbito: aplica a TODOS los sliders de la app (Predictor en Vivo y Esfera de Bloch), así son
   consistentes por construcción y no hay que duplicar selectores por página.
   Notas de implementación (el slider es un componente BaseWeb):
   · El relleno lo pinta BaseWeb como linear-gradient con el % dinámico DENTRO del propio carril,
     así que NO tocamos su background-image (lo romperíamos). Sí fijamos background-color: el tramo
     no relleno es semitransparente, así que el color de debajo lo tiñe hacia la paleta.
   · El pulgar real mide 12x12 y BaseWeb calcula el recorrido con ese tamaño: lo dejamos intacto y
     dibujamos el círculo grande con ::before, así el recorrido sigue siendo exacto.
   · Los ticks usan repeating-linear-gradient con periodos en %, de modo que su número es constante
     y el espaciado se reajusta solo con el ancho y el zoom. */
div[data-baseweb="slider"] {{ padding-top:22px; }}
/* Contenedor del carril: posicionado para colgar de él los ticks */
div[data-baseweb="slider"] > div > div {{ position:relative; }}
/* Ticks: menores cada 2,5 % (cortos) + mayores cada 12,5 % (largos), colgando sobre el carril */
div[data-baseweb="slider"] > div > div::before {{
    content:""; position:absolute; left:0; right:0; bottom:calc(100% + 9px); height:8px;
    pointer-events:none;
    background-image:
        repeating-linear-gradient(to right, {C_MID1} 0 1px, transparent 1px 12.5%),
        repeating-linear-gradient(to right, {t['border']} 0 1px, transparent 1px 2.5%);
    background-size: 100% 100%, 100% 50%;
    background-position: 0 0, 0 0;
    background-repeat: no-repeat, no-repeat;
}}
/* Carril: más grueso, redondeado y hundido */
div[data-baseweb="slider"] > div > div > div:last-child {{
    height:10px !important; border-radius:999px !important;
    background-color:{SLIDER_GROOVE} !important;   /* tiñe el tramo no relleno hacia la paleta */
    position:relative; z-index:0;
    box-shadow: inset 0 1px 3px rgba(20,30,40,0.20), inset 0 -1px 0 rgba(255,255,255,0.45);
}}
/* Degradado del relleno + brillo, sin tocar el background-image dinámico de BaseWeb.
   Truco: un blanco que se desvanece hacia la derecha. Sobre el tramo RELLENO (izquierda) aclara el
   azul → produce el degradado claro→oscuro hasta el pulgar, como en la referencia. Sobre el tramo
   VACÍO (derecha) el blanco ya es casi transparente → lo deja limpio. Se adapta solo al mover el
   pulgar, porque el degradado va referido al ancho del carril, no al del relleno. */
div[data-baseweb="slider"] > div > div > div:last-child::after {{
    content:""; position:absolute; inset:0; border-radius:999px; pointer-events:none;
    background:
        linear-gradient(to bottom, rgba(255,255,255,0.22), rgba(255,255,255,0) 60%, rgba(20,30,40,0.10)),
        linear-gradient(to right, rgba(255,255,255,0.34) 0%, rgba(255,255,255,0.07) 55%, rgba(255,255,255,0) 80%);
}}
/* Pulgar: se mantiene 12x12 (geometría de BaseWeb); el círculo visible es el ::before.
   margin-top recentra el pulgar sobre el carril de 10px (BaseWeb lo centraba sobre uno de 4px). */
div[role="slider"] {{
    background:transparent !important; box-shadow:none !important;
    margin-top:3px; z-index:3 !important;
}}
div[role="slider"]::before {{
    content:""; position:absolute; left:50%; top:50%;
    width:22px; height:22px; transform:translate(-50%, -50%); border-radius:50%;
    background:linear-gradient(180deg, {t['surface']} 0%, {t['surface_alt']} 100%);
    border:1px solid {C_MID2};
    box-shadow: 0 1px 3px rgba(20,30,40,0.30), 0 3px 8px rgba(20,30,40,0.16);
}}
/* Muesca central del pulgar (la rayita vertical de la referencia), en color de marca */
div[role="slider"]::after {{
    content:""; position:absolute; left:50%; top:50%;
    width:2px; height:9px; transform:translate(-50%, -50%);
    border-radius:1px; background:{C_PRIMARY};
}}
div[role="slider"]:hover::before {{ border-color:{C_PRIMARY}; }}
div[role="slider"]:active {{ cursor:grabbing !important; }}
/* El valor sobre el pulgar sube para no chocar con los ticks */
div[data-testid="stSliderThumbValue"] {{ top:-40px; z-index:4; }}
#MainMenu, footer, header {{ visibility:hidden; }}
/* ...pero el botón nativo para ABRIR la sidebar vive DENTRO de ese <header>: al ocultarlo, en móvil
   (donde la sidebar arranca colapsada) el usuario se quedaba sin forma de abrir el menú y no podía
   navegar. Lo devolvemos a la vida y lo vestimos con la paleta. */
/* Ojo: el elemento con este testid ES el <button>, no lo contiene. */
button[data-testid="stExpandSidebarButton"] {{
    visibility:visible !important;
    background-color:{t['surface']} !important;
    border:1px solid {t['border']} !important;
    border-radius:10px !important;
    color:{t['text']} !important;
    box-shadow: 0 2px 6px rgba(20,30,40,0.10), 0 1px 3px rgba(20,30,40,0.08) !important;
}}
button[data-testid="stExpandSidebarButton"]:hover {{
    background-color:{t['surface_alt']} !important; border-color:{C_PRIMARY} !important;
}}

/* ═══════════════ TABLET (≤ 1024 px) ═══════════════
   Entre el ancho de escritorio y el punto de quiebre móvil (768 px) hay una franja —tablets en
   horizontal, ventanas a media pantalla— donde la sidebar fija (270 px) ya no deja tanto aire al
   contenido. Aquí solo se afina la proporción (menos padding, rejillas de 3 columnas a 2) sin
   activar aún el modo overlay de la sidebar, que solo tiene sentido en pantallas realmente estrechas. */
@media (max-width: 1024px) {{
    div[data-testid="stMainBlockContainer"], section.main > div.block-container {{
        padding-left:1.25rem !important; padding-right:1.25rem !important;
    }}
    .compare-grid {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }}
}}

/* ═══════════════ MÓVIL (≤ 768 px) ═══════════════
   Sin esto la app era inusable en el teléfono: la sidebar (270 px fijos, con su botón de colapso
   oculto) se comía dos tercios de la pantalla y empujaba el contenido fuera. Aquí devolvemos a la
   sidebar su comportamiento nativo de overlay y apilamos las rejillas. */
@media (max-width: 768px) {{
    /* Sidebar como panel superpuesto, no como columna fija que roba ancho */
    section[data-testid="stSidebar"] {{
        width:min(84vw, 320px) !important;
        min-width:min(84vw, 320px) !important;
        max-width:min(84vw, 320px) !important;
    }}
    /* Streamlit esconde la sidebar con translateX(-300px) — el ancho inline que ELLA calcula. Como
       nosotros le forzamos otro ancho, sobresalía un pico de 20 px que tapaba la primera letra de
       cada línea. Con -100% se esconde entera, sea cual sea el ancho que le demos. */
    section[data-testid="stSidebar"][aria-expanded="false"] {{
        transform:translateX(-100%) !important;
    }}
    /* Devolvemos el botón de cerrar: en escritorio lo ocultamos porque usamos el toggle propio,
       pero en móvil es la única forma de cerrar el overlay. Y le devolvemos altura a la cabecera de
       la sidebar (que colapsamos a 0 para pegar el logo arriba): sin ella el botón quedaba en y=-14,
       medio fuera de la pantalla e imposible de pulsar. */
    [data-testid="stSidebarCollapseButton"] {{ display:block !important; }}
    [data-testid="stSidebarHeader"] {{
        height:auto !important; min-height:46px !important; padding:8px 10px 0 !important;
    }}
    /* Nuestro toggle circular va anclado a left:270px — en móvil flotaría sobre el contenido */
    .st-key-toggle_sidebar {{ display:none !important; }}
    /* La cápsula de tema y el footer son position:fixed anclados al ancho de la sidebar: si siguen
       fijos, quedan flotando sobre el contenido cuando la sidebar está cerrada. Los devolvemos al
       flujo de la sidebar, así solo se ven cuando el panel está abierto. */
    .st-key-theme_toggle {{
        position:static !important; width:100% !important; margin:18px 0 8px !important;
    }}
    .sidebar-footer {{
        position:static !important; width:100% !important; margin-top:10px; border-top:none;
    }}
    /* Contenido: menos padding lateral y sin el tope de ancho */
    div[data-testid="stMainBlockContainer"], section.main > div.block-container {{
        max-width:100% !important; padding-top:2.5rem !important;
        padding-left:1rem !important; padding-right:1rem !important;
    }}
    /* Columnas apiladas: en móvil Streamlit las mantiene lado a lado y quedan ilegibles */
    div[data-testid="stHorizontalBlock"] {{ flex-wrap:wrap !important; }}
    div[data-testid="stColumn"] {{ flex:1 1 100% !important; min-width:100% !important; }}
    /* Rejilla de tarjetas comparativas: una por fila */
    .compare-grid {{ grid-template-columns:1fr !important; }}
    /* Al apilarse (teléfono) la columna es de ancho completo; el alto ya lo fija la figura Plotly */
    .st-key-bloch_row div[data-testid="stColumn"]:last-of-type div[data-testid="stVerticalBlock"] {{ height:auto !important; }}
    /* Cabecera proporcionada a la pantalla del teléfono */
    .page-title {{ font-size:26px; margin-bottom:10px; }}
    .page-subtitle {{ font-size:15px; margin-bottom:20px; }}
    .kpi-card, .info-card {{ padding:14px 15px; }}
    .stat-card {{ min-height:84px !important; }}
    /* Matriz de confusión: la columna de etiquetas fija en 64 px ahoga las celdas en pantalla estrecha */
    .cm-grid {{ grid-template-columns:52px 1fr 1fr; }}
    .cm-num {{ font-size:19px; }}
    /* El resto de la jerarquía tipográfica también baja un escalón en móvil, en la misma proporción
       que el título/subtítulo de arriba — así todo el texto queda a escala del viewport, no solo la
       cabecera de la página. */
    .section-title {{ font-size:16px; }}
    .section-sub, .clinical-note, .cm-title {{ font-size:13px; }}
    .kpi-value {{ font-size:15px; }}
    .kpi-model, .kpi-label, .stat-label {{ font-size:12px; }}
    .badge {{ font-size:11px; }}
}}

/* ═══════════════ TRANSICIÓN DE PÁGINA ═══════════════
   El contenedor .st-key-page_enter_N (ver header(), key = índice de la página en _MENU_OPTIONS) es
   el único elemento con un remount GARANTIZADO al navegar: key distinta = componente distinto para
   Streamlit, así que se desmonta y se vuelve a montar de verdad. El resto del contenido normalmente
   reutiliza sus nodos entre reruns (para no perder el estado de sliders/botones) y una animación
   "al aparecer" sobre ellos nunca llegaría a dispararse — por eso no se ve nada si se aplica ahí. */
@media (prefers-reduced-motion: no-preference) {{
    div[class*="st-key-page_enter_"] {{
        animation: pageFadeIn 0.38s ease-out;
    }}
}}
@keyframes pageFadeIn {{
    from {{ opacity:0; transform:translateY(10px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
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
             "accuracy": 0.8602, "mcc": 0.0625, "cm": {"tn": 1347, "fp": 0, "fn": 219, "tp": 1}},
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
# step + fmt: por defecto st.slider con límites float usa un paso (max-min)/100 irregular (p. ej. 0,62
# años) y muestra el valor con dos decimales ("45.00"). Fijamos pasos clínicamente naturales —0,1 para
# magnitudes con decimal (HbA1c, IMC), 1 para las enteras— y su formato, para que el valor se lea
# limpio ("45", "5,7") y la interacción sea coherente con cómo se miden esas variables.
QSVM_FEATURES = {
    "LBXGH":    {"label": "HbA1c",               "unit": "%",       "range": (4.0, 15.0),  "default": 5.7,  "importance": 0.2452, "step": 0.1, "fmt": "%.1f"},
    "LBXGLU":   {"label": "Glucosa en ayunas",    "unit": "mg/dL",   "range": (50, 300),     "default": 100,  "importance": 0.1853, "step": 1.0, "fmt": "%d"},
    "RIDAGEYR": {"label": "Edad",                 "unit": "años",    "range": (18, 80),      "default": 45,   "importance": 0.0325, "step": 1.0, "fmt": "%d"},
    "LBDLDL":   {"label": "Colesterol LDL",       "unit": "mg/dL",   "range": (40, 250),     "default": 110,  "importance": 0.0315, "step": 1.0, "fmt": "%d"},
    "BMXWAIST": {"label": "Circunf. cintura",     "unit": "cm",      "range": (60, 150),     "default": 95,   "importance": 0.0284, "step": 1.0, "fmt": "%d"},
    "LBXIN":    {"label": "Insulina",             "unit": "µU/mL",   "range": (2, 60),       "default": 10,   "importance": 0.0264, "step": 1.0, "fmt": "%d"},
    "BMXLEG":   {"label": "Long. pierna",         "unit": "cm",      "range": (30, 50),      "default": 40,   "importance": 0.0226, "step": 1.0, "fmt": "%d"},
    "BMXBMI":   {"label": "IMC",                  "unit": "kg/m²",   "range": (15, 60),      "default": 27,   "importance": 0.0221, "step": 0.1, "fmt": "%.1f"},
}

# Ranking de importancia RF para el gráfico "8 features seleccionadas" (Circuito Cuántico). Versión
# actualizada sin variables DIQ: BMXBMI sustituye a WTINT2YR en la posición 8. Independiente de
# QSVM_FEATURES a propósito — ese dict alimenta Esfera de Bloch / Predictor en Vivo y no cambia.
RF_TOP8_IMPORTANCE = {
    "LBXGH": 0.2454, "LBXGLU": 0.1855, "RIDAGEYR": 0.0323, "LBDLDL": 0.0318,
    "BMXWAIST": 0.0283, "LBXIN": 0.0265, "BMXLEG": 0.0225, "BMXBMI": 0.0221,
}

# ─────────────────────────────────────────────────────────────────────────
# INFERENCIA ONNX REAL (con fallback seguro al sustituto SHAP existente)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_onnx_session(filename: str):
    path = MODELS_DIR / filename
    if ONNX_AVAILABLE and path.exists():
        return ort.InferenceSession(str(path))
    return None


@st.cache_data
def _load_scaler_and_medians():
    scaler_path = MODELS_DIR / "scaler_correcto.json"
    medians_path = MODELS_DIR / "medianas_correctas.json"
    if not (scaler_path.exists() and medians_path.exists()):
        return None
    scaler = json.loads(scaler_path.read_text())
    medians = json.loads(medians_path.read_text())
    return {
        "features": scaler["features"],
        "mean": np.array(scaler["mean"]),
        "scale": np.array(scaler["scale"]),
        "medians": medians,
    }


@st.cache_data
def _load_roc_scores(prefix: str):
    """Scores + etiquetas reales del test para la curva ROC de un modelo. Devuelve
    (scores, y_true) o (None, None) si faltan. Las etiquetas del test son las mismas
    para los tres modelos (mismo split estratificado): si un modelo no trae su propio
    <prefix>_y_test.npy, se reutiliza el del QSVM. Verificado: los AUC resultantes
    coinciden exactamente con los del TFM (LightGBM 0,9485 · SVM-RBF 0,9377)."""
    scores_path = MODELS_DIR / f"{prefix}_y_scores.npy"
    if not scores_path.exists():
        return None, None
    test_path = MODELS_DIR / f"{prefix}_y_test.npy"
    if not test_path.exists():
        test_path = MODELS_DIR / "qsvm_y_test.npy"   # etiquetas compartidas del test
    if not test_path.exists():
        return None, None
    scores, y_true = np.load(scores_path), np.load(test_path)
    if len(scores) != len(y_true):   # desalineados: no dibujar algo incorrecto
        return None, None
    return scores, y_true


def _build_feature_vector(sp: dict, overrides: dict) -> np.ndarray:
    """Construye el vector de 89 features: medianas del conjunto de
    entrenamiento + las variables clínicas del Live Predictor sobrescritas
    en su posición exacta."""
    feats = sp["features"]
    x = np.array([sp["medians"][f] for f in feats], dtype=np.float64)
    for k, v in overrides.items():
        if k in feats:
            x[feats.index(k)] = v
    return x


def predict_real(overrides: dict):
    """Devuelve (prob_lgbm, prob_svm), o None si los modelos ONNX no están
    disponibles (activa el fallback al sustituto por reglas)."""
    sp = _load_scaler_and_medians()
    sess_lgbm = _load_onnx_session("lgbm_final.onnx")
    sess_svm = _load_onnx_session("svm_final.onnx")
    if sp is None or sess_lgbm is None or sess_svm is None:
        return None

    x_raw = _build_feature_vector(sp, overrides)

    # LightGBM: datos CRUDOS, sin escalar (se entrenó así — verificado)
    out_lgbm = sess_lgbm.run(None, {"float_input": x_raw.reshape(1, -1).astype(np.float32)})
    prob_lgbm = float(out_lgbm[1][0][1])

    # SVM-RBF: escalado con el scaler recuperado (verificado AUC=0.9377 exacto)
    x_scaled = ((x_raw - sp["mean"]) / sp["scale"]).reshape(1, -1).astype(np.float32)
    out_svm = sess_svm.run(None, {"float_input": x_scaled})
    prob_svm = float(out_svm[1][0][1])

    return prob_lgbm, prob_svm


def compute_roc_empirical(y_true: np.ndarray, y_scores: np.ndarray):
    """Curva ROC real a partir de scores crudos, sin depender de scikit-learn."""
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    n_pos, n_neg = tps[-1], fps[-1]
    tpr = tps / n_pos if n_pos > 0 else tps.astype(float)
    fpr = fps / n_neg if n_neg > 0 else fps.astype(float)
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])

# Descripción breve de cada variable NHANES (para los tooltips de las gráficas SHAP). Basado en el
# Anexo C del TFM (Diccionario de Variables NHANES Utilizadas en el Pipeline).
VAR_DESC = {
    "LBXGH":      "Hemoglobina glicosilada (HbA1c): glucosa media de los últimos 2-3 meses. Marcador diagnóstico primario de diabetes (ADA: ≥ 6,5 %).",
    "RIDAGEYR":   "Edad del participante en el momento de la exploración (años).",
    "LBXGLU":     "Glucosa en plasma en ayunas: marcador bioquímico del control glucémico (mg/dL).",
    "LBDLDL":     "Colesterol LDL calculado: fracción del colesterol ligada a riesgo cardiovascular (mg/dL).",
    "BMXWAIST":   "Circunferencia de cintura: adiposidad abdominal asociada a resistencia a la insulina (cm).",
    "WTINT2YR":   "Factor de expansión muestral de la entrevista NHANES. Artefacto del diseño muestral, no una variable clínica.",
    "BMXARML":    "Longitud del brazo (acromion → olécranon): medida antropométrica (cm).",
    "BMXLEG":     "Longitud máxima de la pierna (rodilla → suelo): medida antropométrica (cm).",
    "BMXBMI":     "Índice de Masa Corporal (peso/talla²): adiposidad corporal global (kg/m²).",
    "PAD680":     "Minutos de actividad sedentaria al día (tiempo sentado o recostado).",
    "PAD645":     "Minutos semanales de actividad física moderada (trabajo + recreación).",
    "PAQ640":     "Días por semana con actividades de fortalecimiento muscular.",
    "BMXWT":      "Peso corporal total (kg).",
    "LBXIN":      "Insulina sérica en ayunas: marcador de resistencia a la insulina (µU/mL).",
    "INDHHIN2":   "Nivel de ingresos del hogar (variable socioeconómica categórica).",
    "DMDYRSUS":   "Número de años de residencia en Estados Unidos.",
    "BMXARMC":    "Circunferencia media del brazo: medida antropométrica (cm).",
    "PAQ670":     "Minutos semanales de actividad recreativa vigorosa.",
    "BPXSY1":     "Presión arterial sistólica, primera medición (mmHg).",
    "PAD630":     "Minutos semanales de actividad física moderada de recreación.",
    "DMDHHSZE":   "Composición del hogar: número de niños en el hogar.",
    "BPXDI1":     "Presión arterial diastólica, primera medición (mmHg).",
    "LBXTR":      "Triglicéridos séricos: marcador del perfil lipídico (mg/dL).",
    "DMDMARTL_1": "Estado civil = casado (variable dummy tras one-hot encoding).",
    "DMDMARTL_5": "Estado civil = nunca casado (variable dummy tras one-hot encoding).",
    "BPXPLS":     "Pulso: frecuencia cardíaca en reposo (latidos/min).",
    "DMDEDUC2_3": "Nivel educativo intermedio (bachillerato/GED): variable dummy tras one-hot encoding.",
    "SDMVSTRA":   "Estrato de varianza del diseño muestral NHANES (variable metodológica, no clínica).",
    "DMDMARTL_2": "Estado civil = viudo (variable dummy tras one-hot encoding).",
    "DMDHHSZB":   "Composición del hogar: número de adultos en el hogar.",
}

def _wrap_hover(text, width=54):
    return "<br>".join(textwrap.wrap(text, width=width))

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    _qml_logo_b64 = _b64_image(str(ASSETS_DIR / "qml_logov2-sidebar.png"))
    _logo_h = "40px" if narrow else "64px"
    st.markdown(f"""
    <div style="display:flex;justify-content:center;align-items:center;padding:16px 0;margin-bottom:8px;border-bottom:1px solid {t['border']};">
        <img src="data:image/png;base64,{_qml_logo_b64}" style="height:{_logo_h};width:auto;display:block;" alt="QML DataOps">
    </div>
    """, unsafe_allow_html=True)

    _MENU_OPTIONS = ["Resumen", "Resultados", "Análisis SHAP", "Circuito Cuántico", "Esfera de Bloch", "Predictor en Vivo"]
    if "page" not in st.session_state:
        st.session_state.page = _MENU_OPTIONS[0]

    if narrow:
        if st.button("›", key="toggle_sidebar", help="Expandir"):
            st.session_state.sidebar_narrow = False
            st.session_state.menu_force_index = _MENU_OPTIONS.index(st.session_state.page)
            st.rerun()
    else:
        if st.button("‹", key="toggle_sidebar", help="Colapsar"):
            st.session_state.sidebar_narrow = True
            st.session_state.menu_force_index = _MENU_OPTIONS.index(st.session_state.page)
            st.rerun()

    # streamlit-option-menu renderiza dentro de un iframe: el CSS del documento principal
    # (st.markdown) no puede alcanzar sus elementos internos. Por eso el modo narrow se logra
    # aquí, vía el dict "styles" que sí viaja al componente, en vez de con CSS externo.
    nav_link_style = {"font-size": "13.5px", "text-align": "left", "margin": "2px 0", "padding": "10px 8px",
                       "border-radius": "10px", "color": t["text_secondary"], "font-weight": "400",
                       "border-left": "0px solid transparent", "transition": "all 0.12s ease",
                       "--hover-color": t["sidebar_active"]}
    nav_link_selected_style = {"background-color": t["sidebar_active"], "color": t["text"], "font-weight": "600",
                                "border-left": "0px solid transparent", "border-radius": "10px"}
    if narrow:
        nav_link_style.update({"font-size": "0px", "text-align": "center", "padding": "12px 0"})
        nav_link_selected_style.update({"font-size": "0px", "text-align": "center", "padding": "12px 0"})

    # manual_select fuerza al componente a saltar a un indice concreto, pero es un disparo "de un
    # solo uso": si se reenvia en cada rerun (incluido el propio rerun que dispara un clic del
    # usuario en el menu) compite con ese clic y el menu queda oscilando sin parar entre la pestaña
    # vieja y la nueva hasta que se refresca la pagina. Por eso solo se rellena explicitamente justo
    # antes de un st.rerun() disparado por OTRO widget (toggle de sidebar, toggle de tema) y se
    # consume aqui con pop() para que en el resto de reruns (incluidos los clics normales) viaje
    # como None y no interfiera.
    _forced_index = st.session_state.pop("menu_force_index", None)
    page = option_menu(
        menu_title=None,
        options=_MENU_OPTIONS,
        icons=["house", "bar-chart", "diagram-3", "cpu", "globe", "sliders"],
        default_index=_MENU_OPTIONS.index(st.session_state.page),
        manual_select=_forced_index,
        # La key incluye el tema Y el modo narrow a proposito: option_menu vive en un iframe con
        # estado JS propio (Vue) que solo LEE el dict "styles" al montarse — en reruns posteriores con
        # la MISMA key, los cambios en "styles" (colores de tema, o el font-size:0 del modo narrow) no
        # se reaplican de verdad. Con key fija, alternar sidebar_narrow dejaba el menu con el texto
        # aun visible (a tamaño completo) junto al icono, porque el componente seguia usando los
        # estilos con los que se monto la primera vez. Cambiar la key en cada toggle (tema Y narrow)
        # fuerza un remount completo, que si levanta los estilos frescos. default_index/manual_select
        # ya se encargan de que ese remount respete la pestaña activa.
        key=f"main_menu_{st.session_state.theme}_{narrow}",
        styles={
            "container": {"padding": "0", "background-color": t["sidebar_bg"], "border-radius": "0"},
            "icon": {"font-size": "15px", "color": t["text_secondary"]},
            "nav-link": nav_link_style,
            "nav-link-selected": nav_link_selected_style,
        },
    )
    st.session_state.page = page
    # Refuerzo del hover (streamlit-option-menu no expone :hover directo en icon/color de texto)
    st.markdown(f"""
    <style>
    nav[role="navigation"] a.nav-link:hover:not(.active) {{
        background-color:{t['sidebar_active']} !important;
        color:{t['text']} !important;
    }}
    nav[role="navigation"] a.nav-link:hover:not(.active) span {{
        color:{t['text']} !important;
    }}
    nav[role="navigation"] a.nav-link.active i {{
        color:{C_PRIMARY} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    if st.button(" ", key="theme_toggle", help="Cambiar a tema oscuro" if st.session_state.theme == "light" else "Cambiar a tema claro"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

    if narrow:
        _footer_html = ('<div class="footer-name">JAC</div>'
                         '<div class="footer-uni">UEV</div>')
    else:
        _footer_html = ('<div class="footer-name">Juan Albornoz C. · TFM 2026</div>'
                         '<div class="footer-uni">Universidad Europea de Valencia</div>')
    st.markdown(f'<div class="sidebar-footer">{_footer_html}</div>', unsafe_allow_html=True)

def header(eyebrow, title, subtitle):
    # Streamlit reutiliza los mismos nodos del DOM entre reruns (para no perder el estado de sliders/
    # botones), así que una animación CSS "al aparecer" sobre stElementContainer nunca llegaba a
    # dispararse de verdad al cambiar de página — el nodo nunca se desmontaba. Un contenedor con key
    # dependiente de la página SÍ fuerza un remount real cada vez que cambia `page` (key distinta =
    # componente distinto para Streamlit), y NO se repite en reruns dentro de la misma página (mover
    # un slider, alternar el tema) porque ahí la key no cambia y el contenedor se reutiliza tal cual.
    _page_idx = _MENU_OPTIONS.index(page) if page in _MENU_OPTIONS else 0
    with st.container(key=f"page_enter_{_page_idx}"):
        st.markdown(f'<div class="page-title">{title}: <span style="color:{t["text_secondary"]}; font-weight:400;">{eyebrow}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

def nf(x, dec=4):
    """Formato numérico español: coma decimal, punto de millar (consistente con la prosa del TFM)."""
    s = f"{x:,.{dec}f}"                       # formato US: '1,234.5678'
    return s.translate(str.maketrans({",": ".", ".": ","}))  # intercambia separadores en una pasada

def pct(x, dec=1):
    """Porcentaje en formato español: '78,2 %' (x en [0,1])."""
    return f"{nf(x * 100, dec)} %"

def plotly_layout(fig, height=300, **kwargs):
    margin = kwargs.pop("margin", dict(l=40, r=16, t=30, b=36))
    fig.update_layout(
        height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=t["text_secondary"], size=12),
        separators=",.",  # localización ES: coma decimal / punto de millar en ticks y hover
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border"], align="left",
                         font=dict(family="Inter", size=12, color=t["text"])),
        margin=margin, **kwargs,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "Resumen":
    header("Framework DataOps + QML", "Resumen",
           "Pipeline end-to-end sobre Databricks CE + AWS S3, con QSVM cuántico frente a dos baselines "
           "clásicos, validado sobre datos clínicos reales del estudio NHANES (CDC).")

    st.markdown(f"""
    <div class="info-card lead-card" style="margin-bottom:20px;">
    <p style="font-size:16px; color:{t['text_secondary']}; line-height:1.7; margin:0;">
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
            st.markdown(f'<div class="info-card stat-card"><div class="stat-num">{num}</div><div class="stat-label">{lab}</div></div>', unsafe_allow_html=True)

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
            <div class="medallion-item" style="border-left-color:{color};">
                <div style="width:10px;height:10px;border-radius:3px;background:{color};margin-top:5px;flex-shrink:0;"></div>
                <div>
                    <div style="font-size:14px;font-weight:600;color:{t['text']};">{name}</div>
                    <div style="font-size:13px;color:{t['text_secondary']};line-height:1.55;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Distribución variable objetivo (DIQ010)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Target binarizado: 1 = diabetes diagnosticada, 0 = resto</div>', unsafe_allow_html=True)
        pie_labels, pie_values = ["No diabetes", "Diabetes"], [86, 14]
        fig = go.Figure()
        # Sombra sutil y uniforme (misma silueta del donut, desplazada solo un poco hacia abajo)
        # para dar una sensación de elevación/profundidad discreta, sin separar los segmentos
        # ni desplazarlos en diagonal (lo que se veía forzado/poco natural).
        fig.add_trace(go.Pie(
            labels=pie_labels, values=pie_values, hole=0.62,
            marker=dict(colors=[hex_to_rgba(t["text"], 0.14), hex_to_rgba(t["text"], 0.14)]),
            textinfo="none", hoverinfo="skip", sort=False, showlegend=False,
            domain=dict(x=[0.0, 1.0], y=[0.0, 0.965]),
        ))
        fig.add_trace(go.Pie(
            labels=pie_labels, values=pie_values, hole=0.62,
            marker=dict(colors=[C_LIGHT, C_PRIMARY], line=dict(color=t["surface"], width=2)),
            textinfo="label+percent", textposition="outside",
            textfont=dict(size=12, family="Inter", color=t["text"]),
            insidetextorientation="horizontal", sort=False, automargin=True,
            hoverinfo="skip",
            domain=dict(x=[0.0, 1.0], y=[0.035, 1.0]),
        ))
        plotly_layout(fig, height=300, showlegend=False, margin=dict(l=30, r=30, t=45, b=45))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="section-title" style="margin-top:6px;">Comparativa triangulada — objetivo del experimento</div>', unsafe_allow_html=True)
    labels3 = [("LightGBM", "Baseline tabular de referencia", C_PRIMARY),
               ("SVM-RBF", "Puente estructural hacia el componente cuántico", C_DARK),
               ("QSVM", "FidelityQuantumKernel — mismo clasificador, kernel cuántico", C_MID1)]
    # HTML sin saltos ni indentación: Streamlit trataría las líneas con 4+ espacios como bloque de código.
    _compare_cards = "".join(
        f'<div class="info-card" style="border-top:3px solid {color};">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{color};flex-shrink:0;"></span>'
        f'<span style="font-size:14px;font-weight:600;color:{t["text"]};">{name}</span></div>'
        f'<div style="font-size:13px;color:{t["text_secondary"]};line-height:1.55;">{desc}</div></div>'
        for name, desc, color in labels3
    )
    st.markdown(f'<div class="compare-grid">{_compare_cards}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Resultados":
    header("Comparativa triangulada", "Resultados",
           "LightGBM vs. SVM-RBF vs. QSVM sobre el mismo conjunto de test (1.567 instancias).")

    cols = st.columns(3)
    for col, key in zip(cols, MODEL_ORDER):
        m = MODELS[key]
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {m['color']};">
                <div class="kpi-model"><span class="kpi-dot" style="background:{m['color']}"></span>{m['label']}</div>
                <div class="kpi-value-auc" style="color:{m['color']};">{nf(m['auc'])}</div>
                <div class="kpi-label" style="margin-bottom:10px;">AUC-ROC</div>
                <div class="kpi-row"><span class="kpi-label">F1-macro</span><span class="kpi-value">{nf(m['f1_macro'])}</span></div>
                <div class="kpi-row"><span class="kpi-label">Accuracy</span><span class="kpi-value">{nf(m['accuracy'])}</span></div>
                <div class="kpi-row"><span class="kpi-label">MCC</span><span class="kpi-value">{nf(m['mcc'])}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Curvas ROC</div>', unsafe_allow_html=True)

    def roc_curve_for_auc(auc, n=200):
        a = (1.0 / auc) - 1.0
        x = np.linspace(0, 1, n)
        y = np.power(x, a) if a > 0 else np.ones_like(x)
        y[0], y[-1] = 0.0, 1.0
        return x, y

    # Prefijo de archivo de scores por modelo (las keys de MODELS difieren de los nombres de fichero).
    ROC_PREFIX = {"lightgbm": "lgbm", "svm_rbf": "svm", "qsvm": "qsvm"}
    # Cargar scores reales de cada modelo; True si TODOS son empíricos (para el subtítulo).
    roc_data = {k: _load_roc_scores(ROC_PREFIX[k]) for k in MODEL_ORDER}
    all_real = all(sc is not None for sc, _ in roc_data.values())
    st.markdown(
        f'<div class="section-sub">{"Curvas empíricas reales, punto a punto sobre las 1.567 instancias del test (mismos scores que reportan el AUC del TFM)." if all_real else "AUC exacto · forma reconstruida a partir del AUC donde no hay scores por instancia."}</div>',
        unsafe_allow_html=True)

    roc_cols = st.columns(3)
    for col, key in zip(roc_cols, MODEL_ORDER):
        m = MODELS[key]
        scores, y_true = roc_data[key]
        if scores is not None:
            x, y = compute_roc_empirical(y_true, scores)
        else:
            x, y = roc_curve_for_auc(m["auc"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=t["border"], width=1.5, dash="dash"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=m["color"], width=2.5), fill="tozeroy",
                                  fillcolor=hex_to_rgba(m["color"], 0.18), name=m["label"],
                                  hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
        plotly_layout(fig, height=250, showlegend=False,
                      title=dict(text=f"{m['label']} · AUC {nf(m['auc'])}", font=dict(size=14, color=t["text"])),
                      xaxis=dict(title="FPR", range=[0, 1], showgrid=False, zeroline=False, tickfont=dict(size=11), fixedrange=True),
                      yaxis=dict(title="TPR", range=[0, 1], showgrid=True, gridcolor=t["border"], zeroline=False, tickfont=dict(size=11), fixedrange=True))
        with col:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matrices de confusión</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Valores exactos verificados contra el classification report de cada modelo</div>', unsafe_allow_html=True)

    def cm_cell(val, total, tag, color):
        prop = val / total if total else 0.0
        bg = hex_to_rgba(color, 0.12 + 0.80 * prop)
        strong = prop > 0.55
        txt = "#FFFFFF" if strong else t["text"]
        sub = "rgba(255,255,255,0.85)" if strong else t["text_secondary"]
        return (f'<div class="cm-cell" style="background:{bg};color:{txt};">'
                f'<div class="cm-num">{val}</div>'
                f'<div class="cm-tag" style="color:{sub};">{tag}</div></div>')

    cm_cols = st.columns(3)
    for col, key in zip(cm_cols, MODEL_ORDER):
        m = MODELS[key]
        cm = m["cm"]
        r0, r1 = cm["tn"] + cm["fp"], cm["fn"] + cm["tp"]
        with col:
            st.markdown(f"""
            <div class="info-card">
              <div class="cm-title"><span class="kpi-dot" style="background:{m['color']};"></span>{m['label']}</div>
              <div class="cm-grid">
                <div></div>
                <div class="cm-collabel">Pred.<br>No diabetes</div>
                <div class="cm-collabel">Pred.<br>Diabetes</div>
                <div class="cm-rowlabel">Real<br>No diab.</div>
                {cm_cell(cm['tn'], r0, "VN", m['color'])}
                {cm_cell(cm['fp'], r0, "FP", m['color'])}
                <div class="cm-rowlabel">Real<br>Diabetes</div>
                {cm_cell(cm['fn'], r1, "FN", m['color'])}
                {cm_cell(cm['tp'], r1, "VP", m['color'])}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Comparativa de métricas</div>', unsafe_allow_html=True)
    fig = go.Figure()
    metric_keys, metric_labels = ["auc", "f1_macro", "accuracy", "mcc"], ["AUC-ROC", "F1-macro", "Accuracy", "MCC"]
    metric_desc = {
        "auc": "Área bajo la curva ROC: capacidad de separar diabetes vs. no-diabetes. 0,5 = azar, 1 = perfecto.",
        "f1_macro": "Media armónica de precisión y recall promediada por clase (sin ponderar). Penaliza el desbalance.",
        "accuracy": "Proporción de aciertos totales. Con clases desbalanceadas puede reflejar solo la clase mayoritaria.",
        "mcc": "Coef. de correlación de Matthews: calidad global robusta al desbalance. 0 = azar, 1 = perfecto.",
    }
    for key in MODEL_ORDER:
        m = MODELS[key]
        fig.add_trace(go.Bar(name=m["label"], x=metric_labels, y=[m[k] for k in metric_keys], marker_color=m["color"],
                              text=[nf(m[k], 3) for k in metric_keys], textposition="outside", textfont=dict(size=14),
                              customdata=[[nf(m[k], 3), _wrap_hover(metric_desc[k])] for k in metric_keys],
                              hovertemplate="<b>%{fullData.name}</b> · %{x} = %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    plotly_layout(fig, height=460, barmode="group",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=13, color=t["text"])),
                  yaxis=dict(range=[0, 1.08], showgrid=True, gridcolor=t["border"], zeroline=False, fixedrange=True),
                  xaxis=dict(showgrid=False, tickfont=dict(size=13, color=t["text"]), fixedrange=True))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(f"""
    <div class="clinical-note" style="margin-top:6px;">
    <b>Nota sobre el experimento QSVM.</b> El QSVM se entrenó sobre una muestra estratificada de 500 instancias
    (coste O(n²) del kernel cuántico) y se evaluó sobre las 1.567 del test completo. AUC-ROC = 0,5493 indica que el
    modelo apenas supera la clasificación aleatoria — Recall ≈ 0 para la clase diabetes (1 de 220), Accuracy = 0,8602 refleja
    solo la proporción de la clase mayoritaria. El MCC ≈ 0 confirma ausencia de capacidad predictiva real.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 3 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Análisis SHAP":
    header("Interpretabilidad", "Análisis SHAP",
           "Importancia global de variables — TreeExplainer (LightGBM) vs. KernelExplainer (SVM-RBF).")

    tab1, tab2 = st.tabs(["LightGBM · TreeExplainer", "SVM-RBF · KernelExplainer"])

    def shap_chart(data, color, sample_note):
        rev = list(reversed(data))
        names = [code for code, label, _ in rev]
        values = [v for _, _, v in rev]
        customdata = [[code, label, _wrap_hover(VAR_DESC.get(code, label))] for code, label, _ in rev]
        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h", marker_color=color,
            text=[nf(v) for v in values], textposition="outside", textfont=dict(size=11),
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        ))
        plotly_layout(fig, height=520, hovermode="y",
                      xaxis=dict(title="mean(|SHAP value|)", showgrid=True, gridcolor=t["border"], range=[0, max(values) * 1.3], fixedrange=True),
                      yaxis=dict(tickfont=dict(size=12, color=t["text"]), fixedrange=True), margin=dict(l=170, r=60, t=20, b=40))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(f'<div class="section-sub">💡 Pasa el cursor sobre cada barra (fila) para ver el significado de la variable. {sample_note}</div>', unsafe_allow_html=True)

    def shap_summary_image(filename, title, caption):
        # Los SHAP summary plot (beeswarm) requieren los valores SHAP por instancia, que no están
        # en los datos de la app: se incrustan las figuras originales del TFM desde figures/.
        # Se renderiza solo si el PNG existe, así el bloque del SVM aparece en cuanto se añada su figura.
        path = FIGURES_DIR / filename
        if not path.exists():
            return
        b64 = _b64_image(str(path))
        st.markdown(f'<div class="section-title" style="margin-top:24px;">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{caption}</div>', unsafe_allow_html=True)
        # Acotado y centrado (max-width) igual que el circuito, en vez de estirarse a todo el ancho:
        # el beeswarm es una figura de proporción casi cuadrada; a ancho completo dominaba la página y
        # rompía la coherencia con el resto de tarjetas. Centrado queda como "lámina" proporcionada.
        st.markdown(f"""
        <div class="fig-card" style="max-width:840px; margin:0 auto;">
            <img src="data:image/png;base64,{b64}" style="width:100%; display:block; border-radius:6px;">
        </div>
        """, unsafe_allow_html=True)

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
        shap_summary_image(
            "SHAP Summary LGBM.png",
            "SHAP Summary Plot — LightGBM (Figura 27)",
            "Cada punto es una instancia del test; el color indica el valor de la variable (rojo alto, azul bajo) "
            "y la posición horizontal su impacto en la predicción. LBXGH y RIDAGEYR dominan el modelo.",
        )

    with tab2:
        st.markdown(f"""
        <div class="clinical-note" style="margin-bottom:14px;">
        El ranking de SVM-RBF coincide en las variables dominantes con LightGBM (<b>LBXGH</b>, <b>LBXGLU</b>,
        <b>LBDLDL</b>, <b>RIDAGEYR</b>), lo que refuerza la validez clínica del hallazgo al ser independiente del
        algoritmo, dotándolo de mayor robustez metodológica. KernelExplainer trata el modelo como caja negra,
        aplicable a cualquier clasificador.
        </div>
        """, unsafe_allow_html=True)
        shap_chart(SHAP_SVMRBF, C_PRIMARY, "Valores aproximados por muestreo: fondo de 100 instancias, contribuciones sobre 200 instancias de test.")
        shap_summary_image(
            "SHAP Summary SVM.png",
            "SHAP Summary Plot — SVM-RBF (Figura 31)",
            "Cada punto es una instancia; color = valor de la variable, posición = impacto. "
            "KernelExplainer sobre 200 instancias del test.",
        )

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 4 — QUANTUM CIRCUIT
# ═══════════════════════════════════════════════════════════════════════
elif page == "Circuito Cuántico":
    header("Componente cuántico", "Circuito Cuántico",
           "Configuración del ZZFeatureMap y FidelityQuantumKernel implementados en Qiskit sobre Databricks CE.")

    cols = st.columns(4)
    specs = [("8", "Qubits (feature_dimension)"), ("2", "Repeticiones (reps)"), ("Linear", "Entanglement"), ("qiskit 2.5.0", "Versión")]
    for col, (num, lab) in zip(cols, specs):
        with col:
            st.markdown(f'<div class="info-card stat-card" style="min-height:96px;"><div class="stat-num" style="font-size:clamp(20px, 2.4vw, 34px);">{num}</div><div class="stat-label">{lab}</div></div>', unsafe_allow_html=True)

    # "Cómo funciona" a ancho completo: sus dos párrafos son conceptualmente independientes
    # (codificación vs. kernel), así que van lado a lado en vez de apilados — a ancho completo el
    # texto apilado dejaría líneas incómodamente largas, y apilado-estrecho (como antes, dentro de
    # media página) dejaba la columna vecina con un hueco vacío grande por debajo.
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cómo funciona</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card">
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:28px;">
        <p style="font-size:15px; color:{t['text_secondary']}; line-height:1.75; margin:0;">
        El <b style="color:{t['text']}">ZZFeatureMap</b> codifica cada una de las 8 variables clínicas como un ángulo
        de rotación (puerta RZ) en un qubit independiente, tras crear superposición con puertas Hadamard. Su elemento
        distintivo es el <b style="color:{t['text']}">entrelazamiento</b> entre pares de qubits mediante puertas que
        dependen del producto cruzado de dos variables — correlaciones que el kernel RBF clásico no puede representar.
        </p>
        <p style="font-size:15px; color:{t['text_secondary']}; line-height:1.75; margin:0;">
        El <b style="color:{t['text']}">FidelityQuantumKernel</b> mide la similitud entre dos pacientes como la
        fidelidad entre sus estados cuánticos: <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. La implementación usa
        <code>StatevectorSampler</code>, simulando el estado exacto sin ruido — resultados deterministas y reproducibles.
        </p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Fila de dos columnas: la gráfica de importancia (izquierda) ocupa proporcionalmente más ancho
    # que la lista de estadísticas de entrenamiento (derecha) — con 8 barras horizontales necesita más
    # espacio para leerse cómoda; la lista de KPIs es compacta y no lo necesita. El resto de la app
    # estira el ELEMENTO más corto con CSS (mismo truco que .compare-grid / .st-key-bloch_row) en vez
    # de dejar hueco vacío bajo un bloque de altura fija — aquí la nota de la derecha crece hasta
    # igualar la altura de la gráfica.
    st.markdown("<br>", unsafe_allow_html=True)
    qc_row = st.container(key="qc_stats_row")
    col1, col2 = qc_row.columns([1.35, 1])
    with col1:
        st.markdown('<div class="section-title">8 features seleccionadas (Random Forest)</div>', unsafe_allow_html=True)
        # Datos propios de esta gráfica (no QSVM_FEATURES): esa lista alimenta también los sliders de
        # Esfera de Bloch y Predictor en Vivo, y la actualización pedida (BMXBMI sustituye a WTINT2YR,
        # sin variables DIQ) es solo para este ranking del Random Forest — no debe alterar esas páginas.
        names = list(reversed(list(RF_TOP8_IMPORTANCE.keys())))
        values = list(reversed(list(RF_TOP8_IMPORTANCE.values())))
        customdata = [[code, _wrap_hover(VAR_DESC.get(code, code))] for code in names]
        fig = go.Figure()
        # Sombra casi imperceptible detrás de cada barra: misma posición y grosor exactos que la barra
        # real (ambas con width/offset automáticos, sin forzar ningún valor a mano — eso fue lo que se
        # veía tosco), apenas un 3% más larga y muy tenue. Solo se asoma una hebra de color detrás de
        # la punta, como una sombra proyectada suave — nada agresivo.
        fig.add_trace(go.Bar(
            x=[v * 1.03 for v in values], y=names, orientation="h",
            marker_color=hex_to_rgba(t["text"], 0.08), marker_line_width=0,
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Bar(
            x=values, y=names, orientation="h", marker_color=C_PRIMARY, cliponaxis=False,
            text=[nf(v) for v in values], textposition="outside", textfont=dict(size=11),
            customdata=customdata, showlegend=False,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
        ))
        plotly_layout(fig, height=300, barmode="overlay",
                      # Margen izquierdo reducido (150→95): pega las etiquetas al borde de la tarjeta
                      # en vez de dejarlas centradas con aire de sobra. Al ser el ancho de la tarjeta
                      # fijo, ese espacio liberado pasa directo al área de barras — se agrandan solas,
                      # de forma proporcional, sin tocar la tarjeta que las contiene.
                      xaxis=dict(title="Importancia RF", showgrid=True, gridcolor=t["border"], range=[0, max(values) * 1.3], fixedrange=True),
                      yaxis=dict(tickfont=dict(size=12, color=t["text"]), fixedrange=True), margin=dict(l=95, r=70, t=20, b=40))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with col2:
        st.markdown('<div class="section-title">Entrenamiento y evaluación</div>', unsafe_allow_html=True)
        # Lista vertical de KPIs (mismo patrón .kpi-row que Esfera de Bloch / Predictor en Vivo) en vez
        # de tarjetas en grilla: más compacta en una columna estrecha y de un vistazo. Incluye los dos
        # datos que antes solo estaban en la nota de texto (instancias del test y tiempo de inferencia).
        tstats = [
            ("Instancias entrenamiento", "500"),
            ("Tiempo entrenamiento", "21,1 min"),
            ("Instancias test", "1.567"),
            ("Tiempo de inferencia", "132,8 min"),
            ("Support vectors", "[425, 70]"),
        ]
        kpi_rows = "".join(f'<div class="kpi-row"><span class="kpi-label">{l}</span><span class="kpi-value">{v}</span></div>' for l, v in tstats)
        st.markdown(f'<div class="info-card">{kpi_rows}</div>', unsafe_allow_html=True)

    # La nota va DEBAJO de la fila (ancho completo), no dentro de col2: así la tarjeta de KPIs es el
    # único elemento de esa columna y puede estirarse limpio hasta igualar la altura de la gráfica —
    # si la nota se quedara dentro de col2, empujaría esa columna más abajo que la de la gráfica.
    st.markdown(f"""
    <div class="clinical-note" style="margin-top:16px;">
    Por el coste O(n²) del kernel cuántico, el entrenamiento se limitó a una muestra estratificada de 500 instancias
    (el límite operativo de Databricks CE serverless se sitúa ~500-1.000). La evaluación se hizo sobre el test
    completo (1.567 instancias) por lotes de 100, con un tiempo total de predicción de 132,8 minutos.
    </div>
    """, unsafe_allow_html=True)

    # Diagrama del circuito a ancho completo (fuera de col1/col2: con 8 qubits y las 4 secciones de
    # entrelazamiento apiladas, comprimirlo a la mitad de la página dejaría las etiquetas P(...) ilegibles.
    _circuit_path = FIGURES_DIR / "Circuito Cuantico 8qb.png"
    if _circuit_path.exists():
        st.markdown('<div class="section-title" style="margin-top:20px;">Circuito completo (8 qubits)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">ZZFeatureMap con reps=2: codificación (H + P) seguida de dos rondas de entrelazamiento lineal entre qubits adyacentes.</div>', unsafe_allow_html=True)
        _circuit_b64 = _b64_image_autocrop(str(_circuit_path))
        st.markdown(f"""
        <div class="fig-card" style="padding:14px; max-width:900px; margin:0 auto;">
            <img src="data:image/png;base64,{_circuit_b64}" style="width:100%; display:block; border-radius:6px;">
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 5 — BLOCH SPHERE EMULATOR
# ═══════════════════════════════════════════════════════════════════════
elif page == "Esfera de Bloch":
    header("Codificación cuántica", "Esfera de Bloch",
           "Cómo el ZZFeatureMap codifica el valor de una variable clínica como estado cuántico |ψ⟩.")

    # Contenedor con clave (genera .st-key-bloch_row) para poder estirar la gráfica 3D hasta el alto
    # de la columna izquierda y que ambas tarjetas cierren alineadas abajo — ver CSS .st-key-bloch_row.
    bloch_row = st.container(key="bloch_row")
    col1, col2 = bloch_row.columns([1, 1.3])
    with col1:
        var_code = st.selectbox("Variable clínica", list(QSVM_FEATURES.keys()),
                                 format_func=lambda c: f"{c} — {QSVM_FEATURES[c]['label']}")
        v = QSVM_FEATURES[var_code]
        lo, hi = v["range"]
        val = st.slider(f"Valor ({v['unit']})", float(lo), float(hi), float(v["default"]),
                        step=v["step"], format=v["fmt"])

        x_norm = (val - lo) / (hi - lo)
        theta = 2 * x_norm * np.pi
        alpha = np.cos(theta / 2)
        beta = np.sin(theta / 2)
        p0, p1 = alpha**2, beta**2

        st.markdown(f"""
        <div class="info-card">
            <div class="kpi-row"><span class="kpi-label">x normalizado</span><span class="kpi-value">{nf(x_norm, 3)}</span></div>
            <div class="kpi-row"><span class="kpi-label">θ = 2·x_norm·π</span><span class="kpi-value">{nf(theta, 3)} rad</span></div>
            <div class="kpi-row"><span class="kpi-label">α (amplitud |0⟩)</span><span class="kpi-value">{nf(alpha, 3)}</span></div>
            <div class="kpi-row"><span class="kpi-label">β (amplitud |1⟩)</span><span class="kpi-value">{nf(beta, 3)}</span></div>
            <div class="kpi-row"><span class="kpi-label">P(|0⟩)</span><span class="kpi-value">{pct(p0)}</span></div>
            <div class="kpi-row"><span class="kpi-label">P(|1⟩)</span><span class="kpi-value">{pct(p1)}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure()
        # Superficie esférica con sombreado (lighting) para darle volumen de bola real
        u, w = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        xs, ys, zs = np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w)
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs, opacity=0.18, showscale=False, hoverinfo="skip",
            colorscale=[[0, C_LIGHT], [1, C_MID2]],
            lighting=dict(ambient=0.6, diffuse=0.9, specular=0.3, roughness=0.5, fresnel=0.2),
            lightposition=dict(x=120, y=200, z=160),
        ))
        # Círculos máximos (ecuador + 2 meridianos): refuerzan la curvatura al rotar
        circ = np.linspace(0, 2 * np.pi, 120)
        for gx, gy, gz in [
            (np.cos(circ), np.sin(circ), np.zeros_like(circ)),   # ecuador (plano XY)
            (np.cos(circ), np.zeros_like(circ), np.sin(circ)),   # meridiano XZ
            (np.zeros_like(circ), np.cos(circ), np.sin(circ)),   # meridiano YZ
        ]:
            fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", opacity=0.5,
                                        line=dict(color=C_MID2, width=1.5), showlegend=False, hoverinfo="skip"))
        # Ejes cartesianos
        for ax_x, ax_y, ax_z in [([-1.06,1.06],[0,0],[0,0]), ([0,0],[-1.06,1.06],[0,0]), ([0,0],[0,0],[-1.10,1.10])]:
            fig.add_trace(go.Scatter3d(x=ax_x, y=ax_y, z=ax_z, mode="lines",
                                        line=dict(color=t["border"], width=2), showlegend=False, hoverinfo="skip"))
        # Vector de estado |ψ⟩ (φ = 0 → contenido en el plano XZ)
        px, py, pz = np.sin(theta), 0.0, np.cos(theta)
        fig.add_trace(go.Scatter3d(x=[0, px], y=[0, py], z=[0, pz], mode="lines",
                                    line=dict(color=C_PRIMARY, width=7), showlegend=False, hoverinfo="skip"))
        # Punta de flecha (cono) apuntando hacia afuera a lo largo del vector
        fig.add_trace(go.Cone(x=[px], y=[py], z=[pz], u=[px], v=[py], w=[pz],
                              sizemode="absolute", sizeref=0.18, anchor="tip", showscale=False,
                              colorscale=[[0, C_PRIMARY], [1, C_PRIMARY]], hoverinfo="skip"))
        # Proyección vertical al plano ecuatorial (pista de profundidad sutil)
        fig.add_trace(go.Scatter3d(x=[px, px], y=[py, py], z=[pz, 0], mode="lines", opacity=0.45,
                                    line=dict(color=C_PRIMARY, width=2, dash="dot"), showlegend=False, hoverinfo="skip"))
        # Punto del estado (para el hover)
        fig.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz], mode="markers",
                                    marker=dict(size=5, color=C_PRIMARY), showlegend=False,
                                    hovertemplate=f"|ψ⟩ ({var_code})<extra></extra>"))
        # Etiquetas de los polos
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[1.08,-1.08], mode="text", text=["|0⟩","|1⟩"],
                                    textfont=dict(size=15, color=t["text"]), showlegend=False, hoverinfo="skip"))
        # Alto FIJO en píxeles (sin autosize): tamaño idéntico en cada rerun y en cualquier navegador
        # (Firefox incluido). 486 px ≈ alto natural de la columna izquierda (selectbox + slider +
        # tarjeta de métricas), para que el fondo de esta tarjeta quede alineado con el de aquella.
        # Sube/baja este valor para agrandar/reducir la esfera.
        fig.update_layout(
            height=486, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                # Rangos apretados (±1.08 en vez de ±1.4): la esfera de radio 1 llena ~93% del cubo en
                # vez de ~71% → ~30% más grande DENTRO de la misma tarjeta, sin cambiar su tamaño en px.
                # z un poco más holgado (±1.12) para dejar aire a las etiquetas |0⟩/|1⟩.
                xaxis=dict(visible=False, range=[-1.08, 1.08]),
                yaxis=dict(visible=False, range=[-1.08, 1.08]),
                zaxis=dict(visible=False, range=[-1.12, 1.12]),
                aspectmode="cube", dragmode="orbit",
                camera=dict(eye=dict(x=1.45, y=1.45, z=0.75)),
            ),
        )
        # key estable: sin ella Streamlit remonta el iframe de Plotly en cada rerun y la esfera
        # parpadea/encoge; con key reutiliza el mismo componente y solo actualiza los datos.
        st.plotly_chart(fig, width="stretch", key="bloch_sphere",
                        config={"displayModeBar": False, "responsive": True})

    st.markdown(f"""
    <div class="clinical-note">
    Este cálculo reproduce el primer paso de codificación del ZZFeatureMap real: θ = 2·x_norm·π, donde x_norm es el
    valor clínico normalizado al rango fisiológico [0,1]. No incluye el paso de entrelazamiento entre qubits
    (puertas P(2·(π−x_i)·(π−x_j))), que solo es representable en el espacio conjunto de los 8 qubits — ver Circuito Cuántico.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 6 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════
elif page == "Predictor en Vivo":
    header("Predicción interactiva", "Predictor en Vivo",
           "Formulario con las 8 variables clínicas principales — LightGBM y SVM-RBF.")

    _sp = _load_scaler_and_medians()
    _models_ready = (ONNX_AVAILABLE and _sp is not None
                      and _load_onnx_session("lgbm_final.onnx") is not None
                      and _load_onnx_session("svm_final.onnx") is not None)

    if _models_ready:
        st.markdown(f"""
        <div class="clinical-note" style="margin-bottom:16px;">
        <b>Inferencia real (ONNX).</b> Predicciones de LightGBM y SVM-RBF vía <code>onnxruntime</code>,
        con el <code>StandardScaler</code> recuperado del pipeline Gold. Las 8 variables mostradas son las
        de mayor importancia clínica; las 81 features restantes se fijan en la mediana del conjunto de
        entrenamiento. QSVM no está disponible en tiempo real (coste O(n²) del kernel cuántico — 132,8 min
        por instancia), igual que documenta tu TFM.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="clinical-note" style="margin-bottom:16px;">
        ⚠ <b>Aviso técnico y clínico.</b> Este formulario no tiene aún conectados los modelos serializados reales
        (<code>.onnx</code>) de tu repositorio — coloca <code>lgbm_final.onnx</code>, <code>svm_final.onnx</code>,
        <code>scaler_correcto.json</code> y <code>medianas_correctas.json</code> en <code>streamlit/models/</code>.
        La puntuación mostrada abajo es un <b>sustituto transparente</b>: una combinación ponderada por importancia
        SHAP normalizada, solo para fines de maquetación. <b>No reemplaza el diagnóstico médico profesional</b> y
        no debe presentarse como predicción real en la defensa sin antes conectar los modelos entrenados. QSVM no
        está disponible en tiempo real (coste O(n²) del kernel cuántico), igual que documenta tu TFM.
        </div>
        """, unsafe_allow_html=True)

    cols = st.columns(2)
    inputs = {}
    items = list(QSVM_FEATURES.items())
    for i, (code, v) in enumerate(items):
        with cols[i % 2]:
            lo, hi = v["range"]
            inputs[code] = st.slider(f"{v['label']} ({v['unit']})", float(lo), float(hi), float(v["default"]),
                                      step=v["step"], format=v["fmt"], key=f"lp_{code}")

    _real = predict_real(inputs) if _models_ready else None

    if _real is not None:
        risk, _svm_prob = _real
    else:
        # Sustituto transparente: combinacion ponderada por SHAP normalizado (NO es el modelo real)
        weights = {c: v["importance"] for c, v in QSVM_FEATURES.items()}
        wsum = sum(weights.values())
        score = 0.0
        for code, w in weights.items():
            lo, hi = QSVM_FEATURES[code]["range"]
            x_norm = (inputs[code] - lo) / (hi - lo)
            score += (w / wsum) * x_norm
        risk = float(np.clip(score, 0, 1))

    # Categoría interpretable respecto al umbral de decisión (50%)
    if risk < 0.33:
        cat, cat_color, interp = ("Bajo", C_MID1,
            "Los valores introducidos sitúan el score claramente por debajo del umbral de decisión (50%).")
    elif risk < 0.5:
        cat, cat_color, interp = ("Moderado", C_PRIMARY,
            "El score se aproxima al umbral de decisión (50%): zona de incertidumbre.")
    else:
        cat, cat_color, interp = ("Elevado", C_DARK,
            "El score supera el umbral de decisión (50%): el sustituto clasificaría como caso positivo.")

    st.markdown("<br>", unsafe_allow_html=True)
    # Score (izquierda) y velocímetro (derecha) viven dentro de UNA sola tarjeta: es el contenedor con
    # clave "predictor_gauge_row" el que lleva borde/fondo/sombra (ver CSS .st-key-predictor_gauge_row),
    # y sus dos hijos pierden su recuadro individual. vertical_alignment="center" centra ambas mitades
    # entre sí, así que ya no hace falta fijar la altura de la caja de score en px.
    gauge_row = st.container(key="predictor_gauge_row")
    rcol1, rcol2 = gauge_row.columns([1, 1.4], vertical_alignment="center")
    with rcol1:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-label" style="margin-bottom:8px;">Score de riesgo (sustituto)</div>
            <div class="kpi-value-auc" style="color:{cat_color};">{pct(risk)}</div>
            <div style="margin-top:12px;">
                <span style="display:inline-flex; align-items:center; gap:7px; font-size:13px; font-weight:600;
                      padding:5px 12px; border-radius:20px; background:{cat_color}1F; color:{cat_color};">
                    <span style="width:8px; height:8px; border-radius:50%; background:{cat_color};"></span>
                    Riesgo {cat}
                </span>
            </div>
            <div style="font-size:12.5px; color:{t['text_secondary']}; margin-top:14px; line-height:1.55;">{interp}</div>
        </div>
        """, unsafe_allow_html=True)
    with rcol2:
        # Gauge tipo velocímetro (aguja + círculo, zonas segmentadas con separadores) en vez del modo
        # "gauge+number" nativo de Plotly (que solo pinta una barra rellena desde cero, sin aguja).
        # Construido a mano con arcos SVG (fig.add_shape) + una aguja por trigonometría — mismo
        # planteamiento que ya usa el vector de estado en Esfera de Bloch. Paleta propia del proyecto:
        # 5 tonos, de C_LIGHT (bajo riesgo) a C_DARK (alto riesgo), en vez del rojo→verde de referencia.
        def _polar(angle_deg, r):
            rad = np.radians(angle_deg)
            return r * np.cos(rad), r * np.sin(rad)

        def _arc_path(r_in, r_out, a0, a1, n=30):
            a0r, a1r = np.radians(a0), np.radians(a1)
            th_out, th_in = np.linspace(a0r, a1r, n), np.linspace(a1r, a0r, n)
            xs = np.concatenate([r_out * np.cos(th_out), r_in * np.cos(th_in)])
            ys = np.concatenate([r_out * np.sin(th_out), r_in * np.sin(th_in)])
            return f"M {xs[0]},{ys[0]} " + " ".join(f"L{x},{y}" for x, y in zip(xs[1:], ys[1:])) + " Z"

        def _val_to_angle(v):
            return 180 - (v / 100) * 180  # 0 -> 180° (izq.), 100 -> 0° (der.)

        fig = go.Figure()
        r_in, r_out = 0.62, 1.0
        gap_deg = 1.6  # separador blanco entre zonas, como en la referencia
        band_colors = [C_LIGHT, C_MID2, C_MID1, C_PRIMARY, C_DARK]
        n_bands = len(band_colors)
        band_span = 180 / n_bands
        for i, color in enumerate(band_colors):
            a0 = 180 - i * band_span - gap_deg / 2
            a1 = 180 - (i + 1) * band_span + (gap_deg / 2 if i < n_bands - 1 else 0)
            fig.add_shape(type="path", path=_arc_path(r_in, r_out, a0, a1),
                          fillcolor=color, line=dict(color=t["surface"], width=2), layer="below")

        # Marcas finas en 0/25/50/75/100 (sin la rueda de números completa de un eje tradicional)
        for tv in [0, 25, 50, 75, 100]:
            ang = _val_to_angle(tv)
            x0, y0 = _polar(ang, r_out + 0.02)
            x1, y1 = _polar(ang, r_out + 0.08)
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=t["text_secondary"], width=1.5))
            xl, yl = _polar(ang, r_out + 0.20)
            fig.add_annotation(x=xl, y=yl, text=str(tv), showarrow=False,
                               font=dict(size=11, color=t["text_secondary"], family="Inter"))

        # Aguja: línea fina + círculo abierto en el pivote (como la referencia), color según categoría
        # de riesgo — coherente con el número y la insignia de la tarjeta izquierda.
        needle_ang = _val_to_angle(risk * 100)
        nx, ny = _polar(needle_ang, r_out * 0.86)
        fig.add_trace(go.Scatter(x=[0, nx], y=[0, ny], mode="lines",
                                 line=dict(color=cat_color, width=3), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                                 marker=dict(size=16, color=t["surface"], line=dict(color=cat_color, width=3)),
                                 hoverinfo="skip", showlegend=False))

        fig.update_xaxes(visible=False, range=[-1.25, 1.25], fixedrange=True)
        fig.update_yaxes(visible=False, range=[-0.08, 1.25], fixedrange=True, scaleanchor="x", scaleratio=1)
        fig.update_layout(height=260, margin=dict(l=16, r=16, t=6, b=4),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", color=t["text"]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(f'<div class="section-sub" style="text-align:center; margin-top:-6px;">Zonas: bajo · moderado · alto · &nbsp;línea = umbral de decisión (50%)</div>', unsafe_allow_html=True)
