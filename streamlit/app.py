"""
QML DataOps TFM Dashboard — App completa (6 paginas)
Universidad Europea de Valencia · TFM Juan Albornoz

QML DataOps TFM Dashboard — App completa (7 paginas)

Estructura de paginas segun Seccion "Estructura y Paginas de la Aplicacion"
del TFM_UEV_QML_JuanAlbornoz.docx: Overview, Results, SHAP Analysis,
Quantum Circuit, Bloch Sphere Emulator, Live Predictor. Se anade Gobernanza,
que documenta los controles de calidad y trazabilidad del pipeline DataOps.

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

# ═════════════════════════════════════════════════════════════════════════
# SISTEMA DE DISEÑO
# ═════════════════════════════════════════════════════════════════════════
# PALETA BASE (definida por el autor):
#   #F5A623  ámbar           #F9C449  oro       #FBDD8B  crema
#   #E9E9E9  gris claro      #1C1F26  carbón
#
# Medida con el validador, la paleta tiene DOS familias muy desiguales, y de ahí
# sale todo el reparto:
#   · Los tres cálidos caben en 17° de tono (73° → 90°) y son todos altísimos en
#     luminosidad (L 0,784 · 0,847 · 0,905). Sobre blanco dan 2,03:1 · 1,61:1 ·
#     1,33:1 — NINGUNO llega al 3:1 que exige una marca de datos. Sobre el carbón
#     dan 8,1:1 · 10,2:1 · 12,4:1. Es una paleta concebida para fondo oscuro, y el
#     tema oscuro es donde rinde al máximo.
#   · #F5A623 ↔ #F9C449 → ΔE 7,1 en visión normal (el suelo es 15): tal cual, los
#     dos cálidos vecinos son el mismo color para cualquier ojo, no solo para quien
#     tiene daltonismo.
#   · Dos neutros puros que hacen todo el trabajo estructural: #E9E9E9 (croma 0,000)
#     y #1C1F26 (croma 0,014 — 16,5:1 sobre blanco, el ancla de contraste).
#
# Reparto en consecuencia:
#   · INTERFAZ (fondos, superficies, bordes, tinta, acentos) → los 5 tonos. Carbón y
#     gris son superficie y tinta; el ámbar es el acento de marca; oro y crema, realce.
#   · SERIES (los 3 modelos) → NO caben tres cálidos. Se probó por enumeración: el
#     mejor trío cálido posible se queda en ΔE 15,0-15,1 (justo en el filo del suelo)
#     y FALLA en oscuro, además de convertir el tercer tono en un oliva turbio que ya
#     no parece la paleta. El reparto que sí pasa usa las DOS familias —
#     tinta · gris · ámbar— y es además la lectura correcta del relato: los dos
#     modelos clásicos en neutro, el cuántico en el color de marca. El color va donde
#     está la tesis.
#   · RAMP (magnitud) → rampa de un solo tono sobre el eje cálido (h 78°), cinco pasos
#     de luminosidad monótona. Una escala de magnitud tiene que ser monocroma para que
#     el orden se lea sin leyenda.
#
# Resultados del validador con los valores de abajo (--pairs all):
#   SERIES claro  → CVD ΔE 14,3 · visión normal 17,1 · contraste ≥3:1 los tres
#   SERIES oscuro → CVD ΔE 16,8 · visión normal 17,2 · contraste ≥3:1 los tres
#   RAMP claro / RAMP oscuro → monotonía, ΔL, tono único y contraste: TODAS PASS
#   Único FAIL, deliberado: el suelo de croma sobre los dos slots neutros. Ese suelo
#   existe para que un tono no "lea gris" y deje de hacer trabajo de identidad; aquí
#   los neutros SON neutros a propósito y con un solo cromático entre ellos no hay
#   ambigüedad posible (tinta ↔ gris ΔE 32). Ojo: el gris de serie es FRÍO (h 267°,
#   el tono del propio carbón aclarado) y no cálido — un taupe cálido colapsaba
#   contra el ámbar en ΔE 10. Si se retoca cualquier hex, hay que volver a pasar el
#   validador.
# ─────────────────────────────────────────────────────────────────────────

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "sidebar_narrow" not in st.session_state:
    st.session_state.sidebar_narrow = False

_is_dark = st.session_state.theme == "dark"

# ── Paleta base, literal. Referencia única para todo lo demás. ──
P_AMBAR, P_ORO, P_CREMA, P_GRIS, P_CARBON = "#F5A623", "#F9C449", "#FBDD8B", "#E9E9E9", "#1C1F26"

# ── Escala CATEGÓRICA: identidad de cada modelo. Orden fijo, nunca reciclado. ──
# Tinta = baseline clásico · gris = puente estructural · ámbar = cuántico.
# Los dos clásicos van en las dos familias neutras del autor (carbón y gris claro),
# invertidas entre temas para que la que hace de tinta sea siempre la que contrasta.
# El slot cuántico lleva el color de marca: en oscuro es el #F9C449 del autor sin
# tocar (10,2:1); en claro se baja a L 0,64 porque el original da 1,61:1 sobre blanco
# y como relleno de barra sería invisible. Se conserva el tono, se corrige el paso.
SERIES = {
    "lightgbm": P_GRIS    if _is_dark else P_CARBON,
    "svm_rbf":  "#8A8F99" if _is_dark else "#71747C",
    "qsvm":     P_ORO     if _is_dark else "#C07C08",
}

# ── Escala SECUENCIAL: magnitud (matriz de confusión, velocímetro) ──
# Un solo tono (h 78°, el eje cálido de la paleta), luminosidad monótona. El índice 4
# es SIEMPRE el extremo de máxima magnitud: en claro eso es el paso más oscuro y en
# oscuro el más brillante — en ambos casos, el que más se despega de su fondo.
RAMP = (["#724D00", "#9A6A00", "#C48800", "#EBAA2D", "#FFCF83"] if _is_dark
        else ["#EDAB30", "#CC8E00", "#A57200", "#805800", "#604100"])

# ── Acento de marca (cromo de interfaz: navegación, foco, sliders, reglas) ──
# En oscuro es el #F5A623 del autor sin tocar (8,1:1 sobre la superficie). En claro
# ese mismo ámbar da 2,03:1 y no puede llevar texto: el acento baja a L 0,55 del mismo
# tono (4,96:1 sobre blanco, apto para texto pequeño según WCAG) y el ámbar puro queda
# reservado a rellenos y tintes, donde no se le pide legibilidad.
C_PRIMARY = P_AMBAR if _is_dark else "#9A6504"
C_DARK    = P_ORO if _is_dark else "#6B4600"
C_QUANTUM = SERIES["qsvm"]          # acento del componente cuántico (Bloch, ZZFeatureMap)
# La crema es el tono de realce de la paleta, y solo sirve donde contrasta: 12,4:1
# sobre el carbón (vale como texto) frente a 1,33:1 sobre blanco (solo como relleno o
# tinte). Por eso en claro nunca lleva texto encima ni hace de texto.
C_CREMA = P_CREMA
# Alias de compatibilidad: pasos intermedios de la rampa. Se toman del extremo VISIBLE
# de cada una — la clara corre brillante→oscuro y la oscura oscuro→brillante, así que el
# paso "medio-alto" es RAMP[1] en claro y RAMP[3] en oscuro. Con esto el halo del
# interruptor de tema y el anillo de las láminas pesan ópticamente igual en ambos temas.
C_LIGHT = RAMP[0]
C_MID2  = RAMP[3] if _is_dark else RAMP[1]
C_MID1  = RAMP[2]

# ── Colores de ESTADO (reservados: nunca se reutilizan como “serie 4”) ──
# Única excepción deliberada a la paleta base: bien/atención/grave tienen que leerse
# como estado de forma inmediata y universal. Con una marca ámbar hay un riesgo extra —
# el “atención” canónico ES ámbar y se confundiría con el acento—, así que se desplaza
# a un naranja quemado claramente más rojo (h 46° frente a los 73° de la marca) y el
# “grave” a rojo (h 29°). Los tres van SIEMPRE acompañados de su etiqueta de texto.
STATUS = {
    "good":     "#4FBE8C" if _is_dark else "#2F6A4E",
    "warning":  "#F0834A" if _is_dark else "#B4531B",
    "critical": "#EE6A5C" if _is_dark else "#A93226",
}

# ── Tipografía ──
# Newsreader (serif de pantalla) para titulares: da el registro editorial de
# publicación científica. IBM Plex Sans para interfaz e IBM Plex Mono para cifras
# y código — la familia Plex es además la tipografía de IBM, autora de Qiskit.
FONT_SERIF = "'Newsreader', 'Iowan Old Style', Georgia, serif"
FONT_SANS  = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO  = "'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace"
# Plotly no acepta la pila con comillas: solo el nombre de la familia.
PLOTLY_FONT = "IBM Plex Sans"
PLOTLY_MONO = "IBM Plex Mono"

def T():
    """Superficies y tintas, derivadas de la paleta base.

    Las dos superficies clave son literalmente los dos neutros del autor, cada una en
    el tema donde le toca ser el fondo de tarjeta: #1C1F26 en oscuro y —por el otro
    extremo— #E9E9E9 como barra lateral y superficie alterna en claro, con blanco puro
    para las tarjetas. El lienzo de fondo se separa un paso de la tarjeta en ambos
    temas (#F4F3F0 en claro, #12151B en oscuro) para que la tarjeta se vea flotar.

    El par tinta/superficie es el que da el contraste que se pedía: 16,5:1 en claro
    (#1C1F26 sobre blanco) y 13,6:1 en oscuro (#E9E9E9 sobre #1C1F26) — muy por encima
    del 4,5:1 de WCAG. Los dos escalones de tinta apagada mantienen 8,3:1 y 5,0:1 en
    claro, 7,3:1 y 4,8:1 en oscuro, así que TODOS los niveles de texto de la app pasan
    WCAG AA, incluido el más atenuado.

    NOTA: ninguno de los tres cálidos se usa como color de texto sobre blanco: dan
    2,03:1, 1,61:1 y 1,33:1. Viven en bordes, rellenos y realces; cuando el acento
    tiene que llevar texto en claro, usa el paso oscurecido C_PRIMARY (4,96:1).
    """
    if st.session_state.theme == "dark":
        return dict(bg="#12151B", surface=P_CARBON, surface_alt="#262A33",
                     text=P_GRIS, text_secondary="#A9ADB6", text_muted="#868B95",
                     border="#2C313B", border_strong="#3D434F",
                     sidebar_bg="#0E1116", sidebar_active="#232833")
    return dict(bg="#F4F3F0", surface="#FFFFFF", surface_alt=P_GRIS,
                 text=P_CARBON, text_secondary="#4A4E57", text_muted="#6B6F78",
                 border="#DCDBD6", border_strong="#BFBEB8",
                 sidebar_bg=P_GRIS, sidebar_active="#DAD9D3")

t = T()

# ── Elevación: una sola definición de sombra para TODAS las tarjetas ──
SHADOW = ("0 1px 2px rgba(0,0,0,0.30), 0 6px 20px -6px rgba(0,0,0,0.50)" if _is_dark
          else "0 1px 1.5px rgba(11,26,38,0.04), 0 4px 14px -4px rgba(11,26,38,0.07)")
SHADOW_HOVER = ("0 2px 4px rgba(0,0,0,0.35), 0 16px 34px -10px rgba(0,0,0,0.62)" if _is_dark
                else "0 2px 4px rgba(11,26,38,0.05), 0 16px 32px -10px rgba(11,26,38,0.13)")

narrow = st.session_state.sidebar_narrow
SIDEBAR_WIDTH = "84px" if narrow else "270px"
# Color del carril vacío de los sliders: claro en tema claro, hundido en tema oscuro (si usáramos
# un azul fijo, en oscuro el carril quedaría un surco brillante sobre fondo oscuro).
# En claro NO se usa RAMP[0]: ese paso está calibrado para pintar DATO sobre blanco y como
# carril de control resultaba demasiado saturado. Aquí va un tinte más apagado del mismo tono.
SLIDER_GROOVE = "#DFDCD3" if st.session_state.theme == "light" else t["surface_alt"]

# ── Tratamiento de las figuras PNG de fondo blanco (beeswarm SHAP, circuito) según tema ──
# Esas imágenes tienen fondo blanco intrínseco (figuras científicas del TFM). En tema claro se funden
# con la tarjeta. En oscuro, un bloque blanco a ancho completo sobre fondo casi negro "flota" y rompe
# la elegancia del dark mode. En vez de regenerar las imágenes (fuera de alcance), las presentamos
# como "lámina enmarcada": un anillo fino en color de marca (C_MID2) las ata a la paleta, una sombra
# profunda las asienta, y un filtro casi imperceptible baja el fogonazo del blanco puro. En claro se
# mantiene el aspecto plano y sobrio de siempre.
FIG_IMG_FILTER = "brightness(0.965) saturate(1.03)" if _is_dark else "none"
FIG_CARD_SHADOW = (f"0 0 0 1px {C_MID2}33, {SHADOW}" if _is_dark else SHADOW)
FIG_CARD_SHADOW_HOVER = (f"0 0 0 1px {C_PRIMARY}66, {SHADOW_HOVER}" if _is_dark else SHADOW_HOVER)
# Padding-"passe-partout" algo mayor en oscuro: la lámina blanca respira dentro del marco.
FIG_CARD_PAD = "14px" if _is_dark else "10px"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300;6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family:{FONT_SANS} !important; }}
/* Cifras SIEMPRE tabulares: al mover un slider o cambiar de página los números no “bailan”
   horizontalmente, porque todos los dígitos ocupan el mismo ancho. */
body {{ font-variant-numeric: tabular-nums; -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; }}
.stApp {{
    background-color:{t['bg']}; color:{t['text']};
    /* Dos halos muy tenues en el color de marca: dan profundidad al fondo plano sin
       introducir una textura visible. En claro son casi imperceptibles; en oscuro
       evitan que el fondo lea como negro muerto. */
    background-image:
        radial-gradient(1100px 520px at 12% -8%, {C_PRIMARY}{'1E' if _is_dark else '0D'}, transparent 60%),
        radial-gradient(900px 460px at 100% 0%, {C_QUANTUM}{'16' if _is_dark else '0A'}, transparent 62%);
    /* SIN background-attachment:fixed a propósito. Sería redundante —el que hace scroll es
       section[data-testid="stMain"], no .stApp, que ya ocupa el viewport— y en Firefox un
       fondo fijo obliga a repintar los dos degradados en cada scroll, con el tirón
       consiguiente. Sin él se ve igual y el scroll queda limpio. */
}}
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
    background-color:{t['sidebar_bg']};
    /* Velo vertical muy tenue en el tono de marca: el panel gana profundidad y se
       distingue del lienzo principal sin recurrir a un borde marcado. */
    background-image:linear-gradient(180deg, {C_PRIMARY}{'14' if _is_dark else '0A'}, transparent 42%);
    border-right:1px solid {t['border']};
    box-shadow: {"1px 0 0 rgba(255,255,255,0.03), 8px 0 28px rgba(0,0,0,0.42)" if _is_dark
                 else "1px 0 2px rgba(11,26,38,0.04), 6px 0 22px -8px rgba(11,26,38,0.09)"};
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
    background-color:{"#FFFFFF" if st.session_state.theme == "light" else "#080A0E"} !important;
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
.sidebar-footer .footer-name {{ font-size:12.5px; font-weight:500; color:{t['text']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.sidebar-footer .footer-uni {{ font-family:{FONT_MONO}; font-size:10px; font-weight:400; letter-spacing:0.06em;
    color:{t['text_muted']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }}
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
/* ═══════════════ CABECERA DE PÁGINA ═══════════════
   Registro editorial de publicación científica: antetítulo en versalitas monoespaciadas
   (etiqueta de sección), titular en serif de pantalla, subtítulo en sans, y un filete
   que cierra el bloque y lo separa del contenido. */
.page-eyebrow {{
    font-family:{FONT_MONO}; font-size:11.5px; font-weight:500; letter-spacing:0.16em;
    text-transform:uppercase; color:{C_PRIMARY}; margin-bottom:12px;
    display:flex; align-items:center; gap:10px;
}}
.page-eyebrow::before {{
    content:""; width:18px; height:2px; border-radius:1px; background:{C_PRIMARY}; flex-shrink:0;
}}
.page-title {{
    font-family:{FONT_SERIF}; font-size:46px; font-weight:400; color:{t['text']};
    margin-bottom:12px; letter-spacing:-0.015em; line-height:1.12;
}}
.page-subtitle {{
    font-size:16.5px; font-weight:400; color:{t['text_secondary']}; line-height:1.65;
    max-width:76ch; margin-bottom:14px;
}}
/* Filete de cierre: se desvanece hacia la derecha en vez de cortar en seco */
.page-rule {{
    height:1px; margin-bottom:30px;
    background:linear-gradient(90deg, {t['border_strong']}, {t['border']} 45%, transparent);
}}
.kpi-card, .info-card {{
    background-color:{t['surface']}; border:1px solid {t['border']}; border-radius:14px; padding:20px 22px; height:100%;
    box-shadow: {SHADOW};
    transition: box-shadow 0.2s cubic-bezier(0.4,0,0.2,1), transform 0.2s cubic-bezier(0.4,0,0.2,1), border-color 0.2s ease;
}}
.kpi-card:hover, .info-card:hover {{
    box-shadow: {SHADOW_HOVER};
    border-color:{t['border_strong']};
    transform: translateY(-2px);
}}
/* Tarjeta "lead" (párrafo introductorio): filete lateral degradado de marca a violeta —
   el recorrido clásico→cuántico que resume la tesis. Va como pseudo-elemento y NO con
   border-image: esa propiedad sustituye los cuatro bordes a la vez (pintaría el degradado
   también arriba, abajo y a la derecha) y además anula el border-radius de la tarjeta. */
.lead-card {{ position:relative; overflow:hidden; }}
.lead-card::before {{
    content:""; position:absolute; top:0; bottom:0; left:0; width:2px;
    background:linear-gradient(180deg, {C_PRIMARY}, {C_QUANTUM});
}}
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
    background-color:{t['surface']}; border:1px solid {t['border']}; border-radius:16px;
    padding:20px 28px; box-shadow: {SHADOW};
    transition: box-shadow 0.2s cubic-bezier(0.4,0,0.2,1), transform 0.2s cubic-bezier(0.4,0,0.2,1), border-color 0.2s ease;
}}
.st-key-predictor_gauge_row:hover {{
    box-shadow: {SHADOW_HOVER}; border-color:{t['border_strong']};
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
/* Ítem de la arquitectura Medallón como mini-tarjeta con acento lateral por capa.
   El acento lo fija cada ítem en línea con su paso de la rampa secuencial (Bronze →
   Silver → Gold es una progresión, no tres identidades: le corresponde una escala de
   magnitud, no colores categóricos). */
.medallion-item {{
    display:flex; gap:14px; align-items:flex-start; padding:15px 17px; margin-bottom:10px;
    background:{t['surface']}; border:1px solid {t['border']}; border-left:2px solid {C_PRIMARY};
    border-radius:0 12px 12px 0; box-shadow:{SHADOW};
    transition: box-shadow 0.2s cubic-bezier(0.4,0,0.2,1), transform 0.2s cubic-bezier(0.4,0,0.2,1), border-color 0.2s ease;
}}
.medallion-item:hover {{ box-shadow:{SHADOW_HOVER}; transform:translateY(-2px); }}
.medallion-name {{
    font-family:{FONT_MONO}; font-size:11px; font-weight:600; letter-spacing:0.14em;
    text-transform:uppercase; margin-bottom:5px;
}}
/* Enlace de navegación a Gobernanza desde el bloque Medallón. Es un st.button porque hace
   falta ejecutar código (empujar el índice al option_menu), pero NO debe leerse como botón:
   compite con las tarjetas y rompe el registro editorial de la página. Se despoja del chrome
   y queda como enlace de texto en color de marca, alineado con el filete de las tarjetas. */
.st-key-ir_gobernanza button {{
    background:transparent !important; border:none !important; box-shadow:none !important;
    padding:2px 0 !important; margin-top:4px !important; min-height:0 !important;
    color:{C_PRIMARY} !important; font-weight:500 !important;
    transition: opacity 0.15s ease !important;
}}
.st-key-ir_gobernanza button p {{
    font-family:{FONT_MONO} !important; font-size:11.5px !important; font-weight:500 !important;
    letter-spacing:0.09em !important; text-transform:uppercase !important;
}}
.st-key-ir_gobernanza button:hover {{ opacity:0.72; text-decoration:underline; }}
.st-key-ir_gobernanza button:focus:not(:focus-visible) {{ box-shadow:none !important; }}
/* ═══════════════ GOBERNANZA ═══════════════
   Fila de expectativa validada. El estado NUNCA va solo en color: punto + etiqueta de
   texto ("PASSED"), que es lo que exige el sistema de diseño para los colores de estado
   y lo único que funciona en impresión monocroma o con daltonismo. */
.gov-check {{
    display:grid; grid-template-columns:15px minmax(96px, 128px) 1fr auto; gap:12px;
    align-items:center; padding:9px 2px; border-bottom:1px solid {t['border']};
}}
.gov-check:last-child {{ border-bottom:none; }}
.gov-dot {{ width:7px; height:7px; border-radius:50%; margin-left:4px; flex-shrink:0; }}
.gov-col {{ font-family:{FONT_MONO}; font-size:11.5px; font-weight:500; color:{t['text']};
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.gov-rule {{ font-size:12.5px; color:{t['text_secondary']}; line-height:1.5; }}
.gov-state {{ font-family:{FONT_MONO}; font-size:9.5px; font-weight:600; letter-spacing:0.11em;
    text-transform:uppercase; white-space:nowrap; }}
/* Cabecera de dimensión dentro de la suite */
.gov-dim {{
    font-family:{FONT_MONO}; font-size:10px; font-weight:600; letter-spacing:0.14em;
    text-transform:uppercase; color:{t['text_muted']}; margin:16px 0 2px;
    display:flex; align-items:center; gap:10px;
}}
.gov-dim:first-child {{ margin-top:0; }}
.gov-dim::after {{ content:""; flex:1 1 auto; height:1px; background:{t['border']}; }}
/* Tabla del historial Delta: cifras monoespaciadas, filas separadas por filete fino.
   Con overflow-x propio — en móvil la tabla desborda antes que el cuerpo de la página. */
.gov-table-wrap {{ overflow-x:auto; }}
.gov-table {{ width:100%; border-collapse:collapse; font-size:12.5px; min-width:460px; }}
.gov-table th {{
    font-family:{FONT_MONO}; font-size:9.5px; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:{t['text_muted']}; text-align:left;
    padding:0 14px 9px 0; border-bottom:1px solid {t['border_strong']}; white-space:nowrap;
}}
.gov-table td {{
    padding:9px 14px 9px 0; border-bottom:1px solid {t['border']};
    color:{t['text_secondary']}; white-space:nowrap;
}}
.gov-table tr:last-child td {{ border-bottom:none; }}
.gov-table .num {{ font-family:{FONT_MONO}; font-variant-numeric:tabular-nums; color:{t['text']}; }}
/* Bloque de código dentro de una tarjeta (los asserts de la cadena anti-leakage) */
.gov-code {{
    font-family:{FONT_MONO}; font-size:11px; line-height:1.65; color:{t['text_secondary']};
    background:{t['surface_alt']}; border:1px solid {t['border']}; border-radius:8px;
    padding:10px 12px; margin-top:10px; overflow-x:auto; white-space:pre;
}}
/* Contenedores de gráficas Plotly con la misma profundidad sutil Y el mismo hover de elevación que
   las tarjetas .kpi-card / .info-card, para que TODAS las tarjetas reaccionen igual al pasar el ratón. */
div[data-testid="stPlotlyChart"] {{
    background-color:{t['surface']};
    border:1px solid {t['border']};
    border-radius:14px;
    padding:10px;
    box-shadow: {SHADOW};
    transition: box-shadow 0.2s cubic-bezier(0.4,0,0.2,1), transform 0.2s cubic-bezier(0.4,0,0.2,1), border-color 0.2s ease;
}}
div[data-testid="stPlotlyChart"]:hover {{
    box-shadow: {SHADOW_HOVER};
    border-color:{t['border_strong']};
    transform: translateY(-2px);
}}
/* Tarjeta contenedora de una figura/imagen (SHAP summary, circuito cuántico): mismo aspecto y mismo
   hover de elevación que el resto. Fondo blanco fijo porque esas imágenes lo son. En tema oscuro se
   convierte en "lámina enmarcada" (anillo de marca + sombra profunda + brillo atenuado) para que el
   bloque blanco no flote sobre el fondo oscuro — ver FIG_CARD_SHADOW / FIG_IMG_FILTER. */
.fig-card {{
    background:#FFFFFF; border:1px solid {t['border']}; border-radius:14px; padding:{FIG_CARD_PAD};
    box-shadow: {FIG_CARD_SHADOW};
    transition: box-shadow 0.2s cubic-bezier(0.4,0,0.2,1), transform 0.2s cubic-bezier(0.4,0,0.2,1), border-color 0.2s ease;
}}
.fig-card:hover {{
    box-shadow: {FIG_CARD_SHADOW_HOVER};
    transform: translateY(-2px);
}}
/* Atenúa el fogonazo del blanco puro en oscuro (imperceptible en claro: filter:none). El radio
   redondea la imagen igual que el inline style, por si algún navegador lo ignorara. */
.fig-card img {{ filter: {FIG_IMG_FILTER}; }}
/* ═══════════════ MÉTRICAS ═══════════════
   Toda cifra va en monoespaciada con cifras tabulares: alinea las columnas de valores
   entre tarjetas y evita el salto horizontal al recalcularse. */
.kpi-model {{ font-size:12.5px; font-weight:600; margin-bottom:14px; display:flex; align-items:center; gap:8px;
    letter-spacing:0.01em; color:{t['text']}; }}
.kpi-dot {{ width:8px; height:8px; border-radius:2px; flex-shrink:0; }}
.kpi-row {{ display:flex; justify-content:space-between; align-items:baseline; padding:8px 0; border-bottom:1px solid {t['border']}; }}
.kpi-row:last-child {{ border-bottom:none; padding-bottom:0; }}
.kpi-label {{ font-size:13px; color:{t['text_secondary']}; font-weight:400; }}
.kpi-value {{ font-family:{FONT_MONO}; font-size:15px; font-weight:500; color:{t['text']}; font-variant-numeric:tabular-nums; }}
.kpi-value-auc {{ font-family:{FONT_MONO}; font-size:clamp(22px, 2.5vw, 36px); font-weight:500; color:{t['text']};
    letter-spacing:-0.02em; font-variant-numeric:tabular-nums; line-height:1.05; }}
.stat-num {{ font-family:{FONT_MONO}; font-size:clamp(17px, 2.1vw, 31px); font-weight:500; color:{t['text']};
    white-space:nowrap; line-height:1.05; letter-spacing:-0.02em; font-variant-numeric:tabular-nums; }}
.stat-label {{ font-family:{FONT_MONO}; font-size:10.5px; font-weight:500; letter-spacing:0.11em; text-transform:uppercase;
    color:{t['text_muted']}; margin-top:8px; }}
/* Tarjeta de estadística: altura consistente entre las 4 columnas, contenido alineado abajo.
   El filete superior en color de marca las ata visualmente al sistema. */
.stat-card {{ display:flex; flex-direction:column; justify-content:flex-end; min-height:118px; position:relative; overflow:hidden; }}
.stat-card::before {{
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg, {C_PRIMARY}, {C_PRIMARY}00 85%);
}}
/* Variante cuántica: en Circuito Cuántico el filete va en oro, el tono reservado al
   componente cuántico en toda la aplicación (QSVM, vector de estado en la esfera). */
.stat-card.quantum::before {{ background:linear-gradient(90deg, {C_QUANTUM}, {C_QUANTUM}00 85%); }}
/* ═══════════════ TÍTULOS DE SECCIÓN ═══════════════
   Recurso editorial clásico: el título ocupa lo que necesita y un filete fino recorre el
   resto del ancho hasta el margen, marcando el arranque de bloque sin cargar la página. */
.section-title {{
    font-family:{FONT_SERIF}; font-size:23px; font-weight:400; color:{t['text']};
    margin-bottom:5px; display:flex; align-items:center; gap:14px; letter-spacing:-0.01em; line-height:1.3;
}}
.section-title::after {{
    content:""; flex:1 1 auto; height:1px; min-width:20px;
    background:linear-gradient(90deg, {t['border_strong']}, transparent);
}}
.section-sub {{ font-size:13.5px; color:{t['text_secondary']}; margin-bottom:16px; line-height:1.6; max-width:82ch; }}
.badge {{ display:inline-block; font-family:{FONT_MONO}; font-size:11px; font-weight:500; letter-spacing:0.08em;
    text-transform:uppercase; padding:4px 10px; border-radius:6px;
    background:{C_PRIMARY}1A; color:{C_PRIMARY}; margin-right:6px; }}
/* Nota metodológica: filete lateral en color de marca sobre fondo apenas teñido. Sin
   elevación ni hover — es texto de apoyo, no una tarjeta interactiva; que “saltara” al
   pasar el ratón la ponía al mismo nivel jerárquico que los datos. */
.clinical-note {{
    background:{C_PRIMARY}0D; border:1px solid {C_PRIMARY}26; border-left:2px solid {C_PRIMARY};
    border-radius:0 10px 10px 0; padding:14px 18px 14px 20px;
    font-size:13.5px; color:{t['text_secondary']}; line-height:1.7;
}}
.clinical-note b {{ color:{t['text']}; font-weight:600; }}
/* Matriz de confusión (cuadrícula HTML: celdas legibles, etiquetas horizontales) */
.cm-title {{ font-size:13px; font-weight:600; color:{t['text']}; margin-bottom:16px; display:flex; align-items:center; gap:8px; }}
.cm-grid {{ display:grid; grid-template-columns:64px 1fr 1fr; gap:4px; align-items:stretch; }}
.cm-collabel {{ font-family:{FONT_MONO}; font-size:10px; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
    color:{t['text_muted']}; text-align:center; align-self:end; padding-bottom:7px; line-height:1.45; }}
.cm-rowlabel {{ font-family:{FONT_MONO}; font-size:10px; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
    color:{t['text_muted']}; text-align:right; align-self:center; padding-right:10px; line-height:1.45; }}
/* gap:4px + este anillo del color de la superficie = el separador de 2px que exige el
   sistema de diseño entre rellenos contiguos, para que dos celdas de tono parecido no
   se lean como una sola mancha. */
.cm-cell {{ border-radius:8px; aspect-ratio:1 / 0.72; display:flex; flex-direction:column;
    align-items:center; justify-content:center; box-shadow:inset 0 0 0 1px {t['surface']};
    transition:transform 0.15s ease, box-shadow 0.15s ease; }}
.cm-cell:hover {{ transform:scale(1.025); box-shadow:inset 0 0 0 1px {t['surface']}, {SHADOW}; }}
.cm-num {{ font-family:{FONT_MONO}; font-size:24px; font-weight:500; line-height:1; font-variant-numeric:tabular-nums; }}
.cm-tag {{ font-family:{FONT_MONO}; font-size:9.5px; font-weight:500; letter-spacing:0.12em; margin-top:6px; }}
code {{ background:{t['surface_alt']}; padding:2px 6px; border-radius:5px; font-size:12.5px;
    border:1px solid {t['border']}; font-family:{FONT_MONO}; color:{t['text']}; }}
/* ═══════════════ TABS NATIVOS (st.tabs — Análisis SHAP) ═══════════════
   Etiquetas en monoespaciada versalita: leen como un selector de instrumento, no como
   un menú web genérico. El subrayado activo es de 2px en color de marca. */
button[data-baseweb="tab"] {{
    color:{t['text_secondary']} !important;
    font-family:{FONT_MONO} !important; font-size:11.5px !important; font-weight:500 !important;
    letter-spacing:0.09em !important; text-transform:uppercase !important;
    padding-top:10px !important; padding-bottom:10px !important;
    transition: color 0.15s ease !important;
}}
button[data-baseweb="tab"] p {{
    font-family:{FONT_MONO} !important; font-size:11.5px !important; font-weight:500 !important;
    letter-spacing:0.09em !important; text-transform:uppercase !important;
}}
button[data-baseweb="tab"]:hover {{ color:{t['text']} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{C_PRIMARY} !important; }}
[data-baseweb="tab-highlight"] {{ background-color:{C_PRIMARY} !important; height:2px !important; }}
[data-baseweb="tab-border"] {{ background-color:{t['border']} !important; }}
/* Scrollbar fino y en paleta (WebKit + Firefox): sustituye la barra gruesa del SO por un acabado sobrio.
   Ojo: en Firefox las reglas ::-webkit-scrollbar NO se aplican — todo el acabado lo da la línea
   scrollbar-width / scrollbar-color de abajo, sin estado :hover posible. Por eso el pulgar usa
   border_strong y no border: con el tono de borde (#E1E8EE en claro) quedaba casi invisible sobre
   el fondo, y en Firefox no hay hover que lo rescate. */
*::-webkit-scrollbar {{ width:10px; height:10px; }}
*::-webkit-scrollbar-track {{ background:transparent; }}
*::-webkit-scrollbar-thumb {{ background:{t['border_strong']}; border-radius:8px; border:2px solid {t['bg']}; }}
*::-webkit-scrollbar-thumb:hover {{ background:{C_MID2}; }}
* {{ scrollbar-width:thin; scrollbar-color:{t['border_strong']} transparent; }}
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
    .page-eyebrow {{ font-size:10px; letter-spacing:0.13em; margin-bottom:9px; }}
    .page-title {{ font-size:30px; margin-bottom:9px; }}
    .page-subtitle {{ font-size:14.5px; margin-bottom:10px; }}
    .page-rule {{ margin-bottom:22px; }}
    .kpi-card, .info-card {{ padding:16px 16px; }}
    .stat-card {{ min-height:88px !important; }}
    /* Matriz de confusión: la columna de etiquetas fija en 64 px ahoga las celdas en pantalla estrecha */
    .cm-grid {{ grid-template-columns:52px 1fr 1fr; }}
    .cm-num {{ font-size:19px; }}
    /* El resto de la jerarquía tipográfica también baja un escalón en móvil, en la misma proporción
       que el título/subtítulo de arriba — así todo el texto queda a escala del viewport, no solo la
       cabecera de la página. */
    .section-title {{ font-size:19px; gap:10px; }}
    .section-sub, .clinical-note, .cm-title {{ font-size:12.5px; }}
    .kpi-value {{ font-size:14px; }}
    .kpi-model, .kpi-label {{ font-size:12px; }}
    .stat-label {{ font-size:9.5px; letter-spacing:0.09em; }}
    .badge {{ font-size:10px; }}
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
# El color de cada modelo sale de SERIES (escala categórica): el tono sigue a la ENTIDAD
# y es el mismo en las 6 páginas — tarjeta KPI, curva ROC, matriz de confusión y barras
# agrupadas. Antes eran tres pasos de una misma rampa azul y se confundían entre sí.
MODELS = {
    "lightgbm": {"label": "LightGBM", "color": SERIES["lightgbm"], "auc": 0.9485, "f1_macro": 0.6523,
                 "accuracy": 0.7243, "mcc": 0.4566, "cm": {"tn": 924, "fp": 423, "fn": 9, "tp": 211}},
    "svm_rbf": {"label": "SVM-RBF", "color": SERIES["svm_rbf"], "auc": 0.9377, "f1_macro": 0.8243,
                "accuracy": 0.9075, "mcc": 0.6539, "cm": {"tn": 1250, "fp": 97, "fn": 48, "tp": 172}},
    "qsvm": {"label": "QSVM", "color": SERIES["qsvm"], "auc": 0.5493, "f1_macro": 0.4669,
             "accuracy": 0.8602, "mcc": 0.0625, "cm": {"tn": 1347, "fp": 0, "fn": 219, "tp": 1}},
}
MODEL_ORDER = ["lightgbm", "svm_rbf", "qsvm"]

# ─────────────────────────────────────────────────────────────────────────
# GOBERNANZA Y CALIDAD DEL DATO
# ─────────────────────────────────────────────────────────────────────────
# TODAS las cifras de esta sección se han transcrito de las SALIDAS EJECUTADAS de los
# notebooks (los .ipynb del repositorio conservan sus outputs) y de TECHNICAL_NOTES.md.
# Ninguna es estimada. Origen de cada bloque, indicado en su comentario.
#
# La app no puede leerlas en vivo: Streamlit Community Cloud solo ve el repositorio, no
# Unity Catalog Volumes (ver TECHNICAL_NOTES §6). Igual que los .onnx y los .npy, viajan
# embarcadas. Si se re-ejecuta el pipeline y cambian los conteos, hay que actualizarlas.

# Embudo de registros: NB01 c.16/26 (Bronze) y NB02 c.11/13/15 (filtros Silver).
# El "antes" del filtro DIQ010 es 7.835 y no el 7.831 que imprime la celda: esa celda se
# re-ejecutó sobre el dataframe ya filtrado y su contador de partida quedó desplazado. Los
# 4 registros descartados se confirman con el value_counts de la propia celda
# (6.510 + 1.099 + 222 = 7.831).
GOV_EMBUDO = [
    ("Bronze — 3 ciclos unidos", 29400, None,
     "27 ficheros XPT · join por SEQN · 162 columnas comunes a los tres ciclos"),
    ("Filtro edad ≥ 18 años", 17961, 11439, "Restricción a población adulta"),
    ("Filtro ayuno — LBXGLU no nulo", 7835, 10126,
     "Proxy del subgrupo en ayunas: PHAFSTMN no es consistente entre ciclos"),
    ("Filtro DIQ010 válido", 7831, 4,
     "Descarta los códigos 7 «no sabe» y 9 «rehúsa responder», y los nulos"),
]
# NB02 c.19/21/23/25/27 · NB03 c.8/10/12/14
GOV_SILVER_OPS = [
    ("Variables DIQ excluidas por leakage", "6", "DIQ050, DIQ070, DIQ160, DIQ170, DIQ172, DIQ180"),
    ("Columnas sparse eliminadas", "66", "Umbral de >80 % de valores ausentes"),
    ("Variables winsorizadas", "67", "Recorte de outliers por IQR × 3"),
    ("Missing tras imputación", "0", "De 75.855 a 0 en el dataset SVM/QSVM (mediana + moda)"),
]
GOV_GOLD_OPS = [
    ("Features tras codificación", "106", "One-hot de 5 variables categóricas sobre 84 features"),
    ("Descartadas por correlación", "16", "Umbral r > 0,90 entre pares de predictores"),
    ("Features finales", "89", "Conjunto con el que se entrenan los tres modelos"),
    ("Partición estratificada", "6.264 / 1.567", "80/20 · 14,03 % positivos en train, 14,04 % en test"),
]

# NB07 — suite dataframe-expectations 0.7.0, salida de las celdas 10 y 12.
# Valores de respaldo: son los que la suite produjo el 22/06/2026, transcritos de la salida.
GOV_SUITE = {
    "nombre": "silver_quality_suite", "fecha": "2026-06-22", "registros": 7831,
    "total": 15, "passed": 15, "failed": 0, "pass_rate": 1.0, "duracion_s": 0.001765,
}
# ...pero si el CSV que genera el propio NB07 está embarcado, MANDA él. Así las cifras dejan
# de estar transcritas a mano y no pueden desincronizarse de una reejecución del pipeline:
# basta copiar validacion_silver_dfe.csv desde el volumen a streamlit/models/ (el .gitignore
# tiene la excepción para que suba, como ya la tienen los .onnx y los .npy).
# Si no está, se usan los valores de arriba y la página se comporta igual que antes.
GOV_SUITE_FUENTE = "valores verificados del notebook"
_dfe_csv = MODELS_DIR / "validacion_silver_dfe.csv"
if _dfe_csv.exists():
    try:
        import csv as _csv
        with open(_dfe_csv, encoding="utf-8", newline="") as _fh:
            _fila = next(_csv.DictReader(_fh))
        GOV_SUITE = {
            "nombre": _fila["suite_name"], "fecha": _fila["fecha"],
            "registros": int(_fila["dataframe_rows"]),
            "total": int(_fila["total_expectations"]),
            "passed": int(_fila["total_passed"]), "failed": int(_fila["total_failed"]),
            "pass_rate": float(_fila["pass_rate"]), "duracion_s": float(_fila["duration_seconds"]),
        }
        GOV_SUITE_FUENTE = "leído de validacion_silver_dfe.csv"
    except (KeyError, ValueError, StopIteration, OSError):
        # Un CSV con otro esquema o truncado no debe tumbar la página: se ignora y se
        # mantienen los valores de respaldo, que son los publicados en la memoria.
        pass
# Las 15 expectativas, con la descripción literal que devuelve el runner.
GOV_EXPECTATIVAS = [
    ("Completitud", "TARGET", "como máximo 0 nulos"),
    ("Completitud", "LBXGH", "como máximo 0 nulos"),
    ("Completitud", "LBXGLU", "como máximo 0 nulos"),
    ("Completitud", "RIDAGEYR", "como máximo 0 nulos"),
    ("Completitud", "BMXBMI", "como máximo 0 nulos"),
    ("Rangos clínicos", "RIDAGEYR", "mínimo entre 18 y 25"),
    ("Rangos clínicos", "RIDAGEYR", "máximo entre 70 y 120"),
    ("Rangos clínicos", "LBXGH", "mínimo entre 3,0 y 6,0"),
    ("Rangos clínicos", "LBXGH", "máximo entre 8,0 y 20,0"),
    ("Rangos clínicos", "LBXGLU", "mínimo entre 30 y 80"),
    ("Rangos clínicos", "LBXGLU", "máximo entre 150 y 500"),
    ("Rangos clínicos", "BMXBMI", "mínimo entre 10,0 y 18"),
    ("Rangos clínicos", "BMXBMI", "máximo entre 40,0 y 80"),
    ("Volumen", "DataFrame", "al menos 7.000 filas"),
    ("Volumen", "DataFrame", "como máximo 9.000 filas"),
]

# NB03 c.34 — historial Delta de la capa Gold. Se muestran las 6 versiones más recientes
# de las 10 registradas; el resto se purga a las 168 h por retención de Delta.
GOV_DELTA_HISTORY = [
    (9, "2026-07-16 16:58:32", "WRITE", 7831, 757558),
    (8, "2026-07-14 14:22:20", "WRITE", 7831, 757558),
    (7, "2026-06-23 20:13:23", "WRITE", 7831, 755326),
    (6, "2026-06-23 20:11:48", "WRITE", 7831, 755326),
    (5, "2026-06-21 06:13:12", "WRITE", 7831, 754078),
    (4, "2026-06-21 06:04:25", "WRITE", 7831, 754078),
]

# La cadena de custodia contra la fuga de información, en el orden en que actúa.
GOV_LEAKAGE = [
    ("01", "Exclusión en Silver", "NB02 · celda 9",
     "Se eliminan 6 variables DIQ de tratamiento y seguimiento antes de winsorizar: son "
     "consecuencia del diagnóstico, no predictores de él.", None),
    ("02", "Verificación cruzada", "NB02 · celda 18",
     "Se comprueba que ninguna DIQ sobrevive en los 2 Parquet de Silver ni en los 13 de "
     "Gold. Resultado: 15/15 artefactos limpios.", None),
    ("03", "Filtro defensivo del QSVM", "NB03 · celda 8",
     "Segunda barrera antes de la selección por Random Forest. No descarta ninguna columna "
     "(89 de 89 pasan) — precisamente la prueba de que la primera barrera funcionó.", None),
    ("04", "Guarda de pesos de muestreo", "NB03 · celda 6",
     "Detiene el pipeline si aparece cualquier peso de muestreo distinto del conocido. "
     "WTINT2YR sí llega al modelado y está documentado en TECHNICAL_NOTES 2.10.",
     'pesos = [c for c in X_svm.columns if c.startswith(("WTSAF", "WTMEC", "WTINT"))]\n'
     'assert set(pesos) <= {"WTINT2YR"}, f"Peso de muestreo inesperado: {pesos}"'),
]

# Escalado anti-leakage del StandardScaler: NB03 c.16/18.
GOV_SCALER = [
    ("Ajuste", "Solo sobre train", "fit_transform en train · transform en test"),
    ("Columnas evaluadas", "66", "Con varianza > 0"),
    ("Columnas constantes", "23", "Varianza 0 — ver TECHNICAL_NOTES 2.8"),
    ("Media ≈ 0 · desv. ≈ 1", "Verificado", "Assert sobre todas las columnas con dispersión"),
]

# Inventario de frameworks por capa. El primer elemento de cada lista es el framework
# que vertebra la capa; el resto, los que lo acompañan.
# Color del filete: las TRES CAPAS de datos son una progresión de refinamiento y toman
# pasos de la rampa secuencial (igual que la arquitectura Medallón); los TRES MODELOS son
# identidades y toman su color de SERIES, el mismo que llevan en el resto de la aplicación.
GOV_STACK = [
    ("Bronze", "ingesta", RAMP[1],
     ["boto3", "pyreadstat 1.3.5", "Delta Lake", "Databricks Secrets", "PySpark"],
     "boto3 sustituye a spark.conf, bloqueado en Serverless (TN 2.1). Tres asserts "
     "de integridad: 27/27 ficheros, el join por SEQN no duplica filas, y Delta cuadra "
     "con pandas."),
    ("Silver", "calidad", RAMP[2],
     ["dataframe-expectations 0.7.0", "pandas", "NumPy", "PyArrow", "Delta Lake"],
     "El framework de calidad del TFM. Great Expectations es incompatible con el "
     "entorno (TN 2.3). Suite de 15 expectativas en 3 dimensiones, con evidencia "
     "persistida en CSV."),
    ("Gold", "preparación", RAMP[3],
     ["scikit-learn", "StandardScaler", "RandomForest", "joblib", "Delta time travel"],
     "Escalado ajustado solo sobre train, partición estratificada con semilla fija y "
     "exportación del contrato de serving (scaler y medianas en JSON)."),
    ("LightGBM", "modelo", SERIES["lightgbm"],
     ["LightGBM", "SHAP TreeExplainer", "skl2onnx", "onnxmltools", "GridSearchCV"],
     "Interpretabilidad exacta por algoritmo polinomial sobre las 1.567 instancias de "
     "test, y verificación de que el ONNX reproduce el PKL al 100 %."),
    ("SVM-RBF", "modelo", SERIES["svm_rbf"],
     ["scikit-learn SVC", "SHAP KernelExplainer", "shap.kmeans", "skl2onnx"],
     "SHAP agnóstico al modelo, con coste de horas: se calcula una vez sobre 200 "
     "instancias y se persiste en disco para reutilizarlo."),
    ("QSVM", "modelo", SERIES["qsvm"],
     ["Qiskit 2.5.0", "qiskit-machine-learning 0.9.0", "qiskit-algorithms 0.4.0",
      "ZZFeatureMap", "FidelityQuantumKernel"],
     "Sin soporte ONNX: el formato no admite operaciones cuánticas (TN 2.5). La "
     "trazabilidad recae en un CSV de métricas con los 14 campos de configuración."),
]

# Registro de decisiones — las 11 limitaciones de TECHNICAL_NOTES.md §2, resumidas.
# "critical" marca las que condicionan la arquitectura; "warning", las asumidas y
# documentadas sin corregir; "good", las resueltas sin residuo.
GOV_DECISIONES = [
    ("2.1", "critical", "spark.conf bloqueado en Serverless",
     "La configuración de credenciales AWS por spark.conf.set devuelve CONFIG_NOT_AVAILABLE, "
     "el mecanismo estándar para conectar Spark con S3.",
     "boto3 como cliente alternativo. S3 queda como almacenamiento de origen y Unity Catalog "
     "Volumes como capa de procesamiento."),
    ("2.2", "critical", "MLflow bloqueado en Serverless",
     "La integración nativa de MLflow está deshabilitada en la capa gratuita: no hay registro "
     "de experimentos, métricas ni artefactos.",
     "Doble mecanismo sustitutivo: los transaction logs de Delta Lake aportan versión, "
     "timestamp y métricas de operación; y cada notebook persiste sus métricas en CSV."),
    ("2.3", "critical", "Great Expectations incompatible",
     "Requiere una combinación de pandas/numpy que choca con las versiones fijadas del "
     "runtime serverless (pandas 1.5.3 / numpy 1.23.5).",
     "dataframe-expectations 0.7.0 como alternativa compatible. 15 expectativas sobre Silver "
     "en tres dimensiones. Resultado 15/15, pass rate 1,0."),
    ("2.4", "warning", "QSVM — coste computacional O(n²)",
     "Sobre las 6.264 instancias de train, la matriz de kernel exigiría ~39 millones de "
     "evaluaciones del circuito. Con 1.500 el kernel agota la memoria.",
     "Entrenamiento sobre muestra estratificada de 500 instancias (~22 min) preservando el "
     "ratio 86/14. La evaluación sí usa el test completo, para que las métricas comparen."),
    ("2.5", "warning", "QSVM — sin soporte ONNX nativo",
     "El formato ONNX no admite operaciones cuánticas: ni skl2onnx ni onnxmltools pueden "
     "serializar un kernel basado en simulación de estados.",
     "Serialización con joblib. El modelo requiere el entorno Qiskit para inferencia, por lo "
     "que el QSVM no entra en el Predictor en Vivo."),
    ("2.6", "warning", "Versiones de Qiskit no fijables",
     "immutable_package_constraints.txt de Databricks bloquea la instalación de versiones "
     "concretas, así que no hay reproducibilidad exacta de versión.",
     "El pipeline corre con las versiones del entorno (2.5.0 / 0.9.0 / 0.4.0), cuya API es "
     "compatible, y el notebook las imprime en una celda de verificación."),
    ("2.7", "good", "Pérdida de variables por duración de sesión",
     "Las celdas largas (22 min de entrenamiento, 132 de predicción) pueden agotar la sesión "
     "serverless y llevarse las variables en memoria.",
     "Persistencia inmediata tras cada operación costosa y modo TRAINING_MODE que recarga "
     "desde disco en ejecuciones posteriores."),
    ("2.8", "warning", "Winsorización aplicada a categóricas codificadas",
     "NHANES codifica numéricamente muchas categóricas. Si más del 75 % comparte valor, "
     "IQR = 0, los límites colapsan y clip() convierte la variable en constante. "
     "10 columnas quedaron colapsadas así.",
     "Se documenta sin modificar: corregirlo alteraría Silver, Gold y los tres modelos. Las "
     "columnas constantes no sesgan —el modelo no extrae señal de ellas—, pero pierden "
     "información. Corrección identificada como trabajo futuro."),
    ("2.9", "warning", "Correlación calculada antes de particionar",
     "El filtro r > 0,90 se calcula sobre el dataset completo, así que las 16 columnas "
     "descartadas se deciden usando también las observaciones de test.",
     "Se documenta sin modificar. No afecta al escalado ni a la selección de features del "
     "QSVM, ambos ajustados solo sobre train, pero la selección deja de ser estrictamente "
     "ciega al test."),
    ("2.10", "warning", "Peso de muestreo WTINT2YR entre las features",
     "El join intracíclico duplica WTSAF2YR en tres columnas. WTINT2YR no está en la lista "
     "de exclusión y sobrevive al filtro de correlación: es una de las 89 features.",
     "Se documenta sin modificar y se añade un assert que detecta la aparición de cualquier "
     "OTRO peso. Un peso muestral no es una variable clínica: no filtra el objetivo, pero "
     "deja al modelo apoyarse en el diseño de la encuesta."),
    ("2.11", "good", "El QSVM serializado no es recargable entre versiones",
     "El pickle arrastra el ZZFeatureMap con sus ParameterExpression. Si Qiskit cambia de "
     "versión, la deserialización falla — y Serverless actualiza sin aviso.",
     "La carga va envuelta en try/except: si falla, TRAINING_MODE pasa a True y el notebook "
     "re-entrena en lugar de abortar. Queda operativo en los tres escenarios posibles."),
]

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

    # Gobernanza va en SEGUNDA posición, no al final: el título del TFM es "Integración QML en
    # pipeline DataOps" y la mitad DataOps se sostiene en esta página. Puesta aquí, la app se
    # recorre en el orden real de ejecución —el dato antes que los modelos— y enlaza con el
    # bloque Medallón de Resumen, que es su resumen de tres líneas.
    _MENU_OPTIONS = ["Resumen", "Gobernanza", "Resultados", "Análisis SHAP", "Circuito Cuántico", "Esfera de Bloch", "Predictor en Vivo"]
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
    # Ítem activo con filete izquierdo en color de marca (patrón de navegación de producto:
    # el indicador vive en el borde y no depende solo del relleno, que en tema claro es muy
    # tenue). El resto conserva su color de texto secundario y un fondo neutro al hover.
    nav_link_style = {"font-family": "'IBM Plex Sans', system-ui, sans-serif",
                       "font-size": "13.5px", "text-align": "left", "margin": "3px 0", "padding": "10px 10px",
                       "border-radius": "0 9px 9px 0", "color": t["text_secondary"], "font-weight": "400",
                       "border-left": "2px solid transparent", "transition": "all 0.14s ease",
                       "--hover-color": t["sidebar_active"]}
    nav_link_selected_style = {"background-color": t["sidebar_active"], "color": t["text"], "font-weight": "600",
                                "border-left": f"2px solid {C_PRIMARY}", "border-radius": "0 9px 9px 0"}
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
        icons=["house", "shield-check", "bar-chart", "diagram-3", "cpu", "globe", "sliders"],
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
        # El antetítulo pasa a línea propia sobre el titular (antes iba pegado detrás, como
        # "Resultados: Comparativa triangulada"): así el titular queda limpio en serif y la
        # etiqueta de sección se lee como tal. Mismos textos, distinta jerarquía visual.
        st.markdown(
            f'<div class="page-eyebrow">{eyebrow}</div>'
            f'<div class="page-title">{title}</div>'
            f'<div class="page-subtitle">{subtitle}</div>'
            f'<div class="page-rule"></div>',
            unsafe_allow_html=True)

def _hex_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hex_to_rgba(hex_color, alpha):
    r, g, b = _hex_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

def _rel_luminance(rgb):
    """Luminancia relativa WCAG de un color ya compuesto (canales 0-255)."""
    def _lin(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def ink_over(hex_color, alpha, surface):
    """Tinta legible sobre un relleno SEMITRANSPARENTE, calculada, no supuesta.

    El fondo real de una celda pintada con alfa es la MEZCLA del color con la
    superficie de debajo, no el color a secas. Con la paleta azul anterior daba igual
    —todas las series eran oscuras y el blanco valía siempre—, pero con esta no: en
    tema oscuro la serie de LightGBM es #E9E9E9 y la de QSVM #F9C449, así que una
    tinta blanca fija sería blanco sobre blanco. Aquí se compone la mezcla y se elige
    entre carbón y blanco el que más contraste dé, de modo que la regla sigue siendo
    correcta si mañana se vuelve a cambiar la paleta.

    Devuelve (tinta, tinta_atenuada) para el número y su etiqueta.
    """
    mix = tuple(f * alpha + b * (1 - alpha)
                for f, b in zip(_hex_rgb(hex_color), _hex_rgb(surface)))
    lum = _rel_luminance(mix)
    if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / (_rel_luminance(_hex_rgb(P_CARBON)) + 0.05)):
        return "#FFFFFF", "rgba(255,255,255,0.78)"
    return P_CARBON, hex_to_rgba(P_CARBON, 0.72)

def nf(x, dec=4):
    """Formato numérico español: coma decimal, punto de millar (consistente con la prosa del TFM)."""
    s = f"{x:,.{dec}f}"                       # formato US: '1,234.5678'
    return s.translate(str.maketrans({",": ".", ".": ","}))  # intercambia separadores en una pasada

def pct(x, dec=1):
    """Porcentaje en formato español: '78,2 %' (x en [0,1])."""
    return f"{nf(x * 100, dec)} %"

GRID = hex_to_rgba(t["text_secondary"], 0.16)   # rejilla recesiva: se ve, no compite


def plotly_layout(fig, height=300, **kwargs):
    """Tema común a TODAS las gráficas: rejilla recesiva, ejes sin línea, cifras en
    monoespaciada y esquinas de barra redondeadas (4px, como marca el sistema de diseño).
    Al centralizarlo aquí, cualquier ajuste tipográfico o de rejilla se propaga a las
    seis páginas sin tocarlas una a una."""
    margin = kwargs.pop("margin", dict(l=40, r=16, t=30, b=36))
    # Ejes primero: sin línea de eje ni cero marcado, rejilla punteada tenue y cifras en
    # monoespaciada. Van ANTES del update_layout de abajo a propósito — Plotly fusiona
    # propiedad a propiedad, así que lo último aplicado gana. Si estos valores por defecto
    # se aplicaran después, pisarían el tickfont que fija cada página (p. ej. las etiquetas
    # de categoría de la comparativa de métricas, que son palabras y van en sans, no en
    # monoespaciada). Puestos aquí, son solo el punto de partida y la página manda.
    fig.update_xaxes(showline=False, zeroline=False, griddash="dot", gridwidth=1,
                     tickfont=dict(family=PLOTLY_MONO, size=11))
    fig.update_yaxes(showline=False, zeroline=False, griddash="dot", gridwidth=1,
                     tickfont=dict(family=PLOTLY_MONO, size=11))
    fig.update_layout(
        height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOTLY_FONT, color=t["text_secondary"], size=12),
        separators=",.",  # localización ES: coma decimal / punto de millar en ticks y hover
        # barcornerradius es de layout (no de traza): redondea el extremo de dato de TODAS
        # las barras de la figura, verticales y horizontales.
        barcornerradius=4,
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border_strong"], align="left",
                         font=dict(family=PLOTLY_FONT, size=12, color=t["text"])),
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
        # Bronze → Silver → Gold es una PROGRESIÓN de refinamiento, no tres identidades:
        # le corresponde la rampa secuencial, no colores categóricos. Se toman sus tres
        # pasos altos, así el orden de las capas se lee en el propio color sin leyenda —
        # y con esta paleta la coincidencia es literal, la capa "Gold" acaba en el oro.
        #
        # El color va SOLO en el filete y en la muestra sólida. El nombre y el numeral van
        # en tinta: los tonos cálidos de la paleta dan entre 1,3:1 y 2,0:1 sobre blanco y
        # como texto serían ilegibles. El numeral refuerza el orden sin depender del color.
        #
        # Las descripciones son deliberadamente de UNA línea: el detalle de cada control
        # (expectativas, filtros, linaje) vive en la página Gobernanza y no debe contarse
        # dos veces. Aquí solo se nombra qué hace cada capa; el enlace de abajo lleva al resto.
        layers = [
            ("01", "Bronze", "Ingesta desde AWS S3 sin transformación. Preserva la fuente de verdad.", RAMP[2]),
            ("02", "Silver", "Limpieza, imputación, winsorización y validación de calidad.", RAMP[3]),
            ("03", "Gold", "Escalado, codificación y partición estratificada. Listo para modelar.", RAMP[4]),
        ]
        for num, name, desc, color in layers:
            st.markdown(f"""
            <div class="medallion-item" style="border-left-color:{color};">
                <div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex-shrink:0;margin-top:3px;">
                    <div style="font-family:{FONT_MONO};font-size:11px;font-weight:600;
                                color:{t['text_muted']};letter-spacing:0.05em;">{num}</div>
                    <div style="width:9px;height:9px;border-radius:2px;background:{color};"></div>
                </div>
                <div>
                    <div class="medallion-name" style="color:{t['text']};">{name}</div>
                    <div style="font-size:13px;color:{t['text_secondary']};line-height:1.6;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        # Salto a Gobernanza reutilizando el mismo mecanismo que el toggle de la sidebar:
        # option_menu vive en un iframe y no lee session_state.page por su cuenta, así que
        # hay que empujarle el índice con manual_select (menu_force_index se consume al leerse).
        if st.button("Ver los controles de calidad y linaje  →", key="ir_gobernanza"):
            st.session_state.page = "Gobernanza"
            st.session_state.menu_force_index = _MENU_OPTIONS.index("Gobernanza")
            st.rerun()

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
        # Clase minoritaria en color de marca, mayoritaria en un neutro teñido: el ojo va
        # directo al 14 % (que es el dato relevante) sin que dos tonos saturados compitan.
        fig.add_trace(go.Pie(
            labels=pie_labels, values=pie_values, hole=0.68,
            marker=dict(colors=[hex_to_rgba(t["text_secondary"], 0.22), C_PRIMARY],
                        line=dict(color=t["surface"], width=2)),
            textinfo="label+percent", textposition="outside",
            textfont=dict(size=12, family=PLOTLY_FONT, color=t["text"]),
            insidetextorientation="horizontal", sort=False, automargin=True,
            hoverinfo="skip",
            domain=dict(x=[0.0, 1.0], y=[0.035, 1.0]),
        ))
        # El agujero del donut deja de estar vacío: aloja la cifra protagonista. Es el
        # recurso de “número héroe” — la lectura principal no obliga a interpretar el arco.
        fig.add_annotation(text="14 %", x=0.5, y=0.545, xref="paper", yref="paper", showarrow=False,
                           font=dict(family=PLOTLY_MONO, size=34, color=t["text"]))
        fig.add_annotation(text="DIABETES", x=0.5, y=0.40, xref="paper", yref="paper", showarrow=False,
                           font=dict(family=PLOTLY_MONO, size=10.5, color=t["text_muted"]))
        plotly_layout(fig, height=300, showlegend=False, margin=dict(l=30, r=30, t=45, b=45))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="section-title" style="margin-top:6px;">Comparativa triangulada — objetivo del experimento</div>', unsafe_allow_html=True)
    # Mismos tonos que en Resultados: el color sigue al modelo en toda la aplicación.
    labels3 = [("LightGBM", "Baseline tabular de referencia", SERIES["lightgbm"]),
               ("SVM-RBF", "Puente estructural hacia el componente cuántico", SERIES["svm_rbf"]),
               ("QSVM", "FidelityQuantumKernel — mismo clasificador, kernel cuántico", SERIES["qsvm"])]
    # HTML sin saltos ni indentación: Streamlit trataría las líneas con 4+ espacios como bloque de código.
    _compare_cards = "".join(
        f'<div class="info-card" style="border-top:2px solid {color};">'
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:9px;">'
        f'<span style="width:8px;height:8px;border-radius:2px;background:{color};flex-shrink:0;"></span>'
        f'<span style="font-size:13.5px;font-weight:600;color:{t["text"]};letter-spacing:0.01em;">{name}</span></div>'
        f'<div style="font-size:13px;color:{t["text_secondary"]};line-height:1.6;">{desc}</div></div>'
        for name, desc, color in labels3
    )
    st.markdown(f'<div class="compare-grid">{_compare_cards}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 2 — GOBERNANZA
# ═══════════════════════════════════════════════════════════════════════
elif page == "Gobernanza":
    header("Gobernanza · DataOps", "Gobernanza y Calidad del Dato",
           "Los controles que sostienen el pipeline: qué se valida, qué se descarta y por qué, "
           "qué queda registrado y con qué frameworks. Todas las cifras proceden de las salidas "
           "ejecutadas de los notebooks.")

    tab_calidad, tab_linaje, tab_stack = st.tabs(
        ["Calidad del dato", "Linaje y trazabilidad", "Inventario de frameworks"])

    # ─────────────────────────── TAB A — CALIDAD ───────────────────────────
    with tab_calidad:
        _kpis = [
            (f"{GOV_SUITE['passed']}/{GOV_SUITE['total']}", "Expectativas superadas"),
            (nf(GOV_SUITE["pass_rate"], 1), "Pass rate de la suite"),
            (f"{GOV_SUITE['registros']:,}".replace(",", "."), "Registros validados"),
            ("15/15", "Artefactos sin leakage"),
        ]
        st.markdown(
            '<div class="compare-grid" style="grid-template-columns:repeat(4, minmax(0, 1fr));">'
            + "".join(
                f'<div class="info-card stat-card">'
                f'<div class="stat-num">{v}</div>'
                f'<div class="stat-label">{lab}</div></div>'
                for v, lab in _kpis)
            + "</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Embudo de registros</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">De los 29.400 registros de Bronze sobreviven 7.831 a los '
                     'filtros de cohorte de Silver. Cada escalón responde a un criterio explícito, no a una '
                     'limpieza genérica.</div>', unsafe_allow_html=True)

        _fcol, _ncol = st.columns([1.35, 1], gap="medium")
        with _fcol:
            # Las etapas son una PROGRESIÓN (cada una contiene a la siguiente): magnitud, no
            # identidad. Le corresponde la rampa secuencial, no colores categóricos. Se recorre
            # de RAMP[0] a RAMP[3] para que el refinamiento creciente se lea en el propio color.
            _nombres = [e[0] for e in GOV_EMBUDO][::-1]
            _valores = [e[1] for e in GOV_EMBUDO][::-1]
            _colores = [RAMP[i] for i in range(len(GOV_EMBUDO))][::-1]
            _hover = [_wrap_hover(e[3]) for e in GOV_EMBUDO][::-1]
            _perdidos = [e[2] for e in GOV_EMBUDO][::-1]
            _etiquetas = [f"{v:,}".replace(",", ".") for v in _valores]

            fig = go.Figure(go.Bar(
                x=_valores, y=_nombres, orientation="h",
                marker_color=_colores, cliponaxis=False,
                text=_etiquetas, textposition="outside",
                textfont=dict(family=PLOTLY_MONO, size=12, color=t["text"]),
                customdata=list(zip(_hover, [("—" if p is None else f"{p:,}".replace(",", "."))
                                              for p in _perdidos])),
                hovertemplate="<b>%{y}</b><br>Registros: %{x:,}<br>Descartados: %{customdata[1]}"
                              "<br>%{customdata[0]}<extra></extra>",
            ))
            plotly_layout(fig, height=250, showlegend=False,
                          margin=dict(l=8, r=70, t=10, b=28),
                          xaxis=dict(range=[0, max(_valores) * 1.18], showgrid=True,
                                     gridcolor=GRID, fixedrange=True),
                          yaxis=dict(showgrid=False, fixedrange=True,
                                     tickfont=dict(family=PLOTLY_FONT, size=11.5)))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with _ncol:
            _pasos = "".join(
                f'<div class="kpi-row"><span class="kpi-label">{etapa}</span>'
                f'<span class="kpi-value">'
                f'{"—" if perdidos is None else "−" + f"{perdidos:,}".replace(",", ".")}'
                f'</span></div>'
                for etapa, _v, perdidos, _d in GOV_EMBUDO)
            st.markdown(
                f'<div class="info-card"><div class="kpi-model">Registros descartados por filtro</div>'
                f'{_pasos}'
                f'<div class="kpi-row"><span class="kpi-label">Partición Gold 80/20</span>'
                f'<span class="kpi-value">6.264 / 1.567</span></div></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Suite de validación — dataframe-expectations</div>',
                     unsafe_allow_html=True)
        # El separador de millar se sustituye SOLO en la cifra, no en la frase: aplicar el
        # replace a la cadena entera se llevaría por delante las comas de la prosa.
        _n_reg = f'{GOV_SUITE["registros"]:,}'.replace(",", ".")
        st.markdown(
            f'<div class="section-sub">Suite <code>{GOV_SUITE["nombre"]}</code>, ejecutada el '
            f'{GOV_SUITE["fecha"]} sobre los {_n_reg} registros de Silver en '
            f'{nf(GOV_SUITE["duracion_s"], 4)} segundos. Great Expectations es incompatible con las '
            f'versiones fijadas del runtime serverless: esta es la alternativa adoptada.</div>',
            unsafe_allow_html=True)

        _filas, _dim_previa = [], None
        for dim, col, regla in GOV_EXPECTATIVAS:
            if dim != _dim_previa:
                _n = sum(1 for d, _, _ in GOV_EXPECTATIVAS if d == dim)
                _filas.append(f'<div class="gov-dim">{dim} · {_n}</div>')
                _dim_previa = dim
            _filas.append(
                f'<div class="gov-check">'
                f'<span class="gov-dot" style="background:{STATUS["good"]};"></span>'
                f'<span class="gov-col">{col}</span>'
                f'<span class="gov-rule">{regla}</span>'
                f'<span class="gov-state" style="color:{STATUS["good"]};">passed</span></div>')
        st.markdown(f'<div class="info-card">{"".join(_filas)}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Operaciones de calidad por capa</div>',
                     unsafe_allow_html=True)
        _ocol1, _ocol2 = st.columns(2, gap="medium")
        for _col, _titulo, _ops in ((_ocol1, "Silver — limpieza y saneamiento", GOV_SILVER_OPS),
                                     (_ocol2, "Gold — preparación para modelado", GOV_GOLD_OPS)):
            with _col:
                _rows = "".join(
                    f'<div class="kpi-row" style="align-items:flex-start;">'
                    f'<span class="kpi-label" style="max-width:62%;">{lab}'
                    f'<span style="display:block;font-size:11.5px;color:{t["text_muted"]};'
                    f'line-height:1.5;margin-top:3px;">{det}</span></span>'
                    f'<span class="kpi-value">{val}</span></div>'
                    for lab, val, det in _ops)
                st.markdown(
                    f'<div class="info-card"><div class="kpi-model">'
                    f'<span class="kpi-dot" style="background:{C_PRIMARY};"></span>{_titulo}</div>'
                    f'{_rows}</div>', unsafe_allow_html=True)

    # ────────────────────── TAB B — LINAJE Y TRAZABILIDAD ──────────────────────
    with tab_linaje:
        st.markdown('<div class="section-title">Trazabilidad sin MLflow</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">La restricción que más condiciona la arquitectura del '
                     'pipeline, y su mitigación.</div>', unsafe_allow_html=True)

        _lcol1, _lcol2 = st.columns(2, gap="medium")
        with _lcol1:
            st.markdown(
                f'<div class="info-card" style="border-top:2px solid {STATUS["critical"]};">'
                f'<div class="kpi-model"><span class="kpi-dot" style="background:{STATUS["critical"]};">'
                f'</span>Limitación</div>'
                f'<div style="font-size:13.5px;color:{t["text_secondary"]};line-height:1.7;">'
                f'La integración nativa de <b style="color:{t["text"]};">MLflow</b> está deshabilitada '
                f'en Databricks Serverless gratuito. Cualquier llamada a <code>mlflow.start_run()</code> '
                f'o <code>mlflow.log_metric()</code> produce errores de autenticación: no hay registro '
                f'de experimentos, métricas ni artefactos.</div></div>', unsafe_allow_html=True)
        with _lcol2:
            st.markdown(
                f'<div class="info-card" style="border-top:2px solid {STATUS["good"]};">'
                f'<div class="kpi-model"><span class="kpi-dot" style="background:{STATUS["good"]};">'
                f'</span>Mitigación — doble mecanismo</div>'
                f'<div style="font-size:13.5px;color:{t["text_secondary"]};line-height:1.7;">'
                f'<b style="color:{t["text"]};">Transaction logs de Delta Lake</b> — cada escritura '
                f'genera un registro ACID con versión, marca de tiempo y métricas de operación.<br><br>'
                f'<b style="color:{t["text"]};">CSV de métricas por modelo</b> — cada notebook persiste '
                f'sus resultados en Unity Catalog Volumes, y las figuras los leen de ahí en vez de '
                f'llevarlos escritos a mano.</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Historial Delta — capa Gold</div>',
                     unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Seis versiones más recientes de las diez registradas. '
                     'Delta purga las anteriores tras 168 h de retención, comportamiento esperado y no '
                     'un fallo del pipeline.</div>', unsafe_allow_html=True)
        _hist = "".join(
            f'<tr><td class="num">{v}</td><td class="num">{ts}</td><td>{op}</td>'
            f'<td class="num">{filas:,}</td><td class="num">{bytes_/1024:,.0f} KB</td></tr>'
            .replace(",", ".")
            for v, ts, op, filas, bytes_ in GOV_DELTA_HISTORY)
        st.markdown(
            f'<div class="info-card"><div class="gov-table-wrap"><table class="gov-table">'
            f'<thead><tr><th>Versión</th><th>Timestamp</th><th>Operación</th>'
            f'<th>Filas</th><th>Tamaño</th></tr></thead><tbody>{_hist}</tbody></table></div></div>',
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Cadena de custodia contra la fuga de información</div>',
                     unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Cuatro barreras encadenadas. La tercera no descarta '
                     'ninguna columna — y eso es exactamente lo que se quiere ver: prueba que las '
                     'anteriores hicieron su trabajo.</div>', unsafe_allow_html=True)
        # Mismo componente que la arquitectura Medallón: numeral + acento lateral + descripción.
        # El acento sigue la rampa secuencial porque las barreras son una secuencia, no identidades.
        for _i, (num, nombre, origen, desc, codigo) in enumerate(GOV_LEAKAGE):
            _c = RAMP[_i + 1]
            _code_html = (f'<div class="gov-code">{codigo}</div>' if codigo else "")
            st.markdown(
                f'<div class="medallion-item" style="border-left-color:{_c};">'
                f'<div style="display:flex;flex-direction:column;align-items:center;gap:6px;'
                f'flex-shrink:0;margin-top:3px;">'
                f'<div style="font-family:{FONT_MONO};font-size:11px;font-weight:600;'
                f'color:{t["text_muted"]};letter-spacing:0.05em;">{num}</div>'
                f'<div style="width:9px;height:9px;border-radius:2px;background:{_c};"></div></div>'
                f'<div style="min-width:0;">'
                f'<div class="medallion-name" style="color:{t["text"]};">{nombre}'
                f'<span style="font-weight:400;letter-spacing:0.06em;color:{t["text_muted"]};'
                f'margin-left:10px;text-transform:none;">{origen}</span></div>'
                f'<div style="font-size:13px;color:{t["text_secondary"]};line-height:1.6;">{desc}</div>'
                f'{_code_html}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _scol1, _scol2 = st.columns([1, 1.15], gap="medium")
        with _scol1:
            _srows = "".join(
                f'<div class="kpi-row" style="align-items:flex-start;">'
                f'<span class="kpi-label" style="max-width:56%;">{lab}'
                f'<span style="display:block;font-size:11.5px;color:{t["text_muted"]};'
                f'line-height:1.5;margin-top:3px;">{det}</span></span>'
                f'<span class="kpi-value">{val}</span></div>'
                for lab, val, det in GOV_SCALER)
            st.markdown(
                f'<div class="info-card"><div class="kpi-model">'
                f'<span class="kpi-dot" style="background:{C_PRIMARY};"></span>'
                f'Escalado sin fuga estadística</div>{_srows}</div>', unsafe_allow_html=True)
        with _scol2:
            st.markdown(
                '<div class="clinical-note">El <b>StandardScaler</b> se ajusta exclusivamente sobre '
                '<b>train</b>: <code>fit_transform</code> en entrenamiento y <code>transform</code> en '
                'test. Si se ajustara sobre el conjunto completo, la media y la desviación típica del '
                'test se filtrarían al preprocesado y las métricas quedarían optimistas. La selección '
                'de las 8 variables del QSVM sigue la misma regla — el Random Forest se entrena solo '
                'con <code>X_train_svm_scaled</code>.<br><br>El filtro de correlación, en cambio, '
                '<b>sí</b> se calcula antes de particionar. Está documentado y asumido en '
                'TECHNICAL_NOTES 2.9.</div>', unsafe_allow_html=True)

    # ──────────────────── TAB C — INVENTARIO DE FRAMEWORKS ────────────────────
    with tab_stack:
        st.markdown('<div class="section-title">Frameworks por capa</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">El primer distintivo de cada tarjeta es el framework que '
                     'vertebra la capa; el resto lo acompañan.</div>', unsafe_allow_html=True)

        for _inicio in (0, 3):
            _cards = "".join(
                f'<div class="info-card" style="border-top:2px solid {color};">'
                f'<div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px;'
                f'margin-bottom:11px;">'
                f'<span style="font-size:14px;font-weight:600;color:{t["text"]};">{capa}</span>'
                f'<span style="font-family:{FONT_MONO};font-size:9.5px;font-weight:500;'
                f'letter-spacing:0.12em;text-transform:uppercase;color:{t["text_muted"]};">{rol}</span>'
                f'</div>'
                f'<div style="margin-bottom:12px;line-height:2.1;">'
                + "".join(f'<span class="badge">{b}</span>' for b in badges) + "</div>"
                f'<div style="font-size:12.5px;color:{t["text_secondary"]};line-height:1.65;">{nota}</div>'
                f'</div>'
                for capa, rol, color, badges, nota in GOV_STACK[_inicio:_inicio + 3])
            st.markdown(f'<div class="compare-grid">{_cards}</div>', unsafe_allow_html=True)
            if _inicio == 0:
                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Registro de decisiones</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Las once limitaciones documentadas en TECHNICAL_NOTES, '
                     'con su mitigación. Tres condicionan la arquitectura, seis se asumen y documentan '
                     'sin corregir —porque hacerlo invalidaría los resultados ya obtenidos— y dos quedan '
                     'resueltas sin residuo.</div>', unsafe_allow_html=True)

        _ETIQUETA = {"critical": "Arquitectura", "warning": "Asumida", "good": "Resuelta"}
        for ref, nivel, titulo, problema, solucion in GOV_DECISIONES:
            _c = STATUS[nivel]
            with st.expander(f"{ref} · {titulo}"):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
                    f'<span class="gov-dot" style="background:{_c};margin-left:0;"></span>'
                    f'<span class="gov-state" style="color:{_c};">{_ETIQUETA[nivel]}</span></div>'
                    f'<div style="font-size:13px;color:{t["text_secondary"]};line-height:1.7;">'
                    f'<b style="color:{t["text"]};">Problema · </b>{problema}<br><br>'
                    f'<b style="color:{t["text"]};">Solución adoptada · </b>{solucion}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="clinical-note">Las cifras de esta página proceden de las salidas ejecutadas '
            f'de los notebooks del repositorio y de <code>TECHNICAL_NOTES.md</code>; ninguna es '
            f'estimada. La aplicación no puede consultarlas en vivo porque Streamlit Community Cloud '
            f'solo accede al repositorio, no a Unity Catalog Volumes.<br><br>'
            f'Resumen de la suite de calidad: <b>{GOV_SUITE_FUENTE}</b>.</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 3 — RESULTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Resultados":
    header("Comparativa triangulada", "Resultados",
           "LightGBM vs. SVM-RBF vs. QSVM sobre el mismo conjunto de test (1.567 instancias).")

    cols = st.columns(3)
    for col, key in zip(cols, MODEL_ORDER):
        m = MODELS[key]
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:2px solid {m['color']};">
                <div class="kpi-model"><span class="kpi-dot" style="background:{m['color']}"></span>{m['label']}</div>
                <div class="kpi-value-auc" style="color:{m['color']};">{nf(m['auc'])}</div>
                <div class="stat-label" style="margin:6px 0 14px;">AUC-ROC</div>
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
        # Diagonal de azar como referencia recesiva (punteada, gris): es el marco de lectura,
        # no una serie más. Sin entrada de leyenda ni hover.
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(color=hex_to_rgba(t["text_secondary"], 0.35), width=1, dash="dot"),
                                  showlegend=False, hoverinfo="skip"))
        # Marca fina (2px) y relleno tenue: el área sugiere la magnitud del AUC sin tapar la curva.
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=m["color"], width=2, shape="spline", smoothing=0.4),
                                  fill="tozeroy", fillcolor=hex_to_rgba(m["color"], 0.13), name=m["label"],
                                  hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
        # Una sola serie por gráfica: el título la nombra y no hace falta caja de leyenda.
        plotly_layout(fig, height=250, showlegend=False, hovermode="x unified",
                      title=dict(text=f"{m['label']} · AUC {nf(m['auc'])}", x=0.01, xanchor="left",
                                 font=dict(family=PLOTLY_FONT, size=13, color=t["text"])),
                      xaxis=dict(title=dict(text="FPR", font=dict(size=11)), range=[0, 1], showgrid=False, fixedrange=True),
                      yaxis=dict(title=dict(text="TPR", font=dict(size=11)), range=[0, 1], showgrid=True, gridcolor=GRID, fixedrange=True))
        with col:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matrices de confusión</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Valores exactos verificados contra el classification report de cada modelo</div>', unsafe_allow_html=True)

    def cm_cell(val, total, tag, color):
        """Intensidad = proporción sobre la fila (magnitud): rampa de un solo tono, el del
        modelo. El texto NUNCA va en el color de la serie — la tinta se CALCULA sobre la
        mezcla real de relleno y tarjeta (ver ink_over), no por un umbral fijo de
        proporción, que era lo que fallaba con series claras."""
        prop = val / total if total else 0.0
        alpha = 0.10 + 0.82 * prop
        bg = hex_to_rgba(color, alpha)
        txt, sub = ink_over(color, alpha, t["surface"])
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
                              # Etiqueta directa sobre cada barra: los tres modelos quedan identificados
                              # por texto además de por color, así la lectura no depende del tono.
                              text=[nf(m[k], 3) for k in metric_keys], textposition="outside",
                              textfont=dict(family=PLOTLY_MONO, size=12, color=t["text_secondary"]),
                              customdata=[[nf(m[k], 3), _wrap_hover(metric_desc[k])] for k in metric_keys],
                              hovertemplate="<b>%{fullData.name}</b> · %{x} = %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    # bargap/bargroupgap: barras finas con un carril de superficie entre ellas — dos rellenos
    # contiguos nunca se tocan, que es lo que hace legible un grupo de tres.
    plotly_layout(fig, height=460, barmode="group", bargap=0.42, bargroupgap=0.08,
                  legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
                              font=dict(family=PLOTLY_FONT, size=12.5, color=t["text"]),
                              bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
                  yaxis=dict(range=[0, 1.10], showgrid=True, gridcolor=GRID, fixedrange=True),
                  xaxis=dict(showgrid=False, tickfont=dict(family=PLOTLY_FONT, size=13, color=t["text"]), fixedrange=True))
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
            x=values, y=names, orientation="h", marker_color=color, cliponaxis=False,
            # El valor va en monoespaciada y en tinta secundaria, no en el color de la barra:
            # el texto nunca lleva el color de la serie (lo aporta la propia barra al lado).
            text=[nf(v) for v in values], textposition="outside",
            textfont=dict(family=PLOTLY_MONO, size=10.5, color=t["text_secondary"]),
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        ))
        plotly_layout(fig, height=520, hovermode="y", bargap=0.34,
                      xaxis=dict(title=dict(text="mean(|SHAP value|)", font=dict(size=11)),
                                 showgrid=True, gridcolor=GRID, range=[0, max(values) * 1.3], fixedrange=True),
                      yaxis=dict(tickfont=dict(family=PLOTLY_MONO, size=11, color=t["text"]), fixedrange=True),
                      margin=dict(l=170, r=60, t=20, b=40))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(f'<div class="section-sub" style="margin-top:10px;">Pasa el cursor sobre cada barra para ver el significado de la variable. {sample_note}</div>', unsafe_allow_html=True)

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
        # Cada pestaña usa el color de SU modelo (antes las dos iban en el mismo azul de marca):
        # así el ranking se lee sin ambigüedad como perteneciente a LightGBM o a SVM-RBF.
        shap_chart(SHAP_LIGHTGBM, SERIES["lightgbm"], "Valores exactos (algoritmo polinomial) sobre las 1.567 instancias del test.")
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
        shap_chart(SHAP_SVMRBF, SERIES["svm_rbf"], "Valores aproximados por muestreo: fondo de 100 instancias, contribuciones sobre 200 instancias de test.")
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
            st.markdown(f'<div class="info-card stat-card quantum" style="min-height:102px;"><div class="stat-num" style="font-size:clamp(19px, 2.3vw, 32px);">{num}</div><div class="stat-label">{lab}</div></div>', unsafe_allow_html=True)

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
            marker_color=hex_to_rgba(t["text"], 0.07), marker_line_width=0,
            hoverinfo="skip", showlegend=False,
        ))
        # Violeta: son las 8 variables que alimentan el QSVM, así que llevan el acento
        # cuántico y no el azul de marca — la página entera queda cosida al componente.
        fig.add_trace(go.Bar(
            x=values, y=names, orientation="h", marker_color=C_QUANTUM, cliponaxis=False,
            text=[nf(v) for v in values], textposition="outside",
            textfont=dict(family=PLOTLY_MONO, size=10.5, color=t["text_secondary"]),
            customdata=customdata, showlegend=False,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
        ))
        plotly_layout(fig, height=300, barmode="overlay", bargap=0.34,
                      # Margen izquierdo reducido (150→95): pega las etiquetas al borde de la tarjeta
                      # en vez de dejarlas centradas con aire de sobra. Al ser el ancho de la tarjeta
                      # fijo, ese espacio liberado pasa directo al área de barras — se agrandan solas,
                      # de forma proporcional, sin tocar la tarjeta que las contiene.
                      xaxis=dict(title=dict(text="Importancia RF", font=dict(size=11)),
                                 showgrid=True, gridcolor=GRID, range=[0, max(values) * 1.3], fixedrange=True),
                      yaxis=dict(tickfont=dict(family=PLOTLY_MONO, size=11, color=t["text"]), fixedrange=True),
                      margin=dict(l=95, r=70, t=20, b=40))
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
        # La esfera es el CONTENEDOR, no el dato: va en un azul de marca muy diluido para
        # que no compita con el vector. El dato (|ψ⟩) es lo único en violeta saturado.
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs, opacity=0.14, showscale=False, hoverinfo="skip",
            colorscale=[[0, RAMP[0]], [1, RAMP[2]]],
            lighting=dict(ambient=0.66, diffuse=0.9, specular=0.22, roughness=0.6, fresnel=0.25),
            lightposition=dict(x=120, y=200, z=160),
        ))
        # Círculos máximos (ecuador + 2 meridianos): refuerzan la curvatura al rotar
        circ = np.linspace(0, 2 * np.pi, 120)
        for gx, gy, gz in [
            (np.cos(circ), np.sin(circ), np.zeros_like(circ)),   # ecuador (plano XY)
            (np.cos(circ), np.zeros_like(circ), np.sin(circ)),   # meridiano XZ
            (np.zeros_like(circ), np.cos(circ), np.sin(circ)),   # meridiano YZ
        ]:
            fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", opacity=0.42,
                                        line=dict(color=C_MID1, width=1.2), showlegend=False, hoverinfo="skip"))
        # Ejes cartesianos
        for ax_x, ax_y, ax_z in [([-1.06,1.06],[0,0],[0,0]), ([0,0],[-1.06,1.06],[0,0]), ([0,0],[0,0],[-1.10,1.10])]:
            fig.add_trace(go.Scatter3d(x=ax_x, y=ax_y, z=ax_z, mode="lines",
                                        line=dict(color=t["border_strong"], width=1.5), showlegend=False, hoverinfo="skip"))
        # Vector de estado |ψ⟩ (φ = 0 → contenido en el plano XZ), en el acento cuántico
        px, py, pz = np.sin(theta), 0.0, np.cos(theta)
        fig.add_trace(go.Scatter3d(x=[0, px], y=[0, py], z=[0, pz], mode="lines",
                                    line=dict(color=C_QUANTUM, width=7), showlegend=False, hoverinfo="skip"))
        # Punta de flecha (cono) apuntando hacia afuera a lo largo del vector
        fig.add_trace(go.Cone(x=[px], y=[py], z=[pz], u=[px], v=[py], w=[pz],
                              sizemode="absolute", sizeref=0.18, anchor="tip", showscale=False,
                              colorscale=[[0, C_QUANTUM], [1, C_QUANTUM]], hoverinfo="skip"))
        # Proyección vertical al plano ecuatorial (pista de profundidad sutil)
        fig.add_trace(go.Scatter3d(x=[px, px], y=[py, py], z=[pz, 0], mode="lines", opacity=0.42,
                                    line=dict(color=C_QUANTUM, width=2, dash="dot"), showlegend=False, hoverinfo="skip"))
        # Punto del estado: anillo del color de la superficie alrededor del marcador, para que
        # se despegue de la esfera cuando el vector queda por delante de ella.
        fig.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz], mode="markers",
                                    marker=dict(size=7, color=C_QUANTUM,
                                                line=dict(color=t["surface"], width=2)), showlegend=False,
                                    hovertemplate=f"|ψ⟩ ({var_code})<extra></extra>"))
        # Etiquetas de los polos
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[1.08,-1.08], mode="text", text=["|0⟩","|1⟩"],
                                    textfont=dict(family=PLOTLY_MONO, size=14, color=t["text_secondary"]),
                                    showlegend=False, hoverinfo="skip"))
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

    # Categoría interpretable respecto al umbral de decisión (50%). Aquí sí procede la
    # paleta de ESTADO (bien / atención / grave): es una lectura de riesgo, no una serie
    # de datos, y estos tonos están reservados a ese uso en toda la aplicación. La
    # categoría se nombra siempre por texto además de por color — nunca solo color.
    if risk < 0.33:
        cat, cat_color, interp = ("Bajo", STATUS["good"],
            "Los valores introducidos sitúan el score claramente por debajo del umbral de decisión (50%).")
    elif risk < 0.5:
        cat, cat_color, interp = ("Moderado", STATUS["warning"],
            "El score se aproxima al umbral de decisión (50%): zona de incertidumbre.")
    else:
        cat, cat_color, interp = ("Elevado", STATUS["critical"],
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
            <div class="stat-label" style="margin:0 0 10px;">Score de riesgo (sustituto)</div>
            <div class="kpi-value-auc" style="color:{cat_color}; font-size:clamp(30px, 4vw, 52px);">{pct(risk)}</div>
            <div style="margin-top:14px;">
                <span style="display:inline-flex; align-items:center; gap:8px; font-family:{FONT_MONO};
                      font-size:11px; font-weight:600; letter-spacing:0.09em; text-transform:uppercase;
                      padding:6px 13px; border-radius:7px; background:{cat_color}1A;
                      border:1px solid {cat_color}40; color:{cat_color};">
                    <span style="width:7px; height:7px; border-radius:2px; background:{cat_color};"></span>
                    Riesgo {cat}
                </span>
            </div>
            <div style="font-size:12.5px; color:{t['text_secondary']}; margin-top:16px; line-height:1.65;">{interp}</div>
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
        # Las bandas son MAGNITUD (score creciente): rampa secuencial de un solo tono, la
        # validada. La lectura de estado la aporta la aguja, que sí va en color de estado —
        # así el arco no se convierte en un semáforo y el dial se mantiene sobrio.
        band_colors = list(RAMP)
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
                               font=dict(size=10.5, color=t["text_muted"], family=PLOTLY_MONO))

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
                          font=dict(family=PLOTLY_FONT, color=t["text"]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(f'<div class="section-sub" style="text-align:center; margin-top:-6px;">Zonas: bajo · moderado · alto · &nbsp;línea = umbral de decisión (50%)</div>', unsafe_allow_html=True)
