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
import html
import io
import json
import re
import textwrap
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote_plus

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components   # solo para fijar <html lang>; ver más abajo
from PIL import Image
from streamlit_option_menu import option_menu

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import i18n                                   # catálogo de textos ES/EN (módulo local)

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

# ─────────────────────────────────────────────────────────────────────────
# TIRA DE TECNOLOGÍAS (página Resumen)
# ─────────────────────────────────────────────────────────────────────────
# EL ORDEN ES EL DEL PIPELINE, no el alfabético ni el de importancia: la tira se lee como
# se ejecuta el proyecto. AWS guarda el origen; Delta Lake sostiene las tres capas que
# Spark escribe; ONNX serializa lo ya entrenado; GitHub y Streamlit son la entrega. Así
# cuenta la misma historia que el bloque Medallón de encima, pero de un vistazo.
#
# AWS va con UNA sola entrada (el logotipo corporativo) y no con los iconos de servicio de
# S3 e IAM: en una tira que resume el proyecto entero, dos de las doce casillas para dos
# servicios del mismo proveedor desequilibraban la lectura. El detalle de qué servicio hace
# qué está en Gobernanza, que es donde se puede explicar.
#
# Cada entrada es (fichero, nombre). El nombre no se traduce —son marcas— y viaja en alt y
# title: da el tooltip y deja la tira utilizable con lector de pantalla, que si no
# encontraría una docena de imágenes sin nombre.
#
# Los ficheros que todavía no están se SALTAN en silencio, igual que hace shap_summary_image
# con las figuras del beeswarm: la tira se completa sola en cuanto se añadan, sin tocar código.
TECH_STACK = [
    ("tech-aws.svg",          "Amazon Web Services"),
    ("tech-databricks.png",   "Databricks"),
    ("tech-spark.png",        "Apache Spark"),
    ("tech-delta-lake.png",   "Delta Lake"),
    ("tech-python.png",       "Python"),
    ("tech-scikit-learn.png", "scikit-learn"),
    # Los tres modelos, en el orden de MODEL_ORDER, y cada uno detrás del framework que lo
    # implementa: scikit-learn → SVM-RBF, Qiskit → QSVM. LightGBM es las dos cosas a la vez.
    # SVM-RBF y QSVM no son productos y no tienen marca: el primero lleva un icono propio y
    # el segundo el logotipo del TFM, el mismo de la barra lateral.
    ("tech-lightgbm.svg",     "LightGBM"),
    ("tech-svm.png",          "SVM-RBF"),
    ("tech-qiskit.svg",       "Qiskit"),
    ("tech-qsvm.png",         "QSVM"),
    ("tech-onnx.svg",         "ONNX"),
    # Git y GitHub son dos cosas distintas y van seguidas en ese orden: el control de
    # versiones primero y el alojamiento después, que es como se usan.
    ("tech-git.png",          "Git"),
    ("tech-github.svg",       "GitHub"),
    ("tech-streamlit.svg",    "Streamlit"),
]
TECH_ALTO_BASE = 32.0   # alto en pantalla de un isotipo cuadrado
TECH_ALTO_MIN = 21.0    # suelo para el más apaisado

# Marcas que existen de verdad en disco, y cuántas van por fila. La tira se pinta en DOS
# filas equilibradas, así que la anchura de columna es la mitad del recuento redondeando
# hacia arriba: con 14 marcas salen 7 y 7. Se calcula en vez de escribir un 7 fijo para que
# siga partiendo en dos filas parejas si mañana entra o sale alguna.
TECH_N = sum(1 for fichero, _ in TECH_STACK if (ASSETS_DIR / fichero).exists())
TECH_POR_FILA = (TECH_N + 1) // 2


@st.cache_data
def _tech_alto(ruta: str) -> float:
    """Alto en pantalla del logo, normalizado por ÁREA ÓPTICA y no por caja.

    A igual altura, un logotipo apaisado ocupa mucha más superficie que un isotipo
    cuadrado y desequilibra la fila: el de GitHub mide 3,7:1 y se leería como casi cuatro
    veces el de Qiskit. Igualando el área —alto = base / √proporción— los doce pesan lo
    mismo a la vista, que es lo que se pidió. El suelo evita que aplicar la regla a
    rajatabla deje el más apaisado en un hilo ilegible.
    """
    p = Path(ruta)
    if p.suffix.lower() == ".svg":
        # El tamaño intrínseco de un SVG es su viewBox; width/height pueden no estar.
        vb = re.search(r'viewBox="\s*([-\d.eE]+)[,\s]+([-\d.eE]+)[,\s]+([-\d.eE]+)[,\s]+([-\d.eE]+)',
                       p.read_text(encoding="utf-8"))
        ancho, alto = (float(vb.group(3)), float(vb.group(4))) if vb else (1.0, 1.0)
    else:
        with Image.open(p) as im:
            ancho, alto = im.size
    proporcion = (ancho / alto) if alto else 1.0
    return max(TECH_ALTO_MIN, TECH_ALTO_BASE / (proporcion ** 0.5))


def tech_strip():
    """Pinta la tira. Se llama desde Resumen, ya con el tema resuelto."""
    piezas = []
    for fichero, nombre in TECH_STACK:
        ruta = ASSETS_DIR / fichero
        if not ruta.exists():
            continue
        mime = "svg+xml" if ruta.suffix.lower() == ".svg" else "png"
        piezas.append(
            f'<div class="tech-chip" title="{nombre}">'
            f'<img src="data:image/{mime};base64,{_b64_image(str(ruta))}" alt="{nombre}" '
            f'style="height:{_tech_alto(str(ruta)):.1f}px;"></div>')
    if piezas:
        st.markdown(f'<div class="tech-strip">{"".join(piezas)}</div>', unsafe_allow_html=True)

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
if "lang" not in st.session_state:
    st.session_state.lang = i18n.DEFAULT_LANG

_is_dark = st.session_state.theme == "dark"
LANG = st.session_state.lang

def S(key):
    """Texto de la clave en el idioma activo, con caída al español si falta.

    La caída no es una red de seguridad genérica: es lo que permite traducir la app
    PÁGINA A PÁGINA. Mientras STR["en"] no tenga las claves de Gobernanza o del
    Predictor, esas páginas se siguen pintando en español en vez de tumbar la app
    con un KeyError, y el idioma va llegando a cada una según se traduce.
    """
    catalogo = i18n.STR[LANG]
    return catalogo[key] if key in catalogo else i18n.STR["es"][key]

def _flag_uri(lang):
    """SVG de bandera como data-URI en base64, listo para background-image.

    En base64 y no con el SVG en crudo: dentro de la hoja de estilos el marcado
    llevaría comillas, almohadillas de color y signos de mayor/menor que habría que
    escapar dos veces (una para el CSS, otra para el f-string). En base64 no hay
    nada que escapar y el resultado es idéntico.
    """
    return "data:image/svg+xml;base64," + base64.b64encode(
        i18n.FLAG_SVG[lang].encode("utf-8")).decode("ascii")

# El "otro" idioma se precalcula aquí y no dentro de la hoja de estilos: la CSS es un
# f-string gigante y meterle una condicional lo vuelve ilegible justo donde ya hay
# llaves escapadas. Con esto, las reglas de bandera activa/inactiva se leen solas.
LANG_OTRO = "en" if LANG == "es" else "es"
FLAG_ES_URI = _flag_uri("es")
FLAG_EN_URI = _flag_uri("en")

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

# ── Relieve del buscador (neumorfismo) ────────────────────────────────────────
# Dos sombras opuestas en vez de una: la oscura abajo-derecha y la clara arriba-izquierda
# simulan un único foco alto a la izquierda, y la pastilla lee como EXTRUIDA de la barra en
# lugar de apoyada encima. El efecto solo funciona si la pieza comparte color con su fondo
# —una caja de otro color vuelve a leerse como caja—, así que NEU_BG se separa de sidebar_bg
# lo justo para que el relieve tenga de dónde salir, y el contorno lo dibujan las sombras.
# La asimetría entre temas no es un descuido: en claro el papel tiene recorrido hacia el
# blanco y la luz puede ir casi opaca (0,92), mientras que en oscuro cualquier blanco por
# encima del 4-5% sobre #0E1116 se lee como niebla gris y no como luz. Allí el relieve lo
# sostiene la sombra negra, que sí tiene todo el rango por delante.
# NEU_INSET es el mismo relieve del revés (hundido) y marca el estado activo: es el gesto
# propio de este lenguaje —el control se PULSA— y evita añadir un cerco de color que
# rompería el monocromo.
NEU_BG     = "#151920" if _is_dark else "#EDEDEC"
NEU_RAISED = ("5px 5px 11px rgba(0,0,0,0.55), -5px -5px 11px rgba(255,255,255,0.045)" if _is_dark
              else "5px 5px 11px rgba(11,26,38,0.14), -5px -5px 11px rgba(255,255,255,0.92)")
NEU_INSET  = ("inset 3px 3px 7px rgba(0,0,0,0.62), inset -3px -3px 7px rgba(255,255,255,0.05)"
              if _is_dark else
              "inset 3px 3px 7px rgba(11,26,38,0.16), inset -3px -3px 7px rgba(255,255,255,0.95)")

# ── Luz del lienzo: dos halos en las esquinas superiores ──────────────────────
# En OSCURO la luz se hace sumando color: sobre un fondo casi negro un ámbar al 12% sube
# la luminancia y el ojo lo lee como un halo. En CLARO ese mismo gesto no funciona — la
# saturación cuesta luminancia, así que cualquier tono de marca sobre papel lo OSCURECE y
# lee como suciedad, no como luz. Por eso en claro se hace al revés: la esquina sube hacia
# el blanco y la temperatura la pone la crema, el tono que la paleta reserva justo para
# esto (1,33:1 sobre blanco — inservible como texto, perfecto como tinte).
# El segundo halo va en blanco puro: sobre un papel cálido (#F4F3F0) el blanco lee como luz
# FRÍA, y esa diferencia de temperatura entre las dos esquinas es todo el efecto. El oro
# cuántico no aparece aquí porque en claro no puede hacer de luz; se queda donde sí se
# lee, en las gráficas y los filetes.
# Sobre papel el margen es estrechísimo: el lienzo ya está en L 0,896 y el techo absoluto es
# el blanco (1,0), así que ningún halo claro puede pasar de ~1,11:1 contra el fondo — el
# oscuro llega a 1,22:1 porque parte de casi cero y tiene todo el rango por delante. Por eso
# la crema entra solo al 12% en el blanco: cada punto de saturación que se le añade cuesta
# luminancia, y lo que se busca aquí es luz cálida, no un tinte.
C_LUZ = "#FFFBF1"   # la crema disuelta en blanco. Como ella, nunca lleva texto.
HALOS = (f"radial-gradient(1100px 520px at 12% -8%, {C_PRIMARY}1E, transparent 60%),"
         f"radial-gradient(900px 460px at 100% 0%, {C_QUANTUM}16, transparent 62%)"
         if _is_dark else
         f"radial-gradient(1100px 560px at 12% -8%, {C_LUZ}F2, transparent 64%),"
         f"radial-gradient(900px 460px at 100% 0%, #FFFFFFCC, transparent 62%)")
# Mismo principio en el velo de la barra lateral: en oscuro tiñe, en claro alumbra.
VELO_SIDEBAR = (f"linear-gradient(180deg, {C_PRIMARY}14, transparent 42%)" if _is_dark
                else f"linear-gradient(180deg, {C_LUZ}8C, transparent 46%)")

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
    /* Dos halos muy tenues en las esquinas superiores: dan profundidad al fondo plano sin
       introducir una textura visible. En oscuro evitan que el fondo lea como negro muerto;
       en claro alumbran el papel. Cómo se construye cada uno, y por qué no puede ser el
       mismo recurso en los dos temas, está razonado donde se definen (HALOS). */
    background-image: {HALOS};
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
    /* Velo vertical muy tenue: el panel gana profundidad y se distingue del lienzo
       principal sin recurrir a un borde marcado. En oscuro tiñe de marca; en claro
       alumbra, por el mismo motivo que los halos del lienzo (ver VELO_SIDEBAR). */
    background-image:{VELO_SIDEBAR};
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
/* La cursiva del rótulo lleva el nombre accesible del botón ("Colapsar la barra lateral"):
   se recorta, no se apaga. display:none y visibility:hidden lo borrarían también del árbol de
   accesibilidad —justo lo que se quiere conservar—, y font-size:0 deja un nodo de tamaño cero
   que cada lector trata a su manera. El recorte de 1px es el patrón estándar para esto: sigue
   siendo texto renderizado, así que se anuncia, pero no ocupa ni se ve. En absolute, además,
   sale del flujo y la flecha se queda centrada en su círculo de 24px como estaba. */
.st-key-toggle_sidebar button p em {{
    position:absolute !important;
    width:1px !important; height:1px !important;
    margin:-1px !important; padding:0 !important;
    overflow:hidden !important; clip-path:inset(50%) !important;
    white-space:nowrap !important;
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
/* ── Selector de idioma: dos banderas fijas en la esquina superior derecha ──
   Van en el lienzo principal y no en la sidebar a propósito: el idioma afecta a TODA la
   aplicación, no solo a la navegación, y colapsar la sidebar no debe esconderlo. Como el
   <header> nativo de Streamlit está en visibility:hidden (ver más abajo), esa franja
   superior derecha está libre y no hay nada con lo que chocar.

   Los dos contenedores se disuelven con display:contents en LOS DOS niveles (el de Streamlit
   y el del botón), no con el height:0 que usa el toggle de colapso. El motivo es que el
   bloque vertical de Streamlit es un flex con gap: un hijo de altura cero sigue siendo hijo y
   sigue cobrando su hueco, así que height:0 habría dejado dos huecos en blanco por encima del
   titular de la página. Con display:contents el único elemento que queda es el <button>, y
   como es position:fixed ni siquiera cuenta como ítem del flex: no ocupa absolutamente nada.

   La bandera es un background-image: así el botón sigue siendo un botón de Streamlit
   (accesible, con su tooltip y su foco de teclado) y la bandera es solo su piel; poner un
   <img> dentro habría exigido HTML, que no es pulsable. */
.st-key-lang_es, .st-key-lang_en,
.st-key-lang_es div[data-testid="stButton"],
.st-key-lang_en div[data-testid="stButton"] {{ display:contents !important; }}
.st-key-lang_es button, .st-key-lang_en button {{
    position:fixed !important; top:14px !important;
    width:26px !important; height:18px !important; min-height:18px !important;
    padding:0 !important; margin:0 !important;
    border-radius:3px !important; border:1px solid {t['border']} !important;
    background-repeat:no-repeat !important; background-position:center !important;
    background-size:cover !important;
    box-shadow: 0 1px 3px rgba(20,30,40,0.18) !important;
    z-index:1001 !important;
    transition: opacity 0.16s ease, filter 0.16s ease, transform 0.14s ease,
                box-shadow 0.16s ease !important;
}}
/* El texto del botón es un espacio en blanco (la etiqueta real viaja en el tooltip):
   se colapsa a 0 para que no empuje la bandera ni asome bajo ella. */
.st-key-lang_es button p, .st-key-lang_en button p {{ font-size:0 !important; line-height:0 !important; }}
.st-key-lang_es button {{ right:56px !important; background-image:url("{FLAG_ES_URI}") !important; }}
.st-key-lang_en button {{ right:22px !important; background-image:url("{FLAG_EN_URI}") !important; }}
/* El idioma INACTIVO se apaga (medio desaturado y traslúcido) y el activo va a plena tinta
   con un anillo en color de marca. Es el mismo criterio que el ítem activo del menú: el
   estado se lee por contraste entre las dos, no por un adorno añadido. */
.st-key-lang_{LANG_OTRO} button {{
    opacity:0.42 !important; filter:grayscale(0.55) !important;
}}
.st-key-lang_{LANG_OTRO} button:hover {{
    opacity:1 !important; filter:grayscale(0) !important; transform:scale(1.08);
    border-color:{C_PRIMARY} !important;
}}
.st-key-lang_{LANG} button {{
    opacity:1 !important; filter:none !important;
    border-color:{C_PRIMARY} !important;
    box-shadow: 0 0 0 1.5px {C_PRIMARY}66, 0 1px 3px rgba(20,30,40,0.18) !important;
    cursor:default !important;
}}
/* ── Reloj: fecha y hora, a la izquierda de las banderas ──────────────────────
   Comparte la franja de las banderas (top:14px, alto 18px) y se alinea al mismo eje
   vertical, así los tres elementos leen como una sola tira de cabecera y no como piezas
   sueltas. La bandera ES ocupa de right:56 a right:82, de modo que el reloj cierra en 94:
   doce píxeles de aire, el mismo respiro que hay entre las dos banderas.
   El contenido lo escribe un <script> en el documento padre (ver bloque RELOJ más abajo);
   aquí solo vive su aspecto, que así hereda el tema como cualquier otra regla.
   pointer-events:none porque es información, no un control: no debe capturar el cursor
   ni interponerse en un clic dirigido a lo que tenga debajo. */
#tfm-reloj {{
    position:fixed; top:14px; right:94px; height:18px;
    display:flex; align-items:center; gap:6px;
    z-index:1001; pointer-events:none; user-select:none;
    font-size:11.5px; line-height:1; white-space:nowrap;
}}
#tfm-reloj .r-fecha {{
    font-family:{FONT_SANS}; color:{t['text_muted']};
    letter-spacing:0.02em;
}}
#tfm-reloj .r-sep {{ color:{t['text_muted']}; opacity:0.45; }}
/* La hora en mono y con cifras tabulares: sin tabular-nums el ancho de cada dígito cambia
   y el reloj «baila» un par de píxeles a cada minuto, que en un elemento fijo se nota
   mucho más que en una tabla. Va un punto más marcada que la fecha porque es el dato que
   se consulta; la fecha acompaña. */
#tfm-reloj .r-hora {{
    font-family:{FONT_MONO}; color:{t['text_secondary']};
    font-weight:500; font-variant-numeric:tabular-nums;
    letter-spacing:0.04em;
}}
/* Tira de tecnologías (Resumen).
   LA PASTILLA ES CLARA EN LOS DOS TEMAS a propósito, y no es un descuido: la mitad de estas
   marcas son monocromas oscuras —el logotipo de GitHub es #11110F y el de Qiskit #010101, y
   el texto de Spark y de scikit-learn tampoco aguanta— así que sobre el fondo carbón
   desaparecerían. Recolorearlas no es una opción: las guías de marca lo prohíben. Se les da
   entonces el fondo claro para el que fueron diseñadas y la marca viaja intacta; en tema
   oscuro la pastilla baja un punto de blanco para no deslumbrar sobre el carbón.
   El alto de cada imagen lo calcula _tech_alto() por área óptica — ver allí el porqué. */
/* Rejilla de ancho completo, no una fila suelta: las pastillas se reparten TODO el ancho de
   la página en columnas iguales. Con auto-fit las columnas sobrantes se colapsan y las que
   quedan se estiran para ocupar el hueco, así la banda llena la línea sea cual sea el número
   de marcas. Mismo recurso (auto-fit + minmax) que ya usa el bloque "Cómo funciona" del
   Circuito Cuántico.
   POR DEFECTO reparte por ancho (auto-fit), que es lo único que se adapta al contenedor real.
   SOLO con la ventana bien ancha se fija el número exacto de columnas ({TECH_POR_FILA}), que es
   lo que produce las DOS FILAS PAREJAS de {TECH_POR_FILA}: dejarlo en manos de auto-fit las
   partiría por donde cupieran, no por la mitad.
   El umbral se mide sobre el VIEWPORT pero lo que importa es el CONTENEDOR, y la barra lateral
   se lleva 270px: a 1100px de ventana el contenido son 660px, donde las columnas fijas
   aplastaban las pastillas. De ahí que la regla exacta empiece en 1400 y no en 1024. */
.tech-strip {{
    display:grid; grid-template-columns:repeat(auto-fit, minmax(72px, 1fr));
    gap:10px; align-items:center; margin-top:4px;
}}
@media (min-width: 1400px) {{
    .tech-strip {{ grid-template-columns:repeat({TECH_POR_FILA}, minmax(0, 1fr)); }}
}}
.tech-chip {{
    display:flex; align-items:center; justify-content:center;
    /* Alto subido de 46 a 52: al estirarse a ancho completo las pastillas pasan de ~66 px a
       ~125 px de ancho, y con el alto anterior quedaban como cápsulas aplastadas. */
    height:52px; padding:0 11px; box-sizing:border-box;
    background:{'#E9E9EC' if _is_dark else '#F7F7F8'};
    border:1px solid rgba(28,31,38,{0.16 if _is_dark else 0.09});
    border-radius:9px;
    transition: transform 0.14s ease, box-shadow 0.16s ease;
}}
.tech-chip:hover {{ transform:translateY(-2px); box-shadow:0 4px 12px rgba(20,30,40,0.16); }}
.tech-chip img {{ display:block; width:auto; max-width:100%; object-fit:contain; }}
/* Los dos componentes que solo llevan <script> —el que fija <html lang> y el que escribe el
   reloj— no pintan nada y sus iframes sobran. Se colapsan con el MISMO recurso que las
   banderas —display:contents en los envoltorios y position:fixed en el elemento—, así no
   cuentan como ítem del flex ni abren un hueco sobre el titular. No se usa display:none a
   propósito: un iframe así puede no llegar a ejecutar su script, y aquí el script ES todo
   el contenido. */
.st-key-lang_attr, .st-key-lang_attr div[data-testid="stIFrame"],
.st-key-lang_attr div[data-testid="stElementContainer"],
.st-key-reloj, .st-key-reloj div[data-testid="stIFrame"],
.st-key-reloj div[data-testid="stElementContainer"] {{ display:contents !important; }}
.st-key-lang_attr iframe, .st-key-reloj iframe {{
    position:fixed !important; width:0 !important; height:0 !important;
    border:0 !important; opacity:0 !important; pointer-events:none !important;
}}
/* Footer fijo al fondo de la sidebar (por debajo de la cápsula de tema) */
.sidebar-footer {{
    position:fixed; bottom:0; left:0; width:{SIDEBAR_WIDTH};
    padding:8px 6px 10px; text-align:center; box-sizing:border-box;
    border-top:1px solid {t['border']}; background-color:{t['sidebar_bg']};
    color:{t['text_secondary']}; overflow:hidden; z-index:997; line-height:1.35;
    transition: width 0.32s cubic-bezier(0.4,0,0.2,1);
}}
.sidebar-footer .footer-name {{ font-size:12.5px; font-weight:500; color:{t['text']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.sidebar-footer .footer-uni {{ font-family:{FONT_MONO}; font-size:12px; font-weight:400; letter-spacing:0.06em;
    color:{t['text_muted']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }}
/* El option_menu vive en un iframe con fondo propio: igualarlo al de la sidebar (sin caja/sombra
   propia — ya tiene el mismo fondo, así que se funde visualmente con el resto de la sidebar) */
section[data-testid="stSidebar"] iframe {{ background-color:{t['sidebar_bg']} !important; }}
/* ── Ancho del iframe del menú ────────────────────────────────────────────────
   Streamlit dimensiona el iframe de un componente a partir del ancho de su bloque
   contenedor y se lo clava como atributo `width`. Con la barra colapsada ese cálculo
   daba 23 px (medido por CDP) dentro de una barra de 84: el iframe lleva overflow:clip,
   así que el icono —18 px que empiezan en x=18— se cortaba en x=23. Eso, y no un
   line-height, era el recorte: por la DERECHA, no por abajo.
   Se corrige en los dos escalones: se le devuelve al contenido de la barra el ancho
   completo (su relleno lateral de 20 px se reparte ya en cada bloque) y se obliga al
   iframe a ocuparlo. Solo en modo colapsado — en modo ancho el cálculo de Streamlit es
   correcto y no hay nada que forzar. */
{'''section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
    padding-left:0 !important; padding-right:0 !important;
}
section[data-testid="stSidebar"] iframe[data-testid="stCustomComponentV1"] {
    width:100% !important; min-width:100% !important; max-width:100% !important;
}''' if narrow else ''}
section[data-testid="stSidebar"] div[data-testid="stIFrame"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(iframe) {{
    background-color:{t['sidebar_bg']} !important;
}}
/* El menú arranca separado del buscador que ahora tiene encima: sin este aire, la caja de
   búsqueda y el primer ítem del menú se tocaban y se leían como un solo bloque. */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(iframe) {{
    margin-top:14px !important;
}}
/* ── Buscador de la sidebar ──────────────────────────────────────────────────
   Pastilla en relieve (ver NEU_* arriba): terminaciones ovaladas, sin borde, y el contorno
   dibujado solo por las dos sombras. Reposo = extruida; foco = hundida. El radio va en
   999px y no en un valor fijo para que las tapas sigan siendo semicírculos exactos aunque
   cambie el alto — es lo que separa una pastilla de un rectángulo muy redondeado. */
.st-key-nav_search {{ margin-top:2px; }}
/* Lupa del modo colapsado: la MISMA pastilla reducida a círculo. En 84 px no cabe caja de
   texto, así que la entrada se repliega a su icono y al pulsarla despliega la barra (la
   lógica vive en el bloque BUSCADOR). Conserva los 38 px de los iconos del menú de debajo
   para que la columna siga leyéndose alineada. El rótulo del botón es un espacio en blanco
   (la etiqueta real viaja en el tooltip), así que se anula su hueco. */
.st-key-search_expand button {{
    width:38px !important; height:38px !important; min-height:38px !important;
    padding:0 !important; margin:2px auto 0 !important;
    border-radius:999px !important; border:none !important;
    background:{NEU_BG} !important; color:{t['text_secondary']} !important;
    box-shadow:{NEU_RAISED} !important;
    transition: box-shadow 0.18s ease, color 0.18s ease !important;
}}
/* Se hunde al pasar por encima y se queda hundida al pulsar: el mismo par reposo/activo de
   la caja expandida, para que colapsar la barra no cambie el lenguaje del control. */
.st-key-search_expand button:hover {{
    box-shadow:{NEU_INSET} !important; color:{C_PRIMARY} !important;
    background:{NEU_BG} !important;
}}
.st-key-search_expand button:active {{ box-shadow:{NEU_INSET} !important; }}
.st-key-search_expand button p {{ font-size:0 !important; margin:0 !important; }}
.st-key-search_expand button span[data-testid="stIconMaterial"] {{
    font-size:18px !important; margin:0 !important; color:inherit !important;
}}
.st-key-nav_search div[data-baseweb="input"],
.st-key-nav_search div[data-baseweb="base-input"] {{
    background:{NEU_BG} !important;
    border-radius:999px !important;
}}
/* El alto se fija aquí y el relleno se reparte a mano: 16px a la izquierda (0,42 × alto) es
   lo que hace falta para que el texto no se meta en la curva de la tapa, y 14px a la derecha
   dejan al icono a la misma distancia óptica del extremo. Con menos, la pastilla se lee
   apretada; con más, en 270px de barra el hueco de escritura se queda corto. */
.st-key-nav_search div[data-baseweb="input"] {{
    display:flex !important; align-items:center !important;
    height:38px !important; min-height:38px !important;
    padding:0 14px 0 16px !important;
    border:none !important;
    box-shadow:{NEU_RAISED} !important;
    transition: box-shadow 0.18s ease !important;
}}
.st-key-nav_search div[data-baseweb="input"]:focus-within {{
    box-shadow:{NEU_INSET} !important;
}}
/* La lupa a la DERECHA como en la referencia. Streamlit la inyecta como startEnhancer de
   BaseWeb, o sea primer hijo del contenedor flex; no hay parámetro para moverla, así que se
   reordena con `order` en vez de reimplementar el widget con HTML suelto (que perdería el
   binding de session_state del que dependen el vaciado por callback y el colapso). */
.st-key-nav_search [data-testid="stTextInputIcon"] {{
    order:2 !important; margin:0 0 0 8px !important; padding:0 !important;
    color:{t['text_muted']} !important;
    transition: color 0.18s ease !important;
}}
.st-key-nav_search div[data-baseweb="input"]:focus-within [data-testid="stTextInputIcon"] {{
    color:{C_PRIMARY} !important;
}}
.st-key-nav_search div[data-baseweb="base-input"] {{
    flex:1 1 auto !important; min-width:0 !important; padding:0 !important;
}}
.st-key-nav_search input {{
    font-family:{FONT_SANS} !important; font-size:12.5px !important;
    letter-spacing:0.05em !important;
    color:{t['text']} !important; background:transparent !important;
    padding:0 !important; height:auto !important;
}}
/* La referencia pone su placeholder en versalita muy espaciada, y ahí NO se la sigue: su
   "SEARCH…" son seis letras y el nuestro son treinta y una ("Buscar en el panel o en la
   web…", que dice a propósito que la caja busca en los dos sitios). Con caja alta y 0,11em
   esa frase pide ~260px y en la pastilla solo hay ~174 de texto — 270 de barra menos 40 de
   relleno, menos el interior y la lupa. Se conserva el aire de la referencia con un
   espaciado corto, y el ellipsis asegura que si el idioma alarga la frase degrade en puntos
   suspensivos en vez de cortarse a hachazo, igual que las filas de resultados. */
.st-key-nav_search input {{
    text-overflow:ellipsis !important;
}}
.st-key-nav_search input::placeholder {{
    color:{t['text_muted']} !important;
    letter-spacing:0.04em !important;
    font-size:11.5px !important;
    opacity:1 !important;
}}
/* El foco de teclado global (regla `input:focus-visible` de más arriba) dibuja un rectángulo
   de 8px de radio que en una pastilla se ve como un cerco desalineado. Aquí se sustituye por
   un aro que sigue la forma, sobre el estado hundido. El :has() acota el aro al foco POR
   TECLADO: al hacer clic basta con el hundido y la lupa en ámbar. */
.st-key-nav_search input:focus-visible {{ outline:none !important; }}
.st-key-nav_search div[data-baseweb="input"]:has(input:focus-visible) {{
    box-shadow:{NEU_INSET}, 0 0 0 2px {C_PRIMARY} !important;
}}
/* Resultados: lista de opciones, no botonera. Se anula el centrado que la sidebar
   impone a todos sus botones y se les quita la caja — un borde por resultado sería
   siete cajas apiladas donde solo hace falta una zona de hover. */
.st-key-nav_search_results {{ gap:1px !important; margin-top:6px; }}
.st-key-nav_search_results div[data-testid="stButton"] {{
    display:block !important; width:100% !important;
}}
.st-key-nav_search_results button {{
    width:100% !important; min-height:0 !important;
    padding:7px 10px !important; margin:0 !important;
    text-align:left !important; justify-content:flex-start !important;
    background:transparent !important; border:1px solid transparent !important;
    border-radius:8px !important;
    transition: background-color 0.13s ease, color 0.13s ease !important;
}}
.st-key-nav_search_results button:hover {{
    background:{t['sidebar_active']} !important;
    border-color:{t['border']} !important;
}}
.st-key-nav_search_results button p {{
    font-size:12.5px !important; font-weight:400 !important; line-height:1.35 !important;
    color:{t['text_secondary']} !important;
    /* Un rótulo largo se recorta con puntos suspensivos en vez de partirse en dos
       líneas: la lista tiene que conservar el mismo alto por fila para leerse. */
    white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
    display:block !important; text-align:left !important;
}}
.st-key-nav_search_results button:hover p {{ color:{t['text']} !important; }}
.search-none {{
    font-size:12px; color:{t['text_muted']}; padding:6px 10px 2px; line-height:1.4;
}}
/* Salida a la web: se separa de los resultados locales con un filete, porque es una
   acción de otra naturaleza — abandona la aplicación. */
a.search-web {{
    display:flex; align-items:center; gap:6px;
    margin-top:6px; padding:8px 10px 2px;
    border-top:1px solid {t['border']};
    font-size:12.5px; color:{t['text_secondary']}; text-decoration:none !important;
    transition: color 0.13s ease;
}}
a.search-web:hover {{ color:{C_PRIMARY}; }}
a.search-web .search-web-ext {{ font-size:11px; opacity:0.75; }}
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
    font-family:{FONT_MONO}; font-size:13.5px; font-weight:500; letter-spacing:0.16em;
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
/* La entradilla ocupa el ancho completo del contenido (sin la medida de lectura de 76ch)
   y se justifica a ambos márgenes, alineándose con el filete de cierre y las tarjetas. */
.page-subtitle {{
    font-size:16.5px; font-weight:400; color:{t['text_secondary']}; line-height:1.65;
    margin-bottom:14px; text-align:justify; text-justify:inter-word;
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
/* El párrafo y el realce de sus términos clave se visten AQUÍ y no con estilos en línea
   dentro del texto. El párrafo vive en i18n.py, uno por idioma, y con el color incrustado
   en cada <b> la traducción se leería entre atributos CSS repetidos una docena de veces —
   imposible de revisar de corrido y con un color que habría que cambiar en los dos idiomas
   a la vez. Con estas dos reglas, el texto traducido es prosa con <b> y nada más. */
.lead-card p {{
    font-size:16px; color:{t['text_secondary']}; line-height:1.7; margin:0; text-align:justify;
}}
.lead-card p b {{ color:{t['text']}; font-weight:600; }}
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
    font-family:{FONT_MONO}; font-size:13px; font-weight:600; letter-spacing:0.14em;
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
    font-family:{FONT_MONO} !important; font-size:13.5px !important; font-weight:500 !important;
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
.gov-col {{ font-family:{FONT_MONO}; font-size:13.5px; font-weight:500; color:{t['text']};
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.gov-rule {{ font-size:12.5px; color:{t['text_secondary']}; line-height:1.5; }}
.gov-state {{ font-family:{FONT_MONO}; font-size:11.5px; font-weight:600; letter-spacing:0.11em;
    text-transform:uppercase; white-space:nowrap; }}
/* Cabecera de dimensión dentro de la suite */
.gov-dim {{
    font-family:{FONT_MONO}; font-size:12px; font-weight:600; letter-spacing:0.14em;
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
    font-family:{FONT_MONO}; font-size:11.5px; font-weight:600; letter-spacing:0.12em;
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
    font-family:{FONT_MONO}; font-size:13px; line-height:1.65; color:{t['text_secondary']};
    background:{t['surface_alt']}; border:1px solid {t['border']}; border-radius:8px;
    padding:10px 12px; margin-top:10px; overflow-x:auto; white-space:pre;
}}
/* Prosa dentro de tarjeta (el par Limitación / Mitigación de MLflow). Va al mismo cuerpo que
   el .section-sub de encima porque aquí el texto ES el contenido de la sección, no una nota al
   pie de una tabla — y por eso encoge con él en móvil en vez de quedarse fijo en píxeles.
   El título sube en la misma proporción (13 → 14.5) para no aplanar la jerarquía. */
.gov-prose {{ font-size:15px; color:{t['text_secondary']}; line-height:1.7; text-align:justify; }}
.kpi-model.gov-prose-title {{ font-size:14.5px; }}
/* Los dos párrafos de "Cómo funciona" (Circuito Cuántico). Mismo criterio que .lead-card p:
   el realce de los <b> lo pone el CSS y no un style incrustado en cada uno, así el texto
   llega de i18n como prosa limpia y se revisa de corrido en los dos idiomas. */
.qc-prose {{ font-size:15px; color:{t['text_secondary']}; line-height:1.75; margin:0; text-align:justify; }}
.qc-prose b {{ color:{t['text']}; font-weight:600; }}
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
/* Plotly pinta el título de figura (.gtitle) en NEGRITA por defecto, y ese título es el nombre
   del modelo de cada curva ROC. Se devuelve a peso 400 para que coincida con el mismo nombre en
   la tarjeta KPI, en la matriz de confusión y en la leyenda de la comparativa. Se hace por CSS y
   no con font.weight en plotly_layout porque el atributo depende de la versión de plotly.js que
   empaqueta Streamlit; la clase .gtitle es estable y solo la llevan las curvas ROC, que son las
   únicas figuras de la aplicación con título propio (los rótulos de eje son .xtitle / .ytitle). */
div[data-testid="stPlotlyChart"] .gtitle {{ font-weight:400 !important; }}
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
/* 13px / 400: mismas medidas que el nombre del modelo en el resto de tarjetas de Resultados
   (ver .cm-title más abajo). La clase se reutiliza en Gobernanza y en el Predictor, así que
   ese renglón de encabezado queda igual en toda la aplicación, que es lo coherente. */
.kpi-model {{ font-size:13px; font-weight:400; margin-bottom:14px; display:flex; align-items:center; gap:8px;
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
.stat-label {{ font-family:{FONT_MONO}; font-size:12.5px; font-weight:500; letter-spacing:0.11em; text-transform:uppercase;
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
/* ...y el contenido se ancla ARRIBA, no abajo. El flex-end de .stat-card alinea las cifras entre
   sí solo mientras todas las etiquetas ocupen las mismas líneas —que es lo que pasa en Gobernanza →
   Calidad del dato—. Aquí no: "Qubits (feature_dimension)" envuelve a tres líneas y "Entanglement"
   a una, así que cada cifra caía a una altura distinta. Anclando arriba, las cuatro cifras (y la
   primera línea de cada etiqueta) quedan a la misma altura sea cual sea el largo del texto. */
.stat-card.quantum {{ justify-content:flex-start; }}
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
.section-sub {{ font-size:15px; color:{t['text_secondary']}; margin-bottom:16px; line-height:1.6; text-align:justify; }}
.badge {{ display:inline-block; font-family:{FONT_MONO}; font-size:13px; font-weight:500; letter-spacing:0.08em;
    text-transform:uppercase; padding:4px 10px; border-radius:6px;
    background:{C_PRIMARY}1A; color:{C_PRIMARY}; margin-right:6px; }}
/* Nota metodológica: filete lateral en color de marca sobre fondo apenas teñido. Sin
   elevación ni hover — es texto de apoyo, no una tarjeta interactiva; que “saltara” al
   pasar el ratón la ponía al mismo nivel jerárquico que los datos. */
.clinical-note {{
    background:{C_PRIMARY}0D; border:1px solid {C_PRIMARY}26; border-left:2px solid {C_PRIMARY};
    border-radius:0 10px 10px 0; padding:14px 18px 14px 20px;
    font-size:13.5px; color:{t['text_secondary']}; line-height:1.7; text-align:justify;
}}
.clinical-note b {{ color:{t['text']}; font-weight:600; }}
/* Matriz de confusión (cuadrícula HTML: celdas legibles, etiquetas horizontales) */
/* El nombre del modelo se escribe IGUAL en los cuatro sitios donde aparece en Resultados
   —tarjeta KPI, título de la curva ROC, matriz de confusión y leyenda de la comparativa—:
   13px, peso 400 y color de texto principal. Es el mismo dato repetido, así que no debe
   cambiar de aspecto al bajar por la página. Sin negrita en ninguno: la identifica el punto
   de color de al lado, no el grosor de la letra. */
.cm-title {{ font-size:13px; font-weight:400; color:{t['text']}; margin-bottom:16px; display:flex; align-items:center; gap:8px; }}
.cm-grid {{ display:grid; grid-template-columns:64px 1fr 1fr; gap:4px; align-items:stretch; }}
.cm-collabel {{ font-family:{FONT_MONO}; font-size:12px; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
    color:{t['text_muted']}; text-align:center; align-self:end; padding-bottom:7px; line-height:1.45; }}
.cm-rowlabel {{ font-family:{FONT_MONO}; font-size:12px; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
    color:{t['text_muted']}; text-align:right; align-self:center; padding-right:10px; line-height:1.45; }}
/* gap:4px + este anillo del color de la superficie = el separador de 2px que exige el
   sistema de diseño entre rellenos contiguos, para que dos celdas de tono parecido no
   se lean como una sola mancha. */
.cm-cell {{ border-radius:8px; aspect-ratio:1 / 0.72; display:flex; flex-direction:column;
    align-items:center; justify-content:center; box-shadow:inset 0 0 0 1px {t['surface']};
    transition:transform 0.15s ease, box-shadow 0.15s ease; }}
.cm-cell:hover {{ transform:scale(1.025); box-shadow:inset 0 0 0 1px {t['surface']}, {SHADOW}; }}
.cm-num {{ font-family:{FONT_MONO}; font-size:22px; font-weight:500; line-height:1; font-variant-numeric:tabular-nums; }}
.cm-tag {{ font-family:{FONT_MONO}; font-size:11.5px; font-weight:500; letter-spacing:0.12em; margin-top:6px; }}
code {{ background:{t['surface_alt']}; padding:2px 6px; border-radius:5px; font-size:12.5px;
    border:1px solid {t['border']}; font-family:{FONT_MONO}; color:{t['text']}; }}
/* ═══════════════ TABS NATIVOS (st.tabs — Análisis SHAP) ═══════════════
   Etiquetas en monoespaciada versalita: leen como un selector de instrumento, no como
   un menú web genérico. El subrayado activo es de 2px en color de marca. */
button[data-baseweb="tab"] {{
    color:{t['text_secondary']} !important;
    font-family:{FONT_MONO} !important; font-size:13.5px !important; font-weight:500 !important;
    letter-spacing:0.09em !important; text-transform:uppercase !important;
    padding-top:10px !important; padding-bottom:10px !important;
    transition: color 0.15s ease !important;
}}
button[data-baseweb="tab"] p {{
    font-family:{FONT_MONO} !important; font-size:13.5px !important; font-weight:500 !important;
    letter-spacing:0.09em !important; text-transform:uppercase !important;
}}
button[data-baseweb="tab"]:hover {{ color:{t['text']} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{C_PRIMARY} !important; }}
[data-baseweb="tab-highlight"] {{ background-color:{C_PRIMARY} !important; height:2px !important; }}
[data-baseweb="tab-border"] {{ background-color:{t['border']} !important; }}
/* ═══════════════ EXPANDER (Gobernanza · Registro de decisiones) ═══════════════
   Único widget nativo que quedaba sin vestir, y en tema oscuro se rompía: config.toml no fija
   `base`, así que Streamlit pinta el expander con su tema CLARO (barra casi blanca), mientras
   que el rótulo hereda el color de .stApp, que en oscuro es gris claro. Rótulo claro sobre
   barra clara = invisible; solo se leía al pasar el ratón, porque el hover sí aplica acento.
   Aquí se pinta con la paleta de la app, de modo que sigue al tema en ambos sentidos. La
   cabecera va en surface_alt para que se distinga del cuerpo desplegado (surface) tanto en
   claro como en oscuro, sin depender de una sombra. */
[data-testid="stExpander"] {{
    background:{t['surface']} !important;
    border:1px solid {t['border']} !important;
    border-radius:10px !important;
    overflow:hidden;
}}
[data-testid="stExpander"] summary {{
    background:{t['surface_alt']} !important;
    color:{t['text']} !important;
    border-radius:0 !important;
}}
/* El rótulo lo renderiza Streamlit como markdown dentro del summary: sin fijar el color aquí,
   el <p> interior se queda con el del tema base y vuelve a desaparecer. */
[data-testid="stExpander"] summary p {{ color:{t['text']} !important; }}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:hover p {{ color:{C_PRIMARY} !important; }}
/* Flecha de plegado: en tinta secundaria, y acompaña al acento en hover. */
[data-testid="stExpander"] summary svg {{ fill:{t['text_secondary']} !important; }}
[data-testid="stExpander"] summary:hover svg {{ fill:{C_PRIMARY} !important; }}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ background:{t['surface']} !important; }}
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
/* EXCEPCIÓN: los contenedores que Streamlit pone alrededor de cada elemento (stElementContainer)
   y de cada columna (stColumn) nunca deben mostrar carril. Las tarjetas de gráfica llevan
   padding:10px y desbordan por 20px el alto que esos contenedores reservan para la figura, así
   que sacaban una barra vertical gris en el pasillo a la derecha de cada tarjeta —fuera del
   recuadro, por eso no la quitaba ninguna regla aplicada a stPlotlyChart—. Se suprime SOLO el
   carril: ni height ni overflow, que es lo que dispara el bucle de medición de Plotly y estira
   las gráficas sin fin (mismo motivo que el comentario de la esfera de Bloch más arriba).
   scrollbar-width NO se hereda, así que esto afecta a estos dos contenedores y a nada de su
   interior: el scroll horizontal de .gov-table-wrap y .gov-code se conserva intacto. Gana a la
   regla "*" de arriba por especificidad, no por orden. En Firefox scrollbar-width:none es lo
   único que quita el carril (::-webkit-scrollbar no aplica allí); la regla WebKit cubre
   Chrome/Edge. Al liberar ese hueco, el aire de la derecha vuelve a igualar al de la izquierda. */
div[data-testid="stElementContainer"], div[data-testid="stColumn"] {{ scrollbar-width:none; }}
div[data-testid="stElementContainer"]::-webkit-scrollbar,
div[data-testid="stColumn"]::-webkit-scrollbar {{ width:0 !important; height:0 !important; }}
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
    /* El !important NO es decorativo: las filas de 4 KPIs (Gobernanza → Calidad del dato y Circuito
       Cuántico) declaran su repeat(4, ...) en un style EN LÍNEA, que gana a cualquier selector sin
       !important. Sin esto la regla no llegaba a aplicarse nunca en esas dos filas y seguían a cuatro
       columnas hasta los 768 px: con la sidebar todavía fija en 270 px, cuatro tarjetas de ~135 px.
       Las filas que ya declaran repeat(2, ...) en línea no cambian; las de 3 por defecto bajan a 2. */
    .compare-grid {{ grid-template-columns:repeat(2, minmax(0, 1fr)) !important; }}
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
    /* Las banderas SÍ siguen fijas en móvil (no son de la sidebar, son del lienzo), pero se
       arriman al borde y encogen un punto. El botón nativo de abrir la sidebar ocupa la
       esquina superior IZQUIERDA, así que no hay colisión posible. */
    .st-key-lang_es button, .st-key-lang_en button {{
        top:10px !important; width:23px !important; height:16px !important; min-height:16px !important;
    }}
    .st-key-lang_es button {{ right:44px !important; }}
    .st-key-lang_en button {{ right:14px !important; }}
    /* El reloj sigue a las banderas a su nueva posición y se queda SOLO con la hora: en un
       teléfono la franja superior es estrecha y la fecha es el dato prescindible de los dos
       —quien mira un reloj de cabecera mira la hora—. Con la bandera ES ahora en right:44 y
       23px de ancho, el reloj cierra en 77 y conserva sus diez píxeles de aire. */
    #tfm-reloj {{ top:10px; right:77px; height:16px; font-size:11px; }}
    #tfm-reloj .r-fecha, #tfm-reloj .r-sep {{ display:none; }}
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
    /* La tira de tecnologías NO se apila de una en una —serían {TECH_N} filas—: se aprieta a
       pastillas más pequeñas y sigue repartiendo por ancho. */
    .tech-strip {{ grid-template-columns:repeat(auto-fit, minmax(58px, 1fr)) !important; gap:8px; }}
    .tech-chip {{ height:46px; padding:0 8px; }}
    /* Al apilarse (teléfono) la columna es de ancho completo; el alto ya lo fija la figura Plotly */
    .st-key-bloch_row div[data-testid="stColumn"]:last-of-type div[data-testid="stVerticalBlock"] {{ height:auto !important; }}
    /* Cabecera proporcionada a la pantalla del teléfono */
    .page-eyebrow {{ font-size:12px; letter-spacing:0.13em; margin-bottom:9px; }}
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
    /* .cm-title queda FUERA de esta reducción: su pareja visual es el título de las curvas ROC,
       que lo pinta Plotly en píxeles y no baja en móvil. Si encogiera solo este, el nombre del
       modelo volvería a descuadrar entre las dos secciones justo en pantalla estrecha. */
    .section-sub, .clinical-note, .gov-prose, .qc-prose {{ font-size:12.5px; }}
    .kpi-value {{ font-size:14px; }}
    /* .kpi-model fuera de la reducción por el mismo motivo que .cm-title: es el nombre del
       modelo, y su pareja —el título de la curva ROC— la pinta Plotly en píxeles y no encoge.
       Excepción: .gov-prose-title no es un nombre de modelo, así que en móvil devuelve el
       título de tarjeta a los 13 px estándar de .kpi-model, como el resto de la página. */
    .kpi-model.gov-prose-title {{ font-size:13px; }}
    .kpi-label {{ font-size:12px; }}
    .stat-label {{ font-size:11.5px; letter-spacing:0.09em; }}
    .badge {{ font-size:12px; }}
}}

/* ═══════════════ TRANSICIÓN DE PÁGINA ═══════════════
   El contenedor .st-key-page_enter_N (ver header(), key = índice de la página en i18n.PAGE_KEYS) es
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
# UMBRAL: cada modelo se evaluó en SU punto de corte natural, y los tres son correctos por
# separado (verificado recalculando las tres matrices desde los .npy). Pero NO son el mismo
# punto: LightGBM corta en predict_proba >= 0,50, mientras que sklearn's SVC.predict() usa el
# signo de decision_function — que en escala de probabilidad cae en ~0,22, no en 0,50 — y el
# QSVM corta en decision_function > 0, que ni siquiera es una probabilidad. Como accuracy,
# F1-macro y MCC dependen del umbral y el AUC no, hay que decirlo allí donde se comparan.
# "umbral_p" es el equivalente en escala de probabilidad; None = no aplica.
# El RÓTULO del umbral ("p ≥ 0,50") y su procedencia ya no viven aquí: el primero lleva un
# separador decimal que depende del idioma y la segunda es prosa. Los pinta thr_text() /
# S("res_thr_src") a partir de umbral_p, que es el dato.
MODELS = {
    "lightgbm": {"label": "LightGBM", "color": SERIES["lightgbm"], "auc": 0.9485, "f1_macro": 0.6523,
                 "accuracy": 0.7243, "mcc": 0.4566, "cm": {"tn": 924, "fp": 423, "fn": 9, "tp": 211},
                 "umbral_p": 0.50, "umbral_valor": 0.50},
    "svm_rbf": {"label": "SVM-RBF", "color": SERIES["svm_rbf"], "auc": 0.9377, "f1_macro": 0.8243,
                "accuracy": 0.9075, "mcc": 0.6539, "cm": {"tn": 1250, "fp": 97, "fn": 48, "tp": 172},
                # El corte real del SVM es el signo de decision_function, que no se persistió.
                # Su equivalente en la escala de probabilidad guardada cae entre 0,221729 (último
                # score clasificado como negativo) y 0,225167 (primero como positivo): se toma el
                # punto medio para que la reconciliación no dependa de redondeos.
                "umbral_p": 0.2234, "umbral_valor": 0.22344799236138158},
    "qsvm": {"label": "QSVM", "color": SERIES["qsvm"], "auc": 0.5493, "f1_macro": 0.4669,
             "accuracy": 0.8602, "mcc": 0.0625, "cm": {"tn": 1347, "fp": 0, "fn": 219, "tp": 1},
             "umbral_p": None, "umbral_valor": 0.0},
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

# Embudo de registros: conteos de la ingesta Bronze y de los filtros de Silver.
# El "antes" del filtro DIQ010 es 7.835 y no el 7.831 que imprime el pipeline: ese conteo se
# recalculó sobre el dataframe ya filtrado y su contador de partida quedó desplazado. Los
# 4 registros descartados se confirman con el value_counts de la propia salida
# (6.510 + 1.099 + 222 = 7.831).
#
# Aquí quedan SOLO los conteos; el nombre de la etapa y su explicación viven en
# i18n["gov_embudo"] y se emparejan por posición al pintar. Es la misma división que ya
# usaba la arquitectura Medallón de Resumen: la cifra es un dato verificado y no puede
# divergir entre idiomas, el texto sí cambia.
GOV_EMBUDO_N = [(29400, None), (17961, 11439), (7835, 10126), (7831, 4)]
# Valor de cada operación de Silver / Gold, en el orden de i18n["gov_silver_ops"] y
# ["gov_gold_ops"]. La partición se compone con mil() al pintarla porque su separador de
# millar depende del idioma; el resto son enteros pequeños que se escriben igual en ambos.
#
# Los conteos del notebook (106, 90) son COLUMNAS del DataFrame e incluyen TARGET; las
# features son una menos. Antes la tarjeta mezclaba ambos criterios y la resta no cerraba:
# 106 − 16 = 90 ≠ 89. Ahora se nombran los dos como features (105 → 89) y cuadra.
GOV_SILVER_VALS = ["6", "66", "67", "0"]
GOV_GOLD_VALS = ["105", "16", "89", None]   # None = la partición, que se formatea aparte

# Suite dataframe-expectations 0.7.0 sobre la capa Silver.
# Valores de respaldo: son los que la suite produjo el 22/06/2026, transcritos de la salida.
GOV_SUITE = {
    "nombre": "silver_quality_suite", "fecha": "2026-06-22", "registros": 7831,
    "total": 15, "passed": 15, "failed": 0, "pass_rate": 1.0, "duracion_s": 0.001765,
}
# ...pero si el CSV que genera la propia suite está embarcado, MANDA él. Así las cifras dejan
# de estar transcritas a mano y no pueden desincronizarse de una reejecución del pipeline:
# basta copiar validacion_silver_dfe.csv desde el volumen a streamlit/models/ (el .gitignore
# tiene la excepción para que suba, como ya la tienen los .onnx y los .npy).
# Si no está, se usan los valores de arriba y la página se comporta igual que antes.
GOV_SUITE_FUENTE = "gov_suite_src_nb"      # clave i18n, no el rótulo: se traduce al pintarlo
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
        GOV_SUITE_FUENTE = "gov_suite_src_csv"
    except (KeyError, ValueError, StopIteration, OSError):
        # Un CSV con otro esquema o truncado no debe tumbar la página: se ignora y se
        # mantienen los valores de respaldo, que son los publicados en la memoria.
        pass
# Las 15 expectativas (dimensión, columna, regla con la descripción literal que devuelve el
# runner) están en i18n["gov_expectativas"]: son texto de punta a punta, y sus umbrales
# llevan separador decimal, que también cambia de idioma.

# Historial Delta de la capa Gold. Se muestran las 6 versiones más recientes
# de las 10 registradas; el resto se purga a las 168 h por retención de Delta.
GOV_DELTA_HISTORY = [
    (9, "2026-07-16 16:58:32", "WRITE", 7831, 757558),
    (8, "2026-07-14 14:22:20", "WRITE", 7831, 757558),
    (7, "2026-06-23 20:13:23", "WRITE", 7831, 755326),
    (6, "2026-06-23 20:11:48", "WRITE", 7831, 755326),
    (5, "2026-06-21 06:13:12", "WRITE", 7831, 754078),
    (4, "2026-06-21 06:04:25", "WRITE", 7831, 754078),
]

# La cadena de custodia contra la fuga de información, en el orden en que actúa. El numeral
# es estructura (la secuencia de barreras) y no viaja con la traducción; el nombre y la
# descripción de cada barrera están en i18n["gov_leakage"], emparejados por posición.
GOV_LEAKAGE_N = ["01", "02", "03", "04"]

# Escalado anti-leakage del StandardScaler en la capa Gold: i18n["gov_scaler"], completo
# (etiqueta, valor y detalle) porque dos de los cuatro valores son texto, no cifras.

# Inventario de frameworks por capa. El primer elemento de cada lista de badges es el
# framework que vertebra la capa; el resto, los que lo acompañan. Nombre de capa y badges
# son nombres propios y no se traducen; el rol y la nota salen de i18n["gov_stack"], en
# este mismo orden.
# Color del filete: las TRES CAPAS de datos son una progresión de refinamiento y toman
# pasos de la rampa secuencial (igual que la arquitectura Medallón); los TRES MODELOS son
# identidades y toman su color de SERIES, el mismo que llevan en el resto de la aplicación.
GOV_STACK = [
    ("Bronze", RAMP[1],
     ["boto3", "pyreadstat 1.3.5", "Delta Lake", "Databricks Secrets", "PySpark"]),
    ("Silver", RAMP[2],
     ["dataframe-expectations 0.7.0", "pandas", "NumPy", "PyArrow", "Delta Lake"]),
    ("Gold", RAMP[3],
     ["scikit-learn", "StandardScaler", "RandomForest", "joblib", "Delta time travel"]),
    ("LightGBM", SERIES["lightgbm"],
     ["LightGBM", "SHAP TreeExplainer", "skl2onnx", "onnxmltools", "GridSearchCV"]),
    ("SVM-RBF", SERIES["svm_rbf"],
     ["scikit-learn SVC", "SHAP KernelExplainer", "shap.kmeans", "skl2onnx"]),
    ("QSVM", SERIES["qsvm"],
     ["Qiskit 2.5.0", "qiskit-machine-learning 0.9.0", "qiskit-algorithms 0.4.0",
      "ZZFeatureMap", "FidelityQuantumKernel"]),
]

# Registro de decisiones — las 11 limitaciones de TECHNICAL_NOTES.md §2, resumidas.
# "critical" marca las que condicionan la arquitectura; "warning", las asumidas y
# documentadas sin corregir; "good", las resueltas sin residuo.
# Referencia y nivel de cada decisión, en el orden de i18n["gov_decisiones"]. El título,
# el problema y la solución son texto y viven allí; el nivel es estructura (elige el color
# de STATUS y la etiqueta Arquitectura/Asumida/Resuelta) y se queda aquí.
GOV_DECISIONES_META = [
    ("01", "critical"), ("02", "critical"), ("03", "critical"), ("04", "warning"),
    ("05", "warning"), ("06", "warning"), ("07", "good"), ("08", "warning"),
    ("09", "warning"), ("10", "warning"), ("11", "good"),
]

# Ranking SHAP de cada modelo: (código NHANES, valor). El código es el identificador oficial
# de la variable y es lo que se pinta en el eje; su glosa sale de i18n["var_short"] y solo
# aparece en el hover, así que aquí no hace falta repetirla — antes viajaba en la tupla y
# habría obligado a duplicar los 40 valores para tener la lista en inglés.
SHAP_LIGHTGBM = [
    ("LBXGH", 1.1243), ("RIDAGEYR", 0.4654), ("LBXGLU", 0.3161),
    ("LBDLDL", 0.2542), ("BMXWAIST", 0.2012), ("WTINT2YR", 0.1274),
    ("BMXARML", 0.0911), ("BMXLEG", 0.0872), ("BMXBMI", 0.0799),
    ("PAD680", 0.0634), ("PAD645", 0.0450), ("PAQ640", 0.0345),
    ("BMXWT", 0.0336), ("LBXIN", 0.0273), ("INDHHIN2", 0.0264),
    ("DMDYRSUS", 0.0215), ("BMXARMC", 0.0174), ("PAQ670", 0.0170),
    ("BPXSY1", 0.0165), ("PAD630", 0.0158),
]
SHAP_SVMRBF = [
    ("LBXGH", 0.1017), ("LBXGLU", 0.0436), ("LBDLDL", 0.0219),
    ("RIDAGEYR", 0.0141), ("BMXLEG", 0.0107), ("BMXWAIST", 0.0062),
    ("DMDHHSZE", 0.0054), ("PAD680", 0.0042), ("LBXIN", 0.0037),
    ("DMDYRSUS", 0.0025), ("BPXDI1", 0.0023), ("LBXTR", 0.0022),
    ("BMXWT", 0.0021), ("DMDMARTL_1", 0.0020), ("DMDMARTL_5", 0.0018),
    ("BPXPLS", 0.0017), ("DMDEDUC2_3", 0.0017), ("SDMVSTRA", 0.0017),
    ("DMDMARTL_2", 0.0017), ("DMDHHSZB", 0.0016),
]

# Las 8 features seleccionadas por Random Forest para el QSVM (y usadas en Bloch Sphere / Live Predictor)
# step + fmt: por defecto st.slider con límites float usa un paso (max-min)/100 irregular (p. ej. 0,62
# años) y muestra el valor con dos decimales ("45.00"). Fijamos pasos clínicamente naturales —0,1 para
# magnitudes con decimal (HbA1c, IMC), 1 para las enteras— y su formato, para que el valor se lea
# limpio ("45", "5,7") y la interacción sea coherente con cómo se miden esas variables.
#
# "importance": valores del ranking Random Forest tal y como los imprime la celda 20 de
# notebook_03_gold.ipynb ("Top 8 features para QSVM (sin variables DIQ)"). Antes este dict
# llevaba una copia ligeramente distinta (0,2452 / 0,1853 / 0,0325 …) que no coincidía ni con
# el notebook ni con RF_TOP8_IMPORTANCE de más abajo: la misma magnitud aparecía con dos
# valores en la misma aplicación. Ahora ambos salen de la misma fuente — ver
# RF_TOP8_IMPORTANCE, que se deriva de aquí en vez de repetirse a mano.
#
# El rótulo de cada variable sale de i18n["qsvm_labels"] (ver q_label más abajo). La UNIDAD
# sí se queda aquí porque siete de las ocho son símbolos internacionales que no se traducen
# (%, mg/dL, cm, µU/mL, kg/m²); la única palabra, "años", pasa por i18n["qsvm_units"].
#
# El TypedDict declara la forma de cada entrada: como este dict se lee por clave en media
# docena de sitios (sliders del Live Predictor, esfera de Bloch, RF_TOP8_IMPORTANCE), una
# errata como v["rango"] o v["importancia"] no daría error hasta ejecutar esa página
# concreta. Declarada la forma, el editor la marca al escribirla y autocompleta las claves.
# Los valores NO cambian: es solo la anotación.
class QFeature(TypedDict):
    unit: str                     # símbolo de la unidad, sin traducir salvo "años"
    range: tuple[float, float]    # (mínimo, máximo) del slider, en unidades clínicas
    default: float                # valor de partida del slider
    importance: float             # ranking Random Forest, celda 20 de notebook_03_gold
    step: float                   # paso del slider (0,1 con decimal; 1 para enteras)
    fmt: str                      # formato de st.slider para el valor mostrado


QSVM_FEATURES: dict[str, QFeature] = {
    "LBXGH":    {"unit": "%",       "range": (4.0, 15.0),  "default": 5.7,  "importance": 0.2454, "step": 0.1, "fmt": "%.1f"},
    "LBXGLU":   {"unit": "mg/dL",   "range": (50, 300),     "default": 100,  "importance": 0.1855, "step": 1.0, "fmt": "%d"},
    "RIDAGEYR": {"unit": "años",    "range": (18, 80),      "default": 45,   "importance": 0.0323, "step": 1.0, "fmt": "%d"},
    "LBDLDL":   {"unit": "mg/dL",   "range": (40, 250),     "default": 110,  "importance": 0.0318, "step": 1.0, "fmt": "%d"},
    "BMXWAIST": {"unit": "cm",      "range": (60, 150),     "default": 95,   "importance": 0.0283, "step": 1.0, "fmt": "%d"},
    "LBXIN":    {"unit": "µU/mL",   "range": (2, 60),       "default": 10,   "importance": 0.0265, "step": 1.0, "fmt": "%d"},
    "BMXLEG":   {"unit": "cm",      "range": (30, 50),      "default": 40,   "importance": 0.0225, "step": 1.0, "fmt": "%d"},
    "BMXBMI":   {"unit": "kg/m²",   "range": (15, 60),      "default": 27,   "importance": 0.0221, "step": 0.1, "fmt": "%.1f"},
}


def q_label(code):
    """Rótulo traducido de una de las 8 variables del QSVM (sliders y selectores)."""
    return S("qsvm_labels")[code]


def q_unit(code):
    """Unidad traducida. Solo "años" tiene entrada en el mapa; el resto pasan tal cual."""
    u = QSVM_FEATURES[code]["unit"]
    return S("qsvm_units").get(u, u)

# Ranking RF para el gráfico "8 features seleccionadas" (Circuito Cuántico). Se DERIVA de
# QSVM_FEATURES en vez de repetirse: una sola fuente para el mismo dato, así no pueden
# volver a divergir. El orden ya es el del ranking (importancia descendente).
RF_TOP8_IMPORTANCE = {code: v["importance"] for code, v in QSVM_FEATURES.items()}

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


@st.cache_data
def _perfil_base(_features: tuple, medianas: tuple, medias: tuple) -> np.ndarray:
    """Vector base de las 89 features, con las one-hot corregidas.

    La mediana columna a columna NO sirve para variables one-hot: si una categoría tiene
    prevalencia < 50 % su mediana es 0, y como ninguna de las categorías de RIAGENDR,
    RIDRETH1, RIDRETH3 o DMDEDUC2 llega al 50 %, TODAS sus columnas salían a 0. El paciente
    de referencia resultaba imposible: sin sexo, sin etnia y sin nivel educativo. Aquí cada
    grupo one-hot que suma 0 recibe un 1 en su MODA — que se lee de la media del scaler,
    porque para una dummy 0/1 la media ES la prevalencia de esa categoría."""
    feats = list(_features)
    x = np.array(medianas, dtype=np.float64)
    medias = np.array(medias, dtype=np.float64)

    grupos = {}
    for f in feats:
        if "_" in f and f.rsplit("_", 1)[1].replace(".", "").isdigit():
            grupos.setdefault(f.rsplit("_", 1)[0], []).append(f)
    for cols in grupos.values():
        idx = [feats.index(c) for c in cols]
        if x[idx].sum() == 0:                       # grupo one-hot sin categoría activa
            x[max(idx, key=lambda i: medias[i])] = 1.0
    return x


def _build_feature_vector(sp: dict, overrides: dict) -> np.ndarray:
    """Construye el vector de 89 features: perfil base del conjunto de entrenamiento
    (medianas, con las one-hot puestas en su moda) + las variables clínicas del Live
    Predictor sobrescritas en su posición exacta."""
    feats = sp["features"]
    x = _perfil_base(tuple(feats),
                     tuple(sp["medians"][f] for f in feats),
                     tuple(sp["mean"])).copy()
    for k, v in overrides.items():
        # Un nombre desconocido se ignoraba en silencio y la variable simplemente no tenía
        # efecto, sin error visible. Con 89 features y sufijos como "_1.0" eso es una errata
        # a la vuelta de la esquina: mejor romper aquí que devolver una predicción falsa.
        if k not in feats:
            raise KeyError(f"'{k}' no es una feature del modelo (89 disponibles en el scaler)")
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


# ─────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN END-TO-END CONTRA EL "GOLDEN SET"
# reconciliar_metricas() comprueba que las métricas publicadas cuadran con los scores
# guardados, pero no que ESTE camino de inferencia reproduzca los modelos entrenados:
# el conjunto de test no está en el repositorio. El golden set (25 filas del test con la
# probabilidad que devolvió el modelo, generado por los NB04/NB05 — ver
# notebooks/INSTRUCCIONES_exportar_golden_set.md) cierra ese hueco.
#
# Verifica los cuatro pasos de golpe: escalar solo el SVM, convertir a float32, llamar al
# ONNX correcto y LEER BIEN SU SALIDA. Lo último importa más de lo que parece: el NB04
# convirtió con zipmap=False (out[1] es un ndarray (N,2)) y el NB05 sin esa opción (out[1]
# es una lista de dicts). `out[1][i][1]` acierta en ambos, pero por motivos distintos, y una
# regresión ahí devolvería números plausibles y falsos.
# ─────────────────────────────────────────────────────────────────────────
GOLDEN = {
    "lightgbm": {"fichero": "golden_lgbm.npz", "onnx": "lgbm_final.onnx", "escalar": False},
    "svm_rbf":  {"fichero": "golden_svm.npz",  "onnx": "svm_final.onnx",  "escalar": True},
}
GOLDEN_TOL = 1e-4   # los .onnx operan en float32; el ruido esperado es ~1e-7


def _probs_onnx(sess, X: np.ndarray) -> np.ndarray:
    """P(clase 1) sea cual sea la forma del segundo tensor de salida."""
    salida = sess.run(None, {"float_input": X.astype(np.float32)})[1]
    if isinstance(salida, list):                 # ZipMap: lista de {clase: prob}
        return np.array([float(d[1]) for d in salida])
    return np.asarray(salida, dtype=float)[:, 1]  # ndarray (N, 2)


@st.cache_data
def verificar_golden():
    """Reproduce el camino de inferencia del dashboard sobre el golden set y lo compara
    con la probabilidad que registró el notebook. Devuelve {clave: dict|None} más
    '_estado': 'ok' | 'fallo' | 'ausente'. Nunca afirma haber verificado lo que no ha
    podido comprobar: si faltan los ficheros, el estado es 'ausente'."""
    sp = _load_scaler_and_medians()
    res, hay, todo_ok = {}, False, True
    for clave, cfg in GOLDEN.items():
        ruta = MODELS_DIR / cfg["fichero"]
        sess = _load_onnx_session(cfg["onnx"])
        if sp is None or sess is None or not ruta.exists():
            res[clave] = None
            continue
        try:
            d = np.load(ruta, allow_pickle=True)
            X, p_ref = np.asarray(d["X"], dtype=np.float64), np.asarray(d["p"], dtype=np.float64)
            if list(d["features"]) != list(sp["features"]):
                res[clave] = {"ok": False, "n": len(p_ref), "max_dif": float("nan"),
                              "error": "el orden de features del golden set no coincide con el scaler"}
                todo_ok = False
                hay = True
                continue
            X_in = (X - sp["mean"]) / sp["scale"] if cfg["escalar"] else X
            p_app = _probs_onnx(sess, X_in)
            dif = np.abs(p_app - p_ref)
            ok = bool(np.nanmax(dif) <= GOLDEN_TOL)
            res[clave] = {"ok": ok, "n": int(len(p_ref)), "max_dif": float(np.nanmax(dif)),
                          "error": None}
            todo_ok = todo_ok and ok
            hay = True
        except Exception as exc:                  # un .npz corrupto no debe tumbar la página
            res[clave] = {"ok": False, "n": 0, "max_dif": float("nan"), "error": str(exc)[:160]}
            todo_ok = False
            hay = True
    res["_estado"] = ("ok" if todo_ok else "fallo") if hay else "ausente"
    return res


@st.cache_data
def barrer_variable(code: str, valores: tuple, otros: tuple):
    """Recorre `code` sobre `valores` dejando el resto de variables fijas en `otros`.

    Devuelve (prob_lgbm, prob_svm) como arrays, o None si faltan los modelos. Se construye
    UNA matriz con todas las filas y se hace UNA llamada por modelo: barrer punto a punto
    costaba ~100 ms por rerun del slider, y así baja a unos pocos ms. `otros` llega como
    tupla de pares ordenados para que `st.cache_data` pueda hashearlo."""
    sp = _load_scaler_and_medians()
    sess_lgbm = _load_onnx_session("lgbm_final.onnx")
    sess_svm = _load_onnx_session("svm_final.onnx")
    if sp is None or sess_lgbm is None or sess_svm is None:
        return None

    fila = _build_feature_vector(sp, dict(otros))
    X = np.tile(fila, (len(valores), 1))
    X[:, sp["features"].index(code)] = np.asarray(valores, dtype=np.float64)

    out_l = sess_lgbm.run(None, {"float_input": X.astype(np.float32)})
    out_s = sess_svm.run(None, {"float_input": ((X - sp["mean"]) / sp["scale"]).astype(np.float32)})
    p_l = np.array([r[1] for r in out_l[1]], dtype=float)
    p_s = np.array([r[1] for r in out_s[1]], dtype=float)
    return p_l, p_s


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


# ─────────────────────────────────────────────────────────────────────────
# RECONCILIACIÓN DE MÉTRICAS
# Las métricas de MODELS están transcritas de los notebooks. Transcribir a mano se
# desincroniza en cuanto alguien reentrena, así que en vez de confiar en ellas se
# RECALCULAN aquí desde los scores por instancia y se comparan. Si algún día dejan de
# cuadrar, la página lo dice en pantalla en lugar de seguir mostrando cifras muertas.
# El umbral de cada modelo es el suyo (ver comentario de MODELS): no se puede usar 0,5
# para los tres porque solo LightGBM corta ahí.
# ─────────────────────────────────────────────────────────────────────────
def _auc_mann_whitney(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """AUC exacto por rangos, con corrección de empates. Independiente del umbral."""
    order = np.argsort(y_scores, kind="mergesort")
    s_ord = y_scores[order]
    ranks = np.empty(len(y_scores), dtype=float)
    i = 0
    while i < len(s_ord):                      # rangos medios dentro de cada empate
        j = i
        while j + 1 < len(s_ord) and s_ord[j + 1] == s_ord[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _metricas_en_umbral(y_true: np.ndarray, y_scores: np.ndarray, umbral: float) -> dict:
    """Matriz de confusión y métricas derivadas cortando en `umbral`."""
    pred = (y_scores >= umbral).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    def _f1(prec, rec):
        return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    f1_pos = _f1(tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)
    f1_neg = _f1(tn / (tn + fn) if tn + fn else 0.0, tn / (tn + fp) if tn + fp else 0.0)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "cm": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": (tp + tn) / len(y_true),
        "f1_macro": (f1_pos + f1_neg) / 2,
        "mcc": 0.0 if den == 0 else (tp * tn - fp * fn) / den,
    }


@st.cache_data
def reconciliar_metricas():
    """Recalcula las métricas de los tres modelos desde sus .npy y las compara con las
    declaradas en MODELS. Devuelve {clave: {'ok': bool, 'auc': float, 'detalle': [str]}}
    y una entrada especial '_global' con True solo si los tres cuadran. Si a un modelo le
    faltan los scores, queda como None y no se afirma nada sobre él."""
    ROC_PREFIX = {"lightgbm": "lgbm", "svm_rbf": "svm", "qsvm": "qsvm"}
    out, todos_ok, alguno = {}, True, False
    for clave, m in MODELS.items():
        scores, y_true = _load_roc_scores(ROC_PREFIX[clave])
        if scores is None:
            out[clave] = None
            todos_ok = False
            continue
        alguno = True
        auc = _auc_mann_whitney(y_true, scores)
        calc = _metricas_en_umbral(y_true, scores, m["umbral_valor"])
        detalle = []
        if abs(auc - m["auc"]) > 5e-4:
            detalle.append(f"AUC declarado {m['auc']:.4f} vs recalculado {auc:.4f}")
        if calc["cm"] != m["cm"]:
            detalle.append(f"matriz declarada {m['cm']} vs recalculada {calc['cm']}")
        for k in ("accuracy", "f1_macro", "mcc"):
            if abs(calc[k] - m[k]) > 5e-4:
                detalle.append(f"{k} declarado {m[k]:.4f} vs recalculado {calc[k]:.4f}")
        out[clave] = {"ok": not detalle, "auc": auc, "n": int(len(y_true)),
                      "positivos": int(y_true.sum()), "detalle": detalle}
        todos_ok = todos_ok and not detalle
    out["_global"] = todos_ok and alguno
    return out

# Descripción breve de cada variable NHANES (para los tooltips de las gráficas SHAP). Basado en el
# Anexo C del TFM (Diccionario de Variables NHANES Utilizadas en el Pipeline). El texto vive en
# i18n porque es prosa, no dato: aquí solo se resuelve al idioma activo.
VAR_DESC = S("var_desc")

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

    # El ORDEN del menú (y el porqué de que Gobernanza vaya SEGUNDA, no al final) está en
    # i18n.PAGES: el título del TFM es "Integración QML en pipeline DataOps" y la mitad
    # DataOps se sostiene en esa página. Puesta ahí, la app se recorre en el orden real de
    # ejecución —el dato antes que los modelos— y enlaza con el bloque Medallón de Resumen,
    # que es su resumen de tres líneas.
    #
    # La IDENTIDAD de la página es su clave ("governance"), no su rótulo. Lo que se guarda en
    # session_state y lo que se compara en el enrutado es siempre la clave; el rótulo solo
    # existe en el momento de pintarlo. Si la identidad fuera el texto visible, cambiar de
    # idioma dejaría a session_state.page apuntando a una página que ya no existe con ese
    # nombre y la app volvería al Resumen en cada cambio de bandera.
    _MENU_OPTIONS = S("nav")
    if "page" not in st.session_state:
        st.session_state.page = i18n.PAGE_KEYS[0]

    # Sin tooltip, pero con nombre accesible. El globito con "Expandir"/"Colapsar" repetía en
    # palabras lo que la flecha ya dice —apunta siempre al lado al que se moverá la barra—,
    # así que se va; lo que no puede irse es el nombre del botón, porque un <button> cuyo
    # único contenido es "‹" no se anuncia como nada.
    #
    # st.button no acepta aria-label y el help de Streamlit tampoco servía: se traduce en un
    # aria-describedby que solo existe mientras el globito está abierto, o sea una DESCRIPCIÓN
    # ocasional, nunca el nombre. Así que el nombre se construye con el propio rótulo: la
    # palabra viaja dentro en cursiva —el único envoltorio que el markdown de st.button deja
    # crear— y el CSS la recorta. El ojo ve la flecha; el lector de pantalla lee la frase.
    if narrow:
        if st.button(f"› *{S('sidebar_expand')}*", key="toggle_sidebar"):
            st.session_state.sidebar_narrow = False
            st.session_state.menu_force_index = i18n.PAGE_KEYS.index(st.session_state.page)
            st.rerun()
    else:
        if st.button(f"‹ *{S('sidebar_collapse')}*", key="toggle_sidebar"):
            st.session_state.sidebar_narrow = True
            st.session_state.menu_force_index = i18n.PAGE_KEYS.index(st.session_state.page)
            st.rerun()

    # ── BUSCADOR ────────────────────────────────────────────────────────────
    # Busca primero DENTRO del panel (páginas, secciones y las ocho variables clínicas,
    # índice derivado del propio catálogo i18n) y ofrece siempre, debajo, lanzar la misma
    # consulta a la web en una pestaña nueva. Las dos cosas en la misma caja: quien escribe
    # "ZZFeatureMap" quiere ir al Circuito Cuántico, y quien escribe "hemoglobina glicada"
    # quiere una definición — no tiene por qué saber de antemano cuál de las dos es.
    #
    # En los 84 px del modo colapsado una caja de texto no cabe —un buscador que no deja leer
    # lo que escribes es peor que no tenerlo—, pero desaparecer del todo tampoco vale: la
    # entrada sigue estando, reducida a su lupa, y pulsarla despliega la barra con la caja ya
    # puesta. Mismo gesto que hace el propio menú al colapsarse: se queda el icono.
    if narrow:
        if st.button(" ", key="search_expand", icon=":material/search:", help=S("search_expand")):
            st.session_state.sidebar_narrow = False
            st.session_state.menu_force_index = i18n.PAGE_KEYS.index(st.session_state.page)
            st.rerun()
    else:
        def _ir_a_resultado(_pagina):
            """Navega a la página del resultado y vacía la caja.

            Va como CALLBACK y no como código tras el `if st.button(...)`: el valor de un
            widget solo puede tocarse desde un callback (fuera lanza StreamlitAPIException),
            y sin vaciar la caja la lista de resultados seguiría abierta empujando el menú
            hacia abajo después de haber navegado.
            """
            st.session_state.page = _pagina
            st.session_state.menu_force_index = i18n.PAGE_KEYS.index(_pagina)
            st.session_state.nav_search = ""

        _q = (st.text_input(S("search_label"), key="nav_search", placeholder=S("search_ph"),
                            icon=":material/search:", label_visibility="collapsed") or "").strip()
        if _q:
            _hits = i18n.search(_q, LANG)
            with st.container(key="nav_search_results"):
                for _i, _h in enumerate(_hits):
                    # El rótulo de la página va en el tooltip y no en el propio botón: en
                    # 270 px de ancho, "Curva de respuesta · Predictor en Vivo" se parte en
                    # dos líneas y la lista deja de leerse como lista.
                    st.button(_h["label"], key=f"nav_hit_{_i}", width="stretch",
                              on_click=_ir_a_resultado, args=(_h["page"],),
                              help=None if _h["kind"] == 0 else S("search_in").format(
                                  p=_MENU_OPTIONS[i18n.PAGE_KEYS.index(_h["page"])]))
                if not _hits:
                    st.markdown(f'<div class="search-none">{S("search_none")}</div>',
                                unsafe_allow_html=True)
                # Enlace en crudo y no st.link_button: aquí hace falta target="_blank"
                # explícito (la consulta se abre FUERA, no reemplazando el panel) y
                # rel="noopener" para no ceder window.opener al buscador.
                st.markdown(
                    f'<a class="search-web" target="_blank" rel="noopener noreferrer" '
                    f'href="https://www.google.com/search?q={quote_plus(_q)}">'
                    f'{html.escape(S("search_web").format(q=_q))}'
                    f'<span class="search-web-ext">↗</span></a>', unsafe_allow_html=True)

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
    # El icono va en el acento de marca, no en el color del rótulo. C_PRIMARY es justo lo que
    # la paleta reserva al cromo de navegación (ver su bloque más arriba), y el icono es además
    # lo ÚNICO que queda visible con la barra colapsada, donde el rótulo se apaga a font-size:0.
    # El rótulo no se toca —sigue en text_secondary—: así el color distingue glifo de texto en
    # vez de repetirlos, y la jerarquía del ítem la siguen marcando peso, fondo y filete.
    # Contraste sobre sidebar_bg: 9,33:1 en oscuro y 4,08:1 en claro, por encima del 3:1 que
    # WCAG 1.4.11 pide a un elemento gráfico de interfaz.
    icon_style = {"font-size": "15px", "color": C_PRIMARY}
    if narrow:
        # El componente pinta cada ítem como <a class="nav-link"><i class="icon bi-…"></i> Rótulo</a>,
        # y el rótulo es un NODO DE TEXTO SUELTO, sin envoltorio propio. Como el dict "styles" solo
        # llega al <a> y al <i>, la única forma de ocultar el texto es font-size:0 en el <a> — y ahí
        # estaba el recorte de los iconos: con font-size 0 la caja de línea del <a> mide 0, el glifo
        # se dibuja desde una línea base sin altura y se cortaba por la mitad inferior (se veía el
        # tejado de la casa sin paredes, el escudo sin punta).
        #
        # La solución es sacar el glifo del flujo de línea: el <a> pasa a ser flex y el <i> a caja
        # de bloque con ALTURA PROPIA, así su tamaño ya no depende de un line-height heredado que
        # vale cero. El font-size:0 sigue borrando el rótulo, que es lo que se quería.
        _centrado = {"font-size": "0px", "line-height": "0", "padding": "11px 0",
                     "display": "flex", "align-items": "center", "justify-content": "center"}
        nav_link_style.update(_centrado)
        nav_link_selected_style.update(_centrado)
        icon_style.update({"font-size": "18px", "line-height": "1", "height": "22px", "width": "100%",
                           "margin": "0", "display": "flex", "align-items": "center",
                           "justify-content": "center"})

    # manual_select fuerza al componente a saltar a un indice concreto, pero es un disparo "de un
    # solo uso": si se reenvia en cada rerun (incluido el propio rerun que dispara un clic del
    # usuario en el menu) compite con ese clic y el menu queda oscilando sin parar entre la pestaña
    # vieja y la nueva hasta que se refresca la pagina. Por eso solo se rellena explicitamente justo
    # antes de un st.rerun() disparado por OTRO widget (toggle de sidebar, toggle de tema) y se
    # consume aqui con pop() para que en el resto de reruns (incluidos los clics normales) viaje
    # como None y no interfiera.
    _forced_index = st.session_state.pop("menu_force_index", None)
    _seleccion = option_menu(
        menu_title=None,
        options=_MENU_OPTIONS,
        icons=i18n.PAGE_ICONS,
        default_index=i18n.PAGE_KEYS.index(st.session_state.page),
        manual_select=_forced_index,
        # La key incluye el tema Y el modo narrow a proposito: option_menu vive en un iframe con
        # estado JS propio (Vue) que solo LEE el dict "styles" al montarse — en reruns posteriores con
        # la MISMA key, los cambios en "styles" (colores de tema, o el font-size:0 del modo narrow) no
        # se reaplican de verdad. Con key fija, alternar sidebar_narrow dejaba el menu con el texto
        # aun visible (a tamaño completo) junto al icono, porque el componente seguia usando los
        # estilos con los que se monto la primera vez. Cambiar la key en cada toggle (tema Y narrow)
        # fuerza un remount completo, que si levanta los estilos frescos. default_index/manual_select
        # ya se encargan de que ese remount respete la pestaña activa.
        # El IDIOMA entra en la key por el mismo motivo, y aqui es aun mas necesario: no cambian
        # solo los estilos, cambia la lista "options" entera. Sin remount, el componente seguiria
        # mostrando los siete rotulos del idioma con el que se monto.
        key=f"main_menu_{st.session_state.theme}_{narrow}_{LANG}",
        styles={
            # "0!important" y no "0": el componente pone estos estilos EN LÍNEA sobre un div que
            # además lleva la clase Bootstrap `p-3`, y `.p-3` declara `padding:1rem!important`,
            # que gana a un estilo en línea normal. El resultado es que el menú ha llevado
            # siempre 16 px de relleno a cada lado sin que se notara… hasta colapsar la barra:
            # con 84 px, esos 32 px se comían el ancho entero y el icono se salía del iframe.
            # Un !important en línea sí gana a un !important de clase.
            "container": {"padding": "0!important", "background-color": t["sidebar_bg"],
                          "border-radius": "0"},
            "icon": icon_style,
            "nav-link": nav_link_style,
            "nav-link-selected": nav_link_selected_style,
        },
    )
    # option_menu devuelve el ROTULO pulsado, que es lo unico que el componente conoce. Aqui se
    # traduce de vuelta a la clave por posicion —los dos listados salen del mismo i18n.PAGES y
    # comparten orden— y a partir de este punto en toda la app "page" es la clave, nunca el texto.
    page = i18n.PAGE_KEYS[_MENU_OPTIONS.index(_seleccion)]
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
    /* El icono activo sube al acento pareja: como ahora los siete van en C_PRIMARY, ese color
       ya no distingue al seleccionado. C_DARK se separa en la dirección correcta en cada tema
       —más brillante en oscuro (#F9C449), más profundo en claro (#6B4600)—, así que el ítem
       activo gana peso en ambos. Contraste sobre el fondo del ítem activo: 9,15:1 y 5,94:1.
       El !important sigue haciendo falta: nav-link-selected pinta color sobre el <a> y, sin
       forzar, esa cascada se lleva por delante el color en línea del <i>. */
    nav[role="navigation"] a.nav-link.active i {{
        color:{C_DARK} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    if st.button(" ", key="theme_toggle",
                 help=S("theme_to_dark") if st.session_state.theme == "light" else S("theme_to_light")):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

    if narrow:
        _footer_html = (f'<div class="footer-name">{S("footer_name_narrow")}</div>'
                         f'<div class="footer-uni">{S("footer_uni_narrow")}</div>')
    else:
        _footer_html = (f'<div class="footer-name">{S("footer_name")}</div>'
                         f'<div class="footer-uni">{S("footer_uni")}</div>')
    st.markdown(f'<div class="sidebar-footer">{_footer_html}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# SELECTOR DE IDIOMA
# ─────────────────────────────────────────────────────────────────────────
# Los dos botones se pintan en el lienzo principal (no en la sidebar) y el CSS los lleva a la
# esquina superior derecha con position:fixed; ver las reglas .st-key-lang_* de la hoja de
# estilos. Aquí solo importa el mecanismo, que es el mismo del interruptor de tema: escribir
# el idioma en session_state y forzar un rerun.
#
# La etiqueta del botón es un espacio en blanco a propósito: la bandera la pone el CSS como
# fondo, y el nombre del idioma viaja en el tooltip. Un botón vacío del todo perdería su caja
# de foco de teclado.
#
# No hay que invalidar ninguna caché al cambiar de idioma: lo caro que carga esta app —sesión
# ONNX, scaler, scores de ROC, logos en base64— no depende del texto, así que el cambio de
# bandera cuesta exactamente lo mismo que el de tema.
# Sin st.columns para colocarlos uno al lado del otro: el contenedor de columnas SÍ ocuparía
# una fila en el flujo (el height:0 solo aplica a los dos .st-key-lang_*) y empujaría el
# titular de la página hacia abajo. Sueltos, no ocupan nada y el CSS los coloca.
if st.button(" ", key="lang_es", help=S("lang_es_help")) and LANG != "es":
    st.session_state.lang = "es"
    st.rerun()
if st.button(" ", key="lang_en", help=S("lang_en_help")) and LANG != "en":
    st.session_state.lang = "en"
    st.rerun()

# ── El idioma DECLARADO del documento ──
# Streamlit escribe <html lang="en"> fijo y no lo toca al cambiar de bandera: con la app
# en español el documento seguía declarando inglés. No es cosmético. El atributo lang es
# lo que usan los lectores de pantalla para elegir voz y pronunciación (español leído con
# voz inglesa es ininteligible), la separación silábica del navegador y su corrector; y
# es el criterio WCAG 3.1.1 "Language of Page", de nivel A — el mínimo de conformidad.
#
# No hay ajuste de Streamlit para esto, y editar su index.html no vale: se pierde al
# reinstalar y en Community Cloud no se puede tocar. La vía es el componente HTML, cuyo
# iframe se sirve por srcdoc y comparte por tanto el origen de la página, así que puede
# escribir en el documento padre.
#
# El valor va por json.dumps y no interpolado a pelo: es lo que separa "escribir una
# cadena" de "inyectar lo que haya en session_state" dentro de un <script>.
with st.container(key="lang_attr"):
    components.html(
        f"<script>window.parent.document.documentElement.lang = {json.dumps(LANG)};</script>",
        height=0, width=0,
    )

# ── RELOJ ────────────────────────────────────────────────────────────────────
# La hora la pone el NAVEGADOR, no Python: un datetime.now() del servidor se congelaría en el
# instante del rerun —un reloj parado es peor que no tenerlo— y además en Community Cloud
# marcaría la hora del contenedor (UTC), no la de quien mira la pantalla.
#
# Se reutiliza la misma vía que el atributo lang de aquí arriba: el iframe del componente se
# sirve por srcdoc, comparte origen y puede escribir en el documento padre. El elemento se
# crea vacío y su aspecto vive en la hoja de estilos (#tfm-reloj), no aquí, para que siga al
# tema como cualquier otra regla en vez de llevar los colores incrustados en el script.
#
# El setInterval anterior se cancela SIEMPRE al arrancar: cada rerun recarga este iframe y
# vuelve a ejecutar el script, así que sin cancelar quedarían intervalos vivos acumulándose
# —uno por clic— todos escribiendo sobre el mismo nodo. El identificador se guarda en el
# window padre porque es lo único que sobrevive a la recarga del iframe.
#
# Cada 15 s y no cada segundo: sin segundos en pantalla, un tic por segundo son 60 repintados
# por minuto para que cambie un dígito cada 60. Con 15 s el minuto que se ve nunca va más de
# un cuarto de minuto atrasado y el coste es cuatro repintados.
with st.container(key="reloj"):
    components.html(
        f"""<script>
(function () {{
  var doc = window.parent.document;
  var loc = {json.dumps("es-ES" if LANG == "es" else "en-GB")};
  if (window.parent.__tfmRelojTick) {{ window.parent.clearInterval(window.parent.__tfmRelojTick); }}
  var caja = doc.getElementById("tfm-reloj");
  if (!caja) {{
    caja = doc.createElement("div");
    caja.id = "tfm-reloj";
    // Las tres piezas se crean UNA vez y luego solo se les cambia el texto: reconstruir el
    // innerHTML en cada tic tiraría los nodos y con ellos cualquier transición heredada.
    ["r-fecha", "r-sep", "r-hora"].forEach(function (cls) {{
      var s = doc.createElement("span");
      s.className = cls;
      caja.appendChild(s);
    }});
    doc.body.appendChild(caja);
  }}
  var fecha = caja.querySelector(".r-fecha");
  var sep   = caja.querySelector(".r-sep");
  var hora  = caja.querySelector(".r-hora");
  function pinta() {{
    var t = new Date();
    fecha.textContent = t.toLocaleDateString(loc, {{ day: "numeric", month: "short", year: "numeric" }});
    sep.textContent = "\\u00B7";
    // hour12:false en los dos idiomas: el AM/PM cambia el ancho del reloj según la hora del
    // día y en un elemento fijo eso se traduce en que la tira de cabecera se mueve sola.
    hora.textContent = t.toLocaleTimeString(loc, {{ hour: "2-digit", minute: "2-digit", hour12: false }});
  }}
  pinta();
  window.parent.__tfmRelojTick = window.parent.setInterval(pinta, 15000);
}})();
</script>""",
        height=0, width=0,
    )

def header(eyebrow, title, subtitle):
    # Streamlit reutiliza los mismos nodos del DOM entre reruns (para no perder el estado de sliders/
    # botones), así que una animación CSS "al aparecer" sobre stElementContainer nunca llegaba a
    # dispararse de verdad al cambiar de página — el nodo nunca se desmontaba. Un contenedor con key
    # dependiente de la página SÍ fuerza un remount real cada vez que cambia `page` (key distinta =
    # componente distinto para Streamlit), y NO se repite en reruns dentro de la misma página (mover
    # un slider, alternar el tema) porque ahí la key no cambia y el contenedor se reutiliza tal cual.
    _page_idx = i18n.PAGE_KEYS.index(page) if page in i18n.PAGE_KEYS else 0
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
    """Formato numérico según el idioma activo.

    Español: coma decimal y punto de millar ('1.234,5678'), consistente con la prosa
    del TFM. Inglés: la convención inversa ('1,234.5678'), que es la que Python ya
    produce de fábrica. Traducir el texto y dejar las cifras a la española sería un
    error tan visible como no traducir: en una memoria científica el separador forma
    parte de la notación, no de la maquetación.
    """
    s = f"{x:,.{dec}f}"                       # formato US: '1,234.5678'
    if LANG == "es":
        s = s.translate(str.maketrans({",": ".", ".": ","}))  # intercambia separadores en una pasada
    return s

def pct(x, dec=1):
    """Porcentaje en el idioma activo: '78,2 %' en español, '78.2%' en inglés.

    El espacio antes del signo no es un capricho tipográfico: en español la norma
    (y el criterio del SI) lo exige, y en inglés la convención es pegarlo a la cifra.
    """
    return f"{nf(x * 100, dec)} %" if LANG == "es" else f"{nf(x * 100, dec)}%"

def mil(n):
    """Entero con separador de millar en el idioma activo: '29.400' / '29,400'.

    Existe porque la app pinta cifras enteras (registros, filas descartadas) por media
    docena de sitios con un `.replace(",", ".")` a mano. Ese replace es correcto en
    español y erróneo en inglés, así que la decisión se centraliza aquí.
    """
    s = f"{n:,}"
    return s.replace(",", ".") if LANG == "es" else s

def thr_text(clave):
    """Punto de corte de un modelo, con el decimal en el idioma activo.

    El rótulo estaba escrito a mano en MODELS ("p ≥ 0,50") y se quedaba con la coma
    española en la app inglesa — justo en el dato que la tarjeta destaca. Aquí se compone
    desde umbral_p, que es la cifra verificada. None = el corte no es una probabilidad.
    """
    p = MODELS[clave]["umbral_p"]
    plantilla = S("res_thr_label")[clave]
    return plantilla.format(v=nf(p, 2)) if p is not None else plantilla

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
                     tickfont=dict(family=PLOTLY_MONO, size=13))
    fig.update_yaxes(showline=False, zeroline=False, griddash="dot", gridwidth=1,
                     tickfont=dict(family=PLOTLY_MONO, size=13))
    fig.update_layout(
        height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOTLY_FONT, color=t["text_secondary"], size=12),
        # Separadores de los ticks y del hover, en el idioma activo: ES usa coma decimal y
        # punto de millar; EN, la convención inversa. Estaba fijo en ",." y las gráficas
        # seguían dando "0,95" con la app en inglés, justo donde la cifra manda.
        separators=",." if LANG == "es" else ".,",
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
if page == "overview":
    header(S("ov_eyebrow"), S("ov_title"), S("ov_subtitle"))

    # El párrafo llega de i18n.py como prosa con <b>, sin un solo atributo de estilo: el
    # tamaño, el color y el realce los pone .lead-card p / .lead-card p b en la hoja de
    # estilos. Antes cada <b> traía su color incrustado y el texto era irrevisable.
    st.markdown(f'<div class="info-card lead-card" style="margin-bottom:20px;">'
                f'<p>{S("ov_lead")}</p></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">{S("ov_stats_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{S("ov_stats_sub")}</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    # Las dos primeras cifras pasan por mil() en vez de ir escritas: estaban puestas a mano
    # con el punto de millar español ("29.400"), que en inglés se lee como 29 coma 4.
    stats = [(mil(29400), S("ov_stat_bronze")), (mil(7831), S("ov_stat_silver")),
             ("89", S("ov_stat_features")), ("86% / 14%", S("ov_stat_balance"))]
    for col, (num, lab) in zip(cols, stats):
        with col:
            st.markdown(f'<div class="info-card stat-card"><div class="stat-num">{num}</div><div class="stat-label">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<div class="section-title">{S("ov_medallion_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("ov_medallion_sub")}</div>', unsafe_allow_html=True)
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
        #
        # El numeral y el color NO viajan con la traducción: son estructura (orden de las capas
        # y paso de la rampa), no texto. i18n solo aporta el par nombre/descripción y aquí se
        # empareja por posición con lo que es igual en los dos idiomas.
        layers = [(f"{i + 1:02d}", nombre, desc, RAMP[i + 2])
                  for i, (nombre, desc) in enumerate(S("ov_layers"))]
        for num, name, desc, color in layers:
            st.markdown(f"""
            <div class="medallion-item" style="border-left-color:{color};">
                <div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex-shrink:0;margin-top:3px;">
                    <div style="font-family:{FONT_MONO};font-size:13px;font-weight:600;
                                color:{t['text_muted']};letter-spacing:0.05em;">{num}</div>
                    <div style="width:9px;height:9px;border-radius:2px;background:{color};"></div>
                </div>
                <div>
                    <div class="medallion-name" style="color:{t['text']};">{name}</div>
                    <div style="font-size:13px;color:{t['text_secondary']};line-height:1.6;text-align:justify;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        # Salto a Gobernanza reutilizando el mismo mecanismo que el toggle de la sidebar:
        # option_menu vive en un iframe y no lee session_state.page por su cuenta, así que
        # hay que empujarle el índice con manual_select (menu_force_index se consume al leerse).
        if st.button(S("ov_goto_gov"), key="ir_gobernanza"):
            st.session_state.page = "governance"
            st.session_state.menu_force_index = i18n.PAGE_KEYS.index("governance")
            st.rerun()

    with col2:
        st.markdown(f'<div class="section-title">{S("ov_target_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("ov_target_sub")}</div>', unsafe_allow_html=True)
        pie_labels, pie_values = [S("ov_pie_no"), S("ov_pie_yes")], [86, 14]
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
        fig.add_annotation(text=S("ov_donut_center"), x=0.5, y=0.545, xref="paper", yref="paper", showarrow=False,
                           font=dict(family=PLOTLY_MONO, size=34, color=t["text"]))
        fig.add_annotation(text=S("ov_donut_caption"), x=0.5, y=0.40, xref="paper", yref="paper", showarrow=False,
                           font=dict(family=PLOTLY_MONO, size=12.5, color=t["text_muted"]))
        plotly_layout(fig, height=300, showlegend=False, margin=dict(l=30, r=30, t=45, b=45))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(f'<div class="section-title" style="margin-top:6px;">{S("ov_compare_title")}</div>', unsafe_allow_html=True)
    # Mismos tonos que en Resultados: el color sigue al modelo en toda la aplicación. El color
    # se empareja por posición con el par nombre/descripción traducido, en el orden fijo de
    # MODEL_ORDER: el nombre del modelo es el mismo en los dos idiomas, solo cambia la glosa.
    labels3 = [(nombre, desc, SERIES[clave])
               for (nombre, desc), clave in zip(S("ov_compare"), MODEL_ORDER)]
    # HTML sin saltos ni indentación: Streamlit trataría las líneas con 4+ espacios como bloque de código.
    _compare_cards = "".join(
        f'<div class="info-card" style="border-top:2px solid {color};">'
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:9px;">'
        f'<span style="width:8px;height:8px;border-radius:2px;background:{color};flex-shrink:0;"></span>'
        f'<span style="font-size:13.5px;font-weight:600;color:{t["text"]};letter-spacing:0.01em;">{name}</span></div>'
        f'<div style="font-size:13px;color:{t["text_secondary"]};line-height:1.6;text-align:justify;">{desc}</div></div>'
        for name, desc, color in labels3
    )
    st.markdown(f'<div class="compare-grid">{_compare_cards}</div>', unsafe_allow_html=True)

    # Cierre de la página: sobre qué está construido. Va al final y no al principio porque es
    # una credencial, no una explicación — se mira después de saber qué es esto, y el enlace a
    # Gobernanza de más arriba ya lleva a quien quiera el inventario razonado.
    st.markdown(f'<div class="section-title" style="margin-top:26px;">{S("ov_tech_title")}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{S("ov_tech_sub")}</div>', unsafe_allow_html=True)
    tech_strip()

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 2 — GOBERNANZA
# ═══════════════════════════════════════════════════════════════════════
elif page == "governance":
    header(S("gov_eyebrow"), S("gov_title"), S("gov_subtitle"))

    tab_calidad, tab_linaje, tab_stack = st.tabs(S("gov_tabs"))

    # ─────────────────────────── TAB A — CALIDAD ───────────────────────────
    with tab_calidad:
        _kpis = [
            (f"{GOV_SUITE['passed']}/{GOV_SUITE['total']}", S("gov_kpi_expect")),
            (nf(GOV_SUITE["pass_rate"], 1), S("gov_kpi_passrate")),
            (mil(GOV_SUITE["registros"]), S("gov_kpi_records")),
            ("15/15", S("gov_kpi_leakage")),
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
        st.markdown(f'<div class="section-title">{S("gov_funnel_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">'
                    f'{S("gov_funnel_sub").format(bronze=mil(29400), silver=mil(7831))}</div>',
                    unsafe_allow_html=True)

        # Nombre y explicación de cada etapa llegan traducidos; los conteos, de GOV_EMBUDO_N.
        # Se emparejan aquí por posición y el resto de la sección trabaja ya con la tupla unida.
        _embudo = [(nom, val, perd, desc)
                   for (nom, desc), (val, perd) in zip(S("gov_embudo"), GOV_EMBUDO_N)]

        _fcol, _ncol = st.columns([1.35, 1], gap="medium")
        with _fcol:
            # Las etapas son una PROGRESIÓN (cada una contiene a la siguiente): magnitud, no
            # identidad. Le corresponde la rampa secuencial, no colores categóricos. Se recorre
            # de RAMP[0] a RAMP[3] para que el refinamiento creciente se lea en el propio color.
            _nombres = [e[0] for e in _embudo][::-1]
            _valores = [e[1] for e in _embudo][::-1]
            _colores = [RAMP[i] for i in range(len(_embudo))][::-1]
            _hover = [_wrap_hover(e[3]) for e in _embudo][::-1]
            _perdidos = [e[2] for e in _embudo][::-1]
            _etiquetas = [mil(v) for v in _valores]

            fig = go.Figure(go.Bar(
                x=_valores, y=_nombres, orientation="h",
                marker_color=_colores, cliponaxis=False,
                text=_etiquetas, textposition="outside",
                textfont=dict(family=PLOTLY_MONO, size=12, color=t["text"]),
                customdata=list(zip(_hover, [("—" if p is None else mil(p))
                                              for p in _perdidos])),
                hovertemplate=f'<b>%{{y}}</b><br>{S("gov_hover_records")}: %{{x:,}}'
                              f'<br>{S("gov_hover_dropped")}: %{{customdata[1]}}'
                              "<br>%{customdata[0]}<extra></extra>",
            ))
            # 272px = altura natural de la tarjeta vecina ("Registros descartados por filtro":
            # 5 kpi-row + cabecera + relleno). Igualarla deja las dos tarjetas a ras; el margen
            # inferior de 46 da aire a las cifras del eje X, que con 28 quedaban pegadas al borde.
            plotly_layout(fig, height=272, showlegend=False,
                          margin=dict(l=8, r=70, t=10, b=46),
                          xaxis=dict(range=[0, max(_valores) * 1.18], showgrid=True,
                                     gridcolor=GRID, fixedrange=True),
                          yaxis=dict(showgrid=False, fixedrange=True,
                                     tickfont=dict(family=PLOTLY_FONT, size=13.5)))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with _ncol:
            _pasos = "".join(
                f'<div class="kpi-row"><span class="kpi-label">{etapa}</span>'
                f'<span class="kpi-value">'
                f'{"—" if perdidos is None else "−" + mil(perdidos)}'
                f'</span></div>'
                for etapa, _v, perdidos, _d in _embudo)
            st.markdown(
                f'<div class="info-card"><div class="kpi-model">{S("gov_dropped_title")}</div>'
                f'{_pasos}'
                f'<div class="kpi-row"><span class="kpi-label">{S("gov_split_label")}</span>'
                f'<span class="kpi-value">{mil(6264)} / {mil(1567)}</span></div></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{S("gov_suite_title")}</div>',
                     unsafe_allow_html=True)
        # Cada cifra se formatea por separado con mil()/nf(): aplicar un replace a la frase
        # entera se llevaría por delante las comas de la prosa, y además el separador correcto
        # depende del idioma.
        st.markdown(
            f'<div class="section-sub">{S("gov_suite_sub").format(nombre=GOV_SUITE["nombre"], fecha=GOV_SUITE["fecha"], registros=mil(GOV_SUITE["registros"]), duracion=nf(GOV_SUITE["duracion_s"], 4))}</div>',
            unsafe_allow_html=True)

        _expectativas = S("gov_expectativas")
        _filas, _dim_previa = [], None
        for dim, col, regla in _expectativas:
            if dim != _dim_previa:
                _n = sum(1 for d, _, _ in _expectativas if d == dim)
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
        st.markdown(f'<div class="section-title">{S("gov_ops_title")}</div>',
                     unsafe_allow_html=True)
        # La partición 80/20 es la única cifra de estas dos tarjetas con separador de millar,
        # así que se compone aquí con mil() en vez de venir escrita en la lista de valores.
        _gold_vals = [v if v is not None else f"{mil(6264)} / {mil(1567)}" for v in GOV_GOLD_VALS]
        _ocol1, _ocol2 = st.columns(2, gap="medium")
        for _col, _titulo, _textos, _vals in (
                (_ocol1, S("gov_silver_card"), S("gov_silver_ops"), GOV_SILVER_VALS),
                (_ocol2, S("gov_gold_card"), S("gov_gold_ops"), _gold_vals)):
            with _col:
                _rows = "".join(
                    f'<div class="kpi-row" style="align-items:flex-start;">'
                    f'<span class="kpi-label" style="max-width:62%;">{lab}'
                    f'<span style="display:block;font-size:13.5px;color:{t["text_muted"]};'
                    f'line-height:1.5;margin-top:3px;">{det}</span></span>'
                    f'<span class="kpi-value">{val}</span></div>'
                    for (lab, det), val in zip(_textos, _vals))
                st.markdown(
                    f'<div class="info-card"><div class="kpi-model">'
                    f'<span class="kpi-dot" style="background:{C_PRIMARY};"></span>{_titulo}</div>'
                    f'{_rows}</div>', unsafe_allow_html=True)

        # Features efectivas. Un StandardScaler asigna scale_ = 1,0 exacto a las columnas de
        # varianza nula, así que las constantes se pueden CONTAR desde el propio scaler en vez
        # de transcribirlas: si el pipeline se reejecuta, esta cifra se mueve sola. El notebook
        # 03 imprime lo mismo ("Columnas excluidas (varianza = 0): 23"), pero hasta ahora la
        # app no lo decía en ninguna parte y anunciaba 89 features como si todas informaran.
        _sp_gov = _load_scaler_and_medians()
        if _sp_gov is not None:
            _const = [f for f, s in zip(_sp_gov["features"], _sp_gov["scale"]) if float(s) == 1.0]
            _n_tot, _n_const = len(_sp_gov["features"]), len(_const)
            _n_efec = _n_tot - _n_const
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">{S("gov_eff_title")}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="section-sub">'
                        f'{S("gov_eff_sub").format(const=_n_const, total=_n_tot)}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-card">
              <div class="kpi-row"><span class="kpi-label">{S("gov_eff_nominal")}</span>
                   <span class="kpi-value">{_n_tot}</span></div>
              <div class="kpi-row"><span class="kpi-label">{S("gov_eff_const")}</span>
                   <span class="kpi-value" style="color:{STATUS['warning']};">{_n_const}</span></div>
              <div class="kpi-row"><span class="kpi-label">{S("gov_eff_effective")}</span>
                   <span class="kpi-value">{_n_efec}</span></div>
              <div style="margin-top:12px; padding-top:11px; border-top:1px solid {t['border']};
                          font-size:13px; color:{t['text_secondary']}; line-height:1.65; text-align:justify;">
                {S("gov_eff_note")}
              </div>
              <div style="margin-top:10px; font-family:{FONT_MONO}; font-size:11.5px;
                          color:{t['text_muted']}; line-height:1.75; word-break:break-word;">
                {" · ".join(_const)}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ────────────────────── TAB B — LINAJE Y TRAZABILIDAD ──────────────────────
    with tab_linaje:
        st.markdown(f'<div class="section-title">{S("gov_lin_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("gov_lin_sub")}</div>', unsafe_allow_html=True)

        # Rejilla de 2 en vez de st.columns: la limitación es más corta que su mitigación, y con
        # columnas la tarjeta izquierda quedaba visiblemente más baja. El estirado es nativo del
        # grid, así que ambas igualan altura sea cual sea el texto o el zoom.
        # Tipografía en .gov-prose / .gov-prose-title y no inline: así el bloque móvil puede
        # bajarlas junto al .section-sub de arriba, que es con quien comparten tamaño.
        st.markdown(
            f'<div class="compare-grid" style="grid-template-columns:repeat(2, minmax(0, 1fr));">'
            f'<div class="info-card" style="border-top:2px solid {STATUS["critical"]};">'
            f'<div class="kpi-model gov-prose-title">'
            f'<span class="kpi-dot" style="background:{STATUS["critical"]};"></span>'
            f'{S("gov_lin_limit_title")}</div>'
            f'<div class="gov-prose">{S("gov_lin_limit_body")}</div></div>'
            f'<div class="info-card" style="border-top:2px solid {STATUS["good"]};">'
            f'<div class="kpi-model gov-prose-title">'
            f'<span class="kpi-dot" style="background:{STATUS["good"]};"></span>'
            f'{S("gov_lin_mit_title")}</div>'
            f'<div class="gov-prose">{S("gov_lin_mit_body")}</div></div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{S("gov_delta_title")}</div>',
                     unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("gov_delta_sub")}</div>', unsafe_allow_html=True)
        # Las dos cifras se formatean una a una con mil()/nf() en vez de aplicar un replace a
        # la fila entera: el replace global solo era seguro mientras el separador español
        # estuviera fijo, y ahora depende del idioma (además de que arrasaría cualquier coma
        # que llegue a asomar en el nombre de la operación).
        _hist = "".join(
            f'<tr><td class="num">{v}</td><td class="num">{ts}</td><td>{op}</td>'
            f'<td class="num">{mil(filas)}</td><td class="num">{nf(bytes_ / 1024, 0)} KB</td></tr>'
            for v, ts, op, filas, bytes_ in GOV_DELTA_HISTORY)
        _cabeceras = "".join(f"<th>{c}</th>" for c in S("gov_delta_cols"))
        st.markdown(
            f'<div class="info-card"><div class="gov-table-wrap"><table class="gov-table">'
            f'<thead><tr>{_cabeceras}</tr></thead><tbody>{_hist}</tbody></table></div></div>',
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{S("gov_chain_title")}</div>',
                     unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("gov_chain_sub")}</div>', unsafe_allow_html=True)
        # Mismo componente que la arquitectura Medallón: numeral + acento lateral + descripción.
        # El acento sigue la rampa secuencial porque las barreras son una secuencia, no identidades.
        for _i, ((nombre, desc), num) in enumerate(zip(S("gov_leakage"), GOV_LEAKAGE_N)):
            _c = RAMP[_i + 1]
            st.markdown(
                f'<div class="medallion-item" style="border-left-color:{_c};">'
                f'<div style="display:flex;flex-direction:column;align-items:center;gap:6px;'
                f'flex-shrink:0;margin-top:3px;">'
                f'<div style="font-family:{FONT_MONO};font-size:13px;font-weight:600;'
                f'color:{t["text_muted"]};letter-spacing:0.05em;">{num}</div>'
                f'<div style="width:9px;height:9px;border-radius:2px;background:{_c};"></div></div>'
                f'<div style="min-width:0;">'
                f'<div class="medallion-name" style="color:{t["text"]};">{nombre}</div>'
                f'<div style="font-size:13px;color:{t["text_secondary"]};line-height:1.6;text-align:justify;">{desc}</div>'
                f'</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _scol1, _scol2 = st.columns([1, 1.15], gap="medium")
        with _scol1:
            _srows = "".join(
                f'<div class="kpi-row" style="align-items:flex-start;">'
                f'<span class="kpi-label" style="max-width:56%;">{lab}'
                f'<span style="display:block;font-size:13.5px;color:{t["text_muted"]};'
                f'line-height:1.5;margin-top:3px;">{det}</span></span>'
                f'<span class="kpi-value">{val}</span></div>'
                for lab, val, det in S("gov_scaler"))
            st.markdown(
                f'<div class="info-card"><div class="kpi-model">'
                f'<span class="kpi-dot" style="background:{C_PRIMARY};"></span>'
                f'{S("gov_scaler_card")}</div>{_srows}</div>', unsafe_allow_html=True)
        with _scol2:
            st.markdown(f'<div class="clinical-note">{S("gov_scaler_note")}</div>',
                        unsafe_allow_html=True)

        # Verificación end-to-end. Es el cierre del linaje: comprueba que ESTE dashboard
        # reproduce los modelos entrenados, no solo que sus cifras son coherentes entre sí.
        _gold = verificar_golden()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{S("gov_e2e_title")}</div>',
                    unsafe_allow_html=True)
        if _gold["_estado"] == "ausente":
            st.markdown(
                f'<div class="clinical-note">'
                f'{S("gov_e2e_missing").format(color=STATUS["warning"])}</div>',
                unsafe_allow_html=True)
        else:
            _filas = ""
            for _k, _cfg in GOLDEN.items():
                _r = _gold.get(_k)
                _lab = MODELS[_k]["label"]
                if _r is None:
                    _val, _col = S("gov_e2e_unavailable"), t["text_muted"]
                elif _r["error"]:
                    _val, _col = _r["error"], STATUS["critical"]
                elif _r["ok"]:
                    _val = S("gov_e2e_ok_val").format(n=_r["n"], dif=f"{_r['max_dif']:.2e}")
                    _col = STATUS["good"]
                else:
                    _val = S("gov_e2e_bad_val").format(dif=f"{_r['max_dif']:.2e}")
                    _col = STATUS["critical"]
                _esc = S("gov_e2e_scaled") if _cfg["escalar"] else S("gov_e2e_raw")
                _filas += (f'<div class="kpi-row" style="align-items:flex-start;">'
                           f'<span class="kpi-label" style="max-width:52%;">{_lab}'
                           f'<span style="display:block;font-size:13px;color:{t["text_muted"]};'
                           f'line-height:1.5;margin-top:3px;">'
                           f'{S("gov_e2e_path").format(accion=_esc)}</span></span>'
                           f'<span class="kpi-value" style="color:{_col};">{_val}</span></div>')
            _ok = _gold["_estado"] == "ok"
            _titulo = S("gov_e2e_ok_title") if _ok else S("gov_e2e_fail_title")
            _tcol = STATUS["good"] if _ok else STATUS["critical"]
            st.markdown(f"""
            <div class="info-card">
              <div class="kpi-model"><span class="kpi-dot" style="background:{_tcol};"></span>
                   <span style="color:{_tcol};">{_titulo}</span></div>
              {_filas}
              <div style="margin-top:12px; padding-top:11px; border-top:1px solid {t['border']};
                          font-size:13px; color:{t['text_secondary']}; line-height:1.65; text-align:justify;">
                {S("gov_e2e_note").format(tol=f"{GOLDEN_TOL:.0e}")}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ──────────────────── TAB C — INVENTARIO DE FRAMEWORKS ────────────────────
    with tab_stack:
        st.markdown(f'<div class="section-title">{S("gov_stack_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("gov_stack_sub")}</div>', unsafe_allow_html=True)

        # Nombre de capa, color y badges son estructura (GOV_STACK); rol y nota, texto (i18n).
        _stack = [(capa, rol, color, badges, nota)
                  for (capa, color, badges), (rol, nota) in zip(GOV_STACK, S("gov_stack"))]
        for _inicio in (0, 3):
            _cards = "".join(
                f'<div class="info-card" style="border-top:2px solid {color};">'
                f'<div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px;'
                f'margin-bottom:11px;">'
                f'<span style="font-size:14px;font-weight:600;color:{t["text"]};">{capa}</span>'
                f'<span style="font-family:{FONT_MONO};font-size:11.5px;font-weight:500;'
                f'letter-spacing:0.12em;text-transform:uppercase;color:{t["text_muted"]};">{rol}</span>'
                f'</div>'
                f'<div style="margin-bottom:12px;line-height:2.1;">'
                + "".join(f'<span class="badge">{b}</span>' for b in badges) + "</div>"
                f'<div style="font-size:12.5px;color:{t["text_secondary"]};line-height:1.65;text-align:justify;">{nota}</div>'
                f'</div>'
                for capa, rol, color, badges, nota in _stack[_inicio:_inicio + 3])
            st.markdown(f'<div class="compare-grid">{_cards}</div>', unsafe_allow_html=True)
            if _inicio == 0:
                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{S("gov_dec_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("gov_dec_sub")}</div>', unsafe_allow_html=True)

        _etiquetas_nivel = S("gov_dec_tags")
        for (ref, nivel), (titulo, problema, solucion) in zip(GOV_DECISIONES_META, S("gov_decisiones")):
            _c = STATUS[nivel]
            with st.expander(f"{ref} · {titulo}"):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
                    f'<span class="gov-dot" style="background:{_c};margin-left:0;"></span>'
                    f'<span class="gov-state" style="color:{_c};">{_etiquetas_nivel[nivel]}</span></div>'
                    f'<div style="font-size:13px;color:{t["text_secondary"]};line-height:1.7;text-align:justify;">'
                    f'<b style="color:{t["text"]};">{S("gov_dec_problem")}</b>{problema}<br><br>'
                    f'<b style="color:{t["text"]};">{S("gov_dec_solution")}</b>{solucion}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="clinical-note">'
            f'{S("gov_footer_note").format(fuente=S(GOV_SUITE_FUENTE))}</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 3 — RESULTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "results":
    header(S("res_eyebrow"), S("res_title"), S("res_subtitle").format(n=mil(1567)))

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
                <div class="kpi-row" title="{S("res_thr_src")[key]}">
                    <span class="kpi-label" style="color:{t['text_muted']};">{S("res_threshold")}</span>
                    <span class="kpi-value" style="color:{t['text_muted']};">{thr_text(key)}</span></div>
            </div>
            """, unsafe_allow_html=True)

    # Las tres métricas de abajo dependen del umbral y los tres umbrales son distintos: sin este
    # aviso la fila de tarjetas invita a leerlas como directamente comparables, y no lo son.
    # El estado de reconciliación NO se escribe a mano: sale de recalcular las cuatro métricas
    # desde los scores por instancia, así que si un reentrenamiento desincroniza los .npy de las
    # cifras transcritas, la página lo denuncia en vez de seguir mostrando números muertos.
    _rec = reconciliar_metricas()
    if _rec["_global"]:
        _sello = S("res_reconciled").format(color=STATUS["good"])
    else:
        _fallos = [f"{MODELS[k]['label']}: " + (S("res_no_scores") if v is None else "; ".join(v["detalle"]))
                   for k, v in _rec.items() if k != "_global" and (v is None or not v["ok"])]
        _sello = S("res_unreconciled").format(color=STATUS["warning"], fallos=" · ".join(_fallos))
    st.markdown(f"""
    <div class="clinical-note" style="margin-top:14px;">
    {S("res_threshold_note")}
    <div style="margin-top:10px; padding-top:9px; border-top:1px solid {t['border']};">{_sello}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{S("res_roc_title")}</div>', unsafe_allow_html=True)

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
        f'<div class="section-sub">{S("res_roc_sub_real") if all_real else S("res_roc_sub_synth")}</div>',
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
                      xaxis=dict(title=dict(text="FPR", font=dict(size=13)), range=[0, 1], showgrid=False, fixedrange=True),
                      yaxis=dict(title=dict(text="TPR", font=dict(size=13)), range=[0, 1], showgrid=True, gridcolor=GRID, fixedrange=True))
        with col:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{S("res_cm_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{S("res_cm_sub")}</div>', unsafe_allow_html=True)

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

    _cm_tags = S("res_cm_tags")
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
                <div class="cm-collabel">{S("res_cm_pred_no")}</div>
                <div class="cm-collabel">{S("res_cm_pred_yes")}</div>
                <div class="cm-rowlabel">{S("res_cm_real_no")}</div>
                {cm_cell(cm['tn'], r0, _cm_tags["tn"], m['color'])}
                {cm_cell(cm['fp'], r0, _cm_tags["fp"], m['color'])}
                <div class="cm-rowlabel">{S("res_cm_real_yes")}</div>
                {cm_cell(cm['fn'], r1, _cm_tags["fn"], m['color'])}
                {cm_cell(cm['tp'], r1, _cm_tags["tp"], m['color'])}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{S("res_metrics_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{S("res_metrics_sub")}</div>', unsafe_allow_html=True)
    fig = go.Figure()
    metric_keys, metric_labels = ["auc", "f1_macro", "accuracy", "mcc"], ["AUC-ROC", "F1-macro", "Accuracy", "MCC"]
    metric_desc = S("res_metric_desc")
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
                  # 13px: la leyenda es el cuarto sitio donde aparece el nombre del modelo en esta
                  # página, y va con las mismas medidas que los otros tres (tarjeta KPI, título de
                  # la curva ROC y matriz de confusión). Plotly ya pinta la leyenda en peso normal.
                  legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
                              font=dict(family=PLOTLY_FONT, size=13, color=t["text"]),
                              bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
                  yaxis=dict(range=[0, 1.10], showgrid=True, gridcolor=GRID, fixedrange=True),
                  xaxis=dict(showgrid=False, tickfont=dict(family=PLOTLY_FONT, size=13, color=t["text"]), fixedrange=True))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(f"""
    <div class="clinical-note" style="margin-top:6px;">
    {S("res_qsvm_note")}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 3 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif page == "shap":
    header(S("sh_eyebrow"), S("sh_title"), S("sh_subtitle"))

    tab1, tab2 = st.tabs(S("sh_tabs"))

    def shap_chart(data, color, sample_note):
        # El eje lleva el CÓDIGO NHANES (que no se traduce) y el hover, su glosa: por eso la
        # lista de datos solo trae (código, valor) y el rótulo se resuelve aquí, en el idioma
        # activo, en vez de venir incrustado en los datos.
        rev = list(reversed(data))
        _short = S("var_short")
        names = [code for code, _ in rev]
        values = [v for _, v in rev]
        customdata = [[code, _short.get(code, code), _wrap_hover(VAR_DESC.get(code, _short.get(code, code)))]
                      for code, _ in rev]
        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h", marker_color=color, cliponaxis=False,
            # El valor va en monoespaciada y en tinta secundaria, no en el color de la barra:
            # el texto nunca lleva el color de la serie (lo aporta la propia barra al lado).
            text=[nf(v) for v in values], textposition="outside",
            textfont=dict(family=PLOTLY_MONO, size=12.5, color=t["text_secondary"]),
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        ))
        plotly_layout(fig, height=520, hovermode="y", bargap=0.34,
                      xaxis=dict(title=dict(text="mean(|SHAP value|)", font=dict(size=13)),
                                 showgrid=True, gridcolor=GRID, range=[0, max(values) * 1.3], fixedrange=True),
                      # showgrid=False como en el embudo de Gobernanza: en una barra horizontal la
                      # rejilla del eje de categorías pasa por el centro de cada fila, justo por donde
                      # va la cifra del extremo, y la tacha. Sin él la rejilla del eje X ya orienta.
                      yaxis=dict(showgrid=False, tickfont=dict(family=PLOTLY_MONO, size=13, color=t["text"]),
                                 fixedrange=True),
                      margin=dict(l=170, r=60, t=20, b=40))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(f'<div class="section-sub" style="margin-top:10px;">{S("sh_hint").format(nota=sample_note)}</div>', unsafe_allow_html=True)

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
        {S("sh_note_lgbm")}
        </div>
        """, unsafe_allow_html=True)
        # Cada pestaña usa el color de SU modelo (antes las dos iban en el mismo azul de marca):
        # así el ranking se lee sin ambigüedad como perteneciente a LightGBM o a SVM-RBF.
        shap_chart(SHAP_LIGHTGBM, SERIES["lightgbm"], S("sh_sample_lgbm"))
        shap_summary_image("SHAP Summary LGBM.png", S("sh_fig_lgbm_title"), S("sh_fig_lgbm_cap"))

    with tab2:
        st.markdown(f"""
        <div class="clinical-note" style="margin-bottom:14px;">
        {S("sh_note_svm")}
        </div>
        """, unsafe_allow_html=True)
        shap_chart(SHAP_SVMRBF, SERIES["svm_rbf"], S("sh_sample_svm"))
        shap_summary_image("SHAP Summary SVM.png", S("sh_fig_svm_title"), S("sh_fig_svm_cap"))

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 4 — QUANTUM CIRCUIT
# ═══════════════════════════════════════════════════════════════════════
elif page == "circuit":
    header(S("qc_eyebrow"), S("qc_title"), S("qc_subtitle"))

    # Misma construcción que los cuatro KPIs de Gobernanza → Calidad del dato: una sola .compare-grid
    # de cuatro columnas iguales en vez de st.columns. El grid estira las tarjetas a la misma altura
    # y reparte el ancho en fracciones, así que la fila se reescala entera con el zoom en lugar de
    # que cada columna crezca por su cuenta. Lo único propio de esta página es la clase .quantum,
    # que pinta el filete superior en oro.
    _specs = zip(["8", "2", "Linear", "2.5.0"], S("qc_specs"))
    st.markdown(
        '<div class="compare-grid" style="grid-template-columns:repeat(4, minmax(0, 1fr));">'
        + "".join(
            f'<div class="info-card stat-card quantum">'
            f'<div class="stat-num">{v}</div>'
            f'<div class="stat-label">{lab}</div></div>'
            for v, lab in _specs)
        + "</div>", unsafe_allow_html=True)

    # "Cómo funciona" a ancho completo: sus dos párrafos son conceptualmente independientes
    # (codificación vs. kernel), así que van lado a lado en vez de apilados — a ancho completo el
    # texto apilado dejaría líneas incómodamente largas, y apilado-estrecho (como antes, dentro de
    # media página) dejaba la columna vecina con un hueco vacío grande por debajo.
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{S("qc_how_title")}</div>', unsafe_allow_html=True)
    # El realce de los <b> lo pone .qc-prose b en la hoja de estilos, no un style en línea:
    # así los dos párrafos se leen como prosa en los dos idiomas y se revisan de corrido.
    # El suelo de la pista va en min(320px, 100%), no en 320px a secas: auto-fit SIEMPRE coloca al
    # menos una pista y esa pista conserva su mínimo, así que en un teléfono de 360-375px (ancho útil
    # ~294-311px dentro de la tarjeta) los 320px desbordaban y sacaban scroll horizontal a toda la
    # página. Con min() el suelo cede al ancho disponible cuando ya no cabe; por encima de 320px el
    # comportamiento es idéntico al de antes.
    st.markdown(f"""
    <div class="info-card">
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(min(320px, 100%), 1fr)); gap:28px;">
        <p class="qc-prose">{S("qc_how_p1")}</p>
        <p class="qc-prose">{S("qc_how_p2")}</p>
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
        st.markdown(f'<div class="section-title">{S("qc_feat_title")}</div>', unsafe_allow_html=True)
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
            textfont=dict(family=PLOTLY_MONO, size=12.5, color=t["text_secondary"]),
            customdata=customdata, showlegend=False,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
        ))
        plotly_layout(fig, height=300, barmode="overlay", bargap=0.34,
                      # Margen izquierdo reducido (150→95): pega las etiquetas al borde de la tarjeta
                      # en vez de dejarlas centradas con aire de sobra. Al ser el ancho de la tarjeta
                      # fijo, ese espacio liberado pasa directo al área de barras — se agrandan solas,
                      # de forma proporcional, sin tocar la tarjeta que las contiene.
                      xaxis=dict(title=dict(text=S("qc_xaxis"), font=dict(size=13)),
                                 showgrid=True, gridcolor=GRID, range=[0, max(values) * 1.3], fixedrange=True),
                      # showgrid=False como en el embudo de Gobernanza: la rejilla del eje de
                      # categorías cruza el centro de cada fila y tacha la cifra del extremo.
                      yaxis=dict(showgrid=False, tickfont=dict(family=PLOTLY_MONO, size=13, color=t["text"]),
                                 fixedrange=True),
                      margin=dict(l=95, r=70, t=20, b=40))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with col2:
        st.markdown(f'<div class="section-title">{S("qc_train_title")}</div>', unsafe_allow_html=True)
        # Lista vertical de KPIs (mismo patrón .kpi-row que Esfera de Bloch / Predictor en Vivo) en vez
        # de tarjetas en grilla: más compacta en una columna estrecha y de un vistazo. Incluye los dos
        # datos que antes solo estaban en la nota de texto (instancias del test y tiempo de inferencia).
        # Los tres valores con cifra pasan por nf()/mil(): "21,1 min" y "1.567" iban escritos a mano y
        # se quedaban con el separador español al cambiar de idioma.
        tstats = zip(S("qc_tstats"),
                     ["500", f"{nf(21.1, 1)} min", mil(1567), f"{nf(144.5, 1)} min", "[425, 70]"])
        kpi_rows = "".join(f'<div class="kpi-row"><span class="kpi-label">{l}</span><span class="kpi-value">{v}</span></div>' for l, v in tstats)
        st.markdown(f'<div class="info-card">{kpi_rows}</div>', unsafe_allow_html=True)

    # La nota va DEBAJO de la fila (ancho completo), no dentro de col2: así la tarjeta de KPIs es el
    # único elemento de esa columna y puede estirarse limpio hasta igualar la altura de la gráfica —
    # si la nota se quedara dentro de col2, empujaría esa columna más abajo que la de la gráfica.
    st.markdown(f"""
    <div class="clinical-note" style="margin-top:16px;">
    {S("qc_note")}
    </div>
    """, unsafe_allow_html=True)

    # Diagrama del circuito a ancho completo (fuera de col1/col2: con 8 qubits y las 4 secciones de
    # entrelazamiento apiladas, comprimirlo a la mitad de la página dejaría las etiquetas P(...) ilegibles.
    _circuit_path = FIGURES_DIR / "Circuito Cuantico 8qb.png"
    if _circuit_path.exists():
        st.markdown(f'<div class="section-title" style="margin-top:20px;">{S("qc_circuit_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("qc_circuit_sub")}</div>', unsafe_allow_html=True)
        _circuit_b64 = _b64_image_autocrop(str(_circuit_path))
        st.markdown(f"""
        <div class="fig-card" style="padding:14px; max-width:900px; margin:0 auto;">
            <img src="data:image/png;base64,{_circuit_b64}" style="width:100%; display:block; border-radius:6px;">
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 5 — BLOCH SPHERE EMULATOR
# ═══════════════════════════════════════════════════════════════════════
elif page == "bloch":
    header(S("bl_eyebrow"), S("bl_title"), S("bl_subtitle"))

    # Contenedor con clave (genera .st-key-bloch_row) para poder estirar la gráfica 3D hasta el alto
    # de la columna izquierda y que ambas tarjetas cierren alineadas abajo — ver CSS .st-key-bloch_row.
    bloch_row = st.container(key="bloch_row")
    col1, col2 = bloch_row.columns([1, 1.3])
    with col1:
        var_code = st.selectbox(S("bl_var"), list(QSVM_FEATURES.keys()),
                                 format_func=lambda c: f"{c} — {q_label(c)}")
        v = QSVM_FEATURES[var_code]
        lo, hi = v["range"]
        val = st.slider(S("bl_value").format(unidad=q_unit(var_code)),
                        float(lo), float(hi), float(v["default"]),
                        step=v["step"], format=v["fmt"])

        x_norm = (val - lo) / (hi - lo)
        # θ ∈ [0, π], NO [0, 2π]. La parametrización estándar de la esfera de Bloch es
        # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}·sin(θ/2)|1⟩ con θ acotado a [0, π]: recorrer 2π daba una
        # vuelta completa y hacía la representación doblemente degenerada — el mínimo y el
        # máximo de cada variable caían en el MISMO estado (HbA1c 4,0 % y 15,0 % daban ambos
        # P(|0⟩) = 100 %) — y además volvía α = cos(θ/2) negativo en toda la mitad superior
        # del rango, mostrado sin explicación en la tarjeta de amplitudes.
        theta = x_norm * np.pi
        alpha = np.cos(theta / 2)
        beta = np.sin(theta / 2)
        p0, p1 = alpha**2, beta**2

        st.markdown(f"""
        <div class="info-card">
            <div class="kpi-row"><span class="kpi-label">{S("bl_xnorm")}</span><span class="kpi-value">{nf(x_norm, 3)}</span></div>
            <div class="kpi-row"><span class="kpi-label">{S("bl_theta")}</span><span class="kpi-value">{nf(theta, 3)} {S("bl_rad")}</span></div>
            <div class="kpi-row"><span class="kpi-label">{S("bl_alpha")}</span><span class="kpi-value">{nf(alpha, 3)}</span></div>
            <div class="kpi-row"><span class="kpi-label">{S("bl_beta")}</span><span class="kpi-value">{nf(beta, 3)}</span></div>
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
    {S("bl_note")}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 6 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════
elif page == "predictor":
    header(S("lp_eyebrow"), S("lp_title"), S("lp_subtitle"))

    # QUÉ ESTIMA ESTE FORMULARIO (y qué NO). El target del pipeline es TARGET = (DIQ010 == 1),
    # es decir la respuesta a "¿un médico le ha dicho alguna vez que tiene diabetes?". Eso hace
    # que esto sea un DETECTOR de diagnóstico ya existente, no un predictor de riesgo futuro, y
    # la diferencia no es semántica: cambia el signo con el que el modelo usa varias variables
    # (ver la nota de lectura más abajo). Toda la página se nombra en consecuencia.
    st.markdown(f"""
    <div class="clinical-note" style="margin-bottom:16px;">
    {S("lp_what_note")}
    </div>
    """, unsafe_allow_html=True)

    _sp = _load_scaler_and_medians()
    _models_ready = (ONNX_AVAILABLE and _sp is not None
                      and _load_onnx_session("lgbm_final.onnx") is not None
                      and _load_onnx_session("svm_final.onnx") is not None)

    st.markdown(f"""
    <div class="clinical-note" style="margin-bottom:16px;">
    {S("lp_real_note") if _models_ready else S("lp_proxy_note")}
    </div>
    """, unsafe_allow_html=True)

    # Rango de entrenamiento bajo cada slider. Los límites de los sliders son fisiológicos y
    # llegan mucho más lejos de lo que el modelo llegó a ver: con HbA1c el tope de 15 % está a
    # z = +12 de la media de entrenamiento (media 5,72, sd 0,77 tras la winsorización IQR × 3).
    # Fuera de ±3 sd el modelo extrapola y su respuesta se aplana, así que la frontera se marca
    # en pantalla en vez de recortar el slider — media y sd salen del scaler, no de una constante.
    _ref = {}
    if _sp is not None:
        for _c in QSVM_FEATURES:
            if _c in _sp["features"]:
                _i = _sp["features"].index(_c)
                _ref[_c] = (float(_sp["mean"][_i]), float(_sp["scale"][_i]))

    cols = st.columns(2)
    inputs = {}
    items = list(QSVM_FEATURES.items())
    for i, (code, v) in enumerate(items):
        with cols[i % 2]:
            lo, hi = v["range"]
            inputs[code] = st.slider(f"{q_label(code)} ({q_unit(code)})",
                                      float(lo), float(hi), float(v["default"]),
                                      step=v["step"], format=v["fmt"], key=f"lp_{code}")
            _pie = []
            if code in _ref:
                _mu, _sd = _ref[code]
                _dec = 1 if v["step"] < 1 else 0
                # Las cuatro cifras pasan por nf(): son magnitudes clínicas y el separador
                # decimal tiene que seguir al idioma igual que en el resto de la página.
                _pie.append(S("lp_train_range").format(
                    mu=nf(_mu, _dec), sd=nf(_sd, _dec),
                    lo=nf(_mu - 3 * _sd, _dec), hi=nf(_mu + 3 * _sd, _dec)))
                _z = (inputs[code] - _mu) / _sd if _sd else 0.0
                if abs(_z) > 3:
                    _pie.append(f'<span style="color:{STATUS["warning"]};">'
                                + S("lp_extrapolates").format(z=f"{nf(_z, 1)}" if _z < 0 else f"+{nf(_z, 1)}")
                                + "</span>")
            if code == "LBXGH":
                _pie.append(S("lp_ada"))
            if _pie:
                st.markdown(f'<div style="font-size:12px; color:{t["text_muted"]}; line-height:1.6; '
                            f'margin:-8px 0 14px;">{"<br>".join(_pie)}</div>', unsafe_allow_html=True)

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

    # Nombres SIEMPRE condicionados al origen del número. Con los .onnx cargados esto es la
    # probabilidad real de LightGBM y llamarla "sustituto" (como se hacía antes en los dos
    # ramales) contradecía al aviso de "Inferencia real" de arriba, en la misma pantalla.
    _es_real = _real is not None
    _quien = S("lp_who_model") if _es_real else S("lp_who_proxy")
    _etiqueta_score = S("lp_score_real") if _es_real else S("lp_score_proxy")

    # Categoría interpretable respecto al umbral de decisión (50%, el de LightGBM). Se nombra
    # como COMPATIBILIDAD con un diagnóstico ya existente, no como "riesgo": el target del
    # pipeline es DIQ010 == 1, así que un valor alto significa "este perfil se parece al de
    # alguien ya diagnosticado", no "esta persona va a desarrollar diabetes". Se conserva la
    # paleta de ESTADO (bien / atención / grave) y el nombre por texto además del color.
    if risk < 0.33:
        cat, cat_color = S("lp_cat_low"), STATUS["good"]
        interp = S("lp_interp_low").format(quien=_quien)
    elif risk < 0.5:
        cat, cat_color = S("lp_cat_mid"), STATUS["warning"]
        interp = S("lp_interp_mid")
    else:
        cat, cat_color = S("lp_cat_high"), STATUS["critical"]
        interp = S("lp_interp_high").format(quien=_quien)

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
            <div class="stat-label" style="margin:0 0 10px;">{_etiqueta_score}</div>
            <div class="kpi-value-auc" style="color:{cat_color}; font-size:clamp(30px, 4vw, 52px);">{pct(risk)}</div>
            <div style="margin-top:14px;">
                <span style="display:inline-flex; align-items:center; gap:8px; font-family:{FONT_MONO};
                      font-size:13px; font-weight:600; letter-spacing:0.09em; text-transform:uppercase;
                      padding:6px 13px; border-radius:7px; background:{cat_color}1A;
                      border:1px solid {cat_color}40; color:{cat_color};">
                    <span style="width:7px; height:7px; border-radius:2px; background:{cat_color};"></span>
                    {S("lp_badge").format(cat=cat)}
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
                               font=dict(size=12.5, color=t["text_muted"], family=PLOTLY_MONO))

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
        st.markdown(f'<div class="section-sub" style="text-align:center; margin-top:-6px;">{S("lp_gauge_caption")}</div>', unsafe_allow_html=True)

    # Los DOS modelos, cada uno con SU umbral. Antes el SVM se calculaba y se descartaba
    # (`risk, _svm_prob = _real`) aunque el encabezado prometía ambos. Mostrarlo importa porque
    # coinciden en los extremos y divergen justo en la zona de decisión clínica — y porque sus
    # probabilidades NO se leen con el mismo corte: el del SVM está en ≈ 0,22, no en 0,50.
    if _real is not None:
        _umbral_svm = MODELS["svm_rbf"]["umbral_valor"]
        _filas_mod = [
            ("LightGBM", SERIES["lightgbm"], risk, MODELS["lightgbm"]["umbral_valor"],
             thr_text("lightgbm")),
            ("SVM-RBF", SERIES["svm_rbf"], _svm_prob, _umbral_svm, thr_text("svm_rbf")),
        ]
        _cards = ""
        for _nom, _col, _p, _th, _thtxt in _filas_mod:
            _pos = _p >= _th
            _est = S("lp_positive") if _pos else S("lp_negative")
            _estc = STATUS["critical"] if _pos else STATUS["good"]
            _cards += (
                f'<div class="info-card" style="border-top:2px solid {_col};">'
                f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:10px;">'
                f'<span style="width:8px;height:8px;border-radius:2px;background:{_col};"></span>'
                f'<span style="font-size:13.5px;font-weight:600;color:{t["text"]};">{_nom}</span></div>'
                f'<div class="kpi-value-auc" style="font-size:30px;color:{t["text"]};">{pct(_p)}</div>'
                f'<div class="kpi-row" style="margin-top:12px;">'
                f'<span class="kpi-label">{S("lp_own_threshold")}</span>'
                f'<span class="kpi-value">{_thtxt}</span></div>'
                f'<div class="kpi-row"><span class="kpi-label">{S("lp_would_classify")}</span>'
                f'<span class="kpi-value" style="color:{_estc};">{_est}</span></div></div>')
        _dif = abs(risk - _svm_prob)
        st.markdown(f'<div class="section-title" style="margin-top:22px;">{S("lp_side_title")}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("lp_side_sub")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="compare-grid" style="grid-template-columns:repeat(2, minmax(0, 1fr));">'
                    f'{_cards}</div>', unsafe_allow_html=True)
        if _dif > 0.25:
            st.markdown(
                f'<div class="clinical-note" style="margin-top:12px;">'
                f'{S("lp_disagree").format(dif=pct(_dif))}</div>', unsafe_allow_html=True)

    # CURVA DE RESPUESTA. Un ensemble de árboles no produce una rampa: produce una escalera.
    # Sin verla, mover el slider 0,1 y que la probabilidad salte 17 puntos parece un fallo. Con
    # ella se ve que estás en una meseta y dónde está el siguiente peldaño. Se dibuja con
    # line_shape="hv" —escalón literal, jamás spline— porque suavizarla sería dibujar una
    # continuidad que el modelo no tiene. Los cortes de LBXGH caen todos en X,X5: NHANES da la
    # HbA1c con un decimal y LightGBM parte por el punto medio entre valores observados, así que
    # el modelo no distingue por debajo de 0,1 % y el slider nunca puede posarse en un umbral.
    if _real is not None:
        st.markdown(f'<div class="section-title" style="margin-top:22px;">{S("lp_curve_title")}</div>',
                    unsafe_allow_html=True)
        _cur_code = st.selectbox(S("lp_curve_var"), list(QSVM_FEATURES.keys()),
                                 format_func=lambda c: f"{c} — {q_label(c)}",
                                 key="lp_curva")
        _cv = QSVM_FEATURES[_cur_code]
        _clo, _chi = _cv["range"]
        _vals = tuple(round(float(x), 4) for x in
                      np.arange(float(_clo), float(_chi) + _cv["step"] / 2, _cv["step"]))
        _otros = tuple(sorted((k, float(v)) for k, v in inputs.items() if k != _cur_code))
        _bar = barrer_variable(_cur_code, _vals, _otros)
        if _bar is not None:
            _cl, _cs = _bar
            _fig = go.Figure()
            _cur_label = q_label(_cur_code)
            _cur_x = float(inputs[_cur_code])
            _dec = 1 if _cv["step"] < 1 else 0          # decimales del slider, para chip y hover
            # Bandas ADA solo donde significan algo: el eje X es HbA1c en esa variable. El tinte
            # CRECE con la severidad en vez de ir las tres al mismo valor: así el fondo dice que
            # las zonas están ORDENADAS en lugar de leerse como un bloque plano. Sigue siendo el
            # gris de texto, no un color de estado — es marco, no serie. Con la mancha bajo la
            # curva fuera (ver abajo), estas tres franjas son la ÚNICA estructura del lienzo y
            # pueden pesar algo más sin ensuciar nada.
            # Los rótulos NO van dentro del área de datos: van en una tira sobre el lienzo,
            # centrados en su banda. Dentro chocaban dos veces — "prediabetes" anclado a la
            # izquierda de una banda de 0,8 puntos se metía en "diabetes" ("prediabetediabetes"
            # en la captura), y "diabetes" centrado caía justo sobre la meseta del 97 % de
            # LightGBM y desaparecía. Fuera, la tira funciona como cabecera de zonas y no puede
            # colisionar con ninguna curva, sea cual sea la forma de la respuesta.
            if _cur_code == "LBXGH":
                for (_a, _b), _txt, _al in zip([(_clo, 5.7), (5.7, 6.5), (6.5, _chi)],
                                               S("lp_ada_bands"), (0.03, 0.065, 0.105)):
                    _fig.add_vrect(x0=_a, x1=_b, fillcolor=hex_to_rgba(t["text_secondary"], _al),
                                   line_width=0, layer="below")
                    _fig.add_annotation(x=(_a + _b) / 2, yref="paper", y=1, yanchor="bottom", yshift=4,
                                        text=_txt, showarrow=False,
                                        font=dict(family=PLOTLY_FONT, size=10.5, color=t["text_muted"]))
                # Las DOS fronteras ADA, no solo la de 6,5: si una linde se dibuja y la otra no,
                # el ojo lee la banda intermedia como si empezara en el borde del eje.
                for _bx in (5.7, 6.5):
                    _fig.add_vline(x=_bx, layer="below",
                                   line=dict(color=hex_to_rgba(t["text_secondary"], 0.32),
                                             width=1, dash="dot"))
            # Umbral de decisión, rotulado en el margen derecho (fuera del área de datos, donde
            # no puede chocar con las curvas). Sin rótulo, la línea de puntos era un adorno.
            # Dos detalles que costaron una captura cada uno:
            #  · el desplazamiento va en PÍXELES (xshift), no en fracción de papel — con x=1,01
            #    el rótulo se salía del lienzo y se recortaba a "umbra";
            #  · el margen derecho lo comparte con las cifras de punta, así que si una punta cae
            #    sobre el 50 % los dos se imprimen encima (pasa con Edad, que acaba en 48 %). Se
            #    resuelven antes las puntas que se van a rotular y se aparta el umbral al lado
            #    que queda libre.
            _dos_puntas = abs(float(_cl[-1]) - float(_cs[-1])) >= 0.06
            _puntas = [(_cl, SERIES["lightgbm"])] + ([(_cs, SERIES["svm_rbf"])] if _dos_puntas else [])
            _cerca = min((float(_s[-1]) for _s, _ in _puntas), key=lambda v: abs(v - 0.5))
            _thr_dy = 0 if abs(_cerca - 0.5) > 0.055 else (14 if _cerca < 0.5 else -14)
            _fig.add_hline(y=0.5, line=dict(color=hex_to_rgba(t["text_secondary"], 0.45), width=1, dash="dash"))
            _fig.add_annotation(xref="paper", x=1, xshift=9, y=0.5, yshift=_thr_dy,
                                xanchor="left", yanchor="middle",
                                text=S("lp_curve_thr"), showarrow=False,
                                font=dict(family=PLOTLY_FONT, size=10.5, color=t["text_muted"]))
            # SIN mancha bajo la curva. Se probó al 12 % y fue un error de bulto: LightGBM se
            # satura en ~97 % desde HbA1c 7, así que el relleno era una losa gris que cubría
            # cuatro quintos del panel, se tragaba las bandas ADA y dejaba el SVM flotando sobre
            # barro. Aquí el volumen lo da un HALO —la misma línea repetida debajo, ancha y casi
            # transparente— que sigue el contorno de la escalera en vez de rellenar el hueco:
            # cuesta la misma tinta en los tramos planos que en los peldaños y despega las
            # curvas del fondo sin tapar nada. Los dos halos van primero para que ninguna línea
            # nítida quede velada por el halo de la otra.
            _capas = [("LightGBM", _cl, SERIES["lightgbm"], 9, 0.10),
                      ("SVM-RBF", _cs, SERIES["svm_rbf"], 7, 0.07)]
            for _nom, _serie, _col, _w, _a in _capas:
                _fig.add_trace(go.Scatter(x=_vals, y=_serie, mode="lines",
                                          line=dict(color=hex_to_rgba(_col, _a), width=_w, shape="hv"),
                                          showlegend=False, hoverinfo="skip"))
            for _nom, _serie, _col, _w, _a in _capas:
                _fig.add_trace(go.Scatter(x=_vals, y=_serie, mode="lines", name=_nom,
                                          line=dict(color=_col, width=2, shape="hv"),
                                          hovertemplate="%{y:.1%}<extra>" + _nom + "</extra>"))
            # Punta de cada curva rotulada en el margen: el tramo derecho suele ser una meseta
            # larga y muda, y dos cifras ahí lo convierten en información y equilibran la
            # composición. Si las dos puntas convergen (_dos_puntas) se rotula solo LightGBM:
            # apiladas no se leen.
            for _serie, _col in _puntas:
                _yv = float(_serie[-1])
                _fig.add_trace(go.Scatter(x=[_vals[-1]], y=[_yv], mode="markers",
                                          marker=dict(size=8, color=_col,
                                                      line=dict(color=t["surface"], width=2)),
                                          cliponaxis=False, showlegend=False, hoverinfo="skip"))
                _fig.add_annotation(x=_vals[-1], y=_yv, xanchor="left", xshift=10, yanchor="middle",
                                    text=pct(_yv, 0), showarrow=False,
                                    font=dict(family=PLOTLY_MONO, size=11.5, color=t["text_secondary"]))
            # "Estás aquí". Único punto CÁLIDO de la figura, y a propósito: con dos series
            # neutras (tinta y gris) todo lo demás es acromático, así que el ámbar de marca no
            # compite con ninguna y da a la gráfica el foco que le faltaba. No es una tercera
            # serie: es cromo de interfaz —el eco de la posición del slider—, el mismo papel que
            # ya tiene en sliders y anillos de foco. El punto en sí conserva el color de
            # LightGBM, que es de quien es el valor; lo cálido es el halo y la plomada.
            _fig.add_trace(go.Scatter(x=[_cur_x, _cur_x], y=[0, risk], mode="lines",
                                      line=dict(color=hex_to_rgba(C_PRIMARY, 0.50), width=1.5, dash="dot"),
                                      showlegend=False, hoverinfo="skip"))
            # Halo suave + ANILLO abierto. Con dos discos rellenos superpuestos el foco salía
            # como una moneda marrón —el ámbar al 28 % sobre carbón es barro—; el anillo lo
            # convierte en una diana nítida y el halo solo aporta la irradiación.
            _fig.add_trace(go.Scatter(x=[_cur_x], y=[risk], mode="markers",
                                      marker=dict(size=28, color=hex_to_rgba(P_AMBAR, 0.10),
                                                  line=dict(width=0)),
                                      showlegend=False, hoverinfo="skip"))
            # En los símbolos "-open" el trazo lo pinta marker.COLOR, no marker.line.color: con
            # el color puesto en line el anillo salía magenta (el primer color del ciclo por
            # defecto de Plotly). line aquí solo fija el grosor.
            _fig.add_trace(go.Scatter(x=[_cur_x], y=[risk], mode="markers",
                                      marker=dict(size=19, symbol="circle-open", color=C_PRIMARY,
                                                  line=dict(width=1.6)),
                                      showlegend=False, hoverinfo="skip"))
            # Anillo de 2 px en color de SUPERFICIE: el mecanismo del sistema para que una marca
            # se despegue de lo que cruza. Nunca un borde de color, que añadiría tinta de dato.
            _fig.add_trace(go.Scatter(x=[_cur_x], y=[risk], mode="markers",
                                      marker=dict(size=12, color=SERIES["lightgbm"],
                                                  line=dict(color=t["surface"], width=2)),
                                      showlegend=False, hoverinfo="skip"))
            # El valor actual, anclado al eje y con filete ámbar para que se lea como parte del
            # mismo gesto que la plomada. Se ancla por el lado que no se sale del lienzo.
            _frac = (_cur_x - _clo) / (_chi - _clo) if _chi > _clo else 0.5
            _fig.add_annotation(x=_cur_x, y=0, yanchor="bottom", yshift=8,
                                xanchor="left" if _frac < 0.07 else ("right" if _frac > 0.93 else "center"),
                                text=f"{nf(_cur_x, _dec)} {q_unit(_cur_code)}", showarrow=False,
                                font=dict(family=PLOTLY_MONO, size=11.5, color=t["text"]),
                                bgcolor=t["surface"], bordercolor=C_PRIMARY, borderwidth=1, borderpad=4)
            # Hover unificado + retícula: la pregunta de esta gráfica es "¿qué dicen LOS DOS en
            # este valor?", y con un tooltip por traza había que apuntar a cada línea por
            # separado. El margen derecho da sitio al umbral y a las dos cifras de punta.
            # La leyenda baja al pie para dejar la franja superior entera a los rótulos de zona.
            # El eje X se fija al rango exacto de la variable (sin el relleno automático de
            # Plotly) para que las bandas lleguen de borde a borde en vez de dejar una cuña sin
            # teñir a la derecha; los puntos de punta llevan cliponaxis=False y por eso no se
            # cortan por la mitad al caer justo sobre el límite.
            plotly_layout(_fig, height=360, showlegend=True, hovermode="x unified",
                          margin=dict(l=48, r=76, t=30, b=74),
                          legend=dict(orientation="h", yanchor="top", y=-0.26, xanchor="left", x=0,
                                      font=dict(family=PLOTLY_FONT, size=12.5, color=t["text"]),
                                      bgcolor="rgba(0,0,0,0)"),
                          xaxis=dict(title=dict(text=f"{_cur_label} ({q_unit(_cur_code)})",
                                                font=dict(size=13), standoff=10),
                                     range=[_clo, _chi],
                                     showgrid=False, fixedrange=True, hoverformat=f".{_dec}f",
                                     showspikes=True, spikemode="across", spikesnap="data",
                                     spikethickness=1, spikedash="dot",
                                     spikecolor=hex_to_rgba(t["text_secondary"], 0.40)),
                          yaxis=dict(title=dict(text=S("lp_curve_yaxis"), font=dict(size=13), standoff=8),
                                     range=[0, 1], showgrid=True, gridcolor=GRID, tickformat=".0%",
                                     tickvals=[0, 0.25, 0.5, 0.75, 1], fixedrange=True))
            # Panel: el área de datos se separa un paso de la tarjeta. Es lo que faltaba para
            # que las variables SIN bandas ADA (LDL, edad…) no parecieran una línea suelta en
            # mitad de la nada — con el eje Y clavado en 0-100 % (que se mantiene, porque
            # reescalarlo exageraría respuestas planas) media gráfica queda vacía por
            # definición, y ese vacío tiene que leerse como lienzo, no como falta de algo.
            # Va aquí y no en plotly_layout porque esa función ya fija plot_bgcolor.
            _fig.update_layout(plot_bgcolor=hex_to_rgba(t["text_secondary"], 0.05))
            st.plotly_chart(_fig, width="stretch", config={"displayModeBar": False})

            # Peldaños reales de la configuración actual, contados sobre la propia curva.
            _saltos = [(_vals[i], _cl[i] - _cl[i - 1]) for i in range(1, len(_vals))
                       if abs(_cl[i] - _cl[i - 1]) > 1e-9]
            _grandes = sorted(_saltos, key=lambda s: -abs(s[1]))[:3]
            _n_dist = len(set(np.round(_cl, 6)))
            # El valor va con el formato del propio slider (%d / %.1f) y luego por nf() para que
            # su decimal siga al idioma; el salto, por pct() con signo explícito.
            _txt_saltos = " · ".join(
                f"{nf(v, 1 if _cv['step'] < 1 else 0)} ({'+' if d >= 0 else '−'}{pct(abs(d))})"
                for v, d in _grandes) or S("lp_curve_none")
            st.markdown(
                f'<div class="section-sub">'
                f'{S("lp_curve_note").format(n=_n_dist, total=len(_vals), saltos=_txt_saltos)}</div>',
                unsafe_allow_html=True)

    # Nota de lectura: sin esto, dos de las ocho variables se interpretan al revés. Como el
    # target es "ya diagnosticado", el modelo aprende el efecto del TRATAMIENTO además del de
    # la enfermedad, y eso invierte el signo del LDL (los diagnosticados van estatinizados) y
    # da forma de U a la glucosa (hipoglucemias de pacientes tratados). Ambos efectos están
    # medidos sobre los propios .onnx, no supuestos — ver INFORME_AUDITORIA_DASHBOARD.md §1.2.
    st.markdown(f"""
    <div class="clinical-note" style="margin-top:16px;">
    {S("lp_read_note")}
    </div>
    """, unsafe_allow_html=True)
