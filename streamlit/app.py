"""
QML DataOps TFM Dashboard — App completa (7 paginas)
Universidad Europea de Valencia · TFM Juan Albornoz

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
import threading
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote_plus

import numpy as np
# La app no usa pandas, pero Streamlit lo importa por su cuenta y muy adentro. Hecho ahi, en
# el hilo ScriptRunner, el modulo Cython pandas._libs.tslib revienta con "SystemError:
# __pyx_defaults returned a result with an exception set". Importarlo aqui arriba lo carga
# antes, y despues ya es un acierto en sys.modules.
# El camino con el que se diagnostico —option_menu -> create_instance -> is_dataframe_like—
# ya no existe (el menu es de botones nativos desde que el arbol de secciones se fusiono con
# el), pero la linea se queda: lo que revienta es la importacion perezosa DESDE ese hilo, no
# quien la dispara, y precargarla cuesta lo que cuesta importar pandas una vez.
import pandas  # noqa: F401
import plotly.graph_objects as go
# requests NO se anade a requirements.txt a proposito: es dependencia directa de Streamlit
# (que lo usa para su telemetria y para st.connection), asi que ya viaja en el entorno y
# fijarlo aparte solo abriria la puerta a un conflicto de versiones con el propio Streamlit.
# Lo usa unicamente el contador de visitas, contra la API de Gists de GitHub.
import requests
import streamlit as st
import streamlit.components.v1 as components   # solo para fijar <html lang>; ver más abajo
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import i18n                                   # catálogo de textos ES/EN/DE/FR/IT (módulo local)

ASSETS_DIR = Path(__file__).parent / "assets"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
MODELS_DIR = Path(__file__).parent / "models"

# Salida del buscador de la barra lateral (ver bloque BUSCADOR) hacia FUERA del panel.
# Lo que se consulta desde un panel de TFM son conceptos con literatura detrás —"ZZFeatureMap",
# "hemoglobina glicada"—, así que la salida no va a un buscador general: ahí la primera página
# son blogs, cursos y tiendas, y lo que hace falta es algo citable. Los tres destinos cubren las
# dos mitades del trabajo sin obligar a elegir de antemano, que es justo lo que la caja promete:
# Scholar como red académica general, arXiv para el lado cuántico/ML —donde el preprint ES la
# fuente y se adelanta años a la revista— y PubMed para el lado clínico de NHANES. Van de más
# general a más específico, que es también el orden en que se abandonan si el primero no da nada.
# La consulta se interpola con quote_plus: codifica el espacio como "+", válido en la query
# string de los tres, y evita que unas comillas o un "&" del usuario partan la URL.
SEARCH_SOURCES = (
    ("Scholar", "https://scholar.google.com/scholar?q={q}"),
    ("arXiv",   "https://arxiv.org/search/?searchtype=all&query={q}"),
    ("PubMed",  "https://pubmed.ncbi.nlm.nih.gov/?term={q}"),
)

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
# Este número manda SIEMPRE, a cualquier ancho de ventana y a cualquier zoom: el CSS lo usa
# como número exacto de columnas y encoge las pastillas en lugar de reagrupar filas.
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
        # El alto de _tech_alto() ya NO va en px sino multiplicado por --tech-u, la unidad de
        # escala que define .tech-strip y que encoge con el ancho real de la tira: con la
        # rejilla fija en TECH_POR_FILA columnas, un px absoluto aquí dejaría los logos del
        # tamaño de escritorio dentro de pastillas ya estrechadas y se saldrían de la caja.
        # Va el calc() entero en el style —y no un `--th` que el CSS multiplicaría— porque una
        # propiedad estándar sobrevive seguro al saneado del HTML; una custom property en un
        # atributo style depende de qué haga con ella el sanitizador de turno.
        piezas.append(
            f'<div class="tech-chip" title="{nombre}">'
            f'<img src="data:image/{mime};base64,{_b64_image(str(ruta))}" alt="{nombre}" '
            f'style="height:calc({_tech_alto(str(ruta)):.1f} * var(--tech-u));"></div>')
    if piezas:
        # El envoltorio no es decorativo: es el elemento que se MIDE (container-type) para que
        # las cqw de dentro se refieran al hueco real de la tira y no al viewport.
        st.markdown(f'<div class="tech-strip-wrap"><div class="tech-strip">{"".join(piezas)}</div></div>',
                    unsafe_allow_html=True)

# initial_sidebar_state="auto": expandida en escritorio, COLAPSADA en móvil. Con "expanded" se quedaba
# abierta también en el teléfono, comiéndose 270 de los ~390 px de pantalla.
st.set_page_config(page_title="QML DataOps", page_icon="◆", layout="wide", initial_sidebar_state="auto")

# ═════════════════════════════════════════════════════════════════════════
# SISTEMA DE DISEÑO
# ═════════════════════════════════════════════════════════════════════════
# PALETA BASE — REGISTRO CLÍNICO (cuatro anclas del autor):
#   #1565C0  azul clínico (primario)     #00ACC1  cian (acento)
#   #F4F8FB  papel (fondo)               #263238  tinta (texto)
#
# Esta paleta se especifica al revés que las anteriores, y eso cambia el método: las
# tres primeras partían de un LIENZO OSCURO y derivaban el tema claro; esta declara
# el papel y la tinta, o sea el tema CLARO, que es además el registro en el que se lee
# un panel clínico. Medidas:
#   · azul   L 0,513  C 0,160  h 256°     · cian   L 0,682  C 0,118  h 210°
#   · papel  L 0,977  C 0,006  h 240°     · tinta  L 0,309  C 0,019  h 230°
#
# Tres propiedades deciden todo el reparto:
#   · HAY DOS CROMÁTICOS, y no uno. La paleta anterior tenía un solo cálido, así que
#     el acento de marca y el modelo cuántico acababan siendo el MISMO color (ver la
#     nota de C_QUANTUM). Aquí no hace falta: el azul se queda con el cromo de
#     interfaz y el cian con el componente cuántico, que es donde está la tesis. Se
#     separan 46° de tono y ΔL 0,17, o sea que se distinguen por las dos vías.
#   · EL AZUL SÍ PUEDE LLEVAR TEXTO EN CLARO, que es la diferencia práctica más
#     grande con el ámbar. #1565C0 da 5,75:1 sobre la tarjeta, 5,38 sobre el lienzo y
#     4,61 sobre la barra lateral: pasa WCAG AA en los tres fondos que reciben tinta,
#     así que el acento del autor entra SIN TOCAR. El ámbar daba 1,75:1 y había que
#     oscurecerlo a mano hasta L 0,52 antes de dejarle un rótulo.
#   · EL CIAN NO. 2,56:1 sobre el lienzo: es un color concebido para fondo oscuro, y
#     hereda exactamente el papel que tenía el ámbar en claro — vive en rellenos,
#     bordes y tintes, y cuando tiene que llevar texto usa un paso oscurecido.
#
# Lo que la paleta NO trae es el tema oscuro: no hay ni un oscuro entre las cuatro
# anclas. Se deriva, y se deriva CONSERVANDO LA ESCALERA de la paleta anterior —los
# mismos peldaños de luminosidad L 0,122 · 0,165 · 0,212 · 0,307— sobre el eje de la
# tinta (h 230°) en vez del violáceo de antes. Los saltos entre planos siguen siendo
# 1,05 / 1,15 / 1,33:1, así que todo el aspecto "elevado" del tema oscuro sobrevive
# intacto al cambio de color: cambia el tono, no la arquitectura.
#
# El papel #F4F8FB está altísimo (L 0,977, a 1,05:1 del blanco), y eso hay que tenerlo
# presente en todo el tema claro: la tarjeta blanca sobre este lienzo CASI NO SE DESPEGA
# POR COLOR. La elevación en claro la sostienen la sombra y el filete, no el escalón
# tonal — ver SHADOW, que por esto se rehizo a tres capas.
#
# Reparto en consecuencia — las tres familias tienen trabajos DISTINTOS y no se cruzan:
#   · NEUTROS (papel · tinta + la escalera derivada) → ESTRUCTURA: planos de elevación,
#     bordes y todos los niveles de texto. Van sobre el eje de la tinta (h 230°), no en
#     gris: comparten tono con las superficies, así que el texto secundario se hunde en
#     el plano en vez de ensuciarse.
#   · AZUL → CROMO DE INTERFAZ: acento de marca, foco, navegación activa, filetes,
#     halos y la rampa de magnitud. Es lo que la app usa para decir «esto responde».
#   · CIAN → EL COMPONENTE CUÁNTICO: la serie del QSVM, la esfera de Bloch y el
#     ZZFeatureMap. El color va donde está la tesis, y al ser el único de su tono en
#     toda la app, señala sin competir con el cromo.
#   · SERIES (los 3 modelos) → tinta · azul-gris medio · cian. Los dos clásicos en la
#     familia neutra y el cuántico en el acento. El cian separa al cuántico también por
#     TONO, que es lo que sobrevive al daltonismo.
#   · RAMP (magnitud) → rampa fría de un solo tono (h 265°), cinco pasos de
#     luminosidad monótona sobre el eje de la escalera. Va en frío y NO en ámbar a
#     propósito: si la magnitud fuera dorada, el ojo no podría distinguir "esta
#     celda vale mucho" de "esto está seleccionado", que es lo que dice el ámbar en
#     el resto de la aplicación. Una escala de magnitud tiene que ser monocroma
#     para que el orden se lea sin leyenda, y aquí además tiene que ser MUDA.
#
# Resultados del validador con los valores de abajo:
#   SERIES oscuro → ΔE normal 29,4 / 33,9 / 45,9 · peor CVD 28,6 · ≥3:1 los tres
#   SERIES claro  → ΔE normal 22,2 / 54,9 / 46,7 · peor CVD 21,7 · ≥3:1 los tres
#   RAMP claro / RAMP oscuro → monotonía, ΔL constante, tono único: TODAS PASS
#   TINTA, ACENTO y ESTADO → los tres niveles de texto, los dos acentos y los tres
#   colores de estado pasan 4,5:1 contra los CUATRO fondos de su tema (tarjeta,
#   lienzo, barra lateral y superficie alterna), no solo contra el más favorable.
#   Se subió de tres fondos a cuatro respecto de la revisión anterior porque en esta
#   paleta la superficie alterna es el grafito, bastante más claro que la tarjeta, y
#   ahí se colaban dos colores que pasaban en los otros tres. Ver C_PRIMARY, que es
#   donde está la advertencia sobre medir contra el fondo equivocado.
#   Único FAIL, deliberado y heredado: el suelo de croma sobre los slots neutros de
#   SERIES. Ese suelo existe para que un tono no "lea gris" y deje de hacer trabajo
#   de identidad; aquí el neutro ES neutro a propósito y con la pizarra media y el
#   ámbar al lado no hay ambigüedad posible. Si se retoca cualquier hex, hay que
#   volver a pasar el validador.
# ─────────────────────────────────────────────────────────────────────────

# La app ABRE EN OSCURO. Es solo el valor de partida, no una restricción: la cápsula-interruptor
# del pie de la sidebar (st.button key="theme_toggle") sigue alternando en los dos sentidos y el
# tema claro se conserva entero —T(), las rampas, las sombras y el tratamiento de las figuras
# mantienen sus dos ramas, aquí no se ha quitado ninguna—.
# El cambio se hace AQUÍ y no en el base de .streamlit/config.toml a propósito: ese ajuste es del
# servidor, se aplica de una vez para todas las sesiones y no lo puede revertir el interruptor, así
# que poner base="dark" ahí teñiría de oscuro los widgets nativos también en tema claro y dejaría
# la opción clara a medias. El precio de hacerlo en session_state es un parpadeo claro en la
# primera pintura, antes de que entre el <style> de abajo.
#
# El tema TAMBIÉN sobrevive a la recarga, por la vía de ?theme= y por el mismo motivo que el
# idioma: F5 abre una sesión nueva y devolvía a oscuro a quien hubiera elegido claro. Lo que
# NO se hace es preguntarle al navegador por prefers-color-scheme — igual que con el idioma,
# la única forma de cambiar el tema es su interruptor, y adivinarlo en la primera visita
# contradiría el "abre en oscuro" que se acaba de razonar. Se valida contra los dos valores
# posibles: un ?theme=xx desconocido dejaría a T() eligiendo la rama clara sin querer.
if "theme" not in st.session_state:
    _theme_url = st.query_params.get("theme")
    st.session_state.theme = _theme_url if _theme_url in ("dark", "light") else "dark"
if "sidebar_narrow" not in st.session_state:
    st.session_state.sidebar_narrow = False
# El idioma SOBREVIVE A LA RECARGA, y por eso no puede vivir solo en session_state: F5 abre
# una sesión nueva de Streamlit y el estado nace vacío, así que la app volvía al español
# aunque la bandera elegida fuera la italiana. La memoria es la query string —?lang=it—, que
# el navegador conserva al recargar y que viaja además dentro del enlace que se comparte.
#
# Se VALIDA contra i18n.LANGS antes de aceptarla. El parámetro lo escribe el usuario en la
# barra de direcciones con la misma facilidad que la app, y un ?lang=xx sin catálogo tumbaría
# S() con un KeyError en la primera clave que pidiera; lo que no se reconoce cae al idioma
# por defecto en vez de propagarse.
#
# NO se negocia el idioma con el navegador (ni Accept-Language ni navigator.language): la
# única forma de cambiarlo es la bandera, y adivinarlo en la primera visita rompería justo
# eso —abriría en italiano a quien tenga Firefox en italiano sin haber elegido nada—.
if "lang" not in st.session_state:
    _lang_url = st.query_params.get("lang")
    st.session_state.lang = _lang_url if _lang_url in i18n.LANGS else i18n.DEFAULT_LANG
# El desplegable de banderas se abre y se cierra con un CLIC, y su estado vive aquí y no en
# la hoja de estilos: ver el bloque de _ABIERTO más abajo. La clave es NUESTRA y no el id de
# un widget, así que sobrevive al st.rerun() del selector — el mismo motivo que _POS_TAB.
if "lang_open" not in st.session_state:
    st.session_state.lang_open = False

_is_dark = st.session_state.theme == "dark"
LANG = st.session_state.lang
MENU_ABIERTO = st.session_state.lang_open

# La URL se pone al día con el idioma activo, y se hace AQUÍ y no en el callback de la
# bandera: al elegir idioma el botón termina en st.rerun(), y lo que se escriba justo antes
# de reiniciar el script depende de que el mensaje llegue al navegador antes del corte. En
# cambio esta línea corre al principio de la pasada siguiente, cuando el idioma nuevo ya está
# en session_state y no hay nada que pueda interrumpir. Un único sitio, además, en vez de uno
# por cada camino que cambie el idioma.
#
# La condición evita estampar ?lang=es en la primera visita de quien no ha tocado nada: hasta
# que se elige bandera la URL se queda limpia. Una vez que el parámetro existe se mantiene
# siempre —volver al español lo deja en ?lang=es y no lo borra—, porque una URL sin parámetro
# significa "lo que diga el idioma por defecto" y eso desharía la elección en la recarga.
if LANG != i18n.DEFAULT_LANG or "lang" in st.query_params:
    if st.query_params.get("lang") != LANG:
        st.query_params["lang"] = LANG

# Lo mismo para el tema, y aquí el razonamiento del "aquí y no en el callback" es literal: el
# interruptor del pie de la sidebar termina en st.rerun(), y esta línea corre al principio de
# la pasada siguiente, con el tema nuevo ya en session_state. La condición deja la URL limpia
# mientras nadie toque el interruptor; en cuanto se toca, el parámetro se queda —volver a
# oscuro lo deja en ?theme=dark y no lo borra— porque una URL sin parámetro significa "el tema
# de partida" y eso desharía la elección en la recarga.
if st.session_state.theme != "dark" or "theme" in st.query_params:
    if st.query_params.get("theme") != st.session_state.theme:
        st.query_params["theme"] = st.session_state.theme

def S(key):
    """Texto de la clave en el idioma activo, con caída al español si falta.

    La caída no es una red de seguridad genérica: es lo que permite traducir la app
    PÁGINA A PÁGINA. Mientras STR["en"] no tenga las claves de Gobernanza o del
    Predictor, esas páginas se siguen pintando en español en vez de tumbar la app
    con un KeyError, y el idioma va llegando a cada una según se traduce.
    """
    catalogo = i18n.STR[LANG]
    return catalogo[key] if key in catalogo else i18n.STR["es"][key]

# Clave donde tabs_i18n() guarda la posición del tab abierto. El sufijo la mantiene lejos
# del espacio de nombres de las claves de widget: si alguna vez coincidieran, Streamlit la
# trataría como estado de widget y la podaría igual que a la otra, que es justo el fallo
# del que viene todo esto.
_POS_TAB = "{}__pos_tab"

# Páginas que estrenan un grupo de tabs. La lista NO decide qué se dibuja —eso sigue en el
# cuerpo de cada página—: solo la usa el saneo de ?tab= tras el enrutado, para que el
# parámetro no se quede colgando en la URL de una página que no tiene pestañas. Si alguna
# otra página estrena tabs, hay que añadirla aquí.
_PAGINAS_CON_TABS = ("governance", "shap", "circuit")

# Clave (nuestra, no de widget) que marca que ?tab= ya se ha consumido en esta sesión.
_TAB_URL_LEIDA = "__tab_url_leida"

def _recuerda_tab(catalogo, key):
    """Apunta la posición del tab que se acaba de abrir. Callback de tabs_i18n().

    Va aquí y no en el cuerpo de la página porque el cambio de tab no pasa por el script:
    el callback es el único momento en que Streamlit garantiza que session_state[key] ya
    tiene el rótulo NUEVO. Se guarda el índice y no el rótulo, porque el rótulo deja de
    existir en cuanto se cambia de idioma.
    """
    rotulos = S(catalogo)
    abierto = st.session_state.get(key)
    if abierto in rotulos:
        st.session_state[_POS_TAB.format(key)] = rotulos.index(abierto)

def tabs_i18n(catalogo, key):
    """st.tabs que NO se rebobina al cambiar de idioma NI al recargar la página.

    El rótulo de un tab es su nombre para Streamlit, y al cambiar de idioma cambian los
    tres a la vez, así que el widget se da por nuevo y vuelve al primero: quien estaba
    leyendo "Inventario de frameworks" y pulsaba la bandera aparecía de golpe en
    "Calidad del dato". Lo que se conserva aquí es la POSICIÓN, que es lo único que
    significa lo mismo en los dos idiomas.

    NO SIRVE guardarla en el estado del propio widget, que fue el primer intento. El
    botón de la bandera vive ANTES del reparto de páginas y termina en st.rerun(), que
    aborta la pasada: la página nunca llega a dibujarse, el widget no se registra, y
    Streamlit borra su estado por «stale» (on_script_finished → _remove_stale_widgets).
    Y no vale confiar en que el rerun lo libre de la poda: en exec_code.py la excepción
    de rerun pone premature_stop=False justamente para que la limpieza SÍ corra. Cuando
    esta función volvía a mirar el estado del widget, ya no había nada que traducir.

    Por eso la posición vive en una clave PROPIA de session_state. Esa poda solo alcanza
    a las entradas cuya clave es un id de widget (is_element_id), de modo que una clave
    nuestra sobrevive al st.rerun() de la bandera. La escribe el callback de on_change,
    que es además la razón de que on_change no pueda quitarse: la documentación de
    st.tabs dice que el tab activo solo llega a session_state con "rerun" o con un
    callable, y con el "ignore" de serie el servidor ni siquiera sabe cuál está abierto.
    A cambio, cambiar de tab pasa a ser un rerun en vez de un gesto solo del navegador;
    sale barato porque el contenido de esta página ya viene de funciones cacheadas.
    """
    rotulos = S(catalogo)
    _clave_pos = _POS_TAB.format(key)
    # RECARGA (F5): session_state nace vacío, así que la posición se rescata de ?tab=, que es
    # lo único que sobrevive a una sesión nueva — mismo mecanismo que ?lang y ?page.
    #
    # Se consume UNA SOLA VEZ por sesión, en el primer grupo de tabs que se dibuje. El
    # parámetro no dice a qué grupo pertenece —es un número suelto—, y sin ese cerrojo la
    # posición se contagiaría de una página a otra: quien estuviera en el tab 1 de Gobernanza
    # y saltara a Análisis SHAP abriría ahí el segundo tab en vez del primero, porque ese
    # grupo también estrena su clave en ese momento. A partir de la primera lectura manda
    # session_state, que sí guarda una posición por grupo.
    if not st.session_state.get(_TAB_URL_LEIDA):
        st.session_state[_TAB_URL_LEIDA] = True
        _tab_url = st.query_params.get("tab")
        # isdigit() y no int() a secas: el parámetro lo escribe cualquiera en la barra de
        # direcciones, y "?tab=-1" o "?tab=hola" no pueden tumbar la página (el primero
        # además pasaría el rango por la puerta de atrás indexando desde el final).
        if (_clave_pos not in st.session_state and _tab_url is not None
                and _tab_url.isdigit() and int(_tab_url) < len(rotulos)):
            st.session_state[_clave_pos] = int(_tab_url)
    pos = st.session_state.get(_clave_pos, 0)
    if not isinstance(pos, int) or not 0 <= pos < len(rotulos):
        pos = 0
    # La URL se pone al día con el tab abierto. Va aquí, en la pasada SIGUIENTE al clic (el
    # on_change ya ha guardado la posición y ha relanzado el script), por el mismo motivo que
    # ?lang: escribir justo antes de un rerun depende de que el mensaje llegue a tiempo.
    # Y como con ?lang y ?page, hasta que no se toca una pestaña la URL se queda limpia.
    if pos != 0 or "tab" in st.query_params:
        if st.query_params.get("tab") != str(pos):
            st.query_params["tab"] = str(pos)
    # Red de seguridad: si el estado del widget sobreviviera con un rótulo del OTRO
    # idioma, st.tabs recibiría un valor que no está entre sus opciones. Se descarta y
    # manda `default`, que es la vía documentada para fijar el tab inicial.
    if st.session_state.get(key) is not None and st.session_state[key] not in rotulos:
        del st.session_state[key]
    return st.tabs(rotulos, key=key, default=rotulos[pos],
                   on_change=_recuerda_tab, args=(catalogo, key))

def _navegar(pagina, tab=None, seccion=None):
    """Lleva a una página, a su pestaña y —si se pide— a una sección concreta.

    Es el ÚNICO camino de navegación programada de la app: lo usan el buscador y el árbol de
    secciones de la barra lateral. Estaba escrito dentro del primero, y sacarlo aquí no es
    ordenar por ordenar: los dos tienen que abrir la pestaña de destino exactamente igual, y
    de las tres líneas que hacen eso, dos son contraintuitivas y se copiarían mal.

    Escribir `page` basta para que el menú lo obedezca, y eso es nuevo: mientras el menú fue un
    option_menu había que empujarle además el índice a mano (`menu_force_index` -> su
    `manual_select`), porque vivía en un iframe con estado propio y no leía session_state. Al
    pasar a botones nativos, la página activa es la que dice session_state y no hay segunda
    copia del estado que sincronizar.

    `tab` es (grupo, posición) y se abre por la misma vía que usa tabs_i18n para sobrevivir a
    un cambio de idioma: se guarda la posición en su clave propia y se BORRA el estado del
    widget, de modo que en la pasada siguiente manda el `default=rotulos[pos]` de tabs_i18n.
    Escribir la posición sin borrar el estado no basta —Streamlit ignora `default` cuando el
    widget ya tiene valor— y el salto se quedaría en la pestaña que estuviera abierta.

    `seccion` es el RÓTULO del título al que bajar, no su clave: quien lo consume es el
    navegador, que lo busca entre los .section-title del documento. Viaja en session_state
    porque el destino no existe todavía —esta función corre como callback, o sea ANTES de que
    la página de destino se dibuje—, y lo recoge el bloque de navegación en la pasada
    siguiente, ya con el contenido puesto.

    Y de paso despliega la rama de la página de destino, venga el salto de donde venga: llegar a
    una página con su índice plegado obligaría a un clic más para ver dónde has caído. Vale
    también para el buscador, que es la otra puerta de entrada.

    `nav_cerrar` es el aviso para el teléfono, donde la barra no es una columna sino un panel
    SUPERPUESTO: sin él, navegar dejaba el panel abierto encima de la página recién abierta y
    había que cerrarlo a mano para ver a dónde habías ido —y en un salto a una sección, el
    recorrido hasta el título ocurría detrás del panel—. Se pone siempre y lo filtra quien puede
    medir la ventana, que es el navegador; en escritorio se ignora.
    """
    st.session_state.page = pagina
    st.session_state.nav_open = pagina
    st.session_state.nav_cerrar = True
    if tab is not None:
        grupo, pos = tab
        st.session_state[_POS_TAB.format(grupo)] = pos
        st.session_state.pop(grupo, None)
    if seccion:
        st.session_state.nav_scroll = seccion


def _navegar_pagina(pagina):
    """Lo que hace la fila de PÁGINA al pulsarla: ir allí y desplegar su rama; si ya estaba
    desplegada, plegarla.

    Es una envoltura de _navegar y no un parámetro suyo porque el plegado es exclusivo de este
    gesto: pulsar una sección, buscar o seguir un enlace interno siempre tienen que DEJAR la
    rama abierta —van a un destino de dentro—, y solo el clic repetido sobre la raíz significa
    «ya he visto esto, ciérralo».

    Plegar no deja la barra en un estado raro: la fila de la página sigue marcada como activa
    —eso lo dice `page`, no `nav_open`— y lo único que desaparece es su lista de secciones.
    """
    plegar = st.session_state.nav_open == pagina
    _navegar(pagina)
    if plegar:
        st.session_state.nav_open = None


def _flag_uri(lang):
    """SVG de bandera como data-URI en base64, listo para background-image.

    En base64 y no con el SVG en crudo: dentro de la hoja de estilos el marcado
    llevaría comillas, almohadillas de color y signos de mayor/menor que habría que
    escapar dos veces (una para el CSS, otra para el f-string). En base64 no hay
    nada que escapar y el resultado es idéntico.
    """
    return "data:image/svg+xml;base64," + base64.b64encode(
        i18n.FLAG_SVG[lang].encode("utf-8")).decode("ascii")

def _sel_lang(sufijo="", langs=None):
    """Selector CSS que abarca las banderas indicadas (todas, por defecto).

    Las reglas de las banderas eran tres listas escritas a mano con "es" y "en", y el
    idioma inactivo era un `"en" if LANG == "es" else "es"`. Con dos banderas colaba;
    al entrar la tercera dejó de haber UN "otro" idioma, y las listas a mano habrían
    exigido tocar cuatro sitios con la garantía de que uno se queda atrás. Generado
    desde i18n.LANGS, añadir un idioma es añadir su catálogo y su bandera y nada más.
    """
    return ", ".join(f".st-key-lang_{l}{sufijo}" for l in (langs or i18n.LANGS))

LANGS_OTROS = [l for l in i18n.LANGS if l != LANG]

# Estados en los que el desplegable se considera ABIERTO. Son dos y cubren cosas distintas:
#   · el ESTADO DE SESIÓN (MENU_ABIERTO) — el caso normal: se despliega con un clic en la
#     bandera activa y se queda quieto hasta que se elige idioma (o se vuelve a pulsar el
#     disparador), pase lo que pase con el cursor entretanto. El selector es el contenedor a
#     secas, sin condición: la condición ya la ha resuelto Python al decidir si esta línea
#     entra o no en la hoja de estilos.
#   · :has(button:focus-visible) — el teclado. Las banderas del panel siguen siendo enfocables
#     con el tabulador aunque estén a opacidad 0 (por eso se ocultan así y no con visibility,
#     que las sacaría del orden de tabulación), de modo que tabular hasta una la muestra.
#     Va con :focus-visible y no con el :focus-within de antes porque el clic del ratón TAMBIÉN
#     deja el foco en el disparador: con :focus-within, el segundo clic —el de plegar— habría
#     cerrado el menú por estado y lo habría reabierto por foco en la misma pasada.
#
# Ya NO se abre con :hover, que es como funcionaba hasta ahora y era el fallo que esto repara.
# El panel dependía de que el cursor no se saliera ni un píxel de una columna del ancho de UNA
# bandera (_FLAG_ANCHO), y el navegador MUESTREA el puntero: al bajar deprisa hacia las últimas
# la muestra caía fuera de esa columna, el grupo perdía el :hover y el menú se cerraba en la
# cara de quien iba a elegir —solo bajando muy despacio se llegaba abajo—. Con el estado en
# sesión el recorrido del ratón deja de importar: se abre, se elige, se cierra.
_ABIERTO = ((".st-key-lang_switch",) if MENU_ABIERTO else ()) + (
    ".st-key-lang_switch:has(button:focus-visible)",)

def _sel_abierto(sufijo="", langs=None):
    """Selector de las banderas indicadas, pero solo con el desplegable abierto.

    Producto cartesiano de los tres estados por los idiomas: CSS no tiene variables de
    selector y hay que escribir las combinaciones una a una. Generarlas evita el bloque
    de doce selectores a mano que habría que rehacer al tocar un idioma.
    """
    return ", ".join(f"{estado} .st-key-lang_{l}{sufijo}"
                     for estado in _ABIERTO for l in (langs or i18n.LANGS))

# ── Geometría del desplegable de idioma ──
# La bandera ACTIVA hace de disparador y se queda sola en la esquina; las otras cuelgan
# debajo, ocultas hasta que el grupo recibe cursor o foco. Con cinco idiomas la tira
# horizontal medía 162 px y se comía la franja de cabecera entera —y el reloj se iba a
# 196 px del borde—; el desplegable la devuelve al ancho de una sola bandera.
#
# Todas las banderas guardan el MISMO aire (_MENU_HUECO), tanto entre el disparador y el
# panel como entre las del panel: un solo ritmo vertical en vez de dos. Ese aire salía caro
# mientras el menú se abría por :hover —cada hueco era un punto donde el cursor no estaba
# sobre ninguna bandera, y ahí se cerraba—, y había que taparlo una a una con un ::after que
# hacía de puente hacia la de abajo. Abriéndose por clic el hueco ya no significa nada: el
# panel no escucha al cursor. Los puentes se han quitado, y con ellos la franja de clic que
# se comían por debajo de cada bandera.
_FLAG_ANCHO, _FLAG_ALTO = 26, 18
_MENU_TOP, _MENU_HUECO, _MENU_BORDE = 14, 6, 22
_MENU_CARET = 12                # lo que ocupa la flecha a la izquierda del disparador

def _reparto(top, alto, hueco):
    """Coordenada `top` de cada bandera: la activa arriba, las demás en el panel.

    Devuelve un dict por idioma. El orden del panel es el de i18n.LANGS sin la activa,
    así que la lista no se reordena al cambiar de idioma — solo desaparece de ella la
    que ha subido a disparador.

    El paso es `alto + hueco`, el mismo que separa el disparador de la primera del panel:
    así las cinco quedan a intervalos iguales y la columna se lee como una sola serie.
    """
    primera = top + alto + hueco
    reparto = {LANG: top}
    reparto.update({l: primera + i * (alto + hueco) for i, l in enumerate(LANGS_OTROS)})
    return reparto

FLAG_TOP = _reparto(_MENU_TOP, _FLAG_ALTO, _MENU_HUECO)
# El reloj ya solo tiene que esquivar UNA bandera y su flecha, no cinco.
RELOJ_RIGHT = _MENU_BORDE + _FLAG_ANCHO + _MENU_CARET + 12

# Mismas cuentas con las medidas de móvil (ver la media query del final de la hoja).
_FLAG_ANCHO_M, _FLAG_ALTO_M = 23, 16
_MENU_TOP_M, _MENU_HUECO_M, _MENU_BORDE_M = 10, 5, 14
_MENU_CARET_M = 11
FLAG_TOP_M = _reparto(_MENU_TOP_M, _FLAG_ALTO_M, _MENU_HUECO_M)
RELOJ_RIGHT_M = _MENU_BORDE_M + _FLAG_ANCHO_M + _MENU_CARET_M + 10

def _css_banderas(tops, ancho, alto, borde, radio=3, con_imagen=True):
    """Sitio y piel de cada bandera: una regla por idioma.

    `con_imagen` existe porque el data-URI de la bandera de EE. UU. lleva las 50
    estrellas y pesa: la media query de móvil solo cambia medidas, así que repetir ahí
    los cinco data-URI duplicaría la parte más gorda de la hoja de estilos para nada.

    Las cinco redondean sus CUATRO esquinas por igual. Cuando el panel iba pegado hacía
    falta lo contrario —esquinas a escuadra en medio y redondeadas solo arriba del todo y
    abajo del todo, para que las cuatro leyeran como una sola pieza—; separadas, cada una
    es su propia ficha y esa excepción sobra.
    """
    return [
        f'.st-key-lang_{l} button {{ top:{tops[l]}px !important; right:{borde}px !important; '
        f'width:{ancho}px !important; height:{alto}px !important; min-height:{alto}px !important; '
        f'border-radius:{radio}px !important;'
        + (f' background-image:url("{_flag_uri(l)}") !important;' if con_imagen else "")
        + " }"
        for l in i18n.LANGS
    ]

CSS_FLAGS = "\n".join(_css_banderas(FLAG_TOP, _FLAG_ANCHO, _FLAG_ALTO, _MENU_BORDE))
CSS_FLAGS_MOVIL = "\n    ".join(
    _css_banderas(FLAG_TOP_M, _FLAG_ANCHO_M, _FLAG_ALTO_M, _MENU_BORDE_M, con_imagen=False))

# ── Paleta base, literal. Referencia única para todo lo demás. ──
# Las CUATRO ANCLAS del autor, literales y sin tocar.
P_CLINICO, P_CIAN, P_PAPEL, P_TINTA = "#1565C0", "#00ACC1", "#F4F8FB", "#263238"
# La escalera del tema OSCURO, derivada. No está en la paleta del autor: son cuatro pasos
# de luminosidad sobre el tono de la tinta (h 230°, croma 0,020-0,024), en los mismos
# peldaños que traía la paleta anterior —L 0,122 · 0,212 · 0,307— para conservar sus
# saltos de 1,15 y 1,33:1, que es lo que hace que se lean como ALTURAS y no como colores
# distintos. Ver T(), donde se reparten.
P_NOCHE, P_ACERO, P_PLOMO = "#01070C", "#0C1B22", "#243239"
# La tinta del tema oscuro: el papel del autor llevado casi al blanco sobre el mismo eje
# (L 0,968). Es el reflejo de P_TINTA — cada tema escribe con el ancla del otro.
P_NIEBLA = "#F1F5F8"
# Paso alto del azul, derivado subiendo la luminosidad (L 0,513 → 0,762) con el tono
# intacto. Existe porque hay dos sitios que necesitan DOS pasos de marca y no uno: el
# degradado del filete de portada y el acento enfático en tema oscuro (C_DARK). Sin él,
# ese degradado sería un color plano y ese acento no tendría a dónde separarse.
P_CLINICO_ALTO = "#6AB5FF"

# ── Escala CATEGÓRICA: identidad de cada modelo. Orden fijo, nunca reciclado. ──
# Tinta = baseline clásico · azul-gris medio = puente estructural · cian = cuántico.
# Los dos clásicos van en la familia NEUTRA, invertidos entre temas para que el que hace
# de tinta sea siempre el que contrasta: niebla en oscuro y la tinta del autor en claro.
# En el slot del SVM va un paso del eje de la tinta (h 228°) que la paleta no trae hecho:
# sus superficies están todas por debajo de L 0,31 o por encima de L 0,92, y nada entre
# medias llega al 3:1 que exige un relleno de barra.
# DÓNDE se pone ese paso lo decide la separación con el cian, no el contraste con el fondo,
# y es la trampa de esta paleta: con las tres series en la mitad azul del círculo, dos de
# ellas pueden acabar a 18° de tono y a ΔL 0,04 —o sea, el mismo color en una gráfica— sin
# que ningún contraste contra el fondo lo denuncie. Así que el neutro se coloca AL OTRO
# LADO del cian: por encima en oscuro (L 0,80) y por debajo en claro (L 0,44), de modo que
# las tres quedan en una escalera de luminosidad con saltos de 0,12 a 0,17 en los dos temas.
# Y con croma 0,022 frente al 0,118 del cian: el cuántico es el ÚNICO saturado de los tres,
# que es la segunda vía por la que se distingue cuando la primera falla.
# El slot cuántico lleva el ACENTO, y esta es la mejora directa de tener dos cromáticos:
# ya no comparte color con el cromo de interfaz. En oscuro es el #00ACC1 del autor sin
# tocar (4,82:1 en el peor fondo); en claro se baja a L 0,573 porque el original da
# 2,74:1 sobre blanco y como relleno de barra no llegaría al 3:1. Se conserva el tono,
# se corrige el paso.
SERIES = {
    "lightgbm": P_NIEBLA  if _is_dark else P_TINTA,
    "svm_rbf":  "#B0C1C9" if _is_dark else "#46555C",
    "qsvm":     P_CIAN    if _is_dark else "#0091A6",
}

# ── Escala SECUENCIAL: magnitud (matriz de confusión, velocímetro) ──
# Un solo tono (h 256°, el del azul de marca), luminosidad monótona. Va en el azul y no
# en el cian por la razón de arriba: el cian significa "cuántico" en el resto de la app y
# una magnitud cian se leería como una medida del QSVM. El azul, en cambio, es la familia
# del cromo, y aquí aparece en pasos que ninguna pieza de interfaz usa.
# El índice 4 es SIEMPRE el extremo de máxima magnitud: en claro eso es el paso más
# oscuro y en oscuro el más brillante — en ambos casos, el que más se despega de su
# fondo. Los pasos van a ΔL constante (≈0,10 en oscuro, 0,12 en claro), no a ojo: es lo
# que hace que el orden se lea sin leyenda.
RAMP = (["#26364C", "#3F5168", "#5B6D86", "#7A8EA8", "#9FB3CE"] if _is_dark
        else ["#A9C0DE", "#879DBB", "#677C98", "#485C77", "#2B3E57"])

# ── Acento de marca (cromo de interfaz: navegación, foco, sliders, reglas) ──
# En CLARO es el #1565C0 del autor sin tocar, y eso es nuevo: es el primer acento de la
# app que entra intacto en el tema claro. Da 5,75:1 sobre la tarjeta, 5,38 sobre el
# lienzo y 4,61 sobre la barra lateral, o sea que pasa WCAG AA en los TRES fondos que
# reciben tinta y puede llevar rótulo donde caiga.
#
# La regla con la que se comprueba eso importa, porque se repite por toda la paleta: el
# acento no vive solo en las tarjetas. Cae también sobre el lienzo en los antetítulos y
# las cifras del deslizador, y sobre la barra lateral en los enlaces del buscador. Medido
# SOLO contra el blanco cualquier paso hasta L 0,57 parece pasar; contra la barra lateral,
# que es el fondo claro más oscuro que recibe texto, varios de esos se quedan por debajo
# del 4,5:1. Al medir un acento nuevo, medirlo contra ESE fondo y no contra el blanco.
# (La barra ACTIVA —sidebar_active— queda fuera de la cuenta a propósito: es un fondo de
# hover, y allí el rótulo cambia a tinta plena; ninguna pieza escribe en acento sobre él.)
#
# En OSCURO el azul del autor da 2,44:1 sobre la tarjeta y no vale: se sube por el mismo
# tono hasta L 0,676, que es el paso MÁS SATURADO que aún pasa 4,5:1 contra los cuatro
# planos del tema. Se conserva todo el color que se puede conservar.
C_PRIMARY = "#4D98F8" if _is_dark else P_CLINICO
# C_DARK es el acento ENFÁTICO: el mismo azul separado un paso MÁS de la superficie, en
# la dirección que corresponda a cada tema (más brillante en oscuro, más profundo en
# claro). Lo usan los rótulos que tienen que ganarle al acento normal sin cambiar de
# color — el botón de la puerta que toca en la esfera de Bloch, el ítem seleccionado.
C_DARK    = P_CLINICO_ALTO if _is_dark else "#00479F"
C_QUANTUM = SERIES["qsvm"]          # acento del componente cuántico (Bloch, ZZFeatureMap)
# Y aquí está la diferencia con las dos paletas anteriores. En la original, C_QUANTUM y
# C_PRIMARY eran dos cálidos distintos (#F9C449 y #F5A623) que se llevaban ΔE 7,1: el
# mismo color para cualquier ojo. En la siguiente había un solo cálido y los dos pasaron
# a valer literalmente lo mismo. Con dos cromáticos vuelven a ser dos colores de verdad
# —azul 256° y cian 210°—, así que el componente cuántico se distingue del cromo de
# interfaz sin depender de dónde esté puesto.
# Cuando el cian tiene que llevar TEXTO en claro no vale ni el del autor ni el paso de
# relleno: hace falta bajar a L 0,478 (#007186, 4,55:1 en el peor fondo).
C_QUANTUM_TEXTO = P_CIAN if _is_dark else "#007186"
#
# Pasos intermedios. Aquí SE SEPARAN los dos, que antes salían ambos de la rampa:
#   · C_MID1 es dato — la retícula de ticks del deslizador y las líneas guía de las
#     gráficas—, así que sale de la RAMPA, que es la escala del dato.
#   · C_MID2 es BRILLO — el halo del interruptor de tema, el anillo de las láminas,
#     el pulgar del deslizador, el latido de la portada—, así que sale del CROMO, que
#     es la familia que dice "esto responde". No del cian: un halo cuántico alrededor
#     de un deslizador clínico diría algo que no es.
#     Coincide con C_PRIMARY en los dos temas, y se deja declarado aparte igualmente:
#     son roles distintos —uno lleva texto, el otro solo luz— y la próxima paleta puede
#     necesitar separarlos, como ya pasó con el ámbar.
C_MID1  = RAMP[2]
C_MID2  = C_PRIMARY

# ── Colores de ESTADO (reservados: nunca se reutilizan como “serie 4”) ──
# Única excepción deliberada a la paleta base: bien/atención/grave tienen que leerse
# como estado de forma inmediata y universal. Se conservan tal cual venían, y ahora con
# MÁS margen que antes: la razón por la que el "atención" se había desplazado a un
# naranja quemado era no confundirlo con una marca dorada, y esa marca ya no existe —el
# cromo es azul y el acento cian, así que los tres estados quedan a más de 100° de tono
# de todo lo demás—. Se dejan donde están porque están medidos y funcionan; lo que se ha
# revisado es que sigan pasando contra los fondos NUEVOS, que no son los de entonces.
# Los seis pasan 4,5:1 contra los CUATRO fondos de su tema: se usan como TINTA (el
# rótulo "passed", las cifras de los KPI), no solo como puntos de color. El peor caso de
# los seis es el "grave" oscuro sobre el plomo —la superficie alterna, el plano más claro
# del tema oscuro—, y con la escalera nueva se queda en 4,60:1.
STATUS = {
    "good":     "#3FC98B" if _is_dark else "#196646",
    "warning":  "#F5854B" if _is_dark else "#984012",
    "critical": "#F46F66" if _is_dark else "#9C2B24",
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

def T(tema=None):
    """Superficies y tintas, derivadas de la paleta base.

    Con `tema` se puede pedir una paleta que NO es la activa. Hoy no lo usa nadie: lo usaba la
    portada de Resumen, que iba siempre en oscuro, y dejó de hacerlo al revelarse en negativo
    con la app en claro (ver .ov-hero-img). El parámetro se conserva porque es la forma de pedir
    una paleta cruzada sin copiar hexadecimales a mano, que es justo lo que hay que evitar.

    El tema OSCURO usa los tres oscuros del autor tal cual, y en el orden en que ya
    venían escalonados — porque venían escalonados. Es una ESCALERA DE ELEVACIÓN de
    cuatro planos, y de ella sale todo el aspecto "elevado" del tema:

        barra lateral  #0A0E19   L 0,166   el plano que RECIBE (recede)
        lienzo         #05060A   L 0,123   el suelo, el más profundo
        tarjeta        #141826   L 0,212   +1,15:1 sobre el lienzo
        alterna        #2B2F3A   L 0,306   +1,32:1 sobre la tarjeta

    Que los saltos sean de 1,15 y 1,32 no es un defecto: es exactamente lo que se le
    pide a unos planos de elevación. Se distinguen como ALTURAS, no como colores, y
    quien termina de separarlos es la sombra más el filete de .kpi-card. Si el salto
    fuera mayor cada tarjeta parecería un bloque de otro color pegado encima, que es
    el aspecto de panel de 2015 que el sistema lleva evitando desde el principio.

    Hay UNA inversión respecto de las paletas anteriores, y es forzada: antes la barra
    lateral caía POR DEBAJO del lienzo, y aquí no puede. El lienzo está clavado en
    #05060A por decisión de paleta y ya casi no queda recorrido por debajo — el paso
    siguiente hacia abajo es negro puro, que da 1,02:1 contra el lienzo (invisible) y
    encima banda en pantallas malas. Así que la barra sube en vez de bajar: L 0,166,
    1,05:1 sobre el lienzo. Ver .ov-hero-img, que usa sidebar_bg como telón y a la que
    esta inversión sí le cambia el sentido de la profundidad.

    El tema CLARO tiene el problema contrario y hay que tenerlo presente: el lienzo
    #F1F5F9 está en L 0,968, a solo 1,10:1 del blanco. La tarjeta blanca CASI NO SE
    DESPEGA POR COLOR — mucho menos que con el platino anterior (1,31:1). Por eso en
    claro la elevación no la lleva el escalón tonal sino la SOMBRA (rehecha a tres
    capas, ver SHADOW) y el filete de borde, que a 1,47:1 sobre la tarjeta es lo que
    de verdad dibuja su contorno. La barra lateral y la superficie alterna comparten
    un paso más oscuro (#E2E9F0): son el mismo plano, como en las paletas anteriores.

    El par tinta/superficie da el contraste que se pedía: 17,7:1 en claro (#141826
    sobre blanco) y 16,1:1 en oscuro (#F1F5F9 sobre #141826) — muy por encima del
    4,5:1 de WCAG. Los escalones de tinta apagada se midieron contra los CUATRO fondos
    de su tema, no solo contra la tarjeta: 8,9:1 y 6,0:1 en claro (4,9:1 en el peor
    caso), 8,8:1 y 5,6:1 en oscuro (4,5:1 sobre el grafito, que es el fondo que más
    aprieta). TODOS los niveles de texto de la app pasan WCAG AA, incluido el más
    atenuado y sobre el fondo más adverso.
    Las tintas apagadas son FRÍAS (h 265°) y no grises: comparten tono con la escalera,
    así que el texto secundario parece hundirse en el plano en vez de ensuciarse, que
    es lo que pasa cuando se apaga con gris neutro sobre fondo frío. Por lo mismo, los
    neutros claros llevan el tono del propio #F1F5F9 (h 248°) y no croma cero.

    NOTA: el ámbar NO se usa como color de texto sobre blanco — da 1,75:1. Vive en
    bordes, rellenos y realces; cuando el acento tiene que llevar texto en claro, usa
    el paso oscurecido C_PRIMARY (5,59:1 sobre esta superficie, y nunca por debajo de
    4,57:1 en los otros tres fondos claros — ver allí, que es donde está la advertencia
    sobre medir contra el blanco).
    """
    if (tema or st.session_state.theme) == "dark":
        return dict(bg=P_NOCHE, surface=P_ACERO, surface_alt=P_PLOMO,
                     text=P_NIEBLA, text_secondary="#ADB8BD", text_muted="#8E999E",
                     border="#1C272C", border_strong="#3D484E",
                     sidebar_bg="#051016", sidebar_active="#13232A")
    return dict(bg=P_PAPEL, surface="#FFFFFF", surface_alt="#DDE8EE",
                 text=P_TINTA, text_secondary="#44525A", text_muted="#5A6A71",
                 border="#C7D1D7", border_strong="#9EACB3",
                 sidebar_bg="#DDE8EE", sidebar_active="#CFDBE1")

t = T()

# ── Elevación: una sola definición de sombra para TODAS las tarjetas ──
# TRES CAPAS y no dos, y aquí está la mitad del aspecto "elevado" de la app. Una sombra
# de una sola capa dice "hay una caja"; tres dicen a qué ALTURA está, porque es como se
# comporta la luz real: un contacto muy corto y casi opaco justo bajo el borde, una
# sombra media que da el cuerpo, y una difusa y muy abierta que sitúa la pieza en la
# habitación. Es la diferencia entre un panel y una tarjeta que flota.
#
# En CLARO esto no es cosmética, es estructural, y el motivo está en T(): el lienzo
# #F1F5F9 se queda a 1,10:1 de la tarjeta blanca, así que POR COLOR la tarjeta no se
# despega — casi todo el trabajo de separarla lo hacen estas tres capas y el filete de
# borde. Con la paleta anterior (lienzo a 1,31:1) bastaban dos capas flojas; aquí no.
# El tinte es el propio #05060A (5,6,10) y no un negro neutro: sobre un papel frío como
# el #F1F5F9 una sombra neutra lee gris sucio, y teñida del oscuro de la paleta lee como
# sombra de verdad.
#
# En OSCURO el reparto es el contrario: sobre un lienzo casi negro una sombra negra casi
# no tiene recorrido, así que allí quien separa es que la TARJETA ES MÁS CLARA que el
# lienzo (los 1,15:1 de la escalera) más el brillo cenital de CARD_SHEEN. La sombra sigue
# estando —y bien opaca— porque es lo que hunde el borde inferior, pero es la segunda voz.
SHADOW = ("0 1px 2px rgba(0,0,0,0.55), 0 4px 10px -2px rgba(0,0,0,0.45), "
          "0 12px 30px -8px rgba(0,0,0,0.70)" if _is_dark else
          "0 1px 2px rgba(5,6,10,0.06), 0 4px 10px -2px rgba(5,6,10,0.08), "
          "0 12px 28px -8px rgba(5,6,10,0.10)")
SHADOW_HOVER = ("0 2px 4px rgba(0,0,0,0.60), 0 8px 18px -4px rgba(0,0,0,0.55), "
                "0 24px 48px -12px rgba(0,0,0,0.80)" if _is_dark else
                "0 2px 4px rgba(5,6,10,0.08), 0 8px 18px -4px rgba(5,6,10,0.12), "
                "0 24px 44px -12px rgba(5,6,10,0.16)")

# ── Brillo de tarjeta: luz cenital, no un color ───────────────────────────────
# Un degradado vertical brevísimo sobre la superficie. Es lo que separa una tarjeta "plana
# con borde" —el aspecto de panel de 2015— de una que parece tener materia: sugiere que la
# luz cae desde arriba, igual que ya lo sugieren las sombras.
# Va en blanco/negro con alfa y NO en un tono de la paleta a propósito: cualquier color
# desplazaría el matiz del fondo de la tarjeta y arrastraría con él el contraste de TODO el
# texto que lleva encima, que está medido y documentado. Con blanco o negro translúcidos solo
# se mueve la luminosidad, y en una cantidad (3,5% y 2,5%) que no llega a tocar ningún ratio.
# En oscuro la luz suma por arriba; en claro no hay recorrido hacia el blanco desde el blanco,
# así que el volumen se consigue al revés, sombreando muy levemente por abajo.
CARD_SHEEN = ("linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0) 45%)" if _is_dark
              else "linear-gradient(180deg, rgba(5,6,10,0) 55%, rgba(5,6,10,0.022))")

# ── Relieve del buscador (neumorfismo) ────────────────────────────────────────
# Dos sombras opuestas en vez de una: la oscura abajo-derecha y la clara arriba-izquierda
# simulan un único foco alto a la izquierda, y la pastilla lee como EXTRUIDA de la barra en
# lugar de apoyada encima. El efecto solo funciona si la pieza comparte color con su fondo
# —una caja de otro color vuelve a leerse como caja—, así que NEU_BG se separa de sidebar_bg
# lo justo para que el relieve tenga de dónde salir, y el contorno lo dibujan las sombras.
# La asimetría entre temas no es un descuido: en claro el papel tiene recorrido hacia el
# blanco y la luz puede ir casi opaca (0,92), mientras que en oscuro cualquier blanco por
# encima del 4-5% sobre #0A0E19 se lee como niebla gris y no como luz. Allí el relieve lo
# sostiene la sombra negra, que sí tiene todo el rango por delante.
# NEU_INSET es el mismo relieve del revés (hundido) y marca el estado activo: es el gesto
# propio de este lenguaje —el control se PULSA— y evita añadir un cerco de color que
# rompería el monocromo.
NEU_BG     = "#08161C" if _is_dark else "#E4EFF4"
NEU_RAISED = ("5px 5px 11px rgba(0,0,0,0.60), -5px -5px 11px rgba(255,255,255,0.050)" if _is_dark
              else "5px 5px 11px rgba(5,6,10,0.14), -5px -5px 11px rgba(255,255,255,0.95)")
NEU_INSET  = ("inset 3px 3px 7px rgba(0,0,0,0.68), inset -3px -3px 7px rgba(255,255,255,0.055)"
              if _is_dark else
              "inset 3px 3px 7px rgba(5,6,10,0.16), inset -3px -3px 7px rgba(255,255,255,0.98)")

# ── Luz del lienzo: dos halos en las esquinas superiores ──────────────────────
# En OSCURO la luz se hace sumando color: sobre un fondo casi negro un ámbar al 12% sube
# la luminancia y el ojo lo lee como un halo. Y aquí rinde más que nunca, porque el lienzo
# #05060A es el más profundo que ha tenido la app (L 0,123): hay todo el rango por delante,
# así que el halo puede llegar a 1,30:1 contra el fondo sin parecer una mancha.
# En CLARO ese mismo gesto no funciona — la saturación cuesta luminancia, así que cualquier
# tono de marca sobre papel lo OSCURECE y lee como suciedad, no como luz. Por eso en claro
# se hace al revés: la esquina sube hacia el blanco y la temperatura la pone C_LUZ, el
# ámbar disuelto casi del todo en blanco (1,04:1 sobre blanco — inservible como texto,
# perfecto como tinte).
# El segundo halo va en blanco puro: sobre un lienzo FRÍO (#F1F5F9, h 248°) el blanco lee
# como luz cálida-neutra, y esa diferencia de temperatura entre las dos esquinas es todo el
# efecto — invertida respecto de la paleta anterior, donde el lienzo era cálido y el blanco
# hacía de luz fría. El ámbar cuántico no aparece aquí porque en claro no puede hacer de
# luz; se queda donde sí se lee, en las gráficas y los filetes.
# El margen en claro volvió a estrecharse: el lienzo está en L 0,968, a 1,10:1 del blanco,
# así que ningún halo claro puede pasar de ahí. Se dejan casi opacos justamente por eso —
# con tan poco recorrido, bajar la opacidad es apagarlos del todo.
C_LUZ = "#ECFEFF"   # el cian disuelto en blanco. Como él, nunca lleva texto.
HALOS = (f"radial-gradient(1100px 520px at 12% -8%, {C_PRIMARY}24, transparent 60%),"
         f"radial-gradient(900px 460px at 100% 0%, {t['border_strong']}3D, transparent 62%)"
         if _is_dark else
         f"radial-gradient(1100px 560px at 12% -8%, {C_LUZ}FF, transparent 64%),"
         f"radial-gradient(900px 460px at 100% 0%, #FFFFFFF2, transparent 62%)")
# Mismo principio en el velo de la barra lateral: en oscuro tiñe, en claro alumbra.
VELO_SIDEBAR = (f"linear-gradient(180deg, {C_PRIMARY}14, transparent 42%)" if _is_dark
                else f"linear-gradient(180deg, {C_LUZ}B3, transparent 46%)")

narrow = st.session_state.sidebar_narrow
SIDEBAR_WIDTH = "84px" if narrow else "270px"


def _flecha_mask(*trazos: str) -> str:
    """url() de máscara con esos trazos, sobre el lienzo 24×24 de siempre.

    Va como MÁSCARA y no como <img>: el relleno lo pone background-color, así que el icono es
    currentColor y hereda gratis el color del botón —incluido el paso a C_PRIMARY del :hover,
    con su transición—. Con una imagen habría que servir una versión por tema y otra por estado.
    Y va EN LÍNEA (data:) y no como fichero en assets/ porque son tres trazos: un .svg suelto
    sería una petición más y un asset que mantener a mano cada vez que se toque la paleta.
    El stroke es negro y da igual cuál sea: de una máscara solo cuenta el alfa.
    """
    d = "".join(f"%3Cpath d='{t}'/%3E" for t in trazos)
    return ('url("data:image/svg+xml,'
            "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
            "stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E"
            f'{d}%3C/svg%3E")')


# La flecha del toggle de la sidebar: asta con punta y TOPE, el mismo gesto espejado en los dos
# sentidos. Un chevron suelto («‹», lo que había antes) solo dice "hacia allá"; este dice hasta
# dónde, que es lo que de verdad hace el panel — se va contra su tope y se para.
#   colapsar  → punta a la izquierda y tope a la derecha, el borde de la barra yéndose
#   expandir  → el espejo exacto: tope a la izquierda y punta a la derecha
# La dirección se elige aquí, en Python, y no con una clase extra en el HTML: `narrow` ya está
# en ámbito —lo usa también SIDEBAR_WIDTH— y el CSS se regenera entero en cada rerun.
# Los dos sentidos, cada uno con su nombre, porque no todos los mandos eligen igual: el toggle
# propio de escritorio alterna según `narrow`, pero los dos botones NATIVOS —el de cerrar el panel,
# que en el teléfono es la única forma de cerrarlo, y el de volver a abrirlo— tienen cada uno un
# sentido FIJO, que no depende del ancho de la barra.
FLECHA_EXPANDIR = _flecha_mask("M3 5v14", "M21 12H7", "m15 18 6-6-6-6")
FLECHA_COLAPSAR = _flecha_mask("m9 6-6 6 6 6", "M3 12h14", "M21 19V5")
FLECHA_TOGGLE = FLECHA_EXPANDIR if narrow else FLECHA_COLAPSAR
# La flecha del botón «volver arriba», sobre el mismo lienzo de 24×24 y con el mismo trazo que
# las del toggle: asta y punta, no un chevron suelto. Vale aquí el motivo anotado ahí arriba —un
# chevron dice «hacia allá» y este dice hasta dónde—, y pesa incluso más, porque el destino no es
# «un poco más arriba» sino el principio de la página, se pulse donde se pulse.
FLECHA_ARRIBA = _flecha_mask("M12 19V5", "m5 12 7-7 7 7")
# Los signos del árbol de secciones: «+» en la rama plegada y «−» en la desplegada. Van como
# máscara y no como los caracteres «+» y «−» por lo mismo que las flechas —currentColor gratis,
# sin depender de que la fuente traiga el glifo— y por una razón más que aquí pesa: los dos se
# dibujan sobre el MISMO lienzo de 24×24, así que ocupan exactamente igual y la fila no se
# reajusta al cambiar de estado. Con dos caracteres de una fuente proporcional, el rótulo daría
# un salto de un par de píxeles cada vez que se despliega.
# Comparten además el trazo del resto del cromo (2 px, extremos redondeados), que es lo que hace
# que se lean como parte del mismo juego de mandos y no como texto suelto al final de la fila.
SIGNO_MAS = _flecha_mask("M12 5v14", "M5 12h14")
SIGNO_MENOS = _flecha_mask("M5 12h14")
# El disco del toggle va en el tema CONTRARIO al de la app: claro sobre la barra oscura y oscuro
# sobre la barra clara. Es el único control que se sale a propósito de la escala de superficies de
# T(): con sidebar_bg de fondo se mimetizaba con la barra sobre cuyo borde se apoya, y es el gesto
# que más se busca de un vistazo. Son los dos extremos de la paleta —los mismos que T() reparte
# entre tinta y superficie—, aquí intercambiados; al ser un único par, el contraste entre flecha
# y disco es el mismo 16,1:1 en los dos temas.
TOGGLE_DISCO  = P_NIEBLA if _is_dark else P_TINTA
TOGGLE_FLECHA = P_ACERO  if _is_dark else P_NIEBLA

# ── Contador de visitas: la chapa ─────────────────────────────────────────────
# La pieza es una CÁPSULA PARTIDA: a la izquierda un ojo sobre neutro, a la derecha la cifra
# sobre el azul de marca. Sustituye al odómetro de cinco plaquitas, que era una pieza de museo
# —tambor, junturas a media altura, muescas del eje— con seis medidas propias y dos degradados
# que no usaba nadie más. En una barra de 84 px lo que se lee no es el mecanismo: es el número.
#
# EL RÓTULO SE VA Y ENTRA UN OJO. La palabra ("visitas", "Besuche", "visites"…) medía entre 6 y
# 8 caracteres según el idioma y era lo único de la pieza cuyo ancho no controlábamos; el ojo
# mide igual en los cinco. El texto no se pierde: viaja en el title y en el aria-label, que es
# donde lo lee quien lo necesita.
#
# EL RADIO ES DE PÍLDORA (999px) y no los 4-5 px de una chapa al uso, y es lo que la hermana con
# su vecina de fila: el interruptor de tema es una cápsula de 30×15, así que las dos piezas del
# zócalo son ahora la MISMA forma —una entera y otra partida en dos—. Con esquinas de 4 px se
# leían como dos controles de bibliotecas distintas puestos en la misma línea.
#
# El cuerpo IZQUIERDO va en el neutro oscuro de la paleta, y resulta ser casi el mismo color en
# los dos temas —#243239 en oscuro (el plano alterno de la escalera de T()) y el #263238 del
# autor en claro— porque lo que tiene que hacer es lo CONTRARIO en cada uno: subir sobre la
# barra oscura (#051016) y bajar sobre la clara (#DDE8EE). Un solo paso de gris frío resuelve
# los dos casos, y el ojo va en niebla encima: 10,9:1 en oscuro y 13,1:1 en claro.
VC_NEUTRO = t["surface_alt"] if _is_dark else P_TINTA
VC_OJO    = P_NIEBLA
# El cuerpo DERECHO va en C_PRIMARY —el mismo azul del halo del interruptor que tiene al lado,
# que es de donde sale la coherencia entre las dos piezas—, y por eso la TINTA de la cifra
# cambia de tema aunque el fondo no cambie de papel: sobre el azul alto del tema oscuro
# (#4D98F8) la niebla se queda en 2,3:1 y hay que escribir con el fondo de la paleta (8,9:1);
# sobre el #1565C0 del claro, al revés (5,4:1 en niebla). El acento es el mismo en los dos; lo
# que se elige aquí es qué se lee encima de él.
VC_CIFRA = P_NOCHE if _is_dark else P_NIEBLA
# El ojo de Feather, con el trazo de 2 px del resto del cromo (ver _flecha_mask). La pupila va
# como ARCO y no como <circle> porque la máscara solo monta <path>: dos semicírculos encadenados
# dan exactamente el mismo círculo de r=3 centrado en el lienzo de 24×24.
VC_OJO_MASK = _flecha_mask("M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z",
                           "M15 12a3 3 0 1 1-6 0 3 3 0 1 1 6 0")
# UNA SOLA MEDIDA gobierna la chapa entera: relleno, ojo, alto y radio van en em sobre esta, así
# que la pieza crece y encoge de una pieza en vez de descuadrarse por partes.
# Va en clamp(rem) y no en píxeles pelados por el zoom de SOLO TEXTO de Firefox, que escala la
# raíz pero NO el ancho de la barra: con píxeles la chapa se quedaría clavada mientras el texto
# de alrededor crece, y con rem a secas se saldría de una
# barra que no se mueve. El clamp la deja seguir al zoom entre dos topes que sí caben.
# Los topes salen del ancho útil de cada barra. Colapsada son 84 − 12 de relleno = 72 px, y la
# chapa mide 6,8 em (2,4 el cuerpo del ojo y 4,35 el de cinco cifras en monoespaciada): 67 px
# en el tope alto, medidos con la barra colapsada y el contador puesto a 99999. En la ancha
# sobran 190 px, así que ahí el tope lo pone la legibilidad y no el hueco.
VC_FS = "clamp(8px, 0.55rem, 9.8px)" if narrow else "clamp(9.5px, 0.72rem, 12.5px)"
# Color del carril vacío de los sliders: claro en tema claro, hundido en tema oscuro (si usáramos
# un azul fijo, en oscuro el carril quedaría un surco brillante sobre fondo oscuro).
# En claro NO se usa RAMP[0]: ese paso está calibrado para pintar DATO sobre blanco y como
# carril de control resultaba demasiado saturado. Aquí va un tinte más apagado del mismo tono.
SLIDER_GROOVE = "#C3D1E3" if st.session_state.theme == "light" else t["surface_alt"]

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

# ── Mandos de la secuencia de puertas (Esfera de Bloch) ──
# Las claves REALES de los cinco botones, en un solo sitio, y de aquí salen los seis grupos de
# selectores que los visten. Antes se escribían a mano, uno por uno y seis veces, y ahí se coló
# `.st-key-ent_cnot`: las claves son ent_cnot1 y ent_cnot2 (ver el bucle que los crea), así que
# ese selector no casaba con NADA y los dos botones de CNOT salían con el tema base de Streamlit
# —transparentes y con texto rgba(49,51,63,0.4)— en los dos temas. En oscuro eso es 1,46:1: dos
# fantasmas al lado de tres hermanos bien vestidos. El fallo no daba error, que es justo lo que
# lo hizo durar; generándolos se acaba la clase de error. Mismo recurso y mismo motivo que
# _sel_lang() con las banderas.
_ENT_BTN = ("ent_h", "ent_cnot1", "ent_cnot2", "ent_reset", "ent_medir")
_ENT_PASO = ("ent_h", "ent_cnot1", "ent_cnot2")   # los que marcan "te toca": H y los dos CNOT

def _sel_ent(sufijo="button", claves=None):
    """Lista de selectores `.st-key-<clave> <sufijo>` para los mandos de la secuencia."""
    return ", ".join(f".st-key-{k} {sufijo}" for k in (claves or _ENT_BTN))

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
                 else "1px 0 2px rgba(5,6,10,0.04), 6px 0 22px -8px rgba(5,6,10,0.09)"};
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
/* LOS DOS BOTONES NATIVOS de la barra lateral: el de cerrarla (que en el teléfono es la única
   forma de cerrar el panel, porque allí nuestro toggle se esconde) y el de volver a abrirla
   cuando queda oculta del todo. Llevan el MISMO disco invertido y la MISMA flecha que el toggle
   propio de escritorio: es la misma acción, así que no puede tener dos aspectos según por dónde
   se llegue a ella.
   Esto apuntaba antes a [data-testid="collapsedControl"], que es de una versión anterior de
   Streamlit y en la 1.55 NO EXISTE —comprobado en el bundle—, así que era hoja muerta: en
   escritorio no se notaba porque manda nuestro toggle, pero en el teléfono, que es donde se usan
   los nativos, salía la flecha de Streamlit con su forma y su color.
   OJO a la asimetría de los dos testids, que es real y no un descuido: stExpandSidebarButton ES
   el <button>, mientras que stSidebarCollapseButton es el <div> que lo contiene.
   LAS MEDIDAS DE AQUÍ SON LAS DEL TELÉFONO, y por eso no van dentro de la media query: estos dos
   botones NO EXISTEN por encima de los 768px —el de cerrar va en display:none unas líneas más
   arriba y el de abrir ni siquiera llega al DOM mientras la barra esté desplegada, que en
   escritorio es siempre (comprobado a 820, 1024 y 1440px: querySelector devuelve null)—. O sea
   que este bloque es, en la práctica, CSS de móvil, y nada de lo que se toque aquí puede alterar
   el escritorio.
   El disco baja de 34 a 30px: sobre una barra de 320px y una cabecera de 46 era la pieza más
   grande de la pantalla, y con la flecha ya a su tamaño no necesita tanto cuerpo. La pareja
   disco/flecha queda en 30/19, la MISMA proporción que el toggle propio de escritorio (24/15),
   subida a la escala del dedo. */
[data-testid="stSidebarCollapseButton"] button,
button[data-testid="stExpandSidebarButton"] {{
    /* Referencia de posicionamiento para el ::after que amplía el área táctil (ver abajo). */
    position:relative !important;
    background-color:{TOGGLE_DISCO} !important;
    color:{TOGGLE_FLECHA} !important;
    border:1px solid {TOGGLE_DISCO} !important;
    border-radius:50% !important;
    box-shadow: 0 2px 6px rgba(5,6,10,0.10), 0 1px 3px rgba(5,6,10,0.08) !important;
    /* min-width explícito porque Streamlit le pone uno propio de 28px: sin él, el disco no
       bajaría de esa medida por mucho que se le pida un ancho menor. */
    width:30px !important; height:30px !important;
    min-width:30px !important; min-height:30px !important; padding:0 !important;
    display:flex !important; align-items:center !important; justify-content:center !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}}
/* El icono nativo NO es un <svg>: es una ligadura de Material dentro de un <span>, con su propio
   dibujo (una flecha doble) y su propio color. Se retira entero y la flecha la pone el ::before
   con la misma máscara del toggle propio; al pintarse con currentColor entra sola en la
   transición de color del botón y en el C_PRIMARY del :hover.
   OJO AL SELECTOR, que aquí estuvo el fallo: la ligadura viene envuelta en DOS <span> anidados
   —button > span (el contenedor del icono, 24x24) > span[data-testid="stIconMaterial"] (la
   ligadura)— y esta regla apuntaba solo al de dentro. El envoltorio sobrevivía, y aunque no
   pintaba nada seguía ocupando sus 24px de caja... con flex:0 0 auto, o sea INENCOGIBLE. Dentro
   de un disco cuyo interior son 28px, el flex tenía que sacar esos 24 de algún sitio, y el único
   que cedía era nuestra flecha: de los 17px declarados se quedaba en 8 —y, al repartirse la
   línea en orden, arrinconada contra el borde izquierdo—. De ahí lo que se veía en el teléfono:
   un disco grande con una flecha diminuta y descentrada. Ocultando el ENVOLTORIO, el
   stIconMaterial se va con él (es su hijo) y la flecha se queda sola en la caja. */
[data-testid="stSidebarCollapseButton"] button > span,
button[data-testid="stExpandSidebarButton"] > span {{
    display:none !important;
}}
[data-testid="stSidebarCollapseButton"] button::before,
button[data-testid="stExpandSidebarButton"]::before {{
    /* flex:0 0 auto es el cinturón de seguridad de lo anterior: aunque Streamlit vuelva algún día
       a meter algo dentro del botón, la flecha ya no cederá su tamaño para hacerle sitio. */
    content:""; display:block; flex:0 0 auto;
    width:19px; height:19px; background-color:currentColor;
}}
/* Área táctil de 44px —el mínimo de las guías de iOS— sin engordar el disco: un rectángulo
   transparente que desborda el botón por los cuatro lados. Va en position:absolute, así que NO
   entra en la línea flex (no le quita sitio a la flecha) y, al ser hijo del botón, el toque lo
   recibe el botón. Importa porque en el teléfono este es el único mando que abre y cierra el
   panel. Los 8px salen de que el bloque contenedor de un absolute es la caja de RELLENO, que
   aquí mide 28 (30 menos los dos bordes): 28 + 8 + 8 = 44 justos. */
[data-testid="stSidebarCollapseButton"] button::after,
button[data-testid="stExpandSidebarButton"]::after {{
    content:""; position:absolute; inset:-8px; border-radius:50%;
}}
/* Cada uno con su sentido fijo: cerrar apunta a la izquierda, abrir a la derecha. */
[data-testid="stSidebarCollapseButton"] button::before {{
    -webkit-mask:{FLECHA_COLAPSAR} center / contain no-repeat;
    mask:{FLECHA_COLAPSAR} center / contain no-repeat;
}}
button[data-testid="stExpandSidebarButton"]::before {{
    -webkit-mask:{FLECHA_EXPANDIR} center / contain no-repeat;
    mask:{FLECHA_EXPANDIR} center / contain no-repeat;
}}
[data-testid="stSidebarCollapseButton"] button:hover,
button[data-testid="stExpandSidebarButton"]:hover {{
    background-color:{C_PRIMARY} !important; border-color:{C_PRIMARY} !important;
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
    /* Disco en el tema contrario al de la app: ver TOGGLE_DISCO. El borde va del mismo color que
       el relleno —y no en t['border']— para que sea un disco macizo: un aro en el tono de la barra
       alrededor de un disco invertido le pondría un halo justo donde ya hay borde de sidebar. */
    border-radius:50% !important; border:1px solid {TOGGLE_DISCO} !important;
    background-color:{TOGGLE_DISCO} !important; color:{TOGGLE_FLECHA} !important;
    box-shadow: 0 1px 4px rgba(5,6,10,0.15), 0 1px 2px rgba(5,6,10,0.10) !important;
    z-index:1000 !important; margin:0 !important;
    /* Acompaña al borde de la sidebar en el mismo tiempo/curva que su transición de ancho, así el
       botón viaja pegado al borde en vez de saltar de golpe a su nueva posición. */
    transition: left 0.32s cubic-bezier(0.4,0,0.2,1), top 0.32s cubic-bezier(0.4,0,0.2,1),
                background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
}}
/* El realce lo lleva ahora el DISCO, no la flecha. Antes bastaba con teñir la flecha de oro
   porque el disco iba en el tono de la barra; sobre el disco invertido, ese mismo oro quedaba
   lavado (el cian sobre la niebla no llega a 1,3:1). Pintando el disco de marca y dejando la flecha
   en su neutro, el par conserva contraste de sobra en los dos temas y el hover se lee incluso mejor. */
.st-key-toggle_sidebar button:hover {{
    background-color:{C_PRIMARY} !important; border-color:{C_PRIMARY} !important;
}}
/* La flecha la dibuja el ::before, no un carácter en el rótulo: ver FLECHA_TOGGLE, donde se
   elige el sentido y se explica por qué va como máscara. El <p> solo tiene que centrarla.
   El background-color es lo que se ve —la máscara solo recorta—, y al ser currentColor la flecha
   entra sola en la transición de color del botón y en el C_PRIMARY del :hover de aquí debajo. */
.st-key-toggle_sidebar button p {{
    display:flex !important; align-items:center; justify-content:center;
    line-height:1 !important;
}}
.st-key-toggle_sidebar button p::before {{
    content:""; display:block; width:15px; height:15px;
    background-color:currentColor;
    -webkit-mask:{FLECHA_TOGGLE} center / contain no-repeat;
    mask:{FLECHA_TOGGLE} center / contain no-repeat;
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
/* ── Zócalo de la barra: interruptor de tema y contador, en la MISMA fila ──────
   UNA PIEZA Y NO DOS BANDAS APILADAS. Antes el contador iba anclado en bottom:89px y la
   cápsula de tema en bottom:54px, cada una con su fondo opaco y su z-index, y las dos se
   tocaban sin costura para FINGIR que eran un solo zócalo. Ahora lo son de verdad: los dos
   cuelgan del mismo st.container(key="pie_barra"), que es quien lleva el position:fixed, el
   fondo y la transición de ancho. Una sola caja que colocar, y las dos piezas alineadas por
   el mismo align-items en vez de por dos anclajes que había que cuadrar a mano.
   REPARTO. El interruptor se arrima a la esquina izquierda y la chapa al canto derecho, las
   dos centradas en la misma línea. Se consigue con flex-direction:row-reverse, y no dando la
   vuelta al orden del código, porque el contador tiene que seguir siendo el PRIMERO del DOM:
   en la barra colapsada y en el teléfono esta misma fila se apila, y allí el orden que manda
   es el del documento — al revés, el contador saldría debajo del interruptor.
   COLAPSADA SE APILA porque no cabe: 84 px menos el relleno son 72 útiles, y la chapa sola se
   come 67 en su tope alto (medido a cinco cifras). Centrados y en columna, con el contador
   arriba, que es donde estaba.
   El FONDO OPACO hace aquí lo que hacía en las dos bandas anteriores: el árbol de secciones se
   desplaza por detrás y, sin banda, sus filas asomarían alrededor de las piezas y se comerían
   el clic de la que tocara debajo. El z-index queda entre el pie (997) y el toggle de la barra
   (1000), igual que antes.
   El envoltorio que Streamlit 1.55 mete alrededor de todo contenedor con clave (stLayoutWrapper)
   NO lleva la clase st-key-, así que se alcanza por :has(); sin disolverlo seguiría siendo ítem
   del flex raíz y cobraría su hueco de gap al final de la barra aunque midiera cero. */
div[data-testid="stLayoutWrapper"]:has(> .st-key-pie_barra) {{ display:contents !important; }}
.st-key-pie_barra {{
    position:fixed !important; bottom:54px; left:0; width:{SIDEBAR_WIDTH};
    box-sizing:border-box; padding:{'9px 6px' if narrow else '8px 10px'} !important;
    display:flex !important; flex-direction:{'column' if narrow else 'row-reverse'} !important;
    align-items:center !important; justify-content:{'center' if narrow else 'space-between'};
    gap:{'7px' if narrow else '8px'} !important;
    background-color:{t['sidebar_bg']}; z-index:999;
    transition: width 0.32s cubic-bezier(0.4,0,0.2,1);
}}
/* Streamlit sirve sus elementos a ancho completo: en una fila, el primero se quedaría con
   todo el hueco y empujaría al otro contra el canto. Se les devuelve su ancho propio. */
.st-key-pie_barra > div[data-testid="stElementContainer"] {{
    width:auto !important; flex:0 0 auto !important; margin:0 !important;
}}
.st-key-pie_barra div[data-testid="stMarkdownContainer"],
.st-key-pie_barra div[data-testid="stButton"] {{ width:auto !important; }}
/* El interruptor conserva su cápsula y su halo tal cual estaban: es la pieza que ya funcionaba,
   y además la que FIJA el color con el que se hermana el contador —el halo va en C_MID2, que es
   el mismo azul del cuerpo derecho de la chapa—. Lo que cambia es el andamiaje: la colocación
   pasa a ser del zócalo, y el renglón sobrante se lo quita la regla de aquí debajo. */
/* El contenedor del botón hereda el renglón de Streamlit (25,6 px) y le sobran 10 por debajo de
   la cápsula, que mide 15. En escritorio el sobrante quedaba repartido y no se veía, pero en el
   teléfono Streamlit sirve OTRO botón —de un mismo st.button con help= cuelgan dos, y en pantalla
   táctil se pinta el de la variante sin tooltip—, y en esa la cápsula caía al fondo de su caja,
   5 px por debajo de la chapa. Sin renglón, la caja mide lo que mide la cápsula y quien centra
   las dos piezas es el align-items del zócalo, igual en los dos casos. */
.st-key-theme_toggle {{ line-height:0 !important; }}
.st-key-theme_toggle button {{
    width:30px !important; height:15px !important; min-height:15px !important; padding:0 !important;
    border-radius:999px !important; border:none !important;
    background-color:{"#FFFFFF" if st.session_state.theme == "light" else "#060810"} !important;
    box-shadow: 0 0 0 1px {C_MID2}55, 0 0 7px 1.5px {C_MID2}99, 0 0 15px 4px {C_MID2}55 !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}}
.st-key-theme_toggle button:hover {{
    box-shadow: 0 0 0 1px {C_MID2}88, 0 0 10px 2px {C_MID2}CC, 0 0 20px 5px {C_MID2}77 !important;
    transform: scale(1.05);
}}
.st-key-theme_toggle button p {{ font-size:0 !important; }}
/* ── Contador de visitas: la chapa ─────────────────────────────────────────────
   Ojo a la izquierda sobre neutro, cifra a la derecha sobre el azul de marca. El porqué de
   cada color —y de que la tinta de la cifra cambie de tema mientras el fondo no— está donde
   se declaran, en VC_NEUTRO y VC_CIFRA.
   TODO va en em sobre VC_FS: la chapa es UN número. Pasar de barra ancha a colapsada, o que
   Firefox agrande solo el texto, mueve ese número y la pieza entera lo sigue. */
.vc-badge {{
    /* flex y NO inline-flex, que es lo que parecía natural: en línea, la chapa se apoya en la
       BASE del renglón que Streamlit le pone alrededor (un <p> con line-height de 25,6 px) y
       se quedaba 1,4 px por encima del centro de la fila — justo lo que rompía la alineación
       con el interruptor, que sí está centrado. Fuera del flujo en línea no hay renglón, y las
       dos piezas las centra el mismo align-items del zócalo. */
    display:flex; align-items:stretch; flex:0 0 auto;
    font-size:{VC_FS}; line-height:1;
    border-radius:999px; overflow:hidden;
    /* El halo es el MISMO de la cápsula de tema, un punto más flojo: es el gesto por el que las
       dos piezas se leen como una pareja y no como dos cosas que coinciden en la misma línea.
       Va más flojo porque aquí la caja ya está pintada de ese azul —en la cápsula el halo es
       todo lo que hay— y a plena intensidad la chapa quedaría emborronada por su propia luz. */
    box-shadow: 0 0 0 1px {C_MID2}44, 0 0 6px 1px {C_MID2}55,
                0 1px 2px rgba(5,6,10,{'0.55' if _is_dark else '0.18'});
    /* No es un mando: no se pulsa ni recibe foco. El cursor de ayuda es lo que anuncia que sí
       lleva title — el rótulo que se le ha quitado a la vista vive ahí. */
    cursor:help; user-select:none;
}}
.vc-ojo, .vc-num {{ display:flex; align-items:center; height:1.85em; }}
/* Los rellenos son ASIMÉTRICOS a propósito: en una píldora los dos extremos son curvos y se
   comen aire óptico, así que el lado redondo de cada cuerpo pide algo más que el lado recto. */
.vc-ojo {{ background:{VC_NEUTRO}; color:{VC_OJO}; padding:0 0.55em 0 0.72em; }}
.vc-num {{
    background:{C_PRIMARY}; color:{VC_CIFRA}; padding:0 0.75em 0 0.6em;
    font-family:{FONT_MONO}; font-size:1em; font-weight:600; letter-spacing:0.02em;
    font-variant-numeric:tabular-nums;
}}
/* El ojo va como MÁSCARA sobre background-color y no como <img>, por lo mismo que las flechas
   del cromo (ver _flecha_mask): es currentColor, hereda el color del cuerpo y no hay que servir
   una versión por tema. */
.vc-ojo::before {{
    content:""; display:block; width:1.15em; height:1.15em;
    background-color:currentColor;
    -webkit-mask:{VC_OJO_MASK} center / contain no-repeat;
    mask:{VC_OJO_MASK} center / contain no-repeat;
}}
/* ── Selector de idioma: desplegable de banderas en la esquina superior derecha ──
   Va en el lienzo principal y no en la sidebar a propósito: el idioma afecta a TODA la
   aplicación, no solo a la navegación, y colapsar la sidebar no debe esconderlo. Como el
   <header> nativo de Streamlit está en visibility:hidden (ver más abajo), esa franja
   superior derecha está libre y no hay nada con lo que chocar.

   ESTRUCTURA. La bandera del idioma activo es el DISPARADOR y ocupa sola la esquina; las
   otras cuatro forman el PANEL, colgando justo debajo y ocultas hasta que se PULSA el
   disparador. Antes las cinco iban en fila y medían 162 px de franja; ver la geometría
   y el porqué en el bloque _reparto() de arriba.

   ABIERTO Y CERRADO son estado de sesión, no un :hover: el panel se despliega al pulsar la
   bandera activa y se queda ahí hasta que se elige idioma o se vuelve a pulsar el disparador.
   Ver el porqué —y qué fallaba con el hover— en el bloque _ABIERTO de arriba.

   TODOS los contenedores se disuelven con display:contents —el de la agrupación, el de
   cada elemento de Streamlit y el del propio botón—, no con el height:0 que usa el toggle
   de colapso. El motivo es que el bloque vertical de Streamlit es un flex con gap: un hijo
   de altura cero sigue siendo hijo y sigue cobrando su hueco, así que height:0 habría
   dejado huecos en blanco por encima del titular de la página. Con display:contents lo
   único que queda son los <button>, y como son position:fixed ni siquiera cuentan como
   ítems del flex: no ocupan absolutamente nada.

   Ese display:contents en la agrupación es además lo que permite que .st-key-lang_switch
   siga siendo el ancestro común de las cinco banderas —de ahí cuelga la condición de
   «abierto»— sin dibujar ninguna caja que tape la página debajo.

   La bandera es un background-image: así el botón sigue siendo un botón de Streamlit
   (accesible, con su tooltip y su foco de teclado) y la bandera es solo su piel; poner un
   <img> dentro habría exigido HTML, que no es pulsable. */
.st-key-lang_switch,
.st-key-lang_switch > div,
.st-key-lang_switch div[data-testid="stElementContainer"],
.st-key-lang_switch div[data-testid="stButton"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-lang_switch),
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-lang_switch) > div,
{_sel_lang()},
{_sel_lang(' div[data-testid="stButton"]')} {{ display:contents !important; }}
{_sel_lang(" button")} {{
    position:fixed !important;
    padding:0 !important; margin:0 !important;
    border:1px solid {t['border']} !important;
    background-repeat:no-repeat !important; background-position:center !important;
    background-size:cover !important;
    /* overflow visible: la flecha del disparador es un pseudoelemento que sale de la caja
       del botón, y Streamlit recorta los suyos por defecto. */
    overflow:visible !important;
    box-shadow: 0 1px 3px rgba(5,6,10,0.18) !important;
    z-index:1001 !important;
    transition: opacity 0.18s ease, filter 0.18s ease, transform 0.18s ease,
                box-shadow 0.16s ease, border-color 0.16s ease !important;
}}
/* El texto del botón es un espacio en blanco (la etiqueta real viaja en el tooltip):
   se colapsa a 0 para que no empuje la bandera ni asome bajo ella. */
{_sel_lang(" button p")} {{ font-size:0 !important; line-height:0 !important; }}
{CSS_FLAGS}
/* ── Disparador: la bandera activa ──
   A plena tinta y con anillo en color de marca, el mismo criterio que el ítem activo del
   menú: el estado se lee por contraste con las apagadas, no por un adorno añadido. */
.st-key-lang_{LANG} button {{
    opacity:1 !important; filter:none !important;
    border-color:{C_PRIMARY} !important;
    box-shadow: 0 0 0 1.5px {C_PRIMARY}66, 0 1px 3px rgba(5,6,10,0.18) !important;
    cursor:pointer !important;
}}
/* La flecha va DIBUJADA con bordes, no escrita con un carácter tipo "▾": este panel se
   defiende en Windows, donde ya se comprobó con las banderas que no se puede contar con
   que un glifo exista (ver la nota de i18n.FLAG_SVG). Un triángulo de bordes se ve igual
   en cualquier fuente y gira con transform como cualquier otra caja. */
.st-key-lang_{LANG} button::before {{
    content:""; position:absolute; right:100%; top:50%;
    margin-right:5px;
    width:0; height:0;
    border-left:3.5px solid transparent; border-right:3.5px solid transparent;
    border-top:4px solid {t['text_secondary']};
    transform:translateY(-50%);
    transition: transform 0.2s ease, border-top-color 0.16s ease;
    pointer-events:none;
}}
/* ── Panel: los idiomas inactivos ──
   Cerrado, se ocultan con opacidad y NO con visibility ni display: los dos últimos sacan
   el botón del orden de tabulación, y entonces el desplegable no se podría abrir con el
   teclado. Con opacidad 0 siguen siendo enfocables, y el :focus-visible del grupo los
   revela en cuanto el tabulador llega al primero. pointer-events:none evita que ese panel
   invisible intercepte clics dirigidos a la página. */
{_sel_lang(" button", LANGS_OTROS)} {{
    opacity:0 !important;
    filter:grayscale(0.55) !important;
    transform:translateY(-5px) !important;
    pointer-events:none !important;
    box-shadow:none !important;
}}
{_sel_abierto(" button", LANGS_OTROS)} {{
    opacity:0.72 !important;
    transform:translateY(0) !important;
    pointer-events:auto !important;
    box-shadow: 0 2px 8px rgba(5,6,10,0.22) !important;
}}
/* La bandera concreta bajo el cursor sube a plena tinta: dentro del panel abierto, el
   contraste vuelve a distinguir «la que voy a pulsar» de «las demás». */
{_sel_lang(" button:hover", LANGS_OTROS)},
{_sel_lang(" button:focus-visible", LANGS_OTROS)} {{
    opacity:1 !important; filter:grayscale(0) !important;
    border-color:{C_PRIMARY} !important;
    z-index:1002 !important;
}}
/* Abierto, la flecha se da la vuelta y toma el color de marca. */
{_sel_abierto(" button::before", [LANG])} {{
    transform:translateY(-50%) rotate(180deg);
    border-top-color:{C_PRIMARY};
}}
/* ── Reloj: fecha y hora, a la izquierda del selector de idioma ───────────────
   Comparte la franja del disparador (top:14px, alto 18px) y se alinea al mismo eje
   vertical, así los dos elementos leen como una sola tira de cabecera y no como piezas
   sueltas. Su borde derecho (RELOJ_RIGHT) se calcula desde el disparador y su flecha,
   dejando doce píxeles de aire. Calculado y no escrito a mano: esta cifra ya ha cambiado
   tres veces —94 con dos banderas, 196 con las cinco en fila, y ahora que el desplegable
   las recoge vuelve a caber en 72—, y cada vez a mano habría dejado el reloj debajo de
   una bandera. El PANEL desplegado no le afecta: cae por debajo de la franja del reloj,
   no a su lado.
   El contenido lo escribe un <script> en el documento padre (ver bloque RELOJ más abajo);
   aquí solo vive su aspecto, que así hereda el tema como cualquier otra regla.
   pointer-events:none porque es información, no un control: no debe capturar el cursor
   ni interponerse en un clic dirigido a lo que tenga debajo. Eso evita que estorbe al ratón,
   pero no a la vista: siendo fijo, el texto de la página le pasa POR DEBAJO y las dos cosas
   se cruzan. Por eso su opacidad no se fija aquí — la lleva el scroll, desde el bloque RELOJ.
   Las banderas no entran en el fundido: son control, no adorno, y tienen que seguir ahí.
   will-change avisa al compositor de que esa opacidad va a cambiar en cada scroll, para que
   promocione la capa una vez en lugar de replantear la tira de cabecera a cada fotograma.
   Tamaño, peso y color viven en el CONTENEDOR y en ninguna de las dos piezas: así fecha y hora
   son iguales por construcción y no por dos números que haya que mantener a la vez. Ninguna de
   las dos puede redeclarar font-weight — quien lo haga se sale de la negrita común. */
#tfm-reloj {{
    position:fixed; top:14px; right:{RELOJ_RIGHT}px; height:18px;
    display:flex; align-items:center; gap:6px;
    z-index:1001; pointer-events:none; user-select:none;
    font-size:11.5px; font-weight:600; line-height:1; white-space:nowrap;
    color:{t['text_secondary']};
    will-change:opacity;
}}
#tfm-reloj .r-fecha {{
    font-family:{FONT_SANS};
    letter-spacing:0.02em;
}}
/* El separador sí baja de tono: es puntuación, no dato. Sin esto, con las dos piezas ya en
   negrita, el punto medio se lee como un tercer carácter en vez de como la junta entre ambas. */
#tfm-reloj .r-sep {{ color:{t['text_muted']}; opacity:0.5; }}
/* La hora en mono y con cifras tabulares: sin tabular-nums el ancho de cada dígito cambia
   y el reloj «baila» un par de píxeles a cada minuto, que en un elemento fijo se nota
   mucho más que en una tabla. Ya NO lleva peso ni color propios: los tenía para destacar
   sobre una fecha apagada, y ahora que las dos van en la misma negrita ese font-weight:500
   habría hecho lo contrario de lo que parece —dejar la hora MÁS fina que la fecha—, porque
   pisaba el 600 del contenedor. */
#tfm-reloj .r-hora {{
    font-family:{FONT_MONO}; font-variant-numeric:tabular-nums;
    letter-spacing:0.04em;
}}
/* ── Botón «volver arriba» ────────────────────────────────────────────────────
   UNO SOLO, fijo en la esquina inferior derecha, y no uno insertado dentro de cada pestaña:
   solo hay una página y un panel visibles a la vez, así que un elemento por panel serían siete
   copias del mismo control de las que seis estarían siempre ocultas —y siete sitios donde se
   desincroniza—. El destino es el mismo en todas partes, el principio de la página, así que un
   botón por panel no compraría nada. Al ser fijo tampoco empuja el contenido ni se cuelga del
   final del último bloque de cada panel.

   Como el reloj: el nodo lo crea el script en el documento padre y su aspecto vive AQUÍ, para
   que siga al tema junto al resto de la hoja en vez de llevar los colores incrustados en el JS.

   La esquina estaba libre. Los elementos fijos que ya había viven todos arriba —reloj y
   banderas, z 1001— o a la izquierda —toggle de la barra z 1000, cápsula de tema z 999—, así
   que este entra en 1000 sin cruzarse con ninguno. Los 26 px de margen dejan además el disco
   por dentro de la barra de scroll de stMain, que corre pegada a ese mismo borde. */
#tfm-arriba {{
    position:fixed; right:26px; bottom:26px;
    width:36px; height:36px; padding:0;
    display:flex; align-items:center; justify-content:center;
    border-radius:50%; border:1px solid {t['border']};
    background-color:{t['surface']}; color:{t['text_secondary']};
    box-shadow:{SHADOW};
    cursor:pointer; z-index:1000;
    /* Oculto de partida: en la primera pantalla no hay nada a lo que volver, y la portada es
       una lámina a sangre en la que este disco sería lo único puesto por encima. */
    opacity:0; transform:translateY(6px); pointer-events:none;
    transition: opacity 0.22s ease, transform 0.22s ease,
                color 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
}}
/* La clase la pone y la quita el script según el scroll. pointer-events vuelve a auto SOLO
   cuando se ve: un disco invisible pero pulsable en esa esquina se comería los clics de lo que
   tiene debajo, que es justo donde caen las leyendas de varias gráficas. */
#tfm-arriba.visible {{ opacity:1; transform:translateY(0); pointer-events:auto; }}
#tfm-arriba:hover {{
    color:{C_PRIMARY}; border-color:{C_PRIMARY};
    box-shadow:{SHADOW_HOVER}; transform:translateY(-2px);
}}
#tfm-arriba:focus-visible {{ outline:2px solid {C_PRIMARY}; outline-offset:2px; }}
/* La flecha va en máscara y no en carácter por lo mismo que la del toggle: al ser currentColor
   entra sola en la transición a C_PRIMARY del hover, y no depende de que la fuente traiga el
   glifo. Un pseudoelemento además no entra en el árbol de accesibilidad, que es lo que se
   quiere de un dibujo cuando el nombre del botón ya lo pone el aria-label. */
#tfm-arriba::before {{
    content:""; display:block; width:17px; height:17px;
    background-color:currentColor;
    -webkit-mask:{FLECHA_ARRIBA} center / contain no-repeat;
    mask:{FLECHA_ARRIBA} center / contain no-repeat;
}}
/* Tira de tecnologías (Resumen).
   LA PASTILLA ES CLARA EN LOS DOS TEMAS a propósito, y no es un descuido: la mitad de estas
   marcas son monocromas oscuras —el logotipo de GitHub es #11110F y el de Qiskit #010101, y
   el texto de Spark y de scikit-learn tampoco aguanta— así que sobre el fondo pizarra
   desaparecerian. Recolorearlas no es una opcion: las guias de marca lo prohiben. Se les da
   entonces el fondo claro para el que fueron diseñadas y la marca viaja intacta; en tema
   oscuro la pastilla toma la niebla de la paleta, que ya es un blanco roto y no deslumbra
   sobre la pizarra, y en claro sube un punto por encima del lienzo para despegarse de él.
   El alto de cada imagen lo calcula _tech_alto() por área óptica — ver allí el porqué. */
/* Rejilla de DOS FILAS FIJAS de {TECH_POR_FILA}, a cualquier ancho y a cualquier zoom.
   Antes la tira repartía por ancho (auto-fit + minmax) y solo por encima de 1400px se fijaba el
   número de columnas. Eso significaba que en todo el tramo de en medio era el navegador quien
   decidía el corte —8+6, 9+5, 5+5+4…— y la banda cambiaba de forma al redimensionar la ventana
   o al cambiar el zoom. Ahora las columnas son SIEMPRE {TECH_POR_FILA}: lo que se adapta es el
   TAMAÑO de la pastilla, no su número. Es la única manera de garantizar el reparto parejo.
   La escala se mide en unidades de CONTENEDOR (cqw), no de viewport (vw): la sidebar se lleva
   270px y el ancho real de la tira no se deduce del de la ventana —ese desajuste es justo lo
   que obligaba a poner el umbral en 1400 en vez de en 1024—. .tech-strip-wrap es el elemento
   que se mide, así que la tira se ajusta a su hueco de verdad, con la sidebar abierta o
   cerrada y con la página al 100% o al 200%.
   Cada medida lleva delante su valor fijo de escritorio como reserva para navegadores sin
   container queries: allí la tira quedaría apretada, nunca rota (el max-height de la imagen la
   mantiene dentro de la pastilla pase lo que pase). */
.tech-strip-wrap {{ container-type: inline-size; }}
.tech-strip {{
    display:grid; grid-template-columns:repeat({TECH_POR_FILA}, minmax(0, 1fr));
    align-items:center; margin-top:4px;
    gap:10px; gap:clamp(4px, 1cqw, 10px);
    /* Unidad de escala de los logos: vale 1px de ~1000px de tira en adelante —el tamaño de
       siempre en escritorio— y baja proporcional por debajo, con suelo para que no acaben
       ilegibles. La consume .tech-chip img multiplicándola por el --th que _tech_alto()
       calculó para esa marca concreta. */
    --tech-u: clamp(0.60px, 0.1cqw, 1px);
}}
.tech-chip {{
    display:flex; align-items:center; justify-content:center;
    min-width:0;
    /* Alto subido de 46 a 52: al estirarse a ancho completo las pastillas pasan de ~66 px a
       ~125 px de ancho, y con el alto anterior quedaban como cápsulas aplastadas. Ese 52 es
       ahora el TECHO; por debajo de ~1000px de tira encoge con ella para que quepan las siete
       sin aplastarse. */
    height:52px; height:clamp(28px, 5.2cqw, 52px);
    padding:0 11px; padding:0 clamp(4px, 1.1cqw, 11px);
    box-sizing:border-box;
    background:{P_NIEBLA if _is_dark else '#F7FDFF'};
    /* EL MARCO. Hace un trabajo DISTINTO en cada tema, y por eso no es el mismo valor:
         · En CLARO lo sostiene TODO. La placa (#F8FAFD) sobre el lienzo (#F1F5F9) da 1,05:1
           — medido, no estimado: sin marco la pastilla sencillamente no existe, y así estaba
           antes, con un borde al 9% que se quedaba en 1,22:1. Aquí va el borde del tema, que
           es el mismo hilo que ya dibuja el contorno de las tarjetas: 1,34:1 contra el lienzo
           y 1,41:1 contra la placa, o sea visible por los DOS lados, que es lo que hace que
           se lea como un marco y no como una mancha con el canto sucio.
         · En OSCURO la placa ya se separa sola por fuerza bruta (18,5:1 contra el lienzo).
           Ahí el marco no separa: da CANTO. Un negro al 18% sobre la propia placa la baja a
           #C7CACE en su primer píxel (1,50:1), y eso convierte un rectángulo recortado en una
           pieza con borde. Un aro CLARO por fuera habría sido lo natural en otro sitio, pero
           sobre un fondo casi negro leería como halo encendido, no como marco. */
    border:1px solid {'rgba(5,6,10,0.18)' if _is_dark else t['border']};
    border-radius:9px; border-radius:clamp(6px, 0.9cqw, 9px);
    /* Y el ASIENTO, que es la otra mitad de "elevado": hasta ahora la pastilla solo tenía
       sombra al pasar el ratón, así que en reposo —que es como se ve el 99% del tiempo—
       estaba pegada al fondo como una calcomanía. Va la misma gramática que .kpi-card pero a
       escala de pastilla: un hilo interior abajo que le da grosor, un contacto corto y una
       difusa corta. Nada más: son 14 piezas seguidas y una sombra generosa las convertiría en
       una fila de botones. En oscuro la sombra pesa mucho más (0,55/0,60 frente a 0,05/0,08)
       porque cae sobre casi negro y ahí un alfa bajo no oscurece nada. */
    box-shadow:
        inset 0 -1px 0 rgba(5,6,10,0.055),
        0 1px 1px rgba(5,6,10,{0.55 if _is_dark else 0.05}),
        0 3px 8px -2px rgba(5,6,10,{0.60 if _is_dark else 0.08});
    transition: transform 0.14s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}}
/* El hover es LITERALMENTE el de .kpi-card —sube 2px, el borde se tiñe de acento y un aro de
   1px lo acompaña—, y eso es deliberado: son las mismas dos alfas (0x59 en el borde, 0x26 en
   el aro). Que la pastilla más pequeña de la app conteste al ratón igual que la tarjeta más
   grande es justo lo que hace que la interfaz parezca un sistema y no una colección de piezas.
   El aro va en el box-shadow y NO engordando el borde, que desplazaría el logo un píxel. */
.tech-chip:hover {{
    transform:translateY(-2px);
    border-color:{C_PRIMARY}59;
    box-shadow:
        inset 0 -1px 0 rgba(5,6,10,0.055),
        0 0 0 1px {C_PRIMARY}26,
        0 2px 3px rgba(5,6,10,{0.55 if _is_dark else 0.06}),
        0 8px 18px -4px rgba(5,6,10,{0.65 if _is_dark else 0.13});
}}
/* El height va EN LÍNEA, uno por marca (ver tech_strip). Los dos topes de aquí son la red:
   pase lo que pase con ese calc() —incluido un navegador sin container queries, donde
   var(--tech-u) no resuelve y el alto cae a auto— el logo no se sale de su pastilla. */
.tech-chip img {{ display:block; width:auto; max-width:100%; max-height:100%; object-fit:contain; }}
/* Los componentes que solo llevan <script> —el que fija <html lang>, el que escribe el reloj
   y el que hace subir los contadores— no pintan nada y sus iframes sobran. Se colapsan con el
   MISMO recurso que las banderas —display:contents en los envoltorios y position:fixed en el
   elemento—, así no cuentan como ítem del flex ni abren un hueco sobre el titular. No se usa
   display:none a propósito: un iframe así puede no llegar a ejecutar su script, y aquí el
   script ES todo el contenido.
   El position:fixed no es redundante con el width/height a 0: el `height=0` que se le pasa a
   components.html es FALSY, así que el frontal lo descarta y planta 150 px por defecto. Sin
   sacarlo del flujo queda una banda muerta de ese alto donde se monte el componente — que es
   justo lo que le pasó al de la portada y está anotado en su bloque. */
.st-key-lang_attr, .st-key-lang_attr div[data-testid="stIFrame"],
.st-key-lang_attr div[data-testid="stElementContainer"],
.st-key-reloj, .st-key-reloj div[data-testid="stIFrame"],
.st-key-reloj div[data-testid="stElementContainer"],
.st-key-contador_js, .st-key-contador_js div[data-testid="stIFrame"],
.st-key-contador_js div[data-testid="stElementContainer"],
.st-key-nav_js, .st-key-nav_js div[data-testid="stIFrame"],
.st-key-nav_js div[data-testid="stElementContainer"] {{ display:contents !important; }}
.st-key-lang_attr iframe, .st-key-reloj iframe, .st-key-contador_js iframe,
.st-key-nav_js iframe {{
    position:fixed !important; width:0 !important; height:0 !important;
    border:0 !important; opacity:0 !important; pointer-events:none !important;
}}
/* Footer fijo al fondo de la sidebar (por debajo del zócalo) */
.sidebar-footer {{
    position:fixed; bottom:0; left:0; width:{SIDEBAR_WIDTH};
    padding:8px 6px 10px; text-align:center; box-sizing:border-box;
    border-top:1px solid {t['border']}; background-color:{t['sidebar_bg']};
    color:{t['text_secondary']}; overflow:hidden; z-index:997; line-height:1.35;
    transition: width 0.32s cubic-bezier(0.4,0,0.2,1);
}}
.sidebar-footer .footer-name {{ font-size:12.5px; font-weight:500; color:{t['text']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
/* text_secondary y no text_muted, que es lo que llevaba. El fallo original fue medir contra
   SURFACE —blanco puro— un rótulo que no vive ahí sino sobre sidebar_bg: allí el escalón más
   apagado se quedaba en 4,15:1, por debajo del 4,5:1 de WCAG para texto pequeño.
   La paleta nueva ya cierra ese agujero por su cuenta —los tres niveles de tinta se calibraron
   contra los TRES fondos, barra lateral incluida, y text_muted da 4,73:1 allí—, así que hoy
   esto no sería un bug. Se mantiene text_secondary igualmente porque el margen es sano
   (6,4:1 en claro y 8,8:1 en oscuro) y porque la jerarquía del pie no depende de este escalón:
   el nombre sigue por encima —tinta plena, 12,5px y peso 500— y esta línea sigue por debajo,
   más pequeña, en mono y con la caja abierta por el espaciado. */
.sidebar-footer .footer-uni {{ font-family:{FONT_MONO}; font-size:12px; font-weight:400; letter-spacing:0.06em;
    color:{t['text_secondary']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }}
/* ── Suelo de la barra lateral ─────────────────────────────────────────────────
   Abajo del todo hay DOS piezas ancladas con position:fixed —el zócalo con el interruptor y
   el contador (bottom:54px) y el pie (bottom:0)— que FLOTAN sobre lo que pase por debajo.
   Mientras el contenido de la barra terminaba muy por encima no se notaba; con el árbol de
   secciones sí, y no como un defecto visual sino como uno de comportamiento: sus últimas filas
   se metían debajo del pie y dejaban de poder pulsarse —elementFromPoint sobre ellas devolvía
   el pie—, aunque a la vista estuvieran ahí. Un control que se ve y no responde es peor que
   uno que no se ve.
   El relleno inferior le da al contenido sitio donde acabar por encima de las dos. Sale de
   medirlas, no a ojo: el pie son 54 px y el zócalo llega a ~40 en la barra ancha y a ~56 en la
   colapsada (donde la fila se apila y suma el alto de las dos piezas), más un respiro de ~24.
   Eran 166 px cuando el contador y el interruptor iban en dos bandas separadas; hoy la fila
   única deja ese suelo en 118 y 134. Va en los dos anchos de escritorio, no solo en el ancho:
   una barra que puede desplazarse nunca debería terminar debajo de su propio pie.
   En el teléfono NO, y por eso lo deshace la media query de ≤768: allí las dos piezas vuelven
   al flujo del panel y ya no flotan sobre nada, así que este suelo sería una franja muerta al
   final de la lista. */
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
    padding-bottom:{134 if narrow else 118}px !important;
}}
/* ── Ancho del contenido de la barra colapsada ────────────────────────────────
   Se le devuelve el ancho completo: su relleno lateral de 20 px se reparte ya en cada
   bloque, y en 84 px esos 40 px eran casi la mitad de la barra. Lo que se ve al quitarlo es
   que la fila del menú llega de canto a canto, que es lo que hace legible su filete de
   página activa; con el relleno puesto, el filete quedaba flotando a 20 px del borde.
   Solo en modo colapsado — en modo ancho el reparto de Streamlit es el correcto.
   (Esto nació como parche del iframe del option_menu, que Streamlit dimensionaba a 23 px
   dentro de la barra de 84 y recortaba el icono por la derecha. El iframe ya no está; el
   ancho completo sigue haciendo falta por lo de arriba.) */
{'''section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
    padding-left:0 !important; padding-right:0 !important;
}''' if narrow else ''}
/* ── Reparto del aire alrededor del buscador ──────────────────────────────────
   Entre el filete que cierra el logo y la primera fila del menú hay tres cosas: el hueco de
   arriba, la pastilla (38px) y el hueco de abajo. Medido sobre captura, ese reparto era 29/17:
   al buscador le sobraba aire por arriba mientras casi tocaba el primer ítem por abajo, así que
   caja y menú se leían como un solo bloque. Ojo con medirlo por la tinta del icono de "Resumen"
   —da 35 y parece holgado—: lo que el ojo toma por borde del ítem es su CAJA, la que se ve
   sombreada en el ítem activo, y esa empieza 17px por debajo de la pastilla.
   Ahora va 21/25. Ni centrado exacto ni por gusto: el buscador queda colgado de la línea que
   cierra la cabecera, y el menú —siete filas seguidas, un grupo denso— se lleva algo más de
   separación que la que hay dentro del grupo de arriba. Lo gobiernan tres números, y hay que
   moverlos a la vez o el reparto se desequilibra: el margin-bottom del bloque del logo (en
   SIDEBAR, más abajo), el margin-top de .st-key-nav_search y este de aquí. */
.st-key-nav_tree {{ margin-top:22px !important; }}
/* ── Buscador de la sidebar ──────────────────────────────────────────────────
   Pastilla en relieve (ver NEU_* arriba): terminaciones ovaladas, sin borde, y el contorno
   dibujado solo por las dos sombras. Reposo = extruida; foco = hundida. El radio va en
   999px y no en un valor fijo para que las tapas sigan siendo semicírculos exactos aunque
   cambie el alto — es lo que separa una pastilla de un rectángulo muy redondeado. */
/* A cero: los 2px que había aquí eran parte del exceso de aire de arriba (ver "Reparto del
   aire alrededor del buscador"). Se conserva la regla, y no se borra, porque es el ajuste fino
   de ese reparto — el sitio donde tocar si hay que mover la pastilla un pelo. */
.st-key-nav_search {{ margin-top:0; }}
/* Lupa del modo colapsado: la MISMA pastilla reducida a círculo. En 84 px no cabe caja de
   texto, así que la entrada se repliega a su icono y al pulsarla despliega la barra (la
   lógica vive en el bloque BUSCADOR). Conserva los 38 px de los iconos del menú de debajo
   para que la columna siga leyéndose alineada. El rótulo del botón es un espacio en blanco
   (la etiqueta real viaja en el tooltip), así que se anula su hueco. */
/* ── Eje de la lupa colapsada ──────────────────────────────────────────────
   El botón se quedaba unos 19 px a la izquierda de la columna de iconos del menú, y es el
   mismo efecto secundario que se explica ahí arriba: al quitarle a la
   barra su relleno lateral de 20 px, el contenedor del botón conserva el ancho que tenía CON
   él —84 − 2×20 = 44— pero ya sin el desplazamiento que lo colocaba. Resultado: el botón se
   centra dentro de esos 44 px pegados al borde, no dentro de los 84 de la barra. Se le
   devuelve el ancho completo y del centrado ya se encarga la regla general de botones de la
   sidebar (justify-content:center, más arriba).
   Los 2 px de relleno izquierdo no son un ajuste a ojo: reproducen el `border-left:2px`
   transparente que llevan los ítems del menú, que corre su eje 1 px a la derecha. Sin ellos
   las dos columnas quedan alineadas a 1 px, que es justo lo que se venía a arreglar. */
.st-key-search_expand {{
    display:flex !important; justify-content:center !important; box-sizing:border-box !important;
    width:100% !important; min-width:100% !important; max-width:100% !important;
    padding-left:2px !important; padding-right:0 !important;
}}
/* El `help=` del botón lo envuelve en un objetivo de tooltip: si ese envoltorio se ajusta al
   contenido, rompe la cadena de anchos. Se le fuerza el mismo trato que al resto. */
.st-key-search_expand div[data-testid="stButton"],
.st-key-search_expand div[data-testid="stTooltipHoverTarget"] {{
    display:flex !important; justify-content:center !important;
    width:100% !important; padding:0 !important; margin:0 !important;
}}
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
   frenan el texto antes de la curva opuesta —ahí es donde degrada en puntos suspensivos—. Con
   menos, la pastilla se lee apretada; con más, en 270px de barra el hueco de escritura se
   queda corto. */
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
/* La pastilla desplegada NO lleva lupa. Streamlit la inyecta como startEnhancer de BaseWeb
   —primer hijo del contenedor flex, con relleno blanco propio—, y sobre una pastilla en
   relieve ese relleno se leía como un recuadro opaco pegado al texto, no como parte de la
   caja. Se quita en el origen (sin `icon=` en el widget) en vez de repintarla o reordenarla
   por CSS: era pelear con el interior del componente para conservar un glifo que aquí no
   informa de nada, porque el placeholder ya dice qué hace la caja. La lupa se queda donde sí
   es la única señal disponible: el botón del modo colapsado (.st-key-search_expand). */
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
   esa frase pide ~260px y en la pastilla no hay ni 200 de texto — 270 de barra menos el
   relleno de la sidebar y el de la propia pastilla. Se conserva el aire de la referencia con un
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
   TECLADO: al hacer clic basta con el hundido. */
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
/* Salida a la literatura: se separa de los resultados locales con un filete, porque es una
   acción de otra naturaleza — abandona la aplicación. Van dos piezas donde antes había un
   único enlace: un rótulo que repite la consulta y debajo la fila de destinos. El rótulo se
   saca fuera precisamente porque los destinos ahora son tres — meterlo en cada enlace, como
   estaba, escribiría la consulta tres veces en 270px de barra. Por eso además va en color
   apagado: es contexto, y lo pulsable son los nombres. */
.search-web {{
    margin-top:6px; padding:8px 10px 5px;
    border-top:1px solid {t['border']};
    font-size:12px; color:{t['text_muted']};
    /* Una consulta larga se recorta aquí en vez de partir el rótulo en tres líneas y empujar
       los destinos fuera de la vista. */
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}}
/* La fila envuelve: si un idioma alarga los nombres o el usuario agranda la tipografía del
   navegador, el tercer destino baja a una segunda línea en vez de desbordar la barra. */
.search-srcs {{
    display:flex; flex-wrap:wrap; align-items:center; gap:3px 12px;
    padding:0 10px 2px;
}}
a.search-src {{
    font-size:12.5px; color:{t['text_secondary']}; text-decoration:none !important;
    transition: color 0.13s ease;
}}
a.search-src:hover {{ color:{C_PRIMARY}; }}
/* La flecha va en CADA destino y no una vez en la fila: los tres abren pestaña nueva, y una
   sola flecha al final se leería como propiedad del último. */
a.search-src .search-web-ext {{ font-size:10.5px; opacity:0.75; margin-left:2px; }}

/* ── Menú y árbol de secciones ────────────────────────────────────────────────
   La navegación entera de la barra: seis filas de página y, colgando de cada una, sus
   pestañas y sus secciones. Son la MISMA lista y por eso están en el mismo bloque; el
   porqué de que el menú dejara de ser un streamlit-option-menu está donde se pinta.

   Cada rama se despliega al PULSAR su fila de página, y solo entonces: pasar el ratón por
   encima no mueve nada. Aquí abajo están las dos listas plegadas —max-height:0—; la que se abre
   la marca la barra lateral en cada pasada, que es quien sabe cuál es.

   max-height y no height:auto porque auto no se puede animar; el tope va holgado (620px, la
   rama más larga son 14 filas) porque max-height solo tiene que ser MAYOR que el contenido:
   lo que se ve es el alto real, y pasarse solo hace que la apertura empiece un pelín antes.

   La jerarquía se lee por TIPOGRAFÍA y no por sangrado: la fila de página lleva icono y va en
   sans a 13,5 px; la de pestaña, en monoespaciada versalita como todo rótulo de agrupación del
   panel; la de sección, en sans pequeña y apagada. Con las tres al mismo tamaño, un árbol de
   catorce filas se lee como una lista de catorce destinos iguales.
   (El margen superior que lo separa del buscador NO está aquí, sino en el bloque "Reparto del
   aire alrededor del buscador": es uno de los tres números que gobiernan ese hueco.) */
.st-key-nav_tree {{ gap:0 !important; }}
/* Los selectores van por SUBCADENA de clase ([class*=…]) y no por clase exacta: la clave de
   cada fila lleva dentro la página o la sección a la que apunta —.st-key-navp_governance,
   .st-key-navs_bl_zz_title—, que es lo que permite que el callback sepa a dónde ir sin
   registrar nada aparte. La familia se distingue por el prefijo: navb_ rama, navp_ página,
   navt_ pestaña, navs_ sección. */
div[class*="st-key-navb_"] {{ gap:0 !important; }}
div[class*="st-key-navb_"] div[data-testid="stButton"] {{ display:block !important; width:100% !important; }}
/* Se anula la botonera que la barra lateral impone a todos sus botones: aquí son filas de una
   lista, no botones. Mismo criterio y mismas anulaciones que en los resultados del buscador. */
div[class*="st-key-navb_"] button {{
    width:100% !important; min-height:0 !important; margin:0 !important;
    text-align:left !important; justify-content:flex-start !important;
    background:transparent !important; border:none !important;
    border-left:2px solid transparent !important; border-radius:0 8px 8px 0 !important;
    transition: background-color 0.13s ease, border-color 0.13s ease !important;
}}
/* Entre el <button> y el <p> Streamlit mete DOS envoltorios —un <div> y dentro un <span>—, y
   los dos son flex con justify-content:center. El párrafo se queda del ancho de su texto y
   sale centrado: las filas cortas centradas y las largas alineadas, que es exactamente lo que
   delata que aquello no es una lista. Hay que estirar y alinear los dos; con uno solo el otro
   vuelve a centrar. Se apuntan por posición (> div > span) y no por su clase, que es un hash
   de emotion que cambia con cada versión de Streamlit.
   El icono queda FUERA del estirado (de ahí el :not): es hermano del rótulo dentro del mismo
   flex, y estirado al 100% empujaría el texto fuera de la fila. Y el min-width:0 es lo que
   deja funcionar los puntos suspensivos de más abajo: sin él, un elemento flex no baja del
   ancho de su contenido y el rótulo largo desborda en vez de recortarse. */
div[class*="st-key-navb_"] button > div,
div[class*="st-key-navb_"] button > div > span:not([data-testid="stIconMaterial"]),
div[class*="st-key-navb_"] button div[data-testid="stMarkdownContainer"] {{
    width:100% !important; min-width:0 !important; justify-content:flex-start !important;
}}
/* El icono de la página: la única pieza de ancho fijo de la fila. Va en el acento de marca y
   no en el color del rótulo —C_PRIMARY es lo que la paleta reserva al cromo de navegación—, y
   es además lo ÚNICO que queda con la barra colapsada. Contraste sobre sidebar_bg: 9,33:1 en
   oscuro y 4,08:1 en claro, por encima del 3:1 que WCAG 1.4.11 pide a un elemento gráfico.
   El color del ítem activo (C_DARK) lo pone la barra lateral, que es quien sabe cuál es. */
div[class*="st-key-navb_"] button span[data-testid="stIconMaterial"] {{
    flex:0 0 auto !important; width:auto !important; margin:0 !important;
    font-size:15px !important; color:{C_PRIMARY} !important;
    transition: color 0.13s ease !important;
}}
div[class*="st-key-navb_"] button p {{
    width:100% !important;
    display:block !important; text-align:left !important; line-height:1.3 !important;
    /* Un rótulo largo se recorta con puntos suspensivos, igual que en el buscador: la lista
       tiene que conservar el mismo alto por fila para leerse como lista. */
    white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
    transition: color 0.13s ease !important;
}}
div[class*="st-key-navb_"] button:hover {{ background:{t['sidebar_active']} !important; }}
div[class*="st-key-navb_"] button:hover p {{ color:{t['text']} !important; }}
div[class*="st-key-navb_"] button:focus-visible {{
    outline:2px solid {C_PRIMARY} !important; outline-offset:-2px !important;
}}
/* Fila de PÁGINA: la raíz de la rama y, a la vez, el ítem del menú. Las medidas son las que
   tenía el option_menu —13,5 px, 10 de relleno, 3 de margen y la esquina derecha a 9— para que
   el cambio de componente no se note como un cambio de diseño. El hueco entre el icono y el
   rótulo va en `gap` sobre los dos posibles contenedores: Streamlit ha movido el icono de
   nivel entre versiones, y así la fila se compone igual esté donde esté. */
div[class*="st-key-navp_"] button {{
    padding:10px 10px !important; margin:3px 0 !important;
    border-radius:0 9px 9px 0 !important; gap:10px !important;
}}
div[class*="st-key-navp_"] button > div {{ gap:10px !important; }}
div[class*="st-key-navp_"] button p {{
    font-size:13.5px !important; font-weight:400 !important; color:{t['text_secondary']} !important;
}}
/* El signo del final de la fila: dice en qué estado está su rama y, por tanto, qué hace el
   clic. De partida «+» —plegada—; el «−» lo pone la barra lateral en la única fila que esté
   desplegada, cambiando SOLO la imagen de la máscara, para que el hueco, el color y la
   transición sean los mismos en los dos estados.
   Va en el ::after del <button> y no dentro del <p>: el párrafo se recorta con puntos
   suspensivos, así que un signo metido ahí sería lo primero en desaparecer con un rótulo largo
   —justo al revés de lo que hace falta—. Como hermano suyo en el mismo flex, el signo tiene su
   sitio reservado y es el rótulo el que cede.
   El margin-left:auto lo pega al canto derecho aunque un día el rótulo deje de estirarse. */
div[class*="st-key-navp_"] button::after {{
    content:""; flex:0 0 auto; margin-left:auto;
    width:13px; height:13px;
    background-color:{t['text_muted']};
    -webkit-mask:{SIGNO_MAS} center / contain no-repeat;
    mask:{SIGNO_MAS} center / contain no-repeat;
    transition: background-color 0.13s ease;
}}
div[class*="st-key-navp_"] button:hover::after {{ background-color:{t['text_secondary']}; }}
/* Fila de PESTAÑA: se lee como encabezado del grupo, no como destino más —aunque lo sea—, así
   que va en monoespaciada versalita, el mismo recurso con el que el panel marca todo lo que es
   rótulo de agrupación.
   La versalita sola no bastaba: con el mismo color y el mismo eje que sus secciones, la rama se
   leía como una lista seguida y no como grupos. La separan tres cosas, y ninguna añade cromo
   nuevo al árbol:
     · va MENOS sangrada que sus secciones —se arrima al filete— en vez de compartir su eje, que
       es lo que convierte a las de abajo en hijas suyas y no en vecinas;
     · sube un escalón de contraste, a text_secondary sobre el text_muted de las secciones: el
       rótulo del grupo tiene que pesar más que sus destinos aunque sea el más pequeño de los dos;
     · se despega de lo que tiene encima con 7 px de aire, que es lo que de verdad agrupa —el
       hueco de arriba pertenece al grupo que empieza, no al que termina—.
   Se movió la PESTAÑA y no las secciones a propósito: tres de las seis páginas no tienen
   pestañas y sus secciones cuelgan directamente de la página. Sangrarlas más habría abierto en
   ellas un nivel intermedio vacío, que es justo la jerarquía falsa que el árbol evita. */
div[class*="st-key-navt_"] button {{
    padding:6px 8px 5px 8px !important; margin-top:7px !important;
}}
div[class*="st-key-navt_"] button p {{
    font-family:{FONT_MONO}; font-size:10.5px !important; font-weight:500 !important;
    letter-spacing:0.09em; text-transform:uppercase; color:{t['text_secondary']} !important;
}}
/* Fila de SECCIÓN: la hoja del árbol. */
div[class*="st-key-navs_"] button {{ padding:5px 8px 5px 13px !important; }}
div[class*="st-key-navs_"] button p {{
    font-size:12px !important; font-weight:400 !important; color:{t['text_muted']} !important;
}}
{'''/* Modo colapsado: la fila se reduce a su icono, centrado en los 84 px. El rótulo NO se borra
   —es el nombre accesible del botón— sino que sale del flujo con el recorte de 1 px de siempre
   (el mismo patrón, y por los mismos motivos, que la cursiva del toggle de la barra), de modo
   que no arrastra el icono hacia la izquierda ni deja hueco a su derecha. El `help` del botón
   añade el globito con ese mismo nombre: con solo seis dibujos en columna, hace falta poder
   preguntar cuál es cuál.
   El objetivo del tooltip es un envoltorio más entre el bloque y el botón, y si se ajusta al
   contenido rompe la cadena de anchos: se le da el mismo trato que en la lupa colapsada. */
div[class*="st-key-navp_"] div[data-testid="stTooltipHoverTarget"] {
    display:block !important; width:100% !important;
}
div[class*="st-key-navp_"] button {
    padding:11px 0 !important; justify-content:center !important; gap:0 !important;
}
div[class*="st-key-navp_"] button > div,
div[class*="st-key-navp_"] button div[data-testid="stMarkdownContainer"] {
    width:auto !important; justify-content:center !important; gap:0 !important;
}
div[class*="st-key-navp_"] button p {
    position:absolute !important;
    width:1px !important; height:1px !important;
    margin:-1px !important; padding:0 !important;
    overflow:hidden !important; clip-path:inset(50%) !important; white-space:nowrap !important;
}
div[class*="st-key-navp_"] button span[data-testid="stIconMaterial"] { font-size:18px !important; }
/* Sin rama que plegar no hay nada que anunciar, y en 84 px el signo le comería el sitio al
   icono, que es lo único que queda. */
div[class*="st-key-navp_"] button::after { content:none !important; }''' if narrow else ''}
/* ── El árbol cuando se apunta con el dedo ────────────────────────────────────
   Se pregunta por el PUNTERO y no por el ancho, que es lo que de verdad cambia aquí: una
   tableta de 1024 px se maneja con el dedo igual que un teléfono de 390, y un portátil de 1280
   con ratón no necesita nada de esto. `pointer: coarse` responde exactamente a eso —cuál es el
   puntero PRINCIPAL—, de modo que un portátil con pantalla táctil Y ratón se queda en la rama
   fina, que es la correcta para él.
   Lo que cambia es solo el ALTO de las filas. Con ratón, una fila de sección de 26 px es
   cómoda; con el dedo es una diana pequeña, y además van en columna apretada, que es el caso
   en el que un fallo de puntería no te deja donde querías sino en la fila de al lado. Subidas a
   34, y las de página a 42, quedan en el orden de magnitud que piden las guías táctiles sin
   estirar el árbol: la rama más larga pasa de 376 a 484 px, todavía holgada bajo el tope de 620
   con el que se abre.
   No se toca ni el sangrado ni el color: la jerarquía que separa pestaña de sección es la misma
   se apunte con lo que se apunte. */
@media (pointer: coarse) {{
    div[class*="st-key-navp_"] button {{ padding:12px 10px !important; }}
    div[class*="st-key-navt_"] button {{ padding:9px 8px 8px 8px !important; }}
    div[class*="st-key-navs_"] button {{ padding:9px 8px 9px 13px !important; }}
}}
/* La lista que cuelga de cada página. El filete de la izquierda es lo que dibuja la jerarquía:
   sin él, unas filas sangradas se leen como filas sangradas y no como hijas de la de arriba. */
div[class*="st-key-navk_"] {{
    gap:0 !important; overflow:hidden !important;
    max-height:0; opacity:0;
    margin-left:11px !important; border-left:1px solid {t['border']};
    transition: max-height 0.3s cubic-bezier(0.22,1,0.36,1), opacity 0.2s ease;
}}
/* Falta la regla que ABRE una de estas listas, y falta a propósito: cuál está desplegada y
   cuál es la página activa son estado, y esta hoja se escribe antes de que la barra lateral lo
   resuelva. Se emiten allí, en cada pasada, junto al código que pinta la lista. */
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
    background-color:{t['surface']}; background-image:{CARD_SHEEN};
    border:1px solid {t['border']}; border-radius:14px; padding:20px 22px; height:100%;
    box-shadow: {SHADOW};
    transition: box-shadow 0.24s cubic-bezier(0.4,0,0.2,1), transform 0.24s cubic-bezier(0.4,0,0.2,1), border-color 0.24s ease;
}}
/* Al hover la tarjeta no solo sube: el borde se tiñe del acento y un aro de 1px lo acompaña.
   El aro va en el box-shadow y NO engordando el borde, que desplazaría el contenido un píxel
   —el salto que delata una tarjeta mal hecha—. El tinte se queda en alfa 0,35 sobre el borde:
   suficiente para que la tarjeta "conteste" al ratón, lejos de parecer seleccionada. Eso
   último importa aquí, donde ninguna tarjeta es pulsable y un estado de selección mentiría. */
.kpi-card:hover, .info-card:hover {{
    box-shadow: {SHADOW_HOVER}, 0 0 0 1px {C_PRIMARY}26;
    border-color:{C_PRIMARY}59;
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
/* Tarjeta de la figura de arquitectura (Resumen). Lleva MENOS aire lateral que las demás
   —16/18 en vez de 20/22— porque el propio dibujo trae su recuadro discontinuo y, con el
   padding normal, se leían dos marcos concéntricos separados por una franja vacía. El
   margen inferior es el mismo que el de la tarjeta del párrafo que va debajo. */
.arch-card {{ padding:16px 18px; margin-bottom:20px; }}
/* Fila de tarjetas comparativas: grid en vez de st.columns para que las 3 tengan SIEMPRE la
   misma altura (el estirado es nativo del grid), sin importar cuánto texto envuelva ni el zoom. */
.compare-grid {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px; align-items:stretch; }}
.compare-grid .info-card, .compare-grid .clinical-note {{ height:100%; box-sizing:border-box; }}
/* Esfera de Bloch: el alto lo fija la figura Plotly en píxeles (height=BLOCH_H en Python), SIN
   autosize. Así el tamaño es idéntico en cada rerun y en CUALQUIER navegador (Firefox incluido):
   no depende de medir el contenedor (lo que con autosize hacía que "volviera a quedar pequeña" al
   cambiar de variable o mover el slider, con distinto comportamiento entre Firefox y Chrome). Aquí
   solo centramos el recuadro; no forzamos alturas por CSS para no reintroducir el bucle de medición. */
.st-key-bloch_row div[data-testid="stPlotlyChart"] {{ display:flex; align-items:center; justify-content:center; }}
.st-key-bloch_row .info-card {{ height:auto; box-sizing:border-box; }}
/* Q-sphere (sección de entrelazamiento de la misma página): mismo centrado que su hermana de
   arriba y por el mismo motivo — el alto lo fija la figura en píxeles, aquí solo se centra. */
.st-key-ent_row div[data-testid="stPlotlyChart"] {{ display:flex; align-items:center; justify-content:center; }}
.st-key-ent_row .info-card {{ height:auto; box-sizing:border-box; }}
/* ─── Botones de la secuencia de puertas (H · CNOT · Reiniciar) ───
   PRIMER BOTÓN DE ACCIÓN DEL LIENZO PRINCIPAL: los otros cuatro de la app viven en la sidebar
   o son piel de otra cosa (banderas, cápsula de tema) y se visten uno a uno por su clave. Sin
   esta regla saldrían con el tema BASE de Streamlit, que config.toml deja en claro — el mismo
   fallo que tenía el expander: caja casi blanca en tema oscuro. Se visten como una .info-card
   pequeña, que es el lenguaje de superficie de toda la app.
   Se listan por clave y no con un selector global de botón a propósito: un `button {{...}}` a
   secas alcanzaría también a los de la sidebar, que ya tienen su propia forma. Las claves salen
   de _sel_ent() y no escritas a mano — ver allí el porqué. */
{_sel_ent()} {{
    background-color:{t['surface']} !important;
    border:1px solid {t['border']} !important;
    color:{t['text']} !important;
    border-radius:10px !important;
    padding:9px 14px !important; min-height:42px !important;
    font-family:{FONT_SANS} !important; font-size:14px !important; font-weight:500 !important;
    box-shadow:{SHADOW} !important;
    transition: border-color 0.15s ease, color 0.15s ease, transform 0.14s ease, box-shadow 0.16s ease;
}}
{_sel_ent("button p")} {{ color:inherit !important; }}
{_sel_ent("button:hover:enabled")} {{
    border-color:{C_PRIMARY} !important; color:{C_PRIMARY} !important;
    transform:translateY(-2px); box-shadow:{SHADOW_HOVER} !important;
}}
/* El botón de la puerta que TOCA ahora va en acento y con relieve; los otros dos quedan
   apagados y sin sombra. El estado deshabilitado de Streamlit solo baja la opacidad, y a la
   mitad de opacidad los tres se parecían demasiado como para ver de un vistazo cuál es el
   siguiente paso — que es toda la interacción de esta sección. */
/* C_DARK y no C_QUANTUM, que es lo que había: el acento cuántico en claro es #0091A6, y ese
   tono está declarado ARRIBA como relleno de barra —"en claro se baja a L 0,573 (…) como relleno
   de barra no llegaría al 3:1"—, no como tinta. De rótulo se queda en 3,54:1 sobre la superficie
   blanca del botón, por debajo del 4,5:1 de WCAG para texto. C_DARK es el paso oscurecido del
   azul de marca y sube a 9,30:1. En tema oscuro C_DARK es el paso ALTO del azul (#6AB5FF) y también
   gana al acento normal, así que la regla vale igual en los dos temas. */
{_sel_ent("button:enabled", _ENT_PASO)} {{
    border-color:{C_DARK} !important; color:{C_DARK} !important;
}}
{_sel_ent("button:disabled", _ENT_PASO + ("ent_reset",))} {{
    background-color:{t['surface_alt']} !important; border-color:{t['border']} !important;
    color:{t['text_muted']} !important; box-shadow:none !important; opacity:1 !important;
}}
{_sel_ent("button:disabled p", _ENT_PASO + ("ent_reset",))} {{ color:{t['text_muted']} !important; }}
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
   falta ejecutar código (escribir la página en session_state), pero NO debe leerse como botón:
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
/* AQUÍ VIVÍA .gov-dim, la cabecera de dimensión de la suite de calidad, y se ha retirado
   porque ya no la lleva nadie: al plegar las 15 expectativas en un expander por dimensión,
   ese rótulo pasó a ser el título del propio widget y la clase dejó de emitirse. Comprobado
   contra el DOM de las siete páginas: 0 coincidencias. Si algún día vuelven a listarse las
   expectativas seguidas, la cabecera se rehace con las mismas piezas que .gov-state. */
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
/* ── Variante VECTOR: la tarjeta deja de ser blanca ──────────────────────────────────────
   El blanco fijo de arriba está puesto para las LÁMINAS RASTER —el beeswarm de SHAP y el
   circuito de 8 qubits— que traen ese fondo dentro del PNG y no se pueden recolorear. El
   circuito de 3 qubits de la Esfera de Bloch NO es una de ellas: es un SVG que se dibuja en
   cada pulsación con los tokens de la paleta ACTIVA (ver ent_circuito_svg, que lo dice en su
   docstring), y sobre el blanco fijo esos tokens caían en el fondo equivocado. En tema oscuro
   la tinta del rótulo es la niebla #F1F5F9 y daba 1,10:1 contra el blanco: los tres «q₀ |0⟩»
   eran ilegibles —comprobado midiendo el fill sobre el DOM y en captura—. El dibujo se veía
   solo porque los hilos y los rellenos de las puertas sí son oscuros.
   Devolviéndole la superficie del tema, cada color aterriza en el fondo para el que se calculó
   (rótulo 16,1:1, hilo y acento cuántico por encima de 4,5:1) y de paso la tarjeta iguala a la
   de la Q-sphere que tiene al lado, que es su pareja en la misma fila.
   EN TEMA CLARO NO CAMBIA NADA: allí t['surface'] ya ES #FFFFFF y FIG_CARD_SHADOW ya ES SHADOW,
   así que estas dos reglas se resuelven en lo mismo que había. Para volver al aspecto de lámina
   blanca en oscuro basta con quitar la clase fig-vector del marcado. */
.fig-card.fig-vector {{
    background:{t['surface']};
    box-shadow: {SHADOW};
}}
.fig-card.fig-vector:hover {{ box-shadow: {SHADOW_HOVER}; }}
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
/* transform-origin a la izquierda: el filete se dibuja DESDE el título hacia el margen
   cuando entra la página (la animación vive en el bloque de entrada escalonada). Sin esto
   crecería desde su centro hacia los dos lados, que no es un gesto de escritura. */
.section-title::after {{
    content:""; flex:1 1 auto; height:1px; min-width:20px;
    transform-origin:left center;
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
/* La columna de etiquetas va a `min-content` y no a un ancho fijo. Iba a 64px y ahí se leía
   "REAL / DIABET / ES": la palabra se partía por la mitad.
   La CAUSA no era el ancho sino una herencia: el CSS base de Streamlit pone
   `overflow-wrap:break-word` en los contenedores de markdown, y esa propiedad se HEREDA hasta
   aquí. Con ella el navegador tiene permiso para trocear una palabra cuando no cabe; sin ella
   nunca lo haría, por estrecha que fuera la caja. Por eso el arreglo de verdad está abajo, en
   .cm-rowlabel, y esto es solo la mitad que le da el sitio que necesita.
   `min-content` = el ancho de la PALABRA más larga, ni un píxel más. Es justo la garantía que
   hace falta —que ninguna palabra tenga que partirse— sin robarle a las celdas el ancho de la
   FRASE entera, que es lo que pasaba con `auto`: en francés "Pas de diab." se lo llevaba a
   105px y ahogaba la matriz. Con min-content el peor caso son los 8 caracteres de DIABETES
   (es/en/de) = 63,4px de texto + 10 de padding = 73,4; el francés y el italiano piden menos y
   dejan que su frase salte por el espacio, que es un corte legítimo.
   Medido sobre las cajas de línea ya pintadas, en IBM Plex Mono 12px con 0,06em de espaciado.
   Ver el presupuesto de ancho del breakpoint, que tuvo que subir por esto. */
.cm-grid {{ display:grid; grid-template-columns:min-content 1fr 1fr; gap:4px; align-items:stretch; }}
/* El encabezado de COLUMNA lleva el mismo overflow-wrap:normal que el de fila, y por el mismo
   motivo: hereda el break-word de Streamlit. Hoy no se le nota —su celda (73px en el ancho
   mínimo) da de sobra para DIABETES, que pide 63,4—, así que se salva por holgura de
   presupuesto y no por regla. Se le pone igualmente: es el mismo defecto, y basta con que
   alguien estreche una celda o alargue una traducción para que aparezca aquí el "DIABET / ES"
   que se acaba de quitar de la otra etiqueta. */
.cm-collabel {{ font-family:{FONT_MONO}; font-size:12px; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
    color:{t['text_muted']}; text-align:center; align-self:end; padding-bottom:7px; line-height:1.45;
    overflow-wrap:normal; }}
/* overflow-wrap:normal es EL arreglo del "DIABET / ES", y va aquí porque lo que se está
   deshaciendo es una herencia: Streamlit declara `overflow-wrap:break-word` más arriba y esa
   propiedad baja sola hasta esta etiqueta. Devuelta a `normal`, el navegador solo puede cortar
   donde el idioma permite —espacios y el <br> de la cadena—, nunca dentro de una palabra.
   No se usa `white-space:nowrap`: eso prohibiría también el corte por el espacio, que sí es
   legítimo y es el que necesitan el francés y el italiano para no llevarse media matriz.
   Emparejado con el `min-content` de la rejilla, la garantía es completa: la columna siempre
   mide al menos la palabra más larga, así que no hay ancho en el que una palabra no quepa. */
.cm-rowlabel {{ font-family:{FONT_MONO}; font-size:12px; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
    color:{t['text_muted']}; text-align:right; align-self:center; padding-right:10px; line-height:1.45;
    overflow-wrap:normal; }}
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
/* El hover ya no cambia solo la tinta: la pestaña recibe además una base apenas teñida y
   redondeada por arriba, que es lo que da la sensación de superficie pulsable. Sin ella, un
   rótulo que solo cambia de color se lee como texto, no como control. */
button[data-baseweb="tab"] {{
    border-radius:8px 8px 0 0 !important;
    transition: color 0.15s ease, background-color 0.18s ease !important;
}}
button[data-baseweb="tab"]:hover {{
    color:{t['text']} !important;
    background-color:{C_PRIMARY}0F !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{C_PRIMARY} !important; }}
/* BaseWeb mueve el subrayado con `left`/`width` en línea; sin transición salta de una pestaña
   a otra de golpe. Con esto se DESLIZA, que es lo que convierte el cambio de pestaña en un
   gesto continuo y no en un corte. La curva es la misma que la de las tarjetas, para que toda
   la página se mueva con el mismo temperamento. */
[data-baseweb="tab-highlight"] {{
    background-color:{C_PRIMARY} !important; height:2px !important;
    border-radius:2px !important;
    transition: left 0.28s cubic-bezier(0.4,0,0.2,1), width 0.28s cubic-bezier(0.4,0,0.2,1) !important;
}}
[data-baseweb="tab-border"] {{ background-color:{t['border']} !important; }}
/* ── Flechas de desplazamiento de la tira de pestañas ──
   Solo aparecen cuando los rótulos no caben, y por eso este fallo se escondía tan bien: en
   escritorio y en español las tres pestañas de Gobernanza entran de sobra. En el teléfono
   —o en un idioma largo, o con la ventana estrecha— Streamlit añade estos botones con un
   degradado de desvanecido BLANCO INCRUSTADO, `linear-gradient(to right, transparent, #FFF)`,
   heredado de su tema base. Sobre el cadete del tema oscuro eso es una banda blanca de 20px
   pegada al canto de la tira, con la flecha en tinta oscura encima.
   El degradado se rehace contra el fondo REAL de la página (t['bg'], que es sobre lo que se
   apoya la tira) y la flecha toma el color de texto secundario, el mismo que los rótulos
   inactivos. Cada botón desvanece hacia SU lado, de ahí las dos direcciones.
   No lo cazó el detector de widgets sin vestir porque busca `background-color` y aquí el
   blanco viaja en un `background-image`.
   El extremo transparente va con el sufijo `00` sobre el propio color y NO con `transparent`:
   la palabra clave equivale a negro con alfa 0, y al interpolar contra el cadete el degradado
   se ensucia por el medio. Tampoco vale hex_to_rgba() aquí — se define más abajo que esta hoja,
   que se evalúa al importar el módulo. Mismo recurso que el filete dorado de la portada. */
button[data-testid="stTabsScrollRight"] {{
    background-image:linear-gradient(to right, {t['bg']}00, {t['bg']} 40%) !important;
}}
button[data-testid="stTabsScrollLeft"] {{
    background-image:linear-gradient(to left, {t['bg']}00, {t['bg']} 40%) !important;
}}
button[data-testid="stTabsScrollRight"], button[data-testid="stTabsScrollLeft"] {{
    color:{t['text_secondary']} !important;
}}
button[data-testid="stTabsScrollRight"] svg, button[data-testid="stTabsScrollLeft"] svg {{
    fill:{t['text_secondary']} !important;
}}
/* ═══════════════ EXPANDER (Gobernanza · Registro de decisiones) ═══════════════
   Único widget nativo que quedaba sin vestir, y en tema oscuro se rompía: config.toml no fija
   `base`, así que Streamlit pinta el expander con su tema CLARO (barra casi blanca), mientras
   que el rótulo hereda el color de .stApp, que en oscuro es platino. Rótulo claro sobre
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
   border_strong y no border: con el tono de borde (#CDD6DF en claro) quedaba casi invisible sobre
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
   interior: el scroll horizontal de .gov-table-wrap se conserva intacto. Gana a la
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
/* ─── Desplegables (selectbox) ────────────────────────────────────────────────
   Mismo fallo que las etiquetas de aquí arriba, pero en el CONTROL entero y sin arreglar hasta
   ahora: de todo el `data-baseweb="select"` la hoja solo tocaba el foco de teclado, así que la
   caja se quedaba con la paleta base de Streamlit —#F0F2F6 de fondo y #31333F de texto— en
   medio de una interfaz oscura. Se veía en las dos páginas que llevan uno: la Esfera de Bloch
   («Variable clínica») y el Predictor («Variable a recorrer»).
   No lo delató el auditor de contraste porque tinta oscura sobre caja clara SÍ pasa WCAG: es
   un fallo de tema, no de legibilidad, y por eso conviene mirarlo también con los ojos.
   Se viste con el mismo lenguaje que los botones: superficie de tarjeta, borde de la paleta y
   radio de 10px. El :hover tira del acento, igual que el resto de controles.
   Los colores se fuerzan también en los DESCENDIENTES y no solo en la caja: BaseWeb reparte
   `color` por varios divs anidados, y sin esto el valor elegido se queda en el gris del tema
   base aunque la caja ya sea oscura. */
div[data-baseweb="select"] > div {{
    background-color:{t['surface']} !important;
    border:1px solid {t['border']} !important;
    border-radius:10px !important;
}}
div[data-baseweb="select"] > div:hover {{ border-color:{C_PRIMARY} !important; }}
div[data-baseweb="select"], div[data-baseweb="select"] div,
div[data-baseweb="select"] input, div[data-baseweb="select"] span {{
    color:{t['text']} !important;
}}
div[data-baseweb="select"] svg {{ fill:{t['text_secondary']} !important; }}
/* El PANEL desplegado vive fuera del control, colgado del <body>, así que hay que vestirlo
   aparte. El :has(ul) no es adorno: `popover` es el mismo envoltorio que usan los tooltips de
   la app, y sin esa condición se llevarían por delante su forma y su color. */
div[data-baseweb="popover"]:has(ul),
div[data-baseweb="popover"]:has(ul) > div,
div[data-baseweb="popover"]:has(ul) ul {{
    background-color:{t['surface']} !important;
    border-radius:10px !important;
}}
div[data-baseweb="popover"]:has(ul) ul {{
    border:1px solid {t['border']} !important; box-shadow:{SHADOW} !important;
}}
div[data-baseweb="popover"]:has(ul) li, div[data-baseweb="popover"]:has(ul) ul div {{
    color:{t['text']} !important;
}}
div[data-baseweb="popover"]:has(ul) li:hover,
div[data-baseweb="popover"]:has(ul) [role="option"]:hover {{
    background-color:{t['surface_alt']} !important;
}}
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
    box-shadow: inset 0 1px 3px rgba(5,6,10,0.20), inset 0 -1px 0 rgba(255,255,255,0.45);
}}
/* Degradado del relleno + brillo, sin tocar el background-image dinámico de BaseWeb.
   Truco: un blanco que se desvanece hacia la derecha. Sobre el tramo RELLENO (izquierda) aclara el
   azul → produce el degradado claro→oscuro hasta el pulgar, como en la referencia. Sobre el tramo
   VACÍO (derecha) el blanco ya es casi transparente → lo deja limpio. Se adapta solo al mover el
   pulgar, porque el degradado va referido al ancho del carril, no al del relleno. */
div[data-baseweb="slider"] > div > div > div:last-child::after {{
    content:""; position:absolute; inset:0; border-radius:999px; pointer-events:none;
    background:
        linear-gradient(to bottom, rgba(255,255,255,0.22), rgba(255,255,255,0) 60%, rgba(5,6,10,0.10)),
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
    box-shadow: 0 1px 3px rgba(5,6,10,0.30), 0 3px 8px rgba(5,6,10,0.16);
}}
/* Muesca central del pulgar (la rayita vertical de la referencia), en color de marca */
div[role="slider"]::after {{
    content:""; position:absolute; left:50%; top:50%;
    width:2px; height:9px; transform:translate(-50%, -50%);
    border-radius:1px; background:{C_PRIMARY};
}}
div[role="slider"]:hover::before {{ border-color:{C_PRIMARY}; }}
div[role="slider"]:active {{ cursor:grabbing !important; }}
/* El valor sobre el pulgar sube para no chocar con los ticks.
   Y se le pone COLOR, que es lo único del deslizador que se había quedado sin vestir: el carril,
   el aro del pulgar y la muesca ya van en C_PRIMARY (ver arriba), pero la cifra la seguía
   pintando el `primaryColor` de config.toml. Ese ajuste es de SERVIDOR, o sea el mismo #1565C0
   en los dos temas: en claro es justo el acento y va perfecto, pero en oscuro cae a 2,44:1
   sobre la tarjeta —el azul del autor está pensado para papel—, y es
   justo el número que estás leyendo mientras arrastras el mando. C_PRIMARY es el acento que SÍ
   puede llevar texto —es literalmente su definición, "cromo de interfaz: navegación, foco,
   sliders"— y de paso ata la cifra al resto del control.
   La regla alcanza también al <p>: Streamlit envuelve el valor en markdown, y el color efectivo
   lo lleva ese párrafo. */
div[data-testid="stSliderThumbValue"] {{ top:-40px; z-index:4; }}
div[data-testid="stSliderThumbValue"],
div[data-testid="stSliderThumbValue"] p {{ color:{C_PRIMARY} !important; }}
/* Y los EXTREMOS del carril (mínimo y máximo), por lo mismo pero con otro origen: estos ni
   siquiera llegaban al primaryColor, se quedaban en el rgba(49,51,63,0.6) del tema BASE de
   Streamlit — que config.toml deja en claro, el mismo fallo del expander y de los mandos de
   Bloch. Fallaban en los DOS temas: 1,46:1 en oscuro y 3,55:1 en claro, una vez resuelto ese
   0,6 de alfa contra el fondo. text_muted es el escalón que les toca —son el marco de la
   escala, no el dato— y va opaco: 5,3:1 en oscuro y 4,5:1 en claro.
   OJO SI ESTO PARECE CÓDIGO MUERTO: la tira nace en opacity:0 y Streamlit solo la revela al
   pasar el cursor o al enfocar el deslizador (`opacity: isHovered ? 1 : var(--slider-focused,
   0)` en su hoja). Una captura estática NO la enseña; para verla hay que poner esa custom
   property a 1, o tener el foco puesto — que es justo cuando se está usando el control. */
div[data-testid="stSliderTickBar"],
div[data-testid="stSliderTickBar"] p {{ color:{t['text_muted']} !important; }}
#MainMenu, footer, header {{ visibility:hidden; }}
/* ...pero el botón nativo para ABRIR la sidebar vive DENTRO de ese <header>: al ocultarlo, en móvil
   (donde la sidebar arranca colapsada) el usuario se quedaba sin forma de abrir el menú y no podía
   navegar. Lo devolvemos a la vida y lo vestimos con la paleta. */
/* Ojo: el elemento con este testid ES el <button>, no lo contiene.
   Aquí SOLO se le devuelve la visibilidad. El aspecto —disco invertido y flecha propia— se lo da
   el bloque de más arriba, el mismo que viste al botón de cerrar, para que los dos sentidos de la
   misma acción se vean igual. Antes se le pintaba aquí una pastilla rectangular de superficie, y
   como esta regla va DESPUÉS en la hoja, ganaba: en el teléfono el mando de abrir salía con otra
   forma y otro color que todos los demás. */
button[data-testid="stExpandSidebarButton"] {{ visibility:visible !important; }}

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
    /* Lo mismo, pero para las filas de st.columns —que .compare-grid no alcanza porque no son una
       rejilla nuestra sino el flex de Streamlit—. Con la barra fija en 270px, al contenido le
       quedan 550px en una tableta vertical y 754 en horizontal; repartidos entre TRES columnas
       salen 159 y 227px, y en ese ancho hay piezas que sencillamente no caben: la matriz de
       confusión pide 74px de etiquetas más dos celdas que no bajan de ~72 (sus cifras son de 22px),
       o sea 218 mínimos, y la gráfica de Plotly otro tanto. Al no caber NO encogían: se salían por
       la derecha de su tarjeta y se metían encima de la de al lado —las tres matrices y las tres
       curvas ROC de Resultados, comprobado a 820 y 1024px—, y en vertical la tercera llegaba a
       salirse de la página.
       La solución es dejarlas saltar de línea con un ancho MÍNIMO por columna, y ese mínimo no
       puede ser el mismo para todas porque no todas piden lo mismo:
         · filas de TRES (curvas ROC y matrices de Resultados) → 266px. Sale de la cuenta de la
           matriz: 218 de rejilla + 46 de tarjeta = 264, y se redondea a 266 para no ir al filo.
           Con eso quedan a una por fila en vertical (510px, porque 2×266 = 532 no entra) y a
           dos en horizontal (754px), que es justo lo contrario de lo que se quería evitar:
           tarjetas que respiran.
           Los 218 subieron desde 209 al hacer `auto` la columna de etiquetas: los 64px fijos
           de antes partían "DIABETES" a media palabra, y los 74 son lo que de verdad mide el
           token más largo (63,4 de texto + 10 de padding). El aire sale de la fila, no de las
           celdas — que se quedan en sus ~72 y no bajan de ahí.
         · filas de CUATRO o más (las cuatro cifras del Resumen, los KPI de Gobernanza y Circuito)
           → 230px, que las deja en un 2x2 limpio en los dos anchos. Con el mínimo de las de tres
           se habrían apilado de una en una: cuatro cifras sueltas a 510px de ancho cada una.
       Las de DOS se quedan como están: ya caben, y un min-width uniforme les habría roto sus
       proporciones ([1.35, 1], [2.2, 1]...), que están elegidas una a una. De ahí el :has(), que
       es lo que permite contar columnas desde el CSS. */
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(3)) {{
        flex-wrap:wrap !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(3)) > div[data-testid="stColumn"] {{
        min-width:266px !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4)) > div[data-testid="stColumn"] {{
        min-width:230px !important;
    }}
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
    /* El zócalo y el pie son position:fixed anclados al ancho de la sidebar: si siguen fijos,
       quedan flotando sobre el contenido cuando la sidebar está cerrada. Los devolvemos al
       flujo del panel, así solo se ven cuando está abierto.
       La FILA se mantiene aquí aunque se venga del modo colapsado: en el teléfono el panel se
       abre a ancho casi completo, así que el motivo por el que en 84 px se apila —que no cabe—
       no existe. Por eso el row-reverse se repite en vez de heredarse: `narrow` es estado de
       escritorio y puede llegar en cualquiera de sus dos valores. */
    .st-key-pie_barra {{
        position:static !important; width:100% !important;
        flex-direction:row-reverse !important;
        justify-content:space-between !important;
        padding:14px 6px 6px !important; margin:0 !important;
    }}
    .sidebar-footer {{
        position:static !important; width:100% !important; margin-top:10px; border-top:none;
    }}
    /* Y con ellas de vuelta en el flujo, el suelo de 166 px que la barra reserva en escritorio
       deja de tener a quién esquivar: sobre el final de la lista ya no flota nada. Se queda en
       un respiro corto, el mismo que separa cualquier bloque del canto. */
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
        padding-bottom:16px !important;
    }}
    /* El desplegable de idioma SÍ sigue fijo en móvil (no es de la sidebar, es del lienzo),
       pero se arrima al borde y encoge un punto. El botón nativo de abrir la sidebar ocupa
       la esquina superior IZQUIERDA, así que no hay colisión posible. Aquí solo cambian las
       medidas: las banderas y sus estados ya están definidos arriba y se heredan. Que se abra
       por toque y no por hover es además lo que hace que el menú funcione en una pantalla
       táctil, donde no existe el estado sobre el que se apoyaba antes.
       Estas reglas no repiten los data-URI —_css_banderas(con_imagen=False)—, que son con
       diferencia lo más pesado de la hoja. */
    {CSS_FLAGS_MOVIL}
    .st-key-lang_{LANG} button::before {{ margin-right:4px !important; }}
    /* El reloj sigue a las banderas a su nueva posición y se queda SOLO con la hora: en un
       teléfono la franja superior es estrecha y la fecha es el dato prescindible de los dos
       —quien mira un reloj de cabecera mira la hora—. Su borde sale de la misma cuenta que
       en escritorio (RELOJ_RIGHT_M), con diez píxeles de aire en vez de doce. */
    #tfm-reloj {{ top:10px; right:{RELOJ_RIGHT_M}px; height:16px; font-size:11px; }}
    #tfm-reloj .r-fecha, #tfm-reloj .r-sep {{ display:none; }}
    /* El disco de «volver arriba» encoge y se arrima al canto: en un teléfono los 36 px de
       escritorio tapan una franja de contenido que allí es la mitad de ancha. */
    #tfm-arriba {{ right:14px; bottom:14px; width:32px; height:32px; }}
    #tfm-arriba::before {{ width:15px; height:15px; }}
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
    /* La tira de tecnologías tampoco cambia de forma aquí: sigue siendo {TECH_POR_FILA}+{TECH_POR_FILA}.
       Ya no hace falta reajustarla por media query —las cqw de .tech-strip-wrap la encogen
       solas y en el teléfono tocan sus suelos—, y volver a tocarla aquí es justo lo que rompía
       el reparto parejo: el auto-fit anterior repartía por ancho y en un móvil salían tres y
       hasta cuatro filas. */
    /* Al apilarse (teléfono) la columna es de ancho completo; el alto ya lo fija la figura Plotly */
    .st-key-bloch_row div[data-testid="stColumn"]:last-of-type div[data-testid="stVerticalBlock"] {{ height:auto !important; }}
    /* Cabecera proporcionada a la pantalla del teléfono */
    .page-eyebrow {{ font-size:12px; letter-spacing:0.13em; margin-bottom:9px; }}
    .page-title {{ font-size:30px; margin-bottom:9px; }}
    .page-subtitle {{ font-size:14.5px; margin-bottom:10px; }}
    .page-rule {{ margin-bottom:22px; }}
    .kpi-card, .info-card {{ padding:16px 16px; }}
    .stat-card {{ min-height:88px !important; }}
    /* La matriz de confusión YA NO lleva override de columnas. Había uno que bajaba las
       etiquetas de 64 a 52px para no ahogar las celdas en pantalla estrecha, y era justo el
       que peor partía las palabras: si a 64px no cabía "DIABETES", a 52 menos. Ahora la
       columna es `auto` en los dos anchos —mide el token más largo y ya está—, y a quien se
       le da el aire que le falta es a la FILA ENTERA, subiendo su min-width (ver allí). */
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
   "al aparecer" sobre ellos nunca llegaría a dispararse — por eso no se ve nada si se aplica ahí.
   Con esta regla se cierra la lista: NINGÚN movimiento de la aplicación consulta ya
   prefers-reduced-motion. La decisión y su porqué están escritos una sola vez, junto a la entrada
   escalonada del contenido; aquí solo se deja constancia de que este era el último. Como el de
   la barra lateral, el keyframe no lleva retardo ni `backwards`, así que sin animación la página
   se ve igual, solo que sin fundido. */
div[class*="st-key-page_enter_"] {{
    animation: pageFadeIn 0.38s ease-out;
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
    prevalencia < 50 % su mediana es 0, y como ninguna de las categorías de RIDRETH1 o
    DMDEDUC2 llega al 50 %, TODAS sus columnas salían a 0. El paciente de referencia
    resultaba imposible: sin etnia y sin nivel educativo. Cada grupo que suma 0 recibe
    entonces un 1 en su MODA — que se lee de la media del scaler, porque para una dummy 0/1
    la media ES la prevalencia de esa categoría.

    LO QUE HAY QUE MIRAR ANTES ES SI EL GRUPO ESTÁ COMPLETO, y es donde fallaba: no todos
    los grupos traen una columna por categoría, así que "todo a cero" no siempre significa
    "falta el dato". Medido sobre el propio scaler:

        RIDRETH1  5 columnas · Σ prevalencias 1,000 · P(ninguna) 0,000  → completo
        DMDEDUC2  6 columnas · Σ 1,000 · P(ninguna) 0,000               → completo
        DMDMARTL  7 columnas · Σ 1,000 · P(ninguna) 0,000               → completo
        RIAGENDR  1 columna  · Σ 0,485 · P(ninguna) 0,515               → INCOMPLETO
        RIDRETH3  2 columnas · Σ 0,167 · P(ninguna) 0,833               → INCOMPLETO

    RIAGENDR viene codificado dejando fuera una categoría (queda solo RIAGENDR_1.0, varón),
    y de RIDRETH3 sobrevivieron al filtro de correlación únicamente las dos categorías que
    RIDRETH1 no tiene —_6.0 asiático no hispano y _7.0 otro/multirracial—, porque el resto
    eran redundantes con ella. En los dos casos el cero ES una categoría, y además LA MÁS
    FRECUENTE: mujer (51,5 %) y "ni asiático ni otro" (83,3 %).

    Rellenarlos a ciegas construía justo el paciente que este código vino a evitar: varón
    —cuando la moda es mujer— y blanco no hispano por RIDRETH1 Y asiático no hispano por
    RIDRETH3 a la vez, que es una combinación que no existe en el conjunto de datos.

    La regla, por tanto, compara la moda explícita contra esa categoría IMPLÍCITA, cuya
    prevalencia es 1 − Σ. Gana la más probable: en los tres grupos completos, P(ninguna) es
    cero y no cambia nada respecto de antes; en los dos incompletos, el vector se queda a
    ceros, que es la respuesta correcta. El efecto medido sobre el perfil por defecto es de
    −0,18 pp en el SVM-RBF (2,53 % → 2,34 %) y nulo en LightGBM, que no ramifica sobre estas
    dummies; en la zona de decisión llega a pesar más (HbA1c 6,5: SVM 29,7 % → 35,9 %).
    """
    feats = list(_features)
    x = np.array(medianas, dtype=np.float64)
    medias = np.array(medias, dtype=np.float64)

    grupos = {}
    for f in feats:
        if "_" in f and f.rsplit("_", 1)[1].replace(".", "").isdigit():
            grupos.setdefault(f.rsplit("_", 1)[0], []).append(f)
    for cols in grupos.values():
        idx = [feats.index(c) for c in cols]
        if x[idx].sum() != 0:                       # el grupo ya tiene su categoría activa
            continue
        p_col = medias[idx]                         # prevalencia de cada categoría explícita
        p_ninguna = 1.0 - float(p_col.sum())        # ...y la de la implícita, "todas a cero"
        j = int(np.argmax(p_col))
        if p_col[j] > p_ninguna:
            x[idx[j]] = 1.0
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
# CONTADOR DE VISITAS
# ─────────────────────────────────────────────────────────────────────────
# DONDE VIVE EL NUMERO, y por que no vive aqui. Esta app se despliega en Streamlit
# Community Cloud (ver README, «Despliegue»), y alli el sistema de ficheros del contenedor
# es EFIMERO: lo que se escriba en disco desaparece en cada redespliegue y cada vez que la
# app se duerme por inactividad y vuelve a levantarse. Un contador en un .json local, que es
# la primera solucion que uno escribe, marcaria un numero que se reinicia solo cada pocos
# dias — y en una defensa, un contador que pone 3 es peor que no tener contador.
#
# Por eso la cuenta se guarda FUERA, en un Gist de GitHub. Es la opcion que menos deuda deja:
# duradero, gratis, sin cuenta nueva que abrir (el repo ya esta en GitHub) y sin dependencia
# nueva que fijar en requirements.txt, porque requests ya viene con Streamlit. El Gist guarda
# un unico fichero, visitas.json, con la forma {"visitas": N}.
#
# QUE CUENTA COMO VISITA. Una SESION de navegador, no una re-ejecucion: Streamlit vuelve a
# correr el script entero en cada clic, asi que contar por pasada daria el numero de clics.
# El guardarraill es st.session_state, que sobrevive a los reruns dentro de una sesion y no
# entre sesiones — exactamente la definicion que se busca. Una recarga de pagina es una
# visita nueva, que es lo que hace cualquier contador de los de toda la vida.
#
# QUE PASA SI FALLA. Nada visible. Un contador es decoracion; que se caiga la red, caduque el
# token o falte el secreto no puede tumbar el panel, asi que TODO va envuelto y el fallo cae
# al contador en memoria. Si no hay secretos configurados —el caso de cualquiera que clone el
# repo y lo levante en local— la app funciona igual y el contador cuenta las visitas de esa
# sesion del servidor. Ver README para dar de alta el Gist y el token.
VISITS_GIST_FICHERO = "visitas.json"
# Cuatro segundos y no el timeout por defecto de requests, que es ninguno: esta llamada esta
# en el camino critico del primer render de cada sesion, y una API que no responde no puede
# dejar el panel en blanco esperandola. Si no contesta en cuatro, se cuenta en memoria.
VISITS_TIMEOUT = 4


def _secreto(clave):
    """Lee st.secrets[clave] sin reventar cuando no hay secrets.toml.

    En local no existe el fichero, y Streamlit no devuelve vacio en ese caso: lanza al
    TOCAR st.secrets, asi que ni siquiera un .get() bastaria. El try va aqui, una vez, y
    no en cada sitio que necesite un secreto.
    """
    try:
        return str(st.secrets[clave]).strip()
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def _visitas_estado():
    """Cuenta compartida por TODAS las sesiones vivas del proceso.

    cache_resource y no session_state porque el objeto tiene que ser el MISMO para todo el
    que entre: es lo que evita una llamada HTTP por rerun y lo que hace que, cuando el Gist
    no esta disponible, las visitas se sigan sumando entre si en vez de empezar de cero cada
    pestana.

    El Lock no es decorativo. Streamlit atiende cada sesion en su propio hilo, asi que dos
    visitas simultaneas pueden leer el mismo N y escribir el mismo N+1, perdiendo una. Como
    Community Cloud corre un solo contenedor, cerrar el ciclo leer-sumar-escribir bajo el
    lock deja la cuenta EXACTA, no aproximada.
    """
    return {"n": 0, "remoto": False, "lock": threading.Lock()}


def _gist_cabeceras(token):
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _gist_leer(gist_id, token):
    r = requests.get(f"https://api.github.com/gists/{gist_id}",
                     headers=_gist_cabeceras(token), timeout=VISITS_TIMEOUT)
    r.raise_for_status()
    return int(json.loads(r.json()["files"][VISITS_GIST_FICHERO]["content"])["visitas"])


def _gist_escribir(gist_id, token, n):
    r = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers=_gist_cabeceras(token), timeout=VISITS_TIMEOUT,
        json={"files": {VISITS_GIST_FICHERO: {"content": json.dumps({"visitas": n})}}})
    r.raise_for_status()


def contar_visita():
    """Suma UNA visita la primera vez que se llama en cada sesion y devuelve el total."""
    estado = _visitas_estado()
    # La bandera se pone ANTES de tocar la red: si la llamada al Gist falla y el usuario
    # sigue pulsando, cada rerun reintentaria la peticion y le colgaria cuatro segundos de
    # timeout en cada clic. Una visita se cuenta una vez, salga bien o salga mal.
    if st.session_state.get("_visita_contada"):
        return estado["n"]
    st.session_state["_visita_contada"] = True

    gist_id, token = _secreto("gist_visitas_id"), _secreto("gist_visitas_token")
    with estado["lock"]:
        if gist_id and token:
            try:
                n = _gist_leer(gist_id, token) + 1
                _gist_escribir(gist_id, token, n)
                estado["n"], estado["remoto"] = n, True
                return n
            except Exception:
                # Se degrada, no se propaga: a partir de aqui la cuenta sigue en memoria
                # desde el ultimo total conocido, que es lo mas cerca de la verdad que hay.
                estado["remoto"] = False
        estado["n"] += 1
        return estado["n"]


def html_contador(n):
    """Chapa de dos cuerpos: un ojo y la cifra. Solo markup; la forma la pone el CSS.

    Al pasar de 99.999 se queda con los cinco ultimos digitos en vez de crecer, que es lo que
    hacia el odometro del que viene la pieza y lo que acota su ancho: vive en una barra de
    84 px cuando esta colapsada.

    SIN separador de millar y sin ceros a la izquierda. Lo primero no es una eleccion de
    estilo: MILLAR se define con el resto del formato numerico mucho mas abajo, y cuando esta
    funcion se llama —al pintar la barra lateral— todavia no existe. A cinco digitos en
    monoespaciada tampoco aporta legibilidad. Lo segundo si es eleccion: un "00042" era el
    gesto del odometro, y esta pieza ya no lo es.

    El rotulo retirado de la vista viaja en los DOS canales: title para el raton y aria-label
    para el lector de pantalla, que ignora el title cuando hay aria-label. Por eso el
    aria-label repite el rotulo corto con la cifra ("visitas: 22149") en vez de la frase larga
    de la ayuda: es lo que se quiere oir, no lo que se quiere leer al pasar por encima.
    """
    cifra = f"{max(0, int(n))}"[-5:]
    return (f'<span class="vc-badge" title="{html.escape(S("visits_help"))}" role="img" '
            f'aria-label="{html.escape(S("visits_label"))}: {cifra}">'
            f'<span class="vc-ojo" aria-hidden="true"></span>'
            f'<span class="vc-num">{cifra}</span>'
            f'</span>')


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    _qml_logo_b64 = _b64_image(str(ASSETS_DIR / "qml_logov2-sidebar.png"))
    _logo_h = "40px" if narrow else "64px"
    # El filete que cierra la cabecera es el border-bottom de este mismo div, así que su
    # margin-bottom ES la distancia entre esa línea y el buscador. Va corto (2px) porque el
    # aire de verdad lo pone el hueco propio del bloque que Streamlit mete detrás; subirlo
    # descuelga el buscador de la línea y lo empuja contra el menú. El reparto completo, con
    # las medidas, está en el CSS bajo "Reparto del aire alrededor del buscador".
    st.markdown(f"""
    <div style="display:flex;justify-content:center;align-items:center;padding:16px 0;margin-bottom:2px;border-bottom:1px solid {t['border']};">
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
    #
    # Y como la identidad es una CLAVE ESTABLE, la página puede viajar en la query string
    # igual que el idioma: ?page=shap sobrevive a F5 —que abre sesión nueva y estado vacío,
    # ver el bloque de ?lang más arriba— y funciona además como enlace directo a una página
    # concreta, en cualquier idioma. Se valida contra i18n.PAGE_KEYS por el mismo motivo que
    # allí: el parámetro lo puede escribir cualquiera en la barra de direcciones, y un
    # ?page=xx desconocido reventaría el .index() del menú; lo que no se reconoce cae a la
    # primera página en vez de propagarse.
    _MENU_OPTIONS = S("nav")
    if "page" not in st.session_state:
        _page_url = st.query_params.get("page")
        # Una clave RETIRADA no cae al Resumen: se traduce a donde vive hoy su contenido, con su
        # pestaña ya abierta (ver i18n.PAGES_RETIRADAS). Es el caso de ?page=bloch, que se pudo
        # compartir cuando la Esfera de Bloch era una entrada del menú. La posición se escribe en
        # la clave PROPIA de tabs_i18n y no en ?tab=, así que no se pisa con el parámetro y no
        # depende de que este bloque corra antes o después del saneo de la URL.
        if _page_url in i18n.PAGES_RETIRADAS:
            _destino, _grupo, _pos = i18n.PAGES_RETIRADAS[_page_url]
            st.session_state.page = _destino
            st.session_state[_POS_TAB.format(_grupo)] = _pos
        else:
            st.session_state.page = (_page_url if _page_url in i18n.PAGE_KEYS
                                     else i18n.PAGE_KEYS[0])

    # Qué rama del árbol está desplegada. Arranca en la página activa —el índice tiene que decir
    # dónde estás desde el primer dibujo, y esa rama es además la única cuyas secciones están de
    # verdad a un scroll de distancia— y a partir de ahí la mueve el clic. Es estado PROPIO y no
    # se deduce de `page` porque puede no haber ninguna abierta: plegar la de la página en la que
    # estás es un estado legítimo.
    if "nav_open" not in st.session_state:
        st.session_state.nav_open = st.session_state.page

    # Sin tooltip, pero con nombre accesible. El globito con "Expandir"/"Colapsar" repetía en
    # palabras lo que la flecha ya dice —apunta siempre al lado al que se moverá la barra—,
    # así que se va; lo que no puede irse es el nombre del botón, porque un <button> sin más
    # contenido que un dibujo no se anuncia como nada.
    #
    # st.button no acepta aria-label y el help de Streamlit tampoco servía: se traduce en un
    # aria-describedby que solo existe mientras el globito está abierto, o sea una DESCRIPCIÓN
    # ocasional, nunca el nombre. Así que el nombre se construye con el propio rótulo: la
    # palabra viaja dentro en cursiva —el único envoltorio que el markdown de st.button deja
    # crear— y el CSS la recorta. El ojo ve la flecha; el lector de pantalla lee la frase.
    #
    # El rótulo ES SOLO esa cursiva recortada: la flecha ya no es un carácter dentro del texto
    # (antes iba un «‹» / «›» delante) sino el ::before de FLECHA_TOGGLE. Un pseudoelemento no
    # entra en el árbol de accesibilidad, que es justo lo que se quiere de un adorno; y así el
    # dibujo no depende de que la fuente traiga el glifo.
    if narrow:
        if st.button(f"*{S('sidebar_expand')}*", key="toggle_sidebar"):
            st.session_state.sidebar_narrow = False
            st.rerun()
    else:
        if st.button(f"*{S('sidebar_collapse')}*", key="toggle_sidebar"):
            st.session_state.sidebar_narrow = True
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
            st.rerun()
    else:
        def _ir_a_resultado(_pagina, _tab=None):
            """Navega al resultado (ver _navegar) y además vacía la caja de búsqueda.

            Va como CALLBACK y no como código tras el `if st.button(...)`: el valor de un
            widget solo puede tocarse desde un callback (fuera lanza StreamlitAPIException),
            y sin vaciar la caja la lista de resultados seguiría abierta empujando el menú
            hacia abajo después de haber navegado.

            `_tab` llega solo en las secciones que viven DENTRO de una pestaña (hoy las de la
            Esfera de Bloch, en Circuito Cuántico) y es (grupo, posición).
            """
            _navegar(_pagina, _tab)
            st.session_state.nav_search = ""

        _q = (st.text_input(S("search_label"), key="nav_search", placeholder=S("search_ph"),
                            label_visibility="collapsed") or "").strip()
        if _q:
            _hits = i18n.search(_q, LANG)
            with st.container(key="nav_search_results"):
                for _i, _h in enumerate(_hits):
                    # El rótulo de la página va en el tooltip y no en el propio botón: en
                    # 270 px de ancho, "Curva de respuesta · Predictor en Vivo" se parte en
                    # dos líneas y la lista deja de leerse como lista.
                    st.button(_h["label"], key=f"nav_hit_{_i}", width="stretch",
                              on_click=_ir_a_resultado, args=(_h["page"], _h.get("tab")),
                              help=None if _h["kind"] == 0 else S("search_in").format(
                                  p=_MENU_OPTIONS[i18n.PAGE_KEYS.index(_h["page"])]))
                if not _hits:
                    st.markdown(f'<div class="search-none">{S("search_none")}</div>',
                                unsafe_allow_html=True)
                # Enlaces en crudo y no st.link_button: aquí hace falta target="_blank"
                # explícito (la consulta se abre FUERA, no reemplazando el panel) y
                # rel="noopener" para no ceder window.opener a la fuente. Los nombres de los
                # destinos van sin escapar por ser constantes nuestras (SEARCH_SOURCES, arriba);
                # el rótulo sí se escapa, porque lleva dentro lo que el usuario ha tecleado.
                _fuentes = "".join(
                    f'<a class="search-src" target="_blank" rel="noopener noreferrer" '
                    f'href="{_url.format(q=quote_plus(_q))}">{_nombre}'
                    f'<span class="search-web-ext">↗</span></a>'
                    for _nombre, _url in SEARCH_SOURCES)
                st.markdown(
                    f'<div class="search-web">{html.escape(S("search_web").format(q=_q))}</div>'
                    f'<div class="search-srcs">{_fuentes}</div>', unsafe_allow_html=True)

    # ── MENÚ Y ÁRBOL DE SECCIONES ───────────────────────────────
    # La navegación entera de la app en UNA sola lista: las seis páginas y, colgando de cada
    # una, sus pestañas y sus secciones. Al pasar el ratón por una página se abre su rama; al
    # pulsar cualquier fila se va allí. La apertura es CSS puro (ver el bloque "Menú y árbol de
    # secciones" de la hoja de estilos); aquí solo están la estructura y el destino de cada fila.
    #
    # EL MENÚ FUE UN streamlit-option-menu, y ahora son botones nativos. El cambio no es de
    # gusto: ese componente se pinta en un iframe de altura fija y su API es una lista PLANA, así
    # que no había manera de colgarle las secciones —una rama abierta ahí dentro se recorta y,
    # sobre todo, no puede empujar a los ítems de debajo— ni de alcanzarlo con el CSS del
    # documento. Mientras fueron dos piezas, los seis nombres de página salían dos veces en la
    # barra: una como menú y otra como raíz del índice.
    #
    # Lo que se fue con el componente, que era todo deuda del iframe:
    #   · menu_force_index / manual_select: una SEGUNDA copia de "qué página está activa" que
    #     había que empujarle a mano desde cada sitio que navegaba, porque el componente no leía
    #     session_state. Y con su propia trampa —era un disparo de un solo uso que, reenviado en
    #     cada pasada, competía con el clic del usuario y dejaba el menú oscilando—.
    #   · la key con el tema, el idioma y el modo dentro (main_menu_dark_False_es): el componente
    #     solo leía su dict `styles` al montarse, así que cada cambio de esos tres obligaba a un
    #     remount completo para que se reaplicara.
    #   · el fundido que disimulaba ese remount, y las reglas sueltas de fondo, ancho y hover del
    #     iframe. El aspecto del menú vive ahora en la hoja de estilos, junto al del resto.
    #
    # Cada fila es un st.button y no un enlace: navegar aquí es escribir en session_state (la
    # página, la posición de la pestaña, el destino del scroll), o sea trabajo del servidor. Un
    # <a href="?page=…"> recargaría la aplicación entera y perdería la sesión.
    #
    # Y `page` se lee de session_state ANTES de pintar, en vez de salir de lo que devolvía el
    # componente: los callbacks corren al principio de la pasada que dispara el clic, así que
    # cuando esto se ejecuta el estado ya es el de la página de destino. Esa es también la razón
    # de que la lista pueda marcar su fila activa sin preguntarle nada a nadie.
    page = st.session_state.page

    # Las reglas que dependen del ESTADO de la barra —cuál es la página activa y cuál es la rama
    # desplegada—. Van en un <style> propio, aquí, y no en la hoja de arriba, porque aquella se
    # escribe antes de que ese estado se resuelva.
    #
    # La rama abierta la decide el CLIC, no el ratón por encima. Con :hover el despliegue era
    # CSS puro y no costaba una pasada de servidor, pero abría y cerraba ramas al cruzar la barra
    # camino de otra cosa: el índice se movía solo mientras leías, y la lista de debajo saltaba
    # con él. Al clic solo cambia cuando se le pide; el precio es llevar la cuenta a mano
    # (`nav_open`), porque el estado ya no lo guarda el puntero.
    #
    # Sigue habiendo UNA sola rama abierta a la vez, y por el mismo motivo que con el hover: con
    # dos, la lista se iba a 949 px en una pantalla de portátil —más que la ventana entera— y la
    # mitad de abajo quedaba fuera de alcance.
    _abierta = st.session_state.nav_open
    # Se arma fuera del f-string para no tener que escapar las llaves del CSS dentro de otra
    # interpolación. Vacío cuando no hay ninguna desplegada, que es un estado posible: la regla
    # simplemente no se emite y todas las ramas se quedan en el max-height:0 de la hoja general.
    _css_rama = ""
    if _abierta:
        _css_rama = (
            f'.st-key-navb_{_abierta} div[class*="st-key-navk_"] '
            f'{{ max-height:620px !important; opacity:1 !important; }}\n    '
            # Solo la imagen de la máscara: el tamaño, el color y la transición ya están puestos
            # en la hoja general y no tienen por qué repetirse (ni poder desincronizarse).
            f'.st-key-navb_{_abierta} div[class*="st-key-navp_"] button::after '
            f'{{ -webkit-mask-image:{SIGNO_MENOS} !important; mask-image:{SIGNO_MENOS} !important; }}'
        )
    st.markdown(f"""
    <style>
    /* La fila de la página activa, marcada con el mismo filete de marca que llevaba el ítem
       seleccionado del menú. El icono sube a C_DARK: como los seis van en C_PRIMARY, ese color
       ya no distingue al elegido, y C_DARK se separa en la dirección correcta en cada tema
       —más brillante en oscuro (#6AB5FF), más profundo en claro (#00479F)—. Contraste sobre el
       fondo del ítem activo: 7,88:1 y 6,71:1. */
    .st-key-navb_{page} div[class*="st-key-navp_"] button {{
        border-left-color:{C_PRIMARY} !important; background:{t['sidebar_active']} !important;
    }}
    .st-key-navb_{page} div[class*="st-key-navp_"] button p {{
        color:{t['text']} !important; font-weight:600 !important;
    }}
    .st-key-navb_{page} div[class*="st-key-navp_"] button span[data-testid="stIconMaterial"] {{
        color:{C_DARK} !important;
    }}
    /* Y su signo sube con el rótulo, para que la fila entera se lea como un bloque y no como un
       nombre encendido con un adorno apagado al lado. */
    .st-key-navb_{page} div[class*="st-key-navp_"] button::after {{
        background-color:{t['text_secondary']} !important;
    }}
    /* La rama desplegada. Es UNA clase más que el max-height:0 de la hoja general, así que gana
       sin necesitar el !important; lo lleva igualmente porque la de la hoja también lo lleva. */
    {_css_rama}
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="nav_tree"):
        for _pag in i18n.nav_tree(LANG):
            _pk = _pag["page"]
            # El icono es un Material Symbol y se declara en i18n.PAGES junto a la clave de la
            # página. Es la misma familia que ya usa el buscador colapsado, que es su vecino de
            # barra, y la que Streamlit trae de serie: no hay que servir ni mantener un SVG.
            _icono = f":material/{_pag['icon']}:"
            # El contenedor de la rama es el ancestro común de la fila de página y de su lista,
            # y por tanto lo que hace de zona de hover. Sin él no habría manera de que la lista
            # siguiera abierta mientras el ratón está sobre ella.
            with st.container(key=f"navb_{_pk}"):
                if narrow:
                    # En 84 px no cabe el rótulo, y mucho menos una rama de 14 filas: queda el
                    # icono, igual que en el buscador. El nombre viaja dentro de todos modos, en
                    # cursiva, y el CSS lo recorta sin borrarlo del árbol de accesibilidad
                    # —mismo patrón que el toggle de la barra, donde está razonado—; el `help`
                    # pone además el globito, de modo que puntero y lector de pantalla coinciden.
                    st.button(f"*{_pag['label']}*", key=f"navp_{_pk}", icon=_icono,
                              width="stretch", help=_pag["label"],
                              on_click=_navegar, args=(_pk,))   # sin rama que plegar
                    continue
                st.button(_pag["label"], key=f"navp_{_pk}", icon=_icono, width="stretch",
                          on_click=_navegar_pagina, args=(_pk,))
                with st.container(key=f"navk_{_pk}"):
                    for _rama in _pag["ramas"]:
                        # Las páginas sin pestañas devuelven una rama sin rótulo: sus secciones
                        # cuelgan directamente de la página, sin inventarse un nivel que en la
                        # página no existe.
                        if _rama["label"] is not None:
                            st.button(_rama["label"], key=f"navt_{_pk}_{_rama['tab'][1]}",
                                      width="stretch", on_click=_navegar,
                                      args=(_pk, _rama["tab"]))
                        for _sec in _rama["secciones"]:
                            # La clave lleva dentro la clave de la sección, que es única en todo
                            # el catálogo; el destino que viaja al callback es su RÓTULO, porque
                            # quien lo busca es el navegador entre los .section-title del
                            # documento.
                            st.button(_sec["label"], key=f"navs_{_sec['key']}",
                                      width="stretch", on_click=_navegar,
                                      args=(_pk, _rama["tab"], _sec["label"]))

    # La URL se pone al día con la página activa, y se hace AQUÍ —en el único punto por el que
    # pasa toda la navegación— y no en cada sitio que navega (el menú, el buscador, los enlaces internos):
    # todos esos caminos terminan pasando por esta línea, así que un único punto de escritura
    # basta. Escribir una query param NO relanza el script, solo actualiza la barra de
    # direcciones, de modo que esto no se pelea con el rerun del propio clic.
    #
    # La condición replica la de ?lang: hasta que no se navega, la URL se queda limpia; una vez
    # que el parámetro existe se mantiene siempre —volver a Resumen lo deja en ?page=overview y
    # no lo borra—, porque una URL sin parámetro significa "la primera página" y eso desharía la
    # navegación en la recarga.
    if page != i18n.PAGE_KEYS[0] or "page" in st.query_params:
        if st.query_params.get("page") != page:
            st.query_params["page"] = page
    # ?tab= pertenece a la página que se está viendo (ver tabs_i18n), así que en una sin
    # pestañas no significa nada y se borra en vez de quedarse colgando en la barra de
    # direcciones. Va ANTES del cuerpo de las páginas, o sea antes de que tabs_i18n lo lea:
    # justamente en las que sí tienen tabs no se toca.
    if page not in _PAGINAS_CON_TABS and "tab" in st.query_params:
        del st.query_params["tab"]
    # Índice de la página, que viaja DENTRO del nombre de los keyframes de la entrada
    # escalonada (ver el bloque más abajo): al cambiar de página cambia el nombre y las
    # animaciones reinician; dentro de la misma página el nombre no cambia y no se repiten.
    _page_idx_anim = i18n.PAGE_KEYS.index(page)
    st.markdown(f"""
    <style>
    /* ═══════════════ ENTRADA ESCALONADA DEL CONTENIDO ═══════════════
       El contenido no aparece de golpe: sube unos píxeles y se revela en cascada corta,
       cabecera primero y bloques de datos después. Es lo que hace que una página se lea
       como compuesta en vez de volcada, y de paso guía la mirada en el orden correcto.

       EL TRUCO ESTÁ EN EL NOMBRE. Streamlit reutiliza los nodos entre reruns, así que una
       animación "al aparecer" no vuelve a dispararse nunca (ya está razonado en header()).
       Pero una animación SÍ reinicia si le cambia el `animation-name`, y este bloque de
       estilo se regenera en cada rerun: metiendo el índice de página en el nombre del
       keyframe, la cascada se dispara justo cuando cambias de página y NO cuando mueves un
       slider o alternas el tema dentro de la misma. Que es exactamente el criterio que se
       quiere; repetirla en cada rerun sería mareante.

       Se aplica a clases NUESTRAS y no a los data-testid de Streamlit: son las que este
       fichero controla, no cambian con la versión del framework, y así ninguna animación
       puede quedarse colgada de un contenedor interno que un día se renombre.

       backwards es obligatorio: sin él, un bloque con retardo se vería opaco durante ese
       retardo y luego parpadearía a cero para entrar.

       ESTA CASCADA YA NO VIVE DENTRO DE prefers-reduced-motion, y es deliberado. Es la segunda
       excepción de la aplicación, hermana de la del contador y por el mismo motivo: el equipo
       desde el que se trabaja este panel lleva los efectos de animación de Windows apagados
       —SystemParametersInfo(SPI_GETCLIENTAREAANIMATION) = 0—, así que sus navegadores piden
       reduce siempre y la entrada escalonada no se veía NUNCA en escritorio, solo en el móvil y
       en la tableta. Entregadas ya la memoria y la defensa, esto es acabado visual y se ha
       elegido que se vea donde se mira la página. Revertirlo es volver a envolver el bloque en
       @media (prefers-reduced-motion: no-preference).

       Ojo a lo que eso cambia en el `backwards`: antes, quien pedía menos movimiento no recibía
       ni la animación ni el estado inicial, y veía la página quieta y completa. Ahora la recibe
       todo el mundo, así que durante el retardo el bloque está en el `from` del keyframe —opaco
       y 12 px más abajo—. Es el comportamiento buscado, pero implica que estos elementos
       dependen de que la animación llegue a correr para hacerse visibles: si algún día se toca
       el nombre del keyframe o se rompe su declaración, no se quedan sin animar, se quedan sin
       verse. El keyframe se emite tres líneas más abajo, en este mismo <style>. */
    .page-eyebrow, .page-title, .page-subtitle, .page-rule,
    .section-title, .section-sub, .lead-card, .clinical-note,
    .kpi-card, .info-card, .stat-card {{
        animation: tfmEnter{_page_idx_anim} 0.42s cubic-bezier(0.22,1,0.36,1) backwards;
    }}
    .page-title    {{ animation-delay:0.06s; }}
    .page-subtitle {{ animation-delay:0.11s; }}
    .page-rule     {{ animation-delay:0.15s; }}
    .section-title, .section-sub {{ animation-delay:0.17s; }}
    .lead-card, .clinical-note, .kpi-card, .info-card, .stat-card {{ animation-delay:0.20s; }}
    /* El filete del título de sección se dibuja solo, de izquierda a derecha. Es el
       gesto editorial de la página —la regla que cierra el titular— hecho visible. */
    .section-title::after {{
        animation: tfmRule{_page_idx_anim} 0.6s cubic-bezier(0.22,1,0.36,1) 0.22s backwards;
    }}
    @keyframes tfmEnter{_page_idx_anim} {{
        from {{ opacity:0; transform:translateY(12px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes tfmRule{_page_idx_anim} {{
        from {{ transform:scaleX(0); opacity:0; }}
        to   {{ transform:scaleX(1); opacity:1; }}
    }}
    </style>
    """, unsafe_allow_html=True)


    # El contador y el interruptor van en el MISMO contenedor porque son UNA FILA: quien la
    # reparte —interruptor arrimado a la esquina izquierda, chapa al canto derecho, las dos a
    # la misma altura— es .st-key-pie_barra en la hoja de estilos.
    # El contador va ANTES que el interruptor en el codigo, y eso sigue importando aunque en
    # la barra ancha no se note: alli la fila es row-reverse y el contador sale a la derecha
    # de todas formas, pero en la barra colapsada y en el telefono la fila se APILA, y alli el
    # orden que manda es este. Escrito al reves, en el telefono el contador saldria DEBAJO del
    # interruptor.
    with st.container(key="pie_barra"):
        st.markdown(html_contador(contar_visita()), unsafe_allow_html=True)

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
# Los botones se pintan en el lienzo principal (no en la sidebar) y el CSS los lleva a la
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
# una fila en el flujo (el height:0 solo aplica a los .st-key-lang_*) y empujaría el titular
# de la página hacia abajo. Sueltos, no ocupan nada y el CSS los coloca.
#
# En bucle sobre i18n.LANGS y no un `if` por idioma: los cinco botones son el mismo botón
# con otra clave, y escritos a mano el tercero se habría copiado del segundo con la
# comparación del segundo — el fallo silencioso de un selector de idioma, que deja una
# bandera muerta.
#
# El contenedor NO es decorativo y no puede quitarse: es el ancestro común de las cinco
# banderas, y de él cuelga la condición de «desplegable abierto» (.st-key-lang_switch a secas
# cuando lo está, y :has(button:focus-visible) para el teclado). Sin un ancestro propio habría
# que preguntarle al bloque vertical de Streamlit, que envuelve la página entera. No ocupa
# sitio: la hoja de estilos lo disuelve con display:contents junto con los envoltorios que
# Streamlit mete dentro.
#
# La bandera ACTIVA no cambia de idioma —ya es el idioma—, así que su clic es el que pliega y
# despliega el panel. Las otras cuatro eligen idioma y lo cierran de paso: un menú que se queda
# abierto después de elegir tapa la esquina de la página que acabas de traducir.
#
# El st.rerun() es obligatorio también para el simple abrir/cerrar, y no un lujo: la hoja de
# estilos se escribe ARRIBA del script, mucho antes de estos botones, de modo que la pasada en
# la que se pulsa ya ha emitido el CSS con el estado viejo. Sin el rerun el panel no se movería
# hasta la siguiente interacción.
with st.container(key="lang_switch"):
    for _lang in i18n.LANGS:
        if st.button(" ", key=f"lang_{_lang}", help=S(f"lang_{_lang}_help")):
            if _lang == LANG:
                st.session_state.lang_open = not MENU_ABIERTO
            else:
                st.session_state.lang = _lang
                st.session_state.lang_open = False
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
  var loc = {json.dumps({"es": "es-ES", "en": "en-GB", "de": "de-DE",
                          "fr": "fr-FR", "it": "it-IT"}[LANG])};
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

  // ── Fundido al bajar ──────────────────────────────────────────────────────
  // El reloj es fijo y el contenido le pasa POR DEBAJO, así que en cuanto la página baja se
  // cruza con el texto y quedan las dos cosas superpuestas. Se apaga con el scroll en lugar de
  // apartarlo: sigue donde se le espera mientras estás arriba, y deja de existir en cuanto
  // empiezas a leer. Vale para las siete páginas porque cuelga del contenedor de scroll, no de
  // nada que dependa de la página que haya dentro.
  // Quien hace scroll es section[data-testid="stMain"] y no la ventana (mismo motivo que se
  // explica en scrollbar-gutter): en window.scrollY no pasa NUNCA nada. Se deja la ventana
  // como alternativa por si en móvil el que se mueve es el documento.
  var main = doc.querySelector('section[data-testid="stMain"]');
  function desplazamiento() {{ return main ? main.scrollTop : (window.parent.scrollY || 0); }}

  // La curva es cúbica y no una rampa lineal: gasta la opacidad deprisa al principio y luego
  // se arrastra, que es como se lee un fundido en vez de un interruptor. VENTANA son los px de
  // scroll que dura; el exponente, lo brusco que arranca. Son los dos únicos números que tocar.
  // 140 y no los 64 de la primera versión, que resultaron demasiado secos al verlos en marcha.
  // El precio está medido y es real: el contenido arranca a 3rem (48px) y el reloj cierra en
  // y=32, así que el primer texto lo alcanza a los ~16px de scroll, y a esa altura la curva aún
  // va por el 70%. O sea que las primeras líneas cruzan el reloj todavía legible — es el trato
  // que hay que hacer para que el fundido se note, porque con solo 16px de margen no caben las
  // dos cosas. Bajando VENTANA se recupera limpieza y se pierde suavidad, y al revés.
  var VENTANA = 140;
  function opacidad() {{
    var k = Math.max(0, 1 - desplazamiento() / VENTANA);
    caja.style.opacity = (k * k * k).toFixed(3);
  }}

  // Un oyente por rerun y no uno más cada vez: igual que el setInterval de arriba, el iframe se
  // recarga en cada rerun y volvería a suscribirse sobre el mismo elemento. Se guarda también
  // el nodo al que se enganchó, porque es a ESE al que hay que quitárselo.
  if (window.parent.__tfmRelojFadeEl && window.parent.__tfmRelojFade) {{
    window.parent.__tfmRelojFadeEl.removeEventListener("scroll", window.parent.__tfmRelojFade);
  }}
  // rAF: el scroll dispara muchos más eventos que fotogramas tiene la pantalla, y sin esto se
  // recalcularía la opacidad varias veces para pintarla una.
  var pendiente = false;
  var alDesplazar = function () {{
    if (pendiente) {{ return; }}
    pendiente = true;
    window.parent.requestAnimationFrame(function () {{ pendiente = false; opacidad(); }});
  }};
  (main || window.parent).addEventListener("scroll", alDesplazar, {{ passive: true }});
  window.parent.__tfmRelojFade = alDesplazar;
  window.parent.__tfmRelojFadeEl = main || window.parent;
  // Se aplica ya: un rerun (cambiar de página, mover un slider) puede llegar con la página a
  // media altura, y sin esto el reloj reaparecería opaco hasta el siguiente scroll.
  opacidad();
}})();
</script>""",
        height=0, width=0,
    )

# ── NAVEGACIÓN DENTRO DE LA PÁGINA ───────────────────────────────────────────
# Las dos piezas que mueven el scroll sin recargar nada: el disco de «volver arriba» y el salto
# a la sección que se ha pulsado en el árbol de la barra lateral. Van juntas porque comparten lo
# único que tiene miga —el recorrido suave y el contenedor que de verdad hace scroll—, y
# separarlas obligaría a repetir las dos cosas en dos sitios.
#
# Misma vía que el reloj y que el atributo lang: el iframe de components.html se sirve por
# srcdoc, comparte origen y puede escribir en el documento padre. El aspecto vive en la hoja de
# estilos (#tfm-arriba) y aquí solo está el comportamiento.
#
# NINGUNA de las dos es un st.button, y no por capricho: un botón de Streamlit dispara un rerun
# del script entero para hacer algo que ocurre por completo en el navegador. Serían dos segundos
# de recarga, y la página volviendo a montarse, para mover una barra de scroll.
# El destino del salto se consume con pop() y no se lee: es un disparo de un solo uso. Dejado
# en session_state, la página volvería a saltar a la misma sección en cada rerun posterior —o
# sea cada vez que se moviera un slider— sin que nadie lo hubiera pedido.
_nav_scroll = st.session_state.pop("nav_scroll", None)
# Mismo trato de disparo único que el destino del scroll, y por el mismo motivo: dejado puesto,
# el panel se cerraría solo en cada rerun posterior —cada slider, cada pestaña— sin que nadie lo
# haya pedido.
_nav_cerrar = st.session_state.pop("nav_cerrar", False)

with st.container(key="nav_js"):
    components.html(
        f"""<script>
(function () {{
  var W = window.parent, doc = W.document;

  // ── DÓNDE CORRE ESTE CÓDIGO, Y POR QUÉ NO AQUÍ ──────────────────────────────
  // Nada de lo de abajo se ejecuta dentro de este iframe: se inyecta como <script> en el
  // DOCUMENTO PADRE y corre allí. Parece un rodeo y es justo lo contrario — sin él, la mitad
  // de esto no funciona.
  //
  // El motivo es que Streamlit REEMPLAZA el iframe de un componente cada vez que su contenido
  // cambia, y con él se lleva por delante todo lo que vivía en su contexto de JavaScript:
  // funciones, closures, callbacks de requestAnimationFrame y de setTimeout ya programados. No
  // dan error, simplemente dejan de correr. Y no es un caso raro: navegar a otra página dispara
  // DOS pasadas —la del clic y la que devuelve el menú al recibir su manual_select—, o sea que
  // el iframe se renueva un segundo después de cada salto. El recorrido hasta la sección se
  // quedaba congelado a mitad de camino, con la página parada donde le pillara.
  //
  // Inyectado en el padre, el código vive donde viven el botón y el contenedor que hace scroll,
  // y ya no depende de que su iframe siga existiendo. La función se pasa a texto con toString(),
  // así que se sigue escribiendo y leyendo como una función normal en vez de como una cadena; lo
  // único que hay que respetar es que NO puede tomar nada de este ámbito — todo lo que necesita
  // del servidor entra por CFG.
  function comportamiento(CFG) {{
    var W = window, doc = document;   // aquí ya SON los del documento padre

    // El nodo se crea UNA vez y se reutiliza: cuelga de <body>, fuera del árbol que Streamlit
    // reconstruye, así que sobrevive a los reruns y a los cambios de página. Lo único que se
    // reescribe en cada pasada es el rótulo, que es lo que le hace seguir al idioma.
    var btn = doc.getElementById("tfm-arriba");
    var nuevo = !btn;
    if (nuevo) {{
      btn = doc.createElement("button");
      btn.id = "tfm-arriba";
      btn.type = "button";
      doc.body.appendChild(btn);
    }}
    // El nombre accesible va en aria-label: el dibujo es un ::before, y un <button> cuyo único
    // contenido es un adorno no se anuncia como nada. El title pone además el globito del ratón,
    // con el mismo texto, para que puntero y lector de pantalla digan lo mismo.
    var ROTULO = CFG.rotulo;
    btn.setAttribute("aria-label", ROTULO);
    btn.title = ROTULO;

    // Quien hace scroll es section[data-testid="stMain"] y no la ventana (en W.scrollY no pasa
    // nada nunca; mismo motivo que se explica en scrollbar-gutter). Se REBUSCA en cada uso en vez
    // de guardarlo: Streamlit lo reconstruye al cambiar de página y una referencia guardada al
    // arrancar quedaría apuntando a un huérfano — la misma lección que el el(sel) del parallax.
    function contenedor() {{ return doc.querySelector('section[data-testid="stMain"]'); }}

    // ── A dónde sube ──
    // SIEMPRE al principio de la página, con pestañas o sin ellas.
    //
    // La primera versión afinaba: en una página con pestañas paraba en la tira de rótulos, para
    // dejar a la vista en qué pestaña estabas. El resultado es que el mismo botón hacía dos cosas
    // distintas según dónde se pulsara —en Resultados subía del todo y en Gobernanza se quedaba a
    // media altura—, y un control que no aterriza siempre en el mismo sitio obliga a mirar dónde
    // has caído. Un destino único no hay que comprobarlo.
    //
    // Y no se pierde nada por el camino: el titular de la página está arriba del todo, y la tira
    // de pestañas queda igualmente en pantalla, unos 280 px por debajo del borde.

    // ── La subida ──
    // A mano con requestAnimationFrame y no con scrollTo({{behavior:"smooth"}}), que sería la vía
    // corta: el desplazamiento suave NATIVO lo cancelan Chrome y Firefox cuando el sistema pide
    // reducir movimiento, y ahí el botón daría un corte seco en vez de un recorrido. Es la misma
    // decisión ya tomada para el parallax de portada (ver la nota junto a .ov-anim): esto no es
    // un adorno que se pueda perder, es lo que deja ver CUÁNTO se ha subido y desde dónde — sin
    // el recorrido, aterrizar arriba del todo se confunde con haber cambiado de página.
    function abortar() {{
      if (W.__tfmArribaAnim) {{ W.cancelAnimationFrame(W.__tfmArribaAnim); W.__tfmArribaAnim = null; }}
    }}
    function subir(main, hasta) {{
      var desde = main.scrollTop, delta = hasta - desde;
      if (Math.abs(delta) < 2) {{ main.scrollTop = hasta; return; }}
      // Duración proporcional pero acotada por los dos lados: con velocidad fija una página de
      // 4.000 px se haría eterna, y con duración fija un salto de 300 px se ve como un tirón.
      var dur = Math.min(620, Math.max(240, Math.abs(delta) * 0.45));
      var t0 = null;
      abortar();
      function paso(t) {{
        if (t0 === null) {{ t0 = t; }}
        var k = Math.min(1, (t - t0) / dur);
        // easeOutCubic: sale rápido y frena al llegar, que es como se lee un recorrido CON
        // destino. Una rampa lineal parece un ascensor.
        main.scrollTop = desde + delta * (1 - Math.pow(1 - k, 3));
        W.__tfmArribaAnim = k < 1 ? W.requestAnimationFrame(paso) : null;
      }}
      W.__tfmArribaAnim = W.requestAnimationFrame(paso);
    }}

    // ── Cuándo se ve ──
    // A partir de tres cuartos de pantalla de recorrido. Antes de eso el principio sigue a la
    // vista y el botón no ahorraría nada; y se mide contra el ALTO DEL CONTENEDOR y no contra un
    // número fijo de píxeles, para que el criterio sea el mismo en un portátil y en un teléfono.
    // De rebote, en una página que no llega a esa altura el botón no aparece nunca, que es lo
    // correcto: no hay a dónde volver.
    var UMBRAL = 0.75;
    function repinta() {{
      var main = contenedor();
      if (!main) {{ return; }}
      btn.classList.toggle("visible", main.scrollTop > main.clientHeight * UMBRAL);
    }}

    if (nuevo) {{
      btn.addEventListener("click", function () {{
        var main = contenedor();
        if (main) {{ subir(main, 0); }}
      }});
      // Si durante el recorrido se toca la rueda o la pantalla, manda quien la toca: seguir
      // arrastrando el scroll contra el gesto del usuario es de lo más molesto que puede hacer
      // un control como este.
      //
      // Y no basta con parar la animación en curso: el salto a una sección se pasa varios
      // segundos VIGILANDO que el destino siga en su sitio (ver más abajo), así que también hay
      // que darlo por abandonado. Sin esto, apartarse a mirar otra cosa mientras la página
      // termina de dibujarse acababa con el panel devolviéndote a la sección de un tirón.
      var rendirse = function () {{ abortar(); W.__tfmSaltoVivo = false; }};
      doc.addEventListener("wheel", rendirse, {{ passive: true }});
      doc.addEventListener("touchstart", rendirse, {{ passive: true }});
      // El oyente de scroll va sobre el DOCUMENTO y en fase de CAPTURA, no sobre stMain: los
      // eventos de scroll no burbujean pero sí se capturan, y el documento no lo reconstruye
      // nadie. Así este oyente se pone una sola vez en la vida de la página, en lugar de tener
      // que quitarse y volver a ponerse en cada rerun sobre el stMain de turno — que es de donde
      // salen los oyentes acumulados y los que se quedan hablándole a un huérfano.
      var pendiente = false;
      doc.addEventListener("scroll", function () {{
        // rAF: el scroll dispara muchos más eventos que fotogramas tiene la pantalla, y sin esto
        // se recalcularía el estado varias veces para pintarlo una.
        if (pendiente) {{ return; }}
        pendiente = true;
        W.requestAnimationFrame(function () {{ pendiente = false; repinta(); }});
      }}, {{ capture: true, passive: true }});
    }}

    // Se aplica ya: un rerun puede llegar con la página a media altura (mover un slider) o de
    // vuelta arriba del todo (cambiar de página), y sin esto el botón se quedaría como estaba.
    repinta();

    // ── SALTO A UNA SECCIÓN ──────────────────────────────────────────────────────
    // El destino lo pone _navegar() en session_state al pulsar una fila del árbol de la barra
    // lateral, y llega aquí como el RÓTULO del título. No como un id: los .section-title se
    // pintan sin él —son 34 st.markdown repartidos por el fichero—, y el rótulo ya es único
    // dentro de su página, que es todo lo que hace falta para encontrarlo.
    //
    // Se busca DENTRO del panel de pestaña visible cuando lo hay: Streamlit renderiza las tres
    // pestañas de Gobernanza aunque solo enseñe una, así que sin acotar se podría medir un
    // título que está en el DOM pero en display:none — y ahí getBoundingClientRect() devuelve
    // ceros, o sea un salto al principio de la página.
    // ── El panel se cierra solo tras navegar, y SOLO en el teléfono ──
    // Ahí la barra es un overlay que tapa la página entera (ver la media query de ≤768), así que
    // dejarla abierta después de un salto es esconder justo lo que se acaba de pedir. En
    // escritorio y en tableta la barra es una columna que no tapa nada y cerrarla sería perder
    // el menú por las buenas: de ahí el umbral, que es el mismo de la hoja de estilos.
    // Se comprueba aria-expanded para no pulsar el botón cuando el panel ya está cerrado —lo
    // alternaría, abriéndolo—. Y si el botón no estuviera donde se espera, esto no hace nada:
    // se queda como hasta ahora, con el panel abierto.
    if (CFG.cerrar && W.innerWidth <= 768) {{
      var barra = doc.querySelector('section[data-testid="stSidebar"]');
      var cerrar = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                || doc.querySelector('[data-testid="stSidebarCollapseButton"]');
      if (barra && cerrar && barra.getAttribute("aria-expanded") !== "false") {{ cerrar.click(); }}
    }}

    var DESTINO = CFG.destino;
    var AIRE = 24;   // el título aterriza con aire por encima y no pegado al canto: pegado se
                     // lee como cortado, y además el reloj de cabecera ocupa esa franja.
    if (DESTINO) {{
      // Colocar el título no es medir una vez y saltar: es SEGUIRLO hasta que se está quieto.
      //
      // Y hay dos razones distintas para ello, que se resuelven con el mismo bucle. La primera
      // es que cuando este script corre el título todavía no existe —el componente se emite
      // antes que el cuerpo de las páginas—, así que hay que esperarlo. La segunda es más
      // sutil y es la que rompía la versión anterior, que medía una sola vez en cuanto lo
      // encontraba: el título aparece antes que las gráficas que tiene ENCIMA, y cada Plotly
      // que termina de dibujarse lo empuja unos cientos de píxeles hacia abajo. Se aterrizaba
      // en la coordenada correcta de una página que ya no existía; con dos gráficas por medio,
      // el destino acababa muy por debajo del borde inferior.
      //
      // El bucle se rinde cuando lleva DOS comprobaciones seguidas con el título en su sitio
      // —una sola no distingue "ya está" de "aún no ha llegado la siguiente gráfica"— y, en
      // cualquier caso, al agotar el plazo. Acotarlo es obligatorio: si el destino no llega a
      // existir nunca (una sección condicional que hoy no se pinta), esto tiene que dejar la
      // página quieta y callarse, no vigilar el DOM para siempre.
      var pasadas = 0, estables = 0;
      W.__tfmSaltoVivo = true;
      // Se rebusca el título en CADA pasada en vez de guardarse el nodo: Streamlit reemplaza
      // elementos entre reruns, y sobre un nodo desconectado getBoundingClientRect() devuelve
      // ceros — o sea, un salto silencioso al principio de la página.
      var localizar = function () {{
        var vistos = Array.prototype.filter.call(
          doc.querySelectorAll(".section-title"),
          // offsetParent descarta lo que está en el DOM pero oculto: Streamlit renderiza las
          // tres pestañas de Gobernanza aunque solo enseñe una, y un título en display:none
          // tampoco se puede medir.
          function (el) {{ return el.textContent.trim() === DESTINO && el.offsetParent !== null; }});
        return vistos.length ? vistos[0] : null;
      }};
      var ajustar = function () {{
        if (!W.__tfmSaltoVivo) {{ return; }}
        var main = contenedor(), titulo = localizar();
        if (main && titulo) {{
          var delta = titulo.getBoundingClientRect().top - main.getBoundingClientRect().top - AIRE;
          // La tolerancia absorbe el medio píxel de los bordes y los 12 px de la cascada de
          // entrada (tfmEnter), que arranca el bloque desplazado y lo devuelve a su sitio.
          //
          // El segundo caso es el de las últimas secciones de una página: no hay contenido
          // debajo suficiente para subirlas hasta el borde, así que la barra llega a su tope
          // y ahí se acaba el viaje. Sin comprobarlo, el bucle se pasaba sus seis segundos
          // pidiendo una y otra vez un destino que el navegador ya no puede dar.
          var tope = main.scrollHeight - main.clientHeight;
          if (Math.abs(delta) <= 3 || (delta > 0 && main.scrollTop >= tope - 1)) {{ estables++; }}
          else {{ estables = 0; subir(main, Math.max(0, main.scrollTop + delta)); }}
        }}
        if (estables < 2 && ++pasadas < 24) {{ W.setTimeout(ajustar, 250); }}
      }};
      W.requestAnimationFrame(ajustar);
    }}
  }}

  // El <script> se retira del DOM inmediatamente: ya se ha ejecutado al insertarlo, y dejarlo
  // puesto solo acumularía un nodo muerto por rerun.
  var sc = doc.createElement("script");
  sc.textContent = "(" + comportamiento + ")(" + {json.dumps(json.dumps({"rotulo": S("scroll_top"), "destino": _nav_scroll, "cerrar": _nav_cerrar}))} + ");";
  doc.head.appendChild(sc);
  sc.remove();
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
    superficie de debajo, no el color a secas. Con una paleta toda oscura daría igual
    —el blanco valdría siempre—, pero con esta no: en tema oscuro la serie de LightGBM
    es #F1F5F8 y la de QSVM #00ACC1, así que una tinta blanca fija sería blanco sobre
    blanco. Aquí se compone la mezcla y se elige entre el noche y el blanco el que más
    contraste dé, de modo que la regla sigue siendo correcta si mañana se vuelve a
    cambiar la paleta — y de hecho ya ha sobrevivido a TRES cambios enteros sin tocarse:
    lo único que cambió fue el hex de la tinta oscura, que se lee de la paleta.

    Devuelve (tinta, tinta_atenuada) para el número y su etiqueta.
    """
    mix = tuple(f * alpha + b * (1 - alpha)
                for f, b in zip(_hex_rgb(hex_color), _hex_rgb(surface)))
    lum = _rel_luminance(mix)
    if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / (_rel_luminance(_hex_rgb(P_NOCHE)) + 0.05)):
        return "#FFFFFF", "rgba(255,255,255,0.78)"
    return P_NOCHE, hex_to_rgba(P_NOCHE, 0.72)

# ── Separadores numéricos de cada idioma: (millar, decimal) ──
# Cuatro de las cinco lenguas escriben la coma como separador DECIMAL, pero NO comparten
# el de MILLAR: el francés lo pone con un espacio inseparable ("1 567"), no con un punto.
# Por eso esto es un mapa de parejas y no el booleano que bastaba mientras solo convivían
# el español y el inglés — con aquel, el francés habría dado "1.567", que en Francia se
# lee como uno coma quinientos sesenta y siete y no como el número de instancias del test.
#
# Lo consultan las tres funciones de abajo más el `separators` de Plotly. Centralizado
# aquí, añadir una lengua es añadir su fila; escrito como `LANG == "es"` en cada sitio,
# habría que acertar en cinco a la vez y el que se quedara atrás daría "0.9190" en una
# tarjeta y "0,9190" en la de al lado.
#
# El espacio francés es U+00A0 y no uno normal porque partir la línea entre el 1 y el 567
# rompería la cifra en dos. Se usa el inseparable ANCHO y no el fino de U+202F —que es lo
# que prescribe la tipografía francesa moderna— porque el fino falta en bastantes fuentes,
# y donde falta se pinta un rectángulo vacío justo dentro del número.
SEPARADORES = {
    "es": (".", ","),
    "en": (",", "."),
    "de": (".", ","),
    "fr": ("\u00a0", ","),   # escapado y no el carácter literal: se vería como un espacio normal
    "it": (".", ","),
}
MILLAR, DECIMAL = SEPARADORES[LANG]

# El signo de porcentaje va pegado a la cifra SOLO en inglés. En las otras cuatro lleva
# espacio: lo exige la norma del SI, y además la DIN 5008 en alemán y la tipografía
# francesa. El italiano corriente lo omite a menudo, pero esto es una memoria científica
# y la notación manda sobre el uso coloquial.
PCT_ESPACIO = "" if LANG == "en" else " "

def nf(x, dec=4):
    """Formato numérico según el idioma activo.

    Español, alemán e italiano: coma decimal y punto de millar ('1.234,5678'). Francés:
    coma decimal y espacio de millar ('1 234,5678'). Inglés: la convención inversa
    ('1,234.5678'), que es la que Python ya produce de fábrica. Traducir el texto y dejar
    las cifras a la española sería un error tan visible como no traducir: en una memoria
    científica el separador forma parte de la notación, no de la maquetación.

    La sustitución es un translate y no dos replace encadenados: translate recorre la
    cadena UNA vez y mapea cada carácter de forma independiente, así que la coma no puede
    convertirse en punto y ese punto volver a convertirse en coma en la segunda pasada.
    """
    s = f"{x:,.{dec}f}"                       # formato US: '1,234.5678'
    return s.translate(str.maketrans({",": MILLAR, ".": DECIMAL}))

def pct(x, dec=1):
    """Porcentaje en el idioma activo: '78,2 %' en cuatro lenguas, '78.2%' en inglés."""
    return f"{nf(x * 100, dec)}{PCT_ESPACIO}%"

def mil(n):
    """Entero con separador de millar en el idioma activo: '29.400' · '29,400' · '29 400'.

    Existe porque la app pinta cifras enteras (registros, filas descartadas) por media
    docena de sitios con un `.replace(",", ".")` a mano. Ese replace es correcto en
    español y erróneo en las otras cuatro, así que la decisión se centraliza aquí.
    """
    return f"{n:,}".replace(",", MILLAR)

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
        # Separadores de los ticks y del hover, en el idioma activo. Plotly los quiere en
        # una cadena de DOS caracteres y en el orden contrario al del mapa: primero el
        # decimal, después el de millar. Estaba fijo en ",." y las gráficas seguían dando
        # "0,95" con la app en inglés, justo donde la cifra manda.
        separators=DECIMAL + MILLAR,
        # barcornerradius es de layout (no de traza): redondea el extremo de dato de TODAS
        # las barras de la figura, verticales y horizontales.
        barcornerradius=4,
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border_strong"], align="left",
                         font=dict(family=PLOTLY_FONT, size=12, color=t["text"])),
        margin=margin, **kwargs,
    )
    return fig


def esfera_base() -> go.Figure:
    """Figura con la esfera unidad ya puesta: superficie tenue + tres círculos máximos.

    La comparten las DOS esferas de la página: la de Bloch de un qubit y la Q-sphere del
    estado conjunto de dos. Que se parezcan no es casualidad sino requisito — tienen que
    leerse como el mismo objeto visto a dos escalas—, y con el bloque escrito dos veces
    bastaba tocar un `opacity` en una para que dejaran de parecerlo. Aquí la definición
    vive una sola vez y la coincidencia se sostiene sola.

    La esfera es el CONTENEDOR, no el dato: va en la rampa muy diluida para no competir
    con lo que se dibuja encima (el vector |ψ⟩ o los nodos de la Q-sphere), y el
    `lighting` le da volumen de bola real en vez de aspecto de malla. Los tres círculos
    máximos —ecuador XY, meridianos XZ e YZ— refuerzan la curvatura al rotarla.
    """
    fig = go.Figure()
    u, w = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    xs, ys, zs = np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w)
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs, opacity=0.14, showscale=False, hoverinfo="skip",
        colorscale=[[0, RAMP[0]], [1, RAMP[2]]],
        lighting=dict(ambient=0.66, diffuse=0.9, specular=0.22, roughness=0.6, fresnel=0.25),
        lightposition=dict(x=120, y=200, z=160),
    ))
    circ = np.linspace(0, 2 * np.pi, 120)
    for gx, gy, gz in [
        (np.cos(circ), np.sin(circ), np.zeros_like(circ)),   # ecuador (plano XY)
        (np.cos(circ), np.zeros_like(circ), np.sin(circ)),   # meridiano XZ
        (np.zeros_like(circ), np.cos(circ), np.sin(circ)),   # meridiano YZ
    ]:
        fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", opacity=0.42,
                                    line=dict(color=C_MID1, width=1.2), showlegend=False, hoverinfo="skip"))
    return fig

# ═══════════════════════════════════════════════════════════════════════
# ENTRELAZAMIENTO DE TRES QUBITS  (sección final de la página Esfera de Bloch)
# ═══════════════════════════════════════════════════════════════════════
# POR QUÉ ESTO ES NUMPY Y NO QISKIT — la pregunta obvia en un TFM que entrena con Qiskit.
# Qiskit vive en el PIPELINE, no en el panel: está en requirements.txt de la raíz (junto a
# qiskit-machine-learning, pylatexenc y matplotlib, todo lo que necesitan los notebooks de
# Databricks) y NO en streamlit/requirements.txt, que es el entorno que se despliega. Por eso
# el circuito de 8 qubits de la página Circuito Cuántico entra como PNG ya renderizado desde
# el notebook, y por eso app.py no importa qiskit en ninguna línea. qiskit-aer no está en
# ninguno de los dos ficheros.
#
# Traer qiskit + qiskit-aer + matplotlib solo para esta sección rompería esa separación por
# dos motivos, y el segundo pesa más que el primero:
#   · TAMAÑO — varios cientos de MB añadidos a un despliegue de Streamlit Cloud, para un
#     estado de TRES qubits: ocho amplitudes y tres matrices 8x8.
#   · TEMA — plot_state_qsphere() y plot_histogram() devuelven figuras de MATPLOTLIB: fondo
#     blanco fijo, tipografía ajena y ciegas al tema. En una app que se pinta entera en Plotly
#     y alterna claro/oscuro, esas dos figuras serían los únicos recuadros que no cambian.
#
# Lo que sigue no es una aproximación: es el mismo álgebra lineal que ejecutaría Statevector,
# escrita a mano porque a esta escala cabe entera en pantalla y se puede leer. El dibujo
# reutiliza el vocabulario visual de la esfera de Bloch de arriba (superficie tenue, círculos
# máximos, acento cuántico), que es lo que hace que las dos figuras se lean como una sola idea.

# Orden de la base: el índice es 4·q0 + 2·q1 + q2, con q0 a la IZQUIERDA del ket. Es la
# convención de libro de texto, no la de Qiskit —que numera al revés y escribe |q2 q1 q0⟩—.
# Para el estado GHZ las dos coinciden en la etiqueta ("000" y "111" salen igual leídas por
# cualquier lado), pero se deja dicho aquí porque en los pasos intermedios SÍ se separan: tras
# la Hadamard sobre q0 este código da "100" donde Qiskit escribiría "001".
ENT_N = 3
# Las ocho etiquetas se GENERAN del índice en binario en vez de escribirse a mano: así la
# lista y la convención de arriba no pueden discrepar. Escritas a mano, un solo par cambiado
# de sitio daría un histograma con las barras bien altas y mal rotuladas, que es el tipo de
# fallo que nadie ve.
ENT_BASE = tuple(format(i, f"0{ENT_N}b") for i in range(2 ** ENT_N))

_H1 = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)
# CNOT de dos qubits: intercambia |10⟩ y |11⟩ (índices 2 y 3) y deja |00⟩ y |01⟩ intactos —
# el objetivo se voltea solo cuando el control vale 1.
_CNOT1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype=float)
# Cada puerta se sube al espacio de los tres qubits con productos de Kronecker, y el ORDEN de
# los factores es lo que fija la convención de arriba: el factor izquierdo es el qubit más
# significativo del índice. Se levantan así, y no con la maquinaria tensorial de la sección
# del ZZFeatureMap, porque a esta escala son matrices 8×8 que caben enteras en pantalla: la
# estructura del circuito —quién actúa sobre quién y quién se queda al margen— está a la
# vista en los propios kron.
ENT_H0 = np.kron(_H1, np.eye(4))              # H sobre q0; q1 y q2 sin tocar
ENT_CNOT01 = np.kron(_CNOT1, np.eye(2))       # control q0 → objetivo q1; q2 al margen
ENT_CNOT12 = np.kron(np.eye(2), _CNOT1)       # control q1 → objetivo q2; q0 al margen
# El circuito, en orden. La tupla ES la secuencia: el paso n son las n primeras puertas, y
# añadir una cuarta sería añadirla aquí y en el SVG, sin tocar la lógica.
ENT_PUERTAS = (ENT_H0, ENT_CNOT01, ENT_CNOT12)


def ent_statevector(paso: int) -> np.ndarray:
    """Vector de estado de los tres qubits tras `paso` puertas, partiendo de |000⟩.

    Se RECALCULA entero desde |000⟩ en cada rerun en vez de guardarse en session_state y
    mutarse: el estado que persiste es solo el entero `paso`, así que no hay forma de que
    el vector se desincronice del circuito dibujado ni de que un rerun a mitad de camino
    (cambio de tema, de idioma) lo deje aplicado dos veces.
    """
    psi = np.zeros(2 ** ENT_N)
    psi[0] = 1.0
    for puerta in ENT_PUERTAS[:paso]:
        psi = puerta @ psi
    return psi


def _rho_reducida(psi: np.ndarray, ejes) -> np.ndarray:
    """Matriz densidad de los qubits `ejes`, trazando fuera todos los demás.

    `psi` llega como TENSOR de forma (2,)·n, un eje por qubit, y no como vector plano: así
    "trazar fuera el resto" es literalmente reordenar ejes y aplanar, sin aritmética de bits.
    Con los ejes pedidos delante, la matriz M de forma (2^k, resto) cumple ρ = M·M†, que es el
    atajo estándar de la traza parcial sobre un estado puro.

    La usan las DOS secciones de esta página —la de 2 qubits y la del ZZFeatureMap de 8—, que
    es el motivo de que esté aquí fuera: la fórmula es la misma y no debe existir por duplicado.
    """
    n = psi.ndim
    otros = [k for k in range(n) if k not in ejes]
    M = np.transpose(psi, list(ejes) + otros).reshape(2 ** len(ejes), -1)
    return M @ M.conj().T


def _entropia_vn(rho: np.ndarray) -> float:
    """Entropía de von Neumann en BITS (log base 2), no en nats.

    En bits porque la unidad tiene lectura directa: 1 bit es exactamente el entrelazamiento
    de un par de Bell, que es el patrón con el que se compara todo lo demás en esta página.
    Los autovalores nulos se filtran: 0·log0 vale 0 por continuidad, pero log(0) es -inf.
    """
    lam = np.linalg.eigvalsh(rho)
    lam = lam[lam > 1e-12]
    return abs(float(-np.sum(lam * np.log2(lam)))) if lam.size else 0.0


def _concurrencia(rho: np.ndarray) -> float:
    """Concurrencia de Wootters de un par de qubits MEZCLADO: 0 separable, 1 máximo.

    Hace falta una medida distinta de la entropía, y no es un capricho de notación: la ρ del
    par q₀q₁ llega mezclada —es lo que queda tras trazar fuera q₂— y la entropía de
    entrelazamiento solo mide pares en estado PURO. Aplicada a una mezcla contaría además la
    ignorancia clásica sobre el qubit que se ha trazado, y marcaría entrelazamiento donde solo
    hay correlación: exactamente la confusión que esta sección quiere deshacer.

    La receta es la de Wootters, que para dos qubits da la respuesta EXACTA y no una cota: se
    voltea el estado con ρ̃ = (σy⊗σy)·ρ*·(σy⊗σy), se ordenan de mayor a menor las raíces de los
    autovalores de ρ·ρ̃ y C = max(0, λ₁ − λ₂ − λ₃ − λ₄). Ese producto no es hermítico, pero sus
    autovalores son reales y no negativos; el clip a cero es contra el redondeo, que los saca a
    -1e-17 cuando valen 0, y np.sqrt de un negativo daría nan.
    """
    sy = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sy, sy)
    lam = np.sqrt(np.clip(np.real(np.linalg.eigvals(rho @ yy @ rho.conj() @ yy)), 0.0, None))
    lam = np.sort(lam)[::-1]
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def ent_local(psi: np.ndarray) -> dict:
    """Lo que queda de q₀ cuando se ignora al resto, y lo que queda del PAR q₀q₁ sin q₂.

    ESTA es la evidencia dura de que un estado entrelazado no cabe en tres esferas de Bloch.
    Al trazar fuera los otros qubits queda la matriz densidad reducida ρ₀, y de ella salen tres
    cifras que dicen lo mismo desde tres ángulos:

      · |r| — longitud del vector de Bloch de q0. Vale 1 mientras el estado sea puro (la
        flecha llega a la superficie: hay un punto que dibujar) y 0 en el estado de Bell.
        Cero no es "apunta a otro sitio": es que NO HAY flecha, el vector se ha quedado en el
        centro y ningún punto de la esfera describe a ese qubit por separado.
      · Pureza Tr(ρ₀²) — 1 en un estado puro, 0,5 en la mezcla máxima de un qubit.
      · Entropía de entrelazamiento — la de von Neumann de ρ₀, en bits. 0 si q₀ es separable
        del resto, 1 cuando está entrelazado al máximo con ellos.

    Las tres son redundantes a propósito (|r| = √(2·pureza − 1) es exacta, no aproximada):
    quien viene del lado clínico lee la longitud, quien viene del cuántico lee la entropía.

    La CUARTA es la que solo tiene sentido habiendo un tercer qubit, y es la razón de que la
    sección llegue hasta tres: la concurrencia del par q₀q₁ tras trazar fuera q₂. Su lectura a
    lo largo del recorrido es 0 → 0 → 1 → 0, y ese último cero es el dato de la sección. En el
    paso 2 el par ES un estado de Bell —q₂ mira desde fuera, sin entrar—; el tercer CNOT lo
    entrelaza con los otros dos y, al hacerlo, DESHACE el lazo del par: quedan correlacionados
    (medir uno predice el otro) pero ya no entrelazados. El entrelazamiento del GHZ es global,
    no la suma de lazos entre parejas, y es lo que hay que tener en la cabeza para leer la
    matriz de información mutua del ZZFeatureMap de ocho qubits que viene justo debajo.
    """
    tensor = psi.reshape((2,) * ENT_N)
    rho0 = _rho_reducida(tensor, [0])
    pureza = float(np.real(np.trace(rho0 @ rho0)))
    # El max(0, ...) no es paranoia gratuita: en el estado de Bell la pureza sale 0,4999...
    # por redondeo de coma flotante y el radicando se va a -1e-16, que da nan.
    r = float(np.sqrt(max(0.0, 2.0 * pureza - 1.0)))
    return dict(pureza=pureza, r=r, entropia=_entropia_vn(rho0),
                concurrencia=_concurrencia(_rho_reducida(tensor, [0, 1])))


def ent_qsphere_fig(psi: np.ndarray):
    """Q-sphere del estado CONJUNTO: un nodo por estado base, no una flecha por qubit.

    La Q-sphere existe justamente porque la esfera de Bloch no escala: con tres qubits ya no
    hay tres flechas que dibujar, hay un solo estado en un espacio de ocho dimensiones. La
    convención es la de Qiskit: la LATITUD la fija el peso de Hamming del estado base —|000⟩
    en el polo norte, |111⟩ en el sur, y los pesos 1 y 2 en dos anillos intermedios— y el ÁREA
    del nodo es proporcional a su probabilidad. Así el paso de |000⟩ a GHZ se ve como lo que
    es: un único nodo en el polo que se parte en dos, uno en cada polo, y los dos anillos de
    en medio vacíos.

    Se dibuja con el mismo repertorio que la esfera de Bloch de esta página (superficie tenue
    con lighting, tres círculos máximos, acento cuántico para el dato) para que las dos
    figuras se lean como el mismo objeto visto a dos escalas. Ese repertorio compartido es
    literalmente el mismo código: sale de esfera_base().
    """
    fig = esfera_base()

    # Un anillo por peso de Hamming. Con 3 qubits son cuatro: {000} en el polo norte, los tres
    # de peso 1 y los tres de peso 2 en dos anillos intermedios (a z = ⅓ y −⅓, no en el
    # ecuador: con n impar ningún peso cae justo en la mitad) y {111} en el polo sur. El
    # reparto azimutal dentro del anillo es uniforme, así que los tríos salen a 120°.
    por_peso = {}
    for idx, etiqueta in enumerate(ENT_BASE):
        por_peso.setdefault(etiqueta.count("1"), []).append((idx, etiqueta))

    for peso, miembros in por_peso.items():
        z = 1.0 - 2.0 * peso / ENT_N         # polo norte peso 0, polo sur peso n
        radio = float(np.sqrt(max(0.0, 1.0 - z * z)))
        for j, (idx, etiqueta) in enumerate(miembros):
            phi = 2 * np.pi * j / len(miembros)
            x, y = radio * np.cos(phi), radio * np.sin(phi)
            amp = float(psi[idx])
            prob = amp ** 2
            # Los nodos de amplitud nula NO se dibujan, igual que hace Qiskit: pintarlos de
            # tamaño cero deja un punto residual que se confunde con un estado poco probable,
            # que es justo la lectura contraria a la que se busca ("01 y 10 no salen NUNCA").
            if prob < 1e-9:
                continue
            # Radio del nodo desde el centro hasta su punto en la superficie: el "rayo" que
            # ata cada amplitud a su estado base. En tinta apagada — es soporte, no dato.
            fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines", opacity=0.55,
                                        line=dict(color=C_MID1, width=2), showlegend=False, hoverinfo="skip"))
            # ÁREA proporcional a la probabilidad ⇒ diámetro proporcional a su raíz. Es la
            # convención de la Q-sphere y no un ajuste estético: con el diámetro proporcional
            # a la probabilidad, un estado al 50 % se vería como la cuarta parte de uno al
            # 100 % en vez de como la mitad.
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z], mode="markers",
                marker=dict(size=34 * np.sqrt(prob), color=C_QUANTUM,
                            line=dict(color=t["surface"], width=2)),
                showlegend=False,
                hovertemplate=(f"|{etiqueta}⟩<br>{S('bl_ent_hover_amp')} {nf(amp, 3)}"
                               f"<br>{S('bl_ent_hover_prob')} {pct(prob)}<extra></extra>")))
            # La etiqueta se aparta del nodo hacia AFUERA (radio 1,22) en vez de pegarse a él:
            # con el nodo del polo a tamaño máximo, un textposition relativo la metía dentro
            # del propio marcador.
            fig.add_trace(go.Scatter3d(
                x=[x * 1.22 if radio else 0.0], y=[y * 1.22 if radio else 0.0], z=[z * 1.22],
                mode="text", text=[f"|{etiqueta}⟩"],
                textfont=dict(family=PLOTLY_MONO, size=13, color=t["text_secondary"]),
                showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        height=430, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False, range=[-1.30, 1.30]),
            yaxis=dict(visible=False, range=[-1.30, 1.30]),
            zaxis=dict(visible=False, range=[-1.34, 1.34]),
            aspectmode="cube", dragmode="orbit",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.75)),
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# EL ZZFEATUREMAP REAL (8 qubits) — tercer escalón de la página Esfera de Bloch
# ═══════════════════════════════════════════════════════════════════════
# Aquí se deja el ejemplo de libro y se mide EL CIRCUITO DEL TFM. La estructura no es una
# reconstrucción de oído: sale de notebook_06_qsvm.ipynb, celda 6, que instancia la clase de
# Qiskit tal cual —ZZFeatureMap(feature_dimension=8, reps=2, entanglement="linear")— y cuya
# salida impresa confirma los tres parámetros. Cada repetición es:
#     H en los 8  →  P(2·xᵢ) en cada qubit  →  por cada par adyacente (i,j):
#                                              CX(i,j) · P(2·(π−xᵢ)(π−xⱼ)) en j · CX(i,j)
# y el bloque entero se repite reps=2 veces (la capa de H entra en la repetición: es como
# PauliFeatureMap cuenta las suyas).
#
# POR QUÉ NO SE PINTAN AQUÍ LA Q-SPHERE NI EL HISTOGRAMA. Con 8 qubits el estado tiene 256
# amplitudes, y la Q-sphere las repartiría en anillos de 1·8·28·56·70·56·28·8·1 nodos: 70 solo
# en el ecuador, ilegible. Pero el motivo de fondo es peor que el tamaño: tras H⊗8 las 256
# amplitudes son IDÉNTICAS (comprobado: valen 1/16 exacto) y todo lo que viene después —las
# P y los bloques CX·P·CX— es diagonal, o sea que mueve FASES y nunca magnitudes. La Q-sphere
# dimensiona por probabilidad y el histograma mide en la base computacional, que destruye la
# fase: las dos figuras mirarían justo donde este circuito no guarda nada. Las dos magnitudes
# de abajo sí leen la fase, porque salen de la matriz densidad reducida.
ZZ_N = len(QSVM_FEATURES)                        # 8 qubits = 8 features, uno por variable
ZZ_REPS = 2
ZZ_PARES = [(i, i + 1) for i in range(ZZ_N - 1)]  # entanglement="linear": solo vecinos
ZZ_MI_MAX = 2.0    # tope teórico de la información mutua entre dos qubits, en bits


def _zz_h(psi, k):
    return np.moveaxis(np.tensordot(_H1, psi, axes=([1], [k])), 0, k)


def _zz_p(psi, k, theta):
    """P(θ) sobre el qubit k: diag(1, e^{iθ}). Solo toca la mitad del eje k que vale 1."""
    psi = psi.copy()
    idx = [slice(None)] * ZZ_N
    idx[k] = 1
    psi[tuple(idx)] *= np.exp(1j * theta)
    return psi


def _zz_cx(psi, c, t):
    """CNOT: voltea el eje t allí donde el eje c vale 1.

    El índice del eje objetivo se corrige cuando t > c porque al indexar el eje c con un
    entero ese eje DESAPARECE del bloque y todos los posteriores se corren uno. Verificado
    contra la matriz de permutación explícita para todos los pares con n = 2, 3 y 4.
    """
    psi = psi.copy()
    idx = [slice(None)] * ZZ_N
    idx[c] = 1
    psi[tuple(idx)] = np.flip(psi[tuple(idx)], axis=t - 1 if t > c else t)
    return psi


def zz_statevector(x) -> np.ndarray:
    """Estado de los 8 qubits tras el ZZFeatureMap sobre el vector de features ESCALADO.

    Se trabaja con el estado como tensor (2,)·8 y no como vector de 256: cada puerta es
    entonces una operación sobre un eje, sin construir ni una sola matriz 256×256. La pasada
    completa cuesta unos 0,8 ms, que es lo que permite recalcularla en vivo al mover el
    deslizador en vez de precomputar una tabla.
    """
    psi = np.zeros((2,) * ZZ_N, dtype=complex)
    psi[(0,) * ZZ_N] = 1.0
    for _ in range(ZZ_REPS):
        for k in range(ZZ_N):
            psi = _zz_h(psi, k)
        for k in range(ZZ_N):
            psi = _zz_p(psi, k, 2.0 * x[k])
        for i, j in ZZ_PARES:
            psi = _zz_cx(psi, i, j)
            psi = _zz_p(psi, j, 2.0 * (np.pi - x[i]) * (np.pi - x[j]))
            psi = _zz_cx(psi, i, j)
    return psi


@st.cache_data
def zz_metricas(valores: tuple) -> tuple:
    """|r| de cada qubit y matriz 8×8 de información mutua, a partir de las 8 cifras CLÍNICAS.

    El escalado se hace AQUÍ dentro y no en quien llama: el ZZFeatureMap del TFM no recibe
    mg/dL ni años, recibe la salida del StandardScaler ajustado en la capa Gold (celda de
    notebook_03), que es el mismo scaler_correcto.json que ya usa el Predictor en Vivo. Meter
    valores crudos daría ángulos de fase disparatados y una figura que no es la del modelo.

    Se devuelven las dos magnitudes juntas porque comparten el estado, que es lo caro de
    calcular: pedirlas por separado lo evaluaría dos veces.

    · |r| — longitud del vector de Bloch de cada qubit por separado. Es LA MISMA cifra que
      explica la sección de 2 qubits de más arriba, y por eso esta figura se puede leer: 1 =
      el qubit conserva su estado propio, 0 = se lo ha comido el entrelazamiento.
    · Información mutua I(i:j) = S(ρᵢ) + S(ρⱼ) − S(ρᵢⱼ), en bits. Mide cuánta información
      comparten DOS qubits, y es la magnitud que dibuja la topología del circuito: con
      entrelazamiento lineal solo hay puertas entre vecinos, así que los pares lejanos salen
      a cero exacto (comprobado sobre 300 perfiles aleatorios: a distancia ≥ 5 en la cadena,
      máximo 0,0000 bits). El cono de luz del circuito, hecho figura.
    """
    esc = _load_scaler_and_medians()
    if esc is None:
        return None, None
    idx = [esc["features"].index(f) for f in QSVM_FEATURES]
    x = (np.array(valores, dtype=float) - esc["mean"][idx]) / esc["scale"][idx]

    psi = zz_statevector(x)
    rho1 = [_rho_reducida(psi, [k]) for k in range(ZZ_N)]
    pureza = [float(np.real(np.trace(r @ r))) for r in rho1]
    r_len = np.array([np.sqrt(max(0.0, 2.0 * u - 1.0)) for u in pureza])
    S1 = [_entropia_vn(r) for r in rho1]

    MI = np.zeros((ZZ_N, ZZ_N))
    for i in range(ZZ_N):
        for j in range(i + 1, ZZ_N):
            # El max(0,·) es solo higiene numérica: la información mutua es no negativa por
            # subaditividad, pero la resta de tres entropías puede dar -1e-16.
            MI[i, j] = MI[j, i] = max(0.0, S1[i] + S1[j] - _entropia_vn(_rho_reducida(psi, [i, j])))
    return r_len, MI


def ent_circuito_svg(paso: int, medir: bool) -> str:
    """Diagrama del circuito de 3 qubits, dibujado con la paleta activa.

    Va como SVG EN LÍNEA y no como PNG —que es lo que hace la página Circuito Cuántico— por
    la diferencia de fondo entre las dos: allí el circuito es fijo, sale del notebook y se
    sirve ya renderizado; aquí cambia con cada pulsación y con el tema. Un SVG en línea hereda
    la tipografía y los tokens de color de la página, así que sigue al tema en los dos
    sentidos sin generar cuatro imágenes.
    """
    hilo, tinta = t["border_strong"], t["text"]
    apagado = t["text_muted"]
    # Un hilo por qubit, al mismo paso vertical (54 px) y con el mismo aire arriba y abajo que
    # entre hilos: la altura del viewBox sale de esa cuenta, no de un número ajustado a ojo.
    YS = (46, 100, 154)
    piezas = []
    for etiqueta, y in zip(("q₀", "q₁", "q₂"), YS):
        # Hilos: se dibujan enteros de un extremo a otro y las puertas se pintan encima.
        piezas.append(
            f'<line x1="74" y1="{y}" x2="446" y2="{y}" stroke="{hilo}" stroke-width="1.6"/>'
            f'<text x="8" y="{y + 5}" fill="{tinta}" font-family="{FONT_MONO}" '
            f'font-size="14">{etiqueta} |0⟩</text>')

    def _cnot(x, y_control, y_objetivo):
        """CNOT en notación canónica: punto relleno en el control, ⊕ en el objetivo y la
        vertical que los une. El ⊕ se dibuja con dos segmentos y no con un carácter, que
        dependería de que la fuente lo traiga."""
        return (f'<line x1="{x}" y1="{y_control}" x2="{x}" y2="{y_objetivo}" '
                f'stroke="{C_QUANTUM}" stroke-width="1.8"/>'
                f'<circle cx="{x}" cy="{y_control}" r="5.5" fill="{C_QUANTUM}"/>'
                f'<circle cx="{x}" cy="{y_objetivo}" r="13" fill="{t["surface_alt"]}" '
                f'stroke="{C_QUANTUM}" stroke-width="1.8"/>'
                f'<line x1="{x}" y1="{y_objetivo - 13}" x2="{x}" y2="{y_objetivo + 13}" '
                f'stroke="{C_QUANTUM}" stroke-width="1.8"/>'
                f'<line x1="{x - 13}" y1="{y_objetivo}" x2="{x + 13}" y2="{y_objetivo}" '
                f'stroke="{C_QUANTUM}" stroke-width="1.8"/>')

    # Las puertas ya aplicadas van en el acento cuántico; las que aún no, no se dibujan. El
    # diagrama es el registro de lo hecho, no el guion de lo que falta — para eso están los
    # botones, que ya dicen cuál toca.
    if paso >= 1:
        piezas.append(
            f'<rect x="120" y="{YS[0] - 20}" width="40" height="40" rx="7" '
            f'fill="{t["surface_alt"]}" stroke="{C_QUANTUM}" stroke-width="1.8"/>'
            # El RÓTULO va en C_QUANTUM_TEXTO y el trazo de la caja en C_QUANTUM: son el
            # mismo cian a dos pasos, y la diferencia importa solo aquí. Un trazo es
            # elemento gráfico y le basta el 3:1 de WCAG 1.4.11 (el paso de relleno da
            # 3,01 en claro); una letra de 17 px pide 4,5:1, y con el mismo cian se
            # quedaba corta sobre la superficie alterna. El paso oscurecido sube a 4,55.
            f'<text x="140" y="{YS[0] + 6}" text-anchor="middle" fill="{C_QUANTUM_TEXTO}" '
            f'font-family="{FONT_MONO}" font-size="17" font-weight="600">H</text>')
    # Los dos CNOT van escalonados y no en la misma columna: el segundo depende del primero
    # —controla sobre el qubit que el primero acaba de voltear—, y ponerlos alineados los
    # leería como simultáneos, que es justo lo contrario de la cadena que dibujan.
    if paso >= 2:
        piezas.append(_cnot(212, YS[0], YS[1]))
    if paso >= 3:
        piezas.append(_cnot(272, YS[1], YS[2]))
    if medir:
        # Medidor: el arco con la aguja, el símbolo de siempre. En tinta apagada porque la
        # medición no es una puerta más — es donde el estado cuántico deja de existir.
        for y in YS:
            piezas.append(
                f'<rect x="330" y="{y - 20}" width="40" height="40" rx="7" '
                f'fill="{t["surface_alt"]}" stroke="{apagado}" stroke-width="1.6"/>'
                f'<path d="M332 {y + 9} A 18 18 0 0 1 368 {y + 9}" fill="none" '
                f'stroke="{apagado}" stroke-width="1.6"/>'
                f'<line x1="350" y1="{y + 9}" x2="363" y2="{y - 7}" stroke="{apagado}" stroke-width="1.6"/>')
    # SIN atributo height. Un `height="auto"` no es válido en SVG —los atributos de
    # presentación quieren una longitud, y "auto" solo existe como valor CSS—, así que el
    # navegador lo rechazaba y lo dejaba anotado en consola ("Expected length, 'auto'") cada
    # vez que se pinta el circuito, o sea en cada pulsación de puerta. Omitiéndolo, el alto lo
    # deduce el propio viewBox a partir del width del 100 %, que es exactamente lo que se
    # quería; el `height:auto` del style queda además como respaldo explícito y ese sí es CSS.
    return (f'<svg viewBox="0 0 460 {YS[-1] + YS[0]}" width="100%" '
            f'style="display:block; height:auto; max-width:460px; margin:0 auto;" '
            f'role="img" aria-label="{html.escape(S("bl_ent_circuit_alt"))}">'
            + "".join(piezas) + "</svg>")


# ══════════════════════════════════════════════════════════════════════
# DIAGRAMA DE ARQUITECTURA — figura de cabecera de Resumen
# ══════════════════════════════════════════════════════════════════════
# Es la figura figures/arquitectura_tfm.png redibujada como SVG EN LÍNEA. El PNG original mide
# 1404 px de ancho y en esta página se pinta a ~1060: en una pantalla a 2x el navegador lo
# estira y el texto de 13 px sale emborronado, que es justo lo que no se le puede pedir a la
# primera figura del documento. En vector no hay resolución que valga: el mismo dibujo, los
# mismos rótulos y las mismas proporciones, pero nítido a cualquier zoom y en cualquier DPI.
# (los mismos rótulos en castellano: en los otros cuatro idiomas van traducidos, ver abajo).
# Hereda además la tipografía de la app y sigue al tema, como el circuito de tres qubits.
#
# LOS RÓTULOS VIVEN EN i18n.py y siguen al selector de idioma, igual que el resto de la app:
# el PNG solo existía en castellano, y ser texto de verdad —y no píxeles— es justamente lo que
# permite traducirlo. Lo que NO se traduce son los nombres propios (AWS S3, IAM, Databricks
# Community Edition, LightGBM, Qiskit, ZZFeatureMap, ONNX, GitHub…): son lo que son en los
# cinco idiomas. El texto alternativo para lectores de pantalla va aparte, en ov_arch_alt.
#
# Geometría medida sobre el PNG (bounding boxes reales, no a ojo) y desplazada 18/16 px para
# darle el aire lateral que allí no tenía: las cajas llegaban pegadas al borde de la imagen.
ARQ_CAJA_W, ARQ_CAJA_H = 269, 55        # cajas de dentro de cada grupo
ARQ_GRUPO_X = (231, 561, 891)           # esquina izquierda de los tres paneles de grupo
ARQ_FILA_Y = (110, 178, 246)            # tope de las tres filas de cajas
# Lo único de la figura que NO es texto: por grupo, si sus cajas van encadenadas con flechas
# y cuál de ellas va resaltada. El grupo del medio no lleva flechas porque los tres modelos
# son alternativas que se comparan entre sí, no una cadena; en el PNG tampoco las lleva.
ARQ_ESTRUCTURA = ((True, None), (False, 2), (True, None))
# Las cifras no viajan escritas dentro de los rótulos traducidos: llegan por marcador y las
# pone mil(), que las escribe con el separador de millar del idioma activo (29.400 · 29,400 ·
# 29 400). Es el mismo criterio que las cuatro tarjetas de estadísticas de esta página, y por
# el mismo motivo: "29.400" leído en inglés es veintinueve coma cuatro.
ARQ_CIFRAS = {"bronze": 29400, "silver": 7831, "train": 6264, "test": 1567}


def arquitectura_svg() -> str:
    """Diagrama de arquitectura del pipeline, en SVG y con la paleta activa."""
    if _is_dark:
        # En oscuro no valen los grises azulados del original —serían tres manchas claras sobre
        # el lienzo—, así que la ESCALERA se invierte: el panel exterior apenas se separa de la
        # tarjeta, el grupo sube un paso y la caja sube otro. Los recuadros llenos (S3, GitHub,
        # Streamlit, QSVM) se aclaran: el azul marino del original sobre fondo oscuro deja de
        # leerse como relleno y parece un agujero.
        c_out, c_grupo, c_caja = "#0A1519", "#122229", t["surface_alt"]
        c_borde, c_tinta, c_sub = t["border_strong"], t["text"], t["text_secondary"]
        c_acento, c_traza = "#7FA9C4", "#3D6C87"
        c_lleno, c_lleno_alt = "#2B4E66", "#3D6C87"
        c_lleno_tinta, c_lleno_sub = "#FFFFFF", "#CFE0EA"
    else:
        # Los colores exactos del PNG, muestreados sobre el propio fichero.
        c_out, c_grupo, c_caja = "#F2F7FA", "#E6EEF4", "#FFFFFF"
        c_borde, c_tinta, c_sub = "#86A8BC", "#203D50", "#5A6B75"
        c_acento, c_traza = "#3D6C87", "#5D8BA6"
        c_lleno, c_lleno_alt = "#203D50", "#3D6C87"
        c_lleno_tinta, c_lleno_sub = "#FFFFFF", "#CCDCE5"

    def _texto(x, y, txt, color, tam, peso=400):
        return (f'<text x="{x}" y="{y}" text-anchor="middle" fill="{color}" '
                f'font-family="{FONT_SANS}" font-size="{tam}" font-weight="{peso}">'
                f'{html.escape(txt)}</text>')

    def _caja(x, y, w, h, titulo, sub, relleno, borde, tinta, sub_color, rx=8):
        """Caja de dos líneas: rótulo en negrita y detalle debajo, centrados en el recuadro.
        Las bases van a −4 y +13 del centro vertical, la separación medida en el original."""
        cx, cy = x + w / 2, y + h / 2
        trazo = f' stroke="{borde}" stroke-width="1"' if borde else ""
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                f'fill="{relleno}"{trazo}/>'
                + _texto(cx, cy - 4, titulo, tinta, 15, 600)
                + _texto(cx, cy + 13, sub, sub_color, 13.5))

    def _flecha_h(x1, x2, y):
        return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 7}" y2="{y}" stroke="{c_acento}" '
                f'stroke-width="1.6"/><path d="M{x2} {y} L{x2 - 8.5} {y - 4.5} '
                f'L{x2 - 8.5} {y + 4.5} Z" fill="{c_acento}"/>')

    def _flecha_v(x, y1, y2):
        return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 5}" stroke="{c_acento}" '
                f'stroke-width="1.6"/><path d="M{x} {y2} L{x - 4.5} {y2 - 8.5} '
                f'L{x + 4.5} {y2 - 8.5} Z" fill="{c_acento}"/>')

    _s3, _git, _cloud = S("ov_arch_io")
    _cifras = {k: mil(v) for k, v in ARQ_CIFRAS.items()}

    p = [
        # Panel exterior: el recuadro discontinuo marca lo que corre DENTRO de Databricks.
        f'<rect x="212" y="16" width="1001" height="331" rx="14" fill="{c_out}" '
        f'stroke="{c_traza}" stroke-width="1.4" stroke-dasharray="9 5"/>',
        f'<text x="227" y="43.5" fill="{c_tinta}" font-family="{FONT_SANS}" '
        f'font-size="16" font-weight="600">Databricks Community Edition</text>',
        # Origen y destinos, fuera del recuadro: S3 entra por la izquierda; GitHub y Streamlit
        # Cloud salen por la derecha, uno debajo del otro.
        f'<rect x="19" y="149" width="174" height="95" rx="10" fill="{c_lleno}"/>',
        _texto(106, 183.5, _s3[0], c_lleno_tinta, 15, 600),
        _texto(106, 200.5, _s3[1], c_lleno_sub, 13.5),
        _texto(106, 217.5, _s3[2], c_lleno_sub, 13.5),
        _caja(1243, 178, 179, 56, _git[0], _git[1],
              c_lleno, None, c_lleno_tinta, c_lleno_sub, rx=10),
        _caja(1243, 291, 179, 55, _cloud[0], _cloud[1],
              c_lleno, None, c_lleno_tinta, c_lleno_sub, rx=10),
        _flecha_h(193, 231, 196.5),      # S3 → Medallón
        _flecha_h(1193, 1243, 206),      # evaluación → GitHub
        _flecha_v(1332.5, 234, 291),     # GitHub → Streamlit Cloud
    ]
    for i, (gx, (titulo, cajas), (encadenado, destacada)) in enumerate(
            zip(ARQ_GRUPO_X, S("ov_arch_grupos"), ARQ_ESTRUCTURA)):
        p.append(f'<rect x="{gx}" y="71" width="302" height="251" rx="12" fill="{c_grupo}"/>')
        p.append(_texto(gx + 151, 95.5, titulo, c_acento, 15, 600))
        if i:                                    # flecha entre este grupo y el anterior
            p.append(_flecha_h(gx - 28, gx, 196.5))
        for j, (rot, det) in enumerate(cajas):
            y = ARQ_FILA_Y[j]
            det = det.format(**_cifras)
            if j == destacada:
                p.append(_caja(gx + 17, y, ARQ_CAJA_W, ARQ_CAJA_H, rot, det,
                               c_lleno_alt, None, c_lleno_tinta, c_lleno_sub, rx=10))
            else:
                p.append(_caja(gx + 17, y, ARQ_CAJA_W, ARQ_CAJA_H, rot, det,
                               c_caja, c_borde, c_tinta, c_sub))
            if encadenado and j:
                p.append(_flecha_v(gx + 151, ARQ_FILA_Y[j - 1] + ARQ_CAJA_H, y))
    # Sin atributo height, igual que el circuito de tres qubits: el alto lo deduce el viewBox
    # del ancho al 100 %, y "auto" no es un valor válido para el atributo de presentación.
    return (f'<svg viewBox="0 0 1441 363" width="100%" role="img" '
            f'style="display:block; height:auto; margin:0 auto;" '
            f'aria-label="{html.escape(S("ov_arch_alt"))}">' + "".join(p) + "</svg>")


# ═══════════════════════════════════════════════════════════════════════
# PORTADA DE RESUMEN — LÁMINA A PANTALLA COMPLETA (solo en esta página)
# ═══════════════════════════════════════════════════════════════════════
# Al abrir Resumen no se ve la página: se ve una lámina que ocupa la ventana ENTERA a la derecha
# de la barra lateral. La página está debajo, y va apareciendo porque el scroll la sube por
# encima de la lámina, que se queda quieta. El recurso es `position:sticky` puro sobre el primer
# bloque más una hoja opaca —.st-key-ov_sheet— con todo lo demás; el JavaScript solo añade el
# matiz (parallax, desvanecido, barra de avance y entrada de cada bloque) y la pieza se sostiene
# sin él: sin script se ve la lámina, se hace scroll y la página aparece igual.
#
# ESTO YA NO VIVE EN UN IFRAME, y es el cambio de fondo respecto a la primera versión, que era un
# componente de 720 px con su propio scroll interno. Un iframe no se sale de su caja ni se entera
# del scroll de la página, así que a pantalla completa no llegaba por mucho que se le ajustara.
# Al traerlo al documento principal el scroll pasa a ser el de section[data-testid="stMain"] —el
# único contenedor con scroll de Streamlit, el mismo del que ya tira el reloj de cabecera— y lo
# que hay que resolver a cambio es que Streamlit reconstruye sus nodos en cada rerun: de eso se
# ocupa el MutationObserver del script.
#
# EL SANGRADO NO SE CALCULA. En vez de despejar "100vw menos la barra lateral menos la barra de
# scroll" —tres medidas, y la primera además cambia al colapsar el menú—, el bloque principal se
# queda sin padding y sin tope de ancho SOLO en esta página, con lo que su caja pasa a ser
# exactamente el área de contenido y la lámina llega de canto a canto con un simple 100%. El aire
# y el tope de 1500 px se los devuelve la hoja por dentro, con los mismos valores que Streamlit da
# por defecto (5rem a los lados en modo wide, 10rem abajo), así que el texto conserva al píxel la
# medida que tiene en las otras seis páginas.
#
# Y EN CLARO NO SE OSCURECE: SE REVELA EN NEGATIVO. Iba antes siempre en oscuro —con la paleta
# pedida a T("dark")— porque aclarar una fotografía la desarma. Pero esta no es una fotografía:
# es una RETÍCULA CLARA SOBRE UN DEGRADADO PLANO, y eso es justo lo que se puede invertir sin
# perder nada. El tema claro no necesita, por tanto, un segundo archivo: sale del mismo. La lámina
# pasa entonces a tomar la paleta ACTIVA en los dos temas —en oscuro es el telón azul noche de
# siempre, sin tocar una coma; en claro es papel con la retícula en tinta— y con ella se va el
# problema que arrastraba: la tira de cabecera ya no cruza dos capas opuestas (ver su nota, o
# mejor, su ausencia). El cómo está en .ov-hero-img.
#
# NO se usa GSAP ni ninguna librería: lo que hace el script son cuatro interpolaciones lineales
# sobre un único evento de scroll, ya amortiguado con requestAnimationFrame, más un
# IntersectionObserver para la entrada de los bloques.
_OV_RADIO = 30                      # radio del canto superior de la hoja (px)
_OV_SOMBRA = "0 -30px 70px -26px rgba(0,0,0,0.62)"
# Padding lateral del bloque principal en modo wide, que esta página le quita y la hoja devuelve.
# No es un número inventado: es sizes.wideSidePadding del tema de Streamlit, el que se aplica a
# partir de 864 px de ventana. Los dos escalones de abajo replican las media queries de la hoja
# principal (1.25rem en tablet, 1rem en móvil), para que la medida del texto no cambie en ninguna.
_OV_PAD_X = "5rem"

_OV_JS = """
<script>
(function () {
  var W = window.parent, doc = W.document;
  // Quien hace scroll es section[data-testid="stMain"], no la ventana: en W.scrollY no pasa
  // NUNCA nada. Es el mismo contenedor del que cuelga el fundido del reloj de cabecera.
  var sc = doc.querySelector('section[data-testid="stMain"]');
  if (!sc) { return; }
  // Un solo juego de escuchas por ventana: si este iframe se vuelve a montar (cambio de idioma,
  // de tema, ida y vuelta a la página) el anterior se desmonta antes, en vez de acumularse.
  if (W.__ovPortada) { W.__ovPortada.parar(); }
  // Aquí se leía prefers-reduced-motion para saltarse el parallax. Ya no: la portada se mueve
  // siempre, por el mismo motivo anotado junto a .ov-anim en la hoja de la portada.

  // Las referencias se REBUSCAN en cuanto dejan de estar en el documento: Streamlit reconstruye
  // sus nodos en cada rerun y las que se guardaran al arrancar quedarían apuntando a huérfanos.
  var cache = {};
  function el(sel) {
    if (!cache[sel] || !cache[sel].isConnected) { cache[sel] = doc.querySelector(sel); }
    return cache[sel];
  }

  var pedido = false;
  function pinta() {
    pedido = false;
    var lamina = el('.ov-hero');
    if (!lamina) { return; }
    // El avance se mide contra el alto de la lámina y no contra el de la página: lo que se está
    // contando es cuánto queda de portada, que es exactamente una pantalla.
    var p = Math.min(1, Math.max(0, sc.scrollTop / (lamina.clientHeight || 1)));
    var barra = el('.ov-bar');
    if (barra) { barra.style.transform = 'scaleX(' + p.toFixed(4) + ')'; }
    // La pista de scroll se va enseguida: en cuanto el gesto ha empezado ya no informa de nada.
    var pista = el('.ov-hint');
    if (pista) { pista.style.opacity = String(Math.max(0, 1 - p * 7)); }
    // El fondo se mueve a dos tercios de la velocidad del contenido y crece un pelo: es lo que da
    // la profundidad, la sensación de que la lámina está DETRÁS y no pegada al texto.
    var img = el('.ov-hero-img');
    if (img) {
      img.style.transform = 'translate3d(0,' + (p * 70).toFixed(2) + 'px,0) scale(' + (1 + p * 0.08).toFixed(4) + ')';
    }
    // El rótulo se apaga a casi el doble de ritmo que el scroll: desaparece antes de que la hoja
    // llegue a taparlo, y así no se le ve pasar por debajo del canto.
    var dentro = el('.ov-hero-in');
    if (dentro) {
      dentro.style.transform = 'translate3d(0,' + (-p * 80).toFixed(2) + 'px,0)';
      dentro.style.opacity = String(Math.max(0, 1 - p * 1.9));
    }
  }
  function alScroll() { if (!pedido) { pedido = true; W.requestAnimationFrame(pinta); } }

  // ── Entrada de cada bloque ────────────────────────────────────────────────────────────────
  var io = new W.IntersectionObserver(function (entradas) {
    entradas.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add('ov-seen'); io.unobserve(e.target); }
    });
  }, { root: sc, rootMargin: '0px 0px -8% 0px', threshold: 0.01 });

  // El estado escondido lo pone ESTE script, no la hoja de estilos, y es deliberado: si no
  // llegara a correr —iframe bloqueado, error de JS, navegador viejo— la página se ve entera en
  // vez de quedarse en blanco para siempre. La hoja solo describe el viaje, no el punto de salida.
  function registra() {
    var hoja = doc.querySelector('.st-key-ov_sheet');
    if (!hoja) { return; }
    var hijos = hoja.children;
    for (var i = 0; i < hijos.length; i++) {
      var b = hijos[i];
      if (b.dataset.ovAnim) { continue; }
      b.dataset.ovAnim = '1';
      b.classList.add('ov-anim');
      // Lo que ya está en pantalla (o ya ha pasado de largo, con top negativo) se marca visto en
      // esta MISMA tarea: las dos clases entran en el mismo recálculo de estilo, así que la
      // opacidad nunca llega a valer 0 y no hay parpadeo al volver de un rerun.
      if (b.getBoundingClientRect().top < W.innerHeight * 0.92) { b.classList.add('ov-seen'); }
      else { io.observe(b); }
    }
  }

  // Streamlit reconstruye los bloques en cada rerun y los nuevos nacen sin registrar, así que no
  // basta con recorrerlos una vez. Se amortigua con rAF porque un solo rerun dispara decenas de
  // mutaciones, y solo se observan altas y bajas de nodos (childList): las clases que añade
  // registra() son cambios de atributo y no realimentan el observador.
  var toca = false;
  var mo = new W.MutationObserver(function () {
    if (toca) { return; }
    toca = true;
    W.requestAnimationFrame(function () { toca = false; registra(); pinta(); });
  });
  mo.observe(sc, { childList: true, subtree: true });

  sc.addEventListener('scroll', alScroll, { passive: true });
  W.addEventListener('resize', alScroll);
  W.__ovPortada = { parar: function () {
    sc.removeEventListener('scroll', alScroll);
    W.removeEventListener('resize', alScroll);
    io.disconnect();
    mo.disconnect();
  } };
  // Al salir de Resumen, Streamlit desmonta este iframe y con él muere el realm donde viven
  // estas funciones; los observadores, que son del documento padre, se quedarían enganchados a
  // callbacks muertos. Se sueltan a mano en el desmontaje.
  window.addEventListener('pagehide', function () { if (W.__ovPortada) { W.__ovPortada.parar(); W.__ovPortada = null; } });

  registra();
  pinta();
})();
</script>
"""


def portada_resumen():
    """Lámina a pantalla completa, detrás de la página. Solo la usa Resumen.

    Emite en UNA sola llamada la hoja de estilos de la portada y su marcado, y no es capricho:
    el <style> tiene que viajar dentro del mismo stElementContainer que la lámina, porque ese
    contenedor —el único hijo directo del bloque vertical— es justo el nodo que el CSS convierte
    en el bloque pegajoso de 100vh. Separarlos en dos st.markdown crearía dos contenedores y el
    :has() apuntaría al que no es.
    """
    # EL COLOR DE LA LÁMINA, que es el único punto donde los dos temas se separan. En oscuro es el
    # color de la barra lateral, que en ESTA paleta queda un paso por ENCIMA del lienzo y no por
    # debajo (ver T(): el lienzo #05060A ya no deja sitio por abajo). El telón sigue despegándose
    # de la hoja que lo tapa, solo que ahora por arriba — lo que importa es que haya salto, no su
    # signo, porque la hoja es opaca y el borde se lee igual en las dos direcciones. En claro es EXACTAMENTE el lienzo, ni un paso de diferencia, y eso no es pereza sino
    # el requisito de .ov-hero-img: para que el negativo se funda con la página, lámina y hoja
    # tienen que ser el MISMO papel. De aquí salen también los cuatro velos, que no introducen
    # color propio — son este mismo fondo con alfa.
    lamina = t["sidebar_bg"] if _is_dark else t["bg"]
    # La lámina en claro es papel, así que el título va en tinta y el halo que lo despega de la
    # retícula tiene que ser CLARO: la sombra negra de siempre, alrededor de un texto oscuro,
    # solo lo emborrona. Misma inversión para el logotipo, que pasa de proyectar sombra sobre
    # el azul oscuro a apoyarse sobre papel — y ahí un negro al 55% se lee como suciedad, no como relieve.
    sombra_tit  = ("0 2px 16px rgba(0,0,0,0.55)" if _is_dark
                   else f"0 1px 12px {hex_to_rgba(lamina, 0.88)}")
    sombra_logo = ("0 6px 18px rgba(0,0,0,0.55)" if _is_dark
                   else "0 4px 12px rgba(5,6,10,0.14)")
    # EL HALO DE LA PISTA, prestado de la cápsula-interruptor del pie de la barra lateral: mismo
    # color (C_MID2, el paso del ORO que pesa en cada tema — ver su definición: es justo este halo
    # el que obligó a sacarlo de la rampa cuando la rampa se volvió fría—, resuelto para los dos
    # temas) y mismas tres capas —cerco, brillo y difusión—. Aquellas son box-shadow porque la
    # cápsula es una caja; una flecha dibujada a trazo no tiene caja que sombrear, así que aquí
    # van como drop-shadow, que sigue la silueta del trazo en vez de su rectángulo. La conversión
    # pierde el `spread` (drop-shadow no lo tiene) y se compensa en el radio de desenfoque.
    # Los DOS extremos del latido son los dos estados de la cápsula: el bajo es su reposo y el
    # alto, su :hover. Así el gesto no es un efecto nuevo, es el mismo que ya hay en la app
    # puesto a respirar — y si algún día se retoca el halo de allí, este es el sitio a igualar.
    halo_bajo = (f"drop-shadow(0 0 1px {C_MID2}55) drop-shadow(0 0 5px {C_MID2}99)"
                 f" drop-shadow(0 0 11px {C_MID2}55)")
    halo_alto = (f"drop-shadow(0 0 2px {C_MID2}88) drop-shadow(0 0 7px {C_MID2}CC)"
                 f" drop-shadow(0 0 15px {C_MID2}77)")
    # ── EL NEGATIVO: la misma imagen, revelada al revés ─────────────────────────────────────
    # En claro no se carga otro archivo, se invierte el que ya hay. Se puede porque hero-quantum
    # no es una fotografía sino un DIBUJO —retícula clara sobre un degradado plano de un solo
    # tono—, y ahí invertir la luminosidad no deforma nada: lo que era fondo oscuro con líneas
    # claras pasa a ser fondo claro con líneas oscuras. Cuatro pasos, en este orden:
    #
    #   invert(1)          el negativo. Por sí solo también gira el TONO: el cian de la esfera
    #                      sale por su complementario, un naranja apagado que no es el oro de
    #                      la marca y que la ensucia sin llegar a citarla.
    #   hue-rotate(180deg) devuelve el tono a su sitio. Es el complemento exacto del giro que
    #                      acaba de dar la inversión, así que la retícula vuelve a ser azul —la
    #                      misma familia de color que en oscuro— solo que ahora en tinta.
    #   contrast(0.88)     el negativo nace más duro que el original y las líneas competían con
    #                      el titular, que cae encima. Es un fondo: tiene que quedarse detrás.
    #   brightness(1.06)   sube el blanco del negativo hasta rebasar el papel, que es lo que
    #                      permite el paso siguiente.
    #
    # Y el que hace el trabajo de verdad: mix-blend-mode:multiply. Multiplicar es quedarse con
    # LO MÁS OSCURO de cada punto, así que el casi-blanco del fondo invertido deja pasar el
    # lienzo entero y solo sobreviven las líneas. Es decir: la lámina NO TIENE FONDO PROPIO en
    # claro — el fondo es la página. De ahí que el canto de la hoja se lea como un pliegue del
    # papel y no como el corte entre dos superficies distintas, y de ahí también que `lamina`
    # tenga que valer t['bg'] exactamente (ver allí). El grupo de mezcla lo acota .ov-hero.
    #
    # En OSCURO no se emite nada: la imagen va tal cual, como ha ido siempre.
    negativo = "" if _is_dark else (
        "filter:invert(1) hue-rotate(180deg) contrast(0.88) brightness(1.06);"
        " mix-blend-mode:multiply;")
    fondo = _b64_image(str(ASSETS_DIR / "hero-quantum.webp"))
    logo = _b64_image(str(ASSETS_DIR / "qml_logov2-sidebar.png"))
    st.markdown(f"""<style>
/* ── EL BLOQUE PRINCIPAL, A SANGRE ──────────────────────────────────────────────────────────
   Solo en esta página. Estas dos declaraciones deshacen el padding y el tope de ancho de la hoja
   principal —y de sus dos media queries, que también los declaran con !important— y ganan por
   ORDEN DE APARICIÓN, no por especificidad: esta hoja se inyecta desde el cuerpo de la página y
   aquella se escribe entera al principio del script. Al quedarse sin padding, la caja del
   contenido pasa a ser exactamente el área a la derecha de la barra lateral, que es lo que
   necesita la lámina para llegar de canto a canto sin calcular nada. */
div[data-testid="stMainBlockContainer"], section.main > div.block-container {{
    padding:0 !important; max-width:none !important;
}}

/* ── Y A SANGRE TAMBIÉN POR ARRIBA ──────────────────────────────────────────────────────────
   Quitar el padding no bastaba: la lámina seguía empezando 64 px por debajo del canto, y en esa
   franja se veía el fondo de .stApp —con su halo dorado— cortado a cuchillo contra la fotografía,
   justo a la altura del reloj y las banderas. Esos 64 px no son padding de nadie: son HUECO DE
   REJILLA. El bloque vertical raíz es un flex con gap de 1rem y por delante de la lámina van
   cuatro hijos que no pintan nada —la hoja de estilos general y los tres componentes de
   cabecera: banderas, atributo lang y reloj—; sus cajas miden cero, pero siguen siendo ítems del
   flex y cada una cobra su hueco. Cuatro por dieciséis.
   Se anula el hueco del bloque RAÍZ en vez de disolver los cuatro uno a uno: Streamlit 1.55
   envuelve cada st.container en un stLayoutWrapper propio —sin clase st-key- de la que agarrarse—
   que habría que perseguir aparte. Y no se pierde nada, porque en esta página el bloque raíz no
   separa: sus dos únicos hijos con cuerpo son la lámina y la hoja que la tapa, y esas dos se
   relacionan por z-index, no por espaciado. Va aquí dentro y no en la hoja principal a propósito:
   en las otras seis páginas esos mismos 64 px son el aire que hay sobre el titular. */
div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {{
    gap:0 !important;
}}

/* ── LA LÁMINA ──────────────────────────────────────────────────────────────────────────────
   El bloque pegajoso es el stElementContainer de este mismo markdown y no un div nuestro: es el
   único nodo que es hijo directo del bloque vertical, y por tanto el único que puede quedarse
   pegado al techo mientras el resto de la página le pasa por encima. Se localiza con :has() —el
   mismo recurso que ya usan la barra lateral y el conmutador de idioma— en su forma laxa, sin
   combinadores de hijo, para que siga valiendo si Streamlit cambia el andamiaje interno del
   markdown. 100dvh además de 100vh: en el móvil, vh mide la ventana SIN contar la barra del
   navegador y la lámina se pasaba de alto justo en la primera pantalla. */
div[data-testid="stElementContainer"]:has(.ov-hero) {{
    position:sticky; top:0; z-index:0;
    height:100vh; height:100dvh;
}}
/* La lámina se coloca en ABSOLUTO dentro de esa caja: así no depende de que los dos o tres divs
   que Streamlit mete por medio hereden el alto, que es justo lo que cambia entre versiones. */
.ov-hero {{
    position:absolute; inset:0; overflow:hidden;
    display:flex; align-items:center;
    background:{lamina}; color:{t['text']};
    font-family:{FONT_SANS};
    /* El grupo de mezcla del negativo. Sin esto, el multiply de .ov-hero-img no se detendría en
       el fondo de la lámina: buscaría hacia atrás el primer contexto de apilamiento y acabaría
       mezclándose con el lienzo de .stApp Y CON SUS HALOS dorados, que es un fondo distinto y
       además no uniforme. Aquí el contexto lo crearía de todos modos el bloque pegajoso —tiene
       position:sticky con z-index—, pero eso es un efecto colateral de OTRA regla escrita para
       otra cosa; declararlo aquí deja el grupo donde de verdad tiene que estar. */
    isolation:isolate;
}}
/* El fondo desborda su caja (inset negativo) para que el parallax no descubra un borde vacío al
   desplazarlo. La imagen es apaisada de 2.44:1 y una ventana no llega a 1.8:1, así que `cover`
   recorta por los lados: se ancla al 78% porque ahí está la esfera, y el aire que sobra a la
   izquierda es justo donde va el logotipo. */
.ov-hero-img {{
    position:absolute; inset:-10% -5%;
    background:url('data:image/webp;base64,{fondo}') no-repeat 78% center / cover;
    will-change:transform;
    {negativo}
}}
/* Dos velos: uno horizontal que despeja la mitad izquierda —el texto necesita fondo, no suerte—
   y otro vertical que asienta la imagen por arriba y por abajo. Ninguno introduce color propio:
   los dos son el MISMO fondo de la lámina con alfa, así que apagan la imagen sin desplazarle el
   matiz. Por eso el mismo par de degradados sirve para los dos temas sin tocar un solo número:
   en oscuro `lamina` es azul noche y los velos OSCURECEN; en claro es papel y ACLARAN. La operación
   es la misma —acercar la imagen al fondo—, y el reparto de alfas, que es lo que está medido
   contra la caja de texto, no depende de hacia qué lado se acerque. */
.ov-hero-veil {{
    position:absolute; inset:0;
    background:
      linear-gradient(90deg, {lamina} 0%, {hex_to_rgba(lamina, 0.86)} 28%,
                      {hex_to_rgba(lamina, 0.28)} 56%, {hex_to_rgba(lamina, 0)} 80%),
      linear-gradient(180deg, {hex_to_rgba(lamina, 0.55)} 0%, {hex_to_rgba(lamina, 0)} 24%,
                      {hex_to_rgba(lamina, 0)} 58%, {hex_to_rgba(lamina, 0.78)} 100%);
}}
/* El rótulo se ALINEA CON EL TEXTO de la página, no con el borde de la pantalla: el max() elige
   el mayor entre un margen suelto y el canto izquierdo de la columna de 1500 px (la mitad del
   sobrante más el padding que devuelve la hoja). Así, en monitores anchos el logotipo cae a
   plomo sobre el titular que aparece justo debajo al hacer scroll, y en pantallas estrechas
   —donde no hay sobrante— se queda con su margen mínimo. El centrado vertical lo da el flex del
   contenedor y NO un translateY, porque el transform se lo queda el parallax. */
.ov-hero-in {{
    position:relative; z-index:2; flex:0 1 auto;
    /* 660 px y no los 460 de cuando aquí iba un antetítulo de cuatro palabras: el título del
       TFM necesita una medida de lectura decente. El 56% es el tope real —a partir de ahí el
       texto se metería en la mitad derecha de la fotografía, donde está la esfera y donde el
       velo ya casi no oscurece—, y por debajo de 1180 px de área manda ese porcentaje. */
    max-width:min(56%, 660px);
    margin-left:max(clamp(24px, 5vw, 64px), calc(50% - 750px + {_OV_PAD_X}));
    will-change:transform, opacity;
}}
.ov-hero-logo {{
    height:clamp(60px, 8vh, 96px); width:auto; display:block; margin-bottom:26px;
    filter:drop-shadow({sombra_logo});
}}
/* El TÍTULO DEL TFM, y no el antetítulo corto de la página: son dos textos distintos a
   propósito. Sobre la lámina va el título largo —i18n["ov_hero_title"]—, y el "Framework
   DataOps + QML" se queda donde estaba, coronando el titular "Resumen" que aparece al bajar.
   Por eso esto NO reutiliza .page-eyebrow ni su calco de aquí: aquel es mono, en versalitas y
   a 0.16em de espaciado, y con 150 caracteres saldrían ocho líneas de mayúsculas ilegibles.
   Serif, caja normal e interlineado corto, que es como se lee un título.
   La huincha dorada se conserva —es la firma visual de la marca— pero alineada con la PRIMERA
   LÍNEA: flex-start más un margen óptico de 0.62em, porque centrada en un bloque de tres o
   cuatro líneas quedaría flotando en mitad del texto. La sombra no es decorativa: el título
   cae sobre la imagen y el velo no garantiza el mismo fondo en todos los anchos — y por eso
   cambia de signo con el tema (ver `sombra_tit`), porque lo que tiene que hacer es despegar el
   texto de la retícula, no oscurecer. */
.ov-hero-titulo {{
    font-family:{FONT_SERIF}; font-size:clamp(18px, 1.7vw, 25px); font-weight:400;
    line-height:1.3; letter-spacing:-0.01em; color:{t['text']};
    text-shadow:{sombra_tit};
    display:flex; align-items:flex-start; gap:14px;
}}
.ov-hero-titulo::before {{
    content:""; width:22px; height:2px; border-radius:1px; background:{P_CLINICO_ALTO};
    flex-shrink:0; margin-top:0.62em;
}}
.ov-hero-rule {{
    height:1px; margin-top:22px; max-width:280px;
    background:linear-gradient(90deg, {t['border_strong']}, transparent);
}}
/* Pista de desplazamiento: un hilo que cae y un ángulo. Sin palabras — no hay que traducirlo, y
   en una lámina así el gesto se entiende antes leyendo el dibujo que una frase. */
.ov-hint {{
    position:absolute; left:50%; bottom:26px; transform:translateX(-50%); z-index:3;
    display:flex; flex-direction:column; align-items:center; gap:9px;
    transition:opacity 0.25s ease; pointer-events:none;
}}
.ov-hint-line {{ width:1.5px; height:38px; background:linear-gradient(180deg, transparent, {t['text_secondary']}); }}
/* El halo va en el <svg> y NO en .ov-hint, que es quien recibe el desvanecido del scroll: si los
   dos filtros vivieran en el mismo elemento, apagar la pista al bajar apagaría también el latido
   a mitad de ciclo. Y el filtro se declara AQUÍ además de en los fotogramas —repetido a
   propósito— porque es el ESTADO DE REPOSO: si la animación no llega a correr, la flecha
   conserva su halo en vez de quedarse pelada, que es justo lo que se vino a añadir.
   Esa línea nació para cubrir el caso de `prefers-reduced-motion`, que anulaba el latido unas
   reglas más abajo; esa anulación ya no existe (ver la nota junto a .ov-anim), así que hoy cubre
   solo el caso degradado. Se mantiene: cuesta una línea y es la diferencia entre una señal
   apagada y una legible. */
.ov-hint svg {{
    display:block; filter:{halo_bajo};
    animation:ovBaja 1.9s cubic-bezier(0.4,0,0.2,1) infinite;
}}
/* El latido: la flecha cae cinco píxeles y a la vez el halo sube de su reposo a su punto alto.
   El suelo de opacidad es 0.72 y no el 0.55 de antes — con el trazo ya más grueso, bajar tanto
   la hacía titilar en vez de respirar. */
@keyframes ovBaja {{
    0%,100% {{ transform:translateY(0); opacity:0.72; filter:{halo_bajo}; }}
    50%     {{ transform:translateY(5px); opacity:1; filter:{halo_alto}; }}
}}
/* Avance de la portada, pegado al canto superior de la pantalla: la única señal de cuánto queda
   de lámina, ya que la barra de scroll de la página no dice nada de esto. Desaparece sola en
   cuanto la hoja la cubre, porque va dentro de la lámina. */
.ov-bar {{
    position:absolute; top:0; left:0; right:0; height:2px; z-index:4;
    transform-origin:left center; transform:scaleX(0);
    background:linear-gradient(90deg, {P_CLINICO}, {P_CLINICO_ALTO});
    pointer-events:none;
}}

/* ── LA HOJA QUE TAPA ───────────────────────────────────────────────────────────────────────
   Es el contenedor con TODO el resto de la página. Tiene que ser opaca —es lo único que separa
   las dos capas— y va un peldaño por encima de la lámina (z-index 1 contra 0). El fondo repite
   el de .stApp, halos incluidos, para que por debajo del pliegue Resumen se vea exactamente
   igual que las otras seis páginas.
   Se estila el .st-key- A PELO y no un envoltorio, porque en esta versión de Streamlit no hay
   tal envoltorio: un st.container(key=...) es UN solo div (StyledFlexContainerBlock, con
   data-testid="stVerticalBlock" y la clase de la key encima), hijo directo del bloque vertical
   de la página. Las reglas de la hoja principal que apuntan a stVerticalBlockBorderWrapper son
   de una versión anterior y hoy no casan con nada — comprobado en el bundle de 1.55.
   Y de ahí el padding lateral con max(): al ser un único div, el fondo tiene que ir a sangre
   (ancho completo) y la medida del texto la fija el propio padding. El segundo término centra
   una caja de contenido de 1340 px —los 1500 del tope de la app menos sus dos 5rem— y el
   primero es el suelo cuando la pantalla no da para tanto; ambos coinciden en 5rem justo a los
   1500 px, así que no hay salto. 670 = 1500/2 − 80. */
.st-key-ov_sheet {{
    position:relative; z-index:1;
    background-color:{t['bg']}; background-image:{HALOS};
    border-top:1px solid {t['border']};
    border-radius:{_OV_RADIO}px {_OV_RADIO}px 0 0;
    box-shadow:{_OV_SOMBRA};
    padding:3rem max({_OV_PAD_X}, calc(50% - 670px)) 10rem;
}}
/* Filete de luz sobre el canto: marca el borde que avanza sobre la imagen. */
.st-key-ov_sheet::before {{
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    border-radius:{_OV_RADIO}px {_OV_RADIO}px 0 0;
    background:linear-gradient(90deg, {P_CLINICO_ALTO}, {P_CLINICO_ALTO}00 55%);
}}
/* El script vive en un iframe que no pinta nada; sin esto ocuparía un hueco al final de la
   página. Mismo tratamiento —Y COMPLETO— que el reloj de cabecera y el atributo de idioma:
   display:contents en los envoltorios Y position:fixed en el iframe. Faltaba lo segundo, y no
   era un detalle: el `height=0` que se le pasa a components.html es FALSY, así que el frontal
   lo descarta y planta la altura por defecto de 150 px. El iframe seguía por tanto en el flujo
   midiendo 150, y al pie de la página quedaba una banda muerta de ese alto por debajo del
   canto de la hoja, con la lámina asomando detrás. */
.st-key-ov_js, .st-key-ov_js div[data-testid="stIFrame"],
.st-key-ov_js div[data-testid="stElementContainer"] {{ display:contents !important; }}
.st-key-ov_js iframe {{
    position:fixed !important; width:0 !important; height:0 !important;
    border:0 !important; opacity:0 !important; pointer-events:none !important;
}}

/* ── LA TIRA DE CABECERA NO LLEVA NADA, Y ESO ES EL CAMBIO ──────────────────────────────────
   Aquí vivían tres reglas —color del reloj, de su separador y de la flecha del selector— y las
   tres han desaparecido de golpe al pasar la lámina a la paleta activa. Conviene saber por qué
   estaban, porque si algún día la lámina vuelve a ir siempre en oscuro hay que reponerlas:
   el reloj y la flecha son POSITION:FIXED y caen sobre la lámina, no sobre la hoja. Mientras la
   lámina era oscura con la app en claro, esa franja era la única de toda la aplicación donde el
   cromo de interfaz se leía contra un fondo del tema CONTRARIO: el gris del tema daba 2,2:1
   sobre la fotografía y el de la paleta oscura, 2,0:1 sobre el papel al bajar. Ninguno pasaba, y
   la flecha —que cruza las dos capas— tuvo que irse al acento oscurecido por un 3,2:1 que era el
   margen más ajustado de toda la app.
   Ahora las dos capas son del mismo tema, así que los colores por defecto del reloj y de la
   flecha (ver sus bloques) valen tal cual y estas tres reglas serían calcos exactos de lo que ya
   hay. Se van: una excepción sin excepción que documentar es solo ruido. */

/* ── ENTRADA DE CADA BLOQUE ─────────────────────────────────────────────────────────────────
   El punto de partida (invisible y 26 px más abajo) lo pone el SCRIPT añadiendo .ov-anim, no
   esta hoja: si el script no llegara a correr, la página se ve entera en vez de quedarse en
   blanco. Aquí solo se describe el viaje. */
.ov-anim {{
    opacity:0; transform:translateY(26px);
    transition:opacity 0.7s cubic-bezier(0.22,1,0.36,1), transform 0.7s cubic-bezier(0.22,1,0.36,1);
}}
.ov-anim.ov-seen {{ opacity:1; transform:none; }}
/* AQUÍ VIVÍA LA ANULACIÓN POR prefers-reduced-motion, y se ha retirado a propósito. Forzaba
   .ov-anim a opacity:1 / transform:none / transition:none y apagaba el latido de la pista, de
   modo que quien pedía menos movimiento conservaba el revelado —que es scroll, no animación— y
   perdía el fundido de los bloques y el latido. Es la tercera y última excepción de la
   aplicación, hermana de la del contador y la de la entrada escalonada, y por el mismo motivo:
   el equipo desde el que se trabaja este panel tiene los efectos de animación de Windows
   apagados, así que la portada se veía plana en escritorio y con parallax solo en el móvil y la
   tableta. Entregadas ya la memoria y la defensa, esto es acabado visual.
   El punto de partida invisible lo sigue poniendo el SCRIPT y no esta hoja (ver arriba), así que
   retirar la anulación no puede dejar la página en blanco: sin script no hay .ov-anim y los
   bloques se ven enteros. Revertirlo es reponer este bloque tal cual, con sus dos reglas. */
/* PANTALLA BAJA, que en un teléfono es simplemente girarlo: la lámina mide 100dvh, y con la
   ventana en 390 px de alto el logotipo, el título de cuatro líneas y la pista no caben. El
   bloque va centrado con flex dentro de un contenedor con overflow:hidden, así que lo que no
   cabe no empuja: se recorta por arriba y por abajo a la vez. Se mide por ALTO y no por ancho
   porque es el alto lo que falla —una tableta en vertical tiene el mismo ancho y no sufre—, y
   la pista se retira porque en esa franja cae justo encima del texto. */
@media (max-height: 520px) {{
    .ov-hero-logo {{ height:38px; margin-bottom:12px; }}
    .ov-hero-titulo {{ font-size:clamp(14px, 2.7vh, 18px); }}
    .ov-hero-rule {{ margin-top:14px; }}
    .ov-hint {{ display:none; }}
}}
@media (max-width: 1024px) {{
    .st-key-ov_sheet {{ padding:3rem 1.25rem 10rem; }}
    .ov-hero-in {{ max-width:min(70%, 560px); margin-left:clamp(20px, 4vw, 40px); }}
    .ov-hero-titulo {{ font-size:clamp(17px, 2.3vw, 21px); }}
}}
@media (max-width: 768px) {{
    .st-key-ov_sheet {{ padding:2.5rem 1rem 6rem; }}
    /* En el teléfono el texto ocupa casi todo el ancho y la fotografía queda de fondo entero:
       no hay mitad izquierda que respetar porque no cabe la esfera al lado. */
    .ov-hero-in {{ max-width:84%; }}
    .ov-hero-logo {{ height:clamp(46px, 7vh, 64px); margin-bottom:18px; }}
    .ov-hero-titulo {{ font-size:clamp(15px, 3.9vw, 19px); gap:10px; }}
    .ov-hero-titulo::before {{ width:16px; }}
    /* Y por eso el velo cambia de forma, no solo de medida: el de escritorio es HORIZONTAL
       —oscurece la mitad izquierda y se va a transparente al 80%, donde no hay texto— y aquí el
       texto llega justo hasta ese 84%, es decir, hasta la zona sin velo. Sobre una fotografía
       cualquiera eso es una tirada de dados con la legibilidad. En vertical el velo pasa a cubrir
       todo el ancho, con el degradado en el otro eje: la imagen sigue leyéndose y el título tiene
       fondo garantizado esté donde esté el recorte. */
    .ov-hero-veil {{
        background:
          linear-gradient(90deg, {hex_to_rgba(lamina, 0.90)} 0%,
                          {hex_to_rgba(lamina, 0.62)} 100%),
          linear-gradient(180deg, {hex_to_rgba(lamina, 0.38)} 0%,
                          {hex_to_rgba(lamina, 0)} 42%, {hex_to_rgba(lamina, 0.70)} 100%);
    }}
}}
</style>
<div class="ov-hero">
  <div class="ov-hero-img"></div>
  <div class="ov-hero-veil"></div>
  <div class="ov-hero-in">
    <img class="ov-hero-logo" src="data:image/png;base64,{logo}" alt="QML DataOps">
    <div class="ov-hero-titulo">{S("ov_hero_title")}</div>
    <div class="ov-hero-rule"></div>
  </div>
  <div class="ov-hint" aria-hidden="true">
    <span class="ov-hint-line"></span>
    <svg width="22" height="13" viewBox="0 0 22 13" fill="none">
      <path d="M2 2 L11 11 L20 2" stroke="{t['text']}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <div class="ov-bar" aria-hidden="true"></div>
</div>""", unsafe_allow_html=True)


def portada_js():
    """Parallax, avance y entrada de los bloques. Va al FINAL de la página, cuando la hoja ya
    existe en el documento; aun así el script no depende de ese orden, porque el observador de
    mutaciones recoge lo que llegue después."""
    with st.container(key="ov_js"):
        components.html(_OV_JS, height=0, width=0)


# ═══════════════ CONTADORES ═══════════════
# Las cifras que encabezan Resumen (las cuatro del dataset NHANES) y Resultados (los tres
# AUC-ROC) suben desde cero al llegar a ellas. Es el único adorno de la aplicación que toca un
# DATO, y por eso lleva tres cautelas que no son opcionales:
#
#   1. EL SERVIDOR SIGUE ESCRIBIENDO LA CIFRA FINAL EN EL HTML. El script solo la sustituye
#      mientras dura la cuenta, y en el último fotograma repone la cadena original byte a byte
#      —no una reconstrucción suya, que podría diferir en un decimal de lo que dice la memoria—.
#      Si el iframe no llega a ejecutarse (bloqueado, error de JS, navegador viejo) lo que se ve
#      es el número correcto y quieto, nunca un cero ni un hueco. Mismo criterio que el revelado
#      por scroll de la portada: la hoja describe el viaje, jamás el punto de salida.
#   2. NO SE LEE NINGÚN VALOR DE UN ATRIBUTO: se trocea el texto ya renderizado. Así la cuenta
#      hereda gratis la notación del idioma activo —coma o punto decimal, punto o espacio
#      inseparable de millar— sin duplicar aquí la lógica de nf() y mil(). Duplicarla es
#      exactamente por donde se colaría una discrepancia entre lo que cuenta la animación y lo
#      que afirma la tarjeta, y en cinco idiomas hay cinco ocasiones de equivocarse.
#   3. SOLO SE CUENTA AL ENTRAR EN LA PÁGINA. Cualquier rerun —cambiar el tema, el idioma, mover
#      un control— remonta el iframe; sin la guarda, las cifras volverían a correr en cada clic,
#      que además de mareante miente: no ha entrado ningún dato nuevo.
#
# ESTE CONTADOR NO MIRA prefers-reduced-motion, y aquí está escrito el porqué —que vale para
# TODA la aplicación, porque hoy ya no queda un solo movimiento que la consulte—. No es un
# descuido: es una decisión, y conviene leerla antes de "arreglarla".
#
# El equipo desde el que se trabaja este panel lleva los efectos de animación de Windows
# apagados, así que sus navegadores piden reduce SIEMPRE — comprobado con
# SystemParametersInfo(SPI_GETCLIENTAREAANIMATION) = 0 y en la consola con matchMedia. Con las
# guardas puestas, la aplicación se veía animada en el móvil y en la tableta y completamente
# plana en Chrome y en Firefox de escritorio, que es justo donde se compone. La memoria y la
# defensa ya están entregadas: esto es acabado visual, y entre respetar la preferencia del
# sistema y que el trabajo se vea donde se mira, se ha elegido lo segundo.
#
# Las guardas se retiraron en cinco sitios, cada uno con su nota al lado: este contador, la
# entrada escalonada del contenido, la anulación de .ov-anim y del latido en la portada, el
# parallax dentro de _OV_JS, el fundido del menú lateral y el de página al navegar. Revertir
# cualquiera es reponer su condición, que en todos los casos es una línea.
#
# La cadena es RAW (r"""...""") y no una f-string: lleva expresiones regulares, y con las llaves
# dobladas de una f-string los cuantificadores \d{3} serían ilegibles. Los tres valores que
# vienen de Python entran por marcas __ASI__ sustituidas con json.dumps, que es lo que separa
# "escribir una cadena" de "inyectar lo que haya" dentro de un <script> — el mismo cuidado que
# el reloj de cabecera.
_CONTADOR_JS = r"""
<script>
(function () {
  var W = window.parent, doc = W.document;
  // Un solo juego de escuchas por ventana: si este iframe se vuelve a montar, el anterior se
  // desmonta antes en vez de acumularse. parar() además REPONE las cifras que se quedaran a
  // medias, para que un rerun en mitad de la cuenta no congele un número falso en pantalla.
  if (W.__tfmContador) { W.__tfmContador.parar(); W.__tfmContador = null; }

  var MILLAR = __MILLAR__, DECIMAL = __DECIMAL__, PAGINA = __PAGINA__;

  var entrada = (W.__tfmContadorPagina !== PAGINA);
  W.__tfmContadorPagina = PAGINA;
  // Aquí NO se consulta prefers-reduced-motion; el porqué —que hoy vale para toda la
  // aplicación— está arriba, en la cabecera del bloque. Lo único que corta la cuenta es no
  // haber cambiado de página.
  if (!entrada) { return; }

  // 1150 ms es lo que tarda en leerse una cifra sin que la espera se note. El escalón de 90 ms
  // por columna barre la fila de izquierda a derecha, y los 140 de base la dejan arrancar
  // cuando la tarjeta ya ha entrado con su propia animación (.stat-card, retardo 0,20 s).
  var DUR = 1150, BASE = 140, ESCALON = 90;

  function esc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
  // Un número es una tira de dígitos con sus millares opcionales y un decimal opcional. Se
  // construye con los separadores del idioma ACTIVO, así que "29.400" es un solo número en
  // español y "29,400" lo es en inglés, sin que ninguno de los dos parta por donde no debe.
  var RE_NUM = new RegExp('\\d+(?:' + esc(MILLAR) + '\\d{3})*(?:' + esc(DECIMAL) + '\\d+)?', 'g');

  function formatea(v, dec, agrupa) {
    var p = v.toFixed(dec).split('.');
    var ent = p[0];
    // El millar se marca primero con U+0000 y se sustituye después. Escribirlo directo rompería
    // en español, donde el separador de millar es el punto: entraría en el split del decimal.
    if (agrupa) { ent = ent.replace(/\B(?=(\d{3})+(?!\d))/g, '\u0000'); }
    ent = ent.split('\u0000').join(MILLAR);
    return p.length > 1 ? ent + DECIMAL + p[1] : ent;
  }

  // Trocea "86% / 14%" en [86, "% / ", 14, "%"] y "0,9485" en [0,9485]. Lo fijo se conserva tal
  // cual, así que sirve igual para una cifra sola, para dos con texto en medio y para el
  // porcentaje doble del reparto de clases, sin un caso especial por tarjeta.
  function trocea(txt) {
    var piezas = [], ultimo = 0, m;
    RE_NUM.lastIndex = 0;
    while ((m = RE_NUM.exec(txt)) !== null) {
      if (m.index > ultimo) { piezas.push({ fijo: txt.slice(ultimo, m.index) }); }
      var s = m[0], i = s.indexOf(DECIMAL);
      piezas.push({
        num: parseFloat(s.split(MILLAR).join('').split(DECIMAL).join('.')),
        dec: i < 0 ? 0 : s.length - i - DECIMAL.length,
        agrupa: s.indexOf(MILLAR) >= 0
      });
      ultimo = m.index + s.length;
    }
    if (ultimo < txt.length) { piezas.push({ fijo: txt.slice(ultimo) }); }
    return piezas;
  }

  function pinta(piezas, k) {
    var s = '';
    for (var i = 0; i < piezas.length; i++) {
      var p = piezas[i];
      s += (p.fijo !== undefined) ? p.fijo : formatea(p.num * k, p.dec, p.agrupa);
    }
    return s;
  }

  var activos = [], lazo = 0;

  function paso(ahora) {
    lazo = 0;
    var quedan = [];
    for (var i = 0; i < activos.length; i++) {
      var el = activos[i], d = el.__tfmCont;
      if (!el.isConnected) { continue; }          // se lo llevó un rerun por delante
      var t = (ahora - d.t0) / DUR;
      if (t >= 1) { el.textContent = d.original; continue; }   // ← la cadena del servidor
      if (t > 0) { el.textContent = pinta(d.piezas, 1 - Math.pow(1 - t, 4)); }
      quedan.push(el);
    }
    activos = quedan;
    if (activos.length) { lazo = W.requestAnimationFrame(paso); }
  }

  function arranca(el) {
    var d = el.__tfmCont;
    if (!d || d.activo) { return; }
    d.activo = true;
    d.t0 = W.performance.now() + d.retardo;
    el.textContent = pinta(d.piezas, 0);
    activos.push(el);
    if (!lazo) { lazo = W.requestAnimationFrame(paso); }
  }

  // Las cuatro cifras de Resumen viven DEBAJO de la portada a pantalla completa: sin observador
  // contarían a puerta cerrada y al bajar ya estarían quietas.
  //
  // AQUÍ NO SE GUARDA NINGÚN NODO DE STREAMLIT, y eso es lo que hace que esto funcione al cambiar
  // de página. La portada sí puede permitírselo —vive y muere en Resumen—, pero este script cruza
  // páginas, y section[data-testid="stMain"] NO SOBREVIVE al cruce: Streamlit lo reconstruye. Un
  // observador montado sobre el nodo viejo queda hablándole a un huérfano y no vuelve a disparar
  // nunca. Se midió llegando a Resultados con la página desplazada: las tres cifras ni siquiera
  // llegaban a registrarse (__tfmCont sin definir), porque las altas de nodos ocurrían en el
  // árbol nuevo y el MutationObserver seguía escuchando el viejo. Por eso:
  //   · el IntersectionObserver va contra el VIEWPORT en vez de contra la sección con scroll. Para
  //     un elemento dentro de un contenedor que hace scroll la respuesta es idéntica —comprobado
  //     sobre la misma cifra: 0,00 bajo la portada, 1,00 con la tarjeta en pantalla—, y el
  //     viewport no puede quedarse obsoleto.
  //   · el MutationObserver va contra doc.body, que Streamlit tampoco sustituye nunca.
  var io = new W.IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (e.isIntersecting) { io.unobserve(e.target); arranca(e.target); }
    });
  }, { threshold: 0.5 });

  // ...y aun así lo que YA se ve no espera al observador: arranca en esta misma tarea. Es el
  // mismo seguro que se puso en el revelado de la portada, y por el mismo motivo: el primer
  // disparo del observador es asíncrono, y si Streamlit sustituye el nodo entremedias la
  // observación se queda hablándole a un huérfano. Lo que se ve al llegar —los tres AUC de
  // Resultados— no puede depender de esa carrera.
  function seVe(el) {
    var r = el.getBoundingClientRect();
    return r.bottom > 0 && r.top < W.innerHeight * 0.9;
  }

  function registra() {
    var nodos = doc.querySelectorAll('.count-up');
    for (var i = 0; i < nodos.length; i++) {
      var el = nodos[i];
      if (el.__tfmCont) { continue; }
      var original = el.textContent;
      var piezas = trocea(original);
      var hayNum = false;
      for (var j = 0; j < piezas.length; j++) { if (piezas[j].fijo === undefined) { hayNum = true; } }
      if (!hayNum) { continue; }
      // La cascada sale del sitio que ocupa la tarjeta en SU FILA, no del orden en que el
      // observador las va viendo: así el barrido es de izquierda a derecha siempre, aunque las
      // cuatro entren en pantalla en el mismo fotograma.
      var col = el.closest('div[data-testid="stColumn"]');
      var pos = (col && col.parentElement)
        ? Array.prototype.indexOf.call(col.parentElement.children, col) : 0;
      el.__tfmCont = { original: original, piezas: piezas, retardo: BASE + pos * ESCALON };
      if (seVe(el)) { arranca(el); } else { io.observe(el); }
    }
  }

  // Streamlit reconstruye sus nodos en cada rerun y los nuevos nacen sin registrar. Se amortigua
  // con rAF porque un solo rerun dispara decenas de mutaciones, y solo se observan altas y bajas
  // de nodos: escribir textContent es un cambio de dato, no de estructura, así que la cuenta en
  // curso no realimenta el observador.
  var toca = false;
  var mo = new W.MutationObserver(function () {
    if (toca) { return; }
    toca = true;
    W.requestAnimationFrame(function () { toca = false; registra(); });
  });
  mo.observe(doc.body, { childList: true, subtree: true });

  W.__tfmContador = { parar: function () {
    io.disconnect();
    mo.disconnect();
    if (lazo) { W.cancelAnimationFrame(lazo); lazo = 0; }
    for (var i = 0; i < activos.length; i++) {
      if (activos[i].isConnected) { activos[i].textContent = activos[i].__tfmCont.original; }
    }
    activos = [];
  } };
  // Al salir de la página, Streamlit desmonta este iframe y con él muere el realm donde viven
  // estas funciones; los observadores, que son del documento padre, se quedarían enganchados a
  // callbacks muertos. Se sueltan a mano en el desmontaje.
  window.addEventListener('pagehide', function () {
    if (W.__tfmContador) { W.__tfmContador.parar(); W.__tfmContador = null; }
  });

  registra();
})();
</script>
"""


def contadores_js(pagina):
    """Monta el contador. `pagina` es la guarda de "solo al entrar": mientras no cambie, los
    reruns remontan el iframe sin volver a disparar la cuenta."""
    with st.container(key="contador_js"):
        components.html(
            _CONTADOR_JS.replace("__MILLAR__", json.dumps(MILLAR))
                        .replace("__DECIMAL__", json.dumps(DECIMAL))
                        .replace("__PAGINA__", json.dumps(pagina)),
            height=0, width=0)


# ═══════════════════════════════════════════════════════════════════════
# PAGINA 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "overview":
    # La portada va ANTES del titular y a sangre: al abrir la página es lo único que se ve, y
    # todo lo demás —empezando por header()— llega subiendo por encima al hacer scroll. Por eso
    # el resto del cuerpo vive dentro de ov_sheet, que es la hoja opaca que la tapa; el porqué
    # de cada pieza está en el bloque de portada_resumen().
    portada_resumen()

    with st.container(key="ov_sheet"):
        header(S("ov_eyebrow"), S("ov_title"), S("ov_subtitle"))

        # La figura de arquitectura abre el cuerpo, justo encima del párrafo que la explica: se
        # ve el mapa y se lee después el texto que lo recorre. Va a ancho completo de la columna
        # —el diagrama es apaisado y cualquier recorte lo dejaría ilegible— dentro de una
        # .info-card como el resto de la página, con menos aire lateral que las demás porque
        # aquí el recuadro discontinuo del propio dibujo ya hace de marco interior.
        st.markdown(f'<div class="info-card arch-card">{arquitectura_svg()}</div>',
                    unsafe_allow_html=True)

        # El párrafo llega de i18n.py como prosa con <b>, sin un solo atributo de estilo: el
        # tamaño, el color y el realce los pone .lead-card p / .lead-card p b en la hoja de
        # estilos. Antes cada <b> traía su color incrustado y el texto era irrevisable.
        st.markdown(f'<div class="info-card lead-card" style="margin-bottom:20px;">'
                    f'<p>{S("ov_lead")}</p></div>', unsafe_allow_html=True)

        # Las cuatro cifras vuelven a ser tarjetas normales de la página. En la versión anterior eran
        # la capa que subía y tapaba el hero y vivían DENTRO del componente; ahora quien tapa la
        # lámina es la página ENTERA, así que el bloque recupera su sitio y su forma —.info-card
        # .stat-card, la misma de las demás filas de cifras de la aplicación— y la aparición se la da
        # gratis el revelado por scroll, que vale para todos los bloques de la hoja.
        st.markdown(f'<div class="section-title">{S("ov_stats_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("ov_stats_sub")}</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        # Las dos primeras cifras pasan por mil() en vez de ir escritas: estaban puestas a mano
        # con el punto de millar español ("29.400"), que en inglés se lee como 29 coma 4.
        stats = [(mil(29400), S("ov_stat_bronze")), (mil(7831), S("ov_stat_silver")),
                 ("89", S("ov_stat_features")), ("86% / 14%", S("ov_stat_balance"))]
        # count-up: la cifra sube desde cero al entrar en pantalla. Marca de CLASE y no un
        # data-attribute porque las clases son lo que este fichero ya usa por todas partes y
        # sobreviven con seguridad al saneado del HTML de Streamlit. El valor no viaja en la
        # marca: el script trocea este mismo texto, así que "86% / 14%" cuenta sus dos cifras y
        # las dos primeras heredan el separador de millar del idioma sin repetir aquí mil().
        for col, (num, lab) in zip(cols, stats):
            with col:
                st.markdown(f'<div class="info-card stat-card"><div class="stat-num count-up">{num}</div><div class="stat-label">{lab}</div></div>', unsafe_allow_html=True)

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
            # Salto a Gobernanza por el mismo camino que el menú y el buscador. El st.rerun()
            # explícito hace falta porque esto NO es un callback: _navegar escribe el estado y
            # el resto del script ya se está ejecutando con la página vieja.
            if st.button(S("ov_goto_gov"), key="ir_gobernanza"):
                _navegar("governance")
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

    portada_js()
    # Fuera de ov_sheet, como portada_js(): dentro sería un bloque más de la hoja y el revelado
    # por scroll lo trataría como contenido, animando un iframe que no pinta nada.
    contadores_js("overview")

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 2 — GOBERNANZA
# ═══════════════════════════════════════════════════════════════════════
elif page == "governance":
    header(S("gov_eyebrow"), S("gov_title"), S("gov_subtitle"))

    # Vía tabs_i18n y no st.tabs directo: los tres rótulos cambian con el idioma y el tab
    # abierto se perdía al pulsar la bandera. Ver el porqué en tabs_i18n.
    tab_calidad, tab_linaje, tab_stack = tabs_i18n("gov_tabs", key="gov_tabs")

    # ─────────────────────────── TAB A — CALIDAD ───────────────────────────
    with tab_calidad:
        _kpis = [
            (f"{GOV_SUITE['passed']}/{GOV_SUITE['total']}", S("gov_kpi_expect")),
            (nf(GOV_SUITE["pass_rate"], 1), S("gov_kpi_passrate")),
            (mil(GOV_SUITE["registros"]), S("gov_kpi_records")),
            ("15/15", S("gov_kpi_leakage")),
        ]
        # count-up, igual que las cuatro cifras de Resumen y los AUC de Resultados. Las dos
        # razones ("15/15") cuentan sus DOS numeros, que es lo que hace el troceador con
        # cualquier texto mixto; no se les pone un caso especial porque el denominador quieto
        # obligaria a distinguir "15/15" de "86% / 14%", y esa distincion no la puede hacer el
        # script sin que alguien le diga cual es cual desde Python.
        # Aqui las tarjetas no son columnas de st.columns sino celdas de .compare-grid, asi que
        # closest('stColumn') no encuentra nada y las cuatro arrancan con el mismo retardo base:
        # entran a la vez en vez de en cascada. Es correcto para una rejilla propia, donde no hay
        # un orden de columnas de Streamlit al que engancharse.
        st.markdown(
            '<div class="compare-grid" style="grid-template-columns:repeat(4, minmax(0, 1fr));">'
            + "".join(
                f'<div class="info-card stat-card">'
                f'<div class="stat-num count-up">{v}</div>'
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

        # Mismo plegado que el Registro de decisiones (tab C): un expander por dimensión en vez
        # de las 15 expectativas seguidas en una sola tarjeta. Lo que era la cabecera .gov-dim
        # pasa a ser el rótulo del expander, conservando el "{dimensión} · {n}" —el recuento
        # sigue siendo el de todas sus filas—, y el recuadro lo pone ya el propio widget: envolver
        # esto en .info-card dejaría borde dentro de borde. El vestido del expander es la regla
        # global de [data-testid="stExpander"], así que sigue al tema igual que en decisiones.
        _grupos = {}
        for dim, col, regla in S("gov_expectativas"):
            _grupos.setdefault(dim, []).append(
                f'<div class="gov-check">'
                f'<span class="gov-dot" style="background:{STATUS["good"]};"></span>'
                f'<span class="gov-col">{col}</span>'
                f'<span class="gov-rule">{regla}</span>'
                f'<span class="gov-state" style="color:{STATUS["good"]};">passed</span></div>')
        for dim, _checks in _grupos.items():
            with st.expander(f"{dim} · {len(_checks)}"):
                # El <div> envolvente no es decorativo: .gov-check:last-child es quien quita el
                # filete inferior, y solo acierta si las filas son hermanas y cierran el bloque.
                st.markdown(f'<div>{"".join(_checks)}</div>', unsafe_allow_html=True)

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
        # La tabla del escalado y la nota que la explica son el MISMO bloque de discurso, así que
        # tienen que cerrar a la misma altura. Van en UNA sola .compare-grid y no en dos st.columns
        # porque el estirado es nativo del grid: la caja más corta crece hasta la más alta ella
        # sola, sin depender de que height:100% atraviese los divs intermedios de Streamlit ni de
        # cuadrar al píxel el largo del texto en las cinco traducciones. El reparto 1 : 1,15 es el
        # de las columnas que sustituye, y los saltos de 1024 y 768 px de .compare-grid lo
        # reajustan y lo apilan igual que en el resto de la app.
        # La tarjeta reparte en sus filas el sobrante que le quede (space-between, igual que la de
        # Circuito Cuántico): según el idioma y el ancho manda una caja u otra por una línea, y
        # así ese resto se va en cuatro huecos de unos pocos píxeles en vez de amontonarse al pie.
        _srows = "".join(
            f'<div class="kpi-row" style="align-items:flex-start;">'
            f'<span class="kpi-label" style="max-width:56%;">{lab}'
            f'<span style="display:block;font-size:13.5px;color:{t["text_muted"]};'
            f'line-height:1.5;margin-top:3px;">{det}</span></span>'
            f'<span class="kpi-value">{val}</span></div>'
            for lab, val, det in S("gov_scaler"))
        st.markdown(
            f'<div class="compare-grid" '
            f'style="grid-template-columns:minmax(0, 1fr) minmax(0, 1.15fr);">'
            f'<div class="info-card" style="display:flex;flex-direction:column;'
            f'justify-content:space-between;"><div class="kpi-model">'
            f'<span class="kpi-dot" style="background:{C_PRIMARY};"></span>'
            f'{S("gov_scaler_card")}</div>{_srows}</div>'
            f'<div class="clinical-note">{S("gov_scaler_note")}</div>'
            f'</div>', unsafe_allow_html=True)

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

    # Los cuatro KPI de Calidad del dato. Va FUERA de los tres tabs, al nivel de la página: el
    # componente no pinta nada y no tiene por qué vivir dentro de una pestaña. Que las tarjetas
    # estén en el tab que arranca abierto no lo cambia — si algún día dejaran de estarlo, el
    # observador las recogería igual cuando la pestaña se muestre.
    contadores_js("governance")

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
                <div class="kpi-value-auc count-up" style="color:{m['color']};">{nf(m['auc'])}</div>
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
        # SIN spline. Llevaba shape="spline", smoothing=0.4 y era el mismo error cosmético que la
        # curva de respuesta del Predictor evita a propósito: una ROC empírica es una ESCALERA —un
        # peldaño por instancia del test— y suavizarla dibuja una continuidad que los datos no
        # tienen, justo debajo de un subtítulo que promete "punto a punto". Con 1.567 puntos la
        # poligonal se ve igual de limpia y además es literal. (Señalado en
        # INFORME_AUDITORIA_DASHBOARD.md §2.5, que quedó fuera de los grupos aplicados.)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=m["color"], width=2),
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

    # Los tres AUC-ROC de la cabecera. Va al final de la página por el mismo motivo que en
    # Resumen —el script no depende del orden, lo recoge el observador de mutaciones— y así el
    # componente no se cuela entre las tarjetas.
    contadores_js("results")

# ═══════════════════════════════════════════════════════════════════════
# PAGINA 4 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif page == "shap":
    header(S("sh_eyebrow"), S("sh_title"), S("sh_subtitle"))

    # Vía tabs_i18n por lo mismo que los de Gobernanza: con st.tabs pelado, los dos rótulos
    # cambian con la bandera y el widget se daba por nuevo, así que cambiar de idioma —o
    # recargar— devolvía siempre al primer tab.
    tab1, tab2 = tabs_i18n("sh_tabs", key="sh_tabs")

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
# PAGINA 5 — QUANTUM CIRCUIT
# ═══════════════════════════════════════════════════════════════════════
elif page == "circuit":
    header(S("qc_eyebrow"), S("qc_title"), S("qc_subtitle"))

    # DOS PESTAÑAS, y la segunda era una página del menú hasta este cambio. La Esfera de Bloch
    # entra aquí porque es LA MISMA MATERIA a otra escala —un qubit, luego tres, y al final los
    # ocho del modelo, que son los de esta página— y porque un menú de siete entradas dedicaba
    # DOS a visualización cuántica: al dejar de ser la app solo el apoyo de la defensa, ese peso
    # le restaba sitio en el primer nivel a las páginas que mejor explican el proyecto, Resumen
    # y Predictor en Vivo. No se ha tocado NADA de su contenido ni de su interactividad; ha
    # cambiado el nivel al que vive, y solo eso.
    #
    # Vía tabs_i18n y no st.tabs pelado, por lo mismo que Gobernanza y Análisis SHAP: los dos
    # rótulos cambian con la bandera, así que sin esto el widget se da por nuevo y la pestaña
    # abierta se pierde al traducir o al recargar. Y OJO: "circuit" TIENE que estar en
    # _PAGINAS_CON_TABS, o el saneo de ?tab= de más arriba lo borra antes de que tabs_i18n
    # llegue a leerlo y la pestaña no sobreviviría a un F5.
    tab_qc, tab_bloch = tabs_i18n("qc_tabs", key="qc_tabs")

    # ──────────────────── PESTAÑA A — CIRCUITO ZZFEATUREMAP ────────────────────
    with tab_qc:
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
    # PESTAÑA B — ESFERA DE BLOCH
    # ═══════════════════════════════════════════════════════════════════════
    # Lo que era la página 5. Entra con section-title/section-sub y NO con header(): header()
    # abre un contenedor con clave page_enter_<índice de la página>, y llamarlo dos veces en la
    # misma pasada choca por clave duplicada. Sus dos textos son los que ya tenía —bl_title es
    # además la fila con la que se sigue llegando aquí desde el buscador (ver SEARCH_PREFIX)—;
    # el antetítulo bl_eyebrow se retira, porque un antetítulo nombra la categoría de una
    # PÁGINA y esto ya no lo es.
    #
    # Las tres secciones de dentro (la esfera de un qubit, el entrelazamiento de tres y el
    # ZZFeatureMap real de ocho) van juntas en esta pestaña y no repartidas: la tercera usa
    # var_code y val del selector de la primera, a propósito y explicado ahí abajo.
    with tab_bloch:
        st.markdown(f'<div class="section-title">{S("bl_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("bl_subtitle")}</div>', unsafe_allow_html=True)

        # Qué es una esfera de Bloch, antes de enseñar una. El subtítulo dice qué se está viendo
        # (la codificación), pero da por sabido el soporte donde se dibuja; a quien llega desde el
        # lado clínico la figura le queda en una bola con una flecha. Va en .clinical-note, el mismo
        # recurso con el que el Predictor en Vivo aclara qué estima su formulario: nota de entrada,
        # una sola vez, antes de los controles.
        st.markdown(f"""
        <div class="clinical-note" style="margin-bottom:16px;">
        {S("bl_what_note")}
        </div>
        """, unsafe_allow_html=True)

        # Contenedor con clave (genera .st-key-bloch_row) para poder estirar la gráfica 3D hasta el alto
        # de la columna izquierda y que ambas tarjetas cierren alineadas abajo — ver CSS .st-key-bloch_row.
        bloch_row = st.container(key="bloch_row")
        col1, col2 = bloch_row.columns([1, 1.3])
        with col1:
            # LOS DOS CONTROLES LLEVAN CLAVE, y no es cosmética: sin ella, la IDENTIDAD del widget
            # para Streamlit es su rótulo, y los dos rótulos cambian con la bandera. El resultado
            # medido era que pulsar un idioma devolvía el selector a LBXGH y el deslizador a su
            # valor por defecto —quien estuviera mirando la glucosa a 6,9 aparecía en HbA1c a 5,7—,
            # mientras que los ocho deslizadores del Predictor en Vivo, que sí tienen clave desde
            # siempre, conservaban el suyo. Era la misma incoherencia que tabs_i18n arregló para las
            # pestañas, en la única página donde quedaba.
            #
            # La del selector es FIJA porque lo que guarda es el CÓDIGO NHANES ("LBXGH"), que no se
            # traduce. La del deslizador lleva el código DENTRO porque su rango, su paso y su unidad
            # son propios de cada variable: con una clave común, el valor guardado para la glucosa
            # (100 mg/dL) reaparecería al saltar a HbA1c, cuyo eje llega a 15.
            #
            # Eso NO convierte la clave en una memoria por variable, y conviene no confundirlo
            # (comprobado en el navegador): mientras hay otra variable elegida, el deslizador de
            # esta no se dibuja, así que Streamlit poda su estado por «stale» —el mismo mecanismo
            # que se explica largo en tabs_i18n— y al volver arranca de nuevo en su valor por
            # defecto. Que es justo lo que se quiere de un cambio de variable. Lo que la clave
            # arregla es lo otro: los reruns en los que la variable NO cambia (bandera, tema,
            # colapsar la barra), donde antes el rótulo traducido hacía que el widget se diera por
            # nuevo y se perdiera la posición.
            var_code = st.selectbox(S("bl_var"), list(QSVM_FEATURES.keys()),
                                     format_func=lambda c: f"{c} · {q_label(c)}", key="bl_var")
            v = QSVM_FEATURES[var_code]
            lo, hi = v["range"]
            val = st.slider(S("bl_value").format(unidad=q_unit(var_code)),
                            float(lo), float(hi), float(v["default"]),
                            step=v["step"], format=v["fmt"], key=f"bl_val_{var_code}")

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
            # Superficie esférica sombreada + círculos máximos: la misma base que la Q-sphere
            # de más abajo, definida una sola vez en esfera_base().
            fig = esfera_base()
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
            # ── Arco de θ, como en los diagramas canónicos de la esfera de Bloch ──────────
            # θ ES el valor clínico una vez normalizado, así que dibujarlo cierra el circuito entre
            # la cifra de la tarjeta de la izquierda y la figura: se ve DE DÓNDE sale el ángulo. Va
            # en el plano XZ porque φ=0 (ver el vector, más arriba), a radio corto para no tocar ni
            # el vector ni la superficie, y en tinta apagada porque es cota, no dato.
            # Por debajo de ~0,05 rad no se dibuja: un arco de dos píxeles no se lee como arco, se
            # lee como un borrón junto al eje.
            if theta > 0.05:
                _arc = np.linspace(0, theta, 40)
                fig.add_trace(go.Scatter3d(x=0.32*np.sin(_arc), y=np.zeros_like(_arc), z=0.32*np.cos(_arc),
                                            mode="lines", line=dict(color=t["text_muted"], width=2),
                                            showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter3d(x=[0.44*np.sin(theta/2)], y=[0.0], z=[0.44*np.cos(theta/2)],
                                            mode="text", text=["θ"],
                                            textfont=dict(family=PLOTLY_MONO, size=13, color=t["text_muted"]),
                                            showlegend=False, hoverinfo="skip"))
            # ── Valor de la variable, EN el punto ────────────────────────────────────────
            # Sin esto la esfera enseña una posición pero no dice de qué, y hay que mirar al
            # selector de al lado para saber qué representa la flecha. Dos líneas y no una: el
            # rótulo más largo ("Glucosa en ayunas = 100 mg/dL") pide unos 200 px y se saldría de
            # la tarjeta; partido por el "=" la línea más ancha baja a la mitad larga.
            # Siempre a la IZQUIERDA: el vector vive en el semiplano x≥0 (φ=0 y sen θ≥0 en [0,π]),
            # así que en pantalla sale del centro hacia arriba-izquierda y ese lado queda libre en
            # todo el recorrido.
            # Y arriba o abajo SEGÚN EL HEMISFERIO, que no es un adorno: en los extremos del
            # deslizador el punto aterriza justo en un polo, y ahí |0⟩ y |1⟩ ya ocupan sitio —el de
            # arriba por encima, el de abajo por debajo—. Alejándose del ecuador se esquivan los
            # dos: con la variable al mínimo la etiqueta cae bajo el polo norte, y al máximo sube
            # sobre el polo sur. Sin esta regla, al llevar el deslizador al tope la etiqueta tapaba
            # |1⟩ por completo (comprobado exportando la figura).
            _dec = 1 if v["step"] < 1 else 0
            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz], mode="text",
                text=[f"{q_label(var_code)}<br>{nf(val, _dec)} {q_unit(var_code)}"],
                textposition="bottom left" if pz >= 0 else "top left",
                textfont=dict(family=PLOTLY_MONO, size=12, color=C_QUANTUM),
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

        # ═══════════════════════════════════════════════════════════════════
        # ENTRELAZAMIENTO DE TRES QUBITS
        # ═══════════════════════════════════════════════════════════════════
        # POR QUÉ AQUÍ Y NO AL FINAL DE CIRCUITO CUÁNTICO. La nota de arriba (bl_note) cierra la
        # página diciendo que el entrelazamiento "solo es representable en el espacio conjunto":
        # esta sección es esa frase hecha figura. El argumento entero —que un estado entrelazado
        # NO cabe en una esfera de Bloch por qubit— solo se entiende habiendo visto antes la
        # esfera de un qubit, y esa esfera está justo encima; en Circuito Cuántico la sección
        # habría llegado sin ese precedente, entre las especificaciones del ZZFeatureMap y las
        # cifras de entrenamiento, que es una página de fichas y no de didáctica.
        #
        # POR QUÉ TRES Y NO DOS, que es donde estaba la sección antes. Con dos qubits el recorrido
        # termina en el estado de Bell y las tres cifras locales caen a su extremo; con tres, el
        # GHZ da EXACTAMENTE esas mismas tres cifras —el titular no se mueve—, pero aparece una
        # cuarta que dos qubits no pueden enseñar: la concurrencia del par q₀q₁, que sube a 1 en el
        # paso 2 y vuelve a 0 en el 3. Ver ent_local(): el tercer CNOT crea entrelazamiento global
        # DESHACIENDO el del par, y eso es lo que hay que saber para leer la matriz de información
        # mutua por parejas del ZZFeatureMap de ocho qubits que viene justo debajo.
        #
        # El paso ES el estado. Lo único que persiste entre reruns es el entero `ent_paso`; el
        # vector se recalcula desde |000⟩ en cada pasada (ver ent_statevector). Los botones van con
        # on_click y no con `if st.button(...)`: el callback corre ANTES de que el script se
        # reejecute, así que el circuito, la Q-sphere y las métricas se pintan ya con el paso
        # nuevo. Con la forma `if` haría falta un st.rerun() explícito para no ir un paso por
        # detrás — el mismo motivo por el que el toggle de la sidebar sí lo lleva.
        st.session_state.setdefault("ent_paso", 0)
        st.session_state.setdefault("ent_counts", None)

        def _ent_paso(destino):
            """Mueve el circuito al paso `destino` y tira las mediciones anteriores.

            Lo segundo importa tanto como lo primero: un histograma de |00⟩+|11⟩ bajo un circuito
            que ya no tiene el CNOT sería una figura que miente. Al cambiar de paso, la muestra
            vuelve a estar vacía y hay que volver a pedirla.
            """
            st.session_state.ent_paso = destino
            st.session_state.ent_counts = None

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{S("bl_ent_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("bl_ent_sub")}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="clinical-note" style="margin-bottom:16px;">
        {S("bl_ent_intro")}
        </div>
        """, unsafe_allow_html=True)

        paso = st.session_state.ent_paso
        psi = ent_statevector(paso)
        local = ent_local(psi)

        # Botonera: cada puerta se habilita SOLO en su turno. Un circuito no admite aplicar el
        # CNOT antes que la Hadamard —saldría un estado producto sin entrelazar y la sección
        # perdería el hilo—, y el segundo CNOT antes del primero rompería la cadena que construye
        # el GHZ, así que la secuencia la impone el propio control en vez de un aviso a posteriori.
        # El de reiniciar siempre está vivo: se puede rehacer el recorrido entero.
        b1, b2, b3, b4 = st.columns([1.15, 1.5, 1.5, 1])
        for _col, _clave, _rotulo, _destino in (
                (b1, "ent_h", "bl_ent_btn_h", 1),
                (b2, "ent_cnot1", "bl_ent_btn_cnot1", 2),
                (b3, "ent_cnot2", "bl_ent_btn_cnot2", 3)):
            with _col:
                st.button(S(_rotulo), key=_clave, width="stretch",
                          disabled=paso != _destino - 1, on_click=_ent_paso, args=(_destino,))
        with b4:
            st.button(S("bl_ent_btn_reset"), key="ent_reset", width="stretch",
                      disabled=paso == 0, on_click=_ent_paso, args=(0,))

        # Las tres frases del paso actual, en la misma tarjeta de prosa que usa "Cómo funciona"
        # del Circuito Cuántico. Van FUERA de las columnas y a ancho completo: es el texto que
        # explica las dos figuras de debajo, no el pie de ninguna de las dos.
        st.markdown(f'<div class="info-card" style="margin-top:12px;">'
                    f'<p class="qc-prose">{S("bl_ent_step_note")[paso]}</p></div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ent_row = st.container(key="ent_row")
        col1, col2 = ent_row.columns([1, 1.15])
        with col1:
            st.markdown(f'<div class="section-title">{S("bl_ent_circuit_title")}</div>', unsafe_allow_html=True)
            # fig-vector y no .fig-card a secas: este circuito se dibuja con la paleta activa, así
            # que necesita la superficie del tema debajo y no el blanco fijo de las láminas raster
            # —ver el bloque .fig-card.fig-vector de la hoja de estilos, donde está el porqué—.
            st.markdown(f'<div class="fig-card fig-vector" style="padding:18px 14px;">'
                        f'{ent_circuito_svg(paso, st.session_state.ent_counts is not None)}</div>',
                        unsafe_allow_html=True)
            # Las cuatro cifras de ent_local(), que son el argumento cuantitativo de la sección.
            # Mismo patrón .kpi-row que la tarjeta de amplitudes de arriba, a propósito: se leen
            # como su continuación —allí el estado de UN qubit, aquí lo que queda de él dentro
            # del trío—. La longitud del vector va la primera porque es la que se puede contrastar
            # con la figura de esta misma página: 1 = hay flecha que dibujar, 0 = no la hay. La
            # concurrencia va la última porque es la que solo se entiende con las otras tres ya
            # leídas: dice que lo que hay entrelazado es el conjunto y no las parejas.
            _kpi_lab = S("bl_ent_kpi")
            _kpi_val = [nf(local["r"], 3), nf(local["pureza"], 3),
                        f'{nf(local["entropia"], 3)} {S("bl_ent_bits")}',
                        nf(local["concurrencia"], 3)]
            st.markdown(
                '<div class="info-card" style="margin-top:14px;">'
                + "".join(f'<div class="kpi-row"><span class="kpi-label">{l}</span>'
                          f'<span class="kpi-value">{v}</span></div>'
                          for l, v in zip(_kpi_lab, _kpi_val))
                + "</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="section-title">{S("bl_ent_qsphere_title")}</div>', unsafe_allow_html=True)
            # key estable por el mismo motivo que en la esfera de arriba: sin ella Streamlit
            # remonta el iframe en cada rerun y la figura parpadea. Aquí además el rerun ocurre
            # en cada pulsación, así que se notaría el triple.
            st.plotly_chart(ent_qsphere_fig(psi), width="stretch", key="ent_qsphere",
                            config={"displayModeBar": False, "responsive": True})

        # ── Medición ────────────────────────────────────────────────────────
        # La Q-sphere enseña el estado; esto enseña lo que se MIDE, que es lo único observable y
        # la prueba empírica del entrelazamiento: 000 y 111 a partes iguales, y las otras SEIS
        # combinaciones nunca. Se habilita desde el primer paso y no solo en el GHZ, porque el
        # contraste es parte de la lección — tras la Hadamard sola salen 000 y 100, o sea q0 al
        # azar y los otros dos fijos en 0; cada CNOT que se añade ata un qubit más al primero.
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{S("bl_ent_meas_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{S("bl_ent_meas_sub")}</div>', unsafe_allow_html=True)

        m1, m2 = st.columns([2.2, 1])
        with m1:
            # Con clave por lo mismo que los dos controles de arriba: su rótulo se traduce, así que
            # sin ella cambiar de bandera devolvía el número de disparos a 1.000 dejando en pantalla
            # el histograma de la tanda anterior —que sí sobrevive, porque vive en una clave nuestra—.
            # O sea, el control decía 1.000 y la figura de al lado seguía contando 5.000.
            shots = st.slider(S("bl_ent_meas_n"), 100, 10000, 1000, step=100, key="ent_shots")
        with m2:
            # El aire de arriba alinea el botón con el carril del deslizador de al lado: el
            # slider gasta su primera línea en el rótulo y sin esto el botón subía por encima.
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            if st.button(S("bl_ent_meas_btn"), key="ent_medir", width="stretch"):
                # Muestreo multinomial sobre |ψ|². Es EXACTAMENTE lo que hace un simulador ideal
                # sin ruido: cada disparo es un sorteo independiente con las probabilidades de
                # Born, y el simulador no añade nada más cuando no se le pide modelo de ruido.
                # Sin semilla fija a propósito: dos tandas seguidas dan cifras distintas, que es
                # justo lo que se quiere enseñar —la proporción es estable, el recuento exacto
                # no— y con semilla fija parecería un resultado calculado en vez de muestreado.
                rng = np.random.default_rng()
                st.session_state.ent_counts = rng.multinomial(shots, psi ** 2).tolist()

        counts = st.session_state.ent_counts
        if counts is None:
            # Hueco explícito en vez de una figura vacía: una gráfica con ocho barras a cero se
            # lee como un resultado ("no sale nada"), que es lo contrario de "aún no has medido".
            st.markdown(f'<div class="info-card" style="text-align:center; color:{t["text_muted"]}; '
                        f'padding:34px 18px;">{S("bl_ent_meas_empty")}</div>', unsafe_allow_html=True)
        else:
            total = sum(counts)
            fig = go.Figure()
            # Los ocho resultados posibles SIEMPRE en el eje, incluidos los que salen a cero: que
            # las seis combinaciones mixtas aparezcan etiquetadas y vacías es el dato. Si se
            # filtraran las barras nulas, la figura enseñaría dos resultados equiprobables y no
            # habría forma de ver que faltan otros seis — y con tres qubits ese vacío pesa más
            # que con dos: seis de ocho, no dos de cuatro.
            fig.add_trace(go.Bar(
                x=[f"|{b}⟩" for b in ENT_BASE], y=counts,
                marker_color=[C_QUANTUM if c else hex_to_rgba(t["text"], 0.10) for c in counts],
                text=[mil(c) for c in counts], textposition="outside",
                textfont=dict(family=PLOTLY_MONO, size=13, color=t["text_secondary"]),
                customdata=[[pct(c / total)] for c in counts], showlegend=False, cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>%{y} " + S("bl_ent_hover_shots") + "<br>%{customdata[0]}<extra></extra>",
            ))
            plotly_layout(fig, height=320,
                          xaxis=dict(showgrid=False, fixedrange=True,
                                     tickfont=dict(family=PLOTLY_MONO, size=15, color=t["text"])),
                          yaxis=dict(title=dict(text=S("bl_ent_meas_yaxis"), font=dict(size=13)),
                                     showgrid=True, gridcolor=GRID, fixedrange=True,
                                     range=[0, max(counts) * 1.18]),
                          margin=dict(l=60, r=20, t=26, b=40))
            st.plotly_chart(fig, width="stretch", key="ent_hist", config={"displayModeBar": False})
            st.markdown(f'<div class="clinical-note">{S("bl_ent_meas_note")[paso]}</div>',
                        unsafe_allow_html=True)

        # Nota de honestidad metodológica, en la misma línea que bl_note: la sección de arriba
        # dice qué NO reproduce la esfera del ZZFeatureMap, y esta dice con qué está calculado lo
        # que se acaba de ver. En un TFM eso no es un pie de página, es parte del resultado.
        st.markdown(f'<div class="clinical-note" style="margin-top:16px;">{S("bl_ent_impl_note")}</div>',
                    unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════
        # EL ZZFEATUREMAP REAL (8 qubits)
        # ═══════════════════════════════════════════════════════════════════
        # Tercer escalón de la página: un qubit → dos → los ocho del modelo. Cada uno usa el
        # vocabulario del anterior, y ese encadenamiento es lo que hace legible este último — |r|
        # es LA MISMA cifra que la sección de 2 qubits acaba de explicar con el estado de Bell.
        # Sin ese precedente, estas ocho barras serían ocho números entre 0 y 1 sin significado.
        #
        # NO LLEVA CONTROLES PROPIOS: usa el selector y el deslizador del principio de la página.
        # Añadir un segundo par para las mismas ocho variables habría dejado dos controles que
        # dicen lo mismo. Y así el gesto de la página es uno solo: mueves una variable clínica y
        # ves su efecto a las tres escalas, la esfera de arriba y las dos figuras de aquí.
        # Además tiene premio: con entrelazamiento lineal, mover UNA variable solo altera su
        # qubit y sus vecinos inmediatos (verificado barriendo HbA1c por su rango completo: q0,
        # q1 y q2 se mueven; de q3 a q7 no cambian ni un decimal). El cono de luz del circuito se
        # ve arrastrando el deslizador.
        #
        # La sección entera es opcional: si falta scaler_correcto.json no hay forma de escalar las
        # features como las escaló el pipeline, y se omite en silencio en vez de inventar una
        # normalización distinta — mismo criterio que el diagrama del circuito de 8 qubits, que
        # solo se pinta si su PNG está en disco.
        _esc = _load_scaler_and_medians()
        if _esc is not None and all(f in _esc["features"] for f in QSVM_FEATURES):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">{S("bl_zz_title")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-sub">{S("bl_zz_sub")}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="clinical-note" style="margin-bottom:16px;">
            {S("bl_zz_intro")}
            </div>
            """, unsafe_allow_html=True)

            # Las 8 features en su valor de referencia, salvo la que el lector tiene elegida
            # arriba. var_code y val vienen del selector y el deslizador de la esfera: `with`
            # no abre ámbito en Python, así que siguen vivos aquí.
            _perfil = {c: float(f["default"]) for c, f in QSVM_FEATURES.items()}
            _perfil[var_code] = float(val)
            _codigos = list(QSVM_FEATURES.keys())
            r_len, MI = zz_metricas(tuple(_perfil[c] for c in _codigos))

            st.markdown(
                f'<div class="section-sub" style="margin-top:-6px;">'
                f'{S("bl_zz_current").format(var=q_label(var_code), val=nf(val, 1 if v["step"] < 1 else 0), unidad=q_unit(var_code))}'
                f'</div>', unsafe_allow_html=True)

            zz_row = st.container(key="zz_row")
            zc1, zc2 = zz_row.columns([1, 1.15])
            with zc1:
                st.markdown(f'<div class="section-title">{S("bl_zz_r_title")}</div>', unsafe_allow_html=True)
                # Barras horizontales con el mismo tratamiento que el ranking RF del Circuito
                # Cuántico: la lista es la misma y leerlas igual ayuda a cruzarlas. Se invierte
                # el orden porque Plotly apila el eje de categorías de abajo arriba y así el
                # primer qubit queda arriba, como en el diagrama del circuito.
                _orden = list(reversed(_codigos))
                _vals = [float(r_len[_codigos.index(c)]) for c in _orden]
                fig = go.Figure()
                # La variable que el lector está moviendo va en el acento de marca y el resto en
                # el cuántico: sin eso, al arrastrar el deslizador se ve cambiar una barra sin
                # saber cuál se estaba tocando.
                fig.add_trace(go.Bar(
                    x=_vals, y=_orden, orientation="h", cliponaxis=False,
                    marker_color=[C_PRIMARY if c == var_code else C_QUANTUM for c in _orden],
                    text=[nf(x, 3) for x in _vals], textposition="outside",
                    textfont=dict(family=PLOTLY_MONO, size=12.5, color=t["text_secondary"]),
                    customdata=[[q_label(c)] for c in _orden], showlegend=False,
                    hovertemplate="<b>%{customdata[0]}</b><br>|r| = %{x:.3f}<extra></extra>"))
                plotly_layout(fig, height=340,
                              xaxis=dict(title=dict(text=S("bl_zz_r_xaxis"), font=dict(size=13)),
                                         range=[0, 1.16], showgrid=True, gridcolor=GRID, fixedrange=True),
                              yaxis=dict(showgrid=False, fixedrange=True,
                                         tickfont=dict(family=PLOTLY_MONO, size=12, color=t["text"])),
                              margin=dict(l=88, r=54, t=20, b=42))
                st.plotly_chart(fig, width="stretch", key="zz_r", config={"displayModeBar": False})

            with zc2:
                st.markdown(f'<div class="section-title">{S("bl_zz_mi_title")}</div>', unsafe_allow_html=True)
                # Diagonal en blanco: I(i:i) no es cero, es la entropía del propio qubit, y
                # pintarla en la misma escala que los pares sería comparar dos magnitudes
                # distintas. Con NaN, Plotly deja la celda al color del fondo.
                _z = MI.copy().astype(float)
                np.fill_diagonal(_z, np.nan)
                _txt = [["" if i == j else nf(_z[i, j], 2) for j in range(ZZ_N)] for i in range(ZZ_N)]
                fig = go.Figure(go.Heatmap(
                    z=_z, x=_codigos, y=_codigos,
                    # Escala FIJA de 0 al tope teórico (2 bits). Adaptarla al máximo de cada
                    # render habría hecho que el mismo color significara cosas distintas según
                    # dónde estuviera el deslizador, que en un control en vivo es lo peor
                    # posible. El extremo inferior es la superficie de la tarjeta, así que un
                    # par sin correlación se ve literalmente vacío.
                    zmin=0, zmax=ZZ_MI_MAX,
                    colorscale=[[0.0, t["surface_alt"]], [0.25, RAMP[0]], [0.5, RAMP[1]],
                                [0.75, RAMP[2]], [1.0, RAMP[3]]],
                    text=_txt, texttemplate="%{text}", xgap=2, ygap=2,
                    textfont=dict(family=PLOTLY_MONO, size=11, color=t["text"]),
                    hovertemplate="<b>%{y} · %{x}</b><br>%{z:.3f} " + S("bl_ent_bits") + "<extra></extra>",
                    colorbar=dict(title=dict(text=S("bl_zz_mi_cbar"), font=dict(size=12)),
                                  thickness=10, outlinewidth=0, len=0.86,
                                  tickfont=dict(family=PLOTLY_MONO, size=11)),
                ))
                plotly_layout(fig, height=340,
                              xaxis=dict(showgrid=False, fixedrange=True, side="top",
                                         tickfont=dict(family=PLOTLY_MONO, size=10.5, color=t["text_secondary"])),
                              # autorange invertido: la primera variable arriba, para que la
                              # diagonal caiga de arriba-izquierda a abajo-derecha como se lee
                              # cualquier matriz, y no al revés.
                              yaxis=dict(showgrid=False, fixedrange=True, autorange="reversed",
                                         tickfont=dict(family=PLOTLY_MONO, size=10.5, color=t["text_secondary"])),
                              margin=dict(l=76, r=10, t=54, b=10))
                st.plotly_chart(fig, width="stretch", key="zz_mi", config={"displayModeBar": False})

            st.markdown(f'<div class="clinical-note">{S("bl_zz_note")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="clinical-note" style="margin-top:16px;">{S("bl_zz_caveat")}</div>',
                        unsafe_allow_html=True)

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

    inputs = {}
    items = list(QSVM_FEATURES.items())
    # Una fila de columnas POR PAREJA, y no dos columnas para las ocho de golpe. Con un único
    # st.columns(2) cada columna apila sus cuatro variables por su cuenta, así que basta con que
    # un pie tenga una línea de más para que esa columna entera baje y sus sliders dejen de
    # cuadrar con los de al lado. Y pasa de dos maneras: fija —el criterio ADA solo cuelga de
    # HbA1c— y móvil, porque el aviso de extrapolación aparece y desaparece al arrastrar un
    # slider fuera de ±3 sd, o sea que el desajuste cambiaba mientras el usuario juega con la
    # página. Reabriendo las columnas en cada pareja, cada fila arranca alineada y una
    # diferencia de alto se queda dentro de su fila en vez de arrastrar todo lo que va debajo.
    # Se paga en alto: la celda de pie corto queda con aire de sobra hasta la fila siguiente.
    # Es lo que vale una rejilla alineada, y es preferible a fijar min-height en los pies, que
    # obligaría a adivinar cuántas líneas ocupa cada texto en cada idioma y ancho de ventana.
    for _fila in range(0, len(items), 2):
        for _col, (code, v) in zip(st.columns(2), items[_fila:_fila + 2]):
            with _col:
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
        # los 5 pasos de RAMP, de bajo a alto riesgo, en vez del rojo→verde de referencia.
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
                                 format_func=lambda c: f"{c} · {q_label(c)}",
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
            # frías (tinta y acero) todo lo demás es de la familia azul, así que el oro de marca no
            # compite con ninguna y da a la gráfica el foco que le faltaba. No es una tercera
            # serie: es cromo de interfaz —el eco de la posición del slider—, el mismo papel que
            # ya tiene en sliders y anillos de foco. El punto en sí conserva el color de
            # LightGBM, que es de quien es el valor; lo cálido es el halo y la plomada.
            _fig.add_trace(go.Scatter(x=[_cur_x, _cur_x], y=[0, risk], mode="lines",
                                      line=dict(color=hex_to_rgba(C_PRIMARY, 0.50), width=1.5, dash="dot"),
                                      showlegend=False, hoverinfo="skip"))
            # Halo suave + ANILLO abierto. Con dos discos rellenos superpuestos el foco salía
            # como una moneda opaca —un relleno al 28 % sobre el fondo de la tarjeta es una
            # mancha—; el anillo lo convierte en una diana nítida y el halo solo aporta la
            # irradiación. Va en C_MID2, que es la familia del brillo y además SIGUE AL TEMA:
            # antes era un literal de la paleta oscura y en claro pintaba el halo con el color
            # del otro tema.
            _fig.add_trace(go.Scatter(x=[_cur_x], y=[risk], mode="markers",
                                      marker=dict(size=28, color=hex_to_rgba(C_MID2, 0.10),
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
            # El valor actual, anclado al eje y con filete dorado para que se lea como parte del
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
