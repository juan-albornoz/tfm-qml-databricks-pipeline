"""Catálogo de textos ES/EN/DE/FR/IT del dashboard.

Separado de app.py a propósito: el texto es lo único que cambia entre idiomas y
tenerlo en un solo fichero permite revisar la traducción de corrido, sin leerla
entre etiquetas HTML y llamadas a Plotly.

Dos reglas que sostienen el resto:

1. LAS CLAVES SON ESTABLES Y EN INGLÉS. La página "Gobernanza" es `governance`
   en el código y solo se convierte en texto visible al pintarse. Antes el
   enrutado comparaba contra el rótulo español (`if page == "Gobernanza"`), así
   que traducir el menú habría roto la navegación.

2. LOS CATÁLOGOS QUE NO SON EL ESPAÑOL PUEDEN ESTAR INCOMPLETOS. `S()` en app.py
   cae al español cuando falta una clave, de modo que las páginas todavía sin
   traducir siguen funcionando en vez de reventar con KeyError. Esto es lo que
   permite traducir página a página, y lo que permitió añadir el alemán, el
   francés y el italiano enteros sin tocar una sola llamada de app.py.

3. EL TEXTO VA AQUÍ; LOS NÚMEROS, NO. Las cifras verificadas (conteos de registros,
   métricas, importancias) siguen viviendo en app.py, que es donde se documenta su
   procedencia. Este catálogo aporta solo el rótulo y la glosa, y app.py los empareja
   POR POSICIÓN con el dato. Así una cifra no puede divergir entre idiomas. Las
   excepciones son los bloques cuyo contenido es íntegramente texto o notación
   localizable (las 15 expectativas, la tarjeta del escalador): esos van completos.

Estado: traducida la aplicación entera a los cinco idiomas — menú, barra lateral y
las siete páginas.
"""

import re
import unicodedata

LANGS = ("es", "en", "de", "fr", "it")
DEFAULT_LANG = "es"

# ─────────────────────────────────────────────────────────────────────────
# PÁGINAS
# ─────────────────────────────────────────────────────────────────────────
# Clave estable + icono Bootstrap. El ORDEN es el del menú, y Gobernanza va en
# segunda posición por el motivo razonado en app.py (la app se recorre en el
# orden real de ejecución del pipeline). El rótulo visible sale de STR[lang]["nav"].
# El icono es el nombre de un Material Symbol, que es la familia que Streamlit trae de serie y
# la que consume st.button vía `icon=":material/…:"`. Antes eran nombres de Bootstrap Icons,
# porque los pedía streamlit-option-menu; al pasar el menú a botones nativos (ver el bloque
# "Menú y árbol de secciones" de app.py) el que manda es el catálogo de Streamlit. Los nombres
# se validan al pintarse: uno que no exista lanza StreamlitAPIException, no falla en silencio.
PAGES = [
    ("overview",   "home"),
    ("governance", "verified_user"),   # escudo con marca de verificación
    ("results",    "bar_chart"),
    ("shap",       "account_tree"),    # jerarquía de nodos
    ("circuit",    "memory"),          # el chip
    ("predictor",  "tune"),            # los deslizadores
]
PAGE_KEYS = [k for k, _ in PAGES]

# Claves de página RETIRADAS y dónde vive hoy su contenido: (página, grupo de pestañas,
# posición). "bloch" era una entrada del menú y sus enlaces —?page=bloch— se compartieron
# como tales, así que se traducen a su destino actual en vez de caer al Resumen, que es lo
# que hace el saneo con cualquier ?page= que no reconoce. Sin esto, un enlace que alguien
# guardó cuando la Esfera de Bloch era página propia llevaría a la portada sin explicación.
PAGES_RETIRADAS = {"bloch": ("circuit", "qc_tabs", 1)}

# ─────────────────────────────────────────────────────────────────────────
# BANDERAS
# ─────────────────────────────────────────────────────────────────────────
# SVG en línea, no emoji: Windows no incluye glifos de bandera, así que 🇪🇸/🇬🇧
# se dibujarían como las letras "ES"/"GB" — precisamente en la plataforma donde
# se desarrolla y se defiende este TFM. El SVG además escala sin pixelarse y pesa
# menos que cualquier PNG equivalente.
#
# Las CINCO van en el MISMO lienzo 60×40 (3:2) para que ocupen idéntica caja en la
# barra. Solo la española tiene esa proporción oficialmente (la de EE. UU. es
# 10:19, la alemana 3:5, la francesa 2:3 y la italiana 2:3), así que las otras se
# redibujan recalculando sus medidas en proporción — no se estiran—, que es lo que
# hacen los sets de iconos al normalizar a una rejilla común.
#
# La bandera del idioma inglés es la de ESTADOS UNIDOS, no la del Reino Unido,
# porque el catálogo STR["en"] está escrito en inglés americano (ver su cabecera):
# la bandera nombra la variante que de verdad se sirve. Por el mismo criterio, el
# alemán se sirve en su variante estándar de Alemania y lleva la Bundesflagge, y el
# francés en la de Francia (no la de Quebec ni la valona).

# ── Construcción de la bandera de EE. UU. ──
# 13 franjas, roja la primera y la última; cantón de 2/5 del ancho y 7/13 del alto.
_FRANJA = 40 / 13
_CANTON_ANCHO, _CANTON_ALTO = 24.0, 40 * 7 / 13

# Estrella de cinco puntas, centrada en (0,0) y de radio 1. Se declara UNA vez y se
# instancia 50 veces con <use>: repetir el trazado entero cincuenta veces multiplicaría
# por diez el peso del data-URI, que viaja en la hoja de estilos.
_ESTRELLA = ("M0,-1 L0.225,-0.309 L0.951,-0.309 L0.363,0.118 L0.588,0.809 "
             "L0,0.382 L-0.588,0.809 L-0.363,0.118 L-0.951,-0.309 L-0.225,-0.309 Z")

# Disposición oficial de las 50 estrellas: 9 filas alternas de 6 y 5. Las de 6 caen en
# los doceavos IMPARES del ancho del cantón y las de 5 en los PARES, y todas en los
# décimos de su alto. Generarlas aquí en vez de escribir 50 elementos a mano deja a la
# vista la regla —que es lo que hay que poder verificar— en lugar de un bloque opaco.
_ESTRELLAS = "".join(
    f'<use href="#e" x="{col * _CANTON_ANCHO / 12:.2f}" y="{(fila + 1) * _CANTON_ALTO / 10:.2f}"/>'
    for fila in range(9)
    for col in (range(1, 12, 2) if fila % 2 == 0 else range(2, 11, 2))
)
_BARRAS = "".join(
    f'<rect y="{i * _FRANJA:.2f}" width="60" height="{_FRANJA:.2f}" fill="#B22234"/>'
    for i in range(0, 13, 2)
)

# ── Construcción de la bandera de Alemania ──
# Tres franjas horizontales IGUALES, así que el alto de cada una sale de una división
# que no es exacta (40/3 = 13,33…). Se calcula en vez de escribirse redondeado a mano:
# con "13.33" repetido tres veces la última franja cerraría en 39,99 y dejaría una línea
# de fondo asomando por el borde inferior. Cada franja se posiciona por su múltiplo y la
# de abajo se estira hasta el borde del lienzo.
_BANDA_DE = 40 / 3

FLAG_SVG = {
    "es": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">'
        '<rect width="60" height="40" fill="#AA151B"/>'
        '<rect y="10" width="60" height="20" fill="#F1BF00"/>'
        "</svg>"
    ),
    # Colores oficiales: Old Glory Red y Old Glory Blue.
    "en": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">'
        # scale(1.2) en el propio trazado: <use> solo traslada, así que el tamaño de la
        # estrella tiene que venir ya en lo referenciado. 1,2 de radio es el diámetro
        # oficial (0,0616 del alto) ajustado a que el lienzo va comprimido a lo ancho.
        f'<defs><path id="e" transform="scale(1.2)" d="{_ESTRELLA}" fill="#FFFFFF"/></defs>'
        '<rect width="60" height="40" fill="#FFFFFF"/>'
        f"{_BARRAS}"
        f'<rect width="{_CANTON_ANCHO:.2f}" height="{_CANTON_ALTO:.2f}" fill="#3C3B6E"/>'
        f"{_ESTRELLAS}"
        "</svg>"
    ),
    # Schwarz-Rot-Gold en los tonos que publica el Bundesministerium des Innern. El negro
    # es el fondo del lienzo y no una franja aparte: dibujarlo como <rect> sería un
    # rectángulo de más para el mismo resultado.
    "de": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">'
        '<rect width="60" height="40" fill="#000000"/>'
        f'<rect y="{_BANDA_DE:.3f}" width="60" height="{_BANDA_DE:.3f}" fill="#DD0000"/>'
        f'<rect y="{2 * _BANDA_DE:.3f}" width="60" height="{40 - 2 * _BANDA_DE:.3f}" fill="#FFCE00"/>'
        "</svg>"
    ),
    # Tricolores VERTICALES: tres franjas iguales de 20, que aquí sí sale exacto (60/3)
    # y no hace falta la cuenta con decimales de la alemana. La banda central es el
    # fondo blanco del lienzo; solo se pintan las dos de los lados.
    # Colores oficiales: los de la Présidence de la République desde 2020 (el azul
    # marino de 1976, no el más claro de la bandera europea).
    "fr": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">'
        '<rect width="60" height="40" fill="#FFFFFF"/>'
        '<rect width="20" height="40" fill="#002395"/>'
        '<rect x="40" width="20" height="40" fill="#ED2939"/>'
        "</svg>"
    ),
    # Verde e rosso del Reglamento italiano (verde prato / rosso pomodoro).
    "it": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">'
        '<rect width="60" height="40" fill="#FFFFFF"/>'
        '<rect width="20" height="40" fill="#009246"/>'
        '<rect x="40" width="20" height="40" fill="#CE2B37"/>'
        "</svg>"
    ),
}

# ─────────────────────────────────────────────────────────────────────────
# TEXTOS
# ─────────────────────────────────────────────────────────────────────────
STR = {
    # ═══════════════════════════════ ESPAÑOL ═══════════════════════════════
    "es": {
        # ── Navegación y barra lateral ──
        "nav": ["Resumen", "Gobernanza", "Resultados", "Análisis SHAP",
                "Circuito Cuántico", "Predictor en Vivo"],
        # Nombre accesible del botón de colapso, no un tooltip: viaja dentro del rótulo y se
        # recorta por CSS (ver .st-key-toggle_sidebar en app.py). No se ve en pantalla.
        "sidebar_expand": "Expandir la barra lateral",
        "sidebar_collapse": "Colapsar la barra lateral",
        "search_label": "Buscar",
        "search_ph": "Buscar en el panel o en la web…",
        "search_expand": "Buscar: despliega la barra",
        "scroll_top": "Volver arriba",
        "search_in": "en {p}",
        "search_none": "Sin coincidencias en el panel.",
        # Rótulo de la fila de fuentes académicas, no un enlace: los pulsables son los nombres
        # que van debajo (ver SEARCH_SOURCES en app.py). Termina en dos puntos porque la frase
        # la completa esa fila.
        "search_web": "Buscar «{q}» en:",
        "theme_to_dark": "Cambiar a tema oscuro",
        "theme_to_light": "Cambiar a tema claro",
        "lang_es_help": "Ver la aplicación en español",
        "lang_en_help": "Ver la aplicación en inglés",
        "lang_de_help": "Ver la aplicación en alemán",
        "lang_fr_help": "Ver la aplicación en francés",
        "lang_it_help": "Ver la aplicación en italiano",
        "footer_name": "Juan Albornoz C. · TFM 2026",
        "footer_uni": "Universidad Europea de Valencia",
        "footer_name_narrow": "JAC",
        "footer_uni_narrow": "UEV",

        # ── Página 1 · Resumen ──
        "ov_eyebrow": "Framework DataOps + QML",
        # Título del TFM. Solo se pinta sobre la lámina de portada (ver portada_resumen en
        # app.py); el antetítulo de la página sigue siendo el corto de la línea de arriba.
        "ov_hero_title": ("Integración de Quantum Machine Learning en un pipeline DataOps: "
                          "arquitectura Medallón sobre Databricks y comparativa con modelos clásicos "
                          "en predicción clínica"),
        "ov_title": "Resumen",
        "ov_subtitle": ("Pipeline end-to-end sobre Databricks CE + AWS S3, con QSVM cuántico frente a dos "
                        "baselines clásicos, validado sobre datos clínicos reales del estudio NHANES (CDC)."),
        # El color de los <b> lo pone el CSS (.lead-card p b), no un style en línea:
        # así el párrafo se lee como prosa en los dos idiomas y se revisa de corrido.
        "ov_lead": (
            "Este framework diseña e implementa un pipeline <b>DataOps end-to-end</b> sobre "
            "<b>Databricks Community Edition</b>, con <b>AWS S3</b> como capa de almacenamiento cloud "
            "real y una arquitectura <b>Medallón</b> (Bronze → Silver → Gold) sobre Delta Lake como "
            "columna vertebral. Como caso de uso se predice diabetes tipo 2 sobre registros del estudio "
            "<b>NHANES</b> (CDC): el dataset no es el objeto de investigación, sino el vehículo para "
            "demostrar que la arquitectura es viable, reproducible y auditable sobre datos reales a "
            "escala. El núcleo experimental es una <b>comparativa triangulada</b> entre LightGBM "
            "(baseline tabular), SVM con kernel RBF (puente estructural) y un <b>QSVM</b> con "
            "FidelityQuantumKernel en Qiskit, manteniendo idéntico el clasificador subyacente para "
            "atribuir cualquier diferencia de rendimiento al efecto del kernel cuántico."
            " La evaluación cierra el recorrido: cada modelo se mide con AUC-ROC, F1, accuracy y "
            "MCC, <b>SHAP</b> señala sobre LightGBM las 20 variables que más pesan en la predicción, "
            "y los dos modelos clásicos se serializan a <b>ONNX</b> con su portabilidad verificada. "
            "El repositorio de GitHub publica los 7 notebooks que ejecutan ese recorrido, y esta "
            "misma aplicación, desplegada en Streamlit Cloud, es su último eslabón: la predicción "
            "en vivo y su lectura SHAP."),
        "ov_arch_alt": ("Diagrama de la arquitectura del pipeline: AWS S3 alimenta Databricks "
                        "Community Edition, donde la arquitectura Medallón (Bronze, Silver y Gold) "
                        "desemboca en tres modelos (LightGBM, SVM con kernel RBF y QSVM con Qiskit) y "
                        "en la evaluación con métricas, SHAP y serialización a ONNX; la salida va a "
                        "GitHub y a Streamlit Cloud."),
        # Rótulos del diagrama de arquitectura que abre la página (arquitectura_svg()). Las
        # cifras NO van escritas: llegan por marcador y las pone mil(), que usa el separador
        # de millar del idioma. Lo que no viaja aquí es la geometría ni qué caja va resaltada,
        # que son dibujo y no texto.
        "ov_arch_io": (
            ("AWS S3", "NHANES raw · 27 XPT", "IAM"),
            ("GitHub", "7 notebooks · README"),
            ("Streamlit Cloud", "Predicción · SHAP visual"),
        ),
        "ov_arch_grupos": (
            ("Arquitectura Medallón", (
                ("Bronze · ingesta raw",
                 "{bronze} reg · 162 col · Delta Lake ACID"),
                ("Silver · calidad",
                 "{silver} reg · 91 col · expectations"),
                ("Gold · features curados",
                 "89 features · train {train} / test {test}"),
            )),
            ("Modelos · ML / QML", (
                ("LightGBM",
                 "Baseline tabular · GOSS · EFB"),
                ("SVM · Kernel RBF",
                 "Puente directo a QSVM"),
                ("QSVM · Qiskit",
                 "ZZFeatureMap · FidelityQuantumKernel"),
            )),
            ("Evaluación y serialización", (
                ("Métricas",
                 "AUC-ROC · F1 · Accuracy · MCC"),
                ("SHAP",
                 "Explicabilidad LightGBM · top 20"),
                ("Selección · ONNX",
                 "Portabilidad verificada"),
            )),
        ),
        "ov_stats_title": "Estadísticas del dataset NHANES",
        "ov_stats_sub": "Integración de 3 ciclos bienales · pipeline de capas Bronze → Silver → Gold",
        "ov_stat_bronze": "Registros Bronze",
        "ov_stat_silver": "Registros Silver",
        "ov_stat_features": "Features Gold",
        "ov_stat_balance": "Balance de clases",
        "ov_medallion_title": "Arquitectura Medallón",
        "ov_medallion_sub": "Cadena de valor del dato (Curry, 2016) aplicada capa a capa",
        # Descripciones deliberadamente de UNA línea: el detalle de cada control vive
        # en la página Gobernanza y no debe contarse dos veces.
        "ov_layers": [
            ("Bronze", "Ingesta desde AWS S3 sin transformación. Preserva la fuente de verdad."),
            ("Silver", "Limpieza, imputación, winsorización y validación de calidad."),
            ("Gold",   "Escalado, codificación y partición estratificada. Listo para modelar."),
        ],
        "ov_goto_gov": "Ver los controles de calidad y linaje  →",
        "ov_target_title": "Distribución variable objetivo (DIQ010)",
        "ov_target_sub": "Target binarizado: 1 = diabetes diagnosticada, 0 = resto",
        "ov_pie_no": "No diabetes",
        "ov_pie_yes": "Diabetes",
        "ov_donut_center": "14 %",
        "ov_donut_caption": "DIABETES",
        "ov_tech_title": "Construido sobre",
        "ov_tech_sub": ("Plataforma, almacenamiento y librerías del pipeline, en el orden en que "
                        "intervienen · el inventario completo, con la justificación de cada elección, "
                        "está en Gobernanza"),
        "ov_compare_title": "Comparativa triangulada: objetivo del experimento",
        "ov_compare": [
            ("LightGBM", "Baseline tabular de referencia"),
            ("SVM-RBF",  "Puente estructural hacia el componente cuántico"),
            ("QSVM",     "FidelityQuantumKernel, mismo clasificador, kernel cuántico"),
        ],

        # ── Variables NHANES (compartido: SHAP, Circuito, Bloch, Predictor) ──
        # El CÓDIGO (LBXGH, RIDAGEYR…) nunca se traduce: es el identificador oficial de
        # la variable en NHANES y aparece igual en el TFM, en los notebooks y en el eje
        # de las gráficas. Lo que se traduce es su glosa.
        "var_short": {
            "LBXGH": "HbA1c", "RIDAGEYR": "Edad", "LBXGLU": "Glucosa ayunas",
            "LBDLDL": "Colesterol LDL", "BMXWAIST": "Circunf. cintura",
            "WTINT2YR": "Peso muestral*", "BMXARML": "Long. brazo", "BMXLEG": "Long. pierna",
            "BMXBMI": "IMC", "PAD680": "Act. sedentaria", "PAD645": "Act. moderada",
            "PAQ640": "Fortalecim. muscular", "BMXWT": "Peso corporal", "LBXIN": "Insulina",
            "INDHHIN2": "Ingresos hogar", "DMDYRSUS": "Años en EEUU",
            "BMXARMC": "Circunf. brazo", "PAQ670": "Act. vigorosa",
            "BPXSY1": "Presión sistólica", "PAD630": "Act. mod. recreativa",
            "DMDHHSZE": "Tamaño hogar (niños)", "BPXDI1": "Presión diastólica",
            "LBXTR": "Triglicéridos", "DMDMARTL_1": "Estado civil (casado)",
            "DMDMARTL_5": "Estado civil (nunca casado)", "BPXPLS": "Pulso",
            "DMDEDUC2_3": "Educación (nivel 3)", "SDMVSTRA": "Estrato muestral",
            "DMDMARTL_2": "Estado civil (viudo)", "DMDHHSZB": "Tamaño hogar (adultos)",
        },
        "var_desc": {
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
        },
        # Rótulo y unidad de las 8 variables del QSVM (sliders de Bloch y Predictor).
        # Van aparte de "var_short" porque aquí el rótulo se lee como etiqueta de un control
        # y admite más letra que en el eje apretado de una gráfica de barras.
        "qsvm_labels": {
            "LBXGH": "HbA1c", "LBXGLU": "Glucosa en ayunas", "RIDAGEYR": "Edad",
            "LBDLDL": "Colesterol LDL", "BMXWAIST": "Circunf. cintura", "LBXIN": "Insulina",
            "BMXLEG": "Long. pierna", "BMXBMI": "IMC",
        },
        "qsvm_units": {"años": "años"},

        # ── Página 2 · Gobernanza ──
        "gov_eyebrow": "Gobernanza · DataOps",
        "gov_title": "Gobernanza y Calidad del Dato",
        "gov_subtitle": ("Los controles que sostienen el pipeline: qué se valida, qué se descarta y por qué, "
                         "qué queda registrado y con qué frameworks. Todas las cifras proceden de las salidas "
                         "ejecutadas de los notebooks."),
        "gov_tabs": ["Calidad del dato", "Linaje y trazabilidad", "Inventario de frameworks"],
        "gov_kpi_expect": "Expectativas superadas",
        "gov_kpi_passrate": "Pass rate de la suite",
        "gov_kpi_records": "Registros validados",
        "gov_kpi_leakage": "Artefactos sin leakage",
        "gov_funnel_title": "Embudo de registros",
        # Las dos cifras llegan por mil(): con el separador escrito a mano la frase decía
        # "29.400" también en inglés, donde se lee como 29 coma 4.
        "gov_funnel_sub": ("De los {bronze} registros de Bronze sobreviven {silver} a los filtros de cohorte "
                           "de Silver. Cada escalón responde a un criterio explícito, no a una limpieza genérica."),
        "gov_hover_records": "Registros",
        "gov_hover_dropped": "Descartados",
        # Emparejado por posición con los conteos de GOV_EMBUDO_N en app.py.
        "gov_embudo": [
            ("Bronze · 3 ciclos unidos",
             "27 ficheros XPT · join por SEQN · 162 columnas comunes a los tres ciclos"),
            ("Filtro edad ≥ 18 años", "Restricción a población adulta"),
            ("Filtro ayuno · LBXGLU no nulo",
             "Proxy del subgrupo en ayunas: PHAFSTMN no es consistente entre ciclos"),
            ("Filtro DIQ010 válido",
             "Descarta los códigos 7 «no sabe» y 9 «rehúsa responder», y los nulos"),
        ],
        "gov_dropped_title": "Registros descartados por filtro",
        "gov_split_label": "Partición Gold 80/20",
        "gov_suite_title": "Suite de validación · dataframe-expectations",
        "gov_suite_sub": ("Suite <code>{nombre}</code>, ejecutada el {fecha} sobre los {registros} registros de "
                          "Silver en {duracion} segundos. Great Expectations es incompatible con las versiones "
                          "fijadas del runtime serverless: esta es la alternativa adoptada."),
        # Las 15 expectativas van completas (dimensión, columna, regla) y no partidas: la
        # regla lleva los umbrales con separador decimal, que también cambia de idioma.
        "gov_expectativas": [
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
        ],
        "gov_ops_title": "Operaciones de calidad por capa",
        "gov_silver_card": "Silver · limpieza y saneamiento",
        "gov_gold_card": "Gold · preparación para modelado",
        "gov_silver_ops": [
            ("Variables DIQ excluidas por leakage", "DIQ050, DIQ070, DIQ160, DIQ170, DIQ172, DIQ180"),
            ("Columnas sparse eliminadas", "Umbral de >80 % de valores ausentes"),
            ("Variables winsorizadas", "Recorte de outliers por IQR × 3"),
            ("Missing tras imputación", "De 75.855 a 0 en el dataset SVM/QSVM (mediana + moda)"),
        ],
        "gov_gold_ops": [
            ("Features tras codificación", "One-hot de 5 variables categóricas sobre 84 features (106 columnas con TARGET)"),
            ("Descartadas por correlación", "Umbral r > 0,90 entre pares de predictores"),
            ("Features finales", "Conjunto con el que se entrenan los tres modelos"),
            ("Partición estratificada", "80/20 · 14,03 % positivos en train, 14,04 % en test"),
        ],
        "gov_eff_title": "Features efectivas frente a features nominales",
        "gov_eff_sub": ("Contado sobre <code>scaler_correcto.json</code>: {const} de las {total} columnas "
                        "tienen varianza cero y no aportan información al modelo"),
        "gov_eff_nominal": "Features nominales",
        "gov_eff_const": "Constantes (varianza = 0)",
        "gov_eff_effective": "Features efectivas",
        "gov_eff_note": ("Es el efecto colateral de la winsorización IQR × 3 de Silver, que se aplicó también a "
                         "variables categóricas codificadas numéricamente (respuestas 1/2, idioma de la entrevista, "
                         "códigos 7 y 9). Cuando más del 75 % de la muestra responde lo mismo, el recorte colapsa la "
                         "columna a un único valor. Las más recortadas en el notebook 02 (PAQ635, PAQ650, PAQ605, "
                         "DMDHHSZA, DMDCITZN, SIALANG) son exactamente las que aquí aparecen constantes."),
        "gov_lin_title": "Trazabilidad sin MLflow",
        "gov_lin_sub": "La restricción que más condiciona la arquitectura del pipeline, y su mitigación.",
        "gov_lin_limit_title": "Limitación",
        "gov_lin_limit_body": ("La integración nativa de <b>MLflow</b> está deshabilitada en Databricks Serverless "
                               "gratuito. Cualquier llamada a <code>mlflow.start_run()</code> o "
                               "<code>mlflow.log_metric()</code> produce errores de autenticación: no hay registro "
                               "de experimentos, métricas ni artefactos."),
        "gov_lin_mit_title": "Mitigación · doble mecanismo",
        "gov_lin_mit_body": ("<b>Transaction logs de Delta Lake</b>: cada escritura genera un registro ACID con "
                             "versión, marca de tiempo y métricas de operación.<br><br>"
                             "<b>CSV de métricas por modelo</b>: cada notebook persiste sus resultados en Unity "
                             "Catalog Volumes, y las figuras los leen de ahí en vez de llevarlos escritos a mano."),
        "gov_delta_title": "Historial Delta · capa Gold",
        "gov_delta_sub": ("Seis versiones más recientes de las diez registradas. Delta purga las anteriores tras "
                          "168 h de retención, comportamiento esperado y no un fallo del pipeline."),
        "gov_delta_cols": ["Versión", "Timestamp", "Operación", "Filas", "Tamaño"],
        "gov_chain_title": "Cadena de custodia contra la fuga de información",
        "gov_chain_sub": ("Cuatro barreras encadenadas. La tercera no descarta ninguna columna, y eso es "
                          "exactamente lo que se quiere ver: prueba que las anteriores hicieron su trabajo."),
        "gov_leakage": [
            ("Exclusión en Silver",
             "Se eliminan 6 variables DIQ de tratamiento y seguimiento antes de winsorizar: son "
             "consecuencia del diagnóstico, no predictores de él."),
            ("Verificación cruzada",
             "Se comprueba que ninguna DIQ sobrevive en los 2 Parquet de Silver ni en los 13 de "
             "Gold. Resultado: 15/15 artefactos limpios."),
            ("Filtro defensivo del QSVM",
             "Segunda barrera antes de la selección por Random Forest. No descarta ninguna columna "
             "(89 de 89 pasan), precisamente la prueba de que la primera barrera funcionó."),
            ("Guarda de pesos de muestreo",
             "Detiene el pipeline si aparece cualquier peso de muestreo distinto del conocido. "
             "WTINT2YR sí llega al modelado y está documentado en la decisión 10."),
        ],
        "gov_scaler_card": "Escalado sin fuga estadística",
        "gov_scaler": [
            ("Ajuste", "Solo sobre train", "fit_transform en train · transform en test"),
            ("Columnas evaluadas", "66", "Con varianza > 0"),
            ("Columnas constantes", "23", "Varianza 0 · ver decisión 08"),
            ("Media ≈ 0 · desv. ≈ 1", "Verificado", "Assert sobre todas las columnas con dispersión"),
        ],
        "gov_scaler_note": ("El <b>StandardScaler</b> se ajusta exclusivamente sobre <b>train</b>: "
                            "<code>fit_transform</code> en entrenamiento y <code>transform</code> en test. Si se "
                            "ajustara sobre el conjunto completo, la media y la desviación típica del test se "
                            "filtrarían al preprocesado y las métricas quedarían optimistas. La selección de las 8 "
                            "variables del QSVM sigue la misma regla: el Random Forest se entrena solo con "
                            "<code>X_train_svm_scaled</code>.<br><br>El filtro de correlación, en cambio, <b>sí</b> "
                            "se calcula antes de particionar. Está documentado y asumido en la decisión 09."
                            "<br><br>La comprobación no se declara, se ejecuta: sobre las 66 columnas con "
                            "dispersión se exige |media| &lt; 0,01 y desviación entre 0,90 y 1,10. Los parámetros "
                            "ajustados (<code>mean_</code> y <code>scale_</code>) viajan a "
                            "<code>scaler_correcto.json</code>, el mismo fichero que carga el Predictor en Vivo: la "
                            "inferencia reutiliza la escala del train y nunca la recalcula."),
        "gov_e2e_title": "Verificación end-to-end contra los modelos entrenados",
        "gov_e2e_missing": ('<b style="color:{color};">Sin verificar.</b> El conjunto de test no está en el '
                            "repositorio, así que el dashboard no puede comprobar por sí solo que su camino de "
                            "inferencia reproduzca lo que produjeron los modelos entrenados. Para cerrarlo, ejecuta "
                            "las dos celdas de <code>notebooks/INSTRUCCIONES_exportar_golden_set.md</code> y copia "
                            "<code>golden_lgbm.npz</code> y <code>golden_svm.npz</code> a "
                            "<code>streamlit/models/</code>. Mientras falten, esta página no afirma nada que no "
                            "haya podido comprobar."),
        "gov_e2e_unavailable": "no disponible",
        "gov_e2e_ok_val": "{n} filas · dif. máx. {dif}",
        "gov_e2e_bad_val": "DISCREPA · dif. máx. {dif}",
        "gov_e2e_scaled": "escala y llama al ONNX",
        "gov_e2e_raw": "llama al ONNX sin escalar",
        "gov_e2e_path": "El dashboard {accion}",
        "gov_e2e_ok_title": "✓ Camino de inferencia verificado",
        "gov_e2e_fail_title": "⚠ El camino de inferencia no reproduce los modelos",
        "gov_e2e_note": ("Cada fila del <i>golden set</i> es una instancia real del test acompañada de la "
                         "probabilidad que devolvió el modelo entrenado en su notebook. El dashboard la pasa por "
                         "su propio camino (vector crudo, escalado solo del SVM, conversión a <code>float32</code>, "
                         "sesión ONNX y lectura del tensor de salida) y compara. Tolerancia {tol}; el ruido "
                         "esperado por trabajar en <code>float32</code> es de orden 10⁻⁷."),
        "gov_stack_title": "Frameworks por capa",
        "gov_stack_sub": ("El primer distintivo de cada tarjeta es el framework que vertebra la capa; el resto "
                          "lo acompañan."),
        # Rol y nota de cada tarjeta, en el orden de GOV_STACK: Bronze, Silver, Gold,
        # LightGBM, SVM-RBF, QSVM. Nombre de capa y badges no se traducen (son nombres propios).
        "gov_stack": [
            ("ingesta",
             "boto3 sustituye a spark.conf, bloqueado en Serverless (decisión 01). Tres asserts "
             "de integridad: 27/27 ficheros, el join por SEQN no duplica filas, y Delta cuadra "
             "con pandas."),
            ("calidad",
             "El framework de calidad del TFM. Great Expectations es incompatible con el "
             "entorno (decisión 03). Suite de 15 expectativas en 3 dimensiones, con evidencia "
             "persistida en CSV."),
            ("preparación",
             "Escalado ajustado solo sobre train, partición estratificada con semilla fija y "
             "exportación del contrato de serving (scaler y medianas en JSON)."),
            ("modelo",
             "Interpretabilidad exacta por algoritmo polinomial sobre las 1.567 instancias de "
             "test, y verificación de que el ONNX reproduce el PKL al 100 %."),
            ("modelo",
             "SHAP agnóstico al modelo, con coste de horas: se calcula una vez sobre 200 "
             "instancias y se persiste en disco para reutilizarlo."),
            ("modelo",
             "Sin soporte ONNX: el formato no admite operaciones cuánticas (decisión 05). La "
             "trazabilidad recae en un CSV de métricas con los 14 campos de configuración."),
        ],
        "gov_dec_title": "Registro de decisiones",
        "gov_dec_sub": ("Las once limitaciones documentadas en TECHNICAL_NOTES, con su mitigación. Tres "
                        "condicionan la arquitectura, seis se asumen y documentan sin corregir (porque hacerlo "
                        "invalidaría los resultados ya obtenidos) y dos quedan resueltas sin residuo."),
        "gov_dec_tags": {"critical": "Arquitectura", "warning": "Asumida", "good": "Resuelta"},
        "gov_dec_problem": "Problema · ",
        "gov_dec_solution": "Solución adoptada · ",
        # Emparejado por posición con la referencia y el nivel, que viven en app.py.
        "gov_decisiones": [
            ("spark.conf bloqueado en Serverless",
             "La configuración de credenciales AWS por spark.conf.set devuelve CONFIG_NOT_AVAILABLE, "
             "el mecanismo estándar para conectar Spark con S3.",
             "boto3 como cliente alternativo. S3 queda como almacenamiento de origen y Unity Catalog "
             "Volumes como capa de procesamiento."),
            ("MLflow bloqueado en Serverless",
             "La integración nativa de MLflow está deshabilitada en la capa gratuita: no hay registro "
             "de experimentos, métricas ni artefactos.",
             "Doble mecanismo sustitutivo: los transaction logs de Delta Lake aportan versión, "
             "timestamp y métricas de operación; y cada notebook persiste sus métricas en CSV."),
            ("Great Expectations incompatible",
             "Requiere una combinación de pandas/numpy que choca con las versiones fijadas del "
             "runtime serverless (pandas 1.5.3 / numpy 1.23.5).",
             "dataframe-expectations 0.7.0 como alternativa compatible. 15 expectativas sobre Silver "
             "en tres dimensiones. Resultado 15/15, pass rate 1,0."),
            ("QSVM · coste computacional O(n²)",
             "Sobre las 6.264 instancias de train, la matriz de kernel exigiría ~39 millones de "
             "evaluaciones del circuito. Con 1.500 el kernel agota la memoria.",
             "Entrenamiento sobre muestra estratificada de 500 instancias (~22 min) preservando el "
             "ratio 86/14. La evaluación sí usa el test completo, para que las métricas comparen."),
            ("QSVM · sin soporte ONNX nativo",
             "El formato ONNX no admite operaciones cuánticas: ni skl2onnx ni onnxmltools pueden "
             "serializar un kernel basado en simulación de estados.",
             "Serialización con joblib. El modelo requiere el entorno Qiskit para inferencia, por lo "
             "que el QSVM no entra en el Predictor en Vivo."),
            ("Versiones de Qiskit no fijables",
             "immutable_package_constraints.txt de Databricks bloquea la instalación de versiones "
             "concretas, así que no hay reproducibilidad exacta de versión.",
             "El pipeline corre con las versiones del entorno (2.5.0 / 0.9.0 / 0.4.0), cuya API es "
             "compatible, y quedan registradas en una verificación explícita al inicio de la ejecución."),
            ("Pérdida de variables por duración de sesión",
             "Las operaciones largas (22 min de entrenamiento, 132 de predicción) pueden agotar la "
             "sesión serverless y llevarse las variables en memoria.",
             "Persistencia inmediata tras cada operación costosa y modo TRAINING_MODE que recarga "
             "desde disco en ejecuciones posteriores."),
            ("Winsorización aplicada a categóricas codificadas",
             "NHANES codifica numéricamente muchas categóricas. Si más del 75 % comparte valor, "
             "IQR = 0, los límites colapsan y clip() convierte la variable en constante. "
             "10 columnas quedaron colapsadas así.",
             "Se documenta sin modificar: corregirlo alteraría Silver, Gold y los tres modelos. Las "
             "columnas constantes no sesgan (el modelo no extrae señal de ellas), pero pierden "
             "información. Corrección identificada como trabajo futuro."),
            ("Correlación calculada antes de particionar",
             "El filtro r > 0,90 se calcula sobre el dataset completo, así que las 16 columnas "
             "descartadas se deciden usando también las observaciones de test.",
             "Se documenta sin modificar. No afecta al escalado ni a la selección de features del "
             "QSVM, ambos ajustados solo sobre train, pero la selección deja de ser estrictamente "
             "ciega al test."),
            ("Peso de muestreo WTINT2YR entre las features",
             "El join intracíclico duplica WTSAF2YR en tres columnas. WTINT2YR no está en la lista "
             "de exclusión y sobrevive al filtro de correlación: es una de las 89 features.",
             "Se documenta sin modificar y se añade un assert que detecta la aparición de cualquier "
             "OTRO peso. Un peso muestral no es una variable clínica: no filtra el objetivo, pero "
             "deja al modelo apoyarse en el diseño de la encuesta."),
            ("El QSVM serializado no es recargable entre versiones",
             "El pickle arrastra el ZZFeatureMap con sus ParameterExpression. Si Qiskit cambia de "
             "versión, la deserialización falla, y Serverless actualiza sin aviso.",
             "La carga va envuelta en try/except: si falla, TRAINING_MODE pasa a True y el notebook "
             "re-entrena en lugar de abortar. Queda operativo en los tres escenarios posibles."),
        ],
        "gov_footer_note": ("Las cifras de esta página proceden de las salidas ejecutadas de los notebooks del "
                            "repositorio y de <code>TECHNICAL_NOTES.md</code>; ninguna es estimada. La aplicación "
                            "no puede consultarlas en vivo porque Streamlit Community Cloud solo accede al "
                            "repositorio, no a Unity Catalog Volumes.<br><br>"
                            "Resumen de la suite de calidad: <b>{fuente}</b>."),
        "gov_suite_src_csv": "leído de validacion_silver_dfe.csv",
        "gov_suite_src_nb": "valores verificados del notebook",

        # ── Página 3 · Resultados ──
        "res_eyebrow": "Comparativa triangulada",
        "res_title": "Resultados",
        "res_subtitle": "LightGBM vs. SVM-RBF vs. QSVM sobre el mismo conjunto de test ({n} instancias).",
        "res_threshold": "Umbral",
        # Punto de corte de cada modelo. "{v}" lo rellena thr_text() con nf(), porque el
        # separador decimal es notación y cambia de idioma; el del QSVM no lleva cifra
        # (decision_function no está en escala de probabilidad).
        "res_thr_label": {"lightgbm": "p ≥ {v}", "svm_rbf": "p ≈ {v}", "qsvm": "df > 0"},
        "res_thr_src": {"lightgbm": "predict_proba()[:,1] >= 0.5",
                        "svm_rbf": "SVC.predict() · signo de decision_function",
                        "qsvm": "decision_function > 0 (no es probabilidad)"},
        "res_reconciled": ('<span style="color:{color}; font-weight:600;">✓ Reconciliadas</span>: las cuatro '
                           "métricas de los tres modelos se han recalculado desde los scores por instancia y "
                           "coinciden con las publicadas."),
        "res_unreconciled": '<span style="color:{color}; font-weight:600;">⚠ Sin reconciliar</span>: {fallos}',
        "res_no_scores": "scores no disponibles",
        "res_threshold_note": ("<b>Los tres modelos están medidos en umbrales distintos.</b> Cada uno usa su punto "
                               "de corte natural: LightGBM <code>predict_proba ≥ 0,50</code>; SVM-RBF el signo de "
                               "<code>decision_function</code>, que en la escala de probabilidad guardada equivale "
                               "a ≈ 0,22; QSVM <code>decision_function &gt; 0</code>, que no es una probabilidad. "
                               "Cada matriz reproduce exactamente en su propio umbral, pero <b>solo el AUC-ROC es "
                               "comparable entre modelos</b>: es la única de las cuatro métricas independiente del "
                               "punto de corte. A modo de referencia, el SVM-RBF evaluado a 0,50 como LightGBM "
                               "daría accuracy 0,9190 pero solo 131 verdaderos positivos en lugar de 172."),
        "res_roc_title": "Curvas ROC",
        "res_roc_sub_real": ("Curvas empíricas reales, punto a punto sobre las 1.567 instancias del test "
                             "(mismos scores que reportan el AUC del TFM)."),
        "res_roc_sub_synth": ("AUC exacto · forma reconstruida a partir del AUC donde no hay scores por instancia."),
        "res_cm_title": "Matrices de confusión",
        "res_cm_sub": ("Valores verificados contra el classification report de cada modelo, y recalculados desde "
                       "los scores por instancia. Cada matriz corresponde al umbral indicado en su tarjeta"),
        "res_cm_pred_no": "Pred.<br>No diabetes",
        "res_cm_pred_yes": "Pred.<br>Diabetes",
        "res_cm_real_no": "Real<br>No diab.",
        "res_cm_real_yes": "Real<br>Diabetes",
        "res_cm_tags": {"tn": "VN", "fp": "FP", "fn": "FN", "tp": "VP"},
        "res_metrics_title": "Comparativa de métricas",
        "res_metrics_sub": ("Las cuatro métricas se aplican sobre las 1.567 instancias. Accuracy, MCC y F1-macro "
                            "sí penalizan el desbalance de clases, pero dependen del umbral, y cada modelo usa "
                            "el suyo: compara con cautela todo lo que no sea el AUC-ROC"),
        "res_metric_desc": {
            "auc": "Área bajo la curva ROC: capacidad de separar diabetes vs. no-diabetes. 0,5 = azar, 1 = perfecto.",
            "f1_macro": "Media armónica de precisión y recall promediada por clase (sin ponderar). Penaliza el desbalance.",
            "accuracy": "Proporción de aciertos totales. Con clases desbalanceadas puede reflejar solo la clase mayoritaria.",
            "mcc": "Coef. de correlación de Matthews: calidad global robusta al desbalance. 0 = azar, 1 = perfecto.",
        },
        "res_qsvm_note": ("<b>Nota sobre el experimento QSVM.</b> El QSVM se entrenó sobre una muestra "
                          "estratificada de 500 instancias (coste O(n²) del kernel cuántico) y se evaluó sobre "
                          "las 1.567 del test completo. AUC-ROC = 0,5493 indica que el modelo apenas supera la "
                          "clasificación aleatoria: Recall ≈ 0 para la clase diabetes (1 de 220), "
                          "Accuracy = 0,8602 refleja solo la proporción de la clase mayoritaria. El MCC ≈ 0 "
                          "confirma ausencia de capacidad predictiva real."),

        # ── Página 4 · Análisis SHAP ──
        "sh_eyebrow": "Interpretabilidad",
        "sh_title": "Análisis SHAP",
        "sh_subtitle": ("Importancia global de variables: TreeExplainer (LightGBM) vs. "
                        "KernelExplainer (SVM-RBF)."),
        "sh_tabs": ["LightGBM · TreeExplainer", "SVM-RBF · KernelExplainer"],
        "sh_hint": "Pasa el cursor sobre cada barra para ver el significado de la variable. {nota}",
        "sh_sample_lgbm": "Valores exactos (algoritmo polinomial) sobre las 1.567 instancias del test.",
        "sh_sample_svm": ("Valores aproximados por muestreo: fondo de 100 instancias, contribuciones sobre "
                          "200 instancias de test."),
        "sh_note_lgbm": ("<b>LBXGH (HbA1c)</b> domina con amplia diferencia (SHAP medio = 1,1243), coherente con "
                         "su papel como marcador diagnóstico primario de diabetes tipo 2 (ADA: HbA1c ≥ 6,5%). "
                         "<b>RIDAGEYR (edad, 0,4654)</b> refleja el aumento de prevalencia con la edad. "
                         "<b>LBXGLU</b> y <b>LBDLDL</b> completan el bloque bioquímico. <b>WTINT2YR</b> "
                         "(posición 6) es un artefacto del diseño muestral NHANES, no una variable clínica."),
        "sh_note_svm": ("El ranking de SVM-RBF coincide en las variables dominantes con LightGBM (<b>LBXGH</b>, "
                        "<b>LBXGLU</b>, <b>LBDLDL</b>, <b>RIDAGEYR</b>), lo que refuerza la validez clínica del "
                        "hallazgo al ser independiente del algoritmo, dotándolo de mayor robustez metodológica. "
                        "KernelExplainer trata el modelo como caja negra, aplicable a cualquier clasificador."),
        "sh_fig_lgbm_title": "SHAP Summary Plot · LightGBM (Figura 27)",
        "sh_fig_lgbm_cap": ("Cada punto es una instancia del test; el color indica el valor de la variable "
                            "(rojo alto, azul bajo) y la posición horizontal su impacto en la predicción. "
                            "LBXGH y RIDAGEYR dominan el modelo."),
        "sh_fig_svm_title": "SHAP Summary Plot · SVM-RBF (Figura 31)",
        "sh_fig_svm_cap": ("Cada punto es una instancia; color = valor de la variable, posición = impacto. "
                           "KernelExplainer sobre 200 instancias del test."),

        # ── Página 5 · Circuito Cuántico ──
        "qc_eyebrow": "Componente cuántico",
        "qc_title": "Circuito Cuántico",
        "qc_subtitle": ("Configuración del ZZFeatureMap y FidelityQuantumKernel implementados en Qiskit sobre "
                        "Databricks CE."),
        # Las DOS PESTAÑAS de la página. La Esfera de Bloch era una entrada de primer nivel del
        # menú y ahora vive aquí dentro, así que el rótulo de su pestaña es EL MISMO que tenía
        # en el menú —ya traducido y revisado en los cinco idiomas—: quien conociera la app la
        # sigue encontrando por el nombre por el que la conocía. Van por tabs_i18n y no por
        # st.tabs pelado, como las de Gobernanza y Análisis SHAP, porque los dos rótulos
        # cambian con la bandera y sin eso la pestaña abierta se perdía al traducir.
        "qc_tabs": ["Circuito ZZFeatureMap", "Esfera de Bloch"],
        "qc_specs": ["Qubits (feature_dimension)", "Repeticiones (reps)", "Entanglement", "Versión de Qiskit"],
        "qc_how_title": "Cómo funciona",
        "qc_how_p1": ("El <b>ZZFeatureMap</b> codifica cada una de las 8 variables clínicas como un ángulo de "
                      "fase (puerta P) en un qubit independiente, tras crear superposición con puertas Hadamard. "
                      "Su elemento distintivo es el <b>entrelazamiento</b> entre pares de qubits mediante puertas "
                      "que dependen del producto cruzado de dos variables, correlaciones que el kernel RBF "
                      "clásico no puede representar."),
        "qc_how_p2": ("El <b>FidelityQuantumKernel</b> mide la similitud entre dos pacientes como la fidelidad "
                      "entre sus estados cuánticos: <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. La implementación usa "
                      "<code>StatevectorSampler</code>, simulando el estado exacto sin ruido: resultados "
                      "deterministas y reproducibles."),
        "qc_feat_title": "8 features seleccionadas (Random Forest)",
        "qc_xaxis": "Importancia RF",
        "qc_train_title": "Entrenamiento y evaluación",
        "qc_tstats": ["Instancias entrenamiento", "Tiempo entrenamiento", "Instancias test",
                      "Tiempo de inferencia", "Support vectors"],
        "qc_note": ("Por el coste O(n²) del kernel cuántico, el entrenamiento se limitó a una muestra "
                    "estratificada de 500 instancias (el límite operativo de Databricks CE serverless se sitúa "
                    "~500-1.000). La evaluación se hizo sobre el test completo (1.567 instancias) por lotes de "
                    "100, con un tiempo total de predicción de 144,5 minutos."),
        "qc_circuit_title": "Circuito cuántico completo (8 qubits)",
        "qc_circuit_sub": ("ZZFeatureMap con reps=2: codificación (H + P) seguida de dos rondas de entrelazamiento "
                           "lineal entre qubits adyacentes."),

        # ── Página 5 · Circuito Cuántico → pestaña Esfera de Bloch ──
        # AQUÍ VIVÍA bl_eyebrow ("Codificación cuántica") y se ha retirado: un antetítulo nombra
        # la categoría de una PÁGINA, y la Esfera de Bloch pasó a ser una pestaña de Circuito
        # Cuántico —que ya trae el suyo, qc_eyebrow—. bl_title y bl_subtitle sí se conservan:
        # entran como section-title/section-sub de la pestaña, y bl_title es además la fila con
        # la que el buscador sigue llevando hasta aquí por su nombre.
        "bl_title": "Esfera de Bloch",
        "bl_subtitle": "Cómo el ZZFeatureMap codifica el valor de una variable clínica como estado cuántico |ψ⟩.",
        # Nota de entrada de la página, en el mismo formato que lp_what_note: quien llega aquí
        # puede venir del lado clínico y no tener por qué saber qué es una esfera de Bloch, y sin
        # eso la figura es una bola con una flecha. Se explica con el bit clásico como punto de
        # partida y se cierra atando el dibujo a lo que hace el deslizador de al lado, que es lo
        # que convierte la explicación en algo que se puede comprobar moviendo el control.
        "bl_what_note": ("<b>Qué es la esfera de Bloch.</b> Un bit clásico solo puede valer 0 o 1. "
                         "Un qubit admite además cualquier mezcla de los dos, y esa mezcla no cabe "
                         "en un único número: hace falta un mapa. La esfera de Bloch es ese mapa: "
                         "cada estado posible de un qubit es un punto de la superficie de una esfera "
                         "de radio 1. El polo norte es <b>|0⟩</b> y el polo sur <b>|1⟩</b>; entre "
                         "ambos están las superposiciones, y cuanto más cerca queda la flecha de un "
                         "polo, más probable es ese resultado al medir. Aquí el valor clínico se "
                         "traduce en el ángulo θ, así que mover el deslizador hace girar la flecha "
                         "por un meridiano, de |0⟩ a |1⟩."),
        "bl_var": "Variable clínica",
        "bl_value": "Valor ({unidad})",
        "bl_xnorm": "x normalizado",
        "bl_theta": "θ = x_norm·π",
        "bl_alpha": "α (amplitud |0⟩)",
        "bl_beta": "β (amplitud |1⟩)",
        "bl_rad": "rad",
        "bl_note": ("<b>Analogía didáctica del principio de codificación angular</b>, no una réplica del circuito. "
                    "Aquí el valor clínico normalizado a [0,1] se convierte en el ángulo <b>polar</b> "
                    "θ = x_norm·π, de modo que el vector recorre el meridiano de |0⟩ a |1⟩ y P(|0⟩) varía de "
                    "100 % a 0 %: es la forma más legible de ver «un número se vuelve un estado».<br><br>"
                    "El <b>ZZFeatureMap real</b> hace algo distinto: aplica H y después P(2·x<sub>i</sub>), y una "
                    "puerta de fase tras una Hadamard deja el estado <b>sobre el ecuador</b> (θ = π/2 fijo, "
                    "P(|0⟩) = P(|1⟩) = 50 % siempre), codificando el dato en el ángulo <b>azimutal</b> φ, no en el "
                    "polar. Tampoco normaliza a [0,1]: usa el valor escalado directamente. Por eso esta esfera "
                    "ilustra el concepto, pero no reproduce paso a paso el circuito. El entrelazamiento (puertas "
                    "P(2·(π−x<sub>i</sub>)·(π−x<sub>j</sub>))) solo es representable en el espacio conjunto de "
                    "los 8 qubits (ver Circuito Cuántico)."),

        # ── Página 5 · pestaña Esfera de Bloch → sección de entrelazamiento ──
        # Continúa exactamente donde acaba bl_note: esa nota cierra diciendo que el
        # entrelazamiento solo se representa en el espacio conjunto, y esta sección es esa
        # frase convertida en figura. De ahí que el subtítulo la enuncie como un límite del
        # mapa que la página acaba de enseñar, y no como un tema nuevo.
        "bl_ent_title": "Entrelazamiento: tres qubits, un solo estado",
        "bl_ent_sub": ("El límite de la esfera de arriba. Aplica las tres puertas y mira qué le pasa al "
                       "estado local de cada qubit, y al del par que se queda por el camino."),
        "bl_ent_intro": ("<b>Dónde deja de servir la esfera de Bloch.</b> Con un qubit basta una esfera y una "
                         "flecha. Con varios, la tentación es dibujar una esfera por qubit, y para la mayoría "
                         "de los estados funciona. Pero existe una familia de estados en los que <b>no queda "
                         "flecha que dibujar</b>: el conjunto tiene un estado perfectamente definido y ninguno "
                         "de sus miembros lo tiene por separado. Eso es el entrelazamiento, y aquí se construye "
                         "con tres puertas. Aplícalas y sigue las cuatro cifras de la izquierda: las tres "
                         "primeras se rompen en el segundo paso, y la cuarta enseña en el tercero algo que con "
                         "dos qubits no se puede ni plantear."),
        "bl_ent_btn_h": "1 · Hadamard en q₀",
        "bl_ent_btn_cnot1": "2 · CNOT (control q₀ → q₁)",
        "bl_ent_btn_cnot2": "3 · CNOT (control q₁ → q₂)",
        "bl_ent_btn_reset": "Reiniciar a |000⟩",
        # Una entrada por paso, en el orden en que se recorren. Dos o tres frases: lo justo
        # para decir qué acaba de cambiar en las dos figuras de debajo, sin repetir lo que ya
        # dicen los rótulos.
        "bl_ent_step_note": [
            ("<b>Punto de partida.</b> Tres qubits, los tres en |0⟩, sin ninguna puerta aplicada. El estado "
             "conjunto es |000⟩ y todavía no tiene nada de cuántico: equivale exactamente a tres bits "
             "clásicos puestos a cero. En la Q-sphere hay un único nodo, en el polo norte, que se lleva "
             "toda la probabilidad."),
            ("<b>Superposición, todavía sin entrelazar.</b> La Hadamard deja a q₀ a medio camino entre "
             "|0⟩ y |1⟩, mientras q₁ y q₂ siguen firmes en |0⟩: el estado conjunto es (|000⟩ + |100⟩)/√2. "
             "Los tres qubits siguen siendo <b>independientes</b>: cada uno tiene su propio estado puro y "
             "tres esferas de Bloch bastarían para describirlos. Fíjate en que la longitud del vector local "
             "sigue valiendo 1: hay flecha que dibujar."),
            ("<b>Un par de Bell, y un testigo.</b> El primer CNOT voltea q₁ solo cuando q₀ vale 1; aplicado "
             "sobre una superposición, eso ata los dos resultados en uno solo: (|000⟩ + |110⟩)/√2. El mapa se "
             "rompe aquí: la longitud del vector local de q₀ acaba de caer a <b>0</b>, el qubit ya no está en "
             "ningún punto de su esfera porque por separado <b>ya no tiene estado</b>. Y pasa algo más, que "
             "solo se puede ver habiendo un tercer qubit: q₂ se ha quedado FUERA, mirando desde |0⟩, y lo que "
             "hay entrelazado es exactamente la pareja q₀q₁. Su concurrencia marca <b>1</b>, el máximo."),
            ("<b>Estado GHZ.</b> El segundo CNOT engancha a q₂ a la cadena: (|000⟩ + |111⟩)/√2. Los nodos se "
             "han ido a los polos y los dos anillos de en medio han quedado vacíos. Las tres primeras cifras "
             "no se mueven (q₀ sigue sin estado propio), pero la cuarta se desploma: la concurrencia del par "
             "q₀q₁ vuelve a <b>0</b>. Entrelazar a los tres ha DESHECHO el lazo de la pareja. Los dos siguen "
             "correlacionados (medir uno predice el otro), pero ya no entrelazados: el entrelazamiento de un "
             "GHZ es del conjunto entero y <b>no la suma de lazos entre parejas</b>."),
        ],
        "bl_ent_circuit_title": "Circuito",
        "bl_ent_circuit_alt": "Circuito de tres qubits con las puertas aplicadas hasta ahora",
        "bl_ent_qsphere_title": "Q-sphere del estado conjunto",
        # Las tres primeras dicen lo mismo desde tres ángulos (ver ent_local en app.py): se
        # dejan las tres porque cada lector entra por una — la longitud se contrasta con la
        # figura, la pureza es la magnitud estándar y la entropía es la que cita la literatura.
        # La cuarta mide otra cosa y es la que justifica el tercer qubit: cuánto entrelazamiento
        # queda DENTRO del par q₀q₁ cuando se ignora q₂. Recorre 0 → 0 → 1 → 0.
        "bl_ent_kpi": ["Longitud del vector local |r| (q₀)", "Pureza Tr(ρ₀²)",
                       "Entropía de entrelazamiento", "Concurrencia del par q₀q₁"],
        "bl_ent_bits": "bits",
        "bl_ent_hover_amp": "Amplitud:",
        "bl_ent_hover_prob": "Probabilidad:",
        "bl_ent_hover_shots": "mediciones",
        "bl_ent_meas_title": "Medición",
        "bl_ent_meas_sub": ("La Q-sphere enseña el estado; esto enseña lo único que se puede observar. "
                            "Repite la tirada: la proporción es estable, el recuento exacto no."),
        "bl_ent_meas_n": "Número de mediciones",
        "bl_ent_meas_btn": "Simular mediciones",
        "bl_ent_meas_empty": "Elige cuántas mediciones y pulsa «Simular mediciones».",
        "bl_ent_meas_yaxis": "Veces obtenido",
        "bl_ent_meas_note": [
            ("Con los tres qubits en |0⟩ el resultado es <b>000</b> en todos los disparos. No hay nada que "
             "sortear todavía: es el comportamiento de tres bits clásicos."),
            ("Salen <b>000</b> y <b>100</b> a partes iguales: q₀ se comporta como una moneda al aire y q₁ y "
             "q₂ valen 0 pase lo que pase. Los resultados son <b>independientes</b>: saber uno no dice "
             "nada de los otros."),
            ("Salen <b>000</b> y <b>110</b>: q₀ y q₁ dan siempre el mismo valor, así que medir uno "
             "determina el otro, mientras q₂ sigue clavado en 0 y sin enterarse de nada. Dos de los tres "
             "ya están atados; el tercero todavía mira desde fuera."),
            ("<b>Solo salen 000 y 111</b>, cerca del 50 % cada uno. Las seis barras vacías son el dato: "
             "<b>ninguna de las otras seis combinaciones aparece nunca</b>, ni una vez en diez mil "
             "disparos. Cada qubit sigue dando un resultado al azar, pero los tres dan <b>siempre el "
             "mismo</b>: medir uno determina los otros dos al instante. Esa correlación perfecta es el "
             "entrelazamiento visto desde el laboratorio."),
        ],
        "bl_ent_impl_note": ("<b>Cómo está calculado.</b> Las ocho amplitudes salen de álgebra lineal "
                             "<b>exacta</b> en NumPy (las matrices de H⊗I⊗I y de los dos CNOT aplicadas a "
                             "|000⟩), no de una aproximación; la concurrencia del par es la fórmula de "
                             "Wootters, que para dos qubits da el valor exacto y no una cota; y las mediciones "
                             "salen de un muestreo multinomial sobre |ψ|², que es lo que hace un simulador "
                             "ideal sin ruido. El panel <b>no carga Qiskit</b>: el "
                             "entorno que se despliega es Streamlit, NumPy, Plotly y ONNX Runtime, mientras "
                             "que Qiskit vive en el pipeline de Databricks (donde se entrena el QSVM) y sus "
                             "figuras llegan aquí ya renderizadas, como el circuito de 8 qubits de la página "
                             "Circuito Cuántico. La convención de la base es la de libro de texto, |q₀q₁q₂⟩ "
                             "con q₀ a la izquierda; Qiskit numera al revés y escribiría «001» donde el primer "
                             "paso de aquí escribe «100»."),

        # ── Página 5 · pestaña Esfera de Bloch → el ZZFeatureMap real (8 qubits) ──
        # Tercer y último escalón de la página. Aquí se deja el ejemplo de libro y se mide el
        # circuito del TFM, así que el tono sube medio punto: sigue siendo divulgativo, pero
        # ya puede dar por sabido lo que enseñan las dos secciones de arriba.
        "bl_zz_title": "El ZZFeatureMap real: dónde ocurre el entrelazamiento",
        "bl_zz_sub": ("Las mismas cifras, ahora sobre los 8 qubits del QSVM del TFM. Mueve el deslizador "
                      "del principio de la página y mira qué qubits reaccionan."),
        "bl_zz_intro": ("<b>De tres qubits a los ocho del modelo.</b> Cada qubit del ZZFeatureMap lleva "
                        "<b>una variable clínica</b>: q₀ es la HbA1c, q₁ la glucosa, y así hasta el IMC. "
                        "Con 256 amplitudes ya no hay figura del estado que se pueda mirar (ni Q-sphere ni "
                        "histograma), pero sí se pueden medir las <b>mismas dos magnitudes</b> de la "
                        "sección anterior: cuánto estado propio le queda a cada qubit, y cuánta información "
                        "comparte con cada uno de los demás. Eso ya no es un ejemplo de libro: es el "
                        "circuito con el que se entrenó el modelo."),
        "bl_zz_current": "Variable en juego: <b>{var} = {val} {unidad}</b>. Las otras siete, en su valor de referencia.",
        "bl_zz_r_title": "Estado propio de cada qubit",
        "bl_zz_r_xaxis": "|r|: 1 = conserva su estado · 0 = entrelazado del todo",
        "bl_zz_mi_title": "Información mutua entre qubits",
        "bl_zz_mi_cbar": "bits",
        "bl_zz_note": ("<b>Cómo leer la matriz.</b> Cada celda dice cuánta información comparten dos "
                       "qubits: cuanto más encendida, más atados están. Y salta a la vista que el color "
                       "<b>se concentra en una banda junto a la diagonal</b> y las esquinas quedan vacías. "
                       "No es casualidad: el ZZFeatureMap usa <code>entanglement=\"linear\"</code>, o sea "
                       "que <b>solo hay puertas entre qubits vecinos</b>. Con reps=2 esa correlación llega "
                       "como mucho a cuatro eslabones de distancia; más allá vale <b>cero exacto</b> "
                       "(comprobado sobre 300 perfiles: a distancia ≥ 5 en la cadena, 0,0000 bits sin una "
                       "sola excepción). La topología del circuito se dibuja sola.<br><br>"
                       "Y hay una segunda cosa que se ve moviendo el deslizador de arriba: al cambiar "
                       "<b>una</b> variable solo se mueven <b>su qubit y sus vecinos inmediatos</b>: los "
                       "demás no cambian ni un decimal. Es el mismo hecho visto desde el otro lado: el "
                       "cono de luz del circuito, en vivo."),
        "bl_zz_caveat": ("<b>Una advertencia de lectura, y no menor.</b> El entrelazamiento <b>no crece con "
                         "el valor clínico</b>: subiendo la HbA1c por su rango, |r| del primer qubit hace "
                         "0,88 → 1,00 → 0,65 → 0,24 → 0,73 → 0,98 → 0,33. Sube y baja. La razón es que el "
                         "dato entra como un <b>ángulo</b> de fase y los ángulos <b>dan la vuelta</b>: dos "
                         "valores clínicos muy distintos pueden acabar en fases casi iguales. Con las "
                         "features estandarizadas, un caso extremo llega a x ≈ 5, y el producto "
                         "2·(π−xᵢ)(π−xⱼ) del término de entrelazamiento se pasa de 2π varias veces. Es una "
                         "limitación conocida de la codificación angular sin acotar, y conviene tenerla "
                         "presente antes de leer estas figuras como si midieran gravedad clínica: miden "
                         "geometría del circuito, no riesgo."),

        # ── Página 6 · Predictor en Vivo ──
        "lp_eyebrow": "Inferencia interactiva",
        "lp_title": "Predictor en Vivo",
        "lp_subtitle": ("Probabilidad de que un perfil clínico corresponda a una persona ya diagnosticada de "
                        "diabetes: LightGBM sobre las 8 variables de mayor importancia."),
        "lp_what_note": ("<b>Qué estima este formulario.</b> El objetivo del pipeline es "
                         "<code>TARGET = (DIQ010 == 1)</code>, la respuesta a <i>«¿un médico le ha dicho alguna "
                         "vez que tiene diabetes?»</i>. El modelo, por tanto, <b>detecta diabetes ya "
                         "diagnosticada</b>: no predice quién la desarrollará. Es una tarea de detección "
                         "concurrente, no de riesgo prospectivo."),
        "lp_real_note": ("<b>Inferencia real (ONNX).</b> Predicciones de LightGBM y SVM-RBF vía "
                         "<code>onnxruntime</code>, con el <code>StandardScaler</code> recuperado del pipeline "
                         "Gold. Las 8 variables mostradas son las de mayor importancia clínica; las 81 features "
                         "restantes se fijan en la mediana del conjunto de entrenamiento. QSVM no está disponible "
                         "en tiempo real por el coste O(n²) del kernel cuántico: predecir las 1.567 instancias "
                         "del test costó 144,5 minutos."),
        "lp_proxy_note": ("⚠ <b>Aviso técnico y clínico.</b> Este formulario no tiene conectados los modelos "
                          "serializados reales (<code>.onnx</code>): coloca <code>lgbm_final.onnx</code>, "
                          "<code>svm_final.onnx</code>, <code>scaler_correcto.json</code> y "
                          "<code>medianas_correctas.json</code> en <code>streamlit/models/</code>. La puntuación "
                          "mostrada abajo es un <b>sustituto transparente</b>: una combinación ponderada por "
                          "importancia SHAP normalizada, solo para fines de maquetación. <b>No es la salida de "
                          "ningún modelo entrenado</b> y no debe citarse como resultado. QSVM tampoco está "
                          "disponible en tiempo real (coste O(n²) del kernel cuántico)."),
        "lp_train_range": "Entrenamiento: {mu} ± {sd} (±3 sd → {lo} a {hi})",
        "lp_extrapolates": "⚠ z = {z} · fuera del rango entrenado: el modelo extrapola",
        "lp_ada": "Criterio ADA: &lt; 5,7 normal · 5,7–6,4 prediabetes · ≥ 6,5 diabetes",
        "lp_who_model": "el modelo",
        "lp_who_proxy": "el sustituto",
        "lp_score_real": "Probabilidad de diagnóstico existente",
        "lp_score_proxy": "Score de maquetación (sustituto)",
        "lp_cat_low": "Baja",
        "lp_cat_mid": "Intermedia",
        "lp_cat_high": "Alta",
        "lp_interp_low": ("El perfil queda claramente por debajo del umbral de decisión (50%): {quien} lo "
                          "clasificaría como no diagnosticado."),
        "lp_interp_mid": "El valor se aproxima al umbral de decisión (50%): zona de incertidumbre.",
        "lp_interp_high": ("El valor supera el umbral de decisión (50%): {quien} clasificaría este perfil como "
                           "caso positivo."),
        "lp_badge": "Compatibilidad {cat}",
        "lp_gauge_caption": ("Compatibilidad con un diagnóstico existente: baja · intermedia · alta · "
                             "&nbsp;umbral de decisión = 50%"),
        "lp_side_title": "Los dos modelos, lado a lado",
        "lp_side_sub": "Cada probabilidad se juzga con el punto de corte de su propio modelo: no son intercambiables",
        "lp_own_threshold": "Su umbral",
        "lp_would_classify": "Clasificaría como",
        "lp_positive": "positivo",
        "lp_negative": "negativo",
        "lp_disagree": ("<b>Los dos modelos discrepan {dif} en este perfil.</b> Coinciden en los extremos "
                        "(perfiles claramente sanos o claramente diabéticos) y divergen en la banda intermedia, "
                        "que es justamente donde una estimación sería más útil. Tómalo como señal de "
                        "incertidumbre, no como que uno de los dos acierte."),
        "lp_curve_title": "Curva de respuesta",
        "lp_curve_var": "Variable a recorrer",
        "lp_curve_yaxis": "Probabilidad",
        "lp_curve_thr": "umbral",
        "lp_ada_bands": ["normal", "prediabetes", "diabetes"],
        "lp_curve_note": ("LightGBM sobre esta variable devuelve <b>{n} valores distintos</b> en las {total} "
                          "posiciones del slider: es una escalera, no una rampa. Mayores peldaños: {saltos}. "
                          "El punto marca tu valor actual."),
        "lp_curve_none": "ninguno",
        "lp_read_note": ("<b>Cómo leer estos resultados.</b> Como el objetivo es un diagnóstico <i>ya emitido</i>, "
                         "el modelo aprende también el efecto del <b>tratamiento</b>, no solo el de la enfermedad. "
                         "Eso invierte el sentido clínico de dos variables:"
                         '<ul style="margin:8px 0 0; padding-left:20px; line-height:1.7;">'
                         "<li><b>Colesterol LDL</b>: a más LDL, <i>menor</i> probabilidad estimada (de 43% a 18% "
                         "recorriendo el slider). Los diagnosticados suelen estar tratados con estatinas.</li>"
                         "<li><b>Glucosa en ayunas</b>: la respuesta tiene forma de U: los valores muy bajos "
                         "elevan la estimación tanto como los altos, por las hipoglucemias de pacientes en "
                         "tratamiento.</li></ul>"
                         '<div style="margin-top:10px;">Ninguna de las dos debe leerse como un factor de riesgo '
                         "modificable.</div>"),
    },

    # ═══════════════════════════════ INGLÉS ═══════════════════════════════
    # Inglés académico estándar (convención de la literatura de ML): "winsorization",
    # "modeling", "binarized". Los nombres propios no se traducen — ni el de la
    # universidad, ni los de las librerías, ni los códigos de variable NHANES. Tampoco
    # los nombres de fichero del repositorio (scaler_correcto.json,
    # INSTRUCCIONES_exportar_golden_set.md): son rutas reales, y traducirlas daría una
    # instrucción que no se puede seguir.
    #
    # ORTOGRAFÍA AMERICANA, sin mezclar: "artifact", "behavior", "color", "hemoglobin",
    # "glycemic", "hypoglycemia" — que es el registro de la literatura de ML y el que ya
    # imponían "modeling" y "serialize". Convivían las dos variantes ("colour" en el pie
    # de una figura SHAP y "color" en el de la otra, en la misma aplicación).
    #
    # RAYAS: convención inglesa, que distingue por función y no coincide con la española.
    #   · Puntuando una oración (inciso o pausa), va PEGADA a las dos palabras:
    #     "drops no column at all—and that is exactly what we want to see". El español
    #     hace lo contrario, así que aquí las dos versiones no se parecen.
    #   · Separando un rótulo de su glosa (título, cabecera de tarjeta, insignia de
    #     estado, término de una lista), conserva los espacios: "Silver — cleaning and
    #     sanitation". Ahí no puntúa una oración: separa dos bloques, y pegarla daría
    #     "Silver—cleaning".
    #   · Un inciso que llega hasta el punto final no lleva raya de cierre, que en inglés
    #     sobra: "...QSVM decision_function > 0, which is not a probability."
    "en": {
        # ── Navigation and sidebar ──
        "nav": ["Overview", "Governance", "Results", "SHAP Analysis",
                "Quantum Circuit", "Live Predictor"],
        "sidebar_expand": "Expand the sidebar",
        "sidebar_collapse": "Collapse the sidebar",
        "search_label": "Search",
        "search_ph": "Search the dashboard or the web…",
        "search_expand": "Search: expands the sidebar",
        "scroll_top": "Back to top",
        "search_in": "in {p}",
        "search_none": "No matches in the dashboard.",
        "search_web": "Search “{q}” in:",
        "theme_to_dark": "Switch to dark theme",
        "theme_to_light": "Switch to light theme",
        "lang_es_help": "View the application in Spanish",
        "lang_en_help": "View the application in English",
        "lang_de_help": "View the application in German",
        "lang_fr_help": "View the application in French",
        "lang_it_help": "View the application in Italian",
        "footer_name": "Juan Albornoz C. · MSc Thesis 2026",
        "footer_uni": "Universidad Europea de Valencia",
        "footer_name_narrow": "JAC",
        "footer_uni_narrow": "UEV",

        # ── Page 1 · Overview ──
        "ov_eyebrow": "DataOps + QML Framework",
        "ov_hero_title": ("Integrating Quantum Machine Learning into a DataOps pipeline: "
                          "Medallion architecture on Databricks and benchmark against classical models "
                          "in clinical prediction"),
        "ov_title": "Overview",
        "ov_subtitle": ("End-to-end pipeline on Databricks CE + AWS S3, benchmarking a quantum QSVM "
                        "against two classical baselines, validated on real clinical data from the "
                        "NHANES study (CDC)."),
        "ov_lead": (
            "This framework designs and implements an <b>end-to-end DataOps</b> pipeline on "
            "<b>Databricks Community Edition</b>, using <b>AWS S3</b> as a real cloud storage layer and "
            "a <b>Medallion</b> architecture (Bronze → Silver → Gold) over Delta Lake as its backbone. "
            "The use case predicts type 2 diabetes from records of the <b>NHANES</b> study (CDC): the "
            "dataset is not the object of the research but the vehicle for showing that the architecture "
            "is viable, reproducible and auditable on real data at scale. The experimental core is a "
            "<b>triangulated comparison</b> between LightGBM (tabular baseline), an SVM with RBF kernel "
            "(structural bridge) and a <b>QSVM</b> with FidelityQuantumKernel in Qiskit, keeping the "
            "underlying classifier identical so that any difference in performance can be attributed to "
            "the effect of the quantum kernel."
            " Evaluation closes the path: each model is measured with AUC-ROC, F1, accuracy and "
            "MCC, <b>SHAP</b> points out over LightGBM the 20 variables that weigh most in the "
            "prediction, and the two classical models are serialized to <b>ONNX</b> with their "
            "portability verified. The GitHub repository publishes the 7 notebooks that run that "
            "path, and this very application, deployed on Streamlit Cloud, is its last link: live "
            "prediction and its SHAP reading."),
        "ov_arch_alt": ("Diagram of the pipeline architecture: AWS S3 feeds Databricks Community "
                        "Edition, where the Medallion architecture (Bronze, Silver and Gold) flows into "
                        "three models (LightGBM, SVM with RBF kernel and QSVM with Qiskit) and into "
                        "evaluation with metrics, SHAP and ONNX serialization; the output goes to "
                        "GitHub and Streamlit Cloud."),
        # Rótulos del diagrama de arquitectura que abre la página (arquitectura_svg()). Las
        # cifras NO van escritas: llegan por marcador y las pone mil(), que usa el separador
        # de millar del idioma. Lo que no viaja aquí es la geometría ni qué caja va resaltada,
        # que son dibujo y no texto.
        "ov_arch_io": (
            ("AWS S3", "NHANES raw · 27 XPT", "IAM"),
            ("GitHub", "7 notebooks · README"),
            ("Streamlit Cloud", "Prediction · visual SHAP"),
        ),
        "ov_arch_grupos": (
            ("Medallion architecture", (
                ("Bronze · raw ingestion",
                 "{bronze} rows · 162 cols · Delta Lake ACID"),
                ("Silver · quality",
                 "{silver} rows · 91 cols · expectations"),
                ("Gold · curated features",
                 "89 features · train {train} / test {test}"),
            )),
            ("Models · ML / QML", (
                ("LightGBM",
                 "Tabular baseline · GOSS · EFB"),
                ("SVM · RBF kernel",
                 "Direct bridge to QSVM"),
                ("QSVM · Qiskit",
                 "ZZFeatureMap · FidelityQuantumKernel"),
            )),
            ("Evaluation and serialization", (
                ("Metrics",
                 "AUC-ROC · F1 · Accuracy · MCC"),
                ("SHAP",
                 "LightGBM explainability · top 20"),
                ("Selection · ONNX",
                 "Verified portability"),
            )),
        ),
        "ov_stats_title": "NHANES dataset statistics",
        "ov_stats_sub": "Three biennial cycles integrated · Bronze → Silver → Gold layered pipeline",
        "ov_stat_bronze": "Bronze records",
        "ov_stat_silver": "Silver records",
        "ov_stat_features": "Gold features",
        "ov_stat_balance": "Class balance",
        "ov_medallion_title": "Medallion architecture",
        "ov_medallion_sub": "The data value chain (Curry, 2016) applied layer by layer",
        "ov_layers": [
            ("Bronze", "Ingestion from AWS S3 with no transformation. Preserves the source of truth."),
            ("Silver", "Cleaning, imputation, winsorization and quality validation."),
            ("Gold",   "Scaling, encoding and stratified split. Ready for modeling."),
        ],
        "ov_goto_gov": "See the quality and lineage controls  →",
        "ov_target_title": "Target variable distribution (DIQ010)",
        "ov_target_sub": "Binarized target: 1 = diagnosed diabetes, 0 = all others",
        "ov_pie_no": "No diabetes",
        "ov_pie_yes": "Diabetes",
        # Sin espacio antes del signo de porcentaje: en inglés no lo lleva.
        "ov_donut_center": "14%",
        "ov_donut_caption": "DIABETES",
        "ov_tech_title": "Built on",
        "ov_tech_sub": ("Platform, storage and libraries of the pipeline, in the order they come "
                        "into play · the full inventory, with the rationale for each choice, is in "
                        "Governance"),
        "ov_compare_title": "Triangulated comparison: goal of the experiment",
        "ov_compare": [
            ("LightGBM", "Reference tabular baseline"),
            ("SVM-RBF",  "Structural bridge to the quantum component"),
            ("QSVM",     "FidelityQuantumKernel: same classifier, quantum kernel"),
        ],

        # ── NHANES variables (shared: SHAP, Circuit, Bloch, Predictor) ──
        # NHANES variable codes are never translated; only their gloss is.
        "var_short": {
            "LBXGH": "HbA1c", "RIDAGEYR": "Age", "LBXGLU": "Fasting glucose",
            "LBDLDL": "LDL cholesterol", "BMXWAIST": "Waist circumf.",
            "WTINT2YR": "Survey weight*", "BMXARML": "Arm length", "BMXLEG": "Leg length",
            "BMXBMI": "BMI", "PAD680": "Sedentary activity", "PAD645": "Moderate activity",
            "PAQ640": "Muscle strengthening", "BMXWT": "Body weight", "LBXIN": "Insulin",
            "INDHHIN2": "Household income", "DMDYRSUS": "Years in the US",
            "BMXARMC": "Arm circumf.", "PAQ670": "Vigorous activity",
            "BPXSY1": "Systolic pressure", "PAD630": "Moderate rec. activity",
            "DMDHHSZE": "Household size (children)", "BPXDI1": "Diastolic pressure",
            "LBXTR": "Triglycerides", "DMDMARTL_1": "Marital status (married)",
            "DMDMARTL_5": "Marital status (never married)", "BPXPLS": "Pulse",
            "DMDEDUC2_3": "Education (level 3)", "SDMVSTRA": "Sampling stratum",
            "DMDMARTL_2": "Marital status (widowed)", "DMDHHSZB": "Household size (adults)",
        },
        "var_desc": {
            "LBXGH":      "Glycated hemoglobin (HbA1c): average glucose over the last 2-3 months. Primary diagnostic marker for diabetes (ADA: ≥ 6.5%).",
            "RIDAGEYR":   "Age of the participant at the time of the examination (years).",
            "LBXGLU":     "Fasting plasma glucose: biochemical marker of glycemic control (mg/dL).",
            "LBDLDL":     "Calculated LDL cholesterol: the cholesterol fraction linked to cardiovascular risk (mg/dL).",
            "BMXWAIST":   "Waist circumference: abdominal adiposity associated with insulin resistance (cm).",
            "WTINT2YR":   "NHANES interview sampling weight. An artifact of the survey design, not a clinical variable.",
            "BMXARML":    "Upper arm length (acromion → olecranon): anthropometric measurement (cm).",
            "BMXLEG":     "Maximum leg length (knee → floor): anthropometric measurement (cm).",
            "BMXBMI":     "Body Mass Index (weight/height²): overall body adiposity (kg/m²).",
            "PAD680":     "Minutes of sedentary activity per day (time spent sitting or reclining).",
            "PAD645":     "Weekly minutes of moderate physical activity (work + recreation).",
            "PAQ640":     "Days per week with muscle-strengthening activities.",
            "BMXWT":      "Total body weight (kg).",
            "LBXIN":      "Fasting serum insulin: marker of insulin resistance (µU/mL).",
            "INDHHIN2":   "Household income level (categorical socioeconomic variable).",
            "DMDYRSUS":   "Number of years of residence in the United States.",
            "BMXARMC":    "Mid-upper arm circumference: anthropometric measurement (cm).",
            "PAQ670":     "Weekly minutes of vigorous recreational activity.",
            "BPXSY1":     "Systolic blood pressure, first reading (mmHg).",
            "PAD630":     "Weekly minutes of moderate recreational physical activity.",
            "DMDHHSZE":   "Household composition: number of children in the household.",
            "BPXDI1":     "Diastolic blood pressure, first reading (mmHg).",
            "LBXTR":      "Serum triglycerides: marker of the lipid profile (mg/dL).",
            "DMDMARTL_1": "Marital status = married (dummy variable after one-hot encoding).",
            "DMDMARTL_5": "Marital status = never married (dummy variable after one-hot encoding).",
            "BPXPLS":     "Pulse: resting heart rate (beats/min).",
            "DMDEDUC2_3": "Intermediate education level (high school/GED): dummy variable after one-hot encoding.",
            "SDMVSTRA":   "Variance stratum of the NHANES survey design (methodological variable, not clinical).",
            "DMDMARTL_2": "Marital status = widowed (dummy variable after one-hot encoding).",
            "DMDHHSZB":   "Household composition: number of adults in the household.",
        },
        "qsvm_labels": {
            "LBXGH": "HbA1c", "LBXGLU": "Fasting glucose", "RIDAGEYR": "Age",
            "LBDLDL": "LDL cholesterol", "BMXWAIST": "Waist circumf.", "LBXIN": "Insulin",
            "BMXLEG": "Leg length", "BMXBMI": "BMI",
        },
        "qsvm_units": {"años": "years"},

        # ── Page 2 · Governance ──
        "gov_eyebrow": "Governance · DataOps",
        "gov_title": "Data Governance and Quality",
        "gov_subtitle": ("The controls that hold the pipeline together: what is validated, what is dropped and "
                         "why, what is recorded and with which frameworks. Every figure comes from the executed "
                         "outputs of the notebooks."),
        "gov_tabs": ["Data quality", "Lineage and traceability", "Framework inventory"],
        "gov_kpi_expect": "Expectations passed",
        "gov_kpi_passrate": "Suite pass rate",
        "gov_kpi_records": "Records validated",
        "gov_kpi_leakage": "Leakage-free artifacts",
        "gov_funnel_title": "Record funnel",
        "gov_funnel_sub": ("Of the {bronze} Bronze records, {silver} survive the Silver cohort filters. Each step "
                           "answers an explicit criterion, not a generic cleanup."),
        "gov_hover_records": "Records",
        "gov_hover_dropped": "Dropped",
        "gov_embudo": [
            ("Bronze · 3 cycles joined",
             "27 XPT files · join on SEQN · 162 columns common to all three cycles"),
            ("Age filter ≥ 18 years", "Restriction to the adult population"),
            ("Fasting filter · LBXGLU not null",
             "Proxy for the fasting subgroup: PHAFSTMN is not consistent across cycles"),
            ("Valid DIQ010 filter",
             "Drops codes 7 “don't know” and 9 “refused to answer”, and the nulls"),
        ],
        "gov_dropped_title": "Records dropped per filter",
        "gov_split_label": "Gold 80/20 split",
        "gov_suite_title": "Validation suite · dataframe-expectations",
        "gov_suite_sub": ("Suite <code>{nombre}</code>, run on {fecha} over the {registros} Silver records in "
                          "{duracion} seconds. Great Expectations is incompatible with the pinned versions of the "
                          "serverless runtime: this is the alternative adopted."),
        "gov_expectativas": [
            ("Completeness", "TARGET", "at most 0 nulls"),
            ("Completeness", "LBXGH", "at most 0 nulls"),
            ("Completeness", "LBXGLU", "at most 0 nulls"),
            ("Completeness", "RIDAGEYR", "at most 0 nulls"),
            ("Completeness", "BMXBMI", "at most 0 nulls"),
            ("Clinical ranges", "RIDAGEYR", "minimum between 18 and 25"),
            ("Clinical ranges", "RIDAGEYR", "maximum between 70 and 120"),
            ("Clinical ranges", "LBXGH", "minimum between 3.0 and 6.0"),
            ("Clinical ranges", "LBXGH", "maximum between 8.0 and 20.0"),
            ("Clinical ranges", "LBXGLU", "minimum between 30 and 80"),
            ("Clinical ranges", "LBXGLU", "maximum between 150 and 500"),
            ("Clinical ranges", "BMXBMI", "minimum between 10.0 and 18"),
            ("Clinical ranges", "BMXBMI", "maximum between 40.0 and 80"),
            ("Volume", "DataFrame", "at least 7,000 rows"),
            ("Volume", "DataFrame", "at most 9,000 rows"),
        ],
        "gov_ops_title": "Quality operations by layer",
        "gov_silver_card": "Silver · cleaning and sanitation",
        "gov_gold_card": "Gold · preparation for modeling",
        "gov_silver_ops": [
            ("DIQ variables excluded for leakage", "DIQ050, DIQ070, DIQ160, DIQ170, DIQ172, DIQ180"),
            ("Sparse columns removed", "Threshold of >80% missing values"),
            ("Variables winsorized", "Outlier clipping at IQR × 3"),
            ("Missing after imputation", "From 75,855 to 0 in the SVM/QSVM dataset (median + mode)"),
        ],
        "gov_gold_ops": [
            ("Features after encoding", "One-hot of 5 categorical variables over 84 features (106 columns with TARGET)"),
            ("Dropped by correlation", "Threshold r > 0.90 between pairs of predictors"),
            ("Final features", "The set the three models are trained on"),
            ("Stratified split", "80/20 · 14.03% positives in train, 14.04% in test"),
        ],
        "gov_eff_title": "Effective features versus nominal features",
        "gov_eff_sub": ("Counted from <code>scaler_correcto.json</code>: {const} of the {total} columns have zero "
                        "variance and carry no information for the model"),
        "gov_eff_nominal": "Nominal features",
        "gov_eff_const": "Constant (variance = 0)",
        "gov_eff_effective": "Effective features",
        "gov_eff_note": ("This is a side effect of the IQR × 3 winsorization in Silver, which was also applied to "
                         "categorical variables encoded numerically (1/2 answers, interview language, codes 7 and "
                         "9). When more than 75% of the sample gives the same answer, the clipping collapses the "
                         "column to a single value. The most heavily clipped ones in notebook 02 (PAQ635, PAQ650, "
                         "PAQ605, DMDHHSZA, DMDCITZN, SIALANG) are exactly the ones that appear constant here."),
        "gov_lin_title": "Traceability without MLflow",
        "gov_lin_sub": "The constraint that shapes the pipeline architecture the most, and its mitigation.",
        "gov_lin_limit_title": "Limitation",
        "gov_lin_limit_body": ("Native <b>MLflow</b> integration is disabled on the free Databricks Serverless "
                               "tier. Any call to <code>mlflow.start_run()</code> or "
                               "<code>mlflow.log_metric()</code> raises authentication errors: there is no "
                               "tracking of experiments, metrics or artifacts."),
        "gov_lin_mit_title": "Mitigation · a two-fold mechanism",
        "gov_lin_mit_body": ("<b>Delta Lake transaction logs</b>: every write produces an ACID record with "
                             "version, timestamp and operation metrics.<br><br>"
                             "<b>Per-model metrics CSV</b>: each notebook persists its results to Unity Catalog "
                             "Volumes, and the figures read them from there instead of carrying them hard-coded."),
        "gov_delta_title": "Delta history · Gold layer",
        "gov_delta_sub": ("The six most recent of the ten recorded versions. Delta purges the earlier ones after "
                          "168 h of retention: expected behavior, not a pipeline failure."),
        "gov_delta_cols": ["Version", "Timestamp", "Operation", "Rows", "Size"],
        "gov_chain_title": "Chain of custody against information leakage",
        "gov_chain_sub": ("Four chained barriers. The third one drops no column at all, and that is exactly what "
                          "we want to see: proof that the previous ones did their job."),
        "gov_leakage": [
            ("Exclusion in Silver",
             "Six DIQ treatment and follow-up variables are removed before winsorization: they are a "
             "consequence of the diagnosis, not predictors of it."),
            ("Cross-verification",
             "No DIQ variable is allowed to survive in the 2 Silver Parquet files nor in the 13 Gold "
             "ones. Result: 15/15 clean artifacts."),
            ("QSVM defensive filter",
             "A second barrier before Random Forest selection. It drops no column (89 out of 89 pass)"
             ", which is precisely the proof that the first barrier worked."),
            ("Sampling-weight guard",
             "Halts the pipeline if any sampling weight other than the known one appears. WTINT2YR "
             "does reach the modeling stage and is documented in decision 10."),
        ],
        "gov_scaler_card": "Scaling without statistical leakage",
        "gov_scaler": [
            ("Fit", "Train only", "fit_transform on train · transform on test"),
            ("Columns evaluated", "66", "With variance > 0"),
            ("Constant columns", "23", "Variance 0 · see decision 08"),
            ("Mean ≈ 0 · sd ≈ 1", "Verified", "Assert over every column with dispersion"),
        ],
        "gov_scaler_note": ("The <b>StandardScaler</b> is fitted exclusively on <b>train</b>: "
                            "<code>fit_transform</code> on training and <code>transform</code> on test. Were it "
                            "fitted on the full dataset, the mean and standard deviation of the test set would "
                            "leak into preprocessing and the metrics would come out optimistic. The selection of "
                            "the 8 QSVM variables follows the same rule: the Random Forest is trained only on "
                            "<code>X_train_svm_scaled</code>.<br><br>The correlation filter, by contrast, "
                            "<b>is</b> computed before splitting. This is documented and accepted in decision 09."
                            "<br><br>The check is executed, not claimed: over the 66 columns with dispersion it "
                            "demands |mean| &lt; 0.01 and sd between 0.90 and 1.10. The fitted parameters ("
                            "<code>mean_</code> and <code>scale_</code>) are exported to "
                            "<code>scaler_correcto.json</code>, the file the Live Predictor loads: the train "
                            "scale is never recomputed."),
        "gov_e2e_title": "End-to-end verification against the trained models",
        "gov_e2e_missing": ('<b style="color:{color};">Not verified.</b> The test set is not in the repository, so '
                            "the dashboard cannot check on its own that its inference path reproduces what the "
                            "trained models produced. To close this, run the two cells in "
                            "<code>notebooks/INSTRUCCIONES_exportar_golden_set.md</code> and copy "
                            "<code>golden_lgbm.npz</code> and <code>golden_svm.npz</code> into "
                            "<code>streamlit/models/</code>. While they are missing, this page claims nothing it "
                            "has not been able to check."),
        "gov_e2e_unavailable": "not available",
        "gov_e2e_ok_val": "{n} rows · max. diff. {dif}",
        "gov_e2e_bad_val": "MISMATCH · max. diff. {dif}",
        "gov_e2e_scaled": "scales and calls the ONNX model",
        "gov_e2e_raw": "calls the ONNX model without scaling",
        "gov_e2e_path": "The dashboard {accion}",
        "gov_e2e_ok_title": "✓ Inference path verified",
        "gov_e2e_fail_title": "⚠ The inference path does not reproduce the models",
        "gov_e2e_note": ("Each row of the <i>golden set</i> is a real test instance together with the probability "
                         "returned by the model trained in its notebook. The dashboard runs it through its own "
                         "path (raw vector, scaling for the SVM only, conversion to <code>float32</code>, ONNX "
                         "session and reading of the output tensor) and compares. Tolerance {tol}; the noise "
                         "expected from working in <code>float32</code> is of order 10⁻⁷."),
        "gov_stack_title": "Frameworks by layer",
        "gov_stack_sub": ("The first badge on each card is the framework that structures the layer; the rest "
                          "accompany it."),
        "gov_stack": [
            ("ingestion",
             "boto3 replaces spark.conf, blocked on Serverless (decision 01). Three integrity "
             "asserts: 27/27 files, the join on SEQN does not duplicate rows, and Delta matches "
             "pandas."),
            ("quality",
             "The quality framework of this thesis. Great Expectations is incompatible with the "
             "environment (decision 03). A suite of 15 expectations across 3 dimensions, with "
             "evidence persisted to CSV."),
            ("preparation",
             "Scaling fitted on train only, stratified split with a fixed seed and export of the "
             "serving contract (scaler and medians in JSON)."),
            ("model",
             "Exact interpretability via the polynomial algorithm over the 1,567 test instances, "
             "and verification that the ONNX model reproduces the PKL 100%."),
            ("model",
             "Model-agnostic SHAP, at a cost of hours: computed once over 200 instances and "
             "persisted to disk for reuse."),
            ("model",
             "No ONNX support: the format does not admit quantum operations (decision 05). "
             "Traceability falls to a metrics CSV with the 14 configuration fields."),
        ],
        "gov_dec_title": "Decision log",
        "gov_dec_sub": ("The eleven limitations documented in TECHNICAL_NOTES, with their mitigation. Three shape "
                        "the architecture, six are accepted and documented without correction (because correcting "
                        "them would invalidate the results already obtained) and two are resolved with no "
                        "residue."),
        "gov_dec_tags": {"critical": "Architecture", "warning": "Accepted", "good": "Resolved"},
        "gov_dec_problem": "Problem · ",
        "gov_dec_solution": "Solution adopted · ",
        "gov_decisiones": [
            ("spark.conf blocked on Serverless",
             "Configuring AWS credentials through spark.conf.set, the standard mechanism for "
             "connecting Spark to S3, returns CONFIG_NOT_AVAILABLE.",
             "boto3 as an alternative client. S3 remains the source storage and Unity Catalog Volumes "
             "the processing layer."),
            ("MLflow blocked on Serverless",
             "Native MLflow integration is disabled on the free tier: there is no tracking of "
             "experiments, metrics or artifacts.",
             "A two-fold replacement: Delta Lake transaction logs supply version, timestamp and "
             "operation metrics; and each notebook persists its metrics to CSV."),
            ("Great Expectations incompatible",
             "It requires a pandas/numpy combination that clashes with the pinned versions of the "
             "serverless runtime (pandas 1.5.3 / numpy 1.23.5).",
             "dataframe-expectations 0.7.0 as a compatible alternative. 15 expectations on Silver "
             "across three dimensions. Result 15/15, pass rate 1.0."),
            ("QSVM · O(n²) computational cost",
             "Over the 6,264 training instances, the kernel matrix would demand ~39 million circuit "
             "evaluations. At 1,500 the kernel exhausts memory.",
             "Training on a stratified sample of 500 instances (~22 min) preserving the 86/14 ratio. "
             "Evaluation does use the full test set, so that the metrics remain comparable."),
            ("QSVM · no native ONNX support",
             "The ONNX format does not admit quantum operations: neither skl2onnx nor onnxmltools can "
             "serialize a kernel based on state simulation.",
             "Serialization with joblib. The model requires the Qiskit environment for inference, so "
             "the QSVM is not part of the Live Predictor."),
            ("Qiskit versions cannot be pinned",
             "Databricks' immutable_package_constraints.txt blocks the installation of specific "
             "versions, so there is no exact version reproducibility.",
             "The pipeline runs with the environment's versions (2.5.0 / 0.9.0 / 0.4.0), whose API is "
             "compatible, and they are recorded by an explicit check at the start of the run."),
            ("Loss of variables due to session duration",
             "Long operations (22 min of training, 132 of prediction) can exhaust the serverless "
             "session and take the in-memory variables with it.",
             "Immediate persistence after every costly operation, plus a TRAINING_MODE that reloads "
             "from disk on subsequent runs."),
            ("Winsorization applied to encoded categoricals",
             "NHANES encodes many categorical variables numerically. If more than 75% share a value, "
             "IQR = 0, the bounds collapse and clip() turns the variable into a constant. 10 columns "
             "were collapsed this way.",
             "Documented without modification: fixing it would alter Silver, Gold and all three "
             "models. Constant columns do not bias anything (the model extracts no signal from "
             "them), but information is lost. The fix is identified as future work."),
            ("Correlation computed before splitting",
             "The r > 0.90 filter is computed on the full dataset, so the 16 dropped columns are "
             "decided using the test observations as well.",
             "Documented without modification. It affects neither the scaling nor the QSVM feature "
             "selection, both fitted on train only, but the selection is no longer strictly blind to "
             "the test set."),
            ("Sampling weight WTINT2YR among the features",
             "The intra-cycle join duplicates WTSAF2YR across three columns. WTINT2YR is not on the "
             "exclusion list and survives the correlation filter: it is one of the 89 features.",
             "Documented without modification, plus an assert that detects the appearance of any "
             "OTHER weight. A sampling weight is not a clinical variable: it does not leak the "
             "target, but it lets the model lean on the survey design."),
            ("The serialized QSVM is not reloadable across versions",
             "The pickle drags the ZZFeatureMap along with its ParameterExpression objects. If Qiskit "
             "changes version, deserialization fails, and Serverless updates without warning.",
             "Loading is wrapped in try/except: on failure, TRAINING_MODE switches to True and the "
             "notebook retrains instead of aborting. It stays operational in all three scenarios."),
        ],
        "gov_footer_note": ("The figures on this page come from the executed outputs of the repository notebooks "
                            "and from <code>TECHNICAL_NOTES.md</code>; none is estimated. The application cannot "
                            "query them live because Streamlit Community Cloud only reaches the repository, not "
                            "Unity Catalog Volumes.<br><br>"
                            "Quality suite summary: <b>{fuente}</b>."),
        "gov_suite_src_csv": "read from validacion_silver_dfe.csv",
        "gov_suite_src_nb": "values verified from the notebook",

        # ── Page 3 · Results ──
        "res_eyebrow": "Triangulated comparison",
        "res_title": "Results",
        "res_subtitle": "LightGBM vs. SVM-RBF vs. QSVM on the same test set ({n} instances).",
        "res_threshold": "Threshold",
        "res_thr_label": {"lightgbm": "p ≥ {v}", "svm_rbf": "p ≈ {v}", "qsvm": "df > 0"},
        "res_thr_src": {"lightgbm": "predict_proba()[:,1] >= 0.5",
                        "svm_rbf": "SVC.predict() · sign of decision_function",
                        "qsvm": "decision_function > 0 (not a probability)"},
        "res_reconciled": ('<span style="color:{color}; font-weight:600;">✓ Reconciled</span>: the four metrics '
                           "of all three models have been recomputed from the per-instance scores and match the "
                           "published ones."),
        "res_unreconciled": '<span style="color:{color}; font-weight:600;">⚠ Not reconciled</span>: {fallos}',
        "res_no_scores": "scores not available",
        "res_threshold_note": ("<b>The three models are measured at different thresholds.</b> Each one uses its "
                               "natural cut-off point: LightGBM <code>predict_proba ≥ 0.50</code>; SVM-RBF the "
                               "sign of <code>decision_function</code>, which on the stored probability scale is "
                               "equivalent to ≈ 0.22; QSVM <code>decision_function &gt; 0</code>, which is not a "
                               "probability. Each matrix reproduces exactly at its own threshold, but <b>only "
                               "AUC-ROC is comparable across models</b>: it is the only one of the four metrics "
                               "independent of the cut-off point. For reference, SVM-RBF evaluated at 0.50 like "
                               "LightGBM would give accuracy 0.9190 but only 131 true positives instead of 172."),
        "res_roc_title": "ROC curves",
        "res_roc_sub_real": ("Real empirical curves, point by point over the 1,567 test instances (the same "
                             "scores that report the AUC of the thesis)."),
        "res_roc_sub_synth": "Exact AUC · shape reconstructed from the AUC where per-instance scores are missing.",
        "res_cm_title": "Confusion matrices",
        "res_cm_sub": ("Values verified against the classification report of each model, and recomputed from the "
                       "per-instance scores. Each matrix corresponds to the threshold stated on its card"),
        "res_cm_pred_no": "Pred.<br>No diabetes",
        "res_cm_pred_yes": "Pred.<br>Diabetes",
        "res_cm_real_no": "Actual<br>No diab.",
        "res_cm_real_yes": "Actual<br>Diabetes",
        "res_cm_tags": {"tn": "TN", "fp": "FP", "fn": "FN", "tp": "TP"},
        "res_metrics_title": "Metric comparison",
        "res_metrics_sub": ("The four metrics are computed over the 1,567 instances. Accuracy, MCC and F1-macro "
                            "do penalize class imbalance, but they depend on the threshold, and each model uses "
                            "its own: compare anything other than AUC-ROC with caution"),
        "res_metric_desc": {
            "auc": "Area under the ROC curve: ability to separate diabetes vs. non-diabetes. 0.5 = chance, 1 = perfect.",
            "f1_macro": "Harmonic mean of precision and recall averaged per class (unweighted). Penalizes imbalance.",
            "accuracy": "Proportion of overall correct predictions. With imbalanced classes it may reflect only the majority class.",
            "mcc": "Matthews correlation coefficient: overall quality, robust to imbalance. 0 = chance, 1 = perfect.",
        },
        "res_qsvm_note": ("<b>Note on the QSVM experiment.</b> The QSVM was trained on a stratified sample of 500 "
                          "instances (O(n²) cost of the quantum kernel) and evaluated on the full test set of "
                          "1,567. AUC-ROC = 0.5493 indicates that the model barely beats random classification: "
                          "Recall ≈ 0 for the diabetes class (1 out of 220), and Accuracy = 0.8602 reflects only "
                          "the proportion of the majority class. MCC ≈ 0 confirms the absence of any real "
                          "predictive ability."),

        # ── Page 4 · SHAP Analysis ──
        "sh_eyebrow": "Interpretability",
        "sh_title": "SHAP Analysis",
        "sh_subtitle": ("Global feature importance: TreeExplainer (LightGBM) vs. KernelExplainer (SVM-RBF)."),
        "sh_tabs": ["LightGBM · TreeExplainer", "SVM-RBF · KernelExplainer"],
        "sh_hint": "Hover over each bar to see what the variable means. {nota}",
        "sh_sample_lgbm": "Exact values (polynomial algorithm) over the 1,567 test instances.",
        "sh_sample_svm": ("Values approximated by sampling: background of 100 instances, contributions over 200 "
                          "test instances."),
        "sh_note_lgbm": ("<b>LBXGH (HbA1c)</b> dominates by a wide margin (mean SHAP = 1.1243), consistent with "
                         "its role as the primary diagnostic marker for type 2 diabetes (ADA: HbA1c ≥ 6.5%). "
                         "<b>RIDAGEYR (age, 0.4654)</b> reflects the rise in prevalence with age. <b>LBXGLU</b> "
                         "and <b>LBDLDL</b> complete the biochemical block. <b>WTINT2YR</b> (position 6) is an "
                         "artifact of the NHANES survey design, not a clinical variable."),
        "sh_note_svm": ("The SVM-RBF ranking agrees with LightGBM on the dominant variables (<b>LBXGH</b>, "
                        "<b>LBXGLU</b>, <b>LBDLDL</b>, <b>RIDAGEYR</b>), which strengthens the clinical validity "
                        "of the finding by making it independent of the algorithm and methodologically more "
                        "robust. KernelExplainer treats the model as a black box, so it applies to any "
                        "classifier."),
        "sh_fig_lgbm_title": "SHAP Summary Plot · LightGBM (Figure 27)",
        "sh_fig_lgbm_cap": ("Each point is a test instance; color indicates the value of the variable (red high, "
                            "blue low) and the horizontal position its impact on the prediction. LBXGH and "
                            "RIDAGEYR dominate the model."),
        "sh_fig_svm_title": "SHAP Summary Plot · SVM-RBF (Figure 31)",
        "sh_fig_svm_cap": ("Each point is an instance; color = value of the variable, position = impact. "
                           "KernelExplainer over 200 test instances."),

        # ── Page 5 · Quantum Circuit ──
        "qc_eyebrow": "Quantum component",
        "qc_title": "Quantum Circuit",
        "qc_subtitle": ("Configuration of the ZZFeatureMap and FidelityQuantumKernel implemented in Qiskit on "
                        "Databricks CE."),
        "qc_tabs": ["ZZFeatureMap circuit", "Bloch Sphere"],
        "qc_specs": ["Qubits (feature_dimension)", "Repetitions (reps)", "Entanglement", "Qiskit version"],
        "qc_how_title": "How it works",
        "qc_how_p1": ("The <b>ZZFeatureMap</b> encodes each of the 8 clinical variables as a phase angle (P gate) "
                      "on an independent qubit, after creating superposition with Hadamard gates. Its "
                      "distinguishing element is the <b>entanglement</b> between pairs of qubits through gates "
                      "that depend on the cross product of two variables, correlations the classical RBF kernel "
                      "cannot represent."),
        "qc_how_p2": ("The <b>FidelityQuantumKernel</b> measures the similarity between two patients as the "
                      "fidelity between their quantum states: <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. The "
                      "implementation uses <code>StatevectorSampler</code>, simulating the exact state without "
                      "noise: deterministic, reproducible results."),
        "qc_feat_title": "8 selected features (Random Forest)",
        "qc_xaxis": "RF importance",
        "qc_train_title": "Training and evaluation",
        "qc_tstats": ["Training instances", "Training time", "Test instances",
                      "Inference time", "Support vectors"],
        "qc_note": ("Because of the O(n²) cost of the quantum kernel, training was limited to a stratified sample "
                    "of 500 instances (the operational limit of Databricks CE serverless sits around 500-1,000). "
                    "Evaluation was performed on the full test set (1,567 instances) in batches of 100, with a "
                    "total prediction time of 144.5 minutes."),
        "qc_circuit_title": "Complete quantum circuit (8 qubits)",
        "qc_circuit_sub": ("ZZFeatureMap with reps=2: encoding (H + P) followed by two rounds of linear "
                           "entanglement between adjacent qubits."),

        # ── Page 5 · Quantum Circuit → Bloch Sphere tab ──
        "bl_title": "Bloch Sphere",
        "bl_subtitle": "How the ZZFeatureMap encodes the value of a clinical variable as a quantum state |ψ⟩.",
        "bl_what_note": ("<b>What the Bloch sphere is.</b> A classical bit can only be 0 or 1. "
                         "A qubit also admits any mixture of the two, and that mixture does not fit "
                         "into a single number: it needs a map. The Bloch sphere is that map: every "
                         "possible state of one qubit is a point on the surface of a sphere of radius "
                         "1. The north pole is <b>|0⟩</b> and the south pole <b>|1⟩</b>; the "
                         "superpositions lie in between, and the closer the arrow sits to a pole, the "
                         "likelier that outcome when measured. Here the clinical value is turned into "
                         "the angle θ, so moving the slider swings the arrow along a meridian, from "
                         "|0⟩ to |1⟩."),
        "bl_var": "Clinical variable",
        "bl_value": "Value ({unidad})",
        "bl_xnorm": "normalized x",
        "bl_theta": "θ = x_norm·π",
        "bl_alpha": "α (amplitude |0⟩)",
        "bl_beta": "β (amplitude |1⟩)",
        "bl_rad": "rad",
        "bl_note": ("<b>A didactic analogy of the angular encoding principle</b>, not a replica of the circuit. "
                    "Here the clinical value normalized to [0,1] becomes the <b>polar</b> angle θ = x_norm·π, so "
                    "the vector travels the meridian from |0⟩ to |1⟩ and P(|0⟩) varies from 100% to 0%: it is the "
                    "most legible way of seeing “a number becomes a state”.<br><br>"
                    "The <b>real ZZFeatureMap</b> does something different: it applies H and then "
                    "P(2·x<sub>i</sub>), and a phase gate after a Hadamard leaves the state <b>on the "
                    "equator</b> (θ = π/2 fixed, P(|0⟩) = P(|1⟩) = 50% always), encoding the datum in the "
                    "<b>azimuthal</b> angle φ, not the polar one. Nor does it normalize to [0,1]: it uses the "
                    "scaled value directly. That is why this sphere illustrates the concept but does not "
                    "reproduce the circuit step by step. Entanglement (gates "
                    "P(2·(π−x<sub>i</sub>)·(π−x<sub>j</sub>))) is only representable in the joint space of the "
                    "8 qubits (see Quantum Circuit)."),

        # ── Page 5 · Bloch Sphere tab → entanglement section ──
        "bl_ent_title": "Entanglement: three qubits, a single state",
        "bl_ent_sub": ("Where the sphere above stops working. Apply the three gates and watch what happens "
                       "to each qubit's local state, and to the pair left behind."),
        "bl_ent_intro": ("<b>Where the Bloch sphere stops working.</b> One qubit needs one sphere and one "
                         "arrow. With several, the temptation is to draw one sphere per qubit, and for most "
                         "states that works. But there is a family of states where <b>no arrow is left to "
                         "draw</b>: the whole has a perfectly defined state and none of its members has one "
                         "on its own. That is entanglement, and here it takes three gates to build. Apply "
                         "them and follow the four figures on the left: the first three break at the second "
                         "step, and at the third the fourth one shows something two qubits cannot even pose."),
        "bl_ent_btn_h": "1 · Hadamard on q₀",
        "bl_ent_btn_cnot1": "2 · CNOT (control q₀ → q₁)",
        "bl_ent_btn_cnot2": "3 · CNOT (control q₁ → q₂)",
        "bl_ent_btn_reset": "Reset to |000⟩",
        "bl_ent_step_note": [
            ("<b>Starting point.</b> Three qubits, all in |0⟩, no gates applied. The joint state is |000⟩ "
             "and there is nothing quantum about it yet: it is exactly equivalent to three classical bits "
             "set to zero. The Q-sphere shows a single node at the north pole, holding all the probability."),
            ("<b>Superposition, not yet entangled.</b> The Hadamard leaves q₀ halfway between |0⟩ and |1⟩ "
             "while q₁ and q₂ stay firmly in |0⟩: the joint state is (|000⟩ + |100⟩)/√2. The three qubits "
             "are still <b>independent</b>: each has its own pure state, and three Bloch spheres would "
             "describe them perfectly well. Note that the local vector length is still 1: there is an arrow "
             "to draw."),
            ("<b>A Bell pair, and a bystander.</b> The first CNOT flips q₁ only when q₀ is 1; applied to a "
             "superposition, that ties both outcomes into one: (|000⟩ + |110⟩)/√2. This is where the map "
             "breaks: the local vector length of q₀ has just dropped to <b>0</b>, the qubit is no longer at "
             "any point of its sphere because on its own it <b>no longer has a state</b>. And something "
             "else happens, visible only because there is a third qubit: q₂ has been left OUT, watching from "
             "|0⟩, and what is entangled is exactly the pair q₀q₁. Its concurrence reads <b>1</b>, the "
             "maximum."),
            ("<b>GHZ state.</b> The second CNOT hooks q₂ onto the chain: (|000⟩ + |111⟩)/√2. The nodes have "
             "moved to the poles and the two middle rings are empty. The first three figures do not "
             "move (q₀ still has no state of its own), but the fourth collapses: the concurrence of the pair "
             "q₀q₁ is back to <b>0</b>. Entangling all three has UNDONE the link within the pair. The two "
             "remain correlated (measuring one predicts the other), but no longer entangled: the entanglement "
             "of a GHZ belongs to the whole and is <b>not the sum of pairwise links</b>."),
        ],
        "bl_ent_circuit_title": "Circuit",
        "bl_ent_circuit_alt": "Three-qubit circuit showing the gates applied so far",
        "bl_ent_qsphere_title": "Q-sphere of the joint state",
        "bl_ent_kpi": ["Local vector length |r| (q₀)", "Purity Tr(ρ₀²)", "Entanglement entropy",
                       "Concurrence of the pair q₀q₁"],
        "bl_ent_bits": "bits",
        "bl_ent_hover_amp": "Amplitude:",
        "bl_ent_hover_prob": "Probability:",
        "bl_ent_hover_shots": "measurements",
        "bl_ent_meas_title": "Measurement",
        "bl_ent_meas_sub": ("The Q-sphere shows the state; this shows the only thing that can be observed. "
                            "Run it again: the proportion is stable, the exact counts are not."),
        "bl_ent_meas_n": "Number of measurements",
        "bl_ent_meas_btn": "Simulate measurements",
        "bl_ent_meas_empty": "Choose how many measurements and press “Simulate measurements”.",
        "bl_ent_meas_yaxis": "Times obtained",
        "bl_ent_meas_note": [
            ("With all three qubits in |0⟩ the outcome is <b>000</b> on every shot. There is nothing to "
             "sample yet: this is how three classical bits behave."),
            ("<b>000</b> and <b>100</b> come out in equal parts: q₀ behaves like a coin toss while q₁ and "
             "q₂ are 0 no matter what. The outcomes are <b>independent</b>: knowing one tells you nothing "
             "about the others."),
            ("<b>000</b> and <b>110</b> come out: q₀ and q₁ always give the same value, so measuring one "
             "determines the other, while q₂ stays pinned at 0 and none the wiser. Two of the three are "
             "tied together; the third is still watching from outside."),
            ("<b>Only 000 and 111 appear</b>, close to 50% each. The six empty bars are the finding: <b>none "
             "of the other six combinations ever comes out</b>, not once in ten thousand shots. Each qubit "
             "still gives a random outcome, but all three give <b>always the same one</b>: measuring one "
             "determines the other two instantly. That perfect correlation is entanglement seen from the "
             "lab."),
        ],
        "bl_ent_impl_note": ("<b>How this is computed.</b> The eight amplitudes come from <b>exact</b> linear "
                             "algebra in NumPy (the H⊗I⊗I matrix and the two CNOTs applied to |000⟩), not from "
                             "an approximation; the pair's concurrence is Wootters' formula, which for two "
                             "qubits gives the exact value rather than a bound; and the measurements come "
                             "from multinomial sampling over |ψ|², which "
                             "is what an ideal noiseless simulator does. This dashboard <b>does not load "
                             "Qiskit</b>: the deployed environment is Streamlit, NumPy, Plotly and ONNX "
                             "Runtime, while Qiskit lives in the Databricks pipeline (where the QSVM is "
                             "trained) and its figures arrive here already rendered, like the 8-qubit circuit "
                             "on the Quantum Circuit page. The basis convention is the textbook one, |q₀q₁q₂⟩ "
                             "with q₀ on the left; Qiskit numbers the other way and would write “001” where "
                             "the first step here writes “100”."),

        # ── Page 5 · Bloch Sphere tab → the real ZZFeatureMap (8 qubits) ──
        "bl_zz_title": "The real ZZFeatureMap: where entanglement actually happens",
        "bl_zz_sub": ("The same figures, now over the 8 qubits of the TFM's QSVM. Move the slider at the "
                      "top of the page and watch which qubits react."),
        "bl_zz_intro": ("<b>From three qubits to the model's eight.</b> Each qubit of the ZZFeatureMap "
                        "carries <b>one clinical variable</b>: q₀ is HbA1c, q₁ glucose, and so on up to "
                        "BMI. With 256 amplitudes there is no picture of the state left to look at (neither "
                        "Q-sphere nor histogram), but the <b>same two quantities</b> from the previous "
                        "section can still be measured: how much of its own state each qubit keeps, and how "
                        "much information it shares with every other one. This is no longer a textbook "
                        "example: it is the circuit the model was trained with."),
        "bl_zz_current": "Variable in play: <b>{var} = {val} {unidad}</b>. The other seven at their reference value.",
        "bl_zz_r_title": "Each qubit's own state",
        "bl_zz_r_xaxis": "|r|: 1 = keeps its own state · 0 = fully entangled",
        "bl_zz_mi_title": "Mutual information between qubits",
        "bl_zz_mi_cbar": "bits",
        "bl_zz_note": ("<b>How to read the matrix.</b> Each cell says how much information two qubits "
                       "share: the brighter, the more tightly bound. And it jumps out that the colour "
                       "<b>concentrates in a band along the diagonal</b> while the corners stay empty. "
                       "That is no accident: the ZZFeatureMap uses <code>entanglement=\"linear\"</code>, "
                       "meaning <b>there are only gates between neighbouring qubits</b>. With reps=2 that "
                       "correlation reaches at most four links away; beyond that it is <b>exactly zero</b> "
                       "(checked over 300 profiles: at distance ≥ 5 along the chain, 0.0000 bits without a "
                       "single exception). The circuit's topology draws itself.<br><br>"
                       "And there is a second thing you can see by moving the slider above: changing "
                       "<b>one</b> variable moves only <b>its qubit and its immediate neighbours</b>: the "
                       "rest do not shift by a single decimal. It is the same fact from the other side: "
                       "the circuit's light cone, live."),
        "bl_zz_caveat": ("<b>A reading caveat, and not a minor one.</b> Entanglement <b>does not grow with "
                         "the clinical value</b>: raising HbA1c across its range, |r| for the first qubit "
                         "goes 0.88 → 1.00 → 0.65 → 0.24 → 0.73 → 0.98 → 0.33. Up and down. The reason is "
                         "that the datum enters as a phase <b>angle</b>, and angles <b>wrap around</b>: two "
                         "very different clinical values can land on nearly identical phases. With "
                         "standardized features an extreme case reaches x ≈ 5, and the entangling term's "
                         "product 2·(π−xᵢ)(π−xⱼ) passes 2π several times over. This is a known limitation "
                         "of unbounded angle encoding, and worth keeping in mind before reading these "
                         "figures as if they measured clinical severity: they measure circuit geometry, "
                         "not risk."),

        # ── Page 6 · Live Predictor ──
        "lp_eyebrow": "Interactive inference",
        "lp_title": "Live Predictor",
        "lp_subtitle": ("Probability that a clinical profile corresponds to a person already diagnosed with "
                        "diabetes: LightGBM over the 8 most important variables."),
        "lp_what_note": ("<b>What this form estimates.</b> The target of the pipeline is "
                         "<code>TARGET = (DIQ010 == 1)</code>, the answer to <i>“has a doctor ever told you that "
                         "you have diabetes?”</i>. The model therefore <b>detects already diagnosed diabetes</b>: "
                         "it does not predict who will develop it. This is a concurrent detection task, not "
                         "prospective risk."),
        "lp_real_note": ("<b>Real inference (ONNX).</b> Predictions from LightGBM and SVM-RBF via "
                         "<code>onnxruntime</code>, with the <code>StandardScaler</code> recovered from the Gold "
                         "pipeline. The 8 variables shown are the ones of greatest clinical importance; the "
                         "remaining 81 features are fixed at the median of the training set. QSVM is not "
                         "available in real time because of the O(n²) cost of the quantum kernel: predicting the "
                         "1,567 test instances took 144.5 minutes."),
        "lp_proxy_note": ("⚠ <b>Technical and clinical warning.</b> This form does not have the real serialized "
                          "models (<code>.onnx</code>) connected: place <code>lgbm_final.onnx</code>, "
                          "<code>svm_final.onnx</code>, <code>scaler_correcto.json</code> and "
                          "<code>medianas_correctas.json</code> in <code>streamlit/models/</code>. The score "
                          "shown below is a <b>transparent stand-in</b>: a combination weighted by normalized "
                          "SHAP importance, for layout purposes only. <b>It is not the output of any trained "
                          "model</b> and must not be cited as a result. QSVM is likewise unavailable in real time "
                          "(O(n²) cost of the quantum kernel)."),
        "lp_train_range": "Training: {mu} ± {sd} (±3 sd → {lo} to {hi})",
        "lp_extrapolates": "⚠ z = {z} · outside the trained range: the model extrapolates",
        "lp_ada": "ADA criterion: &lt; 5.7 normal · 5.7–6.4 prediabetes · ≥ 6.5 diabetes",
        "lp_who_model": "the model",
        "lp_who_proxy": "the stand-in",
        "lp_score_real": "Probability of an existing diagnosis",
        "lp_score_proxy": "Layout score (stand-in)",
        "lp_cat_low": "Low",
        "lp_cat_mid": "Intermediate",
        "lp_cat_high": "High",
        "lp_interp_low": ("The profile falls clearly below the decision threshold (50%): {quien} would classify "
                          "it as not diagnosed."),
        "lp_interp_mid": "The value approaches the decision threshold (50%): a zone of uncertainty.",
        "lp_interp_high": ("The value exceeds the decision threshold (50%): {quien} would classify this profile "
                           "as a positive case."),
        "lp_badge": "{cat} compatibility",
        "lp_gauge_caption": ("Compatibility with an existing diagnosis: low · intermediate · high · "
                             "&nbsp;decision threshold = 50%"),
        "lp_side_title": "The two models, side by side",
        "lp_side_sub": ("Each probability is judged against the cut-off point of its own model: they are not "
                        "interchangeable"),
        "lp_own_threshold": "Its threshold",
        "lp_would_classify": "Would classify as",
        "lp_positive": "positive",
        "lp_negative": "negative",
        "lp_disagree": ("<b>The two models disagree by {dif} on this profile.</b> They agree at the extremes"
                        "(clearly healthy or clearly diabetic profiles) and diverge in the intermediate band, "
                        "which is precisely where an estimate would be most useful. Read it as a signal of "
                        "uncertainty, not as one of the two being right."),
        "lp_curve_title": "Response curve",
        "lp_curve_var": "Variable to sweep",
        "lp_curve_yaxis": "Probability",
        "lp_curve_thr": "threshold",
        "lp_ada_bands": ["normal", "prediabetes", "diabetes"],
        "lp_curve_note": ("Over this variable LightGBM returns <b>{n} distinct values</b> across the {total} "
                          "slider positions: it is a staircase, not a ramp. Largest steps: {saltos}. The dot "
                          "marks your current value."),
        "lp_curve_none": "none",
        "lp_read_note": ("<b>How to read these results.</b> Because the target is an <i>already issued</i> "
                         "diagnosis, the model also learns the effect of <b>treatment</b>, not only that of the "
                         "disease. This inverts the clinical meaning of two variables:"
                         '<ul style="margin:8px 0 0; padding-left:20px; line-height:1.7;">'
                         "<li><b>LDL cholesterol</b>: the higher the LDL, the <i>lower</i> the estimated "
                         "probability (from 43% to 18% across the slider). Diagnosed patients are usually on "
                         "statins.</li>"
                         "<li><b>Fasting glucose</b>: the response is U-shaped: very low values raise the "
                         "estimate as much as high ones, because of hypoglycemia in treated patients.</li></ul>"
                         '<div style="margin-top:10px;">Neither should be read as a modifiable risk '
                         "factor.</div>"),
    },

    # ═══════════════════════════════ ALEMÁN ════════════════════════════════
    # Alemán estándar de Alemania, registro académico. Rige lo mismo que en inglés para
    # lo que NO se traduce: nombres propios, librerías, códigos de variable NHANES y
    # nombres de fichero del repositorio (scaler_correcto.json,
    # INSTRUCCIONES_exportar_golden_set.md) — son rutas reales y traducirlas daría una
    # instrucción que no se puede seguir.
    #
    # NOTACIÓN NUMÉRICA: la alemana coincide con la española — coma decimal, punto de
    # millar y espacio antes del signo de porcentaje (DIN 5008). Por eso nf(), pct(),
    # mil() y el `separators` de Plotly agrupan "de" con "es" en vez de con "en", y por
    # eso las cifras escritas dentro de estas frases van con coma, igual que arriba.
    #
    # COMILLAS: las alemanas („…“), bajas de apertura y altas de cierre, que es lo que
    # espera el lector y lo que ya hacen las otras dos con las suyas.
    #
    # RAYAS: en alemán las dos funciones que el inglés distingue llevan el mismo signo,
    # con espacios a ambos lados — puntuando la oración («…zwei Torte — und genau das…»)
    # y separando un rótulo de su glosa («Silver — Bereinigung»). Aquí no hay, por tanto,
    # la doble convención de la cabecera inglesa.
    #
    # COMPUESTOS: unidos según el Duden, y con guion cuando un miembro es sigla o nombre
    # propio ("SHAP-Analyse", "Bloch-Kugel", "ZZFeatureMap"). Es lo que evita cadenas
    # ilegibles justo en los rótulos cortos, que es donde más se nota.
    "de": {
        # ── Navigation und Seitenleiste ──
        "nav": ["Übersicht", "Governance", "Ergebnisse", "SHAP-Analyse",
                "Quantenschaltkreis", "Live-Prädiktor"],
        "sidebar_expand": "Seitenleiste ausklappen",
        "sidebar_collapse": "Seitenleiste einklappen",
        "search_label": "Suchen",
        "search_ph": "Im Dashboard oder im Web suchen…",
        "search_expand": "Suchen: klappt die Seitenleiste aus",
        "scroll_top": "Nach oben",
        "search_in": "in {p}",
        "search_none": "Keine Treffer im Dashboard.",
        "search_web": "„{q}“ suchen in:",
        "theme_to_dark": "Zum dunklen Design wechseln",
        "theme_to_light": "Zum hellen Design wechseln",
        "lang_es_help": "Die Anwendung auf Spanisch ansehen",
        "lang_en_help": "Die Anwendung auf Englisch ansehen",
        "lang_de_help": "Die Anwendung auf Deutsch ansehen",
        "lang_fr_help": "Die Anwendung auf Französisch ansehen",
        "lang_it_help": "Die Anwendung auf Italienisch ansehen",
        "footer_name": "Juan Albornoz C. · Masterarbeit 2026",
        "footer_uni": "Universidad Europea de Valencia",
        "footer_name_narrow": "JAC",
        "footer_uni_narrow": "UEV",

        # ── Seite 1 · Übersicht ──
        "ov_eyebrow": "DataOps- und QML-Framework",
        "ov_hero_title": ("Integration von Quantum Machine Learning in eine DataOps-Pipeline: "
                          "Medaillon-Architektur auf Databricks und Vergleich mit klassischen Modellen "
                          "in der klinischen Vorhersage"),
        "ov_title": "Übersicht",
        "ov_subtitle": ("End-to-End-Pipeline auf Databricks CE + AWS S3, mit einer Quanten-QSVM gegen "
                        "zwei klassische Baselines, validiert an echten klinischen Daten der "
                        "NHANES-Studie (CDC)."),
        "ov_lead": (
            "Dieses Framework entwirft und implementiert eine <b>End-to-End-DataOps</b>-Pipeline auf "
            "<b>Databricks Community Edition</b>, mit <b>AWS S3</b> als echter Cloud-Speicherschicht "
            "und einer <b>Medaillon</b>-Architektur (Bronze → Silver → Gold) über Delta Lake als "
            "Rückgrat. Als Anwendungsfall wird Typ-2-Diabetes anhand von Datensätzen der "
            "<b>NHANES</b>-Studie (CDC) vorhergesagt: Der Datensatz ist nicht der Gegenstand der "
            "Untersuchung, sondern das Vehikel, um zu zeigen, dass die Architektur an echten Daten "
            "im großen Maßstab tragfähig, reproduzierbar und prüfbar ist. Der experimentelle Kern "
            "ist ein <b>triangulierter Vergleich</b> zwischen LightGBM (tabellarische Baseline), "
            "einer SVM mit RBF-Kernel (strukturelle Brücke) und einer <b>QSVM</b> mit "
            "FidelityQuantumKernel in Qiskit, wobei der zugrunde liegende Klassifikator identisch "
            "bleibt, damit jeder Leistungsunterschied dem Effekt des Quantenkernels zugeschrieben "
            "werden kann."
            " Die Auswertung schließt den Weg ab: Jedes Modell wird mit AUC-ROC, F1, Accuracy und "
            "MCC gemessen, <b>SHAP</b> zeigt an LightGBM die 20 Variablen, die in der Vorhersage am "
            "schwersten wiegen, und die beiden klassischen Modelle werden mit geprüfter "
            "Portabilität nach <b>ONNX</b> serialisiert. Das GitHub-Repository veröffentlicht die 7 "
            "Notebooks, die diesen Weg ausführen, und diese Anwendung selbst, auf Streamlit Cloud "
            "bereitgestellt, ist ihr letztes Glied: die Vorhersage in Echtzeit und ihre SHAP-Lesart."),
        "ov_arch_alt": ("Diagramm der Pipeline-Architektur: AWS S3 speist Databricks Community "
                        "Edition, wo die Medaillon-Architektur (Bronze, Silver und Gold) in drei "
                        "Modelle (LightGBM, SVM mit RBF-Kernel und QSVM mit Qiskit) und in die "
                        "Auswertung mit Metriken, SHAP und ONNX-Serialisierung mündet; die Ausgabe "
                        "geht an GitHub und Streamlit Cloud."),
        # Rótulos del diagrama de arquitectura que abre la página (arquitectura_svg()). Las
        # cifras NO van escritas: llegan por marcador y las pone mil(), que usa el separador
        # de millar del idioma. Lo que no viaja aquí es la geometría ni qué caja va resaltada,
        # que son dibujo y no texto.
        "ov_arch_io": (
            ("AWS S3", "NHANES roh · 27 XPT", "IAM"),
            ("GitHub", "7 Notebooks · README"),
            ("Streamlit Cloud", "Vorhersage · SHAP visuell"),
        ),
        "ov_arch_grupos": (
            ("Medaillon-Architektur", (
                ("Bronze · Roh-Ingestion",
                 "{bronze} Zeilen · 162 Sp. · Delta Lake ACID"),
                ("Silver · Qualität",
                 "{silver} Zeilen · 91 Sp. · Expectations"),
                ("Gold · kuratierte Features",
                 "89 Features · Train {train} / Test {test}"),
            )),
            ("Modelle · ML / QML", (
                ("LightGBM",
                 "Tabellarische Baseline · GOSS · EFB"),
                ("SVM · RBF-Kernel",
                 "Direkte Brücke zum QSVM"),
                ("QSVM · Qiskit",
                 "ZZFeatureMap · FidelityQuantumKernel"),
            )),
            ("Auswertung und Serialisierung", (
                ("Metriken",
                 "AUC-ROC · F1 · Accuracy · MCC"),
                ("SHAP",
                 "LightGBM-Erklärbarkeit · Top 20"),
                ("Auswahl · ONNX",
                 "Portabilität geprüft"),
            )),
        ),
        "ov_stats_title": "Statistik des NHANES-Datensatzes",
        "ov_stats_sub": "Drei Zweijahreszyklen integriert · Schichtenpipeline Bronze → Silver → Gold",
        "ov_stat_bronze": "Bronze-Datensätze",
        "ov_stat_silver": "Silver-Datensätze",
        "ov_stat_features": "Gold-Features",
        "ov_stat_balance": "Klassenbalance",
        "ov_medallion_title": "Medaillon-Architektur",
        "ov_medallion_sub": "Wertschöpfungskette der Daten (Curry, 2016), Schicht für Schicht angewandt",
        "ov_layers": [
            ("Bronze", "Ingestion aus AWS S3 ohne Transformation. Bewahrt die Ursprungsquelle."),
            ("Silver", "Bereinigung, Imputation, Winsorisierung und Qualitätsprüfung."),
            ("Gold",   "Skalierung, Kodierung und stratifizierte Aufteilung. Bereit zur Modellierung."),
        ],
        "ov_goto_gov": "Qualitäts- und Lineage-Kontrollen ansehen  →",
        "ov_target_title": "Verteilung der Zielvariablen (DIQ010)",
        "ov_target_sub": "Binarisiertes Ziel: 1 = diagnostizierter Diabetes, 0 = alle übrigen",
        "ov_pie_no": "Kein Diabetes",
        "ov_pie_yes": "Diabetes",
        "ov_donut_center": "14 %",
        "ov_donut_caption": "DIABETES",
        "ov_tech_title": "Gebaut auf",
        "ov_tech_sub": ("Plattform, Speicher und Bibliotheken der Pipeline in der Reihenfolge ihres "
                        "Einsatzes · das vollständige Inventar mit der Begründung jeder Wahl steht "
                        "unter Governance"),
        "ov_compare_title": "Triangulierter Vergleich: Ziel des Experiments",
        "ov_compare": [
            ("LightGBM", "Tabellarische Referenz-Baseline"),
            ("SVM-RBF",  "Strukturelle Brücke zur Quantenkomponente"),
            ("QSVM",     "FidelityQuantumKernel: derselbe Klassifikator, Quantenkernel"),
        ],

        # ── NHANES-Variablen (gemeinsam: SHAP, Schaltkreis, Bloch, Prädiktor) ──
        # Die NHANES-Codes werden nie übersetzt, nur ihre Erläuterung.
        "var_short": {
            "LBXGH": "HbA1c", "RIDAGEYR": "Alter", "LBXGLU": "Nüchternglukose",
            "LBDLDL": "LDL-Cholesterin", "BMXWAIST": "Taillenumfang",
            "WTINT2YR": "Stichprobengewicht*", "BMXARML": "Armlänge", "BMXLEG": "Beinlänge",
            "BMXBMI": "BMI", "PAD680": "Sitzende Aktivität", "PAD645": "Moderate Aktivität",
            "PAQ640": "Muskelkräftigung", "BMXWT": "Körpergewicht", "LBXIN": "Insulin",
            "INDHHIN2": "Haushaltseinkommen", "DMDYRSUS": "Jahre in den USA",
            "BMXARMC": "Armumfang", "PAQ670": "Intensive Aktivität",
            "BPXSY1": "Systolischer Druck", "PAD630": "Moderate Freizeitakt.",
            "DMDHHSZE": "Haushaltsgröße (Kinder)", "BPXDI1": "Diastolischer Druck",
            "LBXTR": "Triglyzeride", "DMDMARTL_1": "Familienstand (verheiratet)",
            "DMDMARTL_5": "Familienstand (nie verheiratet)", "BPXPLS": "Puls",
            "DMDEDUC2_3": "Bildung (Stufe 3)", "SDMVSTRA": "Stichprobenschicht",
            "DMDMARTL_2": "Familienstand (verwitwet)", "DMDHHSZB": "Haushaltsgröße (Erwachsene)",
        },
        "var_desc": {
            "LBXGH":      "Glykiertes Hämoglobin (HbA1c): durchschnittlicher Blutzucker der letzten 2-3 Monate. Primärer diagnostischer Marker für Diabetes (ADA: ≥ 6,5 %).",
            "RIDAGEYR":   "Alter der teilnehmenden Person zum Zeitpunkt der Untersuchung (Jahre).",
            "LBXGLU":     "Nüchtern-Plasmaglukose: biochemischer Marker der Blutzuckereinstellung (mg/dL).",
            "LBDLDL":     "Berechnetes LDL-Cholesterin: die mit kardiovaskulärem Risiko verbundene Cholesterinfraktion (mg/dL).",
            "BMXWAIST":   "Taillenumfang: abdominelle Adipositas, assoziiert mit Insulinresistenz (cm).",
            "WTINT2YR":   "Stichprobengewicht des NHANES-Interviews. Ein Artefakt des Erhebungsdesigns, keine klinische Variable.",
            "BMXARML":    "Oberarmlänge (Akromion → Olekranon): anthropometrisches Maß (cm).",
            "BMXLEG":     "Maximale Beinlänge (Knie → Boden): anthropometrisches Maß (cm).",
            "BMXBMI":     "Body-Mass-Index (Gewicht/Größe²): globale Körperfettmasse (kg/m²).",
            "PAD680":     "Minuten sitzender Tätigkeit pro Tag (Zeit im Sitzen oder Liegen).",
            "PAD645":     "Wöchentliche Minuten moderater körperlicher Aktivität (Arbeit + Freizeit).",
            "PAQ640":     "Tage pro Woche mit muskelkräftigenden Aktivitäten.",
            "BMXWT":      "Gesamtes Körpergewicht (kg).",
            "LBXIN":      "Nüchtern-Seruminsulin: Marker für Insulinresistenz (µU/mL).",
            "INDHHIN2":   "Einkommensniveau des Haushalts (kategoriale sozioökonomische Variable).",
            "DMDYRSUS":   "Anzahl der Jahre des Aufenthalts in den Vereinigten Staaten.",
            "BMXARMC":    "Mittlerer Oberarmumfang: anthropometrisches Maß (cm).",
            "PAQ670":     "Wöchentliche Minuten intensiver Freizeitaktivität.",
            "BPXSY1":     "Systolischer Blutdruck, erste Messung (mmHg).",
            "PAD630":     "Wöchentliche Minuten moderater körperlicher Freizeitaktivität.",
            "DMDHHSZE":   "Haushaltszusammensetzung: Anzahl der Kinder im Haushalt.",
            "BPXDI1":     "Diastolischer Blutdruck, erste Messung (mmHg).",
            "LBXTR":      "Serum-Triglyzeride: Marker des Lipidprofils (mg/dL).",
            "DMDMARTL_1": "Familienstand = verheiratet (Dummy-Variable nach One-Hot-Kodierung).",
            "DMDMARTL_5": "Familienstand = nie verheiratet (Dummy-Variable nach One-Hot-Kodierung).",
            "BPXPLS":     "Puls: Ruheherzfrequenz (Schläge/min).",
            "DMDEDUC2_3": "Mittleres Bildungsniveau (High School/GED): Dummy-Variable nach One-Hot-Kodierung.",
            "SDMVSTRA":   "Varianzschicht des NHANES-Erhebungsdesigns (methodische Variable, nicht klinisch).",
            "DMDMARTL_2": "Familienstand = verwitwet (Dummy-Variable nach One-Hot-Kodierung).",
            "DMDHHSZB":   "Haushaltszusammensetzung: Anzahl der Erwachsenen im Haushalt.",
        },
        "qsvm_labels": {
            "LBXGH": "HbA1c", "LBXGLU": "Nüchternglukose", "RIDAGEYR": "Alter",
            "LBDLDL": "LDL-Cholesterin", "BMXWAIST": "Taillenumfang", "LBXIN": "Insulin",
            "BMXLEG": "Beinlänge", "BMXBMI": "BMI",
        },
        "qsvm_units": {"años": "Jahre"},

        # ── Seite 2 · Governance ──
        "gov_eyebrow": "Governance · DataOps",
        "gov_title": "Datengovernance und Datenqualität",
        "gov_subtitle": ("Die Kontrollen, die die Pipeline tragen: was validiert wird, was verworfen wird "
                         "und warum, was protokolliert wird und mit welchen Frameworks. Alle Zahlen stammen "
                         "aus den ausgeführten Ausgaben der Notebooks."),
        "gov_tabs": ["Datenqualität", "Lineage und Nachvollziehbarkeit", "Framework-Inventar"],
        "gov_kpi_expect": "Bestandene Erwartungen",
        "gov_kpi_passrate": "Pass-Rate der Suite",
        "gov_kpi_records": "Validierte Datensätze",
        "gov_kpi_leakage": "Leakage-freie Artefakte",
        "gov_funnel_title": "Datensatz-Trichter",
        "gov_funnel_sub": ("Von den {bronze} Bronze-Datensätzen überstehen {silver} die Kohortenfilter von "
                           "Silver. Jede Stufe folgt einem ausdrücklichen Kriterium, nicht einer generischen "
                           "Bereinigung."),
        "gov_hover_records": "Datensätze",
        "gov_hover_dropped": "Verworfen",
        "gov_embudo": [
            ("Bronze · 3 Zyklen zusammengeführt",
             "27 XPT-Dateien · Join über SEQN · 162 in allen drei Zyklen gemeinsame Spalten"),
            ("Altersfilter ≥ 18 Jahre", "Beschränkung auf die erwachsene Bevölkerung"),
            ("Nüchternfilter · LBXGLU nicht null",
             "Proxy für die nüchterne Teilgruppe: PHAFSTMN ist zyklusübergreifend nicht konsistent"),
            ("Filter auf gültiges DIQ010",
             "Verwirft die Codes 7 „weiß nicht“ und 9 „Antwort verweigert“ sowie die Nullwerte"),
        ],
        "gov_dropped_title": "Pro Filter verworfene Datensätze",
        "gov_split_label": "Gold-Aufteilung 80/20",
        "gov_suite_title": "Validierungssuite · dataframe-expectations",
        "gov_suite_sub": ("Suite <code>{nombre}</code>, ausgeführt am {fecha} über die {registros} "
                          "Silver-Datensätze in {duracion} Sekunden. Great Expectations ist mit den "
                          "festgeschriebenen Versionen der Serverless-Runtime unvereinbar: dies ist die "
                          "gewählte Alternative."),
        "gov_expectativas": [
            ("Vollständigkeit", "TARGET", "höchstens 0 Nullwerte"),
            ("Vollständigkeit", "LBXGH", "höchstens 0 Nullwerte"),
            ("Vollständigkeit", "LBXGLU", "höchstens 0 Nullwerte"),
            ("Vollständigkeit", "RIDAGEYR", "höchstens 0 Nullwerte"),
            ("Vollständigkeit", "BMXBMI", "höchstens 0 Nullwerte"),
            ("Klinische Bereiche", "RIDAGEYR", "Minimum zwischen 18 und 25"),
            ("Klinische Bereiche", "RIDAGEYR", "Maximum zwischen 70 und 120"),
            ("Klinische Bereiche", "LBXGH", "Minimum zwischen 3,0 und 6,0"),
            ("Klinische Bereiche", "LBXGH", "Maximum zwischen 8,0 und 20,0"),
            ("Klinische Bereiche", "LBXGLU", "Minimum zwischen 30 und 80"),
            ("Klinische Bereiche", "LBXGLU", "Maximum zwischen 150 und 500"),
            ("Klinische Bereiche", "BMXBMI", "Minimum zwischen 10,0 und 18"),
            ("Klinische Bereiche", "BMXBMI", "Maximum zwischen 40,0 und 80"),
            ("Volumen", "DataFrame", "mindestens 7.000 Zeilen"),
            ("Volumen", "DataFrame", "höchstens 9.000 Zeilen"),
        ],
        "gov_ops_title": "Qualitätsoperationen je Schicht",
        "gov_silver_card": "Silver · Bereinigung und Aufbereitung",
        "gov_gold_card": "Gold · Vorbereitung für die Modellierung",
        "gov_silver_ops": [
            ("Wegen Leakage ausgeschlossene DIQ-Variablen", "DIQ050, DIQ070, DIQ160, DIQ170, DIQ172, DIQ180"),
            ("Entfernte dünn besetzte Spalten", "Schwelle von >80 % fehlenden Werten"),
            ("Winsorisierte Variablen", "Ausreißerkappung bei IQR × 3"),
            ("Fehlende Werte nach Imputation", "Von 75.855 auf 0 im SVM/QSVM-Datensatz (Median + Modus)"),
        ],
        "gov_gold_ops": [
            ("Features nach der Kodierung", "One-Hot von 5 kategorialen Variablen über 84 Features (106 Spalten mit TARGET)"),
            ("Wegen Korrelation verworfen", "Schwelle r > 0,90 zwischen Prädiktorenpaaren"),
            ("Endgültige Features", "Der Satz, mit dem alle drei Modelle trainiert werden"),
            ("Stratifizierte Aufteilung", "80/20 · 14,03 % Positive im Training, 14,04 % im Test"),
        ],
        "gov_eff_title": "Effektive gegenüber nominalen Features",
        "gov_eff_sub": ("Gezählt anhand von <code>scaler_correcto.json</code>: {const} der {total} Spalten "
                        "haben Varianz null und tragen keine Information für das Modell bei"),
        "gov_eff_nominal": "Nominale Features",
        "gov_eff_const": "Konstant (Varianz = 0)",
        "gov_eff_effective": "Effektive Features",
        "gov_eff_note": ("Das ist eine Nebenwirkung der IQR-×-3-Winsorisierung in Silver, die auch auf "
                         "numerisch kodierte kategoriale Variablen angewandt wurde (Antworten 1/2, "
                         "Interviewsprache, Codes 7 und 9). Wenn mehr als 75 % der Stichprobe dasselbe "
                         "antworten, lässt die Kappung die Spalte auf einen einzigen Wert zusammenfallen. "
                         "Die in Notebook 02 am stärksten gekappten (PAQ635, PAQ650, PAQ605, DMDHHSZA, "
                         "DMDCITZN, SIALANG) sind genau jene, die hier als konstant erscheinen."),
        "gov_lin_title": "Nachvollziehbarkeit ohne MLflow",
        "gov_lin_sub": "Die Einschränkung, die die Architektur der Pipeline am stärksten prägt, und ihre Abhilfe.",
        "gov_lin_limit_title": "Einschränkung",
        "gov_lin_limit_body": ("Die native <b>MLflow</b>-Integration ist im kostenlosen Databricks Serverless "
                               "deaktiviert. Jeder Aufruf von <code>mlflow.start_run()</code> oder "
                               "<code>mlflow.log_metric()</code> erzeugt Authentifizierungsfehler: es gibt "
                               "keine Protokollierung von Experimenten, Metriken oder Artefakten."),
        "gov_lin_mit_title": "Abhilfe · ein zweifacher Mechanismus",
        "gov_lin_mit_body": ("<b>Transaktionsprotokolle von Delta Lake</b>: Jeder Schreibvorgang erzeugt "
                             "einen ACID-Eintrag mit Version, Zeitstempel und Operationsmetriken.<br><br>"
                             "<b>Metriken-CSV je Modell</b>: Jedes Notebook schreibt seine Ergebnisse "
                             "dauerhaft in Unity Catalog Volumes, und die Abbildungen lesen sie von dort, "
                             "statt sie fest einkodiert mitzuführen."),
        "gov_delta_title": "Delta-Historie · Gold-Schicht",
        "gov_delta_sub": ("Die sechs jüngsten der zehn protokollierten Versionen. Delta löscht die älteren "
                          "nach 168 h Aufbewahrung: erwartetes Verhalten, kein Fehler der Pipeline."),
        "gov_delta_cols": ["Version", "Zeitstempel", "Operation", "Zeilen", "Größe"],
        "gov_chain_title": "Lückenlose Kette gegen Informationsleckagen",
        "gov_chain_sub": ("Vier verkettete Barrieren. Die dritte verwirft keine einzige Spalte, und genau "
                          "das will man sehen: der Beleg, dass die vorherigen ihre Arbeit getan haben."),
        "gov_leakage": [
            ("Ausschluss in Silver",
             "Vor der Winsorisierung werden 6 DIQ-Variablen zu Behandlung und Nachsorge entfernt: sie "
             "sind eine Folge der Diagnose, keine Prädiktoren für sie."),
            ("Kreuzprüfung",
             "Es wird geprüft, dass keine DIQ-Variable in den 2 Parquet-Dateien von Silver oder den 13 "
             "von Gold überlebt. Ergebnis: 15/15 saubere Artefakte."),
            ("Defensiver Filter der QSVM",
             "Zweite Barriere vor der Auswahl per Random Forest. Sie verwirft keine Spalte (89 von 89 "
             "passieren), eben der Beleg, dass die erste Barriere funktioniert hat."),
            ("Wächter für Stichprobengewichte",
             "Stoppt die Pipeline, sobald ein anderes als das bekannte Stichprobengewicht auftaucht. "
             "WTINT2YR erreicht die Modellierung tatsächlich und ist in Entscheidung 10 dokumentiert."),
        ],
        "gov_scaler_card": "Skalierung ohne statistisches Leck",
        "gov_scaler": [
            ("Anpassung", "Nur auf dem Training", "fit_transform im Training · transform im Test"),
            ("Ausgewertete Spalten", "66", "Mit Varianz > 0"),
            ("Konstante Spalten", "23", "Varianz 0 · siehe Entscheidung 08"),
            ("Mittelwert ≈ 0 · Std. ≈ 1", "Verifiziert", "Assert über alle Spalten mit Streuung"),
        ],
        "gov_scaler_note": ("Der <b>StandardScaler</b> wird ausschließlich auf dem <b>Training</b> angepasst: "
                            "<code>fit_transform</code> im Training und <code>transform</code> im Test. Würde "
                            "er auf dem vollständigen Datensatz angepasst, sickerten Mittelwert und "
                            "Standardabweichung des Tests in die Vorverarbeitung, und die Metriken fielen zu "
                            "optimistisch aus. Die Auswahl der 8 QSVM-Variablen folgt derselben Regel: Der "
                            "Random Forest wird nur mit <code>X_train_svm_scaled</code> trainiert.<br><br>"
                            "Der Korrelationsfilter hingegen wird <b>sehr wohl</b> vor der Aufteilung "
                            "berechnet. Das ist in Entscheidung 09 dokumentiert und bewusst hingenommen."
                            "<br><br>Die Prüfung wird ausgeführt, nicht behauptet: Über die 66 Spalten mit Streuung "
                            "werden |Mittelwert| &lt; 0,01 und eine Abweichung zwischen 0,90 und 1,10 verlangt. "
                            "Die angepassten Parameter (<code>mean_</code> und <code>scale_</code>) gehen nach "
                            "<code>scaler_correcto.json</code>, die Datei, die der Live-Prädiktor lädt: Die Skala "
                            "des Trainings wird nie neu berechnet."),
        "gov_e2e_title": "End-to-End-Prüfung gegen die trainierten Modelle",
        "gov_e2e_missing": ('<b style="color:{color};">Nicht verifiziert.</b> Der Testdatensatz liegt nicht im '
                            "Repository, deshalb kann das Dashboard nicht von sich aus prüfen, ob sein "
                            "Inferenzpfad das reproduziert, was die trainierten Modelle geliefert haben. Um "
                            "das zu schließen, führe die beiden Zellen aus "
                            "<code>notebooks/INSTRUCCIONES_exportar_golden_set.md</code> aus und kopiere "
                            "<code>golden_lgbm.npz</code> und <code>golden_svm.npz</code> nach "
                            "<code>streamlit/models/</code>. Solange sie fehlen, behauptet diese Seite nichts, "
                            "was sie nicht hat prüfen können."),
        "gov_e2e_unavailable": "nicht verfügbar",
        "gov_e2e_ok_val": "{n} Zeilen · max. Abw. {dif}",
        "gov_e2e_bad_val": "ABWEICHUNG · max. Abw. {dif}",
        "gov_e2e_scaled": "skaliert und ruft das ONNX-Modell auf",
        "gov_e2e_raw": "ruft das ONNX-Modell ohne Skalierung auf",
        "gov_e2e_path": "Das Dashboard {accion}",
        "gov_e2e_ok_title": "✓ Inferenzpfad verifiziert",
        "gov_e2e_fail_title": "⚠ Der Inferenzpfad reproduziert die Modelle nicht",
        "gov_e2e_note": ("Jede Zeile des <i>Golden Set</i> ist eine echte Testinstanz zusammen mit der "
                         "Wahrscheinlichkeit, die das in seinem Notebook trainierte Modell zurückgegeben hat. "
                         "Das Dashboard schickt sie durch den eigenen Pfad (Rohvektor, Skalierung nur für die "
                         "SVM, Umwandlung nach <code>float32</code>, ONNX-Sitzung und Auslesen des "
                         "Ausgabetensors) und vergleicht. Toleranz {tol}; das durch die Arbeit in "
                         "<code>float32</code> erwartete Rauschen liegt in der Größenordnung 10⁻⁷."),
        "gov_stack_title": "Frameworks je Schicht",
        "gov_stack_sub": ("Das erste Abzeichen jeder Karte ist das Framework, das die Schicht trägt; die "
                          "übrigen begleiten es."),
        "gov_stack": [
            ("Ingestion",
             "boto3 ersetzt spark.conf, das unter Serverless blockiert ist (Entscheidung 01). Drei "
             "Integritäts-Asserts: 27/27 Dateien, der Join über SEQN dupliziert keine Zeilen, und "
             "Delta stimmt mit pandas überein."),
            ("Qualität",
             "Das Qualitätsframework dieser Masterarbeit. Great Expectations ist mit der Umgebung "
             "unvereinbar (Entscheidung 03). Eine Suite aus 15 Erwartungen in 3 Dimensionen, mit "
             "in CSV gesicherter Evidenz."),
            ("Vorbereitung",
             "Skalierung nur auf dem Training angepasst, stratifizierte Aufteilung mit festem "
             "Startwert und Export des Serving-Vertrags (Scaler und Mediane in JSON)."),
            ("Modell",
             "Exakte Interpretierbarkeit über den polynomiellen Algorithmus auf den 1.567 "
             "Testinstanzen, und Nachweis, dass das ONNX-Modell das PKL zu 100 % reproduziert."),
            ("Modell",
             "Modellagnostisches SHAP, um den Preis von Stunden: einmal über 200 Instanzen "
             "berechnet und zur Wiederverwendung auf der Platte gesichert."),
            ("Modell",
             "Keine ONNX-Unterstützung: das Format lässt keine Quantenoperationen zu "
             "(Entscheidung 05). Die Nachvollziehbarkeit trägt ein Metriken-CSV mit den 14 "
             "Konfigurationsfeldern."),
        ],
        "gov_dec_title": "Entscheidungsprotokoll",
        "gov_dec_sub": ("Die elf in TECHNICAL_NOTES dokumentierten Einschränkungen samt Abhilfe. Drei prägen "
                        "die Architektur, sechs werden hingenommen und dokumentiert, ohne sie zu korrigieren "
                        "(weil das die bereits erzielten Ergebnisse entwerten würde), und zwei sind "
                        "rückstandslos gelöst."),
        "gov_dec_tags": {"critical": "Architektur", "warning": "Hingenommen", "good": "Gelöst"},
        "gov_dec_problem": "Problem · ",
        "gov_dec_solution": "Gewählte Lösung · ",
        "gov_decisiones": [
            ("spark.conf unter Serverless blockiert",
             "Die Konfiguration der AWS-Zugangsdaten über spark.conf.set, den Standardweg, um Spark "
             "mit S3 zu verbinden, liefert CONFIG_NOT_AVAILABLE.",
             "boto3 als alternativer Client. S3 bleibt der Ursprungsspeicher und Unity Catalog Volumes "
             "die Verarbeitungsschicht."),
            ("MLflow unter Serverless blockiert",
             "Die native MLflow-Integration ist in der kostenlosen Stufe deaktiviert: es gibt keine "
             "Protokollierung von Experimenten, Metriken oder Artefakten.",
             "Ein zweifacher Ersatz: die Transaktionsprotokolle von Delta Lake liefern Version, "
             "Zeitstempel und Operationsmetriken; und jedes Notebook sichert seine Metriken als CSV."),
            ("Great Expectations unvereinbar",
             "Es verlangt eine pandas/numpy-Kombination, die mit den festgeschriebenen Versionen der "
             "Serverless-Runtime kollidiert (pandas 1.5.3 / numpy 1.23.5).",
             "dataframe-expectations 0.7.0 als kompatible Alternative. 15 Erwartungen an Silver in drei "
             "Dimensionen. Ergebnis 15/15, Pass-Rate 1,0."),
            ("QSVM · Rechenaufwand O(n²)",
             "Über die 6.264 Trainingsinstanzen verlangte die Kernelmatrix rund 39 Millionen "
             "Auswertungen des Schaltkreises. Bei 1.500 erschöpft der Kernel den Speicher.",
             "Training auf einer stratifizierten Stichprobe von 500 Instanzen (~22 min) unter Wahrung "
             "des Verhältnisses 86/14. Die Auswertung nutzt sehr wohl den vollständigen Testsatz, damit "
             "die Metriken vergleichbar bleiben."),
            ("QSVM · keine native ONNX-Unterstützung",
             "Das ONNX-Format lässt keine Quantenoperationen zu: weder skl2onnx noch onnxmltools können "
             "einen auf Zustandssimulation beruhenden Kernel serialisieren.",
             "Serialisierung mit joblib. Das Modell benötigt für die Inferenz die Qiskit-Umgebung, "
             "weshalb die QSVM nicht Teil des Live-Prädiktors ist."),
            ("Qiskit-Versionen nicht festschreibbar",
             "Die immutable_package_constraints.txt von Databricks blockiert die Installation "
             "bestimmter Versionen, es gibt also keine exakte Versionsreproduzierbarkeit.",
             "Die Pipeline läuft mit den Versionen der Umgebung (2.5.0 / 0.9.0 / 0.4.0), deren API "
             "kompatibel ist; sie werden durch eine ausdrückliche Prüfung zu Beginn des Laufs "
             "protokolliert."),
            ("Variablenverlust durch die Sitzungsdauer",
             "Lange Operationen (22 min Training, 132 min Vorhersage) können die Serverless-Sitzung "
             "erschöpfen und die Variablen im Arbeitsspeicher mitnehmen.",
             "Sofortige Sicherung nach jeder teuren Operation und ein TRAINING_MODE, der bei späteren "
             "Läufen von der Platte nachlädt."),
            ("Winsorisierung auf kodierte kategoriale Variablen angewandt",
             "NHANES kodiert viele kategoriale Variablen numerisch. Teilen sich mehr als 75 % einen "
             "Wert, ist IQR = 0, die Grenzen fallen zusammen und clip() macht die Variable zur "
             "Konstanten. 10 Spalten sind so kollabiert.",
             "Dokumentiert, ohne es zu ändern: eine Korrektur würde Silver, Gold und alle drei Modelle "
             "verändern. Konstante Spalten verzerren nichts (das Modell zieht kein Signal aus ihnen), "
             "aber Information geht verloren. Die Korrektur ist als künftige Arbeit vermerkt."),
            ("Korrelation vor der Aufteilung berechnet",
             "Der Filter r > 0,90 wird auf dem vollständigen Datensatz berechnet, die 16 verworfenen "
             "Spalten werden also auch anhand der Testbeobachtungen bestimmt.",
             "Dokumentiert, ohne es zu ändern. Es betrifft weder die Skalierung noch die Feature-Auswahl "
             "der QSVM (beide nur auf dem Training angepasst), aber die Auswahl ist nicht mehr streng "
             "blind gegenüber dem Test."),
            ("Stichprobengewicht WTINT2YR unter den Features",
             "Der Join innerhalb des Zyklus dupliziert WTSAF2YR über drei Spalten. WTINT2YR steht nicht "
             "auf der Ausschlussliste und übersteht den Korrelationsfilter: es ist eines der 89 Features.",
             "Dokumentiert, ohne es zu ändern, plus ein Assert, der das Auftauchen JEDES anderen Gewichts "
             "erkennt. Ein Stichprobengewicht ist keine klinische Variable: es verrät das Ziel nicht, "
             "lässt das Modell aber auf dem Erhebungsdesign aufsetzen."),
            ("Die serialisierte QSVM ist versionsübergreifend nicht ladbar",
             "Der Pickle schleppt die ZZFeatureMap mitsamt ihren ParameterExpression mit. Wechselt "
             "Qiskit die Version, scheitert die Deserialisierung, und Serverless aktualisiert ohne "
             "Vorwarnung.",
             "Das Laden ist in try/except gefasst: schlägt es fehl, springt TRAINING_MODE auf True und "
             "das Notebook trainiert neu, statt abzubrechen. Es bleibt in allen drei Szenarien "
             "einsatzfähig."),
        ],
        "gov_footer_note": ("Die Zahlen dieser Seite stammen aus den ausgeführten Ausgaben der Notebooks des "
                            "Repositoriums und aus <code>TECHNICAL_NOTES.md</code>; keine ist geschätzt. Die "
                            "Anwendung kann sie nicht live abfragen, weil Streamlit Community Cloud nur auf "
                            "das Repositorium zugreift, nicht auf Unity Catalog Volumes.<br><br>"
                            "Zusammenfassung der Qualitätssuite: <b>{fuente}</b>."),
        "gov_suite_src_csv": "gelesen aus validacion_silver_dfe.csv",
        "gov_suite_src_nb": "aus dem Notebook verifizierte Werte",

        # ── Seite 3 · Ergebnisse ──
        "res_eyebrow": "Triangulierter Vergleich",
        "res_title": "Ergebnisse",
        "res_subtitle": "LightGBM vs. SVM-RBF vs. QSVM auf demselben Testsatz ({n} Instanzen).",
        "res_threshold": "Schwelle",
        "res_thr_label": {"lightgbm": "p ≥ {v}", "svm_rbf": "p ≈ {v}", "qsvm": "df > 0"},
        "res_thr_src": {"lightgbm": "predict_proba()[:,1] >= 0.5",
                        "svm_rbf": "SVC.predict() · Vorzeichen von decision_function",
                        "qsvm": "decision_function > 0 (keine Wahrscheinlichkeit)"},
        "res_reconciled": ('<span style="color:{color}; font-weight:600;">✓ Abgeglichen</span>: die vier '
                           "Metriken aller drei Modelle wurden aus den Scores je Instanz neu berechnet und "
                           "stimmen mit den veröffentlichten überein."),
        "res_unreconciled": '<span style="color:{color}; font-weight:600;">⚠ Nicht abgeglichen</span>: {fallos}',
        "res_no_scores": "Scores nicht verfügbar",
        "res_threshold_note": ("<b>Die drei Modelle sind an unterschiedlichen Schwellen gemessen.</b> Jedes "
                               "nutzt seinen natürlichen Schnittpunkt: LightGBM "
                               "<code>predict_proba ≥ 0,50</code>; SVM-RBF das Vorzeichen von "
                               "<code>decision_function</code>, was auf der gespeicherten "
                               "Wahrscheinlichkeitsskala ≈ 0,22 entspricht; QSVM "
                               "<code>decision_function &gt; 0</code>, was keine Wahrscheinlichkeit ist. "
                               "Jede Matrix reproduziert exakt an ihrer eigenen Schwelle, aber <b>nur der "
                               "AUC-ROC ist zwischen Modellen vergleichbar</b>: er ist die einzige der vier "
                               "Metriken, die vom Schnittpunkt unabhängig ist. Zur Einordnung: die SVM-RBF, "
                               "bei 0,50 wie LightGBM ausgewertet, ergäbe Accuracy 0,9190, aber nur 131 "
                               "richtig Positive statt 172."),
        "res_roc_title": "ROC-Kurven",
        "res_roc_sub_real": ("Echte empirische Kurven, Punkt für Punkt über die 1.567 Testinstanzen "
                             "(dieselben Scores, die den AUC der Masterarbeit ergeben)."),
        "res_roc_sub_synth": ("Exakter AUC · Form aus dem AUC rekonstruiert, wo Scores je Instanz fehlen."),
        "res_cm_title": "Konfusionsmatrizen",
        "res_cm_sub": ("Werte gegen den Classification Report jedes Modells geprüft und aus den Scores je "
                       "Instanz neu berechnet. Jede Matrix entspricht der auf ihrer Karte genannten Schwelle"),
        "res_cm_pred_no": "Vorhers.<br>Kein Diabetes",
        "res_cm_pred_yes": "Vorhers.<br>Diabetes",
        "res_cm_real_no": "Echt<br>Kein Diab.",
        "res_cm_real_yes": "Echt<br>Diabetes",
        "res_cm_tags": {"tn": "RN", "fp": "FP", "fn": "FN", "tp": "RP"},
        "res_metrics_title": "Metrikvergleich",
        "res_metrics_sub": ("Die vier Metriken werden über die 1.567 Instanzen berechnet. Accuracy, MCC und "
                            "F1-Macro bestrafen das Klassenungleichgewicht sehr wohl, aber sie hängen von "
                            "der Schwelle ab, und jedes Modell nutzt seine eigene: vergleiche alles außer "
                            "dem AUC-ROC mit Vorsicht"),
        "res_metric_desc": {
            "auc": "Fläche unter der ROC-Kurve: Fähigkeit, Diabetes von Nicht-Diabetes zu trennen. 0,5 = Zufall, 1 = perfekt.",
            "f1_macro": "Harmonisches Mittel aus Precision und Recall, je Klasse gemittelt (ungewichtet). Bestraft das Ungleichgewicht.",
            "accuracy": "Anteil der insgesamt richtigen Vorhersagen. Bei unausgewogenen Klassen kann er nur die Mehrheitsklasse widerspiegeln.",
            "mcc": "Matthews-Korrelationskoeffizient: Gesamtgüte, robust gegen Ungleichgewicht. 0 = Zufall, 1 = perfekt.",
        },
        "res_qsvm_note": ("<b>Anmerkung zum QSVM-Experiment.</b> Die QSVM wurde auf einer stratifizierten "
                          "Stichprobe von 500 Instanzen trainiert (Aufwand O(n²) des Quantenkernels) und auf "
                          "den vollständigen 1.567 des Tests ausgewertet. AUC-ROC = 0,5493 zeigt, dass das "
                          "Modell die zufällige Klassifikation kaum übertrifft: Recall ≈ 0 für die Klasse "
                          "Diabetes (1 von 220), Accuracy = 0,8602 spiegelt nur den Anteil der "
                          "Mehrheitsklasse. Der MCC ≈ 0 bestätigt das Fehlen echter Vorhersagekraft."),

        # ── Seite 4 · SHAP-Analyse ──
        "sh_eyebrow": "Interpretierbarkeit",
        "sh_title": "SHAP-Analyse",
        "sh_subtitle": ("Globale Variablenwichtigkeit: TreeExplainer (LightGBM) vs. "
                        "KernelExplainer (SVM-RBF)."),
        "sh_tabs": ["LightGBM · TreeExplainer", "SVM-RBF · KernelExplainer"],
        "sh_hint": "Fahre über einen Balken, um die Bedeutung der Variablen zu sehen. {nota}",
        "sh_sample_lgbm": "Exakte Werte (polynomieller Algorithmus) über die 1.567 Testinstanzen.",
        "sh_sample_svm": ("Durch Stichproben genäherte Werte: Hintergrund aus 100 Instanzen, Beiträge über "
                          "200 Testinstanzen."),
        "sh_note_lgbm": ("<b>LBXGH (HbA1c)</b> dominiert mit großem Abstand (mittlerer SHAP = 1,1243), "
                         "stimmig mit seiner Rolle als primärer diagnostischer Marker für Typ-2-Diabetes "
                         "(ADA: HbA1c ≥ 6,5 %). <b>RIDAGEYR (Alter, 0,4654)</b> spiegelt den Anstieg der "
                         "Prävalenz mit dem Alter. <b>LBXGLU</b> und <b>LBDLDL</b> vervollständigen den "
                         "biochemischen Block. <b>WTINT2YR</b> (Position 6) ist ein Artefakt des "
                         "NHANES-Erhebungsdesigns, keine klinische Variable."),
        "sh_note_svm": ("Das Ranking der SVM-RBF stimmt bei den dominanten Variablen mit LightGBM überein "
                        "(<b>LBXGH</b>, <b>LBXGLU</b>, <b>LBDLDL</b>, <b>RIDAGEYR</b>), was die klinische "
                        "Gültigkeit des Befunds stärkt, weil er damit vom Algorithmus unabhängig und "
                        "methodisch robuster wird. KernelExplainer behandelt das Modell als Blackbox und "
                        "ist deshalb auf jeden Klassifikator anwendbar."),
        "sh_fig_lgbm_title": "SHAP Summary Plot · LightGBM (Abbildung 27)",
        "sh_fig_lgbm_cap": ("Jeder Punkt ist eine Testinstanz; die Farbe zeigt den Wert der Variablen (rot "
                            "hoch, blau niedrig) und die waagerechte Lage ihren Einfluss auf die Vorhersage. "
                            "LBXGH und RIDAGEYR dominieren das Modell."),
        "sh_fig_svm_title": "SHAP Summary Plot · SVM-RBF (Abbildung 31)",
        "sh_fig_svm_cap": ("Jeder Punkt ist eine Instanz; Farbe = Wert der Variablen, Lage = Einfluss. "
                           "KernelExplainer über 200 Testinstanzen."),

        # ── Seite 5 · Quantenschaltkreis ──
        "qc_eyebrow": "Quantenkomponente",
        "qc_title": "Quantenschaltkreis",
        "qc_subtitle": ("Konfiguration der ZZFeatureMap und des FidelityQuantumKernel, implementiert in "
                        "Qiskit auf Databricks CE."),
        "qc_tabs": ["ZZFeatureMap-Schaltkreis", "Bloch-Kugel"],
        "qc_specs": ["Qubits (feature_dimension)", "Wiederholungen (reps)", "Entanglement", "Qiskit-Version"],
        "qc_how_title": "So funktioniert es",
        "qc_how_p1": ("Die <b>ZZFeatureMap</b> kodiert jede der 8 klinischen Variablen als Phasenwinkel "
                      "(P-Gatter) auf einem eigenen Qubit, nachdem sie mit Hadamard-Gattern Superposition "
                      "erzeugt hat. Ihr Unterscheidungsmerkmal ist die <b>Verschränkung</b> zwischen "
                      "Qubit-Paaren über Gatter, die vom Kreuzprodukt zweier Variablen abhängen, "
                      "Korrelationen, die der klassische RBF-Kernel nicht darstellen kann."),
        "qc_how_p2": ("Der <b>FidelityQuantumKernel</b> misst die Ähnlichkeit zweier Patienten als die "
                      "Fidelity zwischen ihren Quantenzuständen: <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. Die "
                      "Implementierung nutzt <code>StatevectorSampler</code> und simuliert den exakten "
                      "Zustand ohne Rauschen: deterministische, reproduzierbare Ergebnisse."),
        "qc_feat_title": "8 ausgewählte Features (Random Forest)",
        "qc_xaxis": "RF-Wichtigkeit",
        "qc_train_title": "Training und Auswertung",
        "qc_tstats": ["Trainingsinstanzen", "Trainingszeit", "Testinstanzen",
                      "Inferenzzeit", "Stützvektoren"],
        "qc_note": ("Wegen des Aufwands O(n²) des Quantenkernels blieb das Training auf eine stratifizierte "
                    "Stichprobe von 500 Instanzen beschränkt (die praktische Grenze von Databricks CE "
                    "serverless liegt bei etwa 500-1.000). Die Auswertung erfolgte auf dem vollständigen "
                    "Testsatz (1.567 Instanzen) in Losen von 100, mit einer Gesamtvorhersagezeit von "
                    "144,5 Minuten."),
        "qc_circuit_title": "Vollständiger Quantenschaltkreis (8 Qubits)",
        "qc_circuit_sub": ("ZZFeatureMap mit reps=2: Kodierung (H + P), gefolgt von zwei Runden linearer "
                           "Verschränkung zwischen benachbarten Qubits."),

        # ── Seite 5 · Quantenschaltkreis → Tab Bloch-Kugel ──
        "bl_title": "Bloch-Kugel",
        "bl_subtitle": "Wie die ZZFeatureMap den Wert einer klinischen Variablen als Quantenzustand |ψ⟩ kodiert.",
        "bl_what_note": ("<b>Was die Bloch-Kugel ist.</b> Ein klassisches Bit kann nur 0 oder 1 "
                         "sein. Ein Qubit lässt zusätzlich jede Mischung aus beidem zu, und diese "
                         "Mischung passt nicht in eine einzige Zahl: es braucht eine Karte. Die "
                         "Bloch-Kugel ist diese Karte: Jeder mögliche Zustand eines Qubits ist ein "
                         "Punkt auf der Oberfläche einer Kugel mit Radius 1. Der Nordpol ist "
                         "<b>|0⟩</b> und der Südpol <b>|1⟩</b>; dazwischen liegen die "
                         "Superpositionen, und je näher der Pfeil an einem Pol steht, desto "
                         "wahrscheinlicher ist dieses Messergebnis. Hier wird der klinische Wert in "
                         "den Winkel θ übersetzt, sodass der Schieberegler den Pfeil entlang eines "
                         "Meridians von |0⟩ nach |1⟩ dreht."),
        "bl_var": "Klinische Variable",
        "bl_value": "Wert ({unidad})",
        "bl_xnorm": "normiertes x",
        "bl_theta": "θ = x_norm·π",
        "bl_alpha": "α (Amplitude |0⟩)",
        "bl_beta": "β (Amplitude |1⟩)",
        "bl_rad": "rad",
        "bl_note": ("<b>Eine didaktische Analogie des Prinzips der Winkelkodierung</b>, keine Nachbildung "
                    "des Schaltkreises. Hier wird der auf [0,1] normierte klinische Wert zum "
                    "<b>polaren</b> Winkel θ = x_norm·π, sodass der Vektor den Meridian von |0⟩ nach |1⟩ "
                    "durchläuft und P(|0⟩) von 100 % auf 0 % fällt: die anschaulichste Art zu sehen, wie "
                    "„eine Zahl zu einem Zustand wird“.<br><br>"
                    "Die <b>echte ZZFeatureMap</b> tut etwas anderes: sie wendet H an und danach "
                    "P(2·x<sub>i</sub>), und ein Phasengatter nach einer Hadamard lässt den Zustand "
                    "<b>auf dem Äquator</b> (θ = π/2 fest, P(|0⟩) = P(|1⟩) = 50 % immer) und kodiert "
                    "den Wert im <b>azimutalen</b> Winkel φ, nicht im polaren. Sie normiert auch nicht "
                    "auf [0,1], sondern nutzt den skalierten Wert direkt. Deshalb veranschaulicht diese "
                    "Kugel das Konzept, bildet den Schaltkreis aber nicht Schritt für Schritt ab. Die "
                    "Verschränkung (Gatter P(2·(π−x<sub>i</sub>)·(π−x<sub>j</sub>))) ist nur im "
                    "gemeinsamen Raum der 8 Qubits darstellbar (siehe Quantenschaltkreis)."),

        # ── Seite 5 · Tab Bloch-Kugel → Abschnitt Verschränkung ──
        "bl_ent_title": "Verschränkung: drei Qubits, ein einziger Zustand",
        "bl_ent_sub": ("Die Grenze der Kugel von oben. Wende die drei Gatter an und sieh, was mit dem "
                       "lokalen Zustand jedes Qubits geschieht, und mit dem des Paares, das übrig bleibt."),
        "bl_ent_intro": ("<b>Wo die Bloch-Kugel aufhört zu taugen.</b> Für ein Qubit genügen eine Kugel "
                         "und ein Pfeil. Bei mehreren liegt die Versuchung nahe, eine Kugel je Qubit zu "
                         "zeichnen, und für die meisten Zustände geht das auf. Aber es gibt eine Familie "
                         "von Zuständen, in der <b>kein Pfeil mehr zu zeichnen bleibt</b>: das Ganze hat "
                         "einen vollkommen bestimmten Zustand, und keines seiner Mitglieder hat einen für "
                         "sich. Das ist die Verschränkung, und hier entsteht sie aus drei Gattern. Wende "
                         "sie an und verfolge die vier Zahlen links: die ersten drei brechen im zweiten "
                         "Schritt ein, und im dritten zeigt die vierte etwas, das sich mit zwei Qubits "
                         "nicht einmal stellen lässt."),
        "bl_ent_btn_h": "1 · Hadamard auf q₀",
        "bl_ent_btn_cnot1": "2 · CNOT (Steuerung q₀ → q₁)",
        "bl_ent_btn_cnot2": "3 · CNOT (Steuerung q₁ → q₂)",
        "bl_ent_btn_reset": "Zurücksetzen auf |000⟩",
        "bl_ent_step_note": [
            ("<b>Ausgangspunkt.</b> Drei Qubits, alle in |0⟩, kein Gatter angewandt. Der gemeinsame "
             "Zustand ist |000⟩ und hat noch nichts Quantenhaftes: er entspricht genau drei klassischen "
             "Bits auf null. In der Q-Sphere gibt es einen einzigen Knoten am Nordpol, der die gesamte "
             "Wahrscheinlichkeit trägt."),
            ("<b>Superposition, noch nicht verschränkt.</b> Die Hadamard lässt q₀ auf halbem Weg "
             "zwischen |0⟩ und |1⟩, während q₁ und q₂ fest in |0⟩ bleiben: der gemeinsame Zustand ist "
             "(|000⟩ + |100⟩)/√2. Die drei Qubits sind weiterhin <b>unabhängig</b>: Jedes hat seinen "
             "eigenen reinen Zustand, und drei Bloch-Kugeln würden sie vollständig beschreiben. Beachte, "
             "dass die Länge des lokalen Vektors weiterhin 1 beträgt: es gibt einen Pfeil zu zeichnen."),
            ("<b>Ein Bell-Paar, und ein Zuschauer.</b> Das erste CNOT kippt q₁ nur dann, wenn q₀ gleich 1 "
             "ist; auf eine Superposition angewandt, bindet das beide Ergebnisse zu einem: "
             "(|000⟩ + |110⟩)/√2. Hier zerbricht die Karte: Die Länge des lokalen Vektors von q₀ ist "
             "gerade auf <b>0</b> gefallen, das Qubit liegt an keinem Punkt seiner Kugel mehr, weil es für "
             "sich genommen <b>keinen Zustand mehr hat</b>. Und es geschieht noch etwas, das nur mit einem "
             "dritten Qubit sichtbar wird: q₂ ist AUSSEN vor geblieben und schaut aus |0⟩ zu, und "
             "verschränkt ist genau das Paar q₀q₁. Seine Konkurrenz steht auf <b>1</b>, dem Maximum."),
            ("<b>GHZ-Zustand.</b> Das zweite CNOT hängt q₂ an die Kette: (|000⟩ + |111⟩)/√2. Die Knoten "
             "sind zu den Polen gewandert und die beiden mittleren Ringe sind leer. Die ersten drei Zahlen "
             "rühren sich nicht (q₀ hat weiterhin keinen eigenen Zustand), aber die vierte bricht ein: "
             "die Konkurrenz des Paares q₀q₁ steht wieder auf <b>0</b>. Alle drei zu verschränken hat das "
             "Band des Paares AUFGELÖST. Die beiden bleiben korreliert (misst man eines, sagt das andere "
             "voraus), aber nicht mehr verschränkt: die Verschränkung eines GHZ gehört dem Ganzen und ist "
             "<b>nicht die Summe paarweiser Bänder</b>."),
        ],
        "bl_ent_circuit_title": "Schaltkreis",
        "bl_ent_circuit_alt": "Drei-Qubit-Schaltkreis mit den bisher angewandten Gattern",
        "bl_ent_qsphere_title": "Q-Sphere des gemeinsamen Zustands",
        "bl_ent_kpi": ["Länge des lokalen Vektors |r| (q₀)", "Reinheit Tr(ρ₀²)",
                       "Verschränkungsentropie", "Konkurrenz des Paares q₀q₁"],
        "bl_ent_bits": "Bit",
        "bl_ent_hover_amp": "Amplitude:",
        "bl_ent_hover_prob": "Wahrscheinlichkeit:",
        "bl_ent_hover_shots": "Messungen",
        "bl_ent_meas_title": "Messung",
        "bl_ent_meas_sub": ("Die Q-Sphere zeigt den Zustand; dies zeigt das Einzige, was sich beobachten "
                            "lässt. Wiederhole den Durchgang: der Anteil ist stabil, die genaue Anzahl nicht."),
        "bl_ent_meas_n": "Anzahl der Messungen",
        "bl_ent_meas_btn": "Messungen simulieren",
        "bl_ent_meas_empty": "Wähle die Anzahl der Messungen und drücke „Messungen simulieren“.",
        "bl_ent_meas_yaxis": "Wie oft erhalten",
        "bl_ent_meas_note": [
            ("Mit allen drei Qubits in |0⟩ lautet das Ergebnis bei jedem Schuss <b>000</b>. Es gibt noch "
             "nichts auszulosen: so verhalten sich drei klassische Bits."),
            ("<b>000</b> und <b>100</b> kommen zu gleichen Teilen heraus: q₀ verhält sich wie ein "
             "Münzwurf, und q₁ und q₂ sind 0, was auch geschieht. Die Ergebnisse sind <b>unabhängig</b>: "
             "Eines zu kennen sagt nichts über die anderen."),
            ("Heraus kommen <b>000</b> und <b>110</b>: q₀ und q₁ liefern immer denselben Wert, eines zu "
             "messen legt also das andere fest, während q₂ auf 0 festgenagelt bleibt und von nichts "
             "erfährt. Zwei der drei sind schon aneinander gebunden; das dritte schaut noch von außen zu."),
            ("<b>Es kommen nur 000 und 111 heraus</b>, jeweils nahe 50 %. Die sechs leeren Balken sind "
             "der Befund: <b>keine der übrigen sechs Kombinationen kommt je vor</b>, kein einziges Mal in "
             "zehntausend Schüssen. Jedes Qubit liefert weiterhin ein zufälliges Ergebnis, aber alle drei "
             "liefern <b>immer dasselbe</b>: eines zu messen legt die anderen beiden augenblicklich fest. "
             "Diese perfekte Korrelation ist die Verschränkung, vom Labor aus gesehen."),
        ],
        "bl_ent_impl_note": ("<b>Wie das berechnet ist.</b> Die acht Amplituden stammen aus <b>exakter</b> "
                             "linearer Algebra in NumPy (der Matrix H⊗I⊗I und den beiden CNOT, angewandt "
                             "auf |000⟩), nicht aus einer Näherung; die Konkurrenz des Paares ist die "
                             "Formel von Wootters, die für zwei Qubits den exakten Wert liefert und keine "
                             "Schranke; und die Messungen stammen aus einer multinomialen "
                             "Ziehung über |ψ|², also dem, was ein idealer rauschfreier Simulator tut. Das "
                             "Panel <b>lädt kein Qiskit</b>: die ausgelieferte Umgebung besteht aus "
                             "Streamlit, NumPy, Plotly und ONNX Runtime, während Qiskit in der "
                             "Databricks-Pipeline lebt (dort wird die QSVM trainiert) und seine "
                             "Abbildungen hier bereits gerendert ankommen, wie der 8-Qubit-Schaltkreis auf "
                             "der Seite Quantenschaltkreis. Die Basiskonvention ist die des Lehrbuchs, "
                             "|q₀q₁q₂⟩ mit q₀ links; Qiskit nummeriert umgekehrt und schriebe „001“, wo der "
                             "erste Schritt hier „100“ schreibt."),

        # ── Seite 5 · Tab Bloch-Kugel → die echte ZZFeatureMap (8 Qubits) ──
        "bl_zz_title": "Die echte ZZFeatureMap: wo die Verschränkung stattfindet",
        "bl_zz_sub": ("Dieselben Zahlen, nun über die 8 Qubits der QSVM dieser Masterarbeit. Bewege den "
                      "Schieberegler am Seitenanfang und sieh, welche Qubits reagieren."),
        "bl_zz_intro": ("<b>Von drei Qubits zu den acht des Modells.</b> Jedes Qubit der ZZFeatureMap "
                        "trägt <b>eine klinische Variable</b>: q₀ ist der HbA1c, q₁ die Glukose, und so "
                        "weiter bis zum BMI. Bei 256 Amplituden gibt es kein Bild des Zustands mehr, das "
                        "man anschauen könnte (weder Q-Sphere noch Histogramm), aber es lassen sich "
                        "weiterhin die <b>gleichen beiden Größen</b> aus dem vorigen Abschnitt messen: "
                        "wie viel eigenen Zustand jedes Qubit behält und wie viel Information es mit "
                        "jedem anderen teilt. Das ist kein Lehrbuchbeispiel mehr: es ist der Schaltkreis, "
                        "mit dem das Modell trainiert wurde."),
        "bl_zz_current": "Variable im Spiel: <b>{var} = {val} {unidad}</b>. Die anderen sieben auf ihrem Referenzwert.",
        "bl_zz_r_title": "Eigener Zustand jedes Qubits",
        "bl_zz_r_xaxis": "|r|: 1 = behält seinen Zustand · 0 = vollständig verschränkt",
        "bl_zz_mi_title": "Wechselseitige Information zwischen Qubits",
        "bl_zz_mi_cbar": "Bit",
        "bl_zz_note": ("<b>Wie die Matrix zu lesen ist.</b> Jede Zelle sagt, wie viel Information zwei "
                       "Qubits teilen: je heller, desto enger sind sie aneinander gebunden. Und es "
                       "springt ins Auge, dass sich die Farbe <b>in einem Band entlang der Diagonalen "
                       "sammelt</b> und die Ecken leer bleiben. Das ist kein Zufall: die ZZFeatureMap "
                       "nutzt <code>entanglement=\"linear\"</code>, es gibt also <b>nur Gatter zwischen "
                       "benachbarten Qubits</b>. Mit reps=2 reicht diese Korrelation höchstens vier "
                       "Glieder weit; darüber hinaus ist sie <b>exakt null</b> (an 300 Profilen geprüft: "
                       "ab Abstand ≥ 5 in der Kette 0,0000 Bit, ohne eine einzige Ausnahme). Die "
                       "Topologie des Schaltkreises zeichnet sich von selbst.<br><br>"
                       "Und es gibt eine zweite Sache, die man sieht, wenn man den Schieberegler oben "
                       "bewegt: ändert man <b>eine</b> Variable, bewegen sich nur <b>ihr Qubit und "
                       "dessen unmittelbare Nachbarn</b>: Die übrigen ändern sich nicht um eine "
                       "Nachkommastelle. Es ist derselbe Sachverhalt von der anderen Seite: der "
                       "Lichtkegel des Schaltkreises, live."),
        "bl_zz_caveat": ("<b>Ein Lesehinweis, und kein kleiner.</b> Die Verschränkung <b>wächst nicht mit "
                         "dem klinischen Wert</b>: erhöht man den HbA1c über seinen Bereich, geht |r| des "
                         "ersten Qubits 0,88 → 1,00 → 0,65 → 0,24 → 0,73 → 0,98 → 0,33. Rauf und runter. "
                         "Der Grund ist, dass der Wert als Phasen<b>winkel</b> eingeht und Winkel "
                         "<b>umlaufen</b>: zwei sehr verschiedene klinische Werte können bei fast "
                         "gleichen Phasen landen. Bei standardisierten Features erreicht ein Extremfall "
                         "x ≈ 5, und das Produkt 2·(π−xᵢ)(π−xⱼ) des Verschränkungsterms überschreitet 2π "
                         "mehrfach. Das ist eine bekannte Grenze der unbeschränkten Winkelkodierung, und "
                         "man sollte sie im Blick haben, bevor man diese Abbildungen liest, als maßen sie "
                         "den klinischen Schweregrad: sie messen die Geometrie des Schaltkreises, nicht "
                         "das Risiko."),

        # ── Seite 6 · Live-Prädiktor ──
        "lp_eyebrow": "Interaktive Inferenz",
        "lp_title": "Live-Prädiktor",
        "lp_subtitle": ("Wahrscheinlichkeit, dass ein klinisches Profil zu einer bereits mit Diabetes "
                        "diagnostizierten Person gehört: LightGBM über die 8 wichtigsten Variablen."),
        "lp_what_note": ("<b>Was dieses Formular schätzt.</b> Das Ziel der Pipeline ist "
                         "<code>TARGET = (DIQ010 == 1)</code>, die Antwort auf <i>„Hat Ihnen jemals ein "
                         "Arzt gesagt, dass Sie Diabetes haben?“</i>. Das Modell <b>erkennt</b> also "
                         "<b>bereits diagnostizierten Diabetes</b>: es sagt nicht voraus, wer ihn "
                         "entwickeln wird. Das ist eine Aufgabe der gleichzeitigen Erkennung, kein "
                         "prospektives Risiko."),
        "lp_real_note": ("<b>Echte Inferenz (ONNX).</b> Vorhersagen von LightGBM und SVM-RBF über "
                         "<code>onnxruntime</code>, mit dem aus der Gold-Pipeline geladenen "
                         "<code>StandardScaler</code>. Die 8 gezeigten Variablen sind die klinisch "
                         "wichtigsten; die übrigen 81 Features werden auf dem Median des Trainingssatzes "
                         "festgehalten. Die QSVM steht wegen des Aufwands O(n²) des Quantenkernels nicht "
                         "in Echtzeit zur Verfügung: die Vorhersage der 1.567 Testinstanzen kostete "
                         "144,5 Minuten."),
        "lp_proxy_note": ("⚠ <b>Technischer und klinischer Hinweis.</b> Dieses Formular hat die echten "
                          "serialisierten Modelle (<code>.onnx</code>) nicht angebunden: Lege "
                          "<code>lgbm_final.onnx</code>, <code>svm_final.onnx</code>, "
                          "<code>scaler_correcto.json</code> und <code>medianas_correctas.json</code> "
                          "in <code>streamlit/models/</code>. Der unten gezeigte Wert ist ein "
                          "<b>transparenter Platzhalter</b>: eine nach normierter SHAP-Wichtigkeit "
                          "gewichtete Kombination, ausschließlich zu Layoutzwecken. <b>Er ist nicht die "
                          "Ausgabe eines trainierten Modells</b> und darf nicht als Ergebnis zitiert "
                          "werden. Auch die QSVM steht nicht in Echtzeit zur Verfügung (Aufwand O(n²) "
                          "des Quantenkernels)."),
        "lp_train_range": "Training: {mu} ± {sd} (±3 SD → {lo} bis {hi})",
        "lp_extrapolates": "⚠ z = {z} · außerhalb des trainierten Bereichs: das Modell extrapoliert",
        "lp_ada": "ADA-Kriterium: &lt; 5,7 normal · 5,7–6,4 Prädiabetes · ≥ 6,5 Diabetes",
        "lp_who_model": "das Modell",
        "lp_who_proxy": "der Platzhalter",
        "lp_score_real": "Wahrscheinlichkeit einer bestehenden Diagnose",
        "lp_score_proxy": "Layout-Score (Platzhalter)",
        "lp_cat_low": "Gering",
        "lp_cat_mid": "Mittel",
        "lp_cat_high": "Hoch",
        "lp_interp_low": ("Das Profil liegt deutlich unter der Entscheidungsschwelle (50 %): {quien} würde "
                          "es als nicht diagnostiziert einstufen."),
        "lp_interp_mid": "Der Wert nähert sich der Entscheidungsschwelle (50 %): Zone der Unsicherheit.",
        "lp_interp_high": ("Der Wert überschreitet die Entscheidungsschwelle (50 %): {quien} würde dieses "
                           "Profil als positiven Fall einstufen."),
        # Dos puntos y no un adjetivo antepuesto: la categoría llega ya declinada desde
        # lp_cat_* y en alemán la terminación del atributo depende del género y del caso
        # ("hohe Übereinstimmung", "mittlere…"), que no se puede componer con un {cat}
        # suelto. Así el rótulo es correcto con las tres, y en versalitas se lee igual.
        "lp_badge": "Übereinstimmung: {cat}",
        "lp_gauge_caption": ("Übereinstimmung mit einer bestehenden Diagnose: gering · mittel · hoch · "
                             "&nbsp;Entscheidungsschwelle = 50 %"),
        "lp_side_title": "Die beiden Modelle nebeneinander",
        "lp_side_sub": ("Jede Wahrscheinlichkeit wird am Schnittpunkt ihres eigenen Modells beurteilt: Sie "
                        "sind nicht austauschbar"),
        "lp_own_threshold": "Seine Schwelle",
        "lp_would_classify": "Würde einstufen als",
        "lp_positive": "positiv",
        "lp_negative": "negativ",
        "lp_disagree": ("<b>Die beiden Modelle weichen bei diesem Profil um {dif} voneinander ab.</b> Sie "
                        "stimmen an den Extremen überein (klar gesunde oder klar diabetische Profile) "
                        "und gehen im mittleren Band auseinander, also genau dort, wo eine Schätzung am "
                        "nützlichsten wäre. Nimm es als Zeichen von Unsicherheit, nicht als Hinweis "
                        "darauf, dass eines der beiden recht hat."),
        "lp_curve_title": "Antwortkurve",
        "lp_curve_var": "Zu durchlaufende Variable",
        "lp_curve_yaxis": "Wahrscheinlichkeit",
        "lp_curve_thr": "Schwelle",
        "lp_ada_bands": ["normal", "Prädiabetes", "Diabetes"],
        "lp_curve_note": ("Über diese Variable liefert LightGBM <b>{n} verschiedene Werte</b> auf den "
                          "{total} Positionen des Schiebereglers: es ist eine Treppe, keine Rampe. Größte "
                          "Stufen: {saltos}. Der Punkt markiert deinen aktuellen Wert."),
        "lp_curve_none": "keine",
        "lp_read_note": ("<b>Wie diese Ergebnisse zu lesen sind.</b> Da das Ziel eine <i>bereits "
                         "gestellte</i> Diagnose ist, lernt das Modell auch den Effekt der "
                         "<b>Behandlung</b>, nicht nur den der Krankheit. Das kehrt die klinische "
                         "Bedeutung zweier Variablen um:"
                         '<ul style="margin:8px 0 0; padding-left:20px; line-height:1.7;">'
                         "<li><b>LDL-Cholesterin</b>: Je höher das LDL, desto <i>geringer</i> die "
                         "geschätzte Wahrscheinlichkeit (von 43 % auf 18 % über den Schieberegler). "
                         "Diagnostizierte Personen sind meist mit Statinen behandelt.</li>"
                         "<li><b>Nüchternglukose</b>: Die Antwort ist U-förmig: sehr niedrige Werte "
                         "heben die Schätzung ebenso wie hohe, wegen der Hypoglykämien behandelter "
                         "Patienten.</li></ul>"
                         '<div style="margin-top:10px;">Keine der beiden darf als veränderbarer '
                         "Risikofaktor gelesen werden.</div>"),
    },

    # ═══════════════════════════════ FRANCÉS ═══════════════════════════════
    # Francés de Francia, registro académico. Rige lo mismo que en los demás para lo que
    # NO se traduce: nombres propios, librerías, códigos de variable NHANES y nombres de
    # fichero del repositorio.
    #
    # NOTACIÓN NUMÉRICA: coma decimal como el español, pero el separador de MILLAR es un
    # espacio inseparable, no un punto — "1 567", nunca "1.567". Es la única de las cinco
    # lenguas que lo hace así, y por eso nf() dejó de ser un intercambio de dos signos y
    # pasó a leer la pareja de separadores de un mapa (ver SEPARADORES en app.py). El
    # signo de porcentaje también va precedido de espacio.
    #
    # ESPACIADO TIPOGRÁFICO: el francés pide espacio ANTES de los dos puntos y dentro de
    # las comillas angulares. Aquí van espacios normales y no inseparables: el inseparable
    # sería lo canónico, pero es un carácter invisible, y sembrar el catálogo de U+00A0
    # que nadie ve al revisarlo cuesta más de lo que arregla — la diferencia se reduce a
    # dónde puede partir la línea. La excepción es el separador de millar, que sí es
    # inseparable porque ahí partir la línea rompería la cifra en dos.
    #
    # COMILLAS: las angulares francesas (« … »), con su espacio interior.
    "fr": {
        # ── Navigation et barre latérale ──
        "nav": ["Aperçu", "Gouvernance", "Résultats", "Analyse SHAP",
                "Circuit quantique", "Prédicteur en direct"],
        "sidebar_expand": "Déplier la barre latérale",
        "sidebar_collapse": "Replier la barre latérale",
        "search_label": "Rechercher",
        "search_ph": "Rechercher dans le tableau de bord ou sur le web…",
        "search_expand": "Rechercher : déplie la barre latérale",
        "scroll_top": "Revenir en haut",
        "search_in": "dans {p}",
        "search_none": "Aucun résultat dans le tableau de bord.",
        "search_web": "Rechercher « {q} » dans :",
        "theme_to_dark": "Passer au thème sombre",
        "theme_to_light": "Passer au thème clair",
        "lang_es_help": "Voir l'application en espagnol",
        "lang_en_help": "Voir l'application en anglais",
        "lang_de_help": "Voir l'application en allemand",
        "lang_fr_help": "Voir l'application en français",
        "lang_it_help": "Voir l'application en italien",
        "footer_name": "Juan Albornoz C. · Mémoire de master 2026",
        "footer_uni": "Universidad Europea de Valencia",
        "footer_name_narrow": "JAC",
        "footer_uni_narrow": "UEV",

        # ── Page 1 · Aperçu ──
        "ov_eyebrow": "Framework DataOps + QML",
        "ov_hero_title": ("Intégration du Quantum Machine Learning dans un pipeline DataOps : "
                          "architecture Médaillon sur Databricks et comparaison avec des modèles classiques "
                          "en prédiction clinique"),
        "ov_title": "Aperçu",
        "ov_subtitle": ("Pipeline de bout en bout sur Databricks CE + AWS S3, avec un QSVM quantique "
                        "face à deux baselines classiques, validé sur des données cliniques réelles "
                        "de l'étude NHANES (CDC)."),
        "ov_lead": (
            "Ce framework conçoit et met en œuvre un pipeline <b>DataOps de bout en bout</b> sur "
            "<b>Databricks Community Edition</b>, avec <b>AWS S3</b> comme véritable couche de "
            "stockage cloud et une architecture <b>Médaillon</b> (Bronze → Silver → Gold) sur Delta "
            "Lake comme colonne vertébrale. Le cas d'usage prédit le diabète de type 2 à partir des "
            "enregistrements de l'étude <b>NHANES</b> (CDC) : le jeu de données n'est pas l'objet de "
            "la recherche, mais le véhicule qui démontre que l'architecture est viable, reproductible "
            "et auditable sur des données réelles à grande échelle. Le cœur expérimental est une "
            "<b>comparaison triangulée</b> entre LightGBM (baseline tabulaire), un SVM à noyau RBF "
            "(pont structurel) et un <b>QSVM</b> avec FidelityQuantumKernel sous Qiskit, en gardant "
            "le classifieur sous-jacent identique afin d'attribuer toute différence de performance à "
            "l'effet du noyau quantique."
            " L'évaluation ferme le parcours : chaque modèle est mesuré avec AUC-ROC, F1, accuracy "
            "et MCC, <b>SHAP</b> désigne sur LightGBM les 20 variables qui pèsent le plus dans la "
            "prédiction, et les deux modèles classiques sont sérialisés en <b>ONNX</b> avec leur "
            "portabilité vérifiée. Le dépôt GitHub publie les 7 notebooks qui exécutent ce "
            "parcours, et cette application même, déployée sur Streamlit Cloud, en est le dernier "
            "maillon : la prédiction en direct et sa lecture SHAP."),
        "ov_arch_alt": ("Schéma de l'architecture du pipeline : AWS S3 alimente Databricks "
                        "Community Edition, où l'architecture Médaillon (Bronze, Silver et Gold) "
                        "débouche sur trois modèles (LightGBM, SVM à noyau RBF et QSVM avec Qiskit) et "
                        "sur l'évaluation avec métriques, SHAP et sérialisation ONNX ; la sortie part "
                        "vers GitHub et Streamlit Cloud."),
        # Rótulos del diagrama de arquitectura que abre la página (arquitectura_svg()). Las
        # cifras NO van escritas: llegan por marcador y las pone mil(), que usa el separador
        # de millar del idioma. Lo que no viaja aquí es la geometría ni qué caja va resaltada,
        # que son dibujo y no texto.
        "ov_arch_io": (
            ("AWS S3", "NHANES brut · 27 XPT", "IAM"),
            ("GitHub", "7 notebooks · README"),
            ("Streamlit Cloud", "Prédiction · SHAP visuel"),
        ),
        "ov_arch_grupos": (
            ("Architecture Médaillon", (
                ("Bronze · ingestion brute",
                 "{bronze} lignes · 162 col. · Delta Lake ACID"),
                ("Silver · qualité",
                 "{silver} lignes · 91 col. · expectations"),
                ("Gold · features curées",
                 "89 features · train {train} / test {test}"),
            )),
            ("Modèles · ML / QML", (
                ("LightGBM",
                 "Baseline tabulaire · GOSS · EFB"),
                ("SVM · noyau RBF",
                 "Pont direct vers le QSVM"),
                ("QSVM · Qiskit",
                 "ZZFeatureMap · FidelityQuantumKernel"),
            )),
            ("Évaluation et sérialisation", (
                ("Métriques",
                 "AUC-ROC · F1 · Accuracy · MCC"),
                ("SHAP",
                 "Explicabilité LightGBM · top 20"),
                ("Sélection · ONNX",
                 "Portabilité vérifiée"),
            )),
        ),
        "ov_stats_title": "Statistiques du jeu de données NHANES",
        "ov_stats_sub": "Trois cycles biennaux intégrés · pipeline en couches Bronze → Silver → Gold",
        "ov_stat_bronze": "Enregistrements Bronze",
        "ov_stat_silver": "Enregistrements Silver",
        "ov_stat_features": "Features Gold",
        "ov_stat_balance": "Équilibre des classes",
        "ov_medallion_title": "Architecture Médaillon",
        "ov_medallion_sub": "Chaîne de valeur de la donnée (Curry, 2016) appliquée couche par couche",
        "ov_layers": [
            ("Bronze", "Ingestion depuis AWS S3 sans transformation. Préserve la source de vérité."),
            ("Silver", "Nettoyage, imputation, winsorisation et validation de la qualité."),
            ("Gold",   "Mise à l'échelle, encodage et partition stratifiée. Prêt à modéliser."),
        ],
        "ov_goto_gov": "Voir les contrôles de qualité et de traçabilité  →",
        "ov_target_title": "Distribution de la variable cible (DIQ010)",
        "ov_target_sub": "Cible binarisée : 1 = diabète diagnostiqué, 0 = le reste",
        "ov_pie_no": "Pas de diabète",
        "ov_pie_yes": "Diabète",
        "ov_donut_center": "14 %",
        "ov_donut_caption": "DIABÈTE",
        "ov_tech_title": "Construit sur",
        "ov_tech_sub": ("Plateforme, stockage et bibliothèques du pipeline, dans leur ordre "
                        "d'intervention · l'inventaire complet, avec la justification de chaque "
                        "choix, se trouve dans Gouvernance"),
        "ov_compare_title": "Comparaison triangulée : objectif de l'expérience",
        "ov_compare": [
            ("LightGBM", "Baseline tabulaire de référence"),
            ("SVM-RBF",  "Pont structurel vers la composante quantique"),
            ("QSVM",     "FidelityQuantumKernel : même classifieur, noyau quantique"),
        ],

        # ── Variables NHANES (partagé : SHAP, Circuit, Bloch, Prédicteur) ──
        # Les codes NHANES ne sont jamais traduits, seulement leur glose.
        "var_short": {
            "LBXGH": "HbA1c", "RIDAGEYR": "Âge", "LBXGLU": "Glycémie à jeun",
            "LBDLDL": "Cholestérol LDL", "BMXWAIST": "Tour de taille",
            "WTINT2YR": "Poids d'échantillon*", "BMXARML": "Longueur du bras",
            "BMXLEG": "Longueur de jambe", "BMXBMI": "IMC",
            "PAD680": "Activité sédentaire", "PAD645": "Activité modérée",
            "PAQ640": "Renforcement musculaire", "BMXWT": "Poids corporel", "LBXIN": "Insuline",
            "INDHHIN2": "Revenu du ménage", "DMDYRSUS": "Années aux États-Unis",
            "BMXARMC": "Circonf. du bras", "PAQ670": "Activité intense",
            "BPXSY1": "Pression systolique", "PAD630": "Activité mod. de loisir",
            "DMDHHSZE": "Taille du ménage (enfants)", "BPXDI1": "Pression diastolique",
            "LBXTR": "Triglycérides", "DMDMARTL_1": "État civil (marié)",
            "DMDMARTL_5": "État civil (jamais marié)", "BPXPLS": "Pouls",
            "DMDEDUC2_3": "Éducation (niveau 3)", "SDMVSTRA": "Strate d'échantillonnage",
            "DMDMARTL_2": "État civil (veuf)", "DMDHHSZB": "Taille du ménage (adultes)",
        },
        "var_desc": {
            "LBXGH":      "Hémoglobine glyquée (HbA1c) : glycémie moyenne des 2-3 derniers mois. Marqueur diagnostique primaire du diabète (ADA : ≥ 6,5 %).",
            "RIDAGEYR":   "Âge du participant au moment de l'examen (années).",
            "LBXGLU":     "Glycémie plasmatique à jeun : marqueur biochimique du contrôle glycémique (mg/dL).",
            "LBDLDL":     "Cholestérol LDL calculé : fraction du cholestérol liée au risque cardiovasculaire (mg/dL).",
            "BMXWAIST":   "Tour de taille : adiposité abdominale associée à l'insulinorésistance (cm).",
            "WTINT2YR":   "Facteur de pondération de l'entretien NHANES. Artefact du plan de sondage, pas une variable clinique.",
            "BMXARML":    "Longueur du bras (acromion → olécrane) : mesure anthropométrique (cm).",
            "BMXLEG":     "Longueur maximale de la jambe (genou → sol) : mesure anthropométrique (cm).",
            "BMXBMI":     "Indice de masse corporelle (poids/taille²) : adiposité corporelle globale (kg/m²).",
            "PAD680":     "Minutes d'activité sédentaire par jour (temps assis ou allongé).",
            "PAD645":     "Minutes hebdomadaires d'activité physique modérée (travail + loisirs).",
            "PAQ640":     "Jours par semaine avec des activités de renforcement musculaire.",
            "BMXWT":      "Poids corporel total (kg).",
            "LBXIN":      "Insuline sérique à jeun : marqueur de l'insulinorésistance (µU/mL).",
            "INDHHIN2":   "Niveau de revenu du ménage (variable socio-économique catégorielle).",
            "DMDYRSUS":   "Nombre d'années de résidence aux États-Unis.",
            "BMXARMC":    "Circonférence moyenne du bras : mesure anthropométrique (cm).",
            "PAQ670":     "Minutes hebdomadaires d'activité de loisir intense.",
            "BPXSY1":     "Pression artérielle systolique, première mesure (mmHg).",
            "PAD630":     "Minutes hebdomadaires d'activité physique modérée de loisir.",
            "DMDHHSZE":   "Composition du ménage : nombre d'enfants dans le foyer.",
            "BPXDI1":     "Pression artérielle diastolique, première mesure (mmHg).",
            "LBXTR":      "Triglycérides sériques : marqueur du profil lipidique (mg/dL).",
            "DMDMARTL_1": "État civil = marié (variable indicatrice après encodage one-hot).",
            "DMDMARTL_5": "État civil = jamais marié (variable indicatrice après encodage one-hot).",
            "BPXPLS":     "Pouls : fréquence cardiaque au repos (battements/min).",
            "DMDEDUC2_3": "Niveau d'études intermédiaire (lycée/GED) : variable indicatrice après encodage one-hot.",
            "SDMVSTRA":   "Strate de variance du plan de sondage NHANES (variable méthodologique, non clinique).",
            "DMDMARTL_2": "État civil = veuf (variable indicatrice après encodage one-hot).",
            "DMDHHSZB":   "Composition du ménage : nombre d'adultes dans le foyer.",
        },
        "qsvm_labels": {
            "LBXGH": "HbA1c", "LBXGLU": "Glycémie à jeun", "RIDAGEYR": "Âge",
            "LBDLDL": "Cholestérol LDL", "BMXWAIST": "Tour de taille", "LBXIN": "Insuline",
            "BMXLEG": "Longueur de jambe", "BMXBMI": "IMC",
        },
        "qsvm_units": {"años": "ans"},

        # ── Page 2 · Gouvernance ──
        "gov_eyebrow": "Gouvernance · DataOps",
        "gov_title": "Gouvernance et qualité de la donnée",
        "gov_subtitle": ("Les contrôles qui soutiennent le pipeline : ce qui est validé, ce qui est "
                         "écarté et pourquoi, ce qui est consigné et avec quels frameworks. Tous les "
                         "chiffres proviennent des sorties exécutées des notebooks."),
        "gov_tabs": ["Qualité de la donnée", "Traçabilité et lignage", "Inventaire des frameworks"],
        "gov_kpi_expect": "Attentes satisfaites",
        "gov_kpi_passrate": "Taux de réussite de la suite",
        "gov_kpi_records": "Enregistrements validés",
        "gov_kpi_leakage": "Artefacts sans fuite",
        "gov_funnel_title": "Entonnoir des enregistrements",
        "gov_funnel_sub": ("Sur les {bronze} enregistrements de Bronze, {silver} survivent aux filtres "
                           "de cohorte de Silver. Chaque palier répond à un critère explicite, pas à un "
                           "nettoyage générique."),
        "gov_hover_records": "Enregistrements",
        "gov_hover_dropped": "Écartés",
        "gov_embudo": [
            ("Bronze · 3 cycles réunis",
             "27 fichiers XPT · jointure par SEQN · 162 colonnes communes aux trois cycles"),
            ("Filtre âge ≥ 18 ans", "Restriction à la population adulte"),
            ("Filtre à jeun · LBXGLU non nul",
             "Proxy du sous-groupe à jeun : PHAFSTMN n'est pas cohérent d'un cycle à l'autre"),
            ("Filtre DIQ010 valide",
             "Écarte les codes 7 « ne sait pas » et 9 « refuse de répondre », ainsi que les valeurs nulles"),
        ],
        "gov_dropped_title": "Enregistrements écartés par filtre",
        "gov_split_label": "Partition Gold 80/20",
        "gov_suite_title": "Suite de validation · dataframe-expectations",
        "gov_suite_sub": ("Suite <code>{nombre}</code>, exécutée le {fecha} sur les {registros} "
                          "enregistrements de Silver en {duracion} secondes. Great Expectations est "
                          "incompatible avec les versions figées du runtime serverless : voici "
                          "l'alternative adoptée."),
        "gov_expectativas": [
            ("Complétude", "TARGET", "au plus 0 valeur nulle"),
            ("Complétude", "LBXGH", "au plus 0 valeur nulle"),
            ("Complétude", "LBXGLU", "au plus 0 valeur nulle"),
            ("Complétude", "RIDAGEYR", "au plus 0 valeur nulle"),
            ("Complétude", "BMXBMI", "au plus 0 valeur nulle"),
            ("Plages cliniques", "RIDAGEYR", "minimum entre 18 et 25"),
            ("Plages cliniques", "RIDAGEYR", "maximum entre 70 et 120"),
            ("Plages cliniques", "LBXGH", "minimum entre 3,0 et 6,0"),
            ("Plages cliniques", "LBXGH", "maximum entre 8,0 et 20,0"),
            ("Plages cliniques", "LBXGLU", "minimum entre 30 et 80"),
            ("Plages cliniques", "LBXGLU", "maximum entre 150 et 500"),
            ("Plages cliniques", "BMXBMI", "minimum entre 10,0 et 18"),
            ("Plages cliniques", "BMXBMI", "maximum entre 40,0 et 80"),
            ("Volume", "DataFrame", "au moins 7 000 lignes"),
            ("Volume", "DataFrame", "au plus 9 000 lignes"),
        ],
        "gov_ops_title": "Opérations de qualité par couche",
        "gov_silver_card": "Silver · nettoyage et assainissement",
        "gov_gold_card": "Gold · préparation à la modélisation",
        "gov_silver_ops": [
            ("Variables DIQ exclues pour fuite", "DIQ050, DIQ070, DIQ160, DIQ170, DIQ172, DIQ180"),
            ("Colonnes creuses supprimées", "Seuil de >80 % de valeurs manquantes"),
            ("Variables winsorisées", "Écrêtage des valeurs aberrantes à IQR × 3"),
            ("Manquants après imputation", "De 75 855 à 0 dans le jeu SVM/QSVM (médiane + mode)"),
        ],
        "gov_gold_ops": [
            ("Features après encodage", "One-hot de 5 variables catégorielles sur 84 features (106 colonnes avec TARGET)"),
            ("Écartées par corrélation", "Seuil r > 0,90 entre paires de prédicteurs"),
            ("Features finales", "L'ensemble sur lequel les trois modèles sont entraînés"),
            ("Partition stratifiée", "80/20 · 14,03 % de positifs à l'entraînement, 14,04 % au test"),
        ],
        "gov_eff_title": "Features effectives face aux features nominales",
        "gov_eff_sub": ("Compté d'après <code>scaler_correcto.json</code> : {const} des {total} colonnes "
                        "ont une variance nulle et n'apportent aucune information au modèle"),
        "gov_eff_nominal": "Features nominales",
        "gov_eff_const": "Constantes (variance = 0)",
        "gov_eff_effective": "Features effectives",
        "gov_eff_note": ("C'est l'effet collatéral de la winsorisation IQR × 3 de Silver, appliquée aussi "
                         "à des variables catégorielles encodées numériquement (réponses 1/2, langue de "
                         "l'entretien, codes 7 et 9). Lorsque plus de 75 % de l'échantillon répond la même "
                         "chose, l'écrêtage réduit la colonne à une valeur unique. Les plus écrêtées dans "
                         "le notebook 02 (PAQ635, PAQ650, PAQ605, DMDHHSZA, DMDCITZN, SIALANG) sont "
                         "exactement celles qui apparaissent ici comme constantes."),
        "gov_lin_title": "Traçabilité sans MLflow",
        "gov_lin_sub": "La contrainte qui conditionne le plus l'architecture du pipeline, et son atténuation.",
        "gov_lin_limit_title": "Limitation",
        "gov_lin_limit_body": ("L'intégration native de <b>MLflow</b> est désactivée dans Databricks "
                               "Serverless gratuit. Tout appel à <code>mlflow.start_run()</code> ou "
                               "<code>mlflow.log_metric()</code> produit des erreurs d'authentification : "
                               "aucun suivi des expériences, des métriques ni des artefacts."),
        "gov_lin_mit_title": "Atténuation · double mécanisme",
        "gov_lin_mit_body": ("<b>Journaux de transactions de Delta Lake</b> : chaque écriture génère un "
                             "enregistrement ACID avec version, horodatage et métriques d'opération."
                             "<br><br><b>CSV de métriques par modèle</b> : chaque notebook persiste ses "
                             "résultats dans Unity Catalog Volumes, et les figures les y lisent au lieu "
                             "de les porter en dur."),
        "gov_delta_title": "Historique Delta · couche Gold",
        "gov_delta_sub": ("Les six versions les plus récentes sur les dix enregistrées. Delta purge les "
                          "précédentes après 168 h de rétention : comportement attendu, et non une "
                          "défaillance du pipeline."),
        "gov_delta_cols": ["Version", "Horodatage", "Opération", "Lignes", "Taille"],
        "gov_chain_title": "Chaîne de contrôle contre la fuite d'information",
        "gov_chain_sub": ("Quatre barrières enchaînées. La troisième n'écarte aucune colonne, et c'est "
                          "exactement ce que l'on veut voir : la preuve que les précédentes ont fait "
                          "leur travail."),
        "gov_leakage": [
            ("Exclusion dans Silver",
             "6 variables DIQ de traitement et de suivi sont supprimées avant la winsorisation : "
             "elles sont une conséquence du diagnostic, non des prédicteurs de celui-ci."),
            ("Vérification croisée",
             "On vérifie qu'aucune DIQ ne survit dans les 2 Parquet de Silver ni dans les 13 de "
             "Gold. Résultat : 15/15 artefacts propres."),
            ("Filtre défensif du QSVM",
             "Deuxième barrière avant la sélection par Random Forest. Elle n'écarte aucune colonne "
             "(89 sur 89 passent), précisément la preuve que la première a fonctionné."),
            ("Garde-fou des poids de sondage",
             "Arrête le pipeline si un poids de sondage autre que celui connu apparaît. WTINT2YR "
             "atteint bel et bien la modélisation et est documenté à la décision 10."),
        ],
        "gov_scaler_card": "Mise à l'échelle sans fuite statistique",
        "gov_scaler": [
            ("Ajustement", "Sur train uniquement", "fit_transform sur train · transform sur test"),
            ("Colonnes évaluées", "66", "Avec variance > 0"),
            ("Colonnes constantes", "23", "Variance 0 · voir décision 08"),
            ("Moyenne ≈ 0 · écart-type ≈ 1", "Vérifié", "Assert sur toutes les colonnes avec dispersion"),
        ],
        "gov_scaler_note": ("Le <b>StandardScaler</b> est ajusté exclusivement sur <b>train</b> : "
                            "<code>fit_transform</code> à l'entraînement et <code>transform</code> au "
                            "test. S'il était ajusté sur l'ensemble complet, la moyenne et l'écart-type "
                            "du test fuiraient dans le prétraitement et les métriques seraient "
                            "optimistes. La sélection des 8 variables du QSVM suit la même règle : le "
                            "Random Forest n'est entraîné que sur <code>X_train_svm_scaled</code>."
                            "<br><br>Le filtre de corrélation, en revanche, <b>est bien</b> calculé "
                            "avant la partition. C'est documenté et assumé à la décision 09."
                            "<br><br>La vérification s'exécute, elle ne se déclare pas : sur les 66 colonnes ayant "
                            "de la dispersion, on exige |moyenne| &lt; 0,01 et un écart-type entre 0,90 et 1,10. "
                            "Les paramètres ajustés (<code>mean_</code> et <code>scale_</code>) partent dans "
                            "<code>scaler_correcto.json</code>, le fichier que charge le Prédicteur en direct : "
                            "l'échelle du train n'est jamais recalculée."),
        "gov_e2e_title": "Vérification de bout en bout face aux modèles entraînés",
        "gov_e2e_missing": ('<b style="color:{color};">Non vérifié.</b> Le jeu de test n\'est pas dans le '
                            "dépôt, le tableau de bord ne peut donc pas vérifier seul que son chemin "
                            "d'inférence reproduit ce qu'ont produit les modèles entraînés. Pour clore ce "
                            "point, exécutez les deux cellules de "
                            "<code>notebooks/INSTRUCCIONES_exportar_golden_set.md</code> et copiez "
                            "<code>golden_lgbm.npz</code> et <code>golden_svm.npz</code> dans "
                            "<code>streamlit/models/</code>. Tant qu'ils manquent, cette page n'affirme "
                            "rien qu'elle n'ait pu vérifier."),
        "gov_e2e_unavailable": "non disponible",
        "gov_e2e_ok_val": "{n} lignes · écart max. {dif}",
        "gov_e2e_bad_val": "DIVERGENCE · écart max. {dif}",
        "gov_e2e_scaled": "met à l'échelle et appelle le modèle ONNX",
        "gov_e2e_raw": "appelle le modèle ONNX sans mise à l'échelle",
        "gov_e2e_path": "Le tableau de bord {accion}",
        "gov_e2e_ok_title": "✓ Chemin d'inférence vérifié",
        "gov_e2e_fail_title": "⚠ Le chemin d'inférence ne reproduit pas les modèles",
        "gov_e2e_note": ("Chaque ligne du <i>golden set</i> est une instance réelle du test accompagnée "
                         "de la probabilité renvoyée par le modèle entraîné dans son notebook. Le "
                         "tableau de bord la fait passer par son propre chemin (vecteur brut, mise à "
                         "l'échelle du SVM seulement, conversion en <code>float32</code>, session ONNX "
                         "et lecture du tenseur de sortie) puis compare. Tolérance {tol} ; le bruit "
                         "attendu du travail en <code>float32</code> est de l'ordre de 10⁻⁷."),
        "gov_stack_title": "Frameworks par couche",
        "gov_stack_sub": ("Le premier badge de chaque carte est le framework qui structure la couche ; "
                          "les autres l'accompagnent."),
        "gov_stack": [
            ("ingestion",
             "boto3 remplace spark.conf, bloqué sous Serverless (décision 01). Trois asserts "
             "d'intégrité : 27/27 fichiers, la jointure par SEQN ne duplique pas de lignes, et "
             "Delta concorde avec pandas."),
            ("qualité",
             "Le framework de qualité de ce mémoire. Great Expectations est incompatible avec "
             "l'environnement (décision 03). Suite de 15 attentes sur 3 dimensions, avec des "
             "preuves persistées en CSV."),
            ("préparation",
             "Mise à l'échelle ajustée sur train uniquement, partition stratifiée à graine fixe "
             "et export du contrat de service (scaler et médianes en JSON)."),
            ("modèle",
             "Interprétabilité exacte par l'algorithme polynomial sur les 1 567 instances de "
             "test, et vérification que le modèle ONNX reproduit le PKL à 100 %."),
            ("modèle",
             "SHAP agnostique au modèle, au prix de plusieurs heures : calculé une fois sur 200 "
             "instances et persisté sur disque pour être réutilisé."),
            ("modèle",
             "Pas de support ONNX : le format n'admet pas d'opérations quantiques (décision 05). "
             "La traçabilité repose sur un CSV de métriques avec les 14 champs de configuration."),
        ],
        "gov_dec_title": "Registre des décisions",
        "gov_dec_sub": ("Les onze limitations documentées dans TECHNICAL_NOTES, avec leur atténuation. "
                        "Trois conditionnent l'architecture, six sont assumées et documentées sans "
                        "correction (car la corriger invaliderait les résultats déjà obtenus) et deux "
                        "sont résolues sans reliquat."),
        "gov_dec_tags": {"critical": "Architecture", "warning": "Assumée", "good": "Résolue"},
        "gov_dec_problem": "Problème · ",
        "gov_dec_solution": "Solution adoptée · ",
        "gov_decisiones": [
            ("spark.conf bloqué sous Serverless",
             "La configuration des identifiants AWS via spark.conf.set renvoie CONFIG_NOT_AVAILABLE, "
             "le mécanisme standard pour connecter Spark à S3.",
             "boto3 comme client alternatif. S3 reste le stockage d'origine et Unity Catalog Volumes "
             "la couche de traitement."),
            ("MLflow bloqué sous Serverless",
             "L'intégration native de MLflow est désactivée dans l'offre gratuite : aucun suivi des "
             "expériences, des métriques ni des artefacts.",
             "Double mécanisme de remplacement : les journaux de transactions de Delta Lake "
             "fournissent version, horodatage et métriques d'opération ; et chaque notebook persiste "
             "ses métriques en CSV."),
            ("Great Expectations incompatible",
             "Il exige une combinaison pandas/numpy qui entre en conflit avec les versions figées du "
             "runtime serverless (pandas 1.5.3 / numpy 1.23.5).",
             "dataframe-expectations 0.7.0 comme alternative compatible. 15 attentes sur Silver dans "
             "trois dimensions. Résultat 15/15, taux de réussite 1,0."),
            ("QSVM · coût de calcul O(n²)",
             "Sur les 6 264 instances d'entraînement, la matrice de noyau exigerait environ "
             "39 millions d'évaluations du circuit. À 1 500, le noyau épuise la mémoire.",
             "Entraînement sur un échantillon stratifié de 500 instances (~22 min) en préservant le "
             "ratio 86/14. L'évaluation utilise bien le test complet, pour que les métriques restent "
             "comparables."),
            ("QSVM · pas de support ONNX natif",
             "Le format ONNX n'admet pas d'opérations quantiques : ni skl2onnx ni onnxmltools ne "
             "peuvent sérialiser un noyau fondé sur la simulation d'états.",
             "Sérialisation avec joblib. Le modèle exige l'environnement Qiskit pour l'inférence, si "
             "bien que le QSVM n'entre pas dans le Prédicteur en direct."),
            ("Versions de Qiskit non figeables",
             "Le fichier immutable_package_constraints.txt de Databricks bloque l'installation de "
             "versions précises : pas de reproductibilité exacte de version.",
             "Le pipeline tourne avec les versions de l'environnement (2.5.0 / 0.9.0 / 0.4.0), dont "
             "l'API est compatible, et elles sont consignées par une vérification explicite au début "
             "de l'exécution."),
            ("Perte de variables due à la durée de session",
             "Les opérations longues (22 min d'entraînement, 132 de prédiction) peuvent épuiser la "
             "session serverless et emporter les variables en mémoire.",
             "Persistance immédiate après chaque opération coûteuse et mode TRAINING_MODE qui "
             "recharge depuis le disque lors des exécutions suivantes."),
            ("Winsorisation appliquée à des catégorielles encodées",
             "NHANES encode numériquement de nombreuses catégorielles. Si plus de 75 % partagent une "
             "valeur, IQR = 0, les bornes s'effondrent et clip() transforme la variable en constante. "
             "10 colonnes se sont ainsi effondrées.",
             "Documenté sans modification : corriger altérerait Silver, Gold et les trois modèles. "
             "Les colonnes constantes ne biaisent rien (le modèle n'en tire aucun signal), mais de "
             "l'information est perdue. La correction est identifiée comme travail futur."),
            ("Corrélation calculée avant la partition",
             "Le filtre r > 0,90 est calculé sur le jeu complet : les 16 colonnes écartées sont donc "
             "décidées en utilisant aussi les observations de test.",
             "Documenté sans modification. Cela n'affecte ni la mise à l'échelle ni la sélection de "
             "features du QSVM, toutes deux ajustées sur train uniquement, mais la sélection cesse "
             "d'être strictement aveugle au test."),
            ("Poids de sondage WTINT2YR parmi les features",
             "La jointure intracyclique duplique WTSAF2YR sur trois colonnes. WTINT2YR n'est pas sur "
             "la liste d'exclusion et survit au filtre de corrélation : c'est l'une des 89 features.",
             "Documenté sans modification, avec un assert qui détecte l'apparition de TOUT autre "
             "poids. Un poids de sondage n'est pas une variable clinique : il ne divulgue pas la "
             "cible, mais laisse le modèle s'appuyer sur le plan de l'enquête."),
            ("Le QSVM sérialisé n'est pas rechargeable d'une version à l'autre",
             "Le pickle entraîne avec lui la ZZFeatureMap et ses ParameterExpression. Si Qiskit change "
             "de version, la désérialisation échoue, et Serverless met à jour sans prévenir.",
             "Le chargement est enveloppé dans un try/except : en cas d'échec, TRAINING_MODE passe à "
             "True et le notebook réentraîne au lieu d'abandonner. Il reste opérationnel dans les "
             "trois scénarios possibles."),
        ],
        "gov_footer_note": ("Les chiffres de cette page proviennent des sorties exécutées des notebooks "
                            "du dépôt et de <code>TECHNICAL_NOTES.md</code> ; aucun n'est estimé. "
                            "L'application ne peut pas les consulter en direct, car Streamlit Community "
                            "Cloud n'accède qu'au dépôt, pas à Unity Catalog Volumes.<br><br>"
                            "Résumé de la suite de qualité : <b>{fuente}</b>."),
        "gov_suite_src_csv": "lu depuis validacion_silver_dfe.csv",
        "gov_suite_src_nb": "valeurs vérifiées du notebook",

        # ── Page 3 · Résultats ──
        "res_eyebrow": "Comparaison triangulée",
        "res_title": "Résultats",
        "res_subtitle": "LightGBM vs SVM-RBF vs QSVM sur le même jeu de test ({n} instances).",
        "res_threshold": "Seuil",
        "res_thr_label": {"lightgbm": "p ≥ {v}", "svm_rbf": "p ≈ {v}", "qsvm": "df > 0"},
        "res_thr_src": {"lightgbm": "predict_proba()[:,1] >= 0.5",
                        "svm_rbf": "SVC.predict() · signe de decision_function",
                        "qsvm": "decision_function > 0 (n'est pas une probabilité)"},
        "res_reconciled": ('<span style="color:{color}; font-weight:600;">✓ Réconciliées</span> : les '
                           "quatre métriques des trois modèles ont été recalculées à partir des scores "
                           "par instance et coïncident avec celles publiées."),
        "res_unreconciled": '<span style="color:{color}; font-weight:600;">⚠ Non réconciliées</span> : {fallos}',
        "res_no_scores": "scores non disponibles",
        "res_threshold_note": ("<b>Les trois modèles sont mesurés à des seuils différents.</b> Chacun "
                               "utilise son point de coupure naturel : LightGBM "
                               "<code>predict_proba ≥ 0,50</code> ; SVM-RBF le signe de "
                               "<code>decision_function</code>, qui sur l'échelle de probabilité "
                               "enregistrée équivaut à ≈ 0,22 ; QSVM <code>decision_function &gt; 0</code>, "
                               "qui n'est pas une probabilité. Chaque matrice se reproduit exactement à "
                               "son propre seuil, mais <b>seule l'AUC-ROC est comparable entre "
                               "modèles</b> : c'est la seule des quatre métriques indépendante du point "
                               "de coupure. À titre de référence, le SVM-RBF évalué à 0,50 comme "
                               "LightGBM donnerait une accuracy de 0,9190 mais seulement 131 vrais "
                               "positifs au lieu de 172."),
        "res_roc_title": "Courbes ROC",
        "res_roc_sub_real": ("Courbes empiriques réelles, point par point sur les 1 567 instances du "
                             "test (les mêmes scores que ceux qui donnent l'AUC du mémoire)."),
        "res_roc_sub_synth": ("AUC exacte · forme reconstruite à partir de l'AUC là où les scores par "
                              "instance manquent."),
        "res_cm_title": "Matrices de confusion",
        "res_cm_sub": ("Valeurs vérifiées face au classification report de chaque modèle, et recalculées "
                       "à partir des scores par instance. Chaque matrice correspond au seuil indiqué sur "
                       "sa carte"),
        "res_cm_pred_no": "Préd.<br>Pas de diabète",
        "res_cm_pred_yes": "Préd.<br>Diabète",
        "res_cm_real_no": "Réel<br>Pas de diab.",
        "res_cm_real_yes": "Réel<br>Diabète",
        "res_cm_tags": {"tn": "VN", "fp": "FP", "fn": "FN", "tp": "VP"},
        "res_metrics_title": "Comparaison des métriques",
        "res_metrics_sub": ("Les quatre métriques portent sur les 1 567 instances. Accuracy, MCC et "
                            "F1-macro pénalisent bien le déséquilibre des classes, mais elles dépendent "
                            "du seuil, et chaque modèle utilise le sien : comparez avec prudence tout ce "
                            "qui n'est pas l'AUC-ROC"),
        "res_metric_desc": {
            "auc": "Aire sous la courbe ROC : capacité à séparer diabète et non-diabète. 0,5 = hasard, 1 = parfait.",
            "f1_macro": "Moyenne harmonique de la précision et du rappel, moyennée par classe (sans pondération). Pénalise le déséquilibre.",
            "accuracy": "Proportion de prédictions correctes au total. Avec des classes déséquilibrées, elle peut ne refléter que la classe majoritaire.",
            "mcc": "Coefficient de corrélation de Matthews : qualité globale robuste au déséquilibre. 0 = hasard, 1 = parfait.",
        },
        "res_qsvm_note": ("<b>Note sur l'expérience QSVM.</b> Le QSVM a été entraîné sur un échantillon "
                          "stratifié de 500 instances (coût O(n²) du noyau quantique) et évalué sur les "
                          "1 567 du test complet. AUC-ROC = 0,5493 indique que le modèle dépasse à peine "
                          "la classification aléatoire : rappel ≈ 0 pour la classe diabète (1 sur 220), "
                          "et une accuracy de 0,8602 qui ne reflète que la proportion de la classe "
                          "majoritaire. Le MCC ≈ 0 confirme l'absence de véritable capacité prédictive."),

        # ── Page 4 · Analyse SHAP ──
        "sh_eyebrow": "Interprétabilité",
        "sh_title": "Analyse SHAP",
        "sh_subtitle": ("Importance globale des variables : TreeExplainer (LightGBM) vs "
                        "KernelExplainer (SVM-RBF)."),
        "sh_tabs": ["LightGBM · TreeExplainer", "SVM-RBF · KernelExplainer"],
        "sh_hint": "Survolez chaque barre pour voir la signification de la variable. {nota}",
        "sh_sample_lgbm": "Valeurs exactes (algorithme polynomial) sur les 1 567 instances du test.",
        "sh_sample_svm": ("Valeurs approchées par échantillonnage : fond de 100 instances, contributions "
                          "sur 200 instances de test."),
        "sh_note_lgbm": ("<b>LBXGH (HbA1c)</b> domine largement (SHAP moyen = 1,1243), en cohérence avec "
                         "son rôle de marqueur diagnostique primaire du diabète de type 2 (ADA : "
                         "HbA1c ≥ 6,5 %). <b>RIDAGEYR (âge, 0,4654)</b> reflète l'augmentation de la "
                         "prévalence avec l'âge. <b>LBXGLU</b> et <b>LBDLDL</b> complètent le bloc "
                         "biochimique. <b>WTINT2YR</b> (position 6) est un artefact du plan de sondage "
                         "NHANES, pas une variable clinique."),
        "sh_note_svm": ("Le classement du SVM-RBF coïncide avec LightGBM sur les variables dominantes "
                        "(<b>LBXGH</b>, <b>LBXGLU</b>, <b>LBDLDL</b>, <b>RIDAGEYR</b>), ce qui renforce "
                        "la validité clinique du résultat en le rendant indépendant de l'algorithme et "
                        "méthodologiquement plus robuste. KernelExplainer traite le modèle comme une "
                        "boîte noire, applicable à n'importe quel classifieur."),
        "sh_fig_lgbm_title": "SHAP Summary Plot · LightGBM (Figure 27)",
        "sh_fig_lgbm_cap": ("Chaque point est une instance du test ; la couleur indique la valeur de la "
                            "variable (rouge élevé, bleu bas) et la position horizontale son impact sur "
                            "la prédiction. LBXGH et RIDAGEYR dominent le modèle."),
        "sh_fig_svm_title": "SHAP Summary Plot · SVM-RBF (Figure 31)",
        "sh_fig_svm_cap": ("Chaque point est une instance ; couleur = valeur de la variable, position = "
                           "impact. KernelExplainer sur 200 instances du test."),

        # ── Page 5 · Circuit quantique ──
        "qc_eyebrow": "Composante quantique",
        "qc_title": "Circuit quantique",
        "qc_subtitle": ("Configuration de la ZZFeatureMap et du FidelityQuantumKernel implémentés sous "
                        "Qiskit sur Databricks CE."),
        "qc_tabs": ["Circuit ZZFeatureMap", "Sphère de Bloch"],
        "qc_specs": ["Qubits (feature_dimension)", "Répétitions (reps)", "Intrication", "Version de Qiskit"],
        "qc_how_title": "Comment ça marche",
        "qc_how_p1": ("La <b>ZZFeatureMap</b> encode chacune des 8 variables cliniques comme un angle de "
                      "phase (porte P) sur un qubit indépendant, après avoir créé la superposition avec "
                      "des portes de Hadamard. Son élément distinctif est l'<b>intrication</b> entre "
                      "paires de qubits au moyen de portes qui dépendent du produit croisé de deux "
                      "variables, des corrélations que le noyau RBF classique ne peut pas représenter."),
        "qc_how_p2": ("Le <b>FidelityQuantumKernel</b> mesure la similarité entre deux patients comme la "
                      "fidélité entre leurs états quantiques : <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. "
                      "L'implémentation utilise <code>StatevectorSampler</code> et simule l'état exact "
                      "sans bruit : des résultats déterministes et reproductibles."),
        "qc_feat_title": "8 features sélectionnées (Random Forest)",
        "qc_xaxis": "Importance RF",
        "qc_train_title": "Entraînement et évaluation",
        "qc_tstats": ["Instances d'entraînement", "Temps d'entraînement", "Instances de test",
                      "Temps d'inférence", "Vecteurs de support"],
        "qc_note": ("En raison du coût O(n²) du noyau quantique, l'entraînement s'est limité à un "
                    "échantillon stratifié de 500 instances (la limite opérationnelle de Databricks CE "
                    "serverless se situe autour de 500-1 000). L'évaluation a porté sur le test complet "
                    "(1 567 instances) par lots de 100, pour un temps total de prédiction de "
                    "144,5 minutes."),
        "qc_circuit_title": "Circuit quantique complet (8 qubits)",
        "qc_circuit_sub": ("ZZFeatureMap avec reps=2 : encodage (H + P) suivi de deux tours d'intrication "
                           "linéaire entre qubits adjacents."),

        # ── Page 5 · Circuit quantique → onglet Sphère de Bloch ──
        "bl_title": "Sphère de Bloch",
        "bl_subtitle": "Comment la ZZFeatureMap encode la valeur d'une variable clinique comme état quantique |ψ⟩.",
        "bl_what_note": ("<b>Ce qu'est la sphère de Bloch.</b> Un bit classique ne peut valoir "
                         "que 0 ou 1. Un qubit admet en plus n'importe quel mélange des deux, et "
                         "ce mélange ne tient pas dans un seul nombre : il faut une carte. La "
                         "sphère de Bloch est cette carte : chaque état possible d'un qubit est "
                         "un point à la surface d'une sphère de rayon 1. Le pôle nord est "
                         "<b>|0⟩</b> et le pôle sud <b>|1⟩</b> ; entre les deux se trouvent les "
                         "superpositions, et plus la flèche est proche d'un pôle, plus ce "
                         "résultat est probable à la mesure. Ici la valeur clinique se traduit en "
                         "l'angle θ, si bien que déplacer le curseur fait tourner la flèche le "
                         "long d'un méridien, de |0⟩ à |1⟩."),
        "bl_var": "Variable clinique",
        "bl_value": "Valeur ({unidad})",
        "bl_xnorm": "x normalisé",
        "bl_theta": "θ = x_norm·π",
        "bl_alpha": "α (amplitude |0⟩)",
        "bl_beta": "β (amplitude |1⟩)",
        "bl_rad": "rad",
        "bl_note": ("<b>Analogie didactique du principe d'encodage angulaire</b>, et non une réplique du "
                    "circuit. Ici la valeur clinique normalisée sur [0,1] devient l'angle <b>polaire</b> "
                    "θ = x_norm·π, de sorte que le vecteur parcourt le méridien de |0⟩ à |1⟩ et que "
                    "P(|0⟩) varie de 100 % à 0 % : c'est la façon la plus lisible de voir « un nombre "
                    "devient un état ».<br><br>"
                    "La <b>vraie ZZFeatureMap</b> fait autre chose : elle applique H puis "
                    "P(2·x<sub>i</sub>), et une porte de phase après une Hadamard laisse l'état <b>sur "
                    "l'équateur</b> (θ = π/2 fixe, P(|0⟩) = P(|1⟩) = 50 % toujours), en encodant la "
                    "donnée dans l'angle <b>azimutal</b> φ, et non dans le polaire. Elle ne normalise "
                    "pas non plus sur [0,1] : elle utilise directement la valeur mise à l'échelle. Voilà "
                    "pourquoi cette sphère illustre le concept sans reproduire le circuit pas à pas. "
                    "L'intrication (portes P(2·(π−x<sub>i</sub>)·(π−x<sub>j</sub>))) n'est représentable "
                    "que dans l'espace conjoint des 8 qubits (voir Circuit quantique)."),

        # ── Page 5 · onglet Sphère de Bloch → section intrication ──
        "bl_ent_title": "Intrication : trois qubits, un seul état",
        "bl_ent_sub": ("La limite de la sphère ci-dessus. Appliquez les trois portes et regardez ce "
                       "qu'il advient de l'état local de chaque qubit, et de celui de la paire laissée "
                       "en chemin."),
        "bl_ent_intro": ("<b>Là où la sphère de Bloch cesse de servir.</b> Avec un qubit, une sphère "
                         "et une flèche suffisent. Avec plusieurs, la tentation est d'en dessiner une par "
                         "qubit, et pour la plupart des états cela fonctionne. Mais il existe une famille "
                         "d'états où <b>il ne reste aucune flèche à dessiner</b> : l'ensemble possède un "
                         "état parfaitement défini et aucun de ses membres n'en a un séparément. C'est "
                         "cela, l'intrication, et ici elle se construit avec trois portes. Appliquez-les "
                         "et suivez les quatre chiffres de gauche : les trois premiers se brisent à la "
                         "deuxième étape, et à la troisième le quatrième montre quelque chose que deux "
                         "qubits ne permettent même pas de poser."),
        "bl_ent_btn_h": "1 · Hadamard sur q₀",
        "bl_ent_btn_cnot1": "2 · CNOT (contrôle q₀ → q₁)",
        "bl_ent_btn_cnot2": "3 · CNOT (contrôle q₁ → q₂)",
        "bl_ent_btn_reset": "Réinitialiser à |000⟩",
        "bl_ent_step_note": [
            ("<b>Point de départ.</b> Trois qubits, tous en |0⟩, aucune porte appliquée. L'état "
             "conjoint est |000⟩ et n'a encore rien de quantique : il équivaut exactement à trois bits "
             "classiques mis à zéro. Sur la Q-sphere il n'y a qu'un seul nœud, au pôle nord, qui "
             "emporte toute la probabilité."),
            ("<b>Superposition, pas encore d'intrication.</b> La Hadamard laisse q₀ à mi-chemin entre "
             "|0⟩ et |1⟩, tandis que q₁ et q₂ restent fermement en |0⟩ : l'état conjoint est "
             "(|000⟩ + |100⟩)/√2. Les trois qubits restent <b>indépendants</b> : chacun a son propre "
             "état pur et trois sphères de Bloch suffiraient à les décrire. Notez que la longueur du "
             "vecteur local vaut toujours 1 : il y a une flèche à dessiner."),
            ("<b>Une paire de Bell, et un témoin.</b> Le premier CNOT retourne q₁ seulement lorsque q₀ "
             "vaut 1 ; appliqué à une superposition, cela lie les deux résultats en un seul : "
             "(|000⟩ + |110⟩)/√2. C'est ici que la carte se rompt : la longueur du vecteur local de q₀ "
             "vient de tomber à <b>0</b>, le qubit n'est plus en aucun point de sa sphère car séparément "
             "il <b>n'a plus d'état</b>. Et il se passe autre chose, visible seulement parce qu'il y a un "
             "troisième qubit : q₂ est resté DEHORS, à regarder depuis |0⟩, et ce qui est intriqué, c'est "
             "exactement la paire q₀q₁. Sa concurrence affiche <b>1</b>, le maximum."),
            ("<b>État GHZ.</b> Le second CNOT accroche q₂ à la chaîne : (|000⟩ + |111⟩)/√2. Les nœuds "
             "sont partis aux pôles et les deux anneaux du milieu sont vides. Les trois premiers "
             "chiffres ne bougent pas (q₀ reste sans état propre), mais le quatrième s'effondre : la "
             "concurrence de la paire q₀q₁ retombe à <b>0</b>. Intriquer les trois a DÉFAIT le lien de "
             "la paire. Les deux restent corrélés (mesurer l'un prédit l'autre), mais ne sont plus "
             "intriqués : l'intrication d'un GHZ appartient à l'ensemble et n'est <b>pas la somme de "
             "liens deux à deux</b>."),
        ],
        "bl_ent_circuit_title": "Circuit",
        "bl_ent_circuit_alt": "Circuit à trois qubits avec les portes appliquées jusqu'ici",
        "bl_ent_qsphere_title": "Q-sphere de l'état conjoint",
        "bl_ent_kpi": ["Longueur du vecteur local |r| (q₀)", "Pureté Tr(ρ₀²)",
                       "Entropie d'intrication", "Concurrence de la paire q₀q₁"],
        "bl_ent_bits": "bits",
        "bl_ent_hover_amp": "Amplitude :",
        "bl_ent_hover_prob": "Probabilité :",
        "bl_ent_hover_shots": "mesures",
        "bl_ent_meas_title": "Mesure",
        "bl_ent_meas_sub": ("La Q-sphere montre l'état ; ceci montre la seule chose observable. "
                            "Relancez le tirage : la proportion est stable, le compte exact ne l'est pas."),
        "bl_ent_meas_n": "Nombre de mesures",
        "bl_ent_meas_btn": "Simuler les mesures",
        "bl_ent_meas_empty": "Choisissez combien de mesures et appuyez sur « Simuler les mesures ».",
        "bl_ent_meas_yaxis": "Nombre d'occurrences",
        "bl_ent_meas_note": [
            ("Avec les trois qubits en |0⟩, le résultat est <b>000</b> à chaque tir. Il n'y a encore "
             "rien à tirer au sort : c'est le comportement de trois bits classiques."),
            ("<b>000</b> et <b>100</b> sortent à parts égales : q₀ se comporte comme un tirage à pile "
             "ou face, et q₁ et q₂ valent 0 quoi qu'il arrive. Les résultats sont <b>indépendants</b> : "
             "connaître l'un ne dit rien des autres."),
            ("Sortent <b>000</b> et <b>110</b> : q₀ et q₁ donnent toujours la même valeur, mesurer l'un "
             "détermine donc l'autre, tandis que q₂ reste cloué à 0 sans rien en savoir. Deux des trois "
             "sont déjà liés ; le troisième regarde encore de l'extérieur."),
            ("<b>Seuls 000 et 111 sortent</b>, à près de 50 % chacun. Les six barres vides sont le "
             "fait marquant : <b>aucune des six autres combinaisons n'apparaît jamais</b>, pas une fois "
             "sur dix mille tirs. Chaque qubit donne toujours un résultat au hasard, mais les trois "
             "donnent <b>toujours le même</b> : mesurer l'un détermine les deux autres instantanément. "
             "Cette corrélation parfaite, c'est l'intrication vue depuis le laboratoire."),
        ],
        "bl_ent_impl_note": ("<b>Comment c'est calculé.</b> Les huit amplitudes proviennent d'une "
                             "algèbre linéaire <b>exacte</b> sous NumPy (la matrice H⊗I⊗I et les deux "
                             "CNOT appliqués à |000⟩), et non d'une approximation ; la concurrence de "
                             "la paire est la formule de Wootters, qui pour deux qubits donne la valeur "
                             "exacte et non une borne ; et les mesures viennent "
                             "d'un tirage multinomial sur |ψ|², ce que fait un simulateur idéal sans "
                             "bruit. Ce panneau <b>ne charge pas Qiskit</b> : l'environnement déployé "
                             "est Streamlit, NumPy, Plotly et ONNX Runtime, tandis que Qiskit vit dans "
                             "le pipeline Databricks (là où le QSVM est entraîné) et ses figures "
                             "arrivent ici déjà rendues, comme le circuit à 8 qubits de la page Circuit "
                             "quantique. La convention de base est celle des manuels, |q₀q₁q₂⟩ avec q₀ à "
                             "gauche ; Qiskit numérote à l'envers et écrirait « 001 » là où la première "
                             "étape écrit ici « 100 »."),

        # ── Page 5 · onglet Sphère de Bloch → la vraie ZZFeatureMap (8 qubits) ──
        "bl_zz_title": "La vraie ZZFeatureMap : où se produit l'intrication",
        "bl_zz_sub": ("Les mêmes chiffres, maintenant sur les 8 qubits du QSVM du mémoire. Déplacez "
                      "le curseur en haut de la page et regardez quels qubits réagissent."),
        "bl_zz_intro": ("<b>De trois qubits aux huit du modèle.</b> Chaque qubit de la ZZFeatureMap "
                        "porte <b>une variable clinique</b> : q₀ est l'HbA1c, q₁ la glycémie, et "
                        "ainsi de suite jusqu'à l'IMC. Avec 256 amplitudes, il n'y a plus de figure "
                        "de l'état à regarder (ni Q-sphere ni histogramme), mais on peut toujours "
                        "mesurer les <b>deux mêmes grandeurs</b> que dans la section précédente : "
                        "combien d'état propre il reste à chaque qubit, et combien d'information il "
                        "partage avec chacun des autres. Ce n'est plus un exemple de manuel : c'est "
                        "le circuit avec lequel le modèle a été entraîné."),
        "bl_zz_current": "Variable en jeu : <b>{var} = {val} {unidad}</b>. Les sept autres à leur valeur de référence.",
        "bl_zz_r_title": "État propre de chaque qubit",
        "bl_zz_r_xaxis": "|r| : 1 = conserve son état · 0 = totalement intriqué",
        "bl_zz_mi_title": "Information mutuelle entre qubits",
        "bl_zz_mi_cbar": "bits",
        "bl_zz_note": ("<b>Comment lire la matrice.</b> Chaque cellule dit combien d'information deux "
                       "qubits partagent : plus c'est allumé, plus ils sont liés. Et il saute aux yeux "
                       "que la couleur <b>se concentre en une bande le long de la diagonale</b> tandis "
                       "que les coins restent vides. Ce n'est pas un hasard : la ZZFeatureMap utilise "
                       "<code>entanglement=\"linear\"</code>, autrement dit <b>il n'y a de portes "
                       "qu'entre qubits voisins</b>. Avec reps=2, cette corrélation atteint au plus "
                       "quatre maillons de distance ; au-delà elle vaut <b>exactement zéro</b> "
                       "(vérifié sur 300 profils : à une distance ≥ 5 dans la chaîne, 0,0000 bit sans "
                       "une seule exception). La topologie du circuit se dessine toute seule.<br><br>"
                       "Et il y a une seconde chose que l'on voit en déplaçant le curseur du haut : en "
                       "changeant <b>une</b> variable, seuls <b>son qubit et ses voisins immédiats</b> "
                       "bougent : les autres ne changent pas d'une décimale. C'est le même fait vu de "
                       "l'autre côté : le cône de lumière du circuit, en direct."),
        "bl_zz_caveat": ("<b>Un avertissement de lecture, et non des moindres.</b> L'intrication <b>ne "
                         "croît pas avec la valeur clinique</b> : en montant l'HbA1c sur sa plage, le "
                         "|r| du premier qubit fait 0,88 → 1,00 → 0,65 → 0,24 → 0,73 → 0,98 → 0,33. Il "
                         "monte et descend. La raison est que la donnée entre comme un <b>angle</b> de "
                         "phase et que les angles <b>font le tour</b> : deux valeurs cliniques très "
                         "différentes peuvent aboutir à des phases presque identiques. Avec des "
                         "features standardisées, un cas extrême atteint x ≈ 5, et le produit "
                         "2·(π−xᵢ)(π−xⱼ) du terme d'intrication dépasse 2π plusieurs fois. C'est une "
                         "limitation connue de l'encodage angulaire non borné, et il convient de la "
                         "garder à l'esprit avant de lire ces figures comme si elles mesuraient une "
                         "gravité clinique : elles mesurent la géométrie du circuit, pas le risque."),

        # ── Page 6 · Prédicteur en direct ──
        "lp_eyebrow": "Inférence interactive",
        "lp_title": "Prédicteur en direct",
        "lp_subtitle": ("Probabilité qu'un profil clinique corresponde à une personne déjà "
                        "diagnostiquée diabétique : LightGBM sur les 8 variables les plus importantes."),
        "lp_what_note": ("<b>Ce que ce formulaire estime.</b> La cible du pipeline est "
                         "<code>TARGET = (DIQ010 == 1)</code>, la réponse à <i>« un médecin vous a-t-il "
                         "déjà dit que vous aviez du diabète ? »</i>. Le modèle <b>détecte donc un "
                         "diabète déjà diagnostiqué</b> : il ne prédit pas qui va le développer. C'est "
                         "une tâche de détection concomitante, pas de risque prospectif."),
        "lp_real_note": ("<b>Inférence réelle (ONNX).</b> Prédictions de LightGBM et de SVM-RBF via "
                         "<code>onnxruntime</code>, avec le <code>StandardScaler</code> récupéré du "
                         "pipeline Gold. Les 8 variables affichées sont celles de plus grande "
                         "importance clinique ; les 81 features restantes sont fixées à la médiane du "
                         "jeu d'entraînement. Le QSVM n'est pas disponible en temps réel à cause du "
                         "coût O(n²) du noyau quantique : prédire les 1 567 instances du test a pris "
                         "144,5 minutes."),
        "lp_proxy_note": ("⚠ <b>Avertissement technique et clinique.</b> Ce formulaire n'a pas les "
                          "vrais modèles sérialisés (<code>.onnx</code>) connectés : placez "
                          "<code>lgbm_final.onnx</code>, <code>svm_final.onnx</code>, "
                          "<code>scaler_correcto.json</code> et <code>medianas_correctas.json</code> "
                          "dans <code>streamlit/models/</code>. Le score affiché ci-dessous est un "
                          "<b>substitut transparent</b> : une combinaison pondérée par l'importance "
                          "SHAP normalisée, à seules fins de maquette. <b>Ce n'est la sortie d'aucun "
                          "modèle entraîné</b> et cela ne doit pas être cité comme résultat. Le QSVM "
                          "n'est pas davantage disponible en temps réel (coût O(n²) du noyau "
                          "quantique)."),
        "lp_train_range": "Entraînement : {mu} ± {sd} (±3 σ → {lo} à {hi})",
        "lp_extrapolates": "⚠ z = {z} · hors de la plage entraînée : le modèle extrapole",
        "lp_ada": "Critère ADA : &lt; 5,7 normal · 5,7–6,4 prédiabète · ≥ 6,5 diabète",
        "lp_who_model": "le modèle",
        "lp_who_proxy": "le substitut",
        "lp_score_real": "Probabilité d'un diagnostic existant",
        "lp_score_proxy": "Score de maquette (substitut)",
        "lp_cat_low": "faible",
        "lp_cat_mid": "intermédiaire",
        "lp_cat_high": "élevée",
        "lp_interp_low": ("Le profil se situe nettement sous le seuil de décision (50 %) : {quien} le "
                          "classerait comme non diagnostiqué."),
        "lp_interp_mid": "La valeur approche le seuil de décision (50 %) : zone d'incertitude.",
        "lp_interp_high": ("La valeur dépasse le seuil de décision (50 %) : {quien} classerait ce profil "
                           "comme un cas positif."),
        # L'adjectif suit le nom en français, donc {cat} arrive déjà accordé au féminin
        # depuis lp_cat_* et se place après « Compatibilité ».
        "lp_badge": "Compatibilité {cat}",
        "lp_gauge_caption": ("Compatibilité avec un diagnostic existant : faible · intermédiaire · "
                             "élevée · &nbsp;seuil de décision = 50 %"),
        "lp_side_title": "Les deux modèles, côte à côte",
        "lp_side_sub": ("Chaque probabilité se juge au point de coupure de son propre modèle : ils ne "
                        "sont pas interchangeables"),
        "lp_own_threshold": "Son seuil",
        "lp_would_classify": "Classerait comme",
        "lp_positive": "positif",
        "lp_negative": "négatif",
        "lp_disagree": ("<b>Les deux modèles divergent de {dif} sur ce profil.</b> Ils s'accordent aux "
                        "extrêmes (profils clairement sains ou clairement diabétiques) et divergent "
                        "dans la bande intermédiaire, justement là où une estimation serait la plus "
                        "utile. Prenez-le comme un signal d'incertitude, non comme le signe que l'un "
                        "des deux a raison."),
        "lp_curve_title": "Courbe de réponse",
        "lp_curve_var": "Variable à parcourir",
        "lp_curve_yaxis": "Probabilité",
        "lp_curve_thr": "seuil",
        "lp_ada_bands": ["normal", "prédiabète", "diabète"],
        "lp_curve_note": ("Sur cette variable, LightGBM renvoie <b>{n} valeurs distinctes</b> sur les "
                          "{total} positions du curseur : c'est un escalier, pas une rampe. Plus "
                          "grandes marches : {saltos}. Le point marque votre valeur actuelle."),
        "lp_curve_none": "aucune",
        "lp_read_note": ("<b>Comment lire ces résultats.</b> Comme la cible est un diagnostic <i>déjà "
                         "posé</i>, le modèle apprend aussi l'effet du <b>traitement</b>, et pas "
                         "seulement celui de la maladie. Cela inverse le sens clinique de deux "
                         "variables :"
                         '<ul style="margin:8px 0 0; padding-left:20px; line-height:1.7;">'
                         "<li><b>Cholestérol LDL</b> : plus le LDL est élevé, <i>plus faible</i> est la "
                         "probabilité estimée (de 43 % à 18 % en parcourant le curseur). Les personnes "
                         "diagnostiquées sont généralement traitées par statines.</li>"
                         "<li><b>Glycémie à jeun</b> : la réponse a une forme en U : les valeurs très "
                         "basses élèvent l'estimation autant que les hautes, à cause des hypoglycémies "
                         "des patients sous traitement.</li></ul>"
                         '<div style="margin-top:10px;">Aucune des deux ne doit se lire comme un '
                         "facteur de risque modifiable.</div>"),
    },

    # ═══════════════════════════════ ITALIANO ══════════════════════════════
    # Italiano estándar, registro académico. Rige lo mismo que en los demás para lo que
    # NO se traduce: nombres propios, librerías, códigos de variable NHANES y nombres de
    # fichero del repositorio.
    #
    # NOTACIÓN NUMÉRICA: idéntica a la española — coma decimal y punto de millar—, así
    # que las cifras dentro de estas frases se copian tal cual del catálogo español. El
    # espacio antes del signo de porcentaje sigue el mismo criterio del SI que allí: el
    # uso corriente italiano lo omite a menudo, pero esta es una memoria científica y la
    # aplicación ya lo aplica en las otras cuatro lenguas.
    #
    # COMILLAS: las angulares italianas, PEGADAS al texto («testo»), sin el espacio
    # interior que sí lleva el francés.
    #
    # APÓSTROFO: se usa el tipográfico (') y no el recto, que es lo que espera el lector
    # italiano —"dell'informazione", no "dell'informazione" con la comilla de máquina—.
    "it": {
        # ── Navigazione e barra laterale ──
        "nav": ["Panoramica", "Governance", "Risultati", "Analisi SHAP",
                "Circuito quantistico", "Predittore in diretta"],
        "sidebar_expand": "Espandi la barra laterale",
        "sidebar_collapse": "Comprimi la barra laterale",
        "search_label": "Cerca",
        "search_ph": "Cerca nella dashboard o sul web…",
        "search_expand": "Cerca: espande la barra laterale",
        "scroll_top": "Torna su",
        "search_in": "in {p}",
        "search_none": "Nessun risultato nella dashboard.",
        "search_web": "Cerca «{q}» in:",
        "theme_to_dark": "Passa al tema scuro",
        "theme_to_light": "Passa al tema chiaro",
        "lang_es_help": "Vedi l'applicazione in spagnolo",
        "lang_en_help": "Vedi l'applicazione in inglese",
        "lang_de_help": "Vedi l'applicazione in tedesco",
        "lang_fr_help": "Vedi l'applicazione in francese",
        "lang_it_help": "Vedi l'applicazione in italiano",
        "footer_name": "Juan Albornoz C. · Tesi di laurea magistrale 2026",
        "footer_uni": "Universidad Europea de Valencia",
        "footer_name_narrow": "JAC",
        "footer_uni_narrow": "UEV",

        # ── Pagina 1 · Panoramica ──
        "ov_eyebrow": "Framework DataOps + QML",
        "ov_hero_title": ("Integrazione del Quantum Machine Learning in una pipeline DataOps: "
                          "architettura Medallion su Databricks e confronto con modelli classici "
                          "nella predizione clinica"),
        "ov_title": "Panoramica",
        "ov_subtitle": ("Pipeline end-to-end su Databricks CE + AWS S3, con una QSVM quantistica a "
                        "confronto con due baseline classiche, validata su dati clinici reali dello "
                        "studio NHANES (CDC)."),
        "ov_lead": (
            "Questo framework progetta e implementa una pipeline <b>DataOps end-to-end</b> su "
            "<b>Databricks Community Edition</b>, con <b>AWS S3</b> come vero livello di archiviazione "
            "cloud e un'architettura <b>Medallion</b> (Bronze → Silver → Gold) su Delta Lake come "
            "spina dorsale. Come caso d'uso si predice il diabete di tipo 2 a partire dai record dello "
            "studio <b>NHANES</b> (CDC): il dataset non è l'oggetto della ricerca, ma il veicolo per "
            "dimostrare che l'architettura è praticabile, riproducibile e verificabile su dati reali "
            "su larga scala. Il nucleo sperimentale è un <b>confronto triangolato</b> fra LightGBM "
            "(baseline tabellare), una SVM con kernel RBF (ponte strutturale) e una <b>QSVM</b> con "
            "FidelityQuantumKernel in Qiskit, mantenendo identico il classificatore sottostante per "
            "attribuire qualsiasi differenza di prestazioni all'effetto del kernel quantistico."
            " La valutazione chiude il percorso: ogni modello si misura con AUC-ROC, F1, accuracy e "
            "MCC, <b>SHAP</b> indica su LightGBM le 20 variabili che pesano di più nella previsione, "
            "e i due modelli classici si serializzano in <b>ONNX</b> con la portabilità verificata. "
            "Il repository GitHub pubblica i 7 notebook che eseguono quel percorso, e questa stessa "
            "applicazione, distribuita su Streamlit Cloud, ne è l'ultimo anello: la previsione in "
            "diretta e la sua lettura SHAP."),
        "ov_arch_alt": ("Diagramma dell'architettura della pipeline: AWS S3 alimenta Databricks "
                        "Community Edition, dove l'architettura Medallion (Bronze, Silver e Gold) "
                        "sfocia in tre modelli (LightGBM, SVM con kernel RBF e QSVM con Qiskit) e "
                        "nella valutazione con metriche, SHAP e serializzazione ONNX; l'output va "
                        "su GitHub e Streamlit Cloud."),
        # Rótulos del diagrama de arquitectura que abre la página (arquitectura_svg()). Las
        # cifras NO van escritas: llegan por marcador y las pone mil(), que usa el separador
        # de millar del idioma. Lo que no viaja aquí es la geometría ni qué caja va resaltada,
        # que son dibujo y no texto.
        "ov_arch_io": (
            ("AWS S3", "NHANES raw · 27 XPT", "IAM"),
            ("GitHub", "7 notebook · README"),
            ("Streamlit Cloud", "Previsione · SHAP visivo"),
        ),
        "ov_arch_grupos": (
            ("Architettura Medallion", (
                ("Bronze · ingestione raw",
                 "{bronze} righe · 162 col · Delta Lake ACID"),
                ("Silver · qualità",
                 "{silver} righe · 91 col · expectations"),
                ("Gold · feature curate",
                 "89 feature · train {train} / test {test}"),
            )),
            ("Modelli · ML / QML", (
                ("LightGBM",
                 "Baseline tabellare · GOSS · EFB"),
                ("SVM · kernel RBF",
                 "Ponte diretto alla QSVM"),
                ("QSVM · Qiskit",
                 "ZZFeatureMap · FidelityQuantumKernel"),
            )),
            ("Valutazione e serializzazione", (
                ("Metriche",
                 "AUC-ROC · F1 · Accuracy · MCC"),
                ("SHAP",
                 "Spiegabilità LightGBM · top 20"),
                ("Selezione · ONNX",
                 "Portabilità verificata"),
            )),
        ),
        "ov_stats_title": "Statistiche del dataset NHANES",
        "ov_stats_sub": "Tre cicli biennali integrati · pipeline a livelli Bronze → Silver → Gold",
        "ov_stat_bronze": "Record Bronze",
        "ov_stat_silver": "Record Silver",
        "ov_stat_features": "Feature Gold",
        "ov_stat_balance": "Bilanciamento delle classi",
        "ov_medallion_title": "Architettura Medallion",
        "ov_medallion_sub": "Catena del valore del dato (Curry, 2016) applicata livello per livello",
        "ov_layers": [
            ("Bronze", "Ingestione da AWS S3 senza trasformazioni. Preserva la fonte di verità."),
            ("Silver", "Pulizia, imputazione, winsorizzazione e validazione della qualità."),
            ("Gold",   "Scalatura, codifica e partizione stratificata. Pronto per la modellazione."),
        ],
        "ov_goto_gov": "Vedi i controlli di qualità e di lineage  →",
        "ov_target_title": "Distribuzione della variabile target (DIQ010)",
        "ov_target_sub": "Target binarizzato: 1 = diabete diagnosticato, 0 = il resto",
        "ov_pie_no": "Nessun diabete",
        "ov_pie_yes": "Diabete",
        "ov_donut_center": "14 %",
        "ov_donut_caption": "DIABETE",
        "ov_tech_title": "Costruito su",
        "ov_tech_sub": ("Piattaforma, archiviazione e librerie della pipeline, nell'ordine in cui "
                        "intervengono · l'inventario completo, con la motivazione di ogni scelta, "
                        "si trova in Governance"),
        "ov_compare_title": "Confronto triangolato: obiettivo dell'esperimento",
        "ov_compare": [
            ("LightGBM", "Baseline tabellare di riferimento"),
            ("SVM-RBF",  "Ponte strutturale verso la componente quantistica"),
            ("QSVM",     "FidelityQuantumKernel: stesso classificatore, kernel quantistico"),
        ],

        # ── Variabili NHANES (condivise: SHAP, Circuito, Bloch, Predittore) ──
        # I codici NHANES non si traducono mai, solo la loro glossa.
        "var_short": {
            "LBXGH": "HbA1c", "RIDAGEYR": "Età", "LBXGLU": "Glicemia a digiuno",
            "LBDLDL": "Colesterolo LDL", "BMXWAIST": "Circonf. vita",
            "WTINT2YR": "Peso campionario*", "BMXARML": "Lunghezza braccio",
            "BMXLEG": "Lunghezza gamba", "BMXBMI": "IMC",
            "PAD680": "Attività sedentaria", "PAD645": "Attività moderata",
            "PAQ640": "Rafforzamento muscolare", "BMXWT": "Peso corporeo", "LBXIN": "Insulina",
            "INDHHIN2": "Reddito familiare", "DMDYRSUS": "Anni negli USA",
            "BMXARMC": "Circonf. braccio", "PAQ670": "Attività intensa",
            "BPXSY1": "Pressione sistolica", "PAD630": "Attività mod. ricreativa",
            "DMDHHSZE": "Dimensione nucleo (bambini)", "BPXDI1": "Pressione diastolica",
            "LBXTR": "Trigliceridi", "DMDMARTL_1": "Stato civile (coniugato)",
            "DMDMARTL_5": "Stato civile (mai coniugato)", "BPXPLS": "Polso",
            "DMDEDUC2_3": "Istruzione (livello 3)", "SDMVSTRA": "Strato campionario",
            "DMDMARTL_2": "Stato civile (vedovo)", "DMDHHSZB": "Dimensione nucleo (adulti)",
        },
        "var_desc": {
            "LBXGH":      "Emoglobina glicata (HbA1c): glicemia media degli ultimi 2-3 mesi. Marcatore diagnostico primario del diabete (ADA: ≥ 6,5 %).",
            "RIDAGEYR":   "Età del partecipante al momento dell'esame (anni).",
            "LBXGLU":     "Glicemia plasmatica a digiuno: marcatore biochimico del controllo glicemico (mg/dL).",
            "LBDLDL":     "Colesterolo LDL calcolato: la frazione del colesterolo legata al rischio cardiovascolare (mg/dL).",
            "BMXWAIST":   "Circonferenza della vita: adiposità addominale associata all'insulino-resistenza (cm).",
            "WTINT2YR":   "Fattore di ponderazione campionaria dell'intervista NHANES. Artefatto del disegno campionario, non una variabile clinica.",
            "BMXARML":    "Lunghezza del braccio (acromion → olecrano): misura antropometrica (cm).",
            "BMXLEG":     "Lunghezza massima della gamba (ginocchio → suolo): misura antropometrica (cm).",
            "BMXBMI":     "Indice di massa corporea (peso/altezza²): adiposità corporea complessiva (kg/m²).",
            "PAD680":     "Minuti di attività sedentaria al giorno (tempo seduti o sdraiati).",
            "PAD645":     "Minuti settimanali di attività fisica moderata (lavoro + tempo libero).",
            "PAQ640":     "Giorni a settimana con attività di rafforzamento muscolare.",
            "BMXWT":      "Peso corporeo totale (kg).",
            "LBXIN":      "Insulina sierica a digiuno: marcatore di insulino-resistenza (µU/mL).",
            "INDHHIN2":   "Livello di reddito del nucleo familiare (variabile socioeconomica categoriale).",
            "DMDYRSUS":   "Numero di anni di residenza negli Stati Uniti.",
            "BMXARMC":    "Circonferenza media del braccio: misura antropometrica (cm).",
            "PAQ670":     "Minuti settimanali di attività ricreativa intensa.",
            "BPXSY1":     "Pressione arteriosa sistolica, prima misurazione (mmHg).",
            "PAD630":     "Minuti settimanali di attività fisica moderata ricreativa.",
            "DMDHHSZE":   "Composizione del nucleo familiare: numero di bambini nel nucleo.",
            "BPXDI1":     "Pressione arteriosa diastolica, prima misurazione (mmHg).",
            "LBXTR":      "Trigliceridi sierici: marcatore del profilo lipidico (mg/dL).",
            "DMDMARTL_1": "Stato civile = coniugato (variabile dummy dopo la codifica one-hot).",
            "DMDMARTL_5": "Stato civile = mai coniugato (variabile dummy dopo la codifica one-hot).",
            "BPXPLS":     "Polso: frequenza cardiaca a riposo (battiti/min).",
            "DMDEDUC2_3": "Livello di istruzione intermedio (diploma/GED): variabile dummy dopo la codifica one-hot.",
            "SDMVSTRA":   "Strato di varianza del disegno campionario NHANES (variabile metodologica, non clinica).",
            "DMDMARTL_2": "Stato civile = vedovo (variabile dummy dopo la codifica one-hot).",
            "DMDHHSZB":   "Composizione del nucleo familiare: numero di adulti nel nucleo.",
        },
        "qsvm_labels": {
            "LBXGH": "HbA1c", "LBXGLU": "Glicemia a digiuno", "RIDAGEYR": "Età",
            "LBDLDL": "Colesterolo LDL", "BMXWAIST": "Circonf. vita", "LBXIN": "Insulina",
            "BMXLEG": "Lunghezza gamba", "BMXBMI": "IMC",
        },
        "qsvm_units": {"años": "anni"},

        # ── Pagina 2 · Governance ──
        "gov_eyebrow": "Governance · DataOps",
        "gov_title": "Governance e qualità del dato",
        "gov_subtitle": ("I controlli che reggono la pipeline: che cosa si valida, che cosa si scarta e "
                         "perché, che cosa resta registrato e con quali framework. Tutte le cifre "
                         "provengono dagli output eseguiti dei notebook."),
        "gov_tabs": ["Qualità del dato", "Lineage e tracciabilità", "Inventario dei framework"],
        "gov_kpi_expect": "Aspettative superate",
        "gov_kpi_passrate": "Pass rate della suite",
        "gov_kpi_records": "Record validati",
        "gov_kpi_leakage": "Artefatti senza leakage",
        "gov_funnel_title": "Imbuto dei record",
        "gov_funnel_sub": ("Dei {bronze} record di Bronze ne sopravvivono {silver} ai filtri di coorte "
                           "di Silver. Ogni gradino risponde a un criterio esplicito, non a una pulizia "
                           "generica."),
        "gov_hover_records": "Record",
        "gov_hover_dropped": "Scartati",
        "gov_embudo": [
            ("Bronze · 3 cicli uniti",
             "27 file XPT · join per SEQN · 162 colonne comuni ai tre cicli"),
            ("Filtro età ≥ 18 anni", "Restrizione alla popolazione adulta"),
            ("Filtro digiuno · LBXGLU non nullo",
             "Proxy del sottogruppo a digiuno: PHAFSTMN non è coerente fra i cicli"),
            ("Filtro DIQ010 valido",
             "Scarta i codici 7 «non so» e 9 «rifiuta di rispondere», e i valori nulli"),
        ],
        "gov_dropped_title": "Record scartati per filtro",
        "gov_split_label": "Partizione Gold 80/20",
        "gov_suite_title": "Suite di validazione · dataframe-expectations",
        "gov_suite_sub": ("Suite <code>{nombre}</code>, eseguita il {fecha} sui {registros} record di "
                          "Silver in {duracion} secondi. Great Expectations è incompatibile con le "
                          "versioni fissate del runtime serverless: questa è l'alternativa adottata."),
        "gov_expectativas": [
            ("Completezza", "TARGET", "al massimo 0 valori nulli"),
            ("Completezza", "LBXGH", "al massimo 0 valori nulli"),
            ("Completezza", "LBXGLU", "al massimo 0 valori nulli"),
            ("Completezza", "RIDAGEYR", "al massimo 0 valori nulli"),
            ("Completezza", "BMXBMI", "al massimo 0 valori nulli"),
            ("Intervalli clinici", "RIDAGEYR", "minimo fra 18 e 25"),
            ("Intervalli clinici", "RIDAGEYR", "massimo fra 70 e 120"),
            ("Intervalli clinici", "LBXGH", "minimo fra 3,0 e 6,0"),
            ("Intervalli clinici", "LBXGH", "massimo fra 8,0 e 20,0"),
            ("Intervalli clinici", "LBXGLU", "minimo fra 30 e 80"),
            ("Intervalli clinici", "LBXGLU", "massimo fra 150 e 500"),
            ("Intervalli clinici", "BMXBMI", "minimo fra 10,0 e 18"),
            ("Intervalli clinici", "BMXBMI", "massimo fra 40,0 e 80"),
            ("Volume", "DataFrame", "almeno 7.000 righe"),
            ("Volume", "DataFrame", "al massimo 9.000 righe"),
        ],
        "gov_ops_title": "Operazioni di qualità per livello",
        "gov_silver_card": "Silver · pulizia e risanamento",
        "gov_gold_card": "Gold · preparazione alla modellazione",
        "gov_silver_ops": [
            ("Variabili DIQ escluse per leakage", "DIQ050, DIQ070, DIQ160, DIQ170, DIQ172, DIQ180"),
            ("Colonne sparse eliminate", "Soglia di >80 % di valori mancanti"),
            ("Variabili winsorizzate", "Taglio degli outlier a IQR × 3"),
            ("Mancanti dopo l'imputazione", "Da 75.855 a 0 nel dataset SVM/QSVM (mediana + moda)"),
        ],
        "gov_gold_ops": [
            ("Feature dopo la codifica", "One-hot di 5 variabili categoriali su 84 feature (106 colonne con TARGET)"),
            ("Scartate per correlazione", "Soglia r > 0,90 fra coppie di predittori"),
            ("Feature finali", "L'insieme con cui si addestrano i tre modelli"),
            ("Partizione stratificata", "80/20 · 14,03 % di positivi nel train, 14,04 % nel test"),
        ],
        "gov_eff_title": "Feature effettive rispetto alle feature nominali",
        "gov_eff_sub": ("Contato su <code>scaler_correcto.json</code>: {const} delle {total} colonne "
                        "hanno varianza zero e non apportano informazione al modello"),
        "gov_eff_nominal": "Feature nominali",
        "gov_eff_const": "Costanti (varianza = 0)",
        "gov_eff_effective": "Feature effettive",
        "gov_eff_note": ("È l'effetto collaterale della winsorizzazione IQR × 3 di Silver, applicata "
                         "anche a variabili categoriali codificate numericamente (risposte 1/2, lingua "
                         "dell'intervista, codici 7 e 9). Quando più del 75 % del campione risponde allo "
                         "stesso modo, il taglio riduce la colonna a un unico valore. Le più tagliate "
                         "nel notebook 02 (PAQ635, PAQ650, PAQ605, DMDHHSZA, DMDCITZN, SIALANG) sono "
                         "esattamente quelle che qui risultano costanti."),
        "gov_lin_title": "Tracciabilità senza MLflow",
        "gov_lin_sub": "Il vincolo che condiziona di più l'architettura della pipeline, e la sua mitigazione.",
        "gov_lin_limit_title": "Limitazione",
        "gov_lin_limit_body": ("L'integrazione nativa di <b>MLflow</b> è disabilitata in Databricks "
                               "Serverless gratuito. Qualsiasi chiamata a <code>mlflow.start_run()</code> "
                               "o <code>mlflow.log_metric()</code> produce errori di autenticazione: non "
                               "c'è registrazione di esperimenti, metriche né artefatti."),
        "gov_lin_mit_title": "Mitigazione · doppio meccanismo",
        "gov_lin_mit_body": ("<b>Transaction log di Delta Lake</b>: ogni scrittura genera un record "
                             "ACID con versione, timestamp e metriche di operazione.<br><br>"
                             "<b>CSV di metriche per modello</b>: ogni notebook rende persistenti i "
                             "propri risultati in Unity Catalog Volumes, e le figure li leggono da lì "
                             "invece di portarli scritti a mano."),
        "gov_delta_title": "Cronologia Delta · livello Gold",
        "gov_delta_sub": ("Le sei versioni più recenti fra le dieci registrate. Delta elimina le "
                          "precedenti dopo 168 h di conservazione: comportamento atteso, non un guasto "
                          "della pipeline."),
        "gov_delta_cols": ["Versione", "Timestamp", "Operazione", "Righe", "Dimensione"],
        "gov_chain_title": "Catena di custodia contro la fuga di informazione",
        "gov_chain_sub": ("Quattro barriere concatenate. La terza non scarta nessuna colonna, ed è "
                          "esattamente ciò che si vuole vedere: la prova che le precedenti hanno fatto "
                          "il loro lavoro."),
        "gov_leakage": [
            ("Esclusione in Silver",
             "Si eliminano 6 variabili DIQ di trattamento e follow-up prima della winsorizzazione: "
             "sono una conseguenza della diagnosi, non predittori di essa."),
            ("Verifica incrociata",
             "Si controlla che nessuna DIQ sopravviva nei 2 Parquet di Silver né nei 13 di Gold. "
             "Risultato: 15/15 artefatti puliti."),
            ("Filtro difensivo della QSVM",
             "Seconda barriera prima della selezione con Random Forest. Non scarta nessuna colonna "
             "(89 su 89 passano), proprio la prova che la prima ha funzionato."),
            ("Guardia sui pesi campionari",
             "Ferma la pipeline se compare un peso campionario diverso da quello noto. WTINT2YR "
             "arriva davvero alla modellazione ed è documentato nella decisione 10."),
        ],
        "gov_scaler_card": "Scalatura senza fuga statistica",
        "gov_scaler": [
            ("Adattamento", "Solo sul train", "fit_transform sul train · transform sul test"),
            ("Colonne valutate", "66", "Con varianza > 0"),
            ("Colonne costanti", "23", "Varianza 0 · vedi decisione 08"),
            ("Media ≈ 0 · dev. ≈ 1", "Verificato", "Assert su tutte le colonne con dispersione"),
        ],
        "gov_scaler_note": ("Lo <b>StandardScaler</b> si adatta esclusivamente sul <b>train</b>: "
                            "<code>fit_transform</code> in addestramento e <code>transform</code> nel "
                            "test. Se si adattasse sull'insieme completo, la media e la deviazione "
                            "standard del test filtrerebbero nel preprocessing e le metriche "
                            "risulterebbero ottimistiche. La selezione delle 8 variabili della QSVM "
                            "segue la stessa regola: il Random Forest si addestra solo su "
                            "<code>X_train_svm_scaled</code>.<br><br>Il filtro di correlazione, invece, "
                            "<b>viene</b> calcolato prima di partizionare. È documentato e accettato "
                            "nella decisione 09."
                            "<br><br>La verifica si esegue, non si dichiara: sulle 66 colonne con dispersione esige "
                            "|media| &lt; 0,01 e una deviazione fra 0,90 e 1,10. I parametri adattati ("
                            "<code>mean_</code> e <code>scale_</code>) finiscono in "
                            "<code>scaler_correcto.json</code>, il file che carica il Predittore in diretta: la "
                            "scala del train non si ricalcola mai."),
        "gov_e2e_title": "Verifica end-to-end rispetto ai modelli addestrati",
        "gov_e2e_missing": ('<b style="color:{color};">Non verificato.</b> Il set di test non è nel '
                            "repository, quindi la dashboard non può controllare da sola che il suo "
                            "percorso di inferenza riproduca ciò che hanno prodotto i modelli "
                            "addestrati. Per chiudere il punto, esegui le due celle di "
                            "<code>notebooks/INSTRUCCIONES_exportar_golden_set.md</code> e copia "
                            "<code>golden_lgbm.npz</code> e <code>golden_svm.npz</code> in "
                            "<code>streamlit/models/</code>. Finché mancano, questa pagina non afferma "
                            "nulla che non abbia potuto verificare."),
        "gov_e2e_unavailable": "non disponibile",
        "gov_e2e_ok_val": "{n} righe · diff. max {dif}",
        "gov_e2e_bad_val": "DISCORDA · diff. max {dif}",
        "gov_e2e_scaled": "scala e chiama il modello ONNX",
        "gov_e2e_raw": "chiama il modello ONNX senza scalare",
        "gov_e2e_path": "La dashboard {accion}",
        "gov_e2e_ok_title": "✓ Percorso di inferenza verificato",
        "gov_e2e_fail_title": "⚠ Il percorso di inferenza non riproduce i modelli",
        "gov_e2e_note": ("Ogni riga del <i>golden set</i> è un'istanza reale del test accompagnata dalla "
                         "probabilità restituita dal modello addestrato nel suo notebook. La dashboard "
                         "la fa passare per il proprio percorso (vettore grezzo, scalatura solo per la "
                         "SVM, conversione in <code>float32</code>, sessione ONNX e lettura del tensore "
                         "di output) e confronta. Tolleranza {tol}; il rumore atteso lavorando in "
                         "<code>float32</code> è dell'ordine di 10⁻⁷."),
        "gov_stack_title": "Framework per livello",
        "gov_stack_sub": ("Il primo distintivo di ogni scheda è il framework che regge il livello; gli "
                          "altri lo accompagnano."),
        "gov_stack": [
            ("ingestione",
             "boto3 sostituisce spark.conf, bloccato in Serverless (decisione 01). Tre assert di "
             "integrità: 27/27 file, il join per SEQN non duplica righe, e Delta coincide con "
             "pandas."),
            ("qualità",
             "Il framework di qualità di questa tesi. Great Expectations è incompatibile con "
             "l'ambiente (decisione 03). Suite di 15 aspettative su 3 dimensioni, con evidenza "
             "resa persistente in CSV."),
            ("preparazione",
             "Scalatura adattata solo sul train, partizione stratificata con seme fisso ed "
             "esportazione del contratto di serving (scaler e mediane in JSON)."),
            ("modello",
             "Interpretabilità esatta tramite algoritmo polinomiale sulle 1.567 istanze di test, "
             "e verifica che il modello ONNX riproduca il PKL al 100 %."),
            ("modello",
             "SHAP agnostico rispetto al modello, al costo di ore: calcolato una volta su 200 "
             "istanze e reso persistente su disco per riutilizzarlo."),
            ("modello",
             "Nessun supporto ONNX: il formato non ammette operazioni quantistiche "
             "(decisione 05). La tracciabilità ricade su un CSV di metriche con i 14 campi di "
             "configurazione."),
        ],
        "gov_dec_title": "Registro delle decisioni",
        "gov_dec_sub": ("Le undici limitazioni documentate in TECHNICAL_NOTES, con la relativa "
                        "mitigazione. Tre condizionano l'architettura, sei sono accettate e documentate "
                        "senza correzione (perché correggerle invaliderebbe i risultati già ottenuti) "
                        "e due restano risolte senza residui."),
        "gov_dec_tags": {"critical": "Architettura", "warning": "Accettata", "good": "Risolta"},
        "gov_dec_problem": "Problema · ",
        "gov_dec_solution": "Soluzione adottata · ",
        "gov_decisiones": [
            ("spark.conf bloccato in Serverless",
             "La configurazione delle credenziali AWS tramite spark.conf.set restituisce "
             "CONFIG_NOT_AVAILABLE, il meccanismo standard per collegare Spark a S3.",
             "boto3 come client alternativo. S3 resta l'archiviazione di origine e Unity Catalog "
             "Volumes il livello di elaborazione."),
            ("MLflow bloccato in Serverless",
             "L'integrazione nativa di MLflow è disabilitata nel piano gratuito: non c'è "
             "registrazione di esperimenti, metriche né artefatti.",
             "Doppio meccanismo sostitutivo: i transaction log di Delta Lake forniscono versione, "
             "timestamp e metriche di operazione; e ogni notebook rende persistenti le proprie "
             "metriche in CSV."),
            ("Great Expectations incompatibile",
             "Richiede una combinazione pandas/numpy che confligge con le versioni fissate del "
             "runtime serverless (pandas 1.5.3 / numpy 1.23.5).",
             "dataframe-expectations 0.7.0 come alternativa compatibile. 15 aspettative su Silver in "
             "tre dimensioni. Risultato 15/15, pass rate 1,0."),
            ("QSVM · costo computazionale O(n²)",
             "Sulle 6.264 istanze di train, la matrice del kernel richiederebbe circa 39 milioni di "
             "valutazioni del circuito. Con 1.500 il kernel esaurisce la memoria.",
             "Addestramento su un campione stratificato di 500 istanze (~22 min) preservando il "
             "rapporto 86/14. La valutazione usa invece il test completo, perché le metriche restino "
             "confrontabili."),
            ("QSVM · nessun supporto ONNX nativo",
             "Il formato ONNX non ammette operazioni quantistiche: né skl2onnx né onnxmltools "
             "possono serializzare un kernel basato sulla simulazione di stati.",
             "Serializzazione con joblib. Il modello richiede l'ambiente Qiskit per l'inferenza, "
             "perciò la QSVM non entra nel Predittore in diretta."),
            ("Versioni di Qiskit non fissabili",
             "Il file immutable_package_constraints.txt di Databricks blocca l'installazione di "
             "versioni specifiche, quindi non c'è riproducibilità esatta di versione.",
             "La pipeline gira con le versioni dell'ambiente (2.5.0 / 0.9.0 / 0.4.0), la cui API è "
             "compatibile, e restano registrate da una verifica esplicita all'inizio "
             "dell'esecuzione."),
            ("Perdita di variabili per durata della sessione",
             "Le operazioni lunghe (22 min di addestramento, 132 di predizione) possono esaurire la "
             "sessione serverless e portarsi via le variabili in memoria.",
             "Persistenza immediata dopo ogni operazione costosa e modalità TRAINING_MODE che "
             "ricarica da disco nelle esecuzioni successive."),
            ("Winsorizzazione applicata a categoriali codificate",
             "NHANES codifica numericamente molte categoriali. Se più del 75 % condivide un valore, "
             "IQR = 0, i limiti collassano e clip() trasforma la variabile in una costante. "
             "10 colonne sono collassate così.",
             "Documentato senza modificare: correggerlo altererebbe Silver, Gold e i tre modelli. Le "
             "colonne costanti non introducono distorsioni (il modello non ne estrae segnale), ma "
             "si perde informazione. La correzione è indicata come lavoro futuro."),
            ("Correlazione calcolata prima di partizionare",
             "Il filtro r > 0,90 si calcola sul dataset completo, quindi le 16 colonne scartate "
             "vengono decise usando anche le osservazioni di test.",
             "Documentato senza modificare. Non influisce né sulla scalatura né sulla selezione di "
             "feature della QSVM, entrambe adattate solo sul train, ma la selezione smette di essere "
             "rigorosamente cieca rispetto al test."),
            ("Peso campionario WTINT2YR fra le feature",
             "Il join intraciclo duplica WTSAF2YR su tre colonne. WTINT2YR non è nell'elenco di "
             "esclusione e sopravvive al filtro di correlazione: è una delle 89 feature.",
             "Documentato senza modificare, con l'aggiunta di un assert che rileva la comparsa di "
             "QUALSIASI altro peso. Un peso campionario non è una variabile clinica: non rivela il "
             "target, ma lascia che il modello si appoggi al disegno dell'indagine."),
            ("La QSVM serializzata non è ricaricabile fra versioni",
             "Il pickle si porta dietro la ZZFeatureMap con le sue ParameterExpression. Se Qiskit "
             "cambia versione, la deserializzazione fallisce, e Serverless aggiorna senza preavviso.",
             "Il caricamento è avvolto in try/except: se fallisce, TRAINING_MODE passa a True e il "
             "notebook riaddestra invece di interrompersi. Resta operativo in tutti e tre gli "
             "scenari possibili."),
        ],
        "gov_footer_note": ("Le cifre di questa pagina provengono dagli output eseguiti dei notebook del "
                            "repository e da <code>TECHNICAL_NOTES.md</code>; nessuna è stimata. "
                            "L'applicazione non può consultarle in diretta perché Streamlit Community "
                            "Cloud accede solo al repository, non a Unity Catalog Volumes.<br><br>"
                            "Riepilogo della suite di qualità: <b>{fuente}</b>."),
        "gov_suite_src_csv": "letto da validacion_silver_dfe.csv",
        "gov_suite_src_nb": "valori verificati dal notebook",

        # ── Pagina 3 · Risultati ──
        "res_eyebrow": "Confronto triangolato",
        "res_title": "Risultati",
        "res_subtitle": "LightGBM vs. SVM-RBF vs. QSVM sullo stesso set di test ({n} istanze).",
        "res_threshold": "Soglia",
        "res_thr_label": {"lightgbm": "p ≥ {v}", "svm_rbf": "p ≈ {v}", "qsvm": "df > 0"},
        "res_thr_src": {"lightgbm": "predict_proba()[:,1] >= 0.5",
                        "svm_rbf": "SVC.predict() · segno di decision_function",
                        "qsvm": "decision_function > 0 (non è una probabilità)"},
        "res_reconciled": ('<span style="color:{color}; font-weight:600;">✓ Riconciliate</span>: le '
                           "quattro metriche dei tre modelli sono state ricalcolate dagli score per "
                           "istanza e coincidono con quelle pubblicate."),
        "res_unreconciled": '<span style="color:{color}; font-weight:600;">⚠ Non riconciliate</span>: {fallos}',
        "res_no_scores": "score non disponibili",
        "res_threshold_note": ("<b>I tre modelli sono misurati a soglie diverse.</b> Ciascuno usa il "
                               "proprio punto di taglio naturale: LightGBM "
                               "<code>predict_proba ≥ 0,50</code>; SVM-RBF il segno di "
                               "<code>decision_function</code>, che sulla scala di probabilità salvata "
                               "equivale a ≈ 0,22; QSVM <code>decision_function &gt; 0</code>, che non è "
                               "una probabilità. Ogni matrice si riproduce esattamente alla propria "
                               "soglia, ma <b>solo l'AUC-ROC è confrontabile fra modelli</b>: è l'unica "
                               "delle quattro metriche indipendente dal punto di taglio. Come "
                               "riferimento, la SVM-RBF valutata a 0,50 come LightGBM darebbe accuracy "
                               "0,9190 ma solo 131 veri positivi invece di 172."),
        "res_roc_title": "Curve ROC",
        "res_roc_sub_real": ("Curve empiriche reali, punto per punto sulle 1.567 istanze del test (gli "
                             "stessi score che danno l'AUC della tesi)."),
        "res_roc_sub_synth": ("AUC esatta · forma ricostruita a partire dall'AUC dove mancano gli score "
                              "per istanza."),
        "res_cm_title": "Matrici di confusione",
        "res_cm_sub": ("Valori verificati rispetto al classification report di ogni modello, e "
                       "ricalcolati dagli score per istanza. Ogni matrice corrisponde alla soglia "
                       "indicata nella sua scheda"),
        "res_cm_pred_no": "Prev.<br>Nessun diabete",
        "res_cm_pred_yes": "Prev.<br>Diabete",
        "res_cm_real_no": "Reale<br>Nessun diab.",
        "res_cm_real_yes": "Reale<br>Diabete",
        "res_cm_tags": {"tn": "VN", "fp": "FP", "fn": "FN", "tp": "VP"},
        "res_metrics_title": "Confronto delle metriche",
        "res_metrics_sub": ("Le quattro metriche si applicano sulle 1.567 istanze. Accuracy, MCC e "
                            "F1-macro penalizzano davvero lo sbilanciamento delle classi, ma dipendono "
                            "dalla soglia, e ogni modello usa la propria: confronta con cautela tutto "
                            "ciò che non sia l'AUC-ROC"),
        "res_metric_desc": {
            "auc": "Area sotto la curva ROC: capacità di separare diabete e non-diabete. 0,5 = caso, 1 = perfetto.",
            "f1_macro": "Media armonica di precisione e richiamo mediata per classe (non pesata). Penalizza lo sbilanciamento.",
            "accuracy": "Proporzione di predizioni corrette totali. Con classi sbilanciate può riflettere solo la classe maggioritaria.",
            "mcc": "Coefficiente di correlazione di Matthews: qualità complessiva robusta allo sbilanciamento. 0 = caso, 1 = perfetto.",
        },
        "res_qsvm_note": ("<b>Nota sull'esperimento QSVM.</b> La QSVM è stata addestrata su un campione "
                          "stratificato di 500 istanze (costo O(n²) del kernel quantistico) e valutata "
                          "sulle 1.567 del test completo. AUC-ROC = 0,5493 indica che il modello supera "
                          "a malapena la classificazione casuale: richiamo ≈ 0 per la classe diabete "
                          "(1 su 220), e accuracy = 0,8602 che riflette solo la proporzione della classe "
                          "maggioritaria. L'MCC ≈ 0 conferma l'assenza di reale capacità predittiva."),

        # ── Pagina 4 · Analisi SHAP ──
        "sh_eyebrow": "Interpretabilità",
        "sh_title": "Analisi SHAP",
        "sh_subtitle": ("Importanza globale delle variabili: TreeExplainer (LightGBM) vs. "
                        "KernelExplainer (SVM-RBF)."),
        "sh_tabs": ["LightGBM · TreeExplainer", "SVM-RBF · KernelExplainer"],
        "sh_hint": "Passa il cursore su ogni barra per vedere il significato della variabile. {nota}",
        "sh_sample_lgbm": "Valori esatti (algoritmo polinomiale) sulle 1.567 istanze del test.",
        "sh_sample_svm": ("Valori approssimati per campionamento: sfondo di 100 istanze, contributi su "
                          "200 istanze di test."),
        "sh_note_lgbm": ("<b>LBXGH (HbA1c)</b> domina con ampio margine (SHAP medio = 1,1243), coerente "
                         "con il suo ruolo di marcatore diagnostico primario del diabete di tipo 2 "
                         "(ADA: HbA1c ≥ 6,5 %). <b>RIDAGEYR (età, 0,4654)</b> riflette l'aumento della "
                         "prevalenza con l'età. <b>LBXGLU</b> e <b>LBDLDL</b> completano il blocco "
                         "biochimico. <b>WTINT2YR</b> (posizione 6) è un artefatto del disegno "
                         "campionario NHANES, non una variabile clinica."),
        "sh_note_svm": ("La classifica della SVM-RBF coincide con LightGBM sulle variabili dominanti "
                        "(<b>LBXGH</b>, <b>LBXGLU</b>, <b>LBDLDL</b>, <b>RIDAGEYR</b>), il che rafforza "
                        "la validità clinica del risultato rendendolo indipendente dall'algoritmo e "
                        "metodologicamente più robusto. KernelExplainer tratta il modello come una "
                        "scatola nera, applicabile a qualsiasi classificatore."),
        "sh_fig_lgbm_title": "SHAP Summary Plot · LightGBM (Figura 27)",
        "sh_fig_lgbm_cap": ("Ogni punto è un'istanza del test; il colore indica il valore della "
                            "variabile (rosso alto, blu basso) e la posizione orizzontale il suo impatto "
                            "sulla predizione. LBXGH e RIDAGEYR dominano il modello."),
        "sh_fig_svm_title": "SHAP Summary Plot · SVM-RBF (Figura 31)",
        "sh_fig_svm_cap": ("Ogni punto è un'istanza; colore = valore della variabile, posizione = "
                           "impatto. KernelExplainer su 200 istanze del test."),

        # ── Pagina 5 · Circuito quantistico ──
        "qc_eyebrow": "Componente quantistica",
        "qc_title": "Circuito quantistico",
        "qc_subtitle": ("Configurazione della ZZFeatureMap e del FidelityQuantumKernel implementati in "
                        "Qiskit su Databricks CE."),
        "qc_tabs": ["Circuito ZZFeatureMap", "Sfera di Bloch"],
        "qc_specs": ["Qubit (feature_dimension)", "Ripetizioni (reps)", "Entanglement", "Versione di Qiskit"],
        "qc_how_title": "Come funziona",
        "qc_how_p1": ("La <b>ZZFeatureMap</b> codifica ciascuna delle 8 variabili cliniche come un "
                      "angolo di fase (porta P) su un qubit indipendente, dopo aver creato la "
                      "sovrapposizione con porte di Hadamard. Il suo elemento distintivo è "
                      "l'<b>entanglement</b> fra coppie di qubit mediante porte che dipendono dal "
                      "prodotto incrociato di due variabili, correlazioni che il kernel RBF classico "
                      "non può rappresentare."),
        "qc_how_p2": ("Il <b>FidelityQuantumKernel</b> misura la somiglianza fra due pazienti come la "
                      "fedeltà fra i loro stati quantistici: <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. "
                      "L'implementazione usa <code>StatevectorSampler</code> e simula lo stato esatto "
                      "senza rumore: risultati deterministici e riproducibili."),
        "qc_feat_title": "8 feature selezionate (Random Forest)",
        "qc_xaxis": "Importanza RF",
        "qc_train_title": "Addestramento e valutazione",
        "qc_tstats": ["Istanze di addestramento", "Tempo di addestramento", "Istanze di test",
                      "Tempo di inferenza", "Vettori di supporto"],
        "qc_note": ("Per il costo O(n²) del kernel quantistico, l'addestramento è stato limitato a un "
                    "campione stratificato di 500 istanze (il limite operativo di Databricks CE "
                    "serverless si colloca intorno a 500-1.000). La valutazione è stata fatta sul test "
                    "completo (1.567 istanze) a lotti di 100, con un tempo totale di predizione di "
                    "144,5 minuti."),
        "qc_circuit_title": "Circuito quantistico completo (8 qubit)",
        "qc_circuit_sub": ("ZZFeatureMap con reps=2: codifica (H + P) seguita da due giri di "
                           "entanglement lineare fra qubit adiacenti."),

        # ── Pagina 5 · Circuito quantistico → scheda Sfera di Bloch ──
        "bl_title": "Sfera di Bloch",
        "bl_subtitle": "Come la ZZFeatureMap codifica il valore di una variabile clinica come stato quantistico |ψ⟩.",
        "bl_what_note": ("<b>Che cos'è la sfera di Bloch.</b> Un bit classico può valere solo 0 "
                         "o 1. Un qubit ammette in più qualsiasi miscela dei due, e quella "
                         "miscela non entra in un solo numero: serve una mappa. La sfera di Bloch "
                         "è quella mappa: ogni stato possibile di un qubit è un punto sulla "
                         "superficie di una sfera di raggio 1. Il polo nord è <b>|0⟩</b> e il polo "
                         "sud <b>|1⟩</b>; in mezzo stanno le sovrapposizioni, e più la freccia si "
                         "avvicina a un polo, più quel risultato è probabile alla misura. Qui il "
                         "valore clinico si traduce nell'angolo θ, così muovere il cursore fa "
                         "ruotare la freccia lungo un meridiano, da |0⟩ a |1⟩."),
        "bl_var": "Variabile clinica",
        "bl_value": "Valore ({unidad})",
        "bl_xnorm": "x normalizzato",
        "bl_theta": "θ = x_norm·π",
        "bl_alpha": "α (ampiezza |0⟩)",
        "bl_beta": "β (ampiezza |1⟩)",
        "bl_rad": "rad",
        "bl_note": ("<b>Analogia didattica del principio di codifica angolare</b>, non una replica del "
                    "circuito. Qui il valore clinico normalizzato su [0,1] diventa l'angolo "
                    "<b>polare</b> θ = x_norm·π, così il vettore percorre il meridiano da |0⟩ a |1⟩ e "
                    "P(|0⟩) varia dal 100 % allo 0 %: è il modo più leggibile di vedere «un numero "
                    "diventa uno stato».<br><br>"
                    "La <b>vera ZZFeatureMap</b> fa qualcosa di diverso: applica H e poi "
                    "P(2·x<sub>i</sub>), e una porta di fase dopo una Hadamard lascia lo stato "
                    "<b>sull'equatore</b> (θ = π/2 fisso, P(|0⟩) = P(|1⟩) = 50 % sempre) "
                    "codificando il dato nell'angolo <b>azimutale</b> φ, non nel polare. Non "
                    "normalizza nemmeno su [0,1]: usa direttamente il valore scalato. Per questo "
                    "questa sfera illustra il concetto, ma non riproduce passo per passo il circuito. "
                    "L'entanglement (porte P(2·(π−x<sub>i</sub>)·(π−x<sub>j</sub>))) è rappresentabile "
                    "solo nello spazio congiunto degli 8 qubit (vedi Circuito quantistico)."),

        # ── Pagina 5 · scheda Sfera di Bloch → sezione entanglement ──
        "bl_ent_title": "Entanglement: tre qubit, un solo stato",
        "bl_ent_sub": ("Il limite della sfera qui sopra. Applica le tre porte e guarda che cosa "
                       "succede allo stato locale di ciascun qubit, e a quello della coppia che resta "
                       "per strada."),
        "bl_ent_intro": ("<b>Dove la sfera di Bloch smette di servire.</b> Con un qubit bastano una "
                         "sfera e una freccia. Con più di uno, la tentazione è disegnare una sfera per "
                         "qubit, e per la maggior parte degli stati funziona. Ma esiste una famiglia di "
                         "stati in cui <b>non resta nessuna freccia da disegnare</b>: l'insieme ha uno "
                         "stato perfettamente definito e nessuno dei suoi membri ce l'ha separatamente. "
                         "Questo è l'entanglement, e qui si costruisce con tre porte. Applicale e segui "
                         "le quattro cifre a sinistra: le prime tre si rompono al secondo passo, e al "
                         "terzo la quarta mostra qualcosa che con due qubit non si può nemmeno porre."),
        "bl_ent_btn_h": "1 · Hadamard su q₀",
        "bl_ent_btn_cnot1": "2 · CNOT (controllo q₀ → q₁)",
        "bl_ent_btn_cnot2": "3 · CNOT (controllo q₁ → q₂)",
        "bl_ent_btn_reset": "Reimposta a |000⟩",
        "bl_ent_step_note": [
            ("<b>Punto di partenza.</b> Tre qubit, tutti in |0⟩, nessuna porta applicata. Lo stato "
             "congiunto è |000⟩ e non ha ancora nulla di quantistico: equivale esattamente a tre bit "
             "classici messi a zero. Nella Q-sphere c'è un unico nodo, al polo nord, che si prende "
             "tutta la probabilità."),
            ("<b>Sovrapposizione, ancora senza entanglement.</b> La Hadamard lascia q₀ a metà strada "
             "fra |0⟩ e |1⟩, mentre q₁ e q₂ restano saldi in |0⟩: lo stato congiunto è "
             "(|000⟩ + |100⟩)/√2. I tre qubit restano <b>indipendenti</b>: ciascuno ha il proprio "
             "stato puro e tre sfere di Bloch basterebbero a descriverli. Nota che la lunghezza del "
             "vettore locale vale ancora 1: c'è una freccia da disegnare."),
            ("<b>Una coppia di Bell, e un testimone.</b> Il primo CNOT ribalta q₁ solo quando q₀ vale "
             "1; applicato a una sovrapposizione, questo lega i due esiti in uno solo: "
             "(|000⟩ + |110⟩)/√2. È qui che la mappa si rompe: la lunghezza del vettore locale di q₀ è "
             "appena caduta a <b>0</b>, il qubit non è più in nessun punto della sua sfera perché "
             "separatamente <b>non ha più stato</b>. E succede anche altro, visibile solo perché c'è un "
             "terzo qubit: q₂ è rimasto FUORI, a guardare da |0⟩, e ciò che è in entanglement è "
             "esattamente la coppia q₀q₁. La sua concorrenza segna <b>1</b>, il massimo."),
            ("<b>Stato GHZ.</b> Il secondo CNOT aggancia q₂ alla catena: (|000⟩ + |111⟩)/√2. I nodi "
             "sono andati ai poli e i due anelli di mezzo sono rimasti vuoti. Le prime tre cifre non si "
             "muovono (q₀ resta senza stato proprio), ma la quarta crolla: la concorrenza della coppia "
             "q₀q₁ torna a <b>0</b>. Mettere in entanglement tutti e tre ha DISFATTO il legame della "
             "coppia. I due restano correlati (misurarne uno predice l'altro), ma non più in "
             "entanglement: l'entanglement di un GHZ è dell'insieme e <b>non è la somma di legami a "
             "due a due</b>."),
        ],
        "bl_ent_circuit_title": "Circuito",
        "bl_ent_circuit_alt": "Circuito a tre qubit con le porte applicate finora",
        "bl_ent_qsphere_title": "Q-sphere dello stato congiunto",
        "bl_ent_kpi": ["Lunghezza del vettore locale |r| (q₀)", "Purezza Tr(ρ₀²)",
                       "Entropia di entanglement", "Concorrenza della coppia q₀q₁"],
        "bl_ent_bits": "bit",
        "bl_ent_hover_amp": "Ampiezza:",
        "bl_ent_hover_prob": "Probabilità:",
        "bl_ent_hover_shots": "misure",
        "bl_ent_meas_title": "Misura",
        "bl_ent_meas_sub": ("La Q-sphere mostra lo stato; questo mostra l'unica cosa osservabile. "
                            "Ripeti il lancio: la proporzione è stabile, il conteggio esatto no."),
        "bl_ent_meas_n": "Numero di misure",
        "bl_ent_meas_btn": "Simula le misure",
        "bl_ent_meas_empty": "Scegli quante misure e premi «Simula le misure».",
        "bl_ent_meas_yaxis": "Volte ottenuto",
        "bl_ent_meas_note": [
            ("Con tutti e tre i qubit in |0⟩ il risultato è <b>000</b> a ogni lancio. Non c'è ancora "
             "nulla da sorteggiare: è il comportamento di tre bit classici."),
            ("Escono <b>000</b> e <b>100</b> in parti uguali: q₀ si comporta come una monetina, e q₁ e "
             "q₂ valgono 0 qualunque cosa accada. Gli esiti sono <b>indipendenti</b>: conoscerne uno "
             "non dice nulla degli altri."),
            ("Escono <b>000</b> e <b>110</b>: q₀ e q₁ danno sempre lo stesso valore, quindi misurarne "
             "uno determina l'altro, mentre q₂ resta inchiodato a 0 senza accorgersi di nulla. Due dei "
             "tre sono già legati; il terzo guarda ancora da fuori."),
            ("<b>Escono solo 000 e 111</b>, vicino al 50 % ciascuno. Le sei barre vuote sono il dato: "
             "<b>nessuna delle altre sei combinazioni compare mai</b>, nemmeno una volta su diecimila "
             "lanci. Ogni qubit continua a dare un esito casuale, ma i tre danno <b>sempre lo "
             "stesso</b>: misurarne uno determina gli altri due all'istante. Quella correlazione "
             "perfetta è l'entanglement visto dal laboratorio."),
        ],
        "bl_ent_impl_note": ("<b>Come è calcolato.</b> Le otto ampiezze vengono da algebra lineare "
                             "<b>esatta</b> in NumPy (la matrice H⊗I⊗I e i due CNOT applicati a |000⟩), "
                             "non da un'approssimazione; la concorrenza della coppia è la formula di "
                             "Wootters, che per due qubit dà il valore esatto e non un limite; e le "
                             "misure vengono da un campionamento multinomiale "
                             "su |ψ|², che è ciò che fa un simulatore ideale senza rumore. Il pannello "
                             "<b>non carica Qiskit</b>: l'ambiente distribuito è Streamlit, NumPy, "
                             "Plotly e ONNX Runtime, mentre Qiskit vive nella pipeline di Databricks "
                             "(dove si addestra la QSVM) e le sue figure arrivano qui già "
                             "renderizzate, come il circuito a 8 qubit della pagina Circuito "
                             "quantistico. La convenzione della base è quella dei manuali, |q₀q₁q₂⟩ con "
                             "q₀ a sinistra; Qiskit numera al contrario e scriverebbe «001» dove il "
                             "primo passo qui scrive «100»."),

        # ── Pagina 5 · scheda Sfera di Bloch → la vera ZZFeatureMap (8 qubit) ──
        "bl_zz_title": "La vera ZZFeatureMap: dove avviene l'entanglement",
        "bl_zz_sub": ("Le stesse cifre, ora sugli 8 qubit della QSVM della tesi. Muovi il cursore "
                      "all'inizio della pagina e guarda quali qubit reagiscono."),
        "bl_zz_intro": ("<b>Da tre qubit agli otto del modello.</b> Ogni qubit della ZZFeatureMap "
                        "porta <b>una variabile clinica</b>: q₀ è l'HbA1c, q₁ la glicemia, e così "
                        "via fino all'IMC. Con 256 ampiezze non c'è più una figura dello stato da "
                        "guardare (né Q-sphere né istogramma), ma si possono comunque misurare le "
                        "<b>stesse due grandezze</b> della sezione precedente: quanto stato proprio "
                        "resta a ciascun qubit, e quanta informazione condivide con ognuno degli "
                        "altri. Questo non è più un esempio da manuale: è il circuito con cui il "
                        "modello è stato addestrato."),
        "bl_zz_current": "Variabile in gioco: <b>{var} = {val} {unidad}</b>. Le altre sette al loro valore di riferimento.",
        "bl_zz_r_title": "Stato proprio di ciascun qubit",
        "bl_zz_r_xaxis": "|r|: 1 = conserva il suo stato · 0 = del tutto in entanglement",
        "bl_zz_mi_title": "Informazione mutua fra qubit",
        "bl_zz_mi_cbar": "bit",
        "bl_zz_note": ("<b>Come leggere la matrice.</b> Ogni cella dice quanta informazione due qubit "
                       "condividono: più è accesa, più sono legati. E salta all'occhio che il colore "
                       "<b>si concentra in una banda lungo la diagonale</b> mentre gli angoli restano "
                       "vuoti. Non è un caso: la ZZFeatureMap usa <code>entanglement=\"linear\"</code>, "
                       "cioè <b>ci sono porte solo fra qubit vicini</b>. Con reps=2 quella correlazione "
                       "arriva al massimo a quattro anelli di distanza; oltre vale <b>esattamente "
                       "zero</b> (verificato su 300 profili: a distanza ≥ 5 nella catena, 0,0000 bit "
                       "senza una sola eccezione). La topologia del circuito si disegna da sé.<br><br>"
                       "E c'è una seconda cosa che si vede muovendo il cursore in alto: cambiando "
                       "<b>una</b> variabile si muovono solo <b>il suo qubit e i vicini immediati</b> "
                       ": gli altri non cambiano di un decimale. È lo stesso fatto visto dall'altro "
                       "lato: il cono di luce del circuito, in diretta."),
        "bl_zz_caveat": ("<b>Un avvertimento di lettura, e non da poco.</b> L'entanglement <b>non "
                         "cresce con il valore clinico</b>: alzando l'HbA1c lungo il suo intervallo, "
                         "|r| del primo qubit fa 0,88 → 1,00 → 0,65 → 0,24 → 0,73 → 0,98 → 0,33. Sale "
                         "e scende. Il motivo è che il dato entra come un <b>angolo</b> di fase e gli "
                         "angoli <b>fanno il giro</b>: due valori clinici molto diversi possono finire "
                         "in fasi quasi uguali. Con le feature standardizzate un caso estremo arriva a "
                         "x ≈ 5, e il prodotto 2·(π−xᵢ)(π−xⱼ) del termine di entanglement supera 2π "
                         "più volte. È una limitazione nota della codifica angolare non limitata, e "
                         "conviene tenerla presente prima di leggere queste figure come se misurassero "
                         "la gravità clinica: misurano la geometria del circuito, non il rischio."),

        # ── Pagina 6 · Predittore in diretta ──
        "lp_eyebrow": "Inferenza interattiva",
        "lp_title": "Predittore in diretta",
        "lp_subtitle": ("Probabilità che un profilo clinico corrisponda a una persona già "
                        "diagnosticata con diabete: LightGBM sulle 8 variabili di maggiore "
                        "importanza."),
        "lp_what_note": ("<b>Che cosa stima questo modulo.</b> L'obiettivo della pipeline è "
                         "<code>TARGET = (DIQ010 == 1)</code>, la risposta a <i>«un medico le ha mai "
                         "detto che ha il diabete?»</i>. Il modello quindi <b>rileva un diabete già "
                         "diagnosticato</b>: non predice chi lo svilupperà. È un compito di "
                         "rilevamento concomitante, non di rischio prospettico."),
        "lp_real_note": ("<b>Inferenza reale (ONNX).</b> Predizioni di LightGBM e SVM-RBF tramite "
                         "<code>onnxruntime</code>, con lo <code>StandardScaler</code> recuperato dalla "
                         "pipeline Gold. Le 8 variabili mostrate sono quelle di maggiore importanza "
                         "clinica; le restanti 81 feature sono fissate alla mediana dell'insieme di "
                         "addestramento. La QSVM non è disponibile in tempo reale per il costo O(n²) "
                         "del kernel quantistico: predire le 1.567 istanze del test è costato "
                         "144,5 minuti."),
        "lp_proxy_note": ("⚠ <b>Avviso tecnico e clinico.</b> Questo modulo non ha collegati i modelli "
                          "serializzati reali (<code>.onnx</code>): colloca "
                          "<code>lgbm_final.onnx</code>, <code>svm_final.onnx</code>, "
                          "<code>scaler_correcto.json</code> e <code>medianas_correctas.json</code> in "
                          "<code>streamlit/models/</code>. Il punteggio mostrato sotto è un "
                          "<b>sostituto trasparente</b>: una combinazione pesata per importanza SHAP "
                          "normalizzata, solo a fini di impaginazione. <b>Non è l'output di alcun "
                          "modello addestrato</b> e non deve essere citato come risultato. Nemmeno la "
                          "QSVM è disponibile in tempo reale (costo O(n²) del kernel quantistico)."),
        "lp_train_range": "Addestramento: {mu} ± {sd} (±3 σ → da {lo} a {hi})",
        "lp_extrapolates": "⚠ z = {z} · fuori dall'intervallo addestrato: il modello estrapola",
        "lp_ada": "Criterio ADA: &lt; 5,7 normale · 5,7–6,4 prediabete · ≥ 6,5 diabete",
        "lp_who_model": "il modello",
        "lp_who_proxy": "il sostituto",
        "lp_score_real": "Probabilità di una diagnosi esistente",
        "lp_score_proxy": "Punteggio di impaginazione (sostituto)",
        "lp_cat_low": "bassa",
        "lp_cat_mid": "intermedia",
        "lp_cat_high": "alta",
        "lp_interp_low": ("Il profilo resta nettamente sotto la soglia di decisione (50 %): {quien} lo "
                          "classificherebbe come non diagnosticato."),
        "lp_interp_mid": "Il valore si avvicina alla soglia di decisione (50 %): zona di incertezza.",
        "lp_interp_high": ("Il valore supera la soglia di decisione (50 %): {quien} classificherebbe "
                           "questo profilo come caso positivo."),
        # In italiano l'aggettivo segue il nome, quindi {cat} arriva già concordato al
        # femminile da lp_cat_* e si colloca dopo «Compatibilità».
        "lp_badge": "Compatibilità {cat}",
        "lp_gauge_caption": ("Compatibilità con una diagnosi esistente: bassa · intermedia · alta · "
                             "&nbsp;soglia di decisione = 50 %"),
        "lp_side_title": "I due modelli, fianco a fianco",
        "lp_side_sub": ("Ogni probabilità si giudica con il punto di taglio del proprio modello: non "
                        "sono intercambiabili"),
        "lp_own_threshold": "La sua soglia",
        "lp_would_classify": "Classificherebbe come",
        "lp_positive": "positivo",
        "lp_negative": "negativo",
        "lp_disagree": ("<b>I due modelli divergono di {dif} su questo profilo.</b> Concordano agli "
                        "estremi (profili chiaramente sani o chiaramente diabetici) e divergono nella "
                        "fascia intermedia, che è proprio dove una stima sarebbe più utile. Prendilo "
                        "come segnale di incertezza, non come prova che uno dei due abbia ragione."),
        "lp_curve_title": "Curva di risposta",
        "lp_curve_var": "Variabile da percorrere",
        "lp_curve_yaxis": "Probabilità",
        "lp_curve_thr": "soglia",
        "lp_ada_bands": ["normale", "prediabete", "diabete"],
        "lp_curve_note": ("Su questa variabile LightGBM restituisce <b>{n} valori distinti</b> nelle "
                          "{total} posizioni del cursore: è una scala a gradini, non una rampa. Gradini "
                          "maggiori: {saltos}. Il punto segna il tuo valore attuale."),
        "lp_curve_none": "nessuno",
        "lp_read_note": ("<b>Come leggere questi risultati.</b> Poiché l'obiettivo è una diagnosi "
                         "<i>già emessa</i>, il modello impara anche l'effetto del <b>trattamento</b>, "
                         "non solo quello della malattia. Questo inverte il senso clinico di due "
                         "variabili:"
                         '<ul style="margin:8px 0 0; padding-left:20px; line-height:1.7;">'
                         "<li><b>Colesterolo LDL</b>: più alto è l'LDL, <i>minore</i> è la probabilità "
                         "stimata (dal 43 % al 18 % percorrendo il cursore). Chi è diagnosticato di "
                         "solito è in terapia con statine.</li>"
                         "<li><b>Glicemia a digiuno</b>: la risposta ha forma di U: i valori molto "
                         "bassi alzano la stima quanto quelli alti, per via delle ipoglicemie dei "
                         "pazienti in terapia.</li></ul>"
                         '<div style="margin-top:10px;">Nessuna delle due va letta come un fattore di '
                         "rischio modificabile.</div>"),
    },
}


# ─────────────────────────────────────────────────────────────────────────
# BUSCADOR
# ─────────────────────────────────────────────────────────────────────────
# El índice NO se escribe a mano: se DERIVA del propio catálogo. Cada clave que
# acaba en "_title" ya es, por construcción, el rótulo de una sección visible, y su
# prefijo dice a qué página pertenece. Mantener una lista paralela de secciones
# habría sido una segunda fuente de verdad que se desincroniza en cuanto se añade,
# se renombra o se traduce una sección — el fallo clásico de los buscadores caseros.
# Así el índice se traduce solo y crece solo.

# Prefijo de clave → página a la que van sus secciones. Ojo a "bl", que es el único que NO
# nombra una página: la Esfera de Bloch dejó de ser entrada del menú y hoy es una pestaña de
# Circuito Cuántico, así que sus siete secciones (entrelazamiento, Q-sphere, medición, el
# ZZFeatureMap real…) tienen que apuntar ahí. Y no es cosmético: lo que se guarda aquí lo
# resuelve app.py con PAGE_KEYS.index(), de modo que un valor que no sea una página REVIENTA
# la lista de resultados en cuanto una de esas secciones coincide con lo buscado.
SEARCH_PREFIX = {
    "ov": "overview", "gov": "governance", "res": "results",
    "sh": "shap", "qc": "circuit", "bl": "circuit", "lp": "predictor",
}
# El "_title" de cada página es el titular de la página entera, no una sección suya:
# ya entra en el índice como fila de página (con el rótulo del menú), así que aquí se
# excluye para no duplicar la entrada.
#
# Por eso este conjunto NO se puede seguir derivando de SEARCH_PREFIX: desde que "bl" apunta a
# circuit hay DOS prefijos para una misma página, y solo uno de los dos —"qc"— da su titular.
# bl_title se queda fuera a propósito y sí genera fila de sección: es como se sigue llegando a
# «Esfera de Bloch» escribiendo su nombre, ahora que ya no hay un ítem del menú con ese rótulo.
_PREFIJO_TITULAR = ("ov", "gov", "res", "sh", "qc", "lp")
_SEARCH_PAGE_TITLES = {f"{p}_title" for p in _PREFIJO_TITULAR}

# Prefijos cuyas secciones no viven en la página a secas, sino DENTRO de una pestaña de esa
# página: (grupo de pestañas, posición). Va en la fila del buscador para que el salto abra la
# pestaña donde de verdad está la sección — sin esto, quien busca "Q-sphere" aterriza en
# Circuito Cuántico pero mirando el ranking del Random Forest, y tiene que adivinar que lo
# que buscaba está detrás de la otra pestaña.
_PREFIJO_PESTANA = {"bl": ("qc_tabs", 1)}

# Términos que un lector buscaría pero que no aparecen literalmente en ningún rótulo:
# nombres propios de tecnología, siglas y sinónimos. No generan fila propia — se suman
# al texto buscable de SU página, de modo que "Qiskit" lleva al Circuito Cuántico
# aunque ninguna sección se llame así. Van los CINCO idiomas en la misma lista y no una
# lista por idioma: unos son notación técnica que nadie traduce ("AUC", "ZZFeatureMap")
# y los demás no estorban — buscar en alemán no puede devolver una página equivocada,
# porque cada término está asignado a la página a la que pertenece en cualquier idioma.
# Y ese solapamiento es además una ventaja: "circuit" sirve al inglés y al francés a la
# vez, y "diabete" al francés y al italiano, así que la lista crece bastante menos que
# el número de lenguas.
# Sin tildes ni diéresis hace falta escribirlos: _plano() las quita en los dos lados de
# la comparación, así que "Qualität" y "qualitat" encuentran lo mismo.
SEARCH_ALIAS = {
    "overview":   ["NHANES", "medallon", "medallion", "medaillon", "medaglione", "bronze",
                   "silver", "gold", "dataset", "datensatz", "pipeline", "Databricks",
                   "ubersicht", "apercu", "panoramica"],
    "governance": ["Great Expectations", "calidad", "quality", "qualitat", "qualite",
                   "linaje", "lineage", "Delta Lake", "Unity Catalog", "DataOps",
                   "auditoria", "audit", "nachvollziehbarkeit", "governance",
                   "tracabilite", "tracciabilita"],
    "results":    ["ROC", "AUC", "matriz de confusion", "confusion matrix",
                   "konfusionsmatrix", "matrice de confusion", "matrice di confusione",
                   "F1", "recall", "precision", "LightGBM", "SVM", "RBF", "QSVM",
                   "metricas", "metrics", "metriken", "metriques", "metriche",
                   "benchmark", "ergebnisse", "resultats", "risultati"],
    "shap":       ["SHAP", "Shapley", "importancia", "importance", "wichtigkeit",
                   "importanza", "beeswarm", "explicabilidad", "explainability",
                   "erklarbarkeit", "interpretabilite", "interpretabilita", "XAI"],
    "circuit":    ["Qiskit", "ZZFeatureMap", "qubit", "kernel cuantico", "quantum kernel",
                   "quantenkernel", "noyau quantique", "kernel quantistico", "circuito",
                   "circuit", "schaltkreis", "puerta", "gate", "gatter", "porte",
                   "entrelazamiento", "entanglement", "verschrankung", "intrication", "IBM",
                   "Bloch", "esfera", "sphere", "kugel", "sfera", "estado", "state", "zustand",
                   "etat", "stato", "amplitud", "amplitude", "ampiezza", "Bell", "Q-sphere",
                   "qsphere", "CNOT", "Hadamard", "medicion", "measurement", "messung",
                   "mesure", "misura", "superposicion", "superposition", "sovrapposizione",
                   "informacion mutua", "mutual information", "wechselseitige information",
                   "information mutuelle", "informazione mutua", "matriz densidad",
                   "density matrix", "dichtematrix", "8 qubits", "cono de luz", "light cone",
                   "lichtkegel", "cone de lumiere", "cono di luce"],
    "predictor":  ["ONNX", "inferencia", "inference", "inferenz", "inferenza", "umbral",
                   "threshold", "schwelle", "seuil", "soglia", "slider", "schieberegler",
                   "curseur", "cursore", "prediccion", "prediction", "vorhersage",
                   "predizione", "what-if", "simulador", "simulator", "simulateur",
                   "simulatore"],
}


def _plano(texto):
    """Minúsculas sin tildes ni diacríticos, para comparar.

    Sin esto, "glucemia" no encontraría "Glucosa" y —peor— buscar "prediccion" sin
    tilde no encontraría "predicción", que es como está escrito en el catálogo. La
    búsqueda tiene que ser indiferente al acento en las DOS direcciones.
    """
    desc = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in desc if unicodedata.category(c) != "Mn")


def search_index(lang):
    """Filas buscables del idioma activo: páginas, secciones y variables clínicas.

    Cada fila lleva `page` (la clave estable a la que navegar), `label` (lo que se
    pinta) y `hay` (el texto ya aplanado contra el que se compara).
    """
    cat = STR[lang]
    base = STR[DEFAULT_LANG]

    def txt(clave):
        return cat[clave] if clave in cat else base[clave]

    filas = []
    for clave_pag, rotulo in zip(PAGE_KEYS, txt("nav")):
        # Al texto buscable de la página se le suman su propio titular y sus alias.
        extra = [txt(f"{p}_title") for p, k in SEARCH_PREFIX.items() if k == clave_pag]
        filas.append({"page": clave_pag, "label": rotulo, "kind": 0,
                      "hay": _plano(" ".join([rotulo, *extra, *SEARCH_ALIAS.get(clave_pag, [])]))})

    # Se recorren las claves del catálogo ESPAÑOL, que es el completo: así una sección
    # todavía sin traducir sigue apareciendo en el índice (con su texto español, igual
    # que hace S() al pintarla) en vez de desaparecer del buscador en inglés.
    for clave in base:
        if not clave.endswith("_title") or clave in _SEARCH_PAGE_TITLES:
            continue
        pagina = SEARCH_PREFIX.get(clave.split("_")[0])
        if pagina is None:
            continue
        rotulo = txt(clave)
        filas.append({"page": pagina, "label": rotulo, "kind": 1, "hay": _plano(rotulo),
                      "tab": _PREFIJO_PESTANA.get(clave.split("_")[0])})

    # Las ocho variables clínicas: se busca tanto por rótulo ("HbA1c") como por su
    # código NHANES ("LBXGH"), que es como aparecen en los informes del TFM.
    for codigo, rotulo in txt("qsvm_labels").items():
        filas.append({"page": "predictor", "label": f"{codigo} · {rotulo}", "kind": 2,
                      "hay": _plano(f"{codigo} {rotulo}")})
    return filas


def search(consulta, lang, limite=6):
    """Coincidencias de `consulta`, ordenadas por lo bien que encajan.

    El orden es: primero lo que EMPIEZA por la consulta, luego lo que la contiene al
    principio de una palabra, y por último lo que la contiene en cualquier posición;
    a igualdad, antes la página que la sección y antes el rótulo corto que el largo.
    Sin este desempate, buscar "res" devolvía seis secciones sueltas y la página
    Resultados quedaba enterrada debajo.
    """
    q = _plano(consulta.strip())
    if len(q) < 2:
        return []

    def rango(fila):
        hay = fila["hay"]
        if hay.startswith(q):
            pos = 0
        elif re.search(rf"\b{re.escape(q)}", hay):
            pos = 1
        else:
            pos = 2
        return (pos, fila["kind"], len(fila["label"]))

    return sorted((f for f in search_index(lang) if q in f["hay"]), key=rango)[:limite]


# ─────────────────────────────────────────────────────────────────────────
# ÍNDICE DE NAVEGACIÓN (árbol de la barra lateral)
# ─────────────────────────────────────────────────────────────────────────
# El árbol de la barra lateral tiene que enseñar las secciones de LAS SEIS PÁGINAS, y no solo
# las de la que se está viendo. Eso obliga a declararlas aquí, porque en el navegador no están:
# Streamlit renderiza únicamente la página activa, así que leer los .section-title del documento
# —que es lo que hacía el índice cuando vivía dentro de la página— solo puede contar cinco
# sextas partes de nada.
#
# Y no se puede DERIVAR del catálogo como hace el buscador. Su regla («toda clave que acaba en
# _title es una sección») vale para buscar, donde un falso positivo de más solo añade una fila
# a una lista de resultados, pero no para un índice: entre las claves _title hay titulares de
# tarjeta (gov_dropped_title, gov_lin_limit_title, gov_e2e_ok_title, ov_hero_title…) que NO se
# pintan con class="section-title" y que en el árbol serían destinos a los que no se puede
# saltar. Tampoco vale el ORDEN del catálogo: en Resumen, "Construido sobre" está escrito antes
# que la comparativa y en la página va después.
#
# Lo que se declara es la ESTRUCTURA —qué secciones, en qué pestaña y en qué orden—; los
# RÓTULOS siguen saliendo de STR, así que el árbol se traduce solo y renombrar una sección
# sigue siendo tocar una única línea del catálogo.
#
# MANTENIMIENTO: añadir una sección a una página es añadir su clave aquí, en la posición que
# ocupa en la página. Si se olvida, la sección existe pero el árbol no la lista (no rompe
# nada); si se escribe una clave que no está en el catálogo, _autocomprobacion() de más abajo
# lo canta al arrancar en vez de dejar una fila en blanco.
#
# Cada página es una lista de RAMAS. Una rama es (grupo_de_pestañas, posición, [claves]); en
# las páginas sin pestañas hay una sola rama con (None, None, [claves]) y el árbol se salta ese
# nivel. El grupo y la posición son exactamente lo que tabs_i18n() necesita para abrir la
# pestaña correcta al saltar, y los mismos dos datos que ya viaja _PREFIJO_PESTANA.
NAV_SECTIONS = {
    "overview": [
        (None, None, ["ov_stats_title", "ov_medallion_title", "ov_target_title",
                      "ov_compare_title", "ov_tech_title"]),
    ],
    "governance": [
        ("gov_tabs", 0, ["gov_funnel_title", "gov_suite_title", "gov_ops_title",
                         "gov_eff_title"]),
        ("gov_tabs", 1, ["gov_lin_title", "gov_delta_title", "gov_chain_title",
                         "gov_e2e_title"]),
        ("gov_tabs", 2, ["gov_stack_title", "gov_dec_title"]),
    ],
    "results": [
        (None, None, ["res_roc_title", "res_cm_title", "res_metrics_title"]),
    ],
    # Una sección por pestaña: el ranking de barras entra sin titular propio (lo encabeza el
    # titular de la página) y la única sección con rótulo es la lámina del beeswarm. Ojo a que
    # ese bloque es CONDICIONAL —shap_summary_image() no pinta nada si falta el PNG en
    # figures/—, así que si un día se retira la figura, el destino deja de existir. El salto lo
    # tolera: si el título no aparece en el documento, no se mueve el scroll.
    "shap": [
        ("sh_tabs", 0, ["sh_fig_lgbm_title"]),
        ("sh_tabs", 1, ["sh_fig_svm_title"]),
    ],
    # La segunda pestaña son las ocho secciones de la Esfera de Bloch, que fue página propia
    # hasta que se fusionó aquí (ver PAGES_RETIRADAS). De ahí que sus claves lleven prefijo
    # "bl" y no "qc": el prefijo dice de dónde viene el texto, la rama dice dónde se pinta.
    "circuit": [
        ("qc_tabs", 0, ["qc_how_title", "qc_feat_title", "qc_train_title", "qc_circuit_title"]),
        ("qc_tabs", 1, ["bl_title", "bl_ent_title", "bl_ent_circuit_title",
                        "bl_ent_qsphere_title", "bl_ent_meas_title", "bl_zz_title",
                        "bl_zz_r_title", "bl_zz_mi_title"]),
    ],
    "predictor": [
        (None, None, ["lp_side_title", "lp_curve_title"]),
    ],
}


def _autocomprobacion():
    """Revienta al importar si una clave del árbol no está en el catálogo.

    Es la red que sostiene el reparto de arriba: la estructura se escribe a mano y los rótulos
    salen del catálogo, así que el único fallo posible es una clave mal escrita o renombrada en
    STR sin actualizar aquí. Sin esta comprobación ese fallo no se ve en el sitio donde está —se
    ve como una fila vacía en la barra lateral que no lleva a ninguna parte, meses después.

    Se hace contra el catálogo ESPAÑOL, que es el completo, por lo mismo que search_index():
    una sección todavía sin traducir sigue siendo una sección.
    """
    base = STR[DEFAULT_LANG]
    faltan = [k for ramas in NAV_SECTIONS.values() for _, _, claves in ramas
              for k in claves if k not in base]
    if faltan:
        raise KeyError(f"NAV_SECTIONS apunta a claves que no existen en STR: {faltan}")
    sobran = [p for p in NAV_SECTIONS if p not in PAGE_KEYS]
    if sobran:
        raise KeyError(f"NAV_SECTIONS apunta a páginas que no existen: {sobran}")


_autocomprobacion()


def nav_tree(lang):
    """El árbol completo ya traducido: páginas → pestañas → secciones.

    Es lo que pinta la barra lateral entera —el menú y el índice son la misma lista—, así que
    cada página viaja además con su `icon`: el nombre del Material Symbol declarado en PAGES,
    sin el envoltorio `:material/…:` que solo entiende Streamlit.

    Devuelve una lista de páginas en el orden del menú, y cada una con sus ramas. Cada rama
    trae `tab`, que es (grupo, posición) o None, listo para dárselo tal cual al mismo camino de
    navegación que usa el buscador; y cada sección trae su `label`, que es además lo que el
    navegador buscará en el documento para saber a dónde bajar.

    Las páginas sin pestañas devuelven una rama única con `tab` a None y sin rótulo: el árbol
    cuelga sus secciones directamente de la página, sin inventarse un nivel intermedio que en
    la página no existe.
    """
    cat = STR[lang]
    base = STR[DEFAULT_LANG]

    def txt(clave):
        return cat[clave] if clave in cat else base[clave]

    arbol = []
    for (pagina, icono), rotulo in zip(PAGES, txt("nav")):
        ramas = []
        for grupo, pos, claves in NAV_SECTIONS.get(pagina, []):
            # El rótulo de la pestaña sale de su propio catálogo ("gov_tabs" y compañía), que
            # es la MISMA lista que pinta st.tabs. Así el árbol no puede acabar diciendo una
            # cosa y la pestaña otra.
            rotulo_tab = txt(grupo)[pos] if grupo else None
            ramas.append({
                "tab": (grupo, pos) if grupo else None,
                "label": rotulo_tab,
                "secciones": [{"key": k, "label": txt(k)} for k in claves],
            })
        arbol.append({"page": pagina, "icon": icono, "label": rotulo, "ramas": ramas})
    return arbol
