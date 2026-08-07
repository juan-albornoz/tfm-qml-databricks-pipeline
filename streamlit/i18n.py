"""Catálogo de textos ES/EN del dashboard.

Separado de app.py a propósito: el texto es lo único que cambia entre idiomas y
tenerlo en un solo fichero permite revisar la traducción de corrido, sin leerla
entre etiquetas HTML y llamadas a Plotly.

Dos reglas que sostienen el resto:

1. LAS CLAVES SON ESTABLES Y EN INGLÉS. La página "Gobernanza" es `governance`
   en el código y solo se convierte en texto visible al pintarse. Antes el
   enrutado comparaba contra el rótulo español (`if page == "Gobernanza"`), así
   que traducir el menú habría roto la navegación.

2. STR["en"] PUEDE ESTAR INCOMPLETO. `S()` en app.py cae al español cuando falta
   una clave, de modo que las páginas todavía sin traducir siguen funcionando en
   vez de reventar con KeyError. Esto es lo que permite traducir página a página.

Estado: menú, barra lateral y página Resumen traducidos. Pendientes Gobernanza,
Resultados, Análisis SHAP, Circuito Cuántico, Esfera de Bloch y Predictor.
"""

LANGS = ("es", "en")
DEFAULT_LANG = "es"

# ─────────────────────────────────────────────────────────────────────────
# PÁGINAS
# ─────────────────────────────────────────────────────────────────────────
# Clave estable + icono Bootstrap. El ORDEN es el del menú, y Gobernanza va en
# segunda posición por el motivo razonado en app.py (la app se recorre en el
# orden real de ejecución del pipeline). El rótulo visible sale de STR[lang]["nav"].
PAGES = [
    ("overview",   "house"),
    ("governance", "shield-check"),
    ("results",    "bar-chart"),
    ("shap",       "diagram-3"),
    ("circuit",    "cpu"),
    ("bloch",      "globe"),
    ("predictor",  "sliders"),
]
PAGE_KEYS = [k for k, _ in PAGES]
PAGE_ICONS = [i for _, i in PAGES]

# ─────────────────────────────────────────────────────────────────────────
# BANDERAS
# ─────────────────────────────────────────────────────────────────────────
# SVG en línea, no emoji: Windows no incluye glifos de bandera, así que 🇪🇸/🇬🇧
# se dibujarían como las letras "ES"/"GB" — precisamente en la plataforma donde
# se desarrolla y se defiende este TFM. El SVG además escala sin pixelarse y pesa
# menos que cualquier PNG equivalente.
#
# Las dos van en el MISMO lienzo 60×40 (3:2) para que ocupen idéntica caja en la
# barra. La Union Jack oficial es 2:1; aquí está redibujada a 3:2 (no estirada:
# los grosores de las bandas están recalculados en proporción), que es lo que
# hacen los sets de iconos de bandera al normalizar a una rejilla común.
FLAG_SVG = {
    "es": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">'
        '<rect width="60" height="40" fill="#AA151B"/>'
        '<rect y="10" width="60" height="20" fill="#F1BF00"/>'
        "</svg>"
    ),
    "en": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">'
        # Recorte "a" = el propio lienzo; "b" = los cuatro cuadrantes que dejan ver
        # solo la mitad de cada diagonal, que es lo que produce el contravirado
        # característico de la cruz de San Patricio.
        '<clipPath id="a"><path d="M0,0 v40 h60 v-40 z"/></clipPath>'
        '<clipPath id="b"><path d="M30,20 h30 v20 z v20 h-30 z h-30 v-20 z v-20 h30 z"/></clipPath>'
        '<g clip-path="url(#a)">'
        '<path d="M0,0 v40 h60 v-40 z" fill="#012169"/>'
        '<path d="M0,0 L60,40 M60,0 L0,40" stroke="#fff" stroke-width="8"/>'
        '<path d="M0,0 L60,40 M60,0 L0,40" clip-path="url(#b)" stroke="#C8102E" stroke-width="5"/>'
        '<path d="M30,0 v40 M0,20 h60" stroke="#fff" stroke-width="13"/>'
        '<path d="M30,0 v40 M0,20 h60" stroke="#C8102E" stroke-width="8"/>'
        "</g></svg>"
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
                "Circuito Cuántico", "Esfera de Bloch", "Predictor en Vivo"],
        "sidebar_expand": "Expandir",
        "sidebar_collapse": "Colapsar",
        "theme_to_dark": "Cambiar a tema oscuro",
        "theme_to_light": "Cambiar a tema claro",
        "lang_es_help": "Ver la aplicación en español",
        "lang_en_help": "Ver la aplicación en inglés",
        "footer_name": "Juan Albornoz C. · TFM 2026",
        "footer_uni": "Universidad Europea de Valencia",
        "footer_name_narrow": "JAC",
        "footer_uni_narrow": "UEV",

        # ── Página 1 · Resumen ──
        "ov_eyebrow": "Framework DataOps + QML",
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
            "<b>NHANES</b> (CDC) — el dataset no es el objeto de investigación, sino el vehículo para "
            "demostrar que la arquitectura es viable, reproducible y auditable sobre datos reales a "
            "escala. El núcleo experimental es una <b>comparativa triangulada</b> entre LightGBM "
            "(baseline tabular), SVM con kernel RBF (puente estructural) y un <b>QSVM</b> con "
            "FidelityQuantumKernel en Qiskit, manteniendo idéntico el clasificador subyacente para "
            "atribuir cualquier diferencia de rendimiento al efecto del kernel cuántico."),
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
        "ov_compare_title": "Comparativa triangulada: objetivo del experimento",
        "ov_compare": [
            ("LightGBM", "Baseline tabular de referencia"),
            ("SVM-RBF",  "Puente estructural hacia el componente cuántico"),
            ("QSVM",     "FidelityQuantumKernel, mismo clasificador, kernel cuántico"),
        ],
    },

    # ═══════════════════════════════ INGLÉS ═══════════════════════════════
    # Inglés académico estándar (convención de la literatura de ML): "winsorization",
    # "modeling", "binarized". Los nombres propios no se traducen — ni el de la
    # universidad, ni los de las librerías, ni los códigos de variable NHANES.
    "en": {
        # ── Navigation and sidebar ──
        "nav": ["Overview", "Governance", "Results", "SHAP Analysis",
                "Quantum Circuit", "Bloch Sphere", "Live Predictor"],
        "sidebar_expand": "Expand",
        "sidebar_collapse": "Collapse",
        "theme_to_dark": "Switch to dark theme",
        "theme_to_light": "Switch to light theme",
        "lang_es_help": "View the application in Spanish",
        "lang_en_help": "View the application in English",
        "footer_name": "Juan Albornoz C. · MSc Thesis 2026",
        "footer_uni": "Universidad Europea de Valencia",
        "footer_name_narrow": "JAC",
        "footer_uni_narrow": "UEV",

        # ── Page 1 · Overview ──
        "ov_eyebrow": "DataOps + QML Framework",
        "ov_title": "Overview",
        "ov_subtitle": ("End-to-end pipeline on Databricks CE + AWS S3, benchmarking a quantum QSVM "
                        "against two classical baselines, validated on real clinical data from the "
                        "NHANES study (CDC)."),
        "ov_lead": (
            "This framework designs and implements an <b>end-to-end DataOps</b> pipeline on "
            "<b>Databricks Community Edition</b>, using <b>AWS S3</b> as a real cloud storage layer and "
            "a <b>Medallion</b> architecture (Bronze → Silver → Gold) over Delta Lake as its backbone. "
            "The use case predicts type 2 diabetes from records of the <b>NHANES</b> study (CDC) — the "
            "dataset is not the object of the research but the vehicle for showing that the architecture "
            "is viable, reproducible and auditable on real data at scale. The experimental core is a "
            "<b>triangulated comparison</b> between LightGBM (tabular baseline), an SVM with RBF kernel "
            "(structural bridge) and a <b>QSVM</b> with FidelityQuantumKernel in Qiskit, keeping the "
            "underlying classifier identical so that any difference in performance can be attributed to "
            "the effect of the quantum kernel."),
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
        "ov_compare_title": "Triangulated comparison: goal of the experiment",
        "ov_compare": [
            ("LightGBM", "Reference tabular baseline"),
            ("SVM-RBF",  "Structural bridge to the quantum component"),
            ("QSVM",     "FidelityQuantumKernel: same classifier, quantum kernel"),
        ],
    },
}
