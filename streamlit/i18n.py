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

3. EL TEXTO VA AQUÍ; LOS NÚMEROS, NO. Las cifras verificadas (conteos de registros,
   métricas, importancias) siguen viviendo en app.py, que es donde se documenta su
   procedencia. Este catálogo aporta solo el rótulo y la glosa, y app.py los empareja
   POR POSICIÓN con el dato. Así una cifra no puede divergir entre idiomas. Las
   excepciones son los bloques cuyo contenido es íntegramente texto o notación
   localizable (las 15 expectativas, la tarjeta del escalador): esos van completos.

Estado: traducida la aplicación entera — menú, barra lateral y las siete páginas.
"""

import re
import unicodedata

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
# barra. Ninguna de las dos tiene esa proporción oficialmente (la española es 3:2,
# la de EE. UU. 10:19), así que la segunda se redibuja recalculando sus medidas en
# proporción — no se estira—, que es lo que hacen los sets de iconos al normalizar
# a una rejilla común.
#
# La bandera del idioma inglés es la de ESTADOS UNIDOS, no la del Reino Unido,
# porque el catálogo STR["en"] está escrito en inglés americano (ver su cabecera):
# la bandera nombra la variante que de verdad se sirve.

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
        # Nombre accesible del botón de colapso, no un tooltip: viaja dentro del rótulo y se
        # recorta por CSS (ver .st-key-toggle_sidebar en app.py). No se ve en pantalla.
        "sidebar_expand": "Expandir la barra lateral",
        "sidebar_collapse": "Colapsar la barra lateral",
        "search_label": "Buscar",
        "search_ph": "Buscar en el panel o en la web…",
        "search_expand": "Buscar — despliega la barra",
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
            ("Bronze — 3 ciclos unidos",
             "27 ficheros XPT · join por SEQN · 162 columnas comunes a los tres ciclos"),
            ("Filtro edad ≥ 18 años", "Restricción a población adulta"),
            ("Filtro ayuno — LBXGLU no nulo",
             "Proxy del subgrupo en ayunas: PHAFSTMN no es consistente entre ciclos"),
            ("Filtro DIQ010 válido",
             "Descarta los códigos 7 «no sabe» y 9 «rehúsa responder», y los nulos"),
        ],
        "gov_dropped_title": "Registros descartados por filtro",
        "gov_split_label": "Partición Gold 80/20",
        "gov_suite_title": "Suite de validación — dataframe-expectations",
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
        "gov_silver_card": "Silver — limpieza y saneamiento",
        "gov_gold_card": "Gold — preparación para modelado",
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
                         "columna a un único valor. Las más recortadas en el notebook 02 —PAQ635, PAQ650, PAQ605, "
                         "DMDHHSZA, DMDCITZN, SIALANG— son exactamente las que aquí aparecen constantes."),
        "gov_lin_title": "Trazabilidad sin MLflow",
        "gov_lin_sub": "La restricción que más condiciona la arquitectura del pipeline, y su mitigación.",
        "gov_lin_limit_title": "Limitación",
        "gov_lin_limit_body": ("La integración nativa de <b>MLflow</b> está deshabilitada en Databricks Serverless "
                               "gratuito. Cualquier llamada a <code>mlflow.start_run()</code> o "
                               "<code>mlflow.log_metric()</code> produce errores de autenticación: no hay registro "
                               "de experimentos, métricas ni artefactos."),
        "gov_lin_mit_title": "Mitigación — doble mecanismo",
        "gov_lin_mit_body": ("<b>Transaction logs de Delta Lake</b> — cada escritura genera un registro ACID con "
                             "versión, marca de tiempo y métricas de operación.<br><br>"
                             "<b>CSV de métricas por modelo</b> — cada notebook persiste sus resultados en Unity "
                             "Catalog Volumes, y las figuras los leen de ahí en vez de llevarlos escritos a mano."),
        "gov_delta_title": "Historial Delta — capa Gold",
        "gov_delta_sub": ("Seis versiones más recientes de las diez registradas. Delta purga las anteriores tras "
                          "168 h de retención, comportamiento esperado y no un fallo del pipeline."),
        "gov_delta_cols": ["Versión", "Timestamp", "Operación", "Filas", "Tamaño"],
        "gov_chain_title": "Cadena de custodia contra la fuga de información",
        "gov_chain_sub": ("Cuatro barreras encadenadas. La tercera no descarta ninguna columna — y eso es "
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
             "(89 de 89 pasan) — precisamente la prueba de que la primera barrera funcionó."),
            ("Guarda de pesos de muestreo",
             "Detiene el pipeline si aparece cualquier peso de muestreo distinto del conocido. "
             "WTINT2YR sí llega al modelado y está documentado en la decisión 10."),
        ],
        "gov_scaler_card": "Escalado sin fuga estadística",
        "gov_scaler": [
            ("Ajuste", "Solo sobre train", "fit_transform en train · transform en test"),
            ("Columnas evaluadas", "66", "Con varianza > 0"),
            ("Columnas constantes", "23", "Varianza 0 — ver decisión 08"),
            ("Media ≈ 0 · desv. ≈ 1", "Verificado", "Assert sobre todas las columnas con dispersión"),
        ],
        "gov_scaler_note": ("El <b>StandardScaler</b> se ajusta exclusivamente sobre <b>train</b>: "
                            "<code>fit_transform</code> en entrenamiento y <code>transform</code> en test. Si se "
                            "ajustara sobre el conjunto completo, la media y la desviación típica del test se "
                            "filtrarían al preprocesado y las métricas quedarían optimistas. La selección de las 8 "
                            "variables del QSVM sigue la misma regla — el Random Forest se entrena solo con "
                            "<code>X_train_svm_scaled</code>.<br><br>El filtro de correlación, en cambio, <b>sí</b> "
                            "se calcula antes de particionar. Está documentado y asumido en la decisión 09."),
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
                         "su propio camino —vector crudo, escalado solo del SVM, conversión a <code>float32</code>, "
                         "sesión ONNX y lectura del tensor de salida— y compara. Tolerancia {tol}; el ruido "
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
                        "condicionan la arquitectura, seis se asumen y documentan sin corregir —porque hacerlo "
                        "invalidaría los resultados ya obtenidos— y dos quedan resueltas sin residuo."),
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
            ("QSVM — coste computacional O(n²)",
             "Sobre las 6.264 instancias de train, la matriz de kernel exigiría ~39 millones de "
             "evaluaciones del circuito. Con 1.500 el kernel agota la memoria.",
             "Entrenamiento sobre muestra estratificada de 500 instancias (~22 min) preservando el "
             "ratio 86/14. La evaluación sí usa el test completo, para que las métricas comparen."),
            ("QSVM — sin soporte ONNX nativo",
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
             "columnas constantes no sesgan —el modelo no extrae señal de ellas—, pero pierden "
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
             "versión, la deserialización falla — y Serverless actualiza sin aviso.",
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
                        "svm_rbf": "SVC.predict() — signo de decision_function",
                        "qsvm": "decision_function > 0 (no es probabilidad)"},
        "res_reconciled": ('<span style="color:{color}; font-weight:600;">✓ Reconciliadas</span> — las cuatro '
                           "métricas de los tres modelos se han recalculado desde los scores por instancia y "
                           "coinciden con las publicadas."),
        "res_unreconciled": '<span style="color:{color}; font-weight:600;">⚠ Sin reconciliar</span> — {fallos}',
        "res_no_scores": "scores no disponibles",
        "res_threshold_note": ("<b>Los tres modelos están medidos en umbrales distintos.</b> Cada uno usa su punto "
                               "de corte natural —LightGBM <code>predict_proba ≥ 0,50</code>; SVM-RBF el signo de "
                               "<code>decision_function</code>, que en la escala de probabilidad guardada equivale "
                               "a ≈ 0,22; QSVM <code>decision_function &gt; 0</code>, que no es una probabilidad—. "
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
                            "sí penalizan el desbalance de clases — pero dependen del umbral, y cada modelo usa "
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
                          "clasificación aleatoria — Recall ≈ 0 para la clase diabetes (1 de 220), "
                          "Accuracy = 0,8602 refleja solo la proporción de la clase mayoritaria. El MCC ≈ 0 "
                          "confirma ausencia de capacidad predictiva real."),

        # ── Página 4 · Análisis SHAP ──
        "sh_eyebrow": "Interpretabilidad",
        "sh_title": "Análisis SHAP",
        "sh_subtitle": ("Importancia global de variables — TreeExplainer (LightGBM) vs. "
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
        "sh_fig_lgbm_title": "SHAP Summary Plot — LightGBM (Figura 27)",
        "sh_fig_lgbm_cap": ("Cada punto es una instancia del test; el color indica el valor de la variable "
                            "(rojo alto, azul bajo) y la posición horizontal su impacto en la predicción. "
                            "LBXGH y RIDAGEYR dominan el modelo."),
        "sh_fig_svm_title": "SHAP Summary Plot — SVM-RBF (Figura 31)",
        "sh_fig_svm_cap": ("Cada punto es una instancia; color = valor de la variable, posición = impacto. "
                           "KernelExplainer sobre 200 instancias del test."),

        # ── Página 5 · Circuito Cuántico ──
        "qc_eyebrow": "Componente cuántico",
        "qc_title": "Circuito Cuántico",
        "qc_subtitle": ("Configuración del ZZFeatureMap y FidelityQuantumKernel implementados en Qiskit sobre "
                        "Databricks CE."),
        "qc_specs": ["Qubits (feature_dimension)", "Repeticiones (reps)", "Entanglement", "Versión de Qiskit"],
        "qc_how_title": "Cómo funciona",
        "qc_how_p1": ("El <b>ZZFeatureMap</b> codifica cada una de las 8 variables clínicas como un ángulo de "
                      "fase (puerta P) en un qubit independiente, tras crear superposición con puertas Hadamard. "
                      "Su elemento distintivo es el <b>entrelazamiento</b> entre pares de qubits mediante puertas "
                      "que dependen del producto cruzado de dos variables — correlaciones que el kernel RBF "
                      "clásico no puede representar."),
        "qc_how_p2": ("El <b>FidelityQuantumKernel</b> mide la similitud entre dos pacientes como la fidelidad "
                      "entre sus estados cuánticos: <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. La implementación usa "
                      "<code>StatevectorSampler</code>, simulando el estado exacto sin ruido — resultados "
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

        # ── Página 6 · Esfera de Bloch ──
        "bl_eyebrow": "Codificación cuántica",
        "bl_title": "Esfera de Bloch",
        "bl_subtitle": "Cómo el ZZFeatureMap codifica el valor de una variable clínica como estado cuántico |ψ⟩.",
        # Nota de entrada de la página, en el mismo formato que lp_what_note: quien llega aquí
        # puede venir del lado clínico y no tener por qué saber qué es una esfera de Bloch, y sin
        # eso la figura es una bola con una flecha. Se explica con el bit clásico como punto de
        # partida y se cierra atando el dibujo a lo que hace el deslizador de al lado, que es lo
        # que convierte la explicación en algo que se puede comprobar moviendo el control.
        "bl_what_note": ("<b>Qué es la esfera de Bloch.</b> Un bit clásico solo puede valer 0 o 1. "
                         "Un qubit admite además cualquier mezcla de los dos, y esa mezcla no cabe "
                         "en un único número: hace falta un mapa. La esfera de Bloch es ese mapa — "
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
                    "puerta de fase tras una Hadamard deja el estado <b>sobre el ecuador</b> —θ = π/2 fijo, "
                    "P(|0⟩) = P(|1⟩) = 50 % siempre— codificando el dato en el ángulo <b>azimutal</b> φ, no en el "
                    "polar. Tampoco normaliza a [0,1]: usa el valor escalado directamente. Por eso esta esfera "
                    "ilustra el concepto, pero no reproduce paso a paso el circuito. El entrelazamiento (puertas "
                    "P(2·(π−x<sub>i</sub>)·(π−x<sub>j</sub>))) solo es representable en el espacio conjunto de "
                    "los 8 qubits — ver Circuito Cuántico."),

        # ── Página 6 · Esfera de Bloch → sección de entrelazamiento ──
        # Continúa exactamente donde acaba bl_note: esa nota cierra diciendo que el
        # entrelazamiento solo se representa en el espacio conjunto, y esta sección es esa
        # frase convertida en figura. De ahí que el subtítulo la enuncie como un límite del
        # mapa que la página acaba de enseñar, y no como un tema nuevo.
        "bl_ent_title": "Entrelazamiento: dos qubits, un solo estado",
        "bl_ent_sub": ("El límite de la esfera de arriba. Aplica las dos puertas y mira qué le pasa al "
                       "estado local de cada qubit."),
        "bl_ent_intro": ("<b>Dónde deja de servir la esfera de Bloch.</b> Con un qubit basta una esfera y una "
                         "flecha. Con dos, la tentación es dibujar dos esferas — y para la mayoría de los "
                         "estados funciona. Pero existe una familia de estados en los que <b>no queda flecha "
                         "que dibujar</b>: el par tiene un estado perfectamente definido y ninguno de sus dos "
                         "miembros lo tiene por separado. Eso es el entrelazamiento, y se construye con dos "
                         "puertas. Aplícalas y sigue las tres cifras de la izquierda."),
        "bl_ent_btn_h": "1 · Hadamard en q₀",
        "bl_ent_btn_cnot": "2 · CNOT (control q₀ → objetivo q₁)",
        "bl_ent_btn_reset": "Reiniciar a |00⟩",
        # Una entrada por paso, en el orden en que se recorren. Dos o tres frases: lo justo
        # para decir qué acaba de cambiar en las dos figuras de debajo, sin repetir lo que ya
        # dicen los rótulos.
        "bl_ent_step_note": [
            ("<b>Punto de partida.</b> Dos qubits, los dos en |0⟩, sin ninguna puerta aplicada. El estado "
             "conjunto es |00⟩ y todavía no tiene nada de cuántico: equivale exactamente a dos bits "
             "clásicos puestos a cero. En la Q-sphere hay un único nodo, en el polo norte, que se lleva "
             "toda la probabilidad."),
            ("<b>Superposición, todavía sin entrelazar.</b> La Hadamard deja a q₀ a medio camino entre "
             "|0⟩ y |1⟩, mientras q₁ sigue firme en |0⟩: el estado conjunto es (|00⟩ + |10⟩)/√2. Los dos "
             "qubits siguen siendo <b>independientes</b> — cada uno tiene su propio estado puro y dos "
             "esferas de Bloch bastarían para describirlos. Fíjate en que la longitud del vector local "
             "sigue valiendo 1: hay flecha que dibujar."),
            ("<b>Estado de Bell.</b> El CNOT voltea q₁ solo cuando q₀ vale 1; aplicado sobre una "
             "superposición, eso ata los dos resultados en uno solo: (|00⟩ + |11⟩)/√2. Los nodos se han "
             "ido a los polos y el ecuador ha quedado vacío. Y el mapa se rompe aquí: la longitud del "
             "vector local acaba de caer a <b>0</b> — el qubit 0 ya no está en ningún punto de su esfera, "
             "porque por separado <b>ya no tiene estado</b>."),
        ],
        "bl_ent_circuit_title": "Circuito",
        "bl_ent_circuit_alt": "Circuito de dos qubits con las puertas aplicadas hasta ahora",
        "bl_ent_qsphere_title": "Q-sphere del estado conjunto",
        # Las tres dicen lo mismo desde tres ángulos (ver ent_local en app.py): se dejan las
        # tres porque cada lector entra por una — la longitud se contrasta con la figura, la
        # pureza es la magnitud estándar y la entropía es la que cita la literatura.
        "bl_ent_kpi": ["Longitud del vector local |r| (q₀)", "Pureza Tr(ρ₀²)",
                       "Entropía de entrelazamiento"],
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
            ("Con los dos qubits en |0⟩ el resultado es <b>00</b> en todos los disparos. No hay nada que "
             "sortear todavía: es el comportamiento de dos bits clásicos."),
            ("Salen <b>00</b> y <b>10</b> a partes iguales: q₀ se comporta como una moneda al aire y q₁ "
             "vale 0 pase lo que pase. Los dos resultados son <b>independientes</b> — saber uno no dice "
             "nada del otro."),
            ("<b>Solo salen 00 y 11</b>, cerca del 50 % cada uno. Las dos barras vacías son el dato: "
             "<b>01 y 10 no aparecen nunca</b>, ni una vez en diez mil disparos. Cada qubit sigue dando "
             "un resultado al azar, pero los dos dan <b>siempre el mismo</b>: medir uno determina el otro "
             "al instante. Esa correlación perfecta es el entrelazamiento visto desde el laboratorio."),
        ],
        "bl_ent_impl_note": ("<b>Cómo está calculado.</b> Las cuatro amplitudes salen de álgebra lineal "
                             "<b>exacta</b> en NumPy —las matrices de H⊗I y del CNOT aplicadas a |00⟩—, no de "
                             "una aproximación, y las mediciones de un muestreo multinomial sobre |ψ|², que es "
                             "lo que hace un simulador ideal sin ruido. El panel <b>no carga Qiskit</b>: el "
                             "entorno que se despliega es Streamlit, NumPy, Plotly y ONNX Runtime, mientras "
                             "que Qiskit vive en el pipeline de Databricks —donde se entrena el QSVM— y sus "
                             "figuras llegan aquí ya renderizadas, como el circuito de 8 qubits de la página "
                             "Circuito Cuántico. La convención de la base es la de libro de texto, |q₀q₁⟩ con "
                             "q₀ a la izquierda; Qiskit numera al revés y escribiría «01» donde el paso "
                             "intermedio de aquí escribe «10»."),

        # ── Página 6 · Esfera de Bloch → el ZZFeatureMap real (8 qubits) ──
        # Tercer y último escalón de la página. Aquí se deja el ejemplo de libro y se mide el
        # circuito del TFM, así que el tono sube medio punto: sigue siendo divulgativo, pero
        # ya puede dar por sabido lo que enseñan las dos secciones de arriba.
        "bl_zz_title": "El ZZFeatureMap real: dónde ocurre el entrelazamiento",
        "bl_zz_sub": ("Las mismas cifras, ahora sobre los 8 qubits del QSVM del TFM. Mueve el deslizador "
                      "del principio de la página y mira qué qubits reaccionan."),
        "bl_zz_intro": ("<b>De dos qubits a los ocho del modelo.</b> Cada qubit del ZZFeatureMap lleva "
                        "<b>una variable clínica</b>: q₀ es la HbA1c, q₁ la glucosa, y así hasta el IMC. "
                        "Con 256 amplitudes ya no hay figura del estado que se pueda mirar —ni Q-sphere ni "
                        "histograma—, pero sí se pueden medir las <b>mismas dos magnitudes</b> de la "
                        "sección anterior: cuánto estado propio le queda a cada qubit, y cuánta información "
                        "comparte con cada uno de los demás. Eso ya no es un ejemplo de libro: es el "
                        "circuito con el que se entrenó el modelo."),
        "bl_zz_current": "Variable en juego: <b>{var} = {val} {unidad}</b>. Las otras siete, en su valor de referencia.",
        "bl_zz_r_title": "Estado propio de cada qubit",
        "bl_zz_r_xaxis": "|r| — 1 = conserva su estado · 0 = entrelazado del todo",
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
                       "demás no cambian ni un decimal. Es el mismo hecho visto desde el otro lado — el "
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

        # ── Página 7 · Predictor en Vivo ──
        "lp_eyebrow": "Inferencia interactiva",
        "lp_title": "Predictor en Vivo",
        "lp_subtitle": ("Probabilidad de que un perfil clínico corresponda a una persona ya diagnosticada de "
                        "diabetes — LightGBM sobre las 8 variables de mayor importancia."),
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
                          "serializados reales (<code>.onnx</code>) — coloca <code>lgbm_final.onnx</code>, "
                          "<code>svm_final.onnx</code>, <code>scaler_correcto.json</code> y "
                          "<code>medianas_correctas.json</code> en <code>streamlit/models/</code>. La puntuación "
                          "mostrada abajo es un <b>sustituto transparente</b>: una combinación ponderada por "
                          "importancia SHAP normalizada, solo para fines de maquetación. <b>No es la salida de "
                          "ningún modelo entrenado</b> y no debe citarse como resultado. QSVM tampoco está "
                          "disponible en tiempo real (coste O(n²) del kernel cuántico)."),
        "lp_train_range": "Entrenamiento: {mu} ± {sd} (±3 sd → {lo} a {hi})",
        "lp_extrapolates": "⚠ z = {z} — fuera del rango entrenado: el modelo extrapola",
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
        "lp_side_sub": "Cada probabilidad se juzga con el punto de corte de su propio modelo — no son intercambiables",
        "lp_own_threshold": "Su umbral",
        "lp_would_classify": "Clasificaría como",
        "lp_positive": "positivo",
        "lp_negative": "negativo",
        "lp_disagree": ("<b>Los dos modelos discrepan {dif} en este perfil.</b> Coinciden en los extremos "
                        "—perfiles claramente sanos o claramente diabéticos— y divergen en la banda intermedia, "
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
                         "<li><b>Colesterol LDL</b> — a más LDL, <i>menor</i> probabilidad estimada (de 43% a 18% "
                         "recorriendo el slider). Los diagnosticados suelen estar tratados con estatinas.</li>"
                         "<li><b>Glucosa en ayunas</b> — la respuesta tiene forma de U: los valores muy bajos "
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
                "Quantum Circuit", "Bloch Sphere", "Live Predictor"],
        "sidebar_expand": "Expand the sidebar",
        "sidebar_collapse": "Collapse the sidebar",
        "search_label": "Search",
        "search_ph": "Search the dashboard or the web…",
        "search_expand": "Search — expands the sidebar",
        "search_in": "in {p}",
        "search_none": "No matches in the dashboard.",
        "search_web": "Search “{q}” in:",
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
            "The use case predicts type 2 diabetes from records of the <b>NHANES</b> study (CDC)—the "
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
            ("Bronze — 3 cycles joined",
             "27 XPT files · join on SEQN · 162 columns common to all three cycles"),
            ("Age filter ≥ 18 years", "Restriction to the adult population"),
            ("Fasting filter — LBXGLU not null",
             "Proxy for the fasting subgroup: PHAFSTMN is not consistent across cycles"),
            ("Valid DIQ010 filter",
             "Drops codes 7 “don't know” and 9 “refused to answer”, and the nulls"),
        ],
        "gov_dropped_title": "Records dropped per filter",
        "gov_split_label": "Gold 80/20 split",
        "gov_suite_title": "Validation suite — dataframe-expectations",
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
        "gov_silver_card": "Silver — cleaning and sanitation",
        "gov_gold_card": "Gold — preparation for modeling",
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
                         "column to a single value. The most heavily clipped ones in notebook 02—PAQ635, PAQ650, "
                         "PAQ605, DMDHHSZA, DMDCITZN, SIALANG—are exactly the ones that appear constant here."),
        "gov_lin_title": "Traceability without MLflow",
        "gov_lin_sub": "The constraint that shapes the pipeline architecture the most, and its mitigation.",
        "gov_lin_limit_title": "Limitation",
        "gov_lin_limit_body": ("Native <b>MLflow</b> integration is disabled on the free Databricks Serverless "
                               "tier. Any call to <code>mlflow.start_run()</code> or "
                               "<code>mlflow.log_metric()</code> raises authentication errors: there is no "
                               "tracking of experiments, metrics or artifacts."),
        "gov_lin_mit_title": "Mitigation — a two-fold mechanism",
        "gov_lin_mit_body": ("<b>Delta Lake transaction logs</b> — every write produces an ACID record with "
                             "version, timestamp and operation metrics.<br><br>"
                             "<b>Per-model metrics CSV</b> — each notebook persists its results to Unity Catalog "
                             "Volumes, and the figures read them from there instead of carrying them hard-coded."),
        "gov_delta_title": "Delta history — Gold layer",
        "gov_delta_sub": ("The six most recent of the ten recorded versions. Delta purges the earlier ones after "
                          "168 h of retention: expected behavior, not a pipeline failure."),
        "gov_delta_cols": ["Version", "Timestamp", "Operation", "Rows", "Size"],
        "gov_chain_title": "Chain of custody against information leakage",
        "gov_chain_sub": ("Four chained barriers. The third one drops no column at all—and that is exactly what "
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
             "—which is precisely the proof that the first barrier worked."),
            ("Sampling-weight guard",
             "Halts the pipeline if any sampling weight other than the known one appears. WTINT2YR "
             "does reach the modeling stage and is documented in decision 10."),
        ],
        "gov_scaler_card": "Scaling without statistical leakage",
        "gov_scaler": [
            ("Fit", "Train only", "fit_transform on train · transform on test"),
            ("Columns evaluated", "66", "With variance > 0"),
            ("Constant columns", "23", "Variance 0 — see decision 08"),
            ("Mean ≈ 0 · sd ≈ 1", "Verified", "Assert over every column with dispersion"),
        ],
        "gov_scaler_note": ("The <b>StandardScaler</b> is fitted exclusively on <b>train</b>: "
                            "<code>fit_transform</code> on training and <code>transform</code> on test. Were it "
                            "fitted on the full dataset, the mean and standard deviation of the test set would "
                            "leak into preprocessing and the metrics would come out optimistic. The selection of "
                            "the 8 QSVM variables follows the same rule—the Random Forest is trained only on "
                            "<code>X_train_svm_scaled</code>.<br><br>The correlation filter, by contrast, "
                            "<b>is</b> computed before splitting. This is documented and accepted in decision 09."),
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
                         "path—raw vector, scaling for the SVM only, conversion to <code>float32</code>, ONNX "
                         "session and reading of the output tensor—and compares. Tolerance {tol}; the noise "
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
                        "the architecture, six are accepted and documented without correction—because correcting "
                        "them would invalidate the results already obtained—and two are resolved with no "
                        "residue."),
        "gov_dec_tags": {"critical": "Architecture", "warning": "Accepted", "good": "Resolved"},
        "gov_dec_problem": "Problem · ",
        "gov_dec_solution": "Solution adopted · ",
        "gov_decisiones": [
            ("spark.conf blocked on Serverless",
             "Configuring AWS credentials through spark.conf.set returns CONFIG_NOT_AVAILABLE—the "
             "standard mechanism for connecting Spark to S3.",
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
            ("QSVM — O(n²) computational cost",
             "Over the 6,264 training instances, the kernel matrix would demand ~39 million circuit "
             "evaluations. At 1,500 the kernel exhausts memory.",
             "Training on a stratified sample of 500 instances (~22 min) preserving the 86/14 ratio. "
             "Evaluation does use the full test set, so that the metrics remain comparable."),
            ("QSVM — no native ONNX support",
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
             "models. Constant columns do not bias anything—the model extracts no signal from them—"
             "but information is lost. The fix is identified as future work."),
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
             "changes version, deserialization fails—and Serverless updates without warning.",
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
                        "svm_rbf": "SVC.predict() — sign of decision_function",
                        "qsvm": "decision_function > 0 (not a probability)"},
        "res_reconciled": ('<span style="color:{color}; font-weight:600;">✓ Reconciled</span> — the four metrics '
                           "of all three models have been recomputed from the per-instance scores and match the "
                           "published ones."),
        "res_unreconciled": '<span style="color:{color}; font-weight:600;">⚠ Not reconciled</span> — {fallos}',
        "res_no_scores": "scores not available",
        "res_threshold_note": ("<b>The three models are measured at different thresholds.</b> Each one uses its "
                               "natural cut-off point—LightGBM <code>predict_proba ≥ 0.50</code>; SVM-RBF the "
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
                            "do penalize class imbalance—but they depend on the threshold, and each model uses "
                            "its own: compare anything other than AUC-ROC with caution"),
        "res_metric_desc": {
            "auc": "Area under the ROC curve: ability to separate diabetes vs. non-diabetes. 0.5 = chance, 1 = perfect.",
            "f1_macro": "Harmonic mean of precision and recall averaged per class (unweighted). Penalizes imbalance.",
            "accuracy": "Proportion of overall correct predictions. With imbalanced classes it may reflect only the majority class.",
            "mcc": "Matthews correlation coefficient: overall quality, robust to imbalance. 0 = chance, 1 = perfect.",
        },
        "res_qsvm_note": ("<b>Note on the QSVM experiment.</b> The QSVM was trained on a stratified sample of 500 "
                          "instances (O(n²) cost of the quantum kernel) and evaluated on the full test set of "
                          "1,567. AUC-ROC = 0.5493 indicates that the model barely beats random classification—"
                          "Recall ≈ 0 for the diabetes class (1 out of 220), and Accuracy = 0.8602 reflects only "
                          "the proportion of the majority class. MCC ≈ 0 confirms the absence of any real "
                          "predictive ability."),

        # ── Page 4 · SHAP Analysis ──
        "sh_eyebrow": "Interpretability",
        "sh_title": "SHAP Analysis",
        "sh_subtitle": ("Global feature importance — TreeExplainer (LightGBM) vs. KernelExplainer (SVM-RBF)."),
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
        "sh_fig_lgbm_title": "SHAP Summary Plot — LightGBM (Figure 27)",
        "sh_fig_lgbm_cap": ("Each point is a test instance; color indicates the value of the variable (red high, "
                            "blue low) and the horizontal position its impact on the prediction. LBXGH and "
                            "RIDAGEYR dominate the model."),
        "sh_fig_svm_title": "SHAP Summary Plot — SVM-RBF (Figure 31)",
        "sh_fig_svm_cap": ("Each point is an instance; color = value of the variable, position = impact. "
                           "KernelExplainer over 200 test instances."),

        # ── Page 5 · Quantum Circuit ──
        "qc_eyebrow": "Quantum component",
        "qc_title": "Quantum Circuit",
        "qc_subtitle": ("Configuration of the ZZFeatureMap and FidelityQuantumKernel implemented in Qiskit on "
                        "Databricks CE."),
        "qc_specs": ["Qubits (feature_dimension)", "Repetitions (reps)", "Entanglement", "Qiskit version"],
        "qc_how_title": "How it works",
        "qc_how_p1": ("The <b>ZZFeatureMap</b> encodes each of the 8 clinical variables as a phase angle (P gate) "
                      "on an independent qubit, after creating superposition with Hadamard gates. Its "
                      "distinguishing element is the <b>entanglement</b> between pairs of qubits through gates "
                      "that depend on the cross product of two variables—correlations the classical RBF kernel "
                      "cannot represent."),
        "qc_how_p2": ("The <b>FidelityQuantumKernel</b> measures the similarity between two patients as the "
                      "fidelity between their quantum states: <code>K(x,y) = |⟨ψ(x)|ψ(y)⟩|²</code>. The "
                      "implementation uses <code>StatevectorSampler</code>, simulating the exact state without "
                      "noise—deterministic, reproducible results."),
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

        # ── Page 6 · Bloch Sphere ──
        "bl_eyebrow": "Quantum encoding",
        "bl_title": "Bloch Sphere",
        "bl_subtitle": "How the ZZFeatureMap encodes the value of a clinical variable as a quantum state |ψ⟩.",
        "bl_what_note": ("<b>What the Bloch sphere is.</b> A classical bit can only be 0 or 1. "
                         "A qubit also admits any mixture of the two, and that mixture does not fit "
                         "into a single number: it needs a map. The Bloch sphere is that map — every "
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
                    "equator</b>—θ = π/2 fixed, P(|0⟩) = P(|1⟩) = 50% always—encoding the datum in the "
                    "<b>azimuthal</b> angle φ, not the polar one. Nor does it normalize to [0,1]: it uses the "
                    "scaled value directly. That is why this sphere illustrates the concept but does not "
                    "reproduce the circuit step by step. Entanglement (gates "
                    "P(2·(π−x<sub>i</sub>)·(π−x<sub>j</sub>))) is only representable in the joint space of the "
                    "8 qubits—see Quantum Circuit."),

        # ── Page 6 · Bloch Sphere → entanglement section ──
        "bl_ent_title": "Entanglement: two qubits, a single state",
        "bl_ent_sub": ("Where the sphere above stops working. Apply both gates and watch what happens to "
                       "each qubit's local state."),
        "bl_ent_intro": ("<b>Where the Bloch sphere stops working.</b> One qubit needs one sphere and one "
                         "arrow. With two, the temptation is to draw two spheres—and for most states that "
                         "works. But there is a family of states where <b>no arrow is left to draw</b>: the "
                         "pair has a perfectly defined state and neither of its members has one on its own. "
                         "That is entanglement, and it takes two gates to build. Apply them and follow the "
                         "three figures on the left."),
        "bl_ent_btn_h": "1 · Hadamard on q₀",
        "bl_ent_btn_cnot": "2 · CNOT (control q₀ → target q₁)",
        "bl_ent_btn_reset": "Reset to |00⟩",
        "bl_ent_step_note": [
            ("<b>Starting point.</b> Two qubits, both in |0⟩, no gates applied. The joint state is |00⟩ and "
             "there is nothing quantum about it yet: it is exactly equivalent to two classical bits set to "
             "zero. The Q-sphere shows a single node at the north pole, holding all the probability."),
            ("<b>Superposition, not yet entangled.</b> The Hadamard leaves q₀ halfway between |0⟩ and |1⟩ "
             "while q₁ stays firmly in |0⟩: the joint state is (|00⟩ + |10⟩)/√2. The two qubits are still "
             "<b>independent</b>—each has its own pure state, and two Bloch spheres would describe them "
             "perfectly well. Note that the local vector length is still 1: there is an arrow to draw."),
            ("<b>Bell state.</b> CNOT flips q₁ only when q₀ is 1; applied to a superposition, that ties both "
             "outcomes into one: (|00⟩ + |11⟩)/√2. The nodes have moved to the poles and the equator is "
             "empty. And this is where the map breaks: the local vector length has just dropped to <b>0</b>—"
             "qubit 0 is no longer at any point of its sphere, because on its own it <b>no longer has a "
             "state</b>."),
        ],
        "bl_ent_circuit_title": "Circuit",
        "bl_ent_circuit_alt": "Two-qubit circuit showing the gates applied so far",
        "bl_ent_qsphere_title": "Q-sphere of the joint state",
        "bl_ent_kpi": ["Local vector length |r| (q₀)", "Purity Tr(ρ₀²)", "Entanglement entropy"],
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
            ("With both qubits in |0⟩ the outcome is <b>00</b> on every shot. There is nothing to sample "
             "yet: this is how two classical bits behave."),
            ("<b>00</b> and <b>10</b> come out in equal parts: q₀ behaves like a coin toss and q₁ is 0 no "
             "matter what. The two outcomes are <b>independent</b>—knowing one tells you nothing about the "
             "other."),
            ("<b>Only 00 and 11 appear</b>, close to 50% each. The two empty bars are the finding: <b>01 and "
             "10 never come out</b>, not once in ten thousand shots. Each qubit still gives a random "
             "outcome, but both give <b>always the same one</b>: measuring one determines the other "
             "instantly. That perfect correlation is entanglement seen from the lab."),
        ],
        "bl_ent_impl_note": ("<b>How this is computed.</b> The four amplitudes come from <b>exact</b> linear "
                             "algebra in NumPy—the H⊗I and CNOT matrices applied to |00⟩—not from an "
                             "approximation, and the measurements from multinomial sampling over |ψ|², which "
                             "is what an ideal noiseless simulator does. This dashboard <b>does not load "
                             "Qiskit</b>: the deployed environment is Streamlit, NumPy, Plotly and ONNX "
                             "Runtime, while Qiskit lives in the Databricks pipeline—where the QSVM is "
                             "trained—and its figures arrive here already rendered, like the 8-qubit circuit "
                             "on the Quantum Circuit page. The basis convention is the textbook one, |q₀q₁⟩ "
                             "with q₀ on the left; Qiskit numbers the other way and would write “01” where "
                             "the intermediate step here writes “10”."),

        # ── Page 6 · Bloch Sphere → the real ZZFeatureMap (8 qubits) ──
        "bl_zz_title": "The real ZZFeatureMap: where entanglement actually happens",
        "bl_zz_sub": ("The same figures, now over the 8 qubits of the TFM's QSVM. Move the slider at the "
                      "top of the page and watch which qubits react."),
        "bl_zz_intro": ("<b>From two qubits to the model's eight.</b> Each qubit of the ZZFeatureMap "
                        "carries <b>one clinical variable</b>: q₀ is HbA1c, q₁ glucose, and so on up to "
                        "BMI. With 256 amplitudes there is no picture of the state left to look at—neither "
                        "Q-sphere nor histogram—but the <b>same two quantities</b> from the previous "
                        "section can still be measured: how much of its own state each qubit keeps, and how "
                        "much information it shares with every other one. This is no longer a textbook "
                        "example: it is the circuit the model was trained with."),
        "bl_zz_current": "Variable in play: <b>{var} = {val} {unidad}</b>. The other seven at their reference value.",
        "bl_zz_r_title": "Each qubit's own state",
        "bl_zz_r_xaxis": "|r| — 1 = keeps its own state · 0 = fully entangled",
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
                       "<b>one</b> variable moves only <b>its qubit and its immediate neighbours</b>—the "
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

        # ── Page 7 · Live Predictor ──
        "lp_eyebrow": "Interactive inference",
        "lp_title": "Live Predictor",
        "lp_subtitle": ("Probability that a clinical profile corresponds to a person already diagnosed with "
                        "diabetes—LightGBM over the 8 most important variables."),
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
                          "models (<code>.onnx</code>) connected—place <code>lgbm_final.onnx</code>, "
                          "<code>svm_final.onnx</code>, <code>scaler_correcto.json</code> and "
                          "<code>medianas_correctas.json</code> in <code>streamlit/models/</code>. The score "
                          "shown below is a <b>transparent stand-in</b>: a combination weighted by normalized "
                          "SHAP importance, for layout purposes only. <b>It is not the output of any trained "
                          "model</b> and must not be cited as a result. QSVM is likewise unavailable in real time "
                          "(O(n²) cost of the quantum kernel)."),
        "lp_train_range": "Training: {mu} ± {sd} (±3 sd → {lo} to {hi})",
        "lp_extrapolates": "⚠ z = {z} — outside the trained range: the model extrapolates",
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
        "lp_side_sub": ("Each probability is judged against the cut-off point of its own model—they are not "
                        "interchangeable"),
        "lp_own_threshold": "Its threshold",
        "lp_would_classify": "Would classify as",
        "lp_positive": "positive",
        "lp_negative": "negative",
        "lp_disagree": ("<b>The two models disagree by {dif} on this profile.</b> They agree at the extremes"
                        "—clearly healthy or clearly diabetic profiles—and diverge in the intermediate band, "
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
                         "<li><b>LDL cholesterol</b> — the higher the LDL, the <i>lower</i> the estimated "
                         "probability (from 43% to 18% across the slider). Diagnosed patients are usually on "
                         "statins.</li>"
                         "<li><b>Fasting glucose</b> — the response is U-shaped: very low values raise the "
                         "estimate as much as high ones, because of hypoglycemia in treated patients.</li></ul>"
                         '<div style="margin-top:10px;">Neither should be read as a modifiable risk '
                         "factor.</div>"),
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
SEARCH_PREFIX = {
    "ov": "overview", "gov": "governance", "res": "results",
    "sh": "shap", "qc": "circuit", "bl": "bloch", "lp": "predictor",
}
# El "_title" de cada página es el titular de la página entera, no una sección suya:
# ya entra en el índice como fila de página (con el rótulo del menú), así que aquí se
# excluye para no duplicar la entrada.
_SEARCH_PAGE_TITLES = {f"{p}_title" for p in SEARCH_PREFIX}

# Términos que un lector buscaría pero que no aparecen literalmente en ningún rótulo:
# nombres propios de tecnología, siglas y sinónimos. No generan fila propia — se suman
# al texto buscable de SU página, de modo que "Qiskit" lleva al Circuito Cuántico
# aunque ninguna sección se llame así. Van en los dos idiomas en la misma lista porque
# son notación técnica: nadie traduce "AUC" ni "ZZFeatureMap".
SEARCH_ALIAS = {
    "overview":   ["NHANES", "medallon", "medallion", "bronze", "silver", "gold",
                   "dataset", "pipeline", "Databricks"],
    "governance": ["Great Expectations", "calidad", "quality", "linaje", "lineage",
                   "Delta Lake", "Unity Catalog", "DataOps", "auditoria", "audit"],
    "results":    ["ROC", "AUC", "matriz de confusion", "confusion matrix", "F1",
                   "recall", "precision", "LightGBM", "SVM", "RBF", "QSVM", "metricas",
                   "metrics", "benchmark"],
    "shap":       ["SHAP", "Shapley", "importancia", "importance", "beeswarm",
                   "explicabilidad", "explainability", "XAI"],
    "circuit":    ["Qiskit", "ZZFeatureMap", "qubit", "kernel cuantico", "quantum kernel",
                   "circuito", "circuit", "puerta", "gate", "entrelazamiento",
                   "entanglement", "IBM"],
    "bloch":      ["Bloch", "esfera", "sphere", "estado", "state", "amplitud", "amplitude",
                   "entrelazamiento", "entanglement", "Bell", "Q-sphere", "qsphere", "CNOT",
                   "Hadamard", "medicion", "measurement", "superposicion", "superposition",
                   "ZZFeatureMap", "informacion mutua", "mutual information", "matriz densidad",
                   "density matrix", "8 qubits", "cono de luz", "light cone"],
    "predictor":  ["ONNX", "inferencia", "inference", "umbral", "threshold", "slider",
                   "prediccion", "prediction", "what-if", "simulador", "simulator"],
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
        filas.append({"page": pagina, "label": rotulo, "kind": 1, "hay": _plano(rotulo)})

    # Las ocho variables clínicas: se busca tanto por rótulo ("HbA1c") como por su
    # código NHANES ("LBXGH"), que es como aparecen en los informes del TFM.
    for codigo, rotulo in txt("qsvm_labels").items():
        filas.append({"page": "predictor", "label": f"{codigo} — {rotulo}", "kind": 2,
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
