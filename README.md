# TFM — Quantum Machine Learning en pipeline DataOps sobre Databricks

**Máster en Análisis de Datos Masivos (Big Data)**  
Universidad Europea de Valencia — Curso 2025-2026  
**Autor:** Juan Albornoz Carrasco  
**Director:** Prof. Ronal Muresano

\---

## Descripción

Este repositorio contiene la implementación completa del Trabajo Fin de Máster:

> \*"Integración de Quantum Machine Learning en un pipeline DataOps: arquitectura Medallón sobre Databricks y comparativa con modelos clásicos en predicción clínica"\*

El proyecto diseña e implementa un framework DataOps end-to-end sobre **Databricks Community Edition**, con **AWS S3** como capa de almacenamiento cloud real y una **arquitectura Medallón** (Bronze → Silver → Gold) sobre **Delta Lake**. El componente diferencial es la integración de un **Quantum Support Vector Machine (QSVM)** con `ZZFeatureMap` y `FidelityQuantumKernel` (Qiskit), comparado experimentalmente contra modelos clásicos de referencia bajo condiciones controladas.

\---

## Stack tecnológico

|Capa|Tecnología|
|-|-|
|Almacenamiento cloud|AWS S3 (`tfm-nhanes`, región `eu-west-1`)|
|Plataforma de datos|Databricks Community Edition (Serverless)|
|Arquitectura de datos|Delta Lake — Medallón (Bronze / Silver / Gold)|
|ML clásico|LightGBM 4.x, SVM-RBF (scikit-learn)|
|Quantum ML|Qiskit, ZZFeatureMap, FidelityQuantumKernel|
|Interpretabilidad|SHAP (TreeExplainer)|
|Serialización|ONNX (LightGBM, SVM-RBF)|
|Calidad del dato|dataframe-expectations 0.7.0|
|Despliegue|Streamlit Community Cloud|

\---

## Dataset

**NHANES** (National Health and Nutrition Examination Survey) — CDC, EE.UU.

* 3 ciclos: 2013-2014 (`\_H`), 2015-2016 (`\_I`), 2017-2018 (`\_J`)
* 27 archivos XPT — 9 módulos por ciclo
* 29.400 registros Bronze → 7.831 registros Silver
* **Variable objetivo:** `TARGET = (DIQ010 == 1)` — respuesta a *«¿un médico le ha dicho alguna vez que tiene diabetes?»*
* Desbalance de clases: 86% negativo / 14% positivo

> **Qué significa el objetivo.** `DIQ010` es un **autoinforme de diagnóstico ya emitido**, no una medida de riesgo futuro. Los modelos, por tanto, resuelven una tarea de **detección concurrente**: estiman si un perfil corresponde a alguien **ya diagnosticado**. Esto tiene consecuencias interpretativas que el dashboard documenta en la propia página del Predictor: como los diagnosticados están tratados, el colesterol LDL aparece con **signo invertido** (más LDL → menor probabilidad estimada, por el efecto de las estatinas) y la glucosa en ayunas tiene una respuesta **en forma de U**. Ninguna de las dos debe leerse como factor de riesgo modificable.

### Features efectivas

De las **89 features** que entran a los modelos, **23 tienen varianza cero** y no aportan información: son categóricas codificadas numéricamente (respuestas 1/2, idioma de entrevista, códigos 7 y 9) que la winsorización IQR × 3 de la capa Silver colapsó a un único valor. El `notebook_03_gold.ipynb` lo imprime (`Columnas excluidas (varianza = 0): 23`) y la página Gobernanza del dashboard las cuenta en vivo desde el propio scaler.

**89 features nominales → 66 efectivas.** Ver [TECHNICAL\_NOTES.md](TECHNICAL_NOTES.md) sección 2, y el detalle con la lista completa y la cadena causal en [INFORME\_AUDITORIA\_DASHBOARD.md](INFORME_AUDITORIA_DASHBOARD.md), sección 3.2.

> ⚠️ Los datos NHANES no se incluyen en este repositorio por restricciones de licencia CDC.  
> Descarga disponible en: https://wwwn.cdc.gov/nchs/nhanes/

\---

## Estructura del repositorio

```
tfm-qml-databricks-pipeline/
├── notebooks/
│   ├── notebook\_01\_bronze.ipynb              # Ingesta ELT desde AWS S3 → Bronze
│   ├── notebook\_02\_silver.ipynb              # Limpieza y transformación → Silver
│   ├── notebook\_03\_gold.ipynb                # Preparación para modelado → Gold
│   ├── notebook\_04\_lgbm.ipynb                # Modelo LightGBM + SHAP + ONNX
│   ├── notebook\_05\_svm.ipynb                 # Modelo SVM-RBF + ONNX
│   ├── notebook\_06\_qsvm.ipynb                # Modelo QSVM (ZZFeatureMap + FidelityQuantumKernel)
│   ├── notebook\_07\_validacion\_silver.ipynb   # Suite dataframe-expectations
│   ├── INSTRUCCIONES\_exportar\_scores\_ROC.md  # Celdas para persistir los scores del test
│   └── INSTRUCCIONES\_exportar\_golden\_set.md  # Celdas para el golden set de verificación
├── streamlit/
│   ├── app.py                                # Aplicación de resultados interactiva
│   ├── .streamlit/config.toml                # Tema de la aplicación
│   ├── assets/                               # Logotipos e imágenes de la interfaz
│   ├── models/                               # Artefactos que consume la app (ver Despliegue)
│   └── requirements.txt                      # Dependencias del despliegue en Streamlit Cloud
├── figures/                                  # Figuras del TFM (SHAP, circuito cuántico)
├── audit\_scripts/
│   ├── audit\_sens.py                         # Barridos de sensibilidad sobre los ONNX
│   ├── audit\_metrics.py                      # Recálculo de métricas desde los .npy
│   └── audit\_baseline.py                     # Vector base, one-hot y rangos de sliders
├── requirements.txt
├── TECHNICAL\_NOTES.md                        # Decisiones técnicas y limitaciones
├── INFORME\_AUDITORIA\_TFM.md                  # Auditoría estática del pipeline
├── INFORME\_AUDITORIA\_DASHBOARD.md            # Auditoría dinámica del dashboard
├── .gitignore
└── LICENSE
```

\---

## Documentación

|Documento|Qué contiene|
|-|-|
|[TECHNICAL\_NOTES.md](TECHNICAL_NOTES.md)|Decisiones técnicas del pipeline, con sus alternativas descartadas y las limitaciones asumidas|
|[INFORME\_AUDITORIA\_TFM.md](INFORME_AUDITORIA_TFM.md)|**Auditoría estática** (2–3 ago 2026) de los 7 notebooks y la app. Ningún notebook se ejecutó: validez de formato, compilación de las 113 celdas, ausencia de credenciales y coherencia de artefactos entre notebooks|
|[INFORME\_AUDITORIA\_DASHBOARD.md](INFORME_AUDITORIA_DASHBOARD.md)|**Auditoría dinámica** (6 ago 2026) del dashboard. Aquí sí se ejecutó código: inferencia ONNX real, recálculo de las métricas de los tres modelos contra los `.npy` y verificación de la cadena de custodia notebooks → artefactos → app|
|[notebooks/INSTRUCCIONES\_exportar\_scores\_ROC.md](notebooks/INSTRUCCIONES_exportar_scores_ROC.md)|Celdas para persistir los scores del test (curvas ROC empíricas)|
|[notebooks/INSTRUCCIONES\_exportar\_golden\_set.md](notebooks/INSTRUCCIONES_exportar_golden_set.md)|Celdas para el *golden set* de verificación end-to-end|

Los scripts de la auditoría dinámica están en [`audit_scripts/`](audit_scripts/). No toman argumentos y resuelven `streamlit/models/` de forma relativa a su ubicación:

```bash
python audit_scripts/audit_sens.py       # sensibilidad de las 8 variables clínicas
python audit_scripts/audit_metrics.py    # AUC, matrices y métricas desde los .npy
python audit_scripts/audit_baseline.py   # vector base, one-hot y rangos de sliders
```

\---

## Pipeline DataOps

```
AWS S3 (27 archivos XPT)
    │
    ▼  boto3
┌─────────────────────────────────────────┐
│  Bronze — Delta Lake                    │
│  29.400 registros · 162 columnas        │
└─────────────────────────────────────────┘
    │  filtros · binarización · winsorización · imputación
    ▼
┌─────────────────────────────────────────┐
│  Silver — Delta Lake                    │
│  7.831 registros · 91 columnas          │
└─────────────────────────────────────────┘
    │  encoding · correlación · split 80/20 · StandardScaler
    ▼
┌─────────────────────────────────────────┐
│  Gold — Delta Lake                      │
│  6.264 train · 1.567 test · 89 features │
└─────────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────────────────┐
    │   datos CRUDOS   │   datos ESCALADOS            │  datos ESCALADOS
    ▼                  ▼                              ▼
┌──────────┐    ┌──────────────┐    ┌─────────────────────────────┐
│ LightGBM │    │   SVM-RBF    │    │           QSVM              │
│GridSearch│    │  C=1.0       │    │  ZZFeatureMap (8q, reps=2)  │
│SHAP+ONNX │    │  ONNX        │    │  FidelityQuantumKernel      │
└──────────┘    └──────────────┘    │  500 instancias train       │
                                    └─────────────────────────────┘
    │
    ▼
Streamlit Community Cloud
```

> **El `StandardScaler` se aplica solo a SVM-RBF y QSVM** (`notebook_03_gold.ipynb`, celda 7), ajustado **únicamente sobre train** para no filtrar estadísticos del test. LightGBM se entrena con los datos crudos y conserva los `NaN`, que maneja de forma nativa. El dashboard reproduce esa asimetría: escala solo para el SVM.
>
> ⚠️ **Trampa de nomenclatura.** Los ficheros `X_train_svm.parquet` y `X_test_svm.parquet` **contienen datos ya escalados** — la celda 26 del NB03 los escribe desde `X_train_svm_scaled` / `X_test_svm_scaled`. Quien los reutilice asumiendo que son crudos los escalaría dos veces.

\---

## Resultados experimentales

Comparativa triangulada sobre el mismo conjunto de test (1.567 instancias):

|Modelo|AUC-ROC|F1-macro|Accuracy|MCC|Umbral|
|-|-|-|-|-|-|
|**LightGBM**|**0.9485**|0.6523|0.7243|0.4566|`predict_proba ≥ 0.50`|
|**SVM-RBF**|0.9377|**0.8243**|**0.9075**|**0.6539**|signo de `decision_function` (≈ 0.22 en probabilidad)|
|**QSVM**|0.5493|0.4669|0.8602|0.0625|`decision_function > 0`|

> QSVM entrenado sobre muestra estratificada de 500 instancias por coste computacional O(n²).  
> Los tres modelos se evalúan sobre el mismo conjunto de test completo (1.567 instancias).

⚠️ **Solo el AUC-ROC es comparable entre modelos.** Cada uno está medido en su punto de corte natural, y los tres son distintos: `SVC.predict()` de scikit-learn usa el signo de `decision_function`, **no** `predict_proba ≥ 0.5` — son inconsistentes por diseño. F1-macro, Accuracy y MCC dependen del umbral y por tanto no se pueden poner en fila sin esa advertencia. A modo de referencia, el SVM-RBF evaluado a 0.50 como LightGBM daría accuracy 0.9190 (mejor) pero **131 verdaderos positivos en lugar de 172** (bastante peor recall).

Las cuatro métricas de los tres modelos **reproducen exactamente** al recalcularlas desde los scores por instancia, cada una en su propio umbral. El dashboard lo comprueba en cada carga — ver *Verificación automática* más abajo — y el recálculo independiente está en [INFORME\_AUDITORIA\_DASHBOARD.md](INFORME_AUDITORIA_DASHBOARD.md), sección 2.

\---

## Ejecución

### Requisitos previos

* Databricks Community Edition con Unity Catalog habilitado
* Bucket AWS S3 con los 27 archivos XPT de NHANES
* Credenciales AWS IAM con permisos de lectura sobre el bucket

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Orden de ejecución de los notebooks

```
1. notebook\_01\_bronze.ipynb          # \~5 min
2. notebook\_02\_silver.ipynb          # \~3 min
3. notebook\_03\_gold.ipynb            # \~2 min
4. notebook\_04\_lgbm.ipynb            # \~10 min (GridSearchCV 5-fold)
5. notebook\_05\_svm.ipynb             # \~15 min
6. notebook\_06\_qsvm.ipynb            # 21,1 min entrenamiento + 144,5 min predicción
7. notebook\_07\_validacion\_silver.ipynb  # \~1 min
```

> Tiempos del QSVM tomados de las salidas del propio `notebook_06_qsvm.ipynb`: **21,1 min** de entrenamiento sobre 500 instancias y **144,5 min** para predecir las 1.567 del test por lotes de 100. Los metadatos persistidos de ese mismo notebook registran `train_time_min: 22.6`; el valor correcto es **21,1**.

### Configuración de credenciales AWS

Las credenciales se gestionan con **Databricks Secrets**: nunca se escriben en el notebook. Alta del scope (una sola vez, desde la CLI de Databricks):

```bash
databricks secrets create-scope --scope aws-nhanes
databricks secrets put-secret   --scope aws-nhanes --key access_key
databricks secrets put-secret   --scope aws-nhanes --key secret_key
```

`notebook_01_bronze.ipynb` las recupera en tiempo de ejecución:

```python
access_key = dbutils.secrets.get(scope="aws-nhanes", key="access_key")
secret_key = dbutils.secrets.get(scope="aws-nhanes", key="secret_key")
```

> ⚠️ Nunca escribas credenciales en claro en los notebooks ni las subas al repositorio.

### Modo ejecución QSVM

El notebook QSVM detecta automáticamente si ya existe un modelo entrenado, para evitar repetir las ~3 horas de cómputo:

```python
TRAINING_MODE = not os.path.exists(f"{models_dir}/qsvm_final.pkl")
# sin qsvm_final.pkl -> True  : entrena desde cero (~166 min en total)
# con qsvm_final.pkl -> False : carga el modelo guardado (~2 min en total)
```

Puede forzarse a `True` manualmente si se desea re-entrenar sobre un modelo ya existente.

> ⚠️ Si el entorno ha actualizado Qiskit desde que se entrenó el modelo, el `.pkl` no puede deserializarse. El notebook detecta el fallo y re-entrena automáticamente — ver [TECHNICAL\_NOTES.md](TECHNICAL_NOTES.md), sección 2.11.

### Despliegue de la aplicación Streamlit

Streamlit Community Cloud solo accede al contenido del repositorio, no a Unity Catalog Volumes. Tras ejecutar los notebooks de modelado hay que copiar a `streamlit/models/` los artefactos que consume la app.

**Obligatorios (8).** Sin ellos el Predictor cae a un sustituto de maquetación, rotulado como tal:

|Artefacto|Origen|Contenido|
|-|-|-|
|`lgbm_final.onnx`|NB04|Modelo LightGBM serializado|
|`svm_final.onnx`|NB05|Modelo SVM-RBF serializado|
|`scaler_correcto.json`|NB03 celda 8|`scaler.mean_` y `scaler.scale_`, ajustados solo sobre train|
|`medianas_correctas.json`|NB03 celda 8|Medianas de train sobre datos **crudos**|
|`lgbm_y_scores.npy`|NB04|Probabilidades del test|
|`svm_y_scores.npy`|NB05|Probabilidades del test|
|`qsvm_y_scores.npy`|NB06|`decision_function` del test|
|`qsvm_y_test.npy`|NB06|Etiquetas del test (compartidas por los tres modelos)|

**Opcionales (3).** Si están, el dashboard activa comprobaciones adicionales; si no, lo dice y no afirma nada:

|Artefacto|Origen|Habilita|
|-|-|-|
|`golden_lgbm.npz`|NB04 — ver [INSTRUCCIONES\_exportar\_golden\_set.md](notebooks/INSTRUCCIONES_exportar_golden_set.md)|Verificación end-to-end del camino de inferencia|
|`golden_svm.npz`|NB05 — íd.|Íd., incluido el paso de escalado|
|`validacion_silver_dfe.csv`|NB07|Cifras de la suite de calidad leídas del fichero en vez de transcritas|

Detalle completo en [TECHNICAL\_NOTES.md](TECHNICAL_NOTES.md), sección 6.

\---

## Verificación automática del dashboard

La aplicación no se limita a mostrar cifras transcritas: **las comprueba contra los artefactos en cada carga** y publica el resultado en pantalla. Si un reentrenamiento desincroniza los ficheros de las constantes, la app lo denuncia en lugar de seguir mostrando números muertos.

### 1. Reconciliación de métricas — página *Resultados*

`reconciliar_metricas()` recalcula AUC-ROC (Mann-Whitney, con corrección de empates), matriz de confusión, accuracy, F1-macro y MCC de los tres modelos desde sus scores por instancia, **cada uno en su propio umbral**, y los compara con lo publicado.

* **✓ Reconciliadas** — las doce cifras coinciden *(estado actual)*
* **⚠ Sin reconciliar** — se listan las discrepancias exactas, modelo a modelo

### 2. Verificación end-to-end — página *Gobernanza → Linaje*

La reconciliación demuestra que las métricas cuadran con los scores, pero no que **este** camino de inferencia reproduzca los modelos entrenados: el conjunto de test no está en el repositorio. El *golden set* cierra ese hueco con 25 filas del test acompañadas de la probabilidad que devolvió el modelo entrenado.

`verificar_golden()` reconstruye el camino completo —vector crudo, escalado solo del SVM, conversión a `float32`, sesión ONNX y lectura del tensor de salida— y compara con tolerancia 1 × 10⁻⁴ (el ruido esperado por trabajar en `float32` es de orden 10⁻⁷).

Ese último paso no es un detalle: **cada modelo devuelve una estructura distinta**. El NB04 convirtió con `zipmap: False`, así que `out[1]` es un `ndarray (N,2)`; el NB05 sin esa opción, así que `out[1]` es una lista de diccionarios. La expresión `out[1][i][1]` acierta en ambos, pero por motivos diferentes, y una regresión ahí devolvería números plausibles y falsos.

Sin los `.npz`, la tarjeta muestra *«Sin verificar»* con las instrucciones — **nunca afirma haber comprobado lo que no ha podido comprobar**.

### 3. Features efectivas — página *Gobernanza → Calidad*

Las 23 columnas de varianza cero se **cuentan desde el propio scaler** (`scale_ == 1.0` marca varianza nula), no se transcriben: si el pipeline se reejecuta, la cifra se mueve sola.

\---

## Cómo leer el Predictor en Vivo

La página estima la **probabilidad de que un perfil corresponda a alguien ya diagnosticado**, no un riesgo futuro. Tres cosas que conviene saber antes de interpretar un número:

**Es una escalera, no una rampa.** LightGBM es un ensemble de árboles: su salida es constante a tramos. Sobre HbA1c hay **14 umbrales**, y caen todos exactamente en `X,X5` — NHANES registra la HbA1c con un decimal y el modelo parte por el punto medio entre valores observados. Como el slider avanza de 0,1 en 0,1, **nunca puede posarse sobre un umbral**: solo se ven los saltos. De las 111 posiciones del slider salen **15 probabilidades distintas**. La *curva de respuesta* de la página lo dibuja con forma de escalón real (nunca suavizada) y cuenta los peldaños de la configuración vigente.

**El modelo decide antes del criterio clínico.** Con los valores por defecto cruza el 50 % en HbA1c ≈ 6,05 — dentro del rango de **prediabetes** de la ADA (5,7–6,4). El 79 % de su recorrido ocurre en esa banda, y al llegar al criterio diagnóstico (≥ 6,5) ya está saturado en 92 %. Es la confusión por tratamiento: los diagnosticados están controlados en la banda 6,0–7,0.

**Extrapola fuera del rango entrenado.** Los sliders cubren rangos fisiológicos que superan con creces lo que el modelo vio: el tope de HbA1c (15 %) está a **z = +12** de la media de entrenamiento (5,72 ± 0,77 tras la winsorización). Por eso cada slider muestra su rango de entrenamiento y avisa al salirse de ±3 sd. Consecuencia práctica: HbA1c 7,3 % y 15 % son **indistinguibles** para el modelo.

\---

## Limitaciones conocidas

Todas están documentadas y asumidas; ninguna se corrige en silencio. Cifras y evidencia en [INFORME\_AUDITORIA\_DASHBOARD.md](INFORME_AUDITORIA_DASHBOARD.md).

|#|Limitación|Detalle|
|-|-|-|
|1|**El objetivo es un diagnóstico, no un riesgo**|`DIQ010` es autoinforme de diagnóstico ya emitido. Los modelos hacen detección concurrente; el AUC de 0,9485 mide eso, no capacidad predictiva prospectiva|
|2|**Confusión por tratamiento**|Invierte el signo del LDL y da forma de U a la glucosa. Advertido en la propia página del Predictor|
|3|**23 de 89 features son constantes**|Efecto colateral de aplicar winsorización IQR × 3 a categóricas codificadas numéricamente|
|4|**`WTINT2YR` es una de las 89 features**|Peso de expansión muestral NHANES, no una variable clínica. Es la 6ª por importancia SHAP y mueve la predicción hasta 9 puntos. TECHNICAL\_NOTES 2.10|
|5|**Filtro de correlación antes de particionar**|Las 16 columnas descartadas por `r > 0,90` se deciden usando también observaciones del test. No afecta al escalado ni a la selección de features del QSVM, ambos ajustados solo sobre train. Decisión 09|
|6|**`lgbm_final.onnx` declara opset 1**|El NB04 pidió `target_opset={'': 12}` pero el fichero registra 1 para el dominio `ai.onnx` — rareza del conversor de LightGBM de `onnxmltools`. Hoy funciona por soporte legacy de `onnxruntime` y da resultados correctos (verificado 100 % contra el pickle), pero conviene reexportar antes de archivar el proyecto. `svm_final.onnx` está correcto (`ai.onnx` 9)|
|7|**Métricas en umbrales distintos**|Ver la advertencia de *Resultados experimentales*|

\---

## Versiones del entorno

|Librería|Versión|
|-|-|
|Python|3.12|
|pandas|1.5.3|
|numpy|1.23.5|
|scikit-learn|1.6.1|
|lightgbm|4.x|
|qiskit|2.5.0|
|qiskit-machine-learning|0.9.0|
|qiskit-algorithms|0.4.0|

\---

## Licencia

MIT License — ver [LICENSE](LICENSE)

\---

*Trabajo Fin de Máster — Universidad Europea de Valencia — 2025-2026*

